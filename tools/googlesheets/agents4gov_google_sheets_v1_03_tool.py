"""
title: Agents4Gov - Google Sheets
author: Agents4Gov
description: Le sempre a primeira aba de uma Google Planilha retornando coordenadas explicitas de linha, coluna e celula para cada valor; quando autorizado, le ou escreve uma celula. Possui debug persistente por sessao, exibido somente quando solicitado.
required_open_webui_version: 0.11.0
requirements: google-auth>=2.38.0, requests>=2.32.0
version: 1.03
license: MIT
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from urllib.parse import quote

import httpx
from google.auth.transport.requests import Request as GoogleAuthRequest
from google.oauth2 import service_account
from pydantic import BaseModel, Field


SHEETS_BASE_URL = "https://sheets.googleapis.com/v4"
SCOPE_READONLY = "https://www.googleapis.com/auth/spreadsheets.readonly"
SCOPE_READWRITE = "https://www.googleapis.com/auth/spreadsheets"


class ToolError(RuntimeError):
    pass


class DebugRun:
    def __init__(
        self,
        *,
        enabled: bool,
        root_dir: str,
        operation: str,
        session_key: str,
        max_chars: int,
    ):
        self.enabled = bool(enabled)
        self.operation = operation
        self.session_key = session_key
        self.max_chars = int(max_chars)
        self.started = time.monotonic()
        self.run_id = (
            datetime.now().strftime("%Y%m%d_%H%M%S")
            + "_"
            + uuid.uuid4().hex[:8]
        )
        self.lines: list[str] = []

        self.root: Optional[Path] = None
        self.log_path: Optional[Path] = None
        self.last_pointer_path: Optional[Path] = None

        if not self.enabled:
            return

        preferred = Path(root_dir).expanduser()
        fallback = Path(tempfile.gettempdir()) / "agents4gov-google-sheets-logs"

        try:
            preferred.mkdir(parents=True, exist_ok=True)
            self.root = preferred
        except Exception:
            fallback.mkdir(parents=True, exist_ok=True)
            self.root = fallback

        run_dir = self.root / "runs"
        session_dir = self.root / "sessions"
        run_dir.mkdir(parents=True, exist_ok=True)
        session_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = run_dir / f"{self.run_id}.log"
        self.last_pointer_path = session_dir / f"{self.session_key}.json"

        self.info(
            "RUN_START",
            f"operation={operation} run_id={self.run_id}",
        )

    @staticmethod
    def _safe_text(value: Any) -> str:
        text = str(value)
        return text.replace("\r", " ").replace("\n", "\\n")

    def write(self, level: str, event: str, message: str) -> None:
        if not self.enabled:
            return

        elapsed = time.monotonic() - self.started
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        line = (
            f"[{stamp}]"
            f"[+{elapsed:08.2f}s]"
            f"[AGENTS4GOV-GSHEETS-V1.03]"
            f"[{level}]"
            f"[{event}] "
            f"{self._safe_text(message)}"
        )

        self.lines.append(line)

        try:
            assert self.log_path is not None
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")
        except Exception:
            pass

        print(line, flush=True)

    def info(self, event: str, message: str) -> None:
        self.write("INFO", event, message)

    def ok(self, event: str, message: str) -> None:
        self.write("OK", event, message)

    def warning(self, event: str, message: str) -> None:
        self.write("WARN", event, message)

    def error(self, event: str, message: str) -> None:
        self.write("ERROR", event, message)

    def finish(self, status: str) -> None:
        if not self.enabled:
            return

        self.info("RUN_END", f"status={status}")

        pointer = {
            "run_id": self.run_id,
            "operation": self.operation,
            "status": status,
            "log_path": str(self.log_path or ""),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        try:
            assert self.last_pointer_path is not None
            tmp = self.last_pointer_path.with_suffix(".tmp")
            tmp.write_text(
                json.dumps(pointer, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            tmp.replace(self.last_pointer_path)
        except Exception:
            pass

    def response_hint(self) -> dict[str, Any]:
        if not self.enabled:
            return {}

        return {
            "debug_disponivel": True,
            "debug_run_id": self.run_id,
            "instrucao_debug_para_llm": (
                "Nao mostre o log automaticamente. "
                "Se o usuario pedir o log completo, detalhes de debug ou todo o fluxo, "
                "chame obter_log_debug para esta mesma sessao."
            ),
        }


class Tools:
    class Valves(BaseModel):
        organizacao: str = Field(
            default="Agents4Gov",
            description="Nome da organizacao responsavel pela configuracao da Tool.",
        )

        google_api_key: str = Field(
            default="",
            description="Google API Key para leitura de planilhas publicas.",
            json_schema_extra={"input": {"type": "password"}},
        )

        service_account_json: str = Field(
            default="",
            description=(
                "JSON completo da Service Account Google. Necessario para planilhas "
                "privadas e obrigatorio para escrita."
            ),
            json_schema_extra={"input": {"type": "password"}},
        )

        preferir_service_account_na_leitura: bool = Field(
            default=True,
            description="Usa Service Account na leitura quando estiver configurada.",
        )

        permitir_escrita: bool = Field(
            default=False,
            description="Habilita escrita quando a Service Account tiver permissao de edicao.",
        )

        exigir_confirmacao_escrita: bool = Field(
            default=True,
            description="Exige confirmacao_usuario=true antes de alterar uma celula.",
        )

        modo_escrita: Literal["RAW", "USER_ENTERED"] = Field(
            default="RAW",
            description=(
                "RAW grava literalmente; USER_ENTERED interpreta como digitacao no Sheets."
            ),
        )

        valor_renderizado: Literal[
            "FORMATTED_VALUE",
            "UNFORMATTED_VALUE",
            "FORMULA",
        ] = Field(
            default="FORMATTED_VALUE",
            description="Forma de renderizacao dos valores lidos.",
        )

        limite_max_linhas: int = Field(
            default=1000,
            ge=1,
            le=10000,
            description="Limite administrativo maximo de linhas por leitura.",
        )

        limite_max_colunas: int = Field(
            default=100,
            ge=1,
            le=1000,
            description="Limite administrativo maximo de colunas por leitura.",
        )

        timeout_segundos: int = Field(
            default=30,
            ge=5,
            le=120,
            description="Timeout das chamadas a Google Sheets API.",
        )

        tentativas_http: int = Field(
            default=3,
            ge=1,
            le=5,
            description="Tentativas em erros HTTP temporarios.",
        )

        debug: bool = Field(
            default=False,
            description=(
                "Quando ativado, registra o fluxo completo de cada execucao. "
                "O log nao e exibido automaticamente; use obter_log_debug quando solicitado."
            ),
        )

        debug_dir: str = Field(
            default="/app/backend/data/agents4gov/google-sheets-logs",
            description="Diretorio persistente para logs de debug.",
        )

        debug_max_chars: int = Field(
            default=50000,
            ge=5000,
            le=200000,
            description="Numero maximo de caracteres retornados por obter_log_debug.",
        )

        debug_ttl_horas: int = Field(
            default=72,
            ge=1,
            le=720,
            description="Tempo de retencao dos arquivos de log de debug.",
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _status(
        self,
        emitter,
        description: str,
        done: bool = False,
    ) -> None:
        if emitter is None:
            return

        await emitter(
            {
                "type": "status",
                "data": {
                    "description": description,
                    "done": done,
                },
            }
        )

    @staticmethod
    def _session_key(
        __user__: Optional[dict],
        __chat_id__: Optional[str],
        __metadata__: Optional[dict],
    ) -> str:
        user = __user__ or {}
        metadata = __metadata__ or {}

        user_id = str(
            user.get("id")
            or metadata.get("user_id")
            or "sem-user"
        )
        chat_id = str(
            __chat_id__
            or metadata.get("chat_id")
            or "sem-chat"
        )

        raw = f"{user_id}|{chat_id}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]

    def _new_debug_run(
        self,
        operation: str,
        *,
        __user__: Optional[dict],
        __chat_id__: Optional[str],
        __metadata__: Optional[dict],
    ) -> DebugRun:
        self._cleanup_debug_logs()

        return DebugRun(
            enabled=self.valves.debug,
            root_dir=self.valves.debug_dir,
            operation=operation,
            session_key=self._session_key(
                __user__,
                __chat_id__,
                __metadata__,
            ),
            max_chars=self.valves.debug_max_chars,
        )

    def _cleanup_debug_logs(self) -> None:
        if not self.valves.debug:
            return

        root = Path(self.valves.debug_dir).expanduser()

        if not root.exists():
            return

        cutoff = time.time() - int(self.valves.debug_ttl_horas) * 3600

        try:
            run_dir = root / "runs"
            if run_dir.exists():
                for path in run_dir.glob("*.log"):
                    try:
                        if path.stat().st_mtime < cutoff:
                            path.unlink(missing_ok=True)
                    except OSError:
                        pass
        except Exception:
            pass

    @staticmethod
    def _validate_spreadsheet_id(spreadsheet_id: str) -> str:
        value = str(spreadsheet_id or "").strip()

        if not value:
            raise ToolError("spreadsheet_id nao foi informado.")

        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ToolError(
                "spreadsheet_id invalido. Informe somente o ID da Google Planilha."
            )

        return value

    @staticmethod
    def _column_number_to_letters(number: int) -> str:
        if number < 1:
            raise ToolError("A coluna deve ser maior ou igual a 1.")

        letters = ""

        while number:
            number, remainder = divmod(number - 1, 26)
            letters = chr(65 + remainder) + letters

        return letters

    @classmethod
    def _normalize_column(cls, column: str) -> tuple[str, int]:
        raw = str(column or "").strip().upper()

        if not raw:
            raise ToolError("A coluna nao foi informada.")

        if raw.isdigit():
            number = int(raw)
            return cls._column_number_to_letters(number), number

        if not re.fullmatch(r"[A-Z]+", raw):
            raise ToolError(
                "Coluna invalida. Use A, B, AA ou numeros como 1, 2, 27."
            )

        number = 0

        for char in raw:
            number = number * 26 + ord(char) - 64

        return raw, number

    @staticmethod
    def _quote_sheet_title(title: str) -> str:
        return "'" + str(title).replace("'", "''") + "'"

    def _parse_service_account_info(
        self,
        debug: Optional[DebugRun] = None,
    ) -> dict[str, Any]:
        raw = str(self.valves.service_account_json or "").strip()

        if not raw:
            raise ToolError("Service Account nao configurada nas Valves.")

        try:
            info = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ToolError(
                "service_account_json nao contem JSON valido."
            ) from exc

        if not isinstance(info, dict) or info.get("type") != "service_account":
            raise ToolError(
                "service_account_json nao corresponde a uma Service Account Google."
            )

        for key in ("client_email", "private_key", "token_uri"):
            if not str(info.get(key) or "").strip():
                raise ToolError(
                    f"Service Account incompleta: campo {key} ausente."
                )

        if debug:
            debug.ok(
                "SERVICE_ACCOUNT_CONFIG",
                (
                    f"client_email={info.get('client_email', '')!r} "
                    "private_key=[REDACTED]"
                ),
            )

        return info

    async def _service_account_token(
        self,
        *,
        write: bool,
        debug: Optional[DebugRun] = None,
    ) -> tuple[str, str]:
        info = self._parse_service_account_info(debug)
        scope = SCOPE_READWRITE if write else SCOPE_READONLY

        if debug:
            debug.info(
                "AUTH_TOKEN_BEGIN",
                f"scope={scope}",
            )

        try:
            credentials = service_account.Credentials.from_service_account_info(
                info,
                scopes=[scope],
            )
            await asyncio.to_thread(
                credentials.refresh,
                GoogleAuthRequest(),
            )
        except Exception as exc:
            if debug:
                debug.error(
                    "AUTH_TOKEN_ERROR",
                    f"{type(exc).__name__}: {exc}",
                )
            raise ToolError(
                f"Falha ao autenticar Service Account: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        token = str(credentials.token or "").strip()

        if not token:
            raise ToolError(
                "A Service Account nao retornou access token."
            )

        if debug:
            debug.ok(
                "AUTH_TOKEN_OK",
                "Access token obtido. token=[REDACTED]",
            )

        return token, str(info.get("client_email") or "")

    async def _auth_context(
        self,
        *,
        write: bool,
        debug: Optional[DebugRun] = None,
    ) -> dict[str, Any]:
        api_key = str(self.valves.google_api_key or "").strip()
        has_sa = bool(str(self.valves.service_account_json or "").strip())

        if debug:
            debug.info(
                "AUTH_SELECT",
                (
                    f"write={write} "
                    f"api_key_configured={bool(api_key)} "
                    f"service_account_configured={has_sa} "
                    f"prefer_service_account={self.valves.preferir_service_account_na_leitura}"
                ),
            )

        if write:
            if not self.valves.permitir_escrita:
                raise ToolError(
                    "Escrita desabilitada nas Valves. Ative permitir_escrita."
                )

            if not has_sa:
                raise ToolError(
                    "Escrita exige Service Account; API Key nao autoriza values.update."
                )

            token, email = await self._service_account_token(
                write=True,
                debug=debug,
            )

            if debug:
                debug.ok(
                    "AUTH_SELECTED",
                    f"mode=service_account write=True email={email!r}",
                )

            return {
                "mode": "service_account",
                "headers": {
                    "Authorization": f"Bearer {token}"
                },
                "params": {"key": api_key} if api_key else {},
                "service_account_email": email,
            }

        if has_sa and self.valves.preferir_service_account_na_leitura:
            token, email = await self._service_account_token(
                write=False,
                debug=debug,
            )

            if debug:
                debug.ok(
                    "AUTH_SELECTED",
                    f"mode=service_account write=False email={email!r}",
                )

            return {
                "mode": "service_account",
                "headers": {
                    "Authorization": f"Bearer {token}"
                },
                "params": {"key": api_key} if api_key else {},
                "service_account_email": email,
            }

        if api_key:
            if debug:
                debug.ok(
                    "AUTH_SELECTED",
                    "mode=api_key key=[REDACTED]",
                )

            return {
                "mode": "api_key",
                "headers": {},
                "params": {"key": api_key},
                "service_account_email": "",
            }

        if has_sa:
            token, email = await self._service_account_token(
                write=False,
                debug=debug,
            )

            if debug:
                debug.ok(
                    "AUTH_SELECTED",
                    f"mode=service_account fallback email={email!r}",
                )

            return {
                "mode": "service_account",
                "headers": {
                    "Authorization": f"Bearer {token}"
                },
                "params": {},
                "service_account_email": email,
            }

        raise ToolError(
            "Configure google_api_key para leitura publica "
            "ou service_account_json para acesso autenticado."
        )

    @staticmethod
    def _google_error(response: httpx.Response) -> str:
        try:
            error = response.json().get("error") or {}

            if isinstance(error, dict):
                status = str(error.get("status") or "").strip()
                message = str(error.get("message") or "").strip()

                if status and message:
                    return f"{status}: {message}"

                if message:
                    return message
        except Exception:
            pass

        return response.text.strip()[:1000] or f"HTTP {response.status_code}"

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        auth: dict[str, Any],
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
        debug: Optional[DebugRun] = None,
    ) -> dict[str, Any]:
        merged_params = dict(auth.get("params") or {})

        if params:
            merged_params.update(params)

        safe_params = {
            key: ("[REDACTED]" if key.lower() == "key" else value)
            for key, value in merged_params.items()
        }

        attempts = int(self.valves.tentativas_http)

        async with httpx.AsyncClient(
            timeout=float(self.valves.timeout_segundos)
        ) as client:
            for attempt in range(1, attempts + 1):
                if debug:
                    debug.info(
                        "HTTP_REQUEST",
                        (
                            f"attempt={attempt}/{attempts} "
                            f"method={method} url={url!r} "
                            f"params={safe_params!r} "
                            f"json_body_present={json_body is not None}"
                        ),
                    )

                started = time.monotonic()

                try:
                    response = await client.request(
                        method,
                        url,
                        headers=auth.get("headers") or {},
                        params=merged_params,
                        json=json_body,
                    )
                except httpx.RequestError as exc:
                    if debug:
                        debug.error(
                            "HTTP_TRANSPORT_ERROR",
                            (
                                f"attempt={attempt} "
                                f"elapsed={time.monotonic() - started:.2f}s "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )

                    if attempt >= attempts:
                        raise ToolError(
                            f"Falha de rede na Google Sheets API: "
                            f"{type(exc).__name__}: {exc}"
                        ) from exc

                    delay = min(2 ** (attempt - 1), 4)

                    if debug:
                        debug.warning(
                            "HTTP_RETRY",
                            f"delay_seconds={delay}",
                        )

                    await asyncio.sleep(delay)
                    continue

                if debug:
                    debug.info(
                        "HTTP_RESPONSE",
                        (
                            f"attempt={attempt} status={response.status_code} "
                            f"elapsed={time.monotonic() - started:.2f}s "
                            f"response_bytes={len(response.content)}"
                        ),
                    )

                if response.status_code < 400:
                    try:
                        data = response.json()
                    except Exception as exc:
                        if debug:
                            debug.error(
                                "HTTP_JSON_ERROR",
                                f"{type(exc).__name__}: {exc}",
                            )
                        raise ToolError(
                            "Google Sheets API retornou JSON invalido."
                        ) from exc

                    if not isinstance(data, dict):
                        raise ToolError(
                            "Google Sheets API retornou formato inesperado."
                        )

                    if debug:
                        debug.ok(
                            "HTTP_SUCCESS",
                            f"top_level_keys={sorted(data.keys())}",
                        )

                    return data

                retryable = (
                    response.status_code == 429
                    or 500 <= response.status_code < 600
                )

                detail = self._google_error(response)

                if debug:
                    debug.error(
                        "HTTP_ERROR",
                        (
                            f"status={response.status_code} "
                            f"retryable={retryable} detail={detail!r}"
                        ),
                    )

                if retryable and attempt < attempts:
                    delay = min(2 ** (attempt - 1), 4)

                    if debug:
                        debug.warning(
                            "HTTP_RETRY",
                            f"delay_seconds={delay}",
                        )

                    await asyncio.sleep(delay)
                    continue

                if response.status_code in {401, 403}:
                    raise ToolError(
                        f"Acesso negado pela Google Sheets API: {detail}"
                    )

                if response.status_code == 404:
                    raise ToolError(
                        f"Planilha nao encontrada ou sem permissao: {detail}"
                    )

                raise ToolError(
                    f"Google Sheets API retornou HTTP "
                    f"{response.status_code}: {detail}"
                )

        raise ToolError(
            "Falha inesperada ao acessar Google Sheets API."
        )

    async def _first_sheet(
        self,
        spreadsheet_id: str,
        *,
        auth: dict[str, Any],
        debug: Optional[DebugRun] = None,
    ) -> dict[str, Any]:
        if debug:
            debug.info(
                "FIRST_SHEET_BEGIN",
                f"spreadsheet_id={spreadsheet_id}",
            )

        data = await self._request_json(
            "GET",
            f"{SHEETS_BASE_URL}/spreadsheets/{spreadsheet_id}",
            auth=auth,
            params={
                "includeGridData": "false",
                "fields": (
                    "properties(title),"
                    "sheets(properties("
                    "sheetId,title,index,"
                    "gridProperties(rowCount,columnCount)"
                    "))"
                ),
            },
            debug=debug,
        )

        sheets = data.get("sheets") or []
        props = [
            sheet.get("properties")
            for sheet in sheets
            if isinstance(sheet, dict)
            and sheet.get("properties")
        ]

        if not props:
            raise ToolError(
                "A planilha nao possui abas acessiveis."
            )

        props.sort(
            key=lambda item: int(
                item.get("index", 10**9)
            )
        )

        first = props[0]
        title = str(first.get("title") or "").strip()

        if not title:
            raise ToolError(
                "A primeira aba nao possui titulo valido."
            )

        grid = first.get("gridProperties") or {}

        result = {
            "spreadsheet_title": str(
                (data.get("properties") or {}).get("title")
                or ""
            ),
            "sheet_id": first.get("sheetId"),
            "sheet_title": title,
            "sheet_index": first.get("index"),
            "row_count": grid.get("rowCount"),
            "column_count": grid.get("columnCount"),
        }

        if debug:
            debug.ok(
                "FIRST_SHEET_OK",
                (
                    f"spreadsheet_title={result['spreadsheet_title']!r} "
                    f"sheet_title={title!r} "
                    f"sheet_id={result['sheet_id']} "
                    f"index={result['sheet_index']} "
                    f"row_count={result['row_count']} "
                    f"column_count={result['column_count']}"
                ),
            )

        return result

    async def _read_range(
        self,
        spreadsheet_id: str,
        range_a1: str,
        *,
        auth: dict[str, Any],
        debug: Optional[DebugRun] = None,
    ) -> dict[str, Any]:
        if debug:
            debug.info(
                "READ_RANGE_BEGIN",
                f"range={range_a1!r}",
            )

        data = await self._request_json(
            "GET",
            (
                f"{SHEETS_BASE_URL}/spreadsheets/"
                f"{spreadsheet_id}/values/"
                f"{quote(range_a1, safe='')}"
            ),
            auth=auth,
            params={
                "majorDimension": "ROWS",
                "valueRenderOption": self.valves.valor_renderizado,
            },
            debug=debug,
        )

        if debug:
            values = data.get("values") or []
            debug.ok(
                "READ_RANGE_OK",
                (
                    f"range_returned={data.get('range', '')!r} "
                    f"rows={len(values) if isinstance(values, list) else 0}"
                ),
            )

        return data

    async def diagnosticar_configuracao(
        self,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """
        Verifica a configuracao sem acessar nenhuma planilha e sem expor segredos.
        """

        debug = self._new_debug_run(
            "diagnosticar_configuracao",
            __user__=__user__,
            __chat_id__=__chat_id__,
            __metadata__=__metadata__,
        )

        try:
            debug.info(
                "CONFIG",
                (
                    f"organizacao={self.valves.organizacao!r} "
                    f"debug={self.valves.debug} "
                    f"permitir_escrita={self.valves.permitir_escrita} "
                    f"exigir_confirmacao_escrita={self.valves.exigir_confirmacao_escrita}"
                ),
            )

            api_key = bool(
                str(self.valves.google_api_key or "").strip()
            )
            sa_configured = bool(
                str(self.valves.service_account_json or "").strip()
            )

            sa_email = ""
            sa_error = ""

            if sa_configured:
                try:
                    sa_email = str(
                        self._parse_service_account_info(
                            debug
                        ).get("client_email")
                        or ""
                    )
                except Exception as exc:
                    sa_error = str(exc)
                    debug.error(
                        "SERVICE_ACCOUNT_INVALID",
                        sa_error,
                    )

            result = {
                "status": "success",
                "versao": "1.03",
                "organizacao": self.valves.organizacao,
                "api_key_configurada": api_key,
                "service_account_configurada": sa_configured,
                "service_account_email": sa_email,
                "service_account_erro": sa_error,
                "leitura_disponivel": bool(
                    api_key
                    or (sa_configured and not sa_error)
                ),
                "escrita_habilitada": self.valves.permitir_escrita,
                "escrita_disponivel": bool(
                    self.valves.permitir_escrita
                    and sa_configured
                    and not sa_error
                ),
                "primeira_aba_apenas": True,
                "exigir_confirmacao_escrita": (
                    self.valves.exigir_confirmacao_escrita
                ),
                "modo_escrita": self.valves.modo_escrita,
            }

            debug.ok(
                "CONFIG_RESULT",
                (
                    f"read_available={result['leitura_disponivel']} "
                    f"write_available={result['escrita_disponivel']}"
                ),
            )
            debug.finish("success")
            result.update(debug.response_hint())
            return result

        except Exception as exc:
            debug.error(
                "CONFIG_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
            debug.finish("error")

            result = {
                "status": "error",
                "mensagem": str(exc),
            }
            result.update(debug.response_hint())
            return result

    async def ler_primeira_aba(
        self,
        spreadsheet_id: str,
        max_linhas: int = 200,
        max_colunas: int = 30,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """
        Le dados da primeira aba da Google Planilha.

        Cada valor retornado inclui coordenadas explicitas em linhas[].celulas:
        linha, coluna, numero_coluna, celula, cabecalho e valor.

        :param spreadsheet_id: ID da Google Planilha, sem a URL completa.
        :param max_linhas: Numero maximo de linhas a retornar.
        :param max_colunas: Numero maximo de colunas a retornar.
        """

        debug = self._new_debug_run(
            "ler_primeira_aba",
            __user__=__user__,
            __chat_id__=__chat_id__,
            __metadata__=__metadata__,
        )

        auth_mode = ""

        try:
            spreadsheet_id = self._validate_spreadsheet_id(
                spreadsheet_id
            )
            max_linhas = max(
                1,
                min(
                    int(max_linhas),
                    int(self.valves.limite_max_linhas),
                ),
            )
            max_colunas = max(
                1,
                min(
                    int(max_colunas),
                    int(self.valves.limite_max_colunas),
                ),
            )

            debug.info(
                "INPUT",
                (
                    f"spreadsheet_id={spreadsheet_id} "
                    f"max_linhas={max_linhas} "
                    f"max_colunas={max_colunas}"
                ),
            )

            await self._status(
                __event_emitter__,
                "Conectando a Google Sheets API...",
            )

            auth = await self._auth_context(
                write=False,
                debug=debug,
            )
            auth_mode = auth["mode"]

            first = await self._first_sheet(
                spreadsheet_id,
                auth=auth,
                debug=debug,
            )

            last_column = self._column_number_to_letters(
                max_colunas
            )
            range_a1 = (
                f"{self._quote_sheet_title(first['sheet_title'])}"
                f"!A1:{last_column}{max_linhas}"
            )

            debug.info(
                "RANGE_SELECTED",
                f"range={range_a1!r}",
            )

            await self._status(
                __event_emitter__,
                f"Lendo a primeira aba '{first['sheet_title']}'...",
            )

            data = await self._read_range(
                spreadsheet_id,
                range_a1,
                auth=auth,
                debug=debug,
            )

            values = data.get("values") or []

            if not isinstance(values, list):
                values = []

            columns = []

            for index in range(1, max_colunas + 1):
                header = ""

                if (
                    values
                    and isinstance(values[0], list)
                    and index <= len(values[0])
                ):
                    header = str(values[0][index - 1])

                columns.append(
                    {
                        "numero": index,
                        "coluna": self._column_number_to_letters(index),
                        "cabecalho": header,
                    }
                )

            rows = []

            for row_number, row in enumerate(
                values,
                start=1,
            ):
                row_values = (
                    row
                    if isinstance(row, list)
                    else []
                )

                cells = []

                for column_number, value in enumerate(
                    row_values,
                    start=1,
                ):
                    column_letters = (
                        self._column_number_to_letters(
                            column_number
                        )
                    )

                    header = ""

                    if (
                        values
                        and isinstance(values[0], list)
                        and column_number <= len(values[0])
                    ):
                        header = str(
                            values[0][column_number - 1]
                        )

                    cells.append(
                        {
                            "linha": row_number,
                            "coluna": column_letters,
                            "numero_coluna": column_number,
                            "celula": (
                                f"{column_letters}{row_number}"
                            ),
                            "cabecalho": header,
                            "valor": value,
                        }
                    )

                rows.append(
                    {
                        "linha": row_number,
                        "valores": row_values,
                        "celulas": cells,
                    }
                )

            explicit_cells = sum(
                len(row.get("celulas") or [])
                for row in rows
            )

            debug.ok(
                "PARSE_VALUES",
                (
                    f"rows_returned={len(rows)} "
                    f"columns_described={len(columns)} "
                    f"explicit_cells={explicit_cells}"
                ),
            )

            await self._status(
                __event_emitter__,
                "Leitura concluida.",
                done=True,
            )

            result = {
                "status": "success",
                "spreadsheet_id": spreadsheet_id,
                "spreadsheet_title": first["spreadsheet_title"],
                "aba": {
                    "titulo": first["sheet_title"],
                    "sheet_id": first["sheet_id"],
                    "indice": first["sheet_index"],
                },
                "primeira_aba": True,
                "intervalo_lido": range_a1,
                "linhas_retornadas": len(rows),
                "colunas_solicitadas": max_colunas,
                "celulas_explicitas": True,
                "colunas": columns,
                "linhas": rows,
                "instrucao_para_llm": (
                    "Cada item de linhas[].celulas informa explicitamente linha, coluna, "
                    "numero_coluna, celula A1, cabecalho e valor. Ao identificar um dado "
                    "para consulta ou correcao, use preferencialmente a coordenada retornada "
                    "em celula, por exemplo D7, ou os campos linha=7 e coluna=D. "
                    "Para escrever, nunca deduza a coordenada apenas pela posicao em uma lista "
                    "se a coordenada explicita estiver disponivel. Confirme celula e novo valor "
                    "com o usuario antes de chamar escrever_celula."
                ),
            }

            debug.ok(
                "RESULT",
                (
                    f"status=success auth_mode={auth_mode} "
                    f"sheet={first['sheet_title']!r}"
                ),
            )
            debug.finish("success")
            result.update(debug.response_hint())
            return result

        except Exception as exc:
            debug.error(
                "OPERATION_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
            debug.finish("error")

            await self._status(
                __event_emitter__,
                "Falha ao ler a Google Planilha.",
                done=True,
            )

            result = {
                "status": "error",
                "mensagem": str(exc),
            }
            result.update(debug.response_hint())
            return result

    async def ler_celula(
        self,
        spreadsheet_id: str,
        linha: int,
        coluna: str,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """
        Le uma celula da primeira aba.

        :param spreadsheet_id: ID da Google Planilha.
        :param linha: Numero da linha, iniciando em 1.
        :param coluna: Letra como C/AA ou numero como 3/27.
        """

        debug = self._new_debug_run(
            "ler_celula",
            __user__=__user__,
            __chat_id__=__chat_id__,
            __metadata__=__metadata__,
        )

        try:
            spreadsheet_id = self._validate_spreadsheet_id(
                spreadsheet_id
            )
            linha = int(linha)

            if linha < 1:
                raise ToolError(
                    "A linha deve ser maior ou igual a 1."
                )

            letters, number = self._normalize_column(
                coluna
            )
            cell = f"{letters}{linha}"

            debug.info(
                "INPUT",
                (
                    f"spreadsheet_id={spreadsheet_id} "
                    f"linha={linha} coluna={letters} cell={cell}"
                ),
            )

            auth = await self._auth_context(
                write=False,
                debug=debug,
            )

            first = await self._first_sheet(
                spreadsheet_id,
                auth=auth,
                debug=debug,
            )

            range_a1 = (
                f"{self._quote_sheet_title(first['sheet_title'])}"
                f"!{cell}"
            )

            debug.info(
                "CELL_SELECTED",
                f"range={range_a1!r}",
            )

            await self._status(
                __event_emitter__,
                f"Lendo celula {cell} da primeira aba...",
            )

            data = await self._read_range(
                spreadsheet_id,
                range_a1,
                auth=auth,
                debug=debug,
            )

            values = data.get("values") or []
            value = (
                values[0][0]
                if (
                    values
                    and isinstance(values[0], list)
                    and values[0]
                )
                else ""
            )

            debug.ok(
                "CELL_VALUE",
                f"cell={cell} value={value!r}",
            )

            await self._status(
                __event_emitter__,
                "Celula lida.",
                done=True,
            )

            result = {
                "status": "success",
                "spreadsheet_id": spreadsheet_id,
                "aba": first["sheet_title"],
                "primeira_aba": True,
                "linha": linha,
                "coluna": letters,
                "numero_coluna": number,
                "celula": cell,
                "valor": value,
            }

            debug.finish("success")
            result.update(debug.response_hint())
            return result

        except Exception as exc:
            debug.error(
                "OPERATION_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
            debug.finish("error")

            await self._status(
                __event_emitter__,
                "Falha ao ler a celula.",
                done=True,
            )

            result = {
                "status": "error",
                "mensagem": str(exc),
            }
            result.update(debug.response_hint())
            return result

    async def escrever_celula(
        self,
        spreadsheet_id: str,
        linha: int,
        coluna: str,
        valor: str,
        confirmacao_usuario: bool = False,
        __event_emitter__=None,
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """
        Escreve uma celula da primeira aba.

        Use somente depois que o usuario identificar claramente linha, coluna e valor.

        :param spreadsheet_id: ID da Google Planilha.
        :param linha: Numero da linha, iniciando em 1.
        :param coluna: Letra como C/AA ou numero como 3/27.
        :param valor: Valor a gravar.
        :param confirmacao_usuario: True somente depois de confirmacao explicita do usuario.
        """

        debug = self._new_debug_run(
            "escrever_celula",
            __user__=__user__,
            __chat_id__=__chat_id__,
            __metadata__=__metadata__,
        )

        try:
            spreadsheet_id = self._validate_spreadsheet_id(
                spreadsheet_id
            )
            linha = int(linha)

            if linha < 1:
                raise ToolError(
                    "A linha deve ser maior ou igual a 1."
                )

            letters, number = self._normalize_column(
                coluna
            )
            cell = f"{letters}{linha}"

            debug.info(
                "INPUT",
                (
                    f"spreadsheet_id={spreadsheet_id} "
                    f"linha={linha} coluna={letters} cell={cell} "
                    f"valor={valor!r} "
                    f"confirmacao_usuario={confirmacao_usuario}"
                ),
            )

            if (
                self.valves.exigir_confirmacao_escrita
                and not confirmacao_usuario
            ):
                debug.warning(
                    "CONFIRMATION_REQUIRED",
                    (
                        f"cell={cell} valor_proposto={valor!r} "
                        "write_not_executed=True"
                    ),
                )
                debug.finish("confirmation_required")

                result = {
                    "status": "confirmation_required",
                    "mensagem": (
                        "A escrita exige confirmacao explicita do usuario."
                    ),
                    "spreadsheet_id": spreadsheet_id,
                    "linha": linha,
                    "coluna": letters,
                    "celula": cell,
                    "valor_proposto": valor,
                    "instrucao_para_llm": (
                        "Mostre ao usuario a celula e o valor proposto. "
                        "Somente depois da confirmacao explicita chame novamente "
                        "escrever_celula com confirmacao_usuario=true."
                    ),
                }
                result.update(debug.response_hint())
                return result

            auth = await self._auth_context(
                write=True,
                debug=debug,
            )

            first = await self._first_sheet(
                spreadsheet_id,
                auth=auth,
                debug=debug,
            )

            range_a1 = (
                f"{self._quote_sheet_title(first['sheet_title'])}"
                f"!{cell}"
            )

            url = (
                f"{SHEETS_BASE_URL}/spreadsheets/"
                f"{spreadsheet_id}/values/"
                f"{quote(range_a1, safe='')}"
            )

            debug.info(
                "WRITE_BEGIN",
                (
                    f"range={range_a1!r} "
                    f"value={valor!r} "
                    f"value_input_option={self.valves.modo_escrita}"
                ),
            )

            await self._status(
                __event_emitter__,
                f"Gravando celula {cell} da primeira aba...",
            )

            data = await self._request_json(
                "PUT",
                url,
                auth=auth,
                params={
                    "valueInputOption": self.valves.modo_escrita,
                    "includeValuesInResponse": "true",
                    "responseValueRenderOption": self.valves.valor_renderizado,
                },
                json_body={
                    "range": range_a1,
                    "majorDimension": "ROWS",
                    "values": [[valor]],
                },
                debug=debug,
            )

            updated = data.get("updatedData") or {}
            updated_values = (
                updated.get("values") or []
                if isinstance(updated, dict)
                else []
            )
            confirmed = (
                updated_values[0][0]
                if (
                    updated_values
                    and isinstance(updated_values[0], list)
                    and updated_values[0]
                )
                else ""
            )

            debug.ok(
                "WRITE_CONFIRMED",
                (
                    f"cell={cell} "
                    f"value_sent={valor!r} "
                    f"value_confirmed={confirmed!r} "
                    f"updated_cells={data.get('updatedCells')}"
                ),
            )

            await self._status(
                __event_emitter__,
                "Celula atualizada.",
                done=True,
            )

            result = {
                "status": "success",
                "mensagem": "Celula atualizada com sucesso.",
                "spreadsheet_id": spreadsheet_id,
                "aba": first["sheet_title"],
                "primeira_aba": True,
                "linha": linha,
                "coluna": letters,
                "numero_coluna": number,
                "celula": cell,
                "valor_enviado": valor,
                "valor_confirmado": confirmed,
                "modo_escrita": self.valves.modo_escrita,
                "updated_cells": data.get("updatedCells"),
                "service_account_email": auth.get(
                    "service_account_email",
                    "",
                ),
            }

            debug.finish("success")
            result.update(debug.response_hint())
            return result

        except Exception as exc:
            debug.error(
                "OPERATION_ERROR",
                f"{type(exc).__name__}: {exc}",
            )
            debug.finish("error")

            await self._status(
                __event_emitter__,
                "Falha ao escrever a celula.",
                done=True,
            )

            result = {
                "status": "error",
                "mensagem": str(exc),
            }
            result.update(debug.response_hint())
            return result

    async def obter_log_debug(
        self,
        run_id: str = "",
        __user__: Optional[dict] = None,
        __chat_id__: Optional[str] = None,
        __metadata__: Optional[dict] = None,
    ) -> dict:
        """
        Retorna o log completo de debug.

        Use SOMENTE quando debug=true e o usuario pedir explicitamente:
        "mostre o log", "todo o log", "detalhes de debug", "o fluxo completo"
        ou expressao equivalente.

        Se run_id estiver vazio, retorna o log da ultima execucao desta sessao.
        Se run_id for informado, ele precisa pertencer a ultima execucao registrada
        para esta mesma sessao.

        :param run_id: Identificador opcional da execucao de debug.
        """

        if not self.valves.debug:
            return {
                "status": "debug_disabled",
                "mensagem": (
                    "O modo debug esta desativado nas Valves."
                ),
            }

        self._cleanup_debug_logs()

        root = Path(self.valves.debug_dir).expanduser()
        session_key = self._session_key(
            __user__,
            __chat_id__,
            __metadata__,
        )
        pointer_path = (
            root
            / "sessions"
            / f"{session_key}.json"
        )

        if not pointer_path.exists():
            return {
                "status": "not_found",
                "mensagem": (
                    "Nenhum log de debug foi encontrado para esta sessao."
                ),
            }

        try:
            pointer = json.loads(
                pointer_path.read_text(
                    encoding="utf-8"
                )
            )
        except Exception as exc:
            return {
                "status": "error",
                "mensagem": (
                    "Nao foi possivel ler o indice do log de debug: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

        last_run_id = str(
            pointer.get("run_id") or ""
        )

        requested_run_id = str(
            run_id or ""
        ).strip()

        if requested_run_id and requested_run_id != last_run_id:
            return {
                "status": "not_found",
                "mensagem": (
                    "O run_id informado nao corresponde a ultima "
                    "execucao desta sessao."
                ),
                "ultimo_run_id": last_run_id,
            }

        log_path = Path(
            str(pointer.get("log_path") or "")
        )

        if not log_path.exists():
            return {
                "status": "not_found",
                "mensagem": (
                    "O arquivo do log de debug nao esta mais disponivel."
                ),
                "run_id": last_run_id,
            }

        try:
            full_log = log_path.read_text(
                encoding="utf-8"
            )
        except Exception as exc:
            return {
                "status": "error",
                "mensagem": (
                    "Nao foi possivel ler o log de debug: "
                    f"{type(exc).__name__}: {exc}"
                ),
            }

        truncated = False
        max_chars = int(
            self.valves.debug_max_chars
        )

        if len(full_log) > max_chars:
            full_log = (
                "... INICIO DO LOG OMITIDO POR LIMITE DE TAMANHO ...\n"
                + full_log[-max_chars:]
            )
            truncated = True

        return {
            "status": "success",
            "run_id": last_run_id,
            "operacao": pointer.get("operation"),
            "resultado_operacao": pointer.get("status"),
            "atualizado_em": pointer.get("updated_at"),
            "truncado": truncated,
            "log": full_log,
            "instrucao_para_llm": (
                "O usuario pediu explicitamente o log. "
                "Apresente o conteudo de log de forma legivel. "
                "Nao invente eventos que nao estejam presentes."
            ),
        }
