"""
title: USP Assina v3.0.2
author: Agents4Gov
description: Cria e envia um único PDF no USP Assina usando Playwright MCP, com autenticação USP coletada pelo Open WebUI, execução determinística, logs persistentes e diagnóstico opcional.
required_open_webui_version: 0.11.0
version: 3.0.2
license: MIT
"""

from __future__ import annotations

import base64
import json
import re
import time
import traceback
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from mcp import ClientSession
from pydantic import BaseModel, Field

try:
    from mcp.client.streamable_http import streamable_http_client
except ImportError:
    from mcp.client.streamable_http import (
        streamablehttp_client as streamable_http_client,
    )


EMAIL_RE = re.compile(r"^[^@\s;]+@[^@\s;]+\.[^@\s;]+$")
NUSP_RE = re.compile(r"^\d+$")

TIPO_ASSINATURA = "ASSINATURA ELETRÔNICA AVANÇADA (ID digital USP)"


class ToolError(RuntimeError):
    pass


class RunLogger:
    def __init__(
        self,
        *,
        requested_root: str,
        debug: bool,
        max_debug_chars: int,
    ):
        self.debug = bool(debug)
        self.max_debug_chars = int(max_debug_chars)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = uuid.uuid4().hex[:8]
        self.run_id = f"{stamp}_{suffix}"

        preferred = Path(requested_root).expanduser()
        fallback = Path("/tmp/usp-assina-v3-logs")

        try:
            preferred.mkdir(parents=True, exist_ok=True)
            self.root = preferred
        except Exception:
            fallback.mkdir(parents=True, exist_ok=True)
            self.root = fallback

        self.run_dir = self.root / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

        self.log_path = self.run_dir / "run.log"
        self.result_path = self.run_dir / "result.json"
        self.state_path = self.run_dir / "state.json"

        self.lines: list[str] = []
        self.current_stage = "INITIALIZING"
        self.started = time.monotonic()

        self.info(
            "RUN_START",
            f"run_id={self.run_id} debug={self.debug}",
        )

    @staticmethod
    def _clean(value: Any) -> str:
        text = str(value)
        text = text.replace("\r", " ").replace("\n", " ")
        return text

    def write(
        self,
        level: str,
        event: str,
        message: str,
    ) -> None:
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.monotonic() - self.started

        line = (
            f"[{stamp}]"
            f"[+{elapsed:08.2f}s]"
            f"[USP-ASSINA-V3.0.2]"
            f"[{level}]"
            f"[{event}] "
            f"{self._clean(message)}"
        )

        self.lines.append(line)

        try:
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

    def stage(self, name: str) -> None:
        self.current_stage = name
        self.info("STAGE", name)
        self.save_state(status="running")

    def save_state(
        self,
        *,
        status: str,
        extra: Optional[dict[str, Any]] = None,
    ) -> None:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "status": status,
            "stage": self.current_stage,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

        if extra:
            payload.update(extra)

        try:
            self.state_path.write_text(
                json.dumps(
                    payload,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def save_result(self, result: dict[str, Any]) -> None:
        try:
            self.result_path.write_text(
                json.dumps(
                    result,
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def debug_text(self) -> str:
        text = "\n".join(self.lines)

        if len(text) > self.max_debug_chars:
            text = (
                "... log truncado; exibindo a parte final ...\n"
                + text[-self.max_debug_chars :]
            )

        return text


class Tools:
    class Valves(BaseModel):
        mcp_url: str = Field(
            default="http://PLAYWRIGHT_MCP_HOST:35010/mcp",
            description="URL Streamable HTTP do Playwright MCP.",
        )

        portal_url: str = Field(
            default="https://portalservicos.usp.br/",
            description="URL do Portal de Serviços USP.",
        )

        assina_url: str = Field(
            default="https://portalservicos.usp.br/assina",
            description="URL do USP Assina.",
        )

        timeout_ms: int = Field(
            default=20000,
            ge=5000,
            le=120000,
            description="Timeout padrão das ações Playwright em milissegundos.",
        )

        max_pdf_mb: int = Field(
            default=10,
            ge=1,
            le=10,
            description="Tamanho máximo do PDF anexado. O USP Assina informa limite de 10 MB por documento.",
        )

        debug: bool = Field(
            default=False,
            description=(
                "Quando ativado, erros retornam ao usuário o log detalhado "
                "da execução. A senha nunca é registrada."
            ),
        )

        log_dir: str = Field(
            default="/app/backend/data/usp-assina-v3-logs",
            description=(
                "Diretório persistente de logs. Se indisponível, usa /tmp."
            ),
        )

        max_debug_chars: int = Field(
            default=24000,
            ge=4000,
            le=100000,
            description="Máximo de caracteres de log devolvidos em modo debug.",
        )

        upload_settle_ms: int = Field(
            default=2500,
            ge=500,
            le=15000,
            description=(
                "Tempo de estabilização após setInputFiles antes da validação do upload."
            ),
        )

    def __init__(self):
        self.valves = self.Valves()

    async def _status(
        self,
        emitter,
        description: str,
        *,
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
    def _mask_nusp(value: str) -> str:
        value = str(value or "")

        if len(value) <= 3:
            return "***"

        return "*" * (len(value) - 3) + value[-3:]

    @staticmethod
    def _normalize_emails(raw: str) -> str:
        values = [
            item.strip()
            for item in str(raw or "").split(";")
            if item.strip()
        ]

        if not values:
            raise ToolError(
                "Informe pelo menos um e-mail para cópia."
            )

        invalid = [
            item
            for item in values
            if not EMAIL_RE.fullmatch(item)
        ]

        if invalid:
            raise ToolError(
                "E-mails inválidos: " + ", ".join(invalid)
            )

        return ";".join(values)

    @staticmethod
    def _normalize_participants(raw: str) -> list[str]:
        values = [
            item.strip()
            for item in re.split(r"[;,]", str(raw or ""))
            if item.strip()
        ]

        if not values:
            raise ToolError(
                "Informe pelo menos um Número USP participante."
            )

        invalid = [
            item
            for item in values
            if not NUSP_RE.fullmatch(item)
        ]

        if invalid:
            raise ToolError(
                "Números USP inválidos: " + ", ".join(invalid)
            )

        deduped: list[str] = []
        seen: set[str] = set()

        for item in values:
            if item not in seen:
                seen.add(item)
                deduped.append(item)

        return deduped

    def _load_single_pdf(
        self,
        files: Any,
        logger: RunLogger,
    ) -> tuple[str, bytes]:
        logger.stage("PDF_VALIDATE")

        files = files or []

        logger.info(
            "PDF_FILES",
            f"attachments_count={len(files)}",
        )

        if len(files) == 0:
            raise ToolError(
                "Nenhum PDF anexado. Anexe exatamente um arquivo PDF."
            )

        if len(files) != 1:
            raise ToolError(
                "Esta operação aceita exatamente um arquivo PDF por vez."
            )

        item = files[0]
        file_info = (
            item.get("file", item)
            if isinstance(item, dict)
            else {}
        )

        filename = str(
            file_info.get("filename")
            or file_info.get("name")
            or ""
        ).strip()

        path_raw = str(
            file_info.get("path")
            or ""
        ).strip()

        logger.info(
            "PDF_METADATA",
            f"filename={Path(filename).name!r} path_available={bool(path_raw)}",
        )

        if not filename:
            raise ToolError(
                "O Open WebUI não informou o nome do arquivo anexado."
            )

        if Path(filename).suffix.lower() != ".pdf":
            raise ToolError(
                "O único arquivo anexado deve ser um PDF."
            )

        if not path_raw:
            raise ToolError(
                "O Open WebUI não forneceu o caminho local do PDF anexado."
            )

        path = Path(path_raw)

        if not path.exists():
            raise ToolError(
                "O PDF anexado não existe no armazenamento do Open WebUI."
            )

        if not path.is_file():
            raise ToolError(
                "O caminho recebido para o PDF não corresponde a um arquivo."
            )

        size = path.stat().st_size
        max_bytes = int(self.valves.max_pdf_mb) * 1024 * 1024

        logger.info(
            "PDF_SIZE",
            f"bytes={size} max_bytes={max_bytes}",
        )

        if size <= 0:
            raise ToolError(
                "O PDF anexado está vazio."
            )

        if size > max_bytes:
            raise ToolError(
                f"O PDF excede o limite configurado de "
                f"{self.valves.max_pdf_mb} MB."
            )

        data = path.read_bytes()

        if not data.startswith(b"%PDF-"):
            raise ToolError(
                "O conteúdo anexado não possui assinatura PDF válida."
            )

        logger.ok(
            "PDF_VALID",
            f"filename={Path(filename).name!r} bytes={len(data)}",
        )

        return Path(filename).name, data

    async def _ask_credentials(
        self,
        event_call,
        logger: RunLogger,
    ) -> tuple[str, str]:
        logger.stage("AUTH_INPUT")

        if event_call is None:
            raise ToolError(
                "A interface segura de autenticação do Open WebUI não está disponível."
            )

        nusp: Optional[str] = None

        for attempt in range(1, 4):
            logger.info(
                "AUTH_NUSP_PROMPT",
                f"attempt={attempt}",
            )

            answer = await event_call(
                {
                    "type": "input",
                    "data": {
                        "title": "Autenticação USP",
                        "message": (
                            "Informe seu Número USP para esta operação."
                        ),
                        "placeholder": "Número USP",
                    },
                }
            )

            if answer is False or answer is None:
                logger.warning(
                    "AUTH_CANCELLED",
                    "Número USP não informado.",
                )
                raise ToolError(
                    "Autenticação cancelada pelo usuário."
                )

            candidate = str(answer).strip()

            if NUSP_RE.fullmatch(candidate):
                nusp = candidate
                logger.ok(
                    "AUTH_NUSP_ACCEPTED",
                    f"nusp={self._mask_nusp(candidate)}",
                )
                break

            logger.warning(
                "AUTH_NUSP_INVALID",
                "Número USP inválido.",
            )

        if not nusp:
            raise ToolError(
                "Número USP inválido após três tentativas."
            )

        logger.info(
            "AUTH_PASSWORD_PROMPT",
            "Solicitando senha em campo protegido.",
        )

        password = await event_call(
            {
                "type": "input",
                "data": {
                    "title": "Autenticação USP",
                    "message": (
                        "Informe sua senha USP. "
                        "Ela será usada somente nesta execução."
                    ),
                    "placeholder": "Senha USP",
                    "type": "password",
                },
            }
        )

        if password is False or password is None:
            logger.warning(
                "AUTH_CANCELLED",
                "Senha não informada.",
            )
            raise ToolError(
                "Autenticação cancelada pelo usuário."
            )

        password_text = str(password)

        if not password_text:
            raise ToolError(
                "A senha USP não pode estar vazia."
            )

        logger.ok(
            "AUTH_PASSWORD_RECEIVED",
            "Senha recebida. Conteúdo não registrado.",
        )

        return nusp, password_text

    @staticmethod
    def _result_text(result: Any) -> str:
        parts: list[str] = []

        for block in getattr(result, "content", None) or []:
            text = getattr(block, "text", None)

            if isinstance(text, str):
                parts.append(text)

        return "\n".join(parts)

    @staticmethod
    def _unwrap_exception(exc: BaseException) -> list[str]:
        messages: list[str] = []

        def visit(item: BaseException, depth: int = 0) -> None:
            prefix = "  " * depth
            messages.append(
                f"{prefix}{type(item).__name__}: {item}"
            )

            children = getattr(item, "exceptions", None)

            if children:
                for child in children:
                    if isinstance(child, BaseException):
                        visit(child, depth + 1)

        visit(exc)
        return messages

    @staticmethod
    def _parse_automation_result(
        text: str,
        logger: RunLogger,
    ) -> dict[str, Any]:
        logger.info(
            "MCP_RESULT_TEXT",
            f"text_length={len(text or '')}",
        )

        if not isinstance(text, str) or not text.strip():
            raise ToolError(
                "O Playwright não retornou conteúdo textual para a automação."
            )

        if "### Result" in text:
            result_section = text.split(
                "### Result",
                1,
            )[1]

            for delimiter in (
                "### Ran Playwright code",
                "### Page state",
                "### Open tabs",
            ):
                if delimiter in result_section:
                    result_section = result_section.split(
                        delimiter,
                        1,
                    )[0]

            result_section = result_section.strip()

            logger.info(
                "MCP_RESULT_SECTION",
                f"chars={len(result_section)}",
            )

            if result_section:
                try:
                    value, _ = json.JSONDecoder().raw_decode(
                        result_section
                    )

                    if isinstance(value, dict):
                        return value

                    if isinstance(value, str):
                        try:
                            nested = json.loads(value)
                            if isinstance(nested, dict):
                                return nested
                        except Exception:
                            pass

                except json.JSONDecodeError as exc:
                    logger.warning(
                        "MCP_RESULT_JSON",
                        f"raw_decode_failed={exc}",
                    )

        stripped = text.strip()

        try:
            value = json.loads(stripped)

            if isinstance(value, dict):
                return value

            if isinstance(value, str):
                nested = json.loads(value)

                if isinstance(nested, dict):
                    return nested

        except Exception as exc:
            logger.warning(
                "MCP_RESULT_FALLBACK",
                f"json_load_failed={type(exc).__name__}: {exc}",
            )

        raise ToolError(
            "Não foi possível interpretar o resultado final do Playwright."
        )

    @staticmethod
    def _safe_mcp_error_text(
        text: str,
        *,
        password: str,
        pdf_b64: str,
    ) -> str:
        safe = str(text or "")

        if password:
            safe = safe.replace(
                password,
                "[PASSWORD_REDACTED]",
            )

        if pdf_b64:
            safe = safe.replace(
                pdf_b64,
                "[PDF_BASE64_REDACTED]",
            )

        # Remove a seção de código executado, que pode conter credenciais
        # e o conteúdo Base64 do PDF.
        if "### Ran Playwright code" in safe:
            safe = safe.split(
                "### Ran Playwright code",
                1,
            )[0]

        return safe[:6000]

    def _build_playwright_code(
        self,
        *,
        nusp: str,
        password: str,
        title: str,
        emails: str,
        participants: list[str],
        pdf_name: str,
        pdf_bytes: bytes,
    ) -> tuple[str, str]:
        pdf_b64 = base64.b64encode(
            pdf_bytes
        ).decode("ascii")

        values = {
            "PORTAL_URL": self.valves.portal_url,
            "ASSINA_URL": self.valves.assina_url,
            "TIMEOUT": int(self.valves.timeout_ms),
            "UPLOAD_SETTLE_MS": int(
                self.valves.upload_settle_ms
            ),
            "NUSP": nusp,
            "PASSWORD": password,
            "TITLE": title,
            "EMAILS": emails,
            "PARTICIPANTS": participants,
            "SIGNATURE_TYPE": TIPO_ASSINATURA,
            "PDF_NAME": pdf_name,
            "PDF_B64": pdf_b64,
        }

        js = {
            key: json.dumps(
                value,
                ensure_ascii=False,
            )
            for key, value in values.items()
        }

        code = f"""
async (page) => {{
    const PORTAL_URL = {js["PORTAL_URL"]};
    const ASSINA_URL = {js["ASSINA_URL"]};
    const TIMEOUT = {js["TIMEOUT"]};
    const UPLOAD_SETTLE_MS = {js["UPLOAD_SETTLE_MS"]};
    const NUSP = {js["NUSP"]};
    const PASSWORD = {js["PASSWORD"]};
    const TITLE = {js["TITLE"]};
    const EMAILS = {js["EMAILS"]};
    const PARTICIPANTS = {js["PARTICIPANTS"]};
    const SIGNATURE_TYPE = {js["SIGNATURE_TYPE"]};
    const PDF_NAME = {js["PDF_NAME"]};
    const PDF_B64 = {js["PDF_B64"]};

    const events = [];
    const startedAt = Date.now();

    let step = "initializing";
    let sendClicked = false;

    page.setDefaultTimeout(TIMEOUT);

    const nowMs = () => Date.now() - startedAt;

    const log = (level, event, message, extra = null) => {{
        const record = {{
            t_ms: nowMs(),
            level,
            event,
            step,
            message
        }};

        if (extra !== null && extra !== undefined) {{
            record.extra = extra;
        }}

        events.push(record);
    }};

    const firstVisible = async (
        locator,
        description,
        timeout = TIMEOUT
    ) => {{
        const item = locator.first();

        try {{
            await item.waitFor({{
                state: "visible",
                timeout
            }});

            return item;
        }} catch (error) {{
            throw new Error(
                `Estado inválido: ${{description}} não está visível.`
            );
        }}
    }};

    const assertValue = async (
        locator,
        expected,
        description
    ) => {{
        const observed = await locator.inputValue();

        if (observed !== expected) {{
            throw new Error(
                `Falha ao validar o campo ${{description}}.`
            );
        }}
    }};

    const runStep = async (name, operation) => {{
        step = name;
        const stepStarted = Date.now();

        log(
            "INFO",
            "STEP_BEGIN",
            name
        );

        try {{
            const result = await operation();

            log(
                "OK",
                "STEP_OK",
                name,
                {{
                    elapsed_ms: Date.now() - stepStarted
                }}
            );

            return result;
        }} catch (error) {{
            log(
                "ERROR",
                "STEP_FAILED",
                String(
                    error && error.message
                        ? error.message
                        : error
                ),
                {{
                    elapsed_ms: Date.now() - stepStarted
                }}
            );

            throw error;
        }}
    }};

    const responseEvents = [];

    const onResponse = async (response) => {{
        try {{
            const url = response.url();

            if (url.includes("usp.br")) {{
                responseEvents.push({{
                    status: response.status(),
                    url: url.slice(0, 500)
                }});

                if (responseEvents.length > 30) {{
                    responseEvents.shift();
                }}
            }}
        }} catch (_) {{}}
    }};

    page.on("response", onResponse);

    try {{
        await runStep(
            "01_open_portal",
            async () => {{
                await page.goto(
                    PORTAL_URL,
                    {{
                        waitUntil: "domcontentloaded"
                    }}
                );

                log(
                    "INFO",
                    "PAGE",
                    "Portal USP aberto.",
                    {{
                        url: page.url(),
                        title: await page.title()
                    }}
                );
            }}
        );

        await runStep(
            "02_login",
            async () => {{
                const profile = page.getByRole(
                    "button",
                    {{
                        name: new RegExp(NUSP)
                    }}
                );

                let alreadyLogged = false;

                if (await profile.count()) {{
                    alreadyLogged = await profile
                        .first()
                        .isVisible()
                        .catch(() => false);
                }}

                if (alreadyLogged) {{
                    log(
                        "INFO",
                        "LOGIN_SKIPPED",
                        "Sessão USP já autenticada."
                    );

                    return;
                }}

                const numero = await firstVisible(
                    page.getByRole(
                        "textbox",
                        {{
                            name: /número usp/i
                        }}
                    ),
                    "campo Número USP"
                );

                const senha = await firstVisible(
                    page.getByRole(
                        "textbox",
                        {{
                            name: /senha/i
                        }}
                    ),
                    "campo Senha"
                );

                const acessar = await firstVisible(
                    page.getByRole(
                        "button",
                        {{
                            name: /acessar minha conta/i
                        }}
                    ),
                    "botão Acessar minha conta"
                );

                await numero.fill(NUSP);
                await assertValue(
                    numero,
                    NUSP,
                    "Número USP"
                );

                await senha.fill(PASSWORD);

                if (!(await senha.inputValue())) {{
                    throw new Error(
                        "O campo Senha permaneceu vazio após o preenchimento."
                    );
                }}

                log(
                    "INFO",
                    "LOGIN_FORM",
                    "Credenciais preenchidas. Senha não registrada."
                );

                await acessar.click();

                await firstVisible(
                    profile,
                    "perfil autenticado",
                    Math.max(
                        TIMEOUT,
                        30000
                    )
                );

                log(
                    "OK",
                    "LOGIN_OK",
                    "Autenticação USP confirmada."
                );
            }}
        );

        await runStep(
            "03_open_usp_assina",
            async () => {{
                await page.goto(
                    ASSINA_URL,
                    {{
                        waitUntil: "domcontentloaded"
                    }}
                );

                await firstVisible(
                    page.getByRole(
                        "heading",
                        {{
                            name: /assinaturas\\s*-\\s*listar/i
                        }}
                    ),
                    "título Assinaturas - Listar"
                );

                await firstVisible(
                    page.getByRole(
                        "link",
                        {{
                            name: /novo documento/i
                        }}
                    ),
                    "link Novo documento"
                );

                log(
                    "INFO",
                    "PAGE",
                    "USP Assina aberto.",
                    {{
                        url: page.url(),
                        title: await page.title()
                    }}
                );
            }}
        );

        await runStep(
            "04_open_new_document",
            async () => {{
                await page
                    .getByRole(
                        "link",
                        {{
                            name: /novo documento/i
                        }}
                    )
                    .click();

                await firstVisible(
                    page.getByRole(
                        "heading",
                        {{
                            name: /lote de documentos/i
                        }}
                    ),
                    "formulário Lote de documentos"
                );

                await firstVisible(
                    page.getByRole(
                        "textbox",
                        {{
                            name: /^título$/i
                        }}
                    ),
                    "campo Título"
                );

                await firstVisible(
                    page.getByRole(
                        "combobox",
                        {{
                            name: /tipo de assinatura.*lote/i
                        }}
                    ),
                    "Tipo de assinatura do lote"
                );
            }}
        );

        await runStep(
            "05_fill_batch",
            async () => {{
                const titleField = await firstVisible(
                    page.getByRole(
                        "textbox",
                        {{
                            name: /^título$/i
                        }}
                    ),
                    "campo Título"
                );

                await titleField.fill(TITLE);
                await assertValue(
                    titleField,
                    TITLE,
                    "Título"
                );

                const signatureSelector = await firstVisible(
                    page.getByRole(
                        "combobox",
                        {{
                            name: /tipo de assinatura.*lote/i
                        }}
                    ),
                    "Tipo de assinatura do lote"
                );

                await signatureSelector.selectOption({{
                    label: SIGNATURE_TYPE
                }});

                const selectedSignature = (
                    await signatureSelector
                        .locator("option:checked")
                        .innerText()
                ).trim();

                if (
                    selectedSignature
                    !== SIGNATURE_TYPE
                ) {{
                    throw new Error(
                        `Tipo de assinatura inesperado: ${{selectedSignature}}`
                    );
                }}

                await page.waitForTimeout(3000);

                const emailField = await firstVisible(
                    page.getByRole(
                        "textbox",
                        {{
                            name: /quando finalizado, enviar email para/i
                        }}
                    ),
                    "campo de e-mails de finalização"
                );

                await emailField.fill(EMAILS);
                await assertValue(
                    emailField,
                    EMAILS,
                    "e-mails de finalização"
                );

                log(
                    "OK",
                    "BATCH_FIELDS",
                    "Título, tipo de assinatura e e-mails preenchidos."
                );
            }}
        );

        await runStep(
            "06_save_batch",
            async () => {{
                const saveButton = await firstVisible(
                    page.getByRole(
                        "button",
                        {{
                            name: /(?:^|\\s)salvar\\s*$/i
                        }}
                    ),
                    "botão Salvar"
                );

                await saveButton.click();
                await page.waitForTimeout(3000);

                await firstVisible(
                    page.getByRole(
                        "heading",
                        {{
                            name: /^participantes$/i
                        }}
                    ),
                    "seção Participantes"
                );

                await firstVisible(
                    page.getByRole(
                        "heading",
                        {{
                            name: /^arquivos$/i
                        }}
                    ),
                    "seção Arquivos"
                );
            }}
        );

        await runStep(
            "07_add_signers",
            async () => {{
                for (
                    let index = 0;
                    index < PARTICIPANTS.length;
                    index++
                ) {{
                    const participant = PARTICIPANTS[index];

                    log(
                        "INFO",
                        "SIGNER_BEGIN",
                        `Adicionando participante ${{index + 1}}/${{PARTICIPANTS.length}}.`
                    );

                    const nuspField = await firstVisible(
                        page.getByRole(
                            "textbox",
                            {{
                                name: /^número usp$/i
                            }}
                        ),
                        `campo Número USP do participante ${{participant}}`
                    );

                    await nuspField.fill(participant);
                    await assertValue(
                        nuspField,
                        participant,
                        `Número USP do participante ${{participant}}`
                    );

                    const searchButton = await firstVisible(
                        page.getByRole(
                            "button",
                            {{
                                name: /^buscar$/i
                            }}
                        ),
                        "botão Buscar participante"
                    );

                    await searchButton.click();

                    const searchHeading = await firstVisible(
                        page.getByRole(
                            "heading",
                            {{
                                name: /resultados da busca/i
                            }}
                        ),
                        "Resultados da busca"
                    );

                    const searchScope = searchHeading.locator(
                        "xpath=../.."
                    );

                    const resultNumber = await firstVisible(
                        searchScope
                            .getByText(
                                new RegExp(
                                    `Número USP:\\\\s*${{participant}}`,
                                    "i"
                                )
                            )
                            .first(),
                        `resultado do Número USP ${{participant}}`
                    );

                    const item = resultNumber.locator(
                        "xpath=ancestor::li[1]"
                    );

                    const profileSelector = await firstVisible(
                        item.getByRole("combobox"),
                        `seletor de perfil de ${{participant}}`
                    );

                    await profileSelector.selectOption({{
                        label: "Assinante"
                    }});

                    const selectedProfile = (
                        await profileSelector
                            .locator("option:checked")
                            .innerText()
                    ).trim();

                    if (
                        selectedProfile.toLocaleLowerCase()
                        !== "assinante"
                    ) {{
                        throw new Error(
                            `Não foi possível selecionar Assinante para ${{participant}}.`
                        );
                    }}

                    const addButton = await firstVisible(
                        item.getByRole(
                            "button",
                            {{
                                name: /adicionar/i
                            }}
                        ),
                        `botão Adicionar de ${{participant}}`
                    );

                    if (await addButton.isDisabled()) {{
                        log(
                            "WARN",
                            "ADD_BUTTON_DISABLED",
                            "Aplicando workaround para habilitar o botão Adicionar."
                        );

                        await addButton.evaluate(
                            element => element.removeAttribute(
                                "disabled"
                            )
                        );
                    }}

                    await addButton.click();

                    await firstVisible(
                        page
                            .getByText(
                                new RegExp(
                                    `Número USP:\\\\s*${{participant}}.*Perfil:\\\\s*Assinante`,
                                    "is"
                                )
                            )
                            .first(),
                        `participante ${{participant}} vinculado como Assinante`,
                        Math.max(
                            TIMEOUT,
                            30000
                        )
                    );

                    log(
                        "OK",
                        "SIGNER_OK",
                        `Participante ${{index + 1}}/${{PARTICIPANTS.length}} adicionado como Assinante.`
                    );
                }}
            }}
        );

        await runStep(
            "08_upload_pdf",
            async () => {{
                log(
                    "INFO",
                    "UPLOAD_BEGIN",
                    "Iniciando upload do PDF diretamente no contexto do Chromium.",
                    {{
                        filename: PDF_NAME,
                        base64_chars: PDF_B64.length
                    }}
                );

                const preferredInputs = page.locator(
                    'input[data-test="hidden-file-input"]'
                );

                const genericInputs = page.locator(
                    'input[type="file"]'
                );

                const preferredCount = await preferredInputs.count();
                const genericCount = await genericInputs.count();

                log(
                    "INFO",
                    "UPLOAD_INPUT_DISCOVERY",
                    "Procurando input de arquivo.",
                    {{
                        preferred_count: preferredCount,
                        generic_count: genericCount
                    }}
                );

                let fileInput = null;

                if (preferredCount > 0) {{
                    fileInput = preferredInputs.first();
                }} else if (genericCount > 0) {{
                    fileInput = genericInputs.first();
                }} else {{
                    throw new Error(
                        "Nenhum input[type=file] foi localizado na página."
                    );
                }}

                await fileInput.waitFor({{
                    state: "attached",
                    timeout: TIMEOUT
                }});

                const inputInfo = await fileInput.evaluate(
                    input => ({{
                        type: input.getAttribute("type"),
                        accept: input.getAttribute("accept"),
                        multiple: Boolean(input.multiple),
                        disabled: Boolean(input.disabled),
                        dataTest: input.getAttribute("data-test"),
                        name: input.getAttribute("name"),
                        id: input.getAttribute("id"),
                        filesBefore: Array
                            .from(input.files || [])
                            .map(file => ({{
                                name: file.name,
                                size: file.size,
                                type: file.type
                            }}))
                    }})
                );

                log(
                    "INFO",
                    "UPLOAD_INPUT",
                    "Input de arquivo localizado.",
                    inputInfo
                );

                if (inputInfo.disabled) {{
                    throw new Error(
                        "O input de arquivo está desabilitado."
                    );
                }}

                const uploadResult = await fileInput.evaluate(
                    (input, payload) => {{
                        if (!(input instanceof HTMLInputElement)) {{
                            throw new Error(
                                "O elemento localizado não é HTMLInputElement."
                            );
                        }}

                        if (input.type !== "file") {{
                            throw new Error(
                                `O input localizado possui type=${{input.type}} em vez de file.`
                            );
                        }}

                        if (
                            typeof atob !== "function"
                            || typeof File !== "function"
                            || typeof DataTransfer !== "function"
                        ) {{
                            throw new Error(
                                "O contexto do Chromium não oferece atob/File/DataTransfer."
                            );
                        }}

                        const binary = atob(payload.base64);
                        const bytes = new Uint8Array(binary.length);

                        for (
                            let index = 0;
                            index < binary.length;
                            index++
                        ) {{
                            bytes[index] = binary.charCodeAt(index);
                        }}

                        const magic = String.fromCharCode(
                            ...bytes.slice(0, 5)
                        );

                        if (magic !== "%PDF-") {{
                            throw new Error(
                                `PDF reconstruído com assinatura inválida: ${{magic}}`
                            );
                        }}

                        const file = new File(
                            [bytes],
                            payload.name,
                            {{
                                type: "application/pdf",
                                lastModified: Date.now()
                            }}
                        );

                        const transfer = new DataTransfer();
                        transfer.items.add(file);

                        input.files = transfer.files;

                        input.dispatchEvent(
                            new Event(
                                "input",
                                {{
                                    bubbles: true,
                                    composed: true
                                }}
                            )
                        );

                        input.dispatchEvent(
                            new Event(
                                "change",
                                {{
                                    bubbles: true,
                                    composed: true
                                }}
                            )
                        );

                        return {{
                            reconstructed_bytes: bytes.length,
                            magic,
                            input_value: input.value,
                            files: Array
                                .from(input.files || [])
                                .map(currentFile => ({{
                                    name: currentFile.name,
                                    size: currentFile.size,
                                    type: currentFile.type
                                }}))
                        }};
                    }},
                    {{
                        name: PDF_NAME,
                        base64: PDF_B64
                    }}
                );

                log(
                    "OK",
                    "UPLOAD_BROWSER_FILE",
                    "PDF reconstruído como File e associado ao input no Chromium.",
                    uploadResult
                );

                const selectedFiles = Array.isArray(
                    uploadResult.files
                )
                    ? uploadResult.files
                    : [];

                log(
                    "INFO",
                    "UPLOAD_FILELIST",
                    "FileList após atribuição no contexto do Chromium.",
                    selectedFiles
                );

                if (selectedFiles.length !== 1) {{
                    throw new Error(
                        `Esperado exatamente um arquivo no input; observado=${{selectedFiles.length}}.`
                    );
                }}

                if (
                    selectedFiles[0].name
                    !== PDF_NAME
                ) {{
                    throw new Error(
                        `Nome do arquivo divergente após upload. esperado=${{PDF_NAME}} observado=${{selectedFiles[0].name}}`
                    );
                }}

                if (
                    selectedFiles[0].size
                    !== uploadResult.reconstructed_bytes
                ) {{
                    throw new Error(
                        `Tamanho do arquivo divergente no FileList. esperado=${{uploadResult.reconstructed_bytes}} observado=${{selectedFiles[0].size}}`
                    );
                }}

                if (
                    selectedFiles[0].type
                    !== "application/pdf"
                ) {{
                    log(
                        "WARN",
                        "UPLOAD_MIME",
                        "O MIME type observado difere de application/pdf.",
                        {{
                            observed: selectedFiles[0].type
                        }}
                    );
                }}

                log(
                    "OK",
                    "UPLOAD_EVENTS_DISPATCHED",
                    "Eventos input e change disparados para o formulário do USP Assina."
                );

                await page.waitForTimeout(
                    UPLOAD_SETTLE_MS
                );

                const uploadedName = page
                    .getByText(
                        PDF_NAME,
                        {{
                            exact: false
                        }}
                    )
                    .first();

                let fileVisible = false;

                if (await uploadedName.count()) {{
                    fileVisible = await uploadedName
                        .isVisible()
                        .catch(() => false);
                }}

                log(
                    fileVisible ? "OK" : "WARN",
                    "UPLOAD_UI_NAME",
                    fileVisible
                        ? "Nome do PDF apareceu na interface."
                        : "Nome do PDF ainda não apareceu visivelmente na interface.",
                    {{
                        filename: PDF_NAME
                    }}
                );

                let filesSectionText = "";

                try {{
                    const filesHeading = page
                        .getByRole(
                            "heading",
                            {{
                                name: /^arquivos$/i
                            }}
                        )
                        .first();

                    if (await filesHeading.count()) {{
                        filesSectionText = (
                            await filesHeading
                                .locator("xpath=../..")
                                .innerText()
                        ).slice(0, 4000);
                    }}
                }} catch (_) {{}}

                log(
                    "INFO",
                    "UPLOAD_UI_SECTION",
                    "Estado textual da seção Arquivos após o upload.",
                    {{
                        files_section_excerpt: filesSectionText
                    }}
                );

                log(
                    "INFO",
                    "UPLOAD_NETWORK",
                    "Últimas respostas HTTP observadas durante a execução.",
                    responseEvents.slice(-20)
                );

                if (!fileVisible) {{
                    throw new Error(
                        `O PDF ${{PDF_NAME}} está presente no FileList, mas não apareceu na seção Arquivos do USP Assina após os eventos input/change.`
                    );
                }}
            }}
        );

        await runStep(
            "09_validate_ready",
            async () => {{
                for (
                    const participant
                    of PARTICIPANTS
                ) {{
                    await firstVisible(
                        page
                            .getByText(
                                new RegExp(
                                    `Número USP:\\\\s*${{participant}}.*Perfil:\\\\s*Assinante`,
                                    "is"
                                )
                            )
                            .first(),
                        `assinante ${{participant}} antes do envio`
                    );
                }}

                await firstVisible(
                    page.getByText(
                        PDF_NAME,
                        {{
                            exact: false
                        }}
                    ),
                    `PDF ${{PDF_NAME}} antes do envio`
                );

                const sendButton = await firstVisible(
                    page.getByRole(
                        "button",
                        {{
                            name: /enviar documentos para assinatura/i
                        }}
                    ),
                    "botão Enviar documentos para assinatura"
                );

                const disabled = await sendButton.isDisabled();

                log(
                    disabled ? "ERROR" : "OK",
                    "SEND_BUTTON_STATE",
                    disabled
                        ? "Botão de envio está desabilitado."
                        : "Botão de envio está habilitado."
                );

                if (disabled) {{
                    throw new Error(
                        "O lote não está pronto para envio: o botão de envio permanece desabilitado."
                    );
                }}
            }}
        );

        await runStep(
            "10_send",
            async () => {{
                const sendButton = await firstVisible(
                    page.getByRole(
                        "button",
                        {{
                            name: /enviar documentos para assinatura/i
                        }}
                    ),
                    "botão Enviar documentos para assinatura"
                );

                sendClicked = true;

                log(
                    "WARN",
                    "SEND_CLICK",
                    "O botão de envio será acionado a partir deste ponto."
                );

                await sendButton.click();

                await page.waitForTimeout(5000);

                await page.waitForURL(
                    /\\/assina\\/?(?:\\?.*)?$/i,
                    {{
                        timeout: Math.max(
                            TIMEOUT,
                            30000
                        )
                    }}
                );

                await firstVisible(
                    page.getByRole(
                        "heading",
                        {{
                            name: /assinaturas\\s*-\\s*listar/i
                        }}
                    ),
                    "listagem Assinaturas - Listar após envio"
                );
            }}
        );

        let finalSituation = (
            "Documento localizado na listagem"
        );

        await runStep(
            "11_validate_final",
            async () => {{
                const pendingForMe = page.locator(
                    "#checkminhapendencia"
                );

                if (
                    await pendingForMe.count()
                    && await pendingForMe
                        .isChecked()
                        .catch(() => false)
                ) {{
                    await pendingForMe.click();
                    await page.waitForTimeout(1000);

                    log(
                        "INFO",
                        "LIST_FILTER",
                        "Filtro Pendentes para eu assinar desmarcado."
                    );
                }}

                let titleLocator = page
                    .getByText(
                        TITLE,
                        {{
                            exact: false
                        }}
                    )
                    .first();

                if (
                    !(await titleLocator.count())
                    || !(await titleLocator
                        .isVisible()
                        .catch(() => false))
                ) {{
                    const titleFilter = page.getByRole(
                        "textbox",
                        {{
                            name: /filtrar por título/i
                        }}
                    );

                    if (
                        await titleFilter.count()
                        && await titleFilter
                            .first()
                            .isVisible()
                            .catch(() => false)
                    ) {{
                        await titleFilter
                            .first()
                            .fill(TITLE);

                        await page.waitForTimeout(
                            1500
                        );
                    }}
                }}

                titleLocator = await firstVisible(
                    page.getByText(
                        TITLE,
                        {{
                            exact: false
                        }}
                    ),
                    `documento enviado com título ${{TITLE}}`,
                    Math.max(
                        TIMEOUT,
                        30000
                    )
                );

                const row = titleLocator.locator(
                    "xpath=ancestor::tr[1]"
                );

                if (await row.count()) {{
                    const rowText = await row.innerText();

                    if (/para assinar/i.test(rowText)) {{
                        finalSituation = "Para assinar";
                    }}
                }}

                log(
                    "OK",
                    "FINAL_DOCUMENT",
                    "Documento localizado na listagem final.",
                    {{
                        situation: finalSituation,
                        url: page.url()
                    }}
                );
            }}
        );

        return {{
            status: "sent",
            safe_to_retry: false,
            title: TITLE,
            participants_count: PARTICIPANTS.length,
            final_url: page.url(),
            final_situation: finalSituation,
            pdf: PDF_NAME,
            send_clicked: sendClicked,
            events
        }};
    }} catch (error) {{
        let pageTitle = "";
        let bodyExcerpt = "";

        try {{
            pageTitle = await page.title();
        }} catch (_) {{}}

        try {{
            bodyExcerpt = (
                await page
                    .locator("body")
                    .innerText()
            ).slice(0, 4000);
        }} catch (_) {{}}

        log(
            "ERROR",
            "AUTOMATION_FAILED",
            String(
                error && error.message
                    ? error.message
                    : error
            ),
            {{
                url: page.url(),
                page_title: pageTitle
            }}
        );

        return {{
            status: sendClicked
                ? "uncertain_after_send"
                : "failed",
            safe_to_retry: !sendClicked,
            step,
            message: String(
                error && error.message
                    ? error.message
                    : error
            ),
            url: page.url(),
            page_title: pageTitle,
            body_excerpt: bodyExcerpt,
            send_clicked: sendClicked,
            events
        }};
    }} finally {{
        try {{
            page.off(
                "response",
                onResponse
            );
        }} catch (_) {{}}

    }}
}}
"""

        return code, pdf_b64

    @staticmethod
    def _log_browser_events(
        logger: RunLogger,
        automation: dict[str, Any],
    ) -> None:
        events = automation.get("events")

        if not isinstance(events, list):
            logger.warning(
                "BROWSER_EVENTS",
                "Resultado não contém lista de eventos do browser.",
            )
            return

        logger.info(
            "BROWSER_EVENTS",
            f"count={len(events)}",
        )

        for item in events:
            if not isinstance(item, dict):
                continue

            level = str(
                item.get("level")
                or "INFO"
            ).upper()

            event = str(
                item.get("event")
                or "BROWSER"
            )

            step = str(
                item.get("step")
                or ""
            )

            t_ms = item.get("t_ms")
            message = str(
                item.get("message")
                or ""
            )

            extra = item.get("extra")

            suffix_parts = []

            if step:
                suffix_parts.append(
                    f"step={step}"
                )

            if t_ms is not None:
                suffix_parts.append(
                    f"t_ms={t_ms}"
                )

            if extra is not None:
                try:
                    suffix_parts.append(
                        "extra="
                        + json.dumps(
                            extra,
                            ensure_ascii=False,
                        )
                    )
                except Exception:
                    suffix_parts.append(
                        f"extra={extra}"
                    )

            full_message = message

            if suffix_parts:
                full_message += (
                    " | "
                    + " ".join(suffix_parts)
                )

            if level == "ERROR":
                logger.error(
                    f"BROWSER_{event}",
                    full_message,
                )
            elif level == "WARN":
                logger.warning(
                    f"BROWSER_{event}",
                    full_message,
                )
            elif level == "OK":
                logger.ok(
                    f"BROWSER_{event}",
                    full_message,
                )
            else:
                logger.info(
                    f"BROWSER_{event}",
                    full_message,
                )

    def _error_response(
        self,
        *,
        logger: RunLogger,
        error: str,
        safe_to_retry: bool,
        stage: Optional[str] = None,
        status: str = "failed",
        extra: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        response: dict[str, Any] = {
            "ok": False,
            "status": status,
            "safe_to_retry": bool(
                safe_to_retry
            ),
            "run_id": logger.run_id,
            "stage": (
                stage
                or logger.current_stage
            ),
            "error": error,
        }

        if extra:
            response.update(extra)

        if self.valves.debug:
            response["debug"] = True
            response["debug_log"] = (
                logger.debug_text()
            )
            response["debug_log_path"] = str(
                logger.log_path
            )

        logger.save_result(response)
        logger.save_state(
            status="failed",
            extra={
                "error": error,
                "safe_to_retry": safe_to_retry,
            },
        )

        return response

    async def enviar_documento_usp_assina(
        self,
        titulo: str,
        emails: str,
        participantes: str,
        __files__=None,
        __event_call__=None,
        __event_emitter__=None,
        __user__=None,
    ) -> dict:
        """
        Cria e envia exatamente um PDF no USP Assina.

        A autenticação USP é solicitada pela própria ferramenta por meio da
        interface segura do Open WebUI. Nunca solicite Número USP ou senha
        diretamente no chat.

        :param titulo: Título do documento/lote.
        :param emails: E-mails para cópia separados por ponto e vírgula.
        :param participantes: Números USP a adicionar como Assinante, separados por ';'.
        :return: Resultado estruturado do envio.
        """

        logger = RunLogger(
            requested_root=self.valves.log_dir,
            debug=self.valves.debug,
            max_debug_chars=self.valves.max_debug_chars,
        )

        nusp = ""
        password = ""
        pdf_b64 = ""

        browser_call_started = False
        browser_closed = False

        try:
            logger.stage("INPUT_VALIDATION")

            title = str(
                titulo
                or ""
            ).strip()

            if not title:
                raise ToolError(
                    "O título do documento é obrigatório."
                )

            normalized_emails = (
                self._normalize_emails(
                    emails
                )
            )

            normalized_participants = (
                self._normalize_participants(
                    participantes
                )
            )

            logger.info(
                "INPUTS",
                (
                    f"title={title!r} "
                    f"emails_count={len(normalized_emails.split(';'))} "
                    f"participants_count={len(normalized_participants)} "
                    f"debug={self.valves.debug}"
                ),
            )

            logger.info(
                "MCP_CONFIG",
                (
                    f"mcp_url={self.valves.mcp_url!r} "
                    f"timeout_ms={self.valves.timeout_ms} "
                    f"upload_settle_ms={self.valves.upload_settle_ms}"
                ),
            )

            await self._status(
                __event_emitter__,
                "Validando o PDF anexado...",
            )

            pdf_name, pdf_bytes = (
                self._load_single_pdf(
                    __files__,
                    logger,
                )
            )

            logger.stage("AUTH")

            await self._status(
                __event_emitter__,
                "Aguardando autenticação USP...",
            )

            nusp, password = (
                await self._ask_credentials(
                    __event_call__,
                    logger,
                )
            )

            logger.stage("MCP_CONNECT")

            await self._status(
                __event_emitter__,
                "Conectando ao Playwright MCP...",
            )

            logger.info(
                "MCP_CONNECT_BEGIN",
                self.valves.mcp_url,
            )

            async with streamable_http_client(
                self.valves.mcp_url
            ) as transport:
                logger.ok(
                    "MCP_TRANSPORT",
                    "Transporte Streamable HTTP aberto.",
                )

                read_stream, write_stream, *_ = (
                    transport
                )

                logger.stage("MCP_SESSION")

                async with ClientSession(
                    read_stream,
                    write_stream,
                ) as session:
                    logger.stage(
                        "MCP_INITIALIZE"
                    )

                    initialize_result = (
                        await session.initialize()
                    )

                    logger.ok(
                        "MCP_INITIALIZED",
                        (
                            "protocol="
                            f"{getattr(initialize_result, 'protocolVersion', 'unknown')}"
                        ),
                    )

                    logger.stage(
                        "MCP_LIST_TOOLS"
                    )

                    tools_result = (
                        await session.list_tools()
                    )

                    tool_names = sorted(
                        tool.name
                        for tool
                        in tools_result.tools
                    )

                    logger.info(
                        "MCP_TOOLS",
                        (
                            f"count={len(tool_names)} "
                            f"names={','.join(tool_names)}"
                        ),
                    )

                    run_tool_name = None

                    if (
                        "browser_run_code_unsafe"
                        in tool_names
                    ):
                        run_tool_name = (
                            "browser_run_code_unsafe"
                        )
                    elif (
                        "browser_run_code"
                        in tool_names
                    ):
                        run_tool_name = (
                            "browser_run_code"
                        )

                    if not run_tool_name:
                        raise ToolError(
                            "O Playwright MCP não oferece "
                            "browser_run_code_unsafe nem browser_run_code."
                        )

                    if (
                        "browser_close"
                        not in tool_names
                    ):
                        raise ToolError(
                            "O Playwright MCP não oferece browser_close."
                        )

                    logger.ok(
                        "MCP_REQUIRED_TOOLS",
                        (
                            f"run_tool={run_tool_name} "
                            "browser_close=available"
                        ),
                    )

                    logger.stage(
                        "BUILD_AUTOMATION"
                    )

                    code, pdf_b64 = (
                        self._build_playwright_code(
                            nusp=nusp,
                            password=password,
                            title=title,
                            emails=normalized_emails,
                            participants=normalized_participants,
                            pdf_name=pdf_name,
                            pdf_bytes=pdf_bytes,
                        )
                    )

                    logger.info(
                        "AUTOMATION_CODE",
                        (
                            f"code_chars={len(code)} "
                            f"pdf_bytes={len(pdf_bytes)} "
                            f"pdf_base64_chars={len(pdf_b64)} "
                            "code_content=NOT_LOGGED"
                        ),
                    )

                    logger.stage(
                        "MCP_RUN_CODE"
                    )

                    await self._status(
                        __event_emitter__,
                        "Executando o fluxo no USP Assina...",
                    )

                    browser_call_started = True

                    run_started = time.monotonic()

                    result = await session.call_tool(
                        run_tool_name,
                        arguments={
                            "code": code,
                        },
                    )

                    logger.info(
                        "MCP_RUN_CODE_RETURN",
                        (
                            f"elapsed={time.monotonic() - run_started:.2f}s "
                            f"isError={bool(getattr(result, 'isError', False))}"
                        ),
                    )

                    result_text = (
                        self._result_text(
                            result
                        )
                    )

                    safe_text = (
                        self._safe_mcp_error_text(
                            result_text,
                            password=password,
                            pdf_b64=pdf_b64,
                        )
                    )

                    logger.info(
                        "MCP_RESPONSE",
                        (
                            f"text_chars={len(result_text)} "
                            f"safe_excerpt={safe_text[:1200]!r}"
                        ),
                    )

                    if bool(
                        getattr(
                            result,
                            "isError",
                            False,
                        )
                    ):
                        raise ToolError(
                            "O Playwright MCP retornou erro ao executar "
                            f"a automação: {safe_text[:3000]}"
                        )

                    logger.stage(
                        "MCP_PARSE_RESULT"
                    )

                    automation = (
                        self._parse_automation_result(
                            result_text,
                            logger,
                        )
                    )

                    self._log_browser_events(
                        logger,
                        automation,
                    )

                    logger.info(
                        "AUTOMATION_RESULT",
                        json.dumps(
                            {
                                key: value
                                for key, value
                                in automation.items()
                                if key
                                not in {
                                    "events",
                                    "body_excerpt",
                                }
                            },
                            ensure_ascii=False,
                        ),
                    )

                    if (
                        automation.get("status")
                        == "sent"
                    ):
                        logger.stage(
                            "MCP_CLOSE_BROWSER"
                        )

                        try:
                            await session.call_tool(
                                "browser_close",
                                arguments={},
                            )

                            browser_closed = True

                            logger.ok(
                                "MCP_BROWSER_CLOSE",
                                "Browser fechado.",
                            )
                        except BaseException as exc:
                            logger.warning(
                                "MCP_BROWSER_CLOSE",
                                "Falha best-effort ao fechar browser: "
                                + " | ".join(
                                    self._unwrap_exception(
                                        exc
                                    )
                                ),
                            )

                        logger.stage("SUCCESS")

                        response = {
                            "ok": True,
                            "status": "sent",
                            "run_id": logger.run_id,
                            "title": title,
                            "participants_count": len(
                                normalized_participants
                            ),
                            "final_situation": automation.get(
                                "final_situation",
                                "Documento localizado na listagem",
                            ),
                        }

                        logger.ok(
                            "RESULT",
                            (
                                f"Documento enviado. "
                                f"title={title!r} "
                                f"participants={len(normalized_participants)} "
                                f"situation={response['final_situation']!r}"
                            ),
                        )

                        logger.save_result(
                            response
                        )

                        logger.save_state(
                            status="success",
                            extra={
                                "title": title,
                                "participants_count": len(
                                    normalized_participants
                                ),
                            },
                        )

                        await self._status(
                            __event_emitter__,
                            "Documento enviado com sucesso.",
                            done=True,
                        )

                        return response

                    if (
                        automation.get("status")
                        == "uncertain_after_send"
                    ):
                        logger.error(
                            "RESULT_UNCERTAIN",
                            (
                                f"step={automation.get('step')} "
                                f"error={automation.get('message')}"
                            ),
                        )

                        await self._status(
                            __event_emitter__,
                            (
                                "O envio foi acionado, mas a validação "
                                "final não pôde ser confirmada."
                            ),
                            done=True,
                        )

                        return self._error_response(
                            logger=logger,
                            error=str(
                                automation.get(
                                    "message"
                                )
                                or "Falha após acionar o envio."
                            ),
                            safe_to_retry=False,
                            stage=str(
                                automation.get(
                                    "step"
                                )
                                or logger.current_stage
                            ),
                            status="uncertain_after_send",
                            extra={
                                "browser_url": automation.get(
                                    "url",
                                    "",
                                ),
                                "browser_page_title": automation.get(
                                    "page_title",
                                    "",
                                ),
                                "browser_body_excerpt": (
                                    automation.get(
                                        "body_excerpt",
                                        "",
                                    )
                                    if self.valves.debug
                                    else ""
                                ),
                            },
                        )

                    logger.error(
                        "RESULT_FAILED",
                        (
                            f"step={automation.get('step')} "
                            f"error={automation.get('message')}"
                        ),
                    )

                    await self._status(
                        __event_emitter__,
                        "A automação falhou antes do envio.",
                        done=True,
                    )

                    return self._error_response(
                        logger=logger,
                        error=str(
                            automation.get(
                                "message"
                            )
                            or "Falha durante a automação."
                        ),
                        safe_to_retry=bool(
                            automation.get(
                                "safe_to_retry",
                                True,
                            )
                        ),
                        stage=str(
                            automation.get(
                                "step"
                            )
                            or logger.current_stage
                        ),
                        extra={
                            "browser_url": automation.get(
                                "url",
                                "",
                            ),
                            "browser_page_title": automation.get(
                                "page_title",
                                "",
                            ),
                            "browser_body_excerpt": (
                                automation.get(
                                    "body_excerpt",
                                    "",
                                )
                                if self.valves.debug
                                else ""
                            ),
                        },
                    )

        except ToolError as exc:
            logger.error(
                "TOOL_ERROR",
                f"{type(exc).__name__}: {exc}",
            )

            await self._status(
                __event_emitter__,
                "Não foi possível concluir a operação.",
                done=True,
            )

            # Se browser_run_code já foi chamado e o erro ocorreu fora do
            # resultado estruturado do browser, não podemos garantir que o
            # clique de envio não ocorreu.
            safe_to_retry = (
                not browser_call_started
            )

            return self._error_response(
                logger=logger,
                error=str(exc),
                safe_to_retry=safe_to_retry,
            )

        except BaseException as exc:
            details = self._unwrap_exception(
                exc
            )

            logger.error(
                "UNEXPECTED_EXCEPTION",
                " | ".join(details),
            )

            try:
                logger.error(
                    "TRACEBACK",
                    traceback.format_exc(),
                )
            except Exception:
                pass

            await self._status(
                __event_emitter__,
                "Erro inesperado durante a operação.",
                done=True,
            )

            return self._error_response(
                logger=logger,
                error=details[-1] if details else (
                    f"{type(exc).__name__}: {exc}"
                ),
                safe_to_retry=(
                    not browser_call_started
                ),
            )

        finally:
            password = ""
            nusp = ""
            pdf_b64 = ""

            if (
                browser_call_started
                and not browser_closed
            ):
                logger.warning(
                    "BROWSER_CLOSE_STATE",
                    (
                        "A execução iniciou uma chamada de browser, "
                        "mas não foi possível confirmar browser_close "
                        "pelo caminho principal."
                    ),
                )
