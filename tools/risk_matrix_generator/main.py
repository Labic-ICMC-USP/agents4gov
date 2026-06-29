import io
import requests
import pandas as pd
import base64
import unicodedata
import numpy as np
from typing import Dict, Any, List, Tuple
from urllib.parse import urlparse, parse_qs
from pydantic import Field

LIKELIHOOD_TERMS = {
    # PT
    "inexistente": 0,
    "impossivel": 1,
    "rarissimo": 2,
    "raro": 3,
    "muito baixo": 4,
    "baixo": 5,
    "pouco provavel": 6,
    "improvavel": 7,
    "ocasional": 8,
    "possivel": 9,
    "moderado": 10,
    "media": 11,
    "provavel": 12,
    "frequente": 13,
    "muito provavel": 14,
    "alto": 15,
    "muito alto": 16,
    "quase certo": 17,
    "certo": 18,
    # EN
    "nonexistent": 0,
    "impossible": 1,
    "extremely rare": 2,
    "very rare": 3,
    "rare": 4,
    "very low": 5,
    "low": 6,
    "unlikely": 7,
    "occasional": 8,
    "possible": 9,
    "moderate": 10,
    "medium": 11,
    "likely": 12,
    "frequent": 13,
    "very likely": 14,
    "high": 15,
    "very high": 16,
    "almost certain": 17,
    "certain": 18,
}

SEVERITY_TERMS = {
    # PT
    "insignificante": 0,
    "desprezivel": 1,
    "muito baixo": 2,
    "baixo": 3,
    "leve": 4,
    "menor": 5,
    "moderado": 6,
    "medio": 7,
    "relevante": 8,
    "significativo": 9,
    "alto": 10,
    "grave": 11,
    "severo": 12,
    "muito grave": 13,
    "critico": 14,
    "catastrofico": 15,
    # EN
    "insignificant": 0,
    "negligible": 1,
    "very low": 2,
    "low": 3,
    "minor": 4,
    "slight": 5,
    "moderate": 6,
    "medium": 7,
    "relevant": 8,
    "significant": 9,
    "high": 10,
    "serious": 11,
    "severe": 12,
    "very severe": 13,
    "critical": 14,
    "catastrophic": 15,
}


class Tools:

    # ---------- MAIN ----------
    def risk_matrix_generator(
        self,
        dataset_link: str = Field(
            ..., description="Public URL or Google Drive link to the dataset."
        ),
        max_rows: int = 100,
        sample_size: int = 10,
        probability_column: str | None = None,
        severity_column: str | None = None,
        prob_min: float | None = None,
        prob_max: float | None = None,
        sev_min: float | None = None,
        sev_max: float | None = None,
        custom_prob: List[str] | None = None,
        custom_sev: List[str] | None = None,
        generate_image: bool = False,
        likelihood_label: str = "Likelihood",
        severity_label: str = "Severity",
        title: str = "Risk Matrix",
        __event_emitter__=None,
    ) -> Dict[str, Any]:

        warning = []  # Store warnings for user

        # Try to download and load dataset
        try:
            file_bytes = self._download_file(dataset_link)
            if not isinstance(file_bytes, bytes):
                raise ValueError("Downloaded content is not bytes. Try again")
            df = self._load_dataframe(file_bytes)
        except Exception as e:
            return {
                "error": {
                    "type": "DATASET_DOWNLOAD_FAILED",
                    "message": str(e),
                }
            }

        # Ensure all labels are strings
        probability_column = str(probability_column)
        severity_column = str(severity_column)
        likelihood_label = str(likelihood_label)
        severity_label = str(severity_label)
        title = str(title)

        # Validate input values
        error = self._values_checking(
            df,
            sev_max,
            sev_min,
            prob_max,
            prob_min,
            max_rows,
            sample_size,
            custom_prob,
            custom_sev,
        )
        if error:
            return error

        # Limit dataset rows if exceeds max_rows
        original_rows = len(df)
        if original_rows > max_rows:
            df = df.iloc[:max_rows]
            warning.append(f"Dataset truncated to first {max_rows} rows.")

        # Automatically infer probability and severity columns
        auto_prob, prob_method = self._infer_risk_columns(df, "prob")
        auto_sev, sev_method = self._infer_risk_columns(df, "sev")

        # Resolve which column to use (user or inferred)
        prob_col, prob_method, prob_warn = self._resolve_column(
            df, probability_column, auto_prob, prob_method
        )
        sev_col, sev_method, sev_warn = self._resolve_column(
            df, severity_column, auto_sev, sev_method
        )

        # Check if any required column is missing
        if prob_col is None or sev_col is None:
            missing = []
            if not prob_col:
                missing.append("probability")
            if not sev_col:
                missing.append("severity")
            return {
                "error": {
                    "type": "COLUMN_INFERENCE_FAILED",
                    "message": f"Could not infer the following columns, or user-provided columns were invalid: {', '.join(missing)}.",
                }
            }

        # Merge warnings
        warning.extend(prob_warn)
        warning.extend(sev_warn)

        # Set final inference method
        inference_method = f"{prob_method}/{sev_method}"

        # Prepare axis values and labels
        prob_values, prob_labels, prob_warn = self._prepare_axis(
            df[prob_col], prob_min, prob_max, "prob", custom_prob
        )
        sev_values, sev_labels, sev_warn = self._prepare_axis(
            df[sev_col], sev_min, sev_max, "sev", custom_sev
        )
        warning.extend(prob_warn)
        warning.extend(sev_warn)

        # Build risk matrix counts
        matrix = {p: {s: 0 for s in sev_labels} for p in prob_labels}
        for p, s in zip(prob_values, sev_values):
            if p in matrix and s in matrix[p]:
                matrix[p][s] += 1

        # Generate optional base64 image
        image_base64 = None
        if generate_image:
            image_base64 = self._generate_matrix_image(
                matrix, prob_labels, sev_labels, title, likelihood_label, severity_label
            )

        # ---------- CRITICAL EVENTS SAMPLE ----------
        # Map labels to numeric indexes
        prob_map = {v: i + 1 for i, v in enumerate(prob_labels)}
        sev_map = {v: i + 1 for i, v in enumerate(sev_labels)}

        tmp = df.copy()
        # Map original columns to normalized scores
        tmp["_prob"] = tmp[prob_col].map(prob_map).fillna(0).astype(int)
        tmp["_sev"] = tmp[sev_col].map(sev_map).fillna(0).astype(int)

        # Compute combined score and sort
        tmp["_score"] = tmp["_prob"] + tmp["_sev"]
        tmp = tmp.sort_values(["_score", "_prob", "_sev"], ascending=False)

        top = tmp.head(sample_size)
        # Extract top critical events
        critical_events = top.drop(columns=["_prob", "_sev", "_score"]).to_dict(
            orient="records"
        )

        # Return structured result
        return {
            "image_base64": image_base64,
            "image_info": "The image is Base64 encoded. Do NOT render it directly. Do NOT show the json base64 code. Inform the user that the image is included in the JSON under 'image_base64' and that they can view it using this site: https://base64-to-image.com/.",
            "metadata": {
                "rows_original": original_rows,
                "rows_processed": len(df),
                "probability_column": prob_col,
                "severity_column": sev_col,
                "inference_method": inference_method,
                "warnings": warning or None,
            },
            "axes": {
                "probability_labels": prob_labels,
                "severity_labels": sev_labels,
            },
            "risk_matrix": matrix,
            "critical_events_sample": critical_events or None,
        }

    # ---------- HELPERS ----------

    # Download file from URL or Google Drive
    def _download_file(self, link: str) -> bytes:
        if "drive.google.com" in link:
            file_id = self._extract_drive_id(link)
            link = f"https://drive.google.com/uc?id={file_id}&export=download"
        r = requests.get(link, timeout=15)
        r.raise_for_status()
        return r.content

    # Extract file ID from Google Drive link
    def _extract_drive_id(self, link: str) -> str:
        parsed = urlparse(link)
        if "id=" in link:
            return parse_qs(parsed.query)["id"][0]
        return parsed.path.split("/")[-2]

    # Try multiple loaders to read dataset
    def _load_dataframe(self, file_bytes: bytes) -> Any:
        loaders = [
            lambda: pd.read_excel(io.BytesIO(file_bytes)),
            lambda: pd.read_csv(io.BytesIO(file_bytes)),
            lambda: pd.read_json(io.BytesIO(file_bytes)),
            lambda: pd.read_parquet(io.BytesIO(file_bytes)),
            lambda: pd.read_feather(io.BytesIO(file_bytes)),
        ]
        for loader in loaders:
            try:
                df = loader()
                if not df.empty:
                    return df
            except Exception:
                pass
        raise ValueError("Unsupported dataset format.")

    # Validate input parameters and dataset
    def _values_checking(
        self,
        df: Any,
        sev_max: int | None,
        sev_min: int | None,
        prob_max: int | None,
        prob_min: int | None,
        max_rows: int,
        sample_size: int,
        custom_sev: list[str] | None,
        custom_prob: list[str] | None,
    ) -> dict | None:
        if df.shape[1] < 2:
            return {
                "error": {
                    "type": "INVALID_DATASET",
                    "message": "Dataset must contain at least two columns.",
                }
            }

        if not isinstance(max_rows, int) or not isinstance(sample_size, int):
            return {
                "error": {
                    "type": "INVALID_TYPE",
                    "message": "max rows and sample size must be integers.",
                }
            }

        if (custom_prob is not None and not isinstance(custom_prob, list)) or (
            custom_sev is not None and not isinstance(custom_sev, list)
        ):
            return {
                "error": {
                    "type": "INVALID_TYPE",
                    "message": "Custom severity and custom likelihood terms must be a list or None.",
                }
            }

        for v in (sev_min, sev_max, prob_min, prob_max):
            if v is not None and not isinstance(v, int):
                return {
                    "error": {
                        "type": "INVALID_TYPE",
                        "message": "Min/Max values must be integers or None.",
                    }
                }

        if sample_size < 0:
            return {
                "error": {
                    "type": "INVALID_VALUE",
                    "message": "Sample size must be greater than or equal to zero.",
                }
            }

        if max_rows <= 0:
            return {
                "error": {
                    "type": "INVALID_VALUE",
                    "message": "Max rows must be greater than zero.",
                }
            }

        if (sev_min is not None and sev_max is not None and sev_min >= sev_max) or (
            prob_min is not None and prob_max is not None and prob_min >= prob_max
        ):
            return {
                "error": {
                    "type": "INVALID_INTERVAL",
                    "message": "Min value must be lower than max value.",
                }
            }

        if (
            (sev_min is not None and sev_min <= 0)
            or (sev_max is not None and sev_max <= 0)
            or (prob_min is not None and prob_min <= 0)
            or (prob_max is not None and prob_max <= 0)
        ):
            return {
                "error": {
                    "type": "INVALID_VALUE",
                    "message": "Value must be greater than zero",
                }
            }

        return None

    # Determine final column to use, user-provided or inferred
    def _resolve_column(
        self,
        df: Any,
        user_col: str | None,
        auto_col: str | None,
        auto_method: str,
    ) -> Tuple[str | None, str, List[str]]:
        warning = []
        if user_col is not None and user_col in df.columns:
            return user_col, "user_provided", warning
        elif user_col is not None:
            warning = [f"Column '{user_col}' not found. Using inferred '{auto_col}'."]
        return auto_col, auto_method, warning

    # Normalize text for matching
    def _normalize_text(self, value: str) -> str:
        value = value.strip().lower()
        value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
        return value

    # ---------- INFERENCE ----------
    # Attempt to infer probability or severity column automatically
    def _infer_risk_columns(self, df: Any, column_type: str) -> Tuple[str | None, str]:
        cols = [c.lower() for c in df.columns]

        prob_keys = [
            # EN
            "probability",
            "likelihood",
            "chance",
            "prob",
            "occurrence",
            "frequency",
            "odds",
            # PT
            "probabilidade",
            "ocorrencia",
            "ocorrência",
            "frequencia",
            "frequência",
            "possibilidade",
            "incidencia",
            "incidência",
        ]

        sev_keys = [
            # EN
            "severity",
            "impact",
            "damage",
            "consequence",
            "criticality",
            "magnitude",
            "loss",
            "harm",
            # PT
            "severidade",
            "impacto",
            "dano",
            "consequencia",
            "consequência",
            "criticidade",
            "gravidade",
            "prejuizo",
            "prejuízo",
            "perda",
        ]

        ref_keys = prob_keys if column_type == "prob" else sev_keys

        column = next(
            (
                df.columns[i]
                for i, c in enumerate(cols)
                if any(k in c for k in ref_keys)
            ),
            None,
        )

        if column:
            return column, "keyword"

        return None, "failed"

    # ---------- AXIS ----------
    # Prepare axis values and labels
    def _prepare_axis(
        self,
        series: Any,
        min_val: float | None,
        max_val: float | None,
        axis_type: str,
        custom_terms: List[str] | None = None,
    ) -> Tuple[Any, list[str | int], List[str]]:

        warnings = []

        try:
            series = series.astype(int)
            is_int = True
        except (ValueError, TypeError):
            is_int = False

        keyword_terms = LIKELIHOOD_TERMS if axis_type == "prob" else SEVERITY_TERMS

        if is_int and custom_terms is None:
            values = series.dropna()
            df_max = int(values.max())
            df_min = int(values.min())

            min_val = min_val if min_val is not None and min_val < df_min else df_min
            max_val = max_val if max_val is not None and max_val > df_max else df_max

            ordered = list(range(min_val, max_val + 1))

            return values, ordered, warnings

        valid = []
        discarded = set()

        if custom_terms is None:
            custom_terms = []

        df_terms = (
            custom_terms
            if is_int
            else set(list(series.dropna().astype(str)) + custom_terms)
        )

        for raw in df_terms:
            raw = str(raw)
            norm = self._normalize_text(raw)
            if norm in keyword_terms:
                valid.append(norm)
            else:
                discarded.add(raw)

        if discarded:
            warnings.append(f"Discarded {axis_type} categories: {sorted(discarded)}")

        ordered = [t for t in keyword_terms if t in valid]

        if is_int:
            df_max = int(series.dropna().max())
            ordered.extend(range(len(ordered) + 1, df_max + 1))

        value_map = {i + 1: v for i, v in enumerate(ordered)}
        values = series.map(value_map).fillna(0) if is_int else series

        return values, ordered, warnings

    # ---------- IMAGE ----------
    # Generate base64 image of risk matrix
    def _generate_matrix_image(
        self,
        matrix: Dict[int, Dict[int, int]],
        prob_labels: List[str],
        sev_labels: List[str],
        title: str,
        likelihood_label: str,
        severity_label: str,
    ) -> str:

        prob_labels = prob_labels[::-1]

        data = np.array([[matrix[p][s] for s in sev_labels] for p in prob_labels])
        max_value = int(data.max()) if data.size else 0

        n_prob = len(prob_labels)
        n_sev = len(sev_labels)
        max_score = n_prob + n_sev

        # Thresholds
        red_score_min = int(np.ceil(0.8 * max_score))
        yellow_score_min = int(np.ceil(0.6 * max_score))
        red_sev_min = int(np.ceil(0.6 * n_sev))
        yellow_sev_min = int(np.ceil(0.4 * n_sev))

        # Layout
        cell = 60
        width = (n_sev + 1) * cell
        height = (n_prob + 2) * cell
        title_font = min(width / 20, 40)

        svg = [
            f'<div style="overflow:auto">',
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
            '<rect width="100%" height="100%" fill="#FFF7F7"/>',
            f'<text x="10" y="30" font-size="{title_font}" font-weight="bold">{title}</text>',
        ]

        header_gray = "#EEF1F6"
        edge = "#FFF7F7"

        # Likelihood header
        svg.append(
            f'<rect x="0" y="60" width="{cell}" height="{cell*2}" fill="{header_gray}" stroke="{edge}" stroke-width="4"/>'
        )
        svg.append(
            f'<text x="{cell/2}" y="{60+cell}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold">{likelihood_label}</text>'
        )

        # Severity header
        svg.append(
            f'<rect x="{cell}" y="60" width="{n_sev*cell}" height="{cell}" fill="{header_gray}" stroke="{edge}" stroke-width="4"/>'
        )
        svg.append(
            f'<text x="{cell + (n_sev*cell)/2}" y="{60+cell/2}" text-anchor="middle" dominant-baseline="middle" font-size="12" font-weight="bold">{severity_label}</text>'
        )

        # Severity labels
        for j, label in enumerate(sev_labels):
            x = cell * (j + 1)
            y = 60 + cell
            text = f"{severity_label[0]}{label}" if str(label).isdigit() else str(label)
            svg.append(
                f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="white" stroke="{edge}" stroke-width="4"/>'
            )
            svg.append(
                f'<text x="{x+cell/2}" y="{y+cell/2}" text-anchor="middle" dominant-baseline="middle" font-size="11">{text}</text>'
            )

        # Body
        for i, label in enumerate(prob_labels):
            y = 60 + cell * (i + 2)
            text = (
                f"{likelihood_label[0]}{label}" if str(label).isdigit() else str(label)
            )
            svg.append(
                f'<rect x="0" y="{y}" width="{cell}" height="{cell}" fill="white" stroke="{edge}" stroke-width="4"/>'
            )
            svg.append(
                f'<text x="{cell/2}" y="{y+cell/2}" text-anchor="middle" dominant-baseline="middle" font-size="11">{text}</text>'
            )

            for j, _ in enumerate(sev_labels):
                x = cell * (j + 1)
                score = n_prob - i + j + 1
                if j + 1 >= red_sev_min and score >= red_score_min:
                    color = "#E85A70"
                elif j + 1 >= yellow_sev_min and score >= yellow_score_min:
                    color = "#F4C542"
                else:
                    color = "#5CC38A"

                value = int(data[i][j]) if data.size else 0
                text_color = "black" if value == max_value else "white"
                svg.append(
                    f'<rect x="{x}" y="{y}" width="{cell}" height="{cell}" fill="{color}" stroke="{edge}" stroke-width="4"/>'
                )
                svg.append(
                    f'<text x="{x+cell/2}" y="{y+cell/2}" text-anchor="middle" dominant-baseline="middle" font-size="16" font-weight="bold" fill="{text_color}">{value}</text>'
                )

        svg.append("</svg></div>")

        svg_content = "\n".join(svg)

        # Save the SVG locally
        #file_path = "matrix.svg"
        #with open(file_path, "w", encoding="utf-8") as f:
        #    f.write(svg_content)

        # Base64 conversion
        svg_bytes = svg_content.encode("utf-8")
        svg_base64 = base64.b64encode(svg_bytes).decode("utf-8")

        return svg_base64