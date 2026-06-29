# Risk Matrix Generator

Generates a probability × severity risk matrix from a dataset. Optionally produces a base64-encoded SVG image of the matrix and extracts the top critical events.

---

## Function

### `risk_matrix_generator`

```python
risk_matrix_generator(
    dataset_link: str,
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
)
```
## Parameters

| Parameter             | Type               | Default       | Description |
|-----------------------|------------------|---------------|-------------|
| `dataset_link`        | `str`             | —             | Public URL or Google Drive link to the dataset. |
| `max_rows`            | `int`             | 100           | Maximum number of dataset rows to process. |
| `sample_size`         | `int`             | 10            | Number of top critical events to return. |
| `probability_column`  | `str` or `None`   | None          | Column name for probability. If not provided, inferred automatically. |
| `severity_column`     | `str` or `None`   | None          | Column name for severity. If not provided, inferred automatically. |
| `prob_min`            | `float` or `None` | None          | Minimum probability value for axis scaling. |
| `prob_max`            | `float` or `None` | None          | Maximum probability value for axis scaling. |
| `sev_min`             | `float` or `None` | None          | Minimum severity value for axis scaling. |
| `sev_max`             | `float` or `None` | None          | Maximum severity value for axis scaling. |
| `custom_prob`         | `List[str]` or None | None       | Custom probability labels (strings) to use instead of defaults. |
| `custom_sev`          | `List[str]` or None | None       | Custom severity labels (strings) to use instead of defaults. |
| `generate_image`      | `bool`            | False         | Whether to generate a base64-encoded SVG image of the matrix. |
| `likelihood_label`    | `str`             | "Likelihood"  | Label to display on the probability axis in the image. |
| `severity_label`      | `str`             | "Severity"    | Label to display on the severity axis in the image. |
| `title`               | `str`             | "Risk Matrix" | Title for the generated image. |

## Returns

A dictionary containing:

- `risk_matrix`: Nested dictionary `{prob_label: {sev_label: count}}` of event counts.  
- `axes`: Dictionary with `probability_labels` and `severity_labels` used.  
- `critical_events_sample`: List of top critical events (as dictionaries).  
- `image_base64`: Base64-encoded SVG image (only if `generate_image=True`).  
- `metadata`: Information about processed rows, columns, inference method, and warnings.

## Usage Example

```python
tools = Tools()
result = tools.risk_matrix_generator(
    dataset_link="https://example.com/data.xlsx",
    probability_column="Probability",
    severity_column="Severity",
    generate_image=True,
    sample_size=5,
)
```

- Access the risk matrix: result["risk_matrix"]

- Access top critical events: result["critical_events_sample"]

- View image: Decode result["image_base64"] using a Base64-to-Image tool, e.g., https://base64-to-image.com/

## Notes

- Probability and severity columns are automatically inferred if not provided.  
- Supported dataset formats: CSV, Excel, JSON, Parquet, Feather.  
- Warnings for discarded or truncated data are included in `metadata["warnings"]`.  
- Axis labels can be customized using `custom_prob` and `custom_sev`.
- To save the SVG image when running locally, uncomment the following lines in the code:

```python
# file_path = "matrix.svg"
# with open(file_path, "w", encoding="utf-8") as f:
# f.write(svg_content)