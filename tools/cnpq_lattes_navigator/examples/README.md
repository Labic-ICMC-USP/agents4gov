# CNPq/Lattes Navigator - Examples

This directory contains example input and output files for the CNPq/Lattes Navigator tool.

## Files

### input_example.json

Example input showing how to structure the researchers list and configuration parameters.

**Key fields:**
- `researchers`: Array of objects with `name` and `lattes_id`
- `time_window`: Number of years to analyze (default: 5)
- `coi_rules_config`: Configuration object to enable/disable specific COI rules

### output_example.json

Example output showing the complete structure of the tool's response when COI is detected.

**Key sections:**
- `execution_metadata`: Information about the analysis run
- `researchers`: Per-researcher profile data with production summaries
- `coi_matrix`: Pairwise conflict of interest detections with evidence
- `summary_text`: Human-readable summary

## Usage in Open WebUI

When using the tool in Open WebUI, you would provide the researchers data as a JSON string:

```
Can you analyze these researchers for conflicts of interest:
[
  {"name": "Ana Silva Santos", "lattes_id": "1234567890123456"},
  {"name": "Carlos Oliveira Lima", "lattes_id": "2345678901234567"}
]
```

The agent will automatically invoke the tool and return structured results.

## Important Notes

1. **Anonymized Data**: The examples use anonymized/fictional data to protect privacy.
2. **Mock Data Warning**: Without browser-use properly configured, the tool will return mock data with warnings.
3. **Evidence**: All COI detections include evidence URLs and specific details.
4. **Confidence Levels**: Each COI detection includes a confidence level (high/medium/low).

## COI Rules Summary

- **R1**: Co-authorship (≥1 shared publication)
- **R2**: Advisor-advisee relationship
- **R3**: Institutional overlap (same department/program)
- **R4**: Project team overlap
- **R5**: Committee/board/event overlap
- **R6**: Frequent co-authorship (≥3 publications)
- **R7**: Strong institutional proximity (same lab/group)

## Testing

To test the tool with the example input:

1. Import the tool into Open WebUI
2. Use the researchers from `input_example.json` in your query
3. Compare the output structure with `output_example.json`

