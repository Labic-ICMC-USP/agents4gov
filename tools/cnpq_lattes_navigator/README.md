# CNPq/Lattes Navigator - COI Detection and 5-Year Production Summary

**File:** `lattes_navigator.py`

**Description:** Automated tool for navigating public CNPq/Lattes profiles to detect Conflicts of Interest (COI) and summarize academic production over a configurable time window.

**Main Method:** `analyze_researchers_coi(researchers_json, time_window, coi_rules_config)`

---

## Overview

The CNPq/Lattes Navigator is a specialized tool designed for the **Agents4Gov (LABIC – ICMC/USP)** project. It analyzes public researcher profiles from the Brazilian National Council for Scientific and Technological Development (CNPq) Lattes platform to:

1. **Detect Conflicts of Interest (COI)** between researchers using 7 predefined rules
2. **Summarize academic production** over the last 5 years (configurable)
3. **Provide evidence-based analysis** with confidence levels and supporting URLs

---

## Features

- **Automated Profile Navigation**: Uses browser-use library to access public Lattes profiles
- **7 COI Detection Rules**: Comprehensive conflict detection covering co-authorship, advising, institutional overlap, and more
- **Configurable Time Window**: Analyze production from the last N years (default: 5)
- **Evidence-Based Results**: All COI detections include supporting evidence and URLs
- **Confidence Scoring**: High/medium/low confidence levels for each detection
- **Rate Limiting**: Built-in delays to respect server resources
- **Graceful Degradation**: Falls back to mock data when browser automation is unavailable

---

## Parameters

### `researchers_json` (required)
JSON string containing list of researchers to analyze.

**Format:**
```json
[
  {"name": "Researcher Name", "lattes_id": "1234567890123456"},
  {"name": "Another Researcher", "lattes_id": "2345678901234567"}
]
```

**Notes:**
- `lattes_id`: 16-digit identifier from the Lattes profile URL
- `name`: Full researcher name for disambiguation

### `time_window` (optional, default: 5)
Number of years to look back for production analysis and COI detection.

**Example:** `time_window=3` analyzes the last 3 years.

### `coi_rules_config` (optional)
JSON string to enable/disable specific COI rules.

**Default (all enabled):**
```json
{
  "R1": true,
  "R2": true,
  "R3": true,
  "R4": true,
  "R5": true,
  "R6": true,
  "R7": true
}
```

**Example (disable some rules):**
```json
{
  "R1": true,
  "R2": true,
  "R3": false,
  "R4": false,
  "R5": false,
  "R6": true,
  "R7": false
}
```

---

## Conflict of Interest (COI) Rules

The tool evaluates **pairwise COI** across all input researchers using publicly available information. A COI flag is raised when **any** activated rule is satisfied.

### R1: Co-authorship
**Condition:** At least 1 co-authored publication within the time window

**Evidence:** Publication entries (title, year, venue) appearing on both profiles

**Confidence:** High (exact title match), Medium (fuzzy match)

### R2: Advisor-Advisee Relationship
**Condition:** One researcher listed as advisor/supervisor of the other's Master/PhD/Postdoc (concluded or ongoing)

**Evidence:** Advising/supervision sections with names, titles, and years

**Confidence:** High (exact name match), Medium (partial match)

### R3: Institutional Overlap
**Condition:** Same department or graduate program affiliation concurrently within the time window

**Evidence:** Affiliation fields (institution, unit/program, time markers)

**Confidence:** High (same department), Medium (same institution)

### R4: Project Team Overlap
**Condition:** Participation in the same funded project within the time window

**Evidence:** Project title, sponsor, role, and years as listed publicly

**Confidence:** High (exact project match)

### R5: Committee/Board/Event Overlap
**Condition:** Publicly listed service on the same committee/board/event organization

**Evidence:** Activities/Services section with event/committee name and year

**Confidence:** Medium (exact match), Low (similar names)

### R6: Frequent Co-Authorship (stronger signal)
**Condition:** 3 or more co-authored publications within the time window

**Evidence:** Multiple publication entries corroborating repeated collaboration

**Confidence:** High (≥3 exact matches)

### R7: Strong Institutional Proximity
**Condition:** Same lab/research group explicitly named in both profiles

**Evidence:** Group/lab names in affiliations or projects

**Confidence:** High (exact lab name match)

---

## Output Structure

The tool returns a JSON string with the following structure:

```json
{
  "status": "success",
  "execution_metadata": {
    "execution_date": "ISO 8601 timestamp",
    "time_window_years": 5,
    "cutoff_date": "ISO 8601 date",
    "num_researchers": 3,
    "coi_rules_active": {...}
  },
  "researchers": [
    {
      "person": {...},
      "production_5y": {
        "publications": {...},
        "projects": {...},
        "advising": {...},
        "activities": [...]
      },
      "affiliations_5y": [...],
      "coauthors_5y": [...],
      "warnings": [...],
      "evidence_urls": [...]
    }
  ],
  "coi_matrix": {
    "pairs": [
      {
        "a_lattes_id": "...",
        "b_lattes_id": "...",
        "a_name": "...",
        "b_name": "...",
        "rules_triggered": ["R1", "R3"],
        "confidence": "high",
        "evidence": [...]
      }
    ]
  },
  "summary_text": "Analysis of 3 researchers..."
}
```

See `schema.json` for complete schema definition.

---

## Usage

### In Open WebUI

After importing the tool into Open WebUI:

```
Can you analyze these researchers for conflicts of interest over the last 5 years:
[
  {"name": "Ana Silva Santos", "lattes_id": "1234567890123456"},
  {"name": "Carlos Oliveira Lima", "lattes_id": "2345678901234567"}
]
```

### With Custom Time Window

```
Analyze these researchers for COI using a 3-year window:
[
  {"name": "Ana Silva Santos", "lattes_id": "1234567890123456"},
  {"name": "Carlos Oliveira Lima", "lattes_id": "2345678901234567"}
]
Time window: 3 years
```

### With Custom Rule Configuration

```
Analyze for COI using only co-authorship and advisor-advisee rules:
[
  {"name": "Ana Silva Santos", "lattes_id": "1234567890123456"},
  {"name": "Carlos Oliveira Lima", "lattes_id": "2345678901234567"}
]
Rules: {"R1": true, "R2": true, "R3": false, "R4": false, "R5": false, "R6": true, "R7": false}
```

---

## Dependencies

```bash
pip install browser-use playwright requests pydantic python-dateutil
```

**Required:**
- `pydantic`: Parameter validation
- `python-dateutil`: Date parsing
- `requests`: HTTP requests (if needed)

**For Full Functionality:**
- `browser-use`: LLM-powered browser automation
- `playwright`: Browser automation backend

**Note:** The tool gracefully degrades when browser-use is not available, returning mock data with warnings.

---

## Limitations

### Technical Limitations

1. **Browser Automation Complexity**: Browser-use requires async execution and LLM configuration. Current implementation includes mock data fallback.

2. **Rate Limiting**: Lattes platform may implement rate limiting. The tool includes 2-second delays between requests.

3. **Dynamic Content**: Some Lattes profiles use dynamic JavaScript content that may not be fully captured.

4. **Name Disambiguation**: Conservative name matching may miss valid matches (false negatives) or flag ambiguous ones with low confidence.

5. **Data Completeness**: Extraction depends on profile completeness and standardization by researchers.

### Scope Limitations

1. **Public Data Only**: Only accesses publicly available information from Lattes profiles.

2. **No Authentication**: Does not access private or restricted profile sections.

3. **Time Window Accuracy**: Date parsing attempts to extract years but may encounter ambiguous formats.

4. **Language**: Optimized for Portuguese names and Brazilian institutional structures.

---

## Ethics and Compliance

### Data Privacy

- **Public Data Only**: Tool exclusively uses publicly available CNPq/Lattes profiles
- **No Authentication**: No login, no access to private data
- **No Data Storage**: Does not persist or cache researcher data
- **Anonymization**: Example files use anonymized/fictional data

### Responsible Use

- **Transparent Evidence**: All COI detections include supporting evidence and URLs
- **Confidence Levels**: Conservative approach with confidence scoring
- **Audit Trail**: Complete action logs for verification
- **False Positives**: Users must verify automated detections

### Compliance

- **Robots.txt**: Respects website terms of service
- **Rate Limiting**: Implements delays to avoid overloading servers
- **Public Interest**: Designed for academic integrity and transparency
- **No Harm**: Conservative disambiguation avoids wrongful accusations

### Intended Use

This tool is designed for:
- Academic peer review processes
- Grant review panels
- Research integrity assessments
- Institutional transparency initiatives

**Not intended for:**
- Automated decision-making without human review
- Public disclosure without verification
- Employment or promotion decisions as sole criterion
- Legal proceedings without corroboration

---

## Troubleshooting

### "browser-use library not installed" Warning

**Issue:** Browser automation is not available.

**Solution:** Install browser-use:
```bash
pip install browser-use playwright
playwright install
```

**Alternative:** The tool works with mock data for testing and development.

### No COI Detected Despite Known Collaboration

**Possible Causes:**
1. Publications outside the time window
2. Different name formats preventing matching
3. Incomplete Lattes profiles
4. Conservative name matching

**Solution:** Review warnings in output, verify profile completeness, consider adjusting time window.

### Slow Execution

**Cause:** Rate limiting delays between profile accesses.

**Solution:** This is intentional to respect server resources. For many researchers, consider breaking into batches.

### Connection Errors

**Cause:** Network issues or Lattes platform unavailability.

**Solution:** Check network connection, verify Lattes platform is accessible, retry after delay.

---

## Examples

See `examples/` directory for:
- `input_example.json`: Sample input structure
- `output_example.json`: Sample output with COI detections
- `README.md`: Usage examples and notes

---

## References

- **CNPq Lattes Platform**: http://lattes.cnpq.br/
- **Search Portal**: https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar
- **Project**: Agents4Gov (LABIC – ICMC/USP)
- **Tool Creation Guide**: [How to Create a Tool](../../docs/how_to_create_tool.md)

---

## License

This tool is part of the Agents4Gov project and is licensed under the **MIT License**.

---

## Contributing

For issues, improvements, or questions about this tool:
1. Review the implementation in `lattes_navigator.py`
2. Check the schema in `schema.json`
3. Test with examples in `examples/`
4. Consult the [tool creation guide](../../docs/how_to_create_tool.md)

---

## Acknowledgments

Developed by **LABIC – Laboratory of Computational Intelligence (ICMC/USP)** as part of the Agents4Gov project for modernizing public sector services with LLM-based tools.
