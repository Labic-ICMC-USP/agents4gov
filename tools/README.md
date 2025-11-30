# Agents4Gov Tools

This directory contains all tool implementations for the Agents4Gov project. Each tool is designed to be imported and used within Open WebUI to provide specific functionality to LLM agents.

## Available Tools

### 1. OpenAlex DOI Metadata Retrieval

**File:** `open_alex_doi.py`

**Description:** Retrieves comprehensive metadata and impact indicators for scientific publications using their DOI (Digital Object Identifier) from the OpenAlex API.

**Main Method:** `get_openalex_metadata_by_doi(doi: str) -> str`

**Features:**
- Fetches basic publication metadata (title, authors, venue, publication year)
- Retrieves citation counts and impact metrics
- Provides normalized percentile rankings
- Calculates Field-Weighted Citation Impact (FWCI)
- Handles multiple DOI formats (with or without prefixes)
- Returns structured JSON output

**Parameters:**
- `doi` (required): The DOI of the publication (e.g., `10.1371/journal.pone.0000000`)
  - Accepts formats: `10.1234/example`, `doi:10.1234/example`, `https://doi.org/10.1234/example`

**Environment Variables:**
- `OPENALEX_EMAIL` (optional): Your email for polite pool access (faster and more reliable API responses)

**Example Output:**
```json
{
  "status": "success",
  "doi": "10.1371/journal.pone.0000000",
  "openalex_id": "https://openalex.org/W2741809807",
  "metadata": {
    "title": "Example Publication Title",
    "authors": ["Author One", "Author Two"],
    "venue": "PLOS ONE",
    "publication_year": 2020,
    "publication_date": "2020-03-15",
    "type": "journal-article"
  },
  "impact_indicators": {
    "cited_by_count": 42,
    "citation_normalized_percentile": {
      "value": 85.5,
      "is_in_top_1_percent": false
    },
    "cited_by_percentile_year": {
      "min": 80,
      "max": 90
    },
    "fwci": 1.5
  },
  "links": {
    "doi_url": "https://doi.org/10.1371/journal.pone.0000000",
    "openalex_url": "https://openalex.org/W2741809807"
  }
}
```

**Use Cases:**
- Research impact analysis
- Literature review automation
- Citation metric extraction
- Publication verification
- Academic database integration

---

### 2. CNPq/Lattes Navigator - COI Detection and 5-Year Production Summary

**File:** `cnpq_lattes_navigator/lattes_navigator.py`

**Description:** Automated tool for navigating public CNPq/Lattes profiles to detect Conflicts of Interest (COI) and summarize academic production over a configurable time window.

**Main Method:** `analyze_researchers_coi(researchers_json, time_window, coi_rules_config)`

**Features:**
- Detects 7 types of conflicts of interest between researchers
- Summarizes academic production over the last N years (default: 5)
- Evidence-based analysis with confidence levels (high/medium/low)
- Browser automation using browser-use library
- Configurable COI rules (enable/disable specific rules)
- Rate limiting and graceful degradation
- Comprehensive output with per-researcher summaries and pairwise COI matrix

**Parameters:**
- `researchers_json` (required): JSON string with list of researchers `[{"name": "...", "lattes_id": "..."}]`
- `time_window` (optional, default: 5): Number of years to analyze
- `coi_rules_config` (optional): JSON string to enable/disable specific rules

**COI Rules:**
- **R1:** Co-authorship (≥1 publication)
- **R2:** Advisor-advisee relationship
- **R3:** Institutional overlap (same department/program)
- **R4:** Project team overlap
- **R5:** Committee/board/event overlap
- **R6:** Frequent co-authorship (≥3 publications)
- **R7:** Strong institutional proximity (same lab/group)

**Dependencies:**
- `browser-use`, `playwright`, `pydantic`, `python-dateutil`

**Use Cases:**
- Academic peer review processes
- Grant review panels
- Research integrity assessments
- Institutional transparency initiatives
- Conflict of interest declarations

**Ethics:**
- Uses only public data
- Provides evidence for all detections
- Conservative disambiguation
- Transparent confidence levels
- Designed for human-in-the-loop verification

---

## How to Use Tools in Open WebUI

### Method 1: Import via UI

1. Start Open WebUI server: `open-webui serve`
2. Access the web interface at [http://localhost:8080](http://localhost:8080)
3. Navigate to **Workspace → Tools**
4. Click **Import Tool** or **+ Create Tool**
5. Copy and paste the content of the tool file (e.g., `open_alex_doi.py`)
6. Save and enable the tool
7. The tool will now be available for agents to use in conversations

### Method 2: Direct File Import

If Open WebUI supports file-based tool loading:

1. Ensure the `tools/` directory is in the Open WebUI tools path
2. Restart Open WebUI to detect new tools
3. Enable the tool in the Tools settings

### Testing a Tool

After importing, test the tool with a simple query:

```
Can you get metadata for the publication with DOI 10.1371/journal.pone.0000000?
```

The agent should automatically invoke the `get_openalex_metadata_by_doi` tool and return the structured results.

---

## Tool Requirements

### General Requirements

All tools in this directory require:
- **Python 3.11+**
- **Open WebUI** installed and running
- **pydantic** library for parameter validation

---

## Creating Your Own Tools

Want to create a new tool? Follow our comprehensive guide:

📖 **[How to Create a Tool Tutorial](../docs/how_to_create_tool.md)**

The tutorial covers:
- Tool structure and class setup
- Parameter validation with Pydantic
- API integration and error handling
- Returning structured JSON data
- Best practices and examples

**Quick Start Template:**

```python
import json
from pydantic import Field

class Tools:
    def __init__(self):
        pass

    def my_tool_method(
        self,
        param: str = Field(
            ...,
            description="Description of parameter"
        )
    ) -> str:
        """
        Tool description.

        Args:
            param: Parameter description

        Returns:
            JSON string with results
        """
        try:
            # Your logic here
            result = {
                'status': 'success',
                'data': 'your data'
            }
            return json.dumps(result, ensure_ascii=False, indent=2)
        except Exception as e:
            error_result = {
                'status': 'error',
                'message': str(e)
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
```

---

## Tool Development Best Practices

1. **Clear Documentation**: Include comprehensive docstrings and parameter descriptions
2. **Error Handling**: Always catch and return structured errors as JSON
3. **Type Hints**: Use Python type hints for all parameters and return values
4. **Structured Output**: Return JSON strings with consistent `status` fields
5. **Environment Variables**: Use env vars for API keys and configuration
6. **Timeouts**: Set timeouts on external API calls
7. **Validation**: Validate and clean input data before processing
8. **Testing**: Test with various inputs including edge cases

---

## Troubleshooting

### Tool Not Appearing in Open WebUI

- Verify the `Tools` class name is correct
- Check for Python syntax errors
- Ensure all required dependencies are installed
- Restart Open WebUI after adding new tools

### Tool Execution Errors

- Check environment variables are set correctly
- Verify internet connectivity for API-based tools
- Review error messages in the JSON response
- Check Open WebUI logs for detailed error information

### Import Errors

- Ensure `pydantic` and other dependencies are installed
- Use Python 3.11+ for compatibility
- Check that the tool file is valid Python code

---

## Contributing New Tools

When adding a new tool to this directory:

1. **Create the tool file** following the structure in `open_alex_doi.py`
2. **Test thoroughly** with various inputs and edge cases
3. **Document the tool** in this README.md under "Available Tools"
4. **Add requirements** if the tool needs specific dependencies
5. **Include examples** showing expected input and output
6. **Follow best practices** outlined in the tutorial

---

## Additional Resources

- **[Project Documentation](../docs/README.md)** - Main documentation hub
- **[Tool Creation Tutorial](../docs/how_to_create_tool.md)** - Step-by-step guide
- **[Open WebUI Tools Guide](https://docs.openwebui.com/features/plugin/tools)** - Official Open WebUI tools documentation
- **[OpenAlex API Documentation](https://docs.openalex.org/)** - For the OpenAlex tool specifically

---

## License

All tools in this directory are part of the Agents4Gov project and are licensed under the **MIT License**.
