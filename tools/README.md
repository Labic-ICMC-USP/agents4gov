# Tools

This directory contains tools that can be used by agents in the Agents4Gov framework. Each tool provides specific functionality that agents can call to perform tasks.

## Available Tools

### OpenAlex
- **[openalex/open_alex_doi.py](openalex/README.md)** - Retrieves metadata and impact indicators for scientific publications using DOI

### OpenML
- **[openml/openml_search.py](openml/README.md)** - Search for machine learning datasets using semantic similarity with embeddings
- **[openml/openml_download.py](openml/README.md)** - Download datasets from OpenML by ID and save as CSV
- **[openml/openml_knn_train.py](openml/README.md)** - Train KNN models with hyperparameter tuning via cross-validation

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
5. Copy and paste the content of the tool file
6. Save and enable the tool
7. The tool will now be available for agents to use in conversations

### Method 2: Direct File Import

If Open WebUI supports file-based tool loading:

1. Ensure the `tools/` directory is in the Open WebUI tools path
2. Restart Open WebUI to detect new tools
3. Enable the tool in the Tools settings

## Tool Requirements

All tools in this directory require:
- **Python 3.11+**
- **Open WebUI** installed and running
- **pydantic** library for parameter validation

## Creating Your Own Tools

Want to create a new tool? Follow our comprehensive guide:

📖 **[How to Create a Tool Tutorial](../docs/how_to_create_tool.md)**

The tutorial covers:
- Tool structure and class setup
- Parameter validation with Pydantic
- API integration and error handling
- Returning structured JSON data
- Best practices and examples

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

## Contributing New Tools

When adding a new tool to this directory:

1. **Create the tool file** following the structure in existing tools
2. **Test thoroughly** with various inputs and edge cases
3. **Document the tool** with a README.md in its subdirectory
4. **Add it to this README** under "Available Tools"
5. **Follow best practices** outlined in the [tutorial](../docs/how_to_create_tool.md)

## Additional Resources

- **[Tool Creation Tutorial](../docs/how_to_create_tool.md)** - Step-by-step guide for creating tools
- **[Open WebUI Tools Guide](https://docs.openwebui.com/features/plugin/tools)** - Official Open WebUI tools documentation
- **[Project Documentation](../docs/README.md)** - Main documentation hub
