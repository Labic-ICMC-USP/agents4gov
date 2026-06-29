# How to Create a Tool for Agents4Gov

This guide will walk you through creating a tool that can be used by agents in the Agents4Gov framework. We'll use the `tools/open_alex_doi.py` file as a reference example.

## Table of Contents
1. [Tool Structure Overview](#tool-structure-overview)
2. [Step 1: Set Up Basic Class Structure](#step-1-set-up-basic-class-structure)
3. [Step 2: Define Helper Methods](#step-2-define-helper-methods)
4. [Step 3: Create the Main Tool Method](#step-3-create-the-main-tool-method)
5. [Step 4: Add Parameter Definitions with Pydantic](#step-4-add-parameter-definitions-with-pydantic)
6. [Step 5: Write Comprehensive Docstrings](#step-5-write-comprehensive-docstrings)
7. [Step 6: Implement the Core Logic](#step-6-implement-the-core-logic)
8. [Step 7: Handle Errors Gracefully](#step-7-handle-errors-gracefully)
9. [Step 8: Return Structured Data](#step-8-return-structured-data)
10. [Best Practices](#best-practices)

---

## Tool Structure Overview

A tool in Agents4Gov is a Python class that provides specific functionality to agents. Each tool:
- Lives in the `tools/` directory
- Contains a `Tools` class with methods that agents can call
- Uses Pydantic for parameter validation and description
- Returns structured data (typically JSON strings)
- Includes comprehensive error handling

---

## Step 1: Set Up Basic Class Structure

Create a new Python file in the `tools/` directory (e.g., `tools/my_tool.py`).

Start with the basic imports and class structure:

```python
import os
import requests
import json
from pydantic import Field

class Tools:
    def __init__(self):
        pass
```

**Key Points:**
- Import necessary libraries (`requests` for API calls, `json` for data handling, `pydantic` for validation)
- Always name the class `Tools`
- Include an `__init__` method (even if it just passes)

**Reference:** `tools/open_alex_doi.py:1-8`

---

## Step 2: Define Helper Methods

Helper methods are private methods (prefixed with `_`) that support your main tool functionality.

```python
def _clean_doi(self, doi: str) -> str:
    """
    Clean and normalize a DOI string by removing common prefixes.

    Args:
        doi: The DOI string to clean

    Returns:
        Cleaned DOI string without prefixes like 'doi:', 'https://doi.org/', etc.
    """
    doi_clean = doi.strip()

    # Remove common DOI prefixes
    if doi_clean.lower().startswith('doi:'):
        doi_clean = doi_clean[4:].strip()
    if doi_clean.startswith('https://doi.org/'):
        doi_clean = doi_clean.replace('https://doi.org/', '')
    if doi_clean.startswith('http://doi.org/'):
        doi_clean = doi_clean.replace('http://doi.org/', '')

    return doi_clean
```

**Key Points:**
- Use underscore prefix (`_`) for private methods
- Add type hints for parameters and return values
- Include docstrings explaining purpose, arguments, and return values
- Keep helper methods focused on a single task

**Reference:** `tools/open_alex_doi.py:10-30`

---

## Step 3: Create the Main Tool Method

This is the method that agents will actually call. It should be public (no underscore prefix).

```python
def get_openalex_metadata_by_doi(
    self,
    doi: str = Field(
        ...,
        description="The DOI (Digital Object Identifier) of the publication"
    )
) -> str:
    """
    Retrieve metadata for a scientific publication from OpenAlex API.

    Args:
        doi: The DOI of the publication to query

    Returns:
        JSON string with structured publication data
    """
    # Implementation here
```

**Key Points:**
- Use descriptive method names that clearly indicate functionality
- Method should accept `self` as first parameter
- Return type should typically be `str` (JSON string) for complex data

**Reference:** `tools/open_alex_doi.py:32-51`

---

## Step 4: Add Parameter Definitions with Pydantic

Use Pydantic's `Field` to define parameters with descriptions that help agents understand how to use your tool.

```python
def my_tool_method(
    self,
    required_param: str = Field(
        ...,  # The ellipsis (...) means this parameter is required
        description="Clear description of what this parameter does and example values"
    ),
    optional_param: str = Field(
        default="default_value",
        description="Description of optional parameter with its default value"
    )
) -> str:
```

**Key Points:**
- `...` in `Field(...)` indicates a required parameter
- Always include a descriptive `description` that includes:
  - What the parameter is for
  - Expected format or examples
  - Any constraints or special values
- Use appropriate types (str, int, bool, etc.)

**Reference:** `tools/open_alex_doi.py:33-37`

---

## Step 5: Write Comprehensive Docstrings

Every method needs a docstring that explains what it does, its parameters, and what it returns.

```python
def get_openalex_metadata_by_doi(self, doi: str = Field(...)) -> str:
    """
    Retrieve essential metadata and impact indicators for a scientific publication from OpenAlex API.

    Returns a JSON string containing:
    - Basic metadata (title, authors, venue, publication year)
    - Impact indicators (citations, percentiles, FWCI)

    Args:
        doi: The DOI of the publication to query

    Returns:
        JSON string with structured publication data and impact metrics
    """
```

**Key Points:**
- Start with a one-line summary
- Add detailed description if needed
- List what data the method returns
- Document all parameters in the Args section
- Specify return type in the Returns section

**Reference:** `tools/open_alex_doi.py:39-51`

---

## Step 6: Implement the Core Logic

Implement the main functionality of your tool with clear comments and sections.

```python
# Clean the input
doi_clean = self._clean_doi(doi)

# Build API endpoint URL
base_url = f"https://api.openalex.org/works/doi:{doi_clean}"

# Handle environment variables for configuration
email = os.getenv("OPENALEX_EMAIL", None)
params = {}
if email:
    params['mailto'] = email

try:
    # Make API request
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    # ========================================
    # BASIC METADATA EXTRACTION
    # ========================================

    title = data.get('title', None)
    publication_year = data.get('publication_year', None)

    # Extract and format complex nested data
    authors_list = data.get('authorships', [])
    authors = [
        author_info.get('author', {}).get('display_name')
        for author_info in authors_list
    ]
```

**Key Points:**
- Use clear section comments with visual separators
- Call helper methods for data cleaning/processing
- Support environment variables for API keys or configuration
- Always set timeouts on API requests
- Use `.get()` for safe dictionary access
- Handle nested data structures carefully

**Reference:** `tools/open_alex_doi.py:53-94`

---

## Step 7: Handle Errors Gracefully

Implement comprehensive error handling to help users understand what went wrong.

```python
try:
    # Main logic here
    response = requests.get(base_url, params=params, timeout=10)
    response.raise_for_status()
    # ... processing ...

except requests.exceptions.HTTPError as e:
    # Handle HTTP errors (e.g., 404 Not Found)
    error_result = {
        'status': 'error',
        'error_type': 'http_error',
        'error_code': e.response.status_code,
        'message': f'Publication not found for DOI: {doi_clean}' if e.response.status_code == 404 else str(e),
        'doi': doi_clean
    }
    return json.dumps(error_result, ensure_ascii=False, indent=2)

except requests.exceptions.RequestException as e:
    # Handle connection errors
    error_result = {
        'status': 'error',
        'error_type': 'connection_error',
        'message': f'Error connecting to API: {str(e)}',
        'doi': doi_clean
    }
    return json.dumps(error_result, ensure_ascii=False, indent=2)

except Exception as e:
    # Handle any other unexpected errors
    error_result = {
        'status': 'error',
        'error_type': 'unexpected_error',
        'message': f'Unexpected error: {str(e)}',
        'doi': doi_clean
    }
    return json.dumps(error_result, ensure_ascii=False, indent=2)
```

**Key Points:**
- Catch specific exceptions first, then general ones
- Return structured error information as JSON
- Include `status` field to indicate success/failure
- Include `error_type` to categorize the error
- Provide helpful error messages
- Include relevant context (e.g., the DOI that was queried)

**Reference:** `tools/open_alex_doi.py:166-195`

---

## Step 8: Return Structured Data

Return data in a consistent, well-structured JSON format.

```python
# Build structured response
result = {
    'status': 'success',
    'doi': doi_clean,
    'openalex_id': data.get('id'),

    # Group related data into nested objects
    'metadata': {
        'title': title,
        'authors': authors,
        'venue': venue_name,
        'publication_year': publication_year,
        'publication_date': publication_date,
        'type': type_crossref
    },

    # Group impact metrics separately
    'impact_indicators': {
        'cited_by_count': cited_by_count,
        'citation_normalized_percentile': {
            'value': percentile_value,
            'is_in_top_1_percent': is_top_1_percent
        },
        'cited_by_percentile_year': {
            'min': percentile_min,
            'max': percentile_max
        },
        'fwci': fwci
    },

    # Provide useful links
    'links': {
        'doi_url': f'https://doi.org/{doi_clean}',
        'openalex_url': data.get('id')
    }
}

# Return as formatted JSON string
return json.dumps(result, ensure_ascii=False, indent=2)
```

**Key Points:**
- Always include a `status` field ('success' or 'error')
- Group related data into nested objects
- Use consistent naming conventions (snake_case)
- Use `ensure_ascii=False` to properly handle unicode characters
- Use `indent=2` for readable output
- Return as JSON string, not dictionary

**Reference:** `tools/open_alex_doi.py:123-160`

---

## Best Practices

### 1. **Clear Naming**
- Use descriptive method names: `get_openalex_metadata_by_doi` (good) vs `get_data` (bad)
- Use verb + noun pattern: `get_`, `fetch_`, `create_`, `update_`, etc.

### 2. **Input Validation**
- Clean and normalize inputs using helper methods
- Validate parameters before using them
- Use Pydantic Field descriptions to guide users

### 3. **Environment Variables**
- Use environment variables for API keys and configuration
- Provide defaults with `os.getenv("VAR_NAME", default_value)`
- Document required environment variables in docstrings

### 4. **API Best Practices**
- Always set timeouts on requests
- Use appropriate HTTP methods
- Handle rate limiting if applicable
- Include user agent or email for polite API access

### 5. **Error Messages**
- Be specific about what went wrong
- Include context (what operation failed, with what input)
- Suggest solutions when possible
- Return errors as structured JSON, not by raising exceptions

### 6. **Documentation**
- Write clear docstrings for all public methods
- Include examples in docstrings when helpful
- Comment complex logic sections
- Use visual separators for different sections

### 7. **Testing Considerations**
- Make methods testable by isolating concerns
- Use helper methods for reusable logic
- Consider edge cases in error handling
- Test with invalid inputs

### 8. **Return Format**
- Always return JSON strings for complex data
- Include status indicator in responses
- Group related fields into nested objects
- Use consistent field naming across tools

---

## Complete Example Template

Here's a complete template you can use as a starting point:

```python
import os
import requests
import json
from pydantic import Field

class Tools:
    def __init__(self):
        pass

    def _helper_method(self, input_data: str) -> str:
        """
        Brief description of what this helper does.

        Args:
            input_data: Description of input

        Returns:
            Description of output
        """
        # Implementation
        return processed_data

    def main_tool_method(
        self,
        required_param: str = Field(
            ...,
            description="Clear description with examples"
        ),
        optional_param: str = Field(
            default="default",
            description="Description of optional parameter"
        )
    ) -> str:
        """
        Brief description of what this tool does.

        Longer description with details about:
        - What data it returns
        - What operations it performs
        - Any important notes

        Args:
            required_param: Description of required parameter
            optional_param: Description of optional parameter

        Returns:
            JSON string with structured results
        """

        # Clean/validate inputs
        processed_input = self._helper_method(required_param)

        # Get configuration
        api_key = os.getenv("API_KEY", None)

        try:
            # Main logic
            response = requests.get(
                "https://api.example.com/endpoint",
                headers={"Authorization": f"Bearer {api_key}"} if api_key else {},
                timeout=10
            )
            response.raise_for_status()
            data = response.json()

            # Extract and structure data
            result = {
                'status': 'success',
                'input': processed_input,
                'data': {
                    'field1': data.get('field1'),
                    'field2': data.get('field2')
                }
            }

            return json.dumps(result, ensure_ascii=False, indent=2)

        except requests.exceptions.HTTPError as e:
            error_result = {
                'status': 'error',
                'error_type': 'http_error',
                'error_code': e.response.status_code,
                'message': str(e)
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)

        except Exception as e:
            error_result = {
                'status': 'error',
                'error_type': 'unexpected_error',
                'message': str(e)
            }
            return json.dumps(error_result, ensure_ascii=False, indent=2)
```

---

## Next Steps

1. **Create your tool file** in the `tools/` directory
2. **Implement the basic structure** following this guide
3. **Test your tool** with various inputs including edge cases
4. **Document any environment variables** needed
5. **Add your tool to the agent's configuration** so it can be discovered and used

## Additional Resources

- Review `tools/open_alex_doi.py` for a complete working example
- Check Pydantic documentation for advanced field validation
- See the agents configuration to understand how tools are loaded

---

**Remember:** A good tool is reliable, well-documented, and handles errors gracefully. Take time to write clear code that other developers (and AI agents) can easily understand and use.
