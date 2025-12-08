# OpenAlex DOI Metadata Retrieval

**File:** `open_alex_doi.py`

**Description:** Retrieves comprehensive metadata and impact indicators for scientific publications using their DOI (Digital Object Identifier) from the OpenAlex API.

**Main Method:** `get_openalex_metadata_by_doi(doi: str) -> str`

## Features

- Fetches basic publication metadata (title, authors, venue, publication year)
- Retrieves citation counts and impact metrics
- Provides normalized percentile rankings
- Calculates Field-Weighted Citation Impact (FWCI)
- Handles multiple DOI formats (with or without prefixes)
- Returns structured JSON output

## Parameters

- `doi` (required): The DOI of the publication (e.g., `10.1371/journal.pone.0000000`)
  - Accepts formats: `10.1234/example`, `doi:10.1234/example`, `https://doi.org/10.1234/example`

## Environment Variables

- `OPENALEX_EMAIL` (optional): Your email for polite pool access (faster and more reliable API responses)

## Example Output

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

## Use Cases

- Research impact analysis
- Literature review automation
- Citation metric extraction
- Publication verification
- Academic database integration

## Usage

After importing this tool in Open WebUI, test it with a query like:

```
Can you get metadata for the publication with DOI 10.1371/journal.pone.0000000?
```

The agent will automatically invoke the `get_openalex_metadata_by_doi` tool and return the structured results.

## Additional Resources

- **[OpenAlex API Documentation](https://docs.openalex.org/)** - Official API documentation
- **[How to Create a Tool](../../docs/how_to_create_tool.md)** - Guide for creating your own tools
