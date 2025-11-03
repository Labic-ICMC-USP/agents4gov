import os
import requests
import json
from pydantic import Field

class Tools:
    def __init__(self):
        pass

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

    def get_openalex_metadata_by_doi(
        self,
        doi: str = Field(
            ...,
            description="The DOI (Digital Object Identifier) of the publication, e.g., '10.1371/journal.pone.0000000'"
        )
    ) -> str:
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
        
        # Clean the DOI using the helper function
        doi_clean = self._clean_doi(doi)
        
        # Build OpenAlex API endpoint URL
        base_url = f"https://api.openalex.org/works/doi:{doi_clean}"
        
        # Optional: Add email for polite pool access (faster and more reliable)
        # Set OPENALEX_EMAIL environment variable to use this feature
        email = os.getenv("OPENALEX_EMAIL", None)
        params = {}
        if email:
            params['mailto'] = email
        
        try:
            # Make request to OpenAlex API
            response = requests.get(base_url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # ========================================
            # BASIC METADATA EXTRACTION
            # ========================================
            
            # Extract core publication information
            title = data.get('title', None)
            publication_year = data.get('publication_year', None)
            publication_date = data.get('publication_date', None)
            type_crossref = data.get('type_crossref', None)
            
            # Extract and format authors list
            # Only include author name for simplicity
            authors_list = data.get('authorships', [])
            authors = [
                author_info.get('author', {}).get('display_name')
                for author_info in authors_list
            ]
            
            # Extract venue/journal information
            primary_location = data.get('primary_location', {})
            source = primary_location.get('source', {}) or {}
            venue_name = source.get('display_name')
            
            # ========================================
            # IMPACT INDICATORS EXTRACTION
            # ========================================
            
            # Total number of citations
            cited_by_count = data.get('cited_by_count', 0)
            
            # Citation normalized percentile
            # Compares citation count to similar publications (by year, type, field)
            citation_normalized_percentile = data.get('citation_normalized_percentile', {}) or {}
            percentile_value = citation_normalized_percentile.get('value')
            is_top_1_percent = citation_normalized_percentile.get('is_in_top_1_percent', False)
            
            # Cited by percentile year
            # Percentile ranking among publications from the same year
            cited_by_percentile_year = data.get('cited_by_percentile_year', {}) or {}
            percentile_min = cited_by_percentile_year.get('min')
            percentile_max = cited_by_percentile_year.get('max')
            
            # Field-Weighted Citation Impact (FWCI)
            # Value of 1.0 means average for the field
            # >1.0 means above average, <1.0 means below average
            fwci = data.get('fwci')
            
            # ========================================
            # BUILD STRUCTURED RESPONSE
            # ========================================
            
            result = {
                'status': 'success',
                'doi': doi_clean,
                'openalex_id': data.get('id'),
                
                # Basic publication metadata
                'metadata': {
                    'title': title,
                    'authors': authors,
                    'venue': venue_name,
                    'publication_year': publication_year,
                    'publication_date': publication_date,
                    'type': type_crossref
                },
                
                # Citation and impact metrics
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
                
                # Useful links
                'links': {
                    'doi_url': f'https://doi.org/{doi_clean}',
                    'openalex_url': data.get('id')
                }
            }
            
            # Return as formatted JSON string
            return json.dumps(result, ensure_ascii=False, indent=2)
            
        # ========================================
        # ERROR HANDLING
        # ========================================
        
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
                'message': f'Error connecting to OpenAlex API: {str(e)}',
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