import os
import json
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dateutil import parser as date_parser
from pydantic import Field
import re
from collections import defaultdict
import time

# Browser-use imports
try:
    from browser_use import Agent
    from langchain_openai import ChatOpenAI
    BROWSER_USE_AVAILABLE = True
except ImportError:
    BROWSER_USE_AVAILABLE = False

class Tools:
    def __init__(self):
        """Initialize the CNPq/Lattes Navigator tool."""
        self.start_url = "https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar"
        self.current_year = datetime.now().year
        self.browser_available = BROWSER_USE_AVAILABLE
        self.rate_limit_delay = 2.0  # seconds between requests
    
    def analyze_researchers_coi(
        self,
        researchers_json: str = Field(
            ...,
            description='JSON string containing list of researchers with format: [{"name": "Researcher Name", "lattes_id": "1234567890123456"}]'
        ),
        time_window: int = Field(
            default=5,
            description="Number of years to look back for production analysis and COI detection (default: 5)"
        ),
        coi_rules_config: str = Field(
            default='{"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}',
            description='JSON string to enable/disable COI rules. Format: {"R1": true, "R2": false, ...}'
        )
    ) -> str:
        """
        Analyze researchers from CNPq/Lattes platform for Conflicts of Interest (COI) and summarize their academic production.
        
        Navigates public CNPq/Lattes profiles to extract:
        - Academic production over the last N years (publications, projects, advising)
        - Institutional affiliations and collaborations
        - Potential conflicts of interest between researchers
        
        Returns a JSON string containing:
        - Per-researcher production summaries
        - Pairwise COI matrix with evidence
        - Confidence levels for each COI detection
        - Evidence URLs and warnings
        
        Args:
            researchers_json: JSON string with list of researchers (name and lattes_id)
            time_window: Years to analyze (default: 5)
            coi_rules_config: JSON to enable/disable specific COI rules
        
        Returns:
            JSON string with structured analysis results
        """
        
        try:
            # Parse input parameters
            researchers = json.loads(researchers_json)
            coi_config = json.loads(coi_rules_config)
            
            # Validate input
            if not isinstance(researchers, list) or len(researchers) == 0:
                return self._error_response("invalid_input", "researchers_json must be a non-empty list")
            
            # Calculate date cutoff
            cutoff_date = datetime.now() - timedelta(days=time_window * 365)
            
            # Initialize results structure
            results = {
                'status': 'success',
                'execution_metadata': {
                    'execution_date': datetime.now().isoformat(),
                    'time_window_years': time_window,
                    'cutoff_date': cutoff_date.isoformat(),
                    'num_researchers': len(researchers),
                    'coi_rules_active': coi_config
                },
                'researchers': [],
                'coi_matrix': {
                    'pairs': []
                },
                'summary_text': ''
            }
            
            # Extract data for each researcher
            researcher_data = []
            for researcher in researchers:
                name = researcher.get('name', '')
                lattes_id = researcher.get('lattes_id', '')
                
                if not name or not lattes_id:
                    results['researchers'].append({
                        'person': {'name': name, 'lattes_id': lattes_id},
                        'warnings': ['Missing name or lattes_id'],
                        'production_5y': {},
                        'coauthors_5y': [],
                        'evidence_urls': []
                    })
                    continue
                
                # Extract researcher profile data
                profile_data = self._extract_researcher_profile(name, lattes_id, cutoff_date)
                researcher_data.append(profile_data)
                results['researchers'].append(profile_data)
            
            # Perform pairwise COI analysis
            coi_pairs = self._analyze_coi_pairwise(researcher_data, coi_config, cutoff_date)
            results['coi_matrix']['pairs'] = coi_pairs
            
            # Generate summary text
            results['summary_text'] = self._generate_summary(results)
            
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError as e:
            return self._error_response('json_parse_error', f'Invalid JSON input: {str(e)}')
        except Exception as e:
            return self._error_response('unexpected_error', f'Unexpected error: {str(e)}')
    
    # ========================================
    # HELPER METHODS - DATA EXTRACTION
    # ========================================
    
    def _extract_researcher_profile(self, name: str, lattes_id: str, cutoff_date: datetime) -> Dict[str, Any]:
        """
        Extract profile data for a single researcher from CNPq/Lattes.
        
        Args:
            name: Researcher name
            lattes_id: Lattes ID
            cutoff_date: Date threshold for filtering data
        
        Returns:
            Dictionary with researcher profile data
        """
        profile_url = f"http://lattes.cnpq.br/{lattes_id}"
        warnings = []
        evidence_urls = [profile_url]
        
        # Check if browser-use is available
        if not self.browser_available:
            warnings.append("browser-use library not installed - using mock data. Install with: pip install browser-use")
            return self._mock_researcher_profile(name, lattes_id, profile_url, warnings)
        
        # Rate limiting
        time.sleep(self.rate_limit_delay)
        
        try:
            # Use browser-use to extract data
            extracted_data = self._navigate_and_extract_lattes(profile_url, name, cutoff_date)
            
            if extracted_data is None:
                warnings.append("Failed to extract data from profile - using partial/mock data")
                return self._mock_researcher_profile(name, lattes_id, profile_url, warnings)
            
            # Process and structure the extracted data
            profile_data = {
                'person': {
                    'name': name,
                    'lattes_id': lattes_id,
                    'profile_url': profile_url,
                    'last_update': extracted_data.get('last_update')
                },
                'production_5y': self._process_production_data(extracted_data, cutoff_date),
                'affiliations_5y': self._process_affiliations(extracted_data, cutoff_date),
                'coauthors_5y': self._extract_coauthors(extracted_data, cutoff_date),
                'warnings': warnings + extracted_data.get('warnings', []),
                'evidence_urls': evidence_urls
            }
            
            return profile_data
            
        except Exception as e:
            warnings.append(f"Error during extraction: {str(e)}")
            return self._mock_researcher_profile(name, lattes_id, profile_url, warnings)
    
    def _mock_researcher_profile(self, name: str, lattes_id: str, profile_url: str, warnings: List[str]) -> Dict[str, Any]:
        """
        Create a mock profile structure when actual extraction is not available.
        
        Args:
            name: Researcher name
            lattes_id: Lattes ID
            profile_url: Profile URL
            warnings: List of warnings
        
        Returns:
            Mock profile data structure
        """
        return {
            'person': {
                'name': name,
                'lattes_id': lattes_id,
                'profile_url': profile_url,
                'last_update': None
            },
            'production_5y': {
                'publications': {
                    'total': 0,
                    'by_type': {},
                    'top_items': []
                },
                'projects': {
                    'total': 0,
                    'active': [],
                    'concluded': []
                },
                'advising': {
                    'total': 0,
                    'ongoing': [],
                    'concluded': []
                },
                'activities': []
            },
            'affiliations_5y': [],
            'coauthors_5y': [],
            'warnings': warnings,
            'evidence_urls': [profile_url]
        }
    
    def _navigate_and_extract_lattes(self, profile_url: str, name: str, cutoff_date: datetime) -> Optional[Dict[str, Any]]:
        """
        Use browser-use to navigate to Lattes profile and extract data.
        
        Args:
            profile_url: URL of the Lattes profile
            name: Researcher name
            cutoff_date: Date threshold for filtering
        
        Returns:
            Extracted data dictionary or None if failed
        """
        # NOTE: Browser-use requires async execution and an LLM
        # This is a placeholder that shows the intended structure
        # In production, this would be executed in an async context with proper LLM setup
        
        warnings = []
        warnings.append("Browser navigation is mocked - actual browser-use integration requires async context and LLM setup")
        
        # Placeholder structure for what would be extracted
        return {
            'last_update': None,
            'warnings': warnings,
            'publications': [],
            'projects': [],
            'advising': [],
            'affiliations': [],
            'activities': []
        }
    
    def _process_production_data(self, extracted_data: Dict[str, Any], cutoff_date: datetime) -> Dict[str, Any]:
        """
        Process and structure production data.
        
        Args:
            extracted_data: Raw extracted data
            cutoff_date: Date threshold
        
        Returns:
            Structured production data
        """
        publications = extracted_data.get('publications', [])
        projects = extracted_data.get('projects', [])
        advising = extracted_data.get('advising', [])
        activities = extracted_data.get('activities', [])
        
        # Filter by date and count by type
        pub_by_type = defaultdict(int)
        filtered_pubs = []
        
        for pub in publications:
            year = self._parse_year(pub.get('year'))
            if self._is_within_window(year, cutoff_date):
                filtered_pubs.append(pub)
                pub_type = pub.get('type', 'other')
                pub_by_type[pub_type] += 1
        
        # Filter projects
        active_projects = []
        concluded_projects = []
        
        for proj in projects:
            start_year = self._parse_year(proj.get('start_year'))
            end_year = self._parse_year(proj.get('end_year'))
            
            # Check if project overlaps with window
            if start_year and self._is_within_window(start_year, cutoff_date):
                if proj.get('status') == 'active':
                    active_projects.append(proj)
                else:
                    concluded_projects.append(proj)
            elif end_year and self._is_within_window(end_year, cutoff_date):
                concluded_projects.append(proj)
        
        # Filter advising
        ongoing_advising = []
        concluded_advising = []
        
        for adv in advising:
            year = self._parse_year(adv.get('year'))
            if self._is_within_window(year, cutoff_date):
                if adv.get('status') == 'ongoing':
                    ongoing_advising.append(adv)
                else:
                    concluded_advising.append(adv)
        
        # Filter activities
        filtered_activities = []
        for act in activities:
            year = self._parse_year(act.get('year'))
            if self._is_within_window(year, cutoff_date):
                filtered_activities.append(act)
        
        return {
            'publications': {
                'total': len(filtered_pubs),
                'by_type': dict(pub_by_type),
                'top_items': filtered_pubs[:10]  # Top 10 most recent
            },
            'projects': {
                'total': len(active_projects) + len(concluded_projects),
                'active': active_projects,
                'concluded': concluded_projects
            },
            'advising': {
                'total': len(ongoing_advising) + len(concluded_advising),
                'ongoing': ongoing_advising,
                'concluded': concluded_advising
            },
            'activities': filtered_activities
        }
    
    def _process_affiliations(self, extracted_data: Dict[str, Any], cutoff_date: datetime) -> List[Dict[str, Any]]:
        """
        Process and filter affiliation data.
        
        Args:
            extracted_data: Raw extracted data
            cutoff_date: Date threshold
        
        Returns:
            List of affiliations within the time window
        """
        affiliations = extracted_data.get('affiliations', [])
        filtered = []
        
        for aff in affiliations:
            start_year = self._parse_year(aff.get('start_year'))
            end_year = self._parse_year(aff.get('end_year'))
            
            # Check if affiliation overlaps with window
            if start_year and self._is_within_window(start_year, cutoff_date):
                filtered.append(aff)
            elif end_year and self._is_within_window(end_year, cutoff_date):
                filtered.append(aff)
            elif start_year is None and end_year is None:
                # Current affiliation with no dates
                filtered.append(aff)
        
        return filtered
    
    def _extract_coauthors(self, extracted_data: Dict[str, Any], cutoff_date: datetime) -> List[Dict[str, str]]:
        """
        Extract unique coauthors from publications.
        
        Args:
            extracted_data: Raw extracted data
            cutoff_date: Date threshold
        
        Returns:
            List of unique coauthors with counts
        """
        publications = extracted_data.get('publications', [])
        coauthor_counts = defaultdict(int)
        
        for pub in publications:
            year = self._parse_year(pub.get('year'))
            if self._is_within_window(year, cutoff_date):
                authors = pub.get('authors', [])
                for author in authors:
                    normalized = self._normalize_name(author)
                    if normalized:
                        coauthor_counts[author] += 1
        
        # Sort by count
        sorted_coauthors = sorted(
            coauthor_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'name': name, 'count': count}
            for name, count in sorted_coauthors
        ]
    
    # ========================================
    # HELPER METHODS - NAME NORMALIZATION
    # ========================================
    
    def _normalize_name(self, name: str) -> str:
        """
        Normalize a person's name for comparison.
        
        Args:
            name: Name to normalize
        
        Returns:
            Normalized name (lowercase, no accents, trimmed)
        """
        if not name:
            return ""
        
        # Convert to lowercase and strip
        normalized = name.lower().strip()
        
        # Remove extra whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Simple accent removal (for Portuguese names)
        accent_map = {
            'á': 'a', 'à': 'a', 'â': 'a', 'ã': 'a', 'ä': 'a',
            'é': 'e', 'è': 'e', 'ê': 'e', 'ë': 'e',
            'í': 'i', 'ì': 'i', 'î': 'i', 'ï': 'i',
            'ó': 'o', 'ò': 'o', 'ô': 'o', 'õ': 'o', 'ö': 'o',
            'ú': 'u', 'ù': 'u', 'û': 'u', 'ü': 'u',
            'ç': 'c', 'ñ': 'n'
        }
        
        for accented, plain in accent_map.items():
            normalized = normalized.replace(accented, plain)
        
        return normalized
    
    def _names_match(self, name1: str, name2: str, threshold: float = 0.8) -> Tuple[bool, str]:
        """
        Check if two names likely refer to the same person.
        
        Args:
            name1: First name
            name2: Second name
            threshold: Similarity threshold (0-1)
        
        Returns:
            Tuple of (match_bool, confidence_level)
        """
        norm1 = self._normalize_name(name1)
        norm2 = self._normalize_name(name2)
        
        # Exact match
        if norm1 == norm2:
            return True, 'high'
        
        # Check if one is substring of other (e.g., "J. Silva" vs "João Silva")
        if norm1 in norm2 or norm2 in norm1:
            return True, 'medium'
        
        # Extract last names
        parts1 = norm1.split()
        parts2 = norm2.split()
        
        if len(parts1) > 0 and len(parts2) > 0:
            last1 = parts1[-1]
            last2 = parts2[-1]
            
            # Same last name is a good indicator
            if last1 == last2:
                return True, 'medium'
        
        return False, 'low'
    
    # ========================================
    # HELPER METHODS - DATE FILTERING
    # ========================================
    
    def _parse_year(self, year_value: Any) -> Optional[int]:
        """
        Parse year from various formats.
        
        Args:
            year_value: Year as string, int, or other format
        
        Returns:
            Year as integer, or None if cannot parse
        """
        if year_value is None:
            return None
        
        # If already an integer
        if isinstance(year_value, int):
            return year_value if 1900 <= year_value <= 2100 else None
        
        # Try to parse as string
        year_str = str(year_value).strip()
        
        # Extract first 4-digit number
        match = re.search(r'\b(19|20)\d{2}\b', year_str)
        if match:
            return int(match.group(0))
        
        return None
    
    def _is_within_window(self, year: Optional[int], cutoff_date: datetime) -> bool:
        """
        Check if a year is within the analysis window.
        
        Args:
            year: Year to check
            cutoff_date: Cutoff date for the window
        
        Returns:
            True if within window, False otherwise
        """
        if year is None:
            return False
        
        cutoff_year = cutoff_date.year
        return year >= cutoff_year
    
    # ========================================
    # HELPER METHODS - COI RULES
    # ========================================
    
    def _check_coi_r1_coauthorship(
        self, 
        researcher_a: Dict[str, Any], 
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R1: Check for co-authorship (at least 1 shared publication).
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        pubs_a = researcher_a.get('production_5y', {}).get('publications', {}).get('top_items', [])
        pubs_b = researcher_b.get('production_5y', {}).get('publications', {}).get('top_items', [])
        
        evidence = []
        shared_count = 0
        
        # Compare publications by title similarity
        for pub_a in pubs_a:
            title_a = self._normalize_name(pub_a.get('title', ''))
            for pub_b in pubs_b:
                title_b = self._normalize_name(pub_b.get('title', ''))
                
                if title_a and title_b and title_a == title_b:
                    shared_count += 1
                    evidence.append(f"Shared publication: {pub_a.get('title')} ({pub_a.get('year')})")
        
        if shared_count > 0:
            return True, 'high', evidence
        
        return False, 'low', []
    
    def _check_coi_r2_advisor_advisee(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R2: Check for advisor-advisee relationship.
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        evidence = []
        
        # Check if researcher_a advised researcher_b
        advising_a = researcher_a.get('production_5y', {}).get('advising', {})
        name_b = researcher_b.get('person', {}).get('name', '')
        
        for advisee in advising_a.get('ongoing', []) + advising_a.get('concluded', []):
            advisee_name = advisee.get('name', '')
            matches, confidence = self._names_match(name_b, advisee_name)
            if matches:
                evidence.append(f"{researcher_a.get('person', {}).get('name')} advised {advisee_name}")
                return True, confidence, evidence
        
        # Check if researcher_b advised researcher_a
        advising_b = researcher_b.get('production_5y', {}).get('advising', {})
        name_a = researcher_a.get('person', {}).get('name', '')
        
        for advisee in advising_b.get('ongoing', []) + advising_b.get('concluded', []):
            advisee_name = advisee.get('name', '')
            matches, confidence = self._names_match(name_a, advisee_name)
            if matches:
                evidence.append(f"{researcher_b.get('person', {}).get('name')} advised {advisee_name}")
                return True, confidence, evidence
        
        return False, 'low', []
    
    def _check_coi_r3_institutional_overlap(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R3: Check for institutional overlap (same department/program concurrently).
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        affiliations_a = researcher_a.get('affiliations_5y', [])
        affiliations_b = researcher_b.get('affiliations_5y', [])
        
        evidence = []
        
        for aff_a in affiliations_a:
            inst_a = self._normalize_name(aff_a.get('institution', ''))
            dept_a = self._normalize_name(aff_a.get('department', ''))
            
            for aff_b in affiliations_b:
                inst_b = self._normalize_name(aff_b.get('institution', ''))
                dept_b = self._normalize_name(aff_b.get('department', ''))
                
                # Same institution and department
                if inst_a and inst_b and inst_a == inst_b:
                    if dept_a and dept_b and dept_a == dept_b:
                        evidence.append(f"Same affiliation: {aff_a.get('institution')} - {aff_a.get('department')}")
                        return True, 'high', evidence
                    else:
                        evidence.append(f"Same institution: {aff_a.get('institution')}")
                        return True, 'medium', evidence
        
        return False, 'low', []
    
    def _check_coi_r4_project_overlap(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R4: Check for project team overlap.
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        projects_a = researcher_a.get('production_5y', {}).get('projects', {})
        projects_b = researcher_b.get('production_5y', {}).get('projects', {})
        
        all_projects_a = projects_a.get('active', []) + projects_a.get('concluded', [])
        all_projects_b = projects_b.get('active', []) + projects_b.get('concluded', [])
        
        evidence = []
        
        for proj_a in all_projects_a:
            title_a = self._normalize_name(proj_a.get('title', ''))
            for proj_b in all_projects_b:
                title_b = self._normalize_name(proj_b.get('title', ''))
                
                if title_a and title_b and title_a == title_b:
                    evidence.append(f"Shared project: {proj_a.get('title')}")
                    return True, 'high', evidence
        
        return False, 'low', []
    
    def _check_coi_r5_committee_overlap(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R5: Check for committee/board/event overlap.
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        activities_a = researcher_a.get('production_5y', {}).get('activities', [])
        activities_b = researcher_b.get('production_5y', {}).get('activities', [])
        
        evidence = []
        
        for act_a in activities_a:
            name_a = self._normalize_name(act_a.get('name', ''))
            for act_b in activities_b:
                name_b = self._normalize_name(act_b.get('name', ''))
                
                if name_a and name_b and name_a == name_b:
                    evidence.append(f"Shared activity: {act_a.get('name')}")
                    return True, 'medium', evidence
        
        return False, 'low', []
    
    def _check_coi_r6_frequent_coauthorship(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R6: Check for frequent co-authorship (>= 3 publications).
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        # First check R1
        triggered, confidence, evidence = self._check_coi_r1_coauthorship(researcher_a, researcher_b, cutoff_date)
        
        # Count shared publications
        shared_count = len(evidence)
        
        if shared_count >= 3:
            return True, 'high', evidence
        
        return False, 'low', []
    
    def _check_coi_r7_lab_group_overlap(
        self,
        researcher_a: Dict[str, Any],
        researcher_b: Dict[str, Any],
        cutoff_date: datetime
    ) -> Tuple[bool, str, List[str]]:
        """
        R7: Check for strong institutional proximity (same lab/group).
        
        Returns:
            Tuple of (triggered, confidence, evidence_list)
        """
        affiliations_a = researcher_a.get('affiliations_5y', [])
        affiliations_b = researcher_b.get('affiliations_5y', [])
        
        evidence = []
        
        for aff_a in affiliations_a:
            lab_a = self._normalize_name(aff_a.get('lab_group', ''))
            
            for aff_b in affiliations_b:
                lab_b = self._normalize_name(aff_b.get('lab_group', ''))
                
                if lab_a and lab_b and lab_a == lab_b:
                    evidence.append(f"Same lab/group: {aff_a.get('lab_group')}")
                    return True, 'high', evidence
        
        return False, 'low', []
    
    # ========================================
    # HELPER METHODS - PAIRWISE COI ANALYSIS
    # ========================================
    
    def _analyze_coi_pairwise(
        self,
        researcher_data: List[Dict[str, Any]],
        coi_config: Dict[str, bool],
        cutoff_date: datetime
    ) -> List[Dict[str, Any]]:
        """
        Perform pairwise COI analysis for all researchers.
        
        Args:
            researcher_data: List of researcher profile data
            coi_config: Configuration for which rules to apply
            cutoff_date: Date cutoff for analysis
        
        Returns:
            List of COI pairs with triggered rules and evidence
        """
        coi_pairs = []
        
        # Compare each pair
        for i in range(len(researcher_data)):
            for j in range(i + 1, len(researcher_data)):
                researcher_a = researcher_data[i]
                researcher_b = researcher_data[j]
                
                lattes_a = researcher_a.get('person', {}).get('lattes_id')
                lattes_b = researcher_b.get('person', {}).get('lattes_id')
                
                # Check each enabled COI rule
                rules_triggered = []
                all_evidence = []
                confidence_levels = []
                
                rule_checks = {
                    'R1': self._check_coi_r1_coauthorship,
                    'R2': self._check_coi_r2_advisor_advisee,
                    'R3': self._check_coi_r3_institutional_overlap,
                    'R4': self._check_coi_r4_project_overlap,
                    'R5': self._check_coi_r5_committee_overlap,
                    'R6': self._check_coi_r6_frequent_coauthorship,
                    'R7': self._check_coi_r7_lab_group_overlap
                }
                
                for rule_id, check_func in rule_checks.items():
                    if coi_config.get(rule_id, True):  # Default to enabled
                        triggered, confidence, evidence = check_func(researcher_a, researcher_b, cutoff_date)
                        
                        if triggered:
                            rules_triggered.append(rule_id)
                            all_evidence.extend(evidence)
                            confidence_levels.append(confidence)
                
                # If any rule triggered, add to pairs
                if rules_triggered:
                    # Determine overall confidence
                    if 'high' in confidence_levels:
                        overall_confidence = 'high'
                    elif 'medium' in confidence_levels:
                        overall_confidence = 'medium'
                    else:
                        overall_confidence = 'low'
                    
                    coi_pairs.append({
                        'a_lattes_id': lattes_a,
                        'b_lattes_id': lattes_b,
                        'a_name': researcher_a.get('person', {}).get('name'),
                        'b_name': researcher_b.get('person', {}).get('name'),
                        'rules_triggered': rules_triggered,
                        'confidence': overall_confidence,
                        'evidence': all_evidence
                    })
        
        return coi_pairs
    
    # ========================================
    # HELPER METHODS - SUMMARY GENERATION
    # ========================================
    
    def _generate_summary(self, results: Dict[str, Any]) -> str:
        """
        Generate a human-readable summary of the analysis.
        
        Args:
            results: Complete results dictionary
        
        Returns:
            Summary text
        """
        num_researchers = results['execution_metadata']['num_researchers']
        time_window = results['execution_metadata']['time_window_years']
        num_coi_pairs = len(results['coi_matrix']['pairs'])
        
        summary = f"Analysis of {num_researchers} researchers over the last {time_window} years. "
        
        if num_coi_pairs == 0:
            summary += "No conflicts of interest detected."
        else:
            summary += f"Detected {num_coi_pairs} potential conflict(s) of interest. "
            
            # Count by confidence
            high = sum(1 for p in results['coi_matrix']['pairs'] if p['confidence'] == 'high')
            medium = sum(1 for p in results['coi_matrix']['pairs'] if p['confidence'] == 'medium')
            low = sum(1 for p in results['coi_matrix']['pairs'] if p['confidence'] == 'low')
            
            summary += f"Confidence levels: {high} high, {medium} medium, {low} low."
        
        return summary
    
    # ========================================
    # HELPER METHODS - ERROR HANDLING
    # ========================================
    
    def _error_response(self, error_type: str, message: str) -> str:
        """
        Generate a standardized error response.
        
        Args:
            error_type: Type of error
            message: Error message
        
        Returns:
            JSON string with error information
        """
        error_result = {
            'status': 'error',
            'error_type': error_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        return json.dumps(error_result, ensure_ascii=False, indent=2)

