import os
import json
import asyncio
import re
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict
from pydantic import Field

BROWSER_USE_AVAILABLE = False
BROWSER_IMPORT_ERROR = None

try:
    from browser_use import Agent, Browser, ChatOpenAI
    BROWSER_USE_AVAILABLE = True
except Exception as e:
    BROWSER_IMPORT_ERROR = str(e)


class Tools:
    def __init__(self):
        self.start_url = "https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar"
        self.current_year = datetime.now().year
        self.browser_available = BROWSER_USE_AVAILABLE
        self.rate_limit_delay = 2.0
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        self.use_cloud_browser = os.getenv("BROWSER_USE_CLOUD", "true").lower() == "true"
    
    def analyze_researchers_coi(
        self,
        researchers_json: str = Field(..., description='JSON list: [{"name": "...", "lattes_id": "..."}]'),
        time_window: int = Field(default=5, description="Years to analyze"),
        coi_rules_config: str = Field(
            default='{"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}',
            description='JSON to enable/disable COI rules'
        )
    ) -> str:
        try:
            researchers = json.loads(researchers_json)
            coi_config = json.loads(coi_rules_config)
            
            if not isinstance(researchers, list) or len(researchers) == 0:
                return self._error_response("invalid_input", "researchers_json must be a non-empty list")
            
            cutoff_date = datetime.now() - timedelta(days=time_window * 365)
            
            results = {
                'status': 'success',
                'execution_metadata': {
                    'execution_date': datetime.now().isoformat(),
                    'time_window_years': time_window,
                    'cutoff_date': cutoff_date.isoformat(),
                    'num_researchers': len(researchers),
                    'coi_rules_active': coi_config,
                    'browser_use_available': self.browser_available
                },
                'researchers': [],
                'coi_matrix': {'pairs': []},
                'summary_text': ''
            }
            
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
                
                profile_data = self._extract_researcher_profile(name, lattes_id, cutoff_date)
                researcher_data.append(profile_data)
                results['researchers'].append(profile_data)
            
            coi_pairs = self._analyze_coi_pairwise(researcher_data, coi_config, cutoff_date)
            results['coi_matrix']['pairs'] = coi_pairs
            results['summary_text'] = self._generate_summary(results)
            
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        except json.JSONDecodeError as e:
            return self._error_response('json_parse_error', f'Invalid JSON: {str(e)}')
        except Exception as e:
            return self._error_response('unexpected_error', str(e))
    
    def _extract_researcher_profile(self, name: str, lattes_id: str, cutoff_date: datetime, is_student: bool = False) -> Dict[str, Any]:
        profile_url = f"http://lattes.cnpq.br/{lattes_id}"
        warnings = []
        
        if not self.browser_available:
            warnings.append("browser-use not installed")
            return self._mock_profile(name, lattes_id, profile_url, warnings)
        
        if not self.openai_api_key:
            warnings.append("OPENAI_API_KEY not set")
            return self._mock_profile(name, lattes_id, profile_url, warnings)
        
        time.sleep(self.rate_limit_delay)
        
        try:
            extracted_data = self._run_browser_extraction(profile_url, name, lattes_id, cutoff_date, is_student)
            
            if extracted_data is None:
                warnings.append("Extraction failed")
                return self._mock_profile(name, lattes_id, profile_url, warnings)
            
            # Check for error warnings in extracted data
            data_warnings = extracted_data.get('warnings', [])
            if any(w in data_warnings for w in ['profile_not_found', 'captcha_blocked', 'page_error']):
                warnings.extend(data_warnings)
                return self._mock_profile(name, lattes_id, profile_url, warnings, extracted_data.get('agent_logs', []))
            
            production = self._process_production(extracted_data, cutoff_date)
            coauthors = production.pop('coauthors_extracted', []) or extracted_data.get('coauthors', [])
            
            return {
                'person': {
                    'name': name,
                    'lattes_id': lattes_id,
                    'profile_url': profile_url,
                    'last_update': extracted_data.get('last_update')
                },
                'production_5y': production,
                'affiliations_5y': extracted_data.get('affiliations', []),
                'coauthors_5y': coauthors,
                'warnings': warnings + data_warnings,
                'evidence_urls': [profile_url],
                'agent_logs': extracted_data.get('agent_logs', [])
            }
        except Exception as e:
            warnings.append(f"Error: {str(e)}")
            return self._mock_profile(name, lattes_id, profile_url, warnings)
    
    def _mock_profile(self, name: str, lattes_id: str, profile_url: str, warnings: List[str], agent_logs: List[Dict] = None) -> Dict[str, Any]:
        return {
            'person': {'name': name, 'lattes_id': lattes_id, 'profile_url': profile_url, 'last_update': None},
            'production_5y': {
                'publications': {'total': 0, 'by_type': {}, 'top_items': []},
                'projects': {'total': 0, 'active': [], 'concluded': []},
                'advising': {'total': 0, 'ongoing': [], 'concluded': []},
                'activities': []
            },
            'affiliations_5y': [],
            'coauthors_5y': [],
            'warnings': warnings,
            'evidence_urls': [profile_url],
            'agent_logs': agent_logs or []
        }
    
    def _run_browser_extraction(self, profile_url: str, name: str, lattes_id: str, cutoff_date: datetime, is_student: bool = False) -> Optional[Dict[str, Any]]:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_extraction(profile_url, name, lattes_id, cutoff_date, is_student))
            finally:
                loop.close()
        except Exception as e:
            return {'warnings': [str(e)], 'publications': [], 'projects': [], 'advising': [], 'affiliations': [], 'coauthors': [], 'last_update': None}
    
    async def _async_extraction(self, profile_url: str, name: str, lattes_id: str, cutoff_date: datetime, is_student: bool = False) -> Dict[str, Any]:
        cutoff_year = cutoff_date.year
        current_year = datetime.now().year
        
        llm = ChatOpenAI(model=self.openai_model)
        
        # Build checkbox step only for students
        checkbox_step = ""
        if is_student:
            checkbox_step = """
3. CHECK the checkbox with CSS selector "#buscarDemais" - REQUIRED for student search
4. """
        else:
            checkbox_step = """
3. """
        
        task = f"""
TASK: Find and extract Lattes CV data for "{name}" (Lattes ID: {lattes_id}).

TARGET LATTES ID: {lattes_id}

NAVIGATION:
1. Go to: https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar
2. Type "{name}" in the search field
{checkbox_step}CLICK button "#botaoBuscaFiltros"
{"5" if is_student else "4"}. CLICK link containing "{name}" in results
{"6" if is_student else "5"}. CLICK button "#idbtnabrircurriculo"
{"7" if is_student else "6"}. VERIFY ID: Look at top of CV for "ID Lattes:" text followed by a number.
   The ID must be exactly "{lattes_id}". 
   If the ID shown is DIFFERENT, go BACK and try the NEXT result in the list.

CSS SELECTORS:
- Checkbox: #buscarDemais (use CHECK action)
- Search button: #botaoBuscaFiltros (use CLICK action)
- Open CV button: #idbtnabrircurriculo (use CLICK action)

ID LATTES LOCATION (in CV page):
The ID appears at top of CV like: "ID Lattes: {lattes_id}"
HTML: <li>ID Lattes: <span style="font-weight: bold; color: #326C99;">{lattes_id}</span></li>

EXTRACT (years {cutoff_year}-{current_year}):
- Institution, publications, projects, advising, coauthors

OUTPUT JSON:
```json
{{
  "last_update": null,
  "affiliations": [{{"institution": "...", "department": "..."}}],
  "publications": [{{"title": "...", "year": 2024, "type": "journal", "coauthors": ["..."]}}],
  "projects": [{{"title": "...", "start_year": 2022}}],
  "advising": [{{"name": "...", "level": "PhD", "year": 2023}}],
  "coauthors": [{{"name": "...", "count": 1}}],
  "warnings": []
}}
```

ERRORS (only if NO data found):
- {{"warnings": ["profile_not_found"], ...}} if ID {lattes_id} not found in any result
- {{"warnings": ["captcha_blocked"], ...}} if completely blocked
"""
        
        browser = None
        if self.use_cloud_browser:
            browser = Browser(
                use_cloud=True,
                cloud_proxy_country_code='br',
                cloud_timeout=15,
                wait_between_actions=3.0, 
                wait_for_network_idle_page_load_time=5.0,
                minimum_wait_page_load_time=3.0,
            )
        
        agent = Agent(
            task=task, 
            llm=llm,
            browser=browser,
            max_actions_per_step=1
        )
        
        max_retries = 1
        last_error = None
        
        for attempt in range(max_retries + 1):
            try:
                history = await agent.run(max_steps=50)  # Increased to allow iteration through search results 
                break
            except Exception as retry_error:
                last_error = retry_error
                if attempt < max_retries:
                    await asyncio.sleep(5)
                    continue
                else:
                    return {
                        'warnings': [f'Failed after {max_retries + 1} attempts: {str(last_error)}'],
                        'publications': [], 'projects': [], 'advising': [],
                        'affiliations': [], 'coauthors': [], 'last_update': None,
                        'agent_logs': [{'error': str(last_error)}]
                    }
        
        try:
            agent_logs = []
            all_content = []
            
            if hasattr(history, 'all_results'):
                for i, r in enumerate(history.all_results):
                    step_log = {'step': i + 1}
                    if hasattr(r, 'extracted_content') and r.extracted_content:
                        all_content.append(str(r.extracted_content))
                        step_log['content'] = str(r.extracted_content)[:200]
                    if hasattr(r, 'long_term_memory') and r.long_term_memory:
                        step_log['memory'] = str(r.long_term_memory)[:200]
                    if hasattr(r, 'error') and r.error:
                        step_log['error'] = str(r.error)
                    agent_logs.append(step_log)
            
            if hasattr(history, 'final_result') and history.final_result:
                all_content.append(str(history.final_result))
            
            full_text = '\n'.join(all_content)
            
            json_block = re.search(r'```json\s*([\s\S]*?)\s*```', full_text)
            if json_block:
                try:
                    result = json.loads(json_block.group(1))
                    result['agent_logs'] = agent_logs
                    return result
                except json.JSONDecodeError:
                    pass
            
            json_match = re.search(r'\{[^{}]*"warnings"[^{}]*\}', full_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    result['agent_logs'] = agent_logs
                    return result
                except json.JSONDecodeError:
                    pass
            
            json_match = re.search(r'\{[\s\S]*\}', full_text)
            if json_match:
                try:
                    result = json.loads(json_match.group())
                    result['agent_logs'] = agent_logs
                    return result
                except json.JSONDecodeError:
                    pass
            
            return {
                'warnings': [f'No JSON in response'],
                'publications': [], 'projects': [], 'advising': [], 
                'affiliations': [], 'coauthors': [], 'last_update': None,
                'agent_logs': agent_logs,
                'raw_content': full_text[:1000]
            }
        except Exception as e:
            return {'warnings': [f'Error: {str(e)}'], 'publications': [], 'projects': [], 'advising': [], 'affiliations': [], 'coauthors': [], 'last_update': None, 'agent_logs': []}
    
    def _deduplicate_publications(self, pubs: List[Dict]) -> List[Dict]:
        seen = set()
        unique = []
        for pub in pubs:
            doi = pub.get('doi')
            if doi:
                key = doi.lower()
            else:
                title = self._normalize_name(pub.get('title', ''))
                year = pub.get('year', '')
                key = f"{title}_{year}"
            if key and key not in seen:
                seen.add(key)
                unique.append(pub)
        return unique
    
    def _extract_coauthors(self, pubs: List[Dict]) -> List[Dict]:
        coauthor_count = defaultdict(int)
        for pub in pubs:
            for coauthor in pub.get('coauthors', []):
                if coauthor:
                    norm_name = self._normalize_name(coauthor)
                    coauthor_count[coauthor] += 1
        return [{'name': name, 'count': count} for name, count in sorted(coauthor_count.items(), key=lambda x: -x[1])[:20]]
    
    def _process_production(self, data: Dict[str, Any], cutoff_date: datetime) -> Dict[str, Any]:
        pub_by_type = defaultdict(int)
        filtered_pubs = []
        
        for pub in data.get('publications', []):
            year = self._parse_year(pub.get('year'))
            if self._in_window(year, cutoff_date):
                filtered_pubs.append(pub)
                pub_by_type[pub.get('type', 'other')] += 1
        
        filtered_pubs = self._deduplicate_publications(filtered_pubs)
        coauthors = self._extract_coauthors(filtered_pubs)
        
        active_proj, concluded_proj = [], []
        for proj in data.get('projects', []):
            if self._in_window(self._parse_year(proj.get('start_year')), cutoff_date):
                (active_proj if proj.get('status') == 'active' else concluded_proj).append(proj)
        
        ongoing_adv, concluded_adv = [], []
        for adv in data.get('advising', []):
            if self._in_window(self._parse_year(adv.get('year')), cutoff_date):
                (ongoing_adv if adv.get('status') == 'ongoing' else concluded_adv).append(adv)
        
        activities = []
        for act in data.get('activities', []):
            if self._in_window(self._parse_year(act.get('year')), cutoff_date):
                activities.append(act)
        
        return {
            'publications': {'total': len(filtered_pubs), 'by_type': dict(pub_by_type), 'top_items': filtered_pubs[:10]},
            'projects': {'total': len(active_proj) + len(concluded_proj), 'active': active_proj, 'concluded': concluded_proj},
            'advising': {'total': len(ongoing_adv) + len(concluded_adv), 'ongoing': ongoing_adv, 'concluded': concluded_adv},
            'activities': activities,
            'coauthors_extracted': coauthors
        }
    
    def _normalize_name(self, name: str) -> str:
        if not name:
            return ""
        normalized = re.sub(r'\s+', ' ', name.lower().strip())
        for a, p in [('á','a'),('à','a'),('â','a'),('ã','a'),('é','e'),('ê','e'),('í','i'),('ó','o'),('ô','o'),('õ','o'),('ú','u'),('ç','c')]:
            normalized = normalized.replace(a, p)
        return normalized
    
    def _names_match(self, n1: str, n2: str) -> Tuple[bool, str]:
        norm1, norm2 = self._normalize_name(n1), self._normalize_name(n2)
        if norm1 == norm2:
            return True, 'high'
        if norm1 in norm2 or norm2 in norm1:
            return True, 'medium'
        p1, p2 = norm1.split(), norm2.split()
        if p1 and p2 and p1[-1] == p2[-1]:
            return True, 'medium'
        return False, 'low'
    
    def _parse_year(self, val: Any) -> Optional[int]:
        if val is None:
            return None
        if isinstance(val, int):
            return val if 1900 <= val <= 2100 else None
        match = re.search(r'\b(19|20)\d{2}\b', str(val))
        return int(match.group(0)) if match else None
    
    def _in_window(self, year: Optional[int], cutoff: datetime) -> bool:
        return year is not None and year >= cutoff.year
    
    def _check_r1(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        pubs_a = a.get('production_5y', {}).get('publications', {}).get('top_items', [])
        pubs_b = b.get('production_5y', {}).get('publications', {}).get('top_items', [])
        evidence = []
        
        for pa in pubs_a:
            ta = self._normalize_name(pa.get('title', ''))
            for pb in pubs_b:
                if ta and ta == self._normalize_name(pb.get('title', '')):
                    evidence.append(f"Shared: {pa.get('title')} ({pa.get('year')})")
        
        name_b = b.get('person', {}).get('name', '')
        for co in a.get('coauthors_5y', []):
            if self._names_match(co.get('name', ''), name_b)[0]:
                evidence.append(f"Coauthor: {co.get('name')} ({co.get('count', 1)}x)")
        
        return (True, 'high', evidence) if evidence else (False, 'low', [])
    
    def _check_r2(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        for src, tgt, src_name in [(a, b, a), (b, a, b)]:
            adv = src.get('production_5y', {}).get('advising', {})
            name = tgt.get('person', {}).get('name', '')
            for advisee in adv.get('ongoing', []) + adv.get('concluded', []):
                match, conf = self._names_match(name, advisee.get('name', ''))
                if match:
                    return True, conf, [f"{src_name.get('person', {}).get('name')} advised {advisee.get('name')}"]
        return False, 'low', []
    
    def _check_r3(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        for aa in a.get('affiliations_5y', []):
            ia = self._normalize_name(aa.get('institution', ''))
            da = self._normalize_name(aa.get('department', ''))
            for ab in b.get('affiliations_5y', []):
                ib = self._normalize_name(ab.get('institution', ''))
                if ia and ia == ib:
                    if da and da == self._normalize_name(ab.get('department', '')):
                        return True, 'high', [f"Same dept: {aa.get('institution')} - {aa.get('department')}"]
                    return True, 'medium', [f"Same inst: {aa.get('institution')}"]
        return False, 'low', []
    
    def _check_r4(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        pa = a.get('production_5y', {}).get('projects', {})
        pb = b.get('production_5y', {}).get('projects', {})
        all_a = pa.get('active', []) + pa.get('concluded', [])
        all_b = pb.get('active', []) + pb.get('concluded', [])
        for p1 in all_a:
            t1 = self._normalize_name(p1.get('title', ''))
            for p2 in all_b:
                if t1 and t1 == self._normalize_name(p2.get('title', '')):
                    return True, 'high', [f"Shared project: {p1.get('title')}"]
        return False, 'low', []
    
    def _check_r5(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        for aa in a.get('production_5y', {}).get('activities', []):
            na = self._normalize_name(aa.get('name', ''))
            for ab in b.get('production_5y', {}).get('activities', []):
                if na and na == self._normalize_name(ab.get('name', '')):
                    return True, 'medium', [f"Shared activity: {aa.get('name')}"]
        return False, 'low', []
    
    def _check_r6(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        _, _, evidence = self._check_r1(a, b, cutoff)
        return (True, 'high', evidence) if len(evidence) >= 3 else (False, 'low', [])
    
    def _check_r7(self, a: Dict, b: Dict, cutoff: datetime) -> Tuple[bool, str, List[str]]:
        for aa in a.get('affiliations_5y', []):
            la = self._normalize_name(aa.get('lab_group', ''))
            for ab in b.get('affiliations_5y', []):
                if la and la == self._normalize_name(ab.get('lab_group', '')):
                    return True, 'high', [f"Same lab: {aa.get('lab_group')}"]
        return False, 'low', []
    
    def _analyze_coi_pairwise(self, data: List[Dict], config: Dict[str, bool], cutoff: datetime) -> List[Dict]:
        pairs = []
        checks = {'R1': self._check_r1, 'R2': self._check_r2, 'R3': self._check_r3, 'R4': self._check_r4, 'R5': self._check_r5, 'R6': self._check_r6, 'R7': self._check_r7}
        rule_descriptions = {
            'R1': 'Co-authorship (shared publication)',
            'R2': 'Advisor-advisee relationship',
            'R3': 'Institutional overlap',
            'R4': 'Project overlap',
            'R5': 'Committee/event overlap',
            'R6': 'Frequent co-authorship (3+ publications)',
            'R7': 'Same lab/research group'
        }
        
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                a, b = data[i], data[j]
                rules_detail = []
                all_evidence = []
                levels = []
                
                for rule, fn in checks.items():
                    if config.get(rule, True):
                        triggered, conf, ev = fn(a, b, cutoff)
                        if triggered:
                            rules_detail.append({
                                'rule': rule,
                                'description': rule_descriptions[rule],
                                'confidence': conf,
                                'evidence': ev
                            })
                            all_evidence.extend(ev)
                            levels.append(conf)
                
                if rules_detail:
                    pairs.append({
                        'a_lattes_id': a.get('person', {}).get('lattes_id'),
                        'b_lattes_id': b.get('person', {}).get('lattes_id'),
                        'a_name': a.get('person', {}).get('name'),
                        'b_name': b.get('person', {}).get('name'),
                        'a_profile_url': a.get('person', {}).get('profile_url'),
                        'b_profile_url': b.get('person', {}).get('profile_url'),
                        'rules_triggered': [r['rule'] for r in rules_detail],
                        'rules_detail': rules_detail,
                        'confidence': 'high' if 'high' in levels else ('medium' if 'medium' in levels else 'low'),
                        'evidence_summary': all_evidence
                    })
        return pairs
    
    def _generate_summary(self, results: Dict) -> str:
        n = results['execution_metadata']['num_researchers']
        w = results['execution_metadata']['time_window_years']
        p = len(results['coi_matrix']['pairs'])
        
        if p == 0:
            return f"Analyzed {n} researchers over {w} years. No COI detected."
        
        h = sum(1 for x in results['coi_matrix']['pairs'] if x['confidence'] == 'high')
        m = sum(1 for x in results['coi_matrix']['pairs'] if x['confidence'] == 'medium')
        l = p - h - m
        return f"Analyzed {n} researchers over {w} years. {p} COI found ({h} high, {m} medium, {l} low)."
    
    def _error_response(self, error_type: str, message: str) -> str:
        return json.dumps({'status': 'error', 'error_type': error_type, 'message': message, 'timestamp': datetime.now().isoformat()}, ensure_ascii=False, indent=2)
    
    def _collect_all_profiles(
        self,
        student: Dict[str, str],
        advisor: Dict[str, str],
        committee_members: List[Dict[str, Any]],
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """
        Browser Tool: Collect all profiles from Lattes platform.
        Returns dict with student_data, advisor_data, and members_data.
        
        Note: The checkbox "Demais pesquisadores" is only needed for student search.
        If student extraction fails, the entire collection is aborted.
        """
        collection_log = []
        total = 2 + len([m for m in committee_members if m.get('lattes_id') != advisor.get('lattes_id') and m.get('role') != 'advisor'])
        current = 0
        
        # Extract student FIRST (requires checkbox "Demais pesquisadores")
        current += 1
        collection_log.append(f"Extracting {current}/{total}: {student.get('name', 'Unknown')} (student - requires checkbox)")
        student_data = self._extract_researcher_profile(
            student.get('name', ''),
            student.get('lattes_id', ''),
            cutoff_date,
            is_student=True  # This enables checkbox verification in the prompt
        )
        
        # Check if student extraction failed - if so, abort the entire collection
        student_warnings = student_data.get('warnings', [])
        student_failed = any(w in student_warnings for w in ['profile_not_found', 'captcha_blocked', 'page_error', 'Extraction failed'])
        
        if student_failed:
            collection_log.append(f"ABORTED: Student extraction failed. Warnings: {student_warnings}")
            return {
                'student_data': student_data,
                'advisor_data': None,
                'members_data': [],
                'collection_log': collection_log,
                'aborted': True,
                'abort_reason': f"Student extraction failed: {student_warnings}"
            }
        
        # Extract advisor (no checkbox needed - established researchers appear in default search)
        current += 1
        collection_log.append(f"Extracting {current}/{total}: {advisor.get('name', 'Unknown')} (advisor)")
        advisor_data = self._extract_researcher_profile(
            advisor.get('name', ''),
            advisor.get('lattes_id', ''),
            cutoff_date,
            is_student=False
        )
        
        # Extract committee members (excluding advisor) - no checkbox needed
        members_data = []
        for member in committee_members:
            member_role = member.get('role', 'unknown')
            if member_role == 'advisor' or member.get('lattes_id') == advisor.get('lattes_id'):
                continue
            
            current += 1
            collection_log.append(f"Extracting {current}/{total}: {member.get('name', 'Unknown')} ({member_role})")
            member_data = self._extract_researcher_profile(
                member.get('name', ''),
                member.get('lattes_id', ''),
                cutoff_date,
                is_student=False
            )
            members_data.append({
                'member_info': member,
                'profile_data': member_data
            })
        
        return {
            'student_data': student_data,
            'advisor_data': advisor_data,
            'members_data': members_data,
            'collection_log': collection_log,
            'aborted': False
        }
    
    def _judge_committee(
        self,
        student_data: Dict[str, Any],
        members_data: List[Dict[str, Any]],
        coi_config: Dict[str, bool],
        cutoff_date: datetime
    ) -> Dict[str, Any]:
        """
        Judge Tool: Analyze COI between student and each committee member.
        No browser operations - pure data analysis.
        """
        members_analysis = []
        conflicts = []
        
        for member_entry in members_data:
            member_info = member_entry['member_info']
            member_profile = member_entry['profile_data']
            member_role = member_info.get('role', 'unknown')
            
            # Analyze COI between student and this member
            coi_result = self._analyze_coi_pair(student_data, member_profile, coi_config, cutoff_date)
            
            member_analysis = {
                'member': {
                    'name': member_info.get('name'),
                    'lattes_id': member_info.get('lattes_id'),
                    'role': member_role,
                    'institution': member_info.get('institution'),
                    'profile_url': member_profile.get('person', {}).get('profile_url')
                },
                'extraction_warnings': member_profile.get('warnings', []),
                'coi_detected': coi_result['has_coi'],
                'coi_details': coi_result['details']
            }
            
            members_analysis.append(member_analysis)
            
            if coi_result['has_coi']:
                conflicts.append({
                    'student_name': student_data.get('person', {}).get('name'),
                    'member_name': member_info.get('name'),
                    'member_role': member_role,
                    'rules_triggered': coi_result['rules_triggered'],
                    'confidence': coi_result['confidence'],
                    'evidence': coi_result['evidence']
                })
        
        return {
            'members_analysis': members_analysis,
            'conflicts': conflicts,
            'has_conflicts': len(conflicts) > 0
        }
    
    def validate_committee(
        self,
        student: Dict[str, str],
        advisor: Dict[str, str],
        committee_members: List[Dict[str, Any]],
        time_window: int = 5,
        coi_rules_config: Dict[str, bool] = None
    ) -> str:
        """
        Validate academic committee for conflicts of interest.
        
        Architecture:
        1. _collect_all_profiles() - Browser Tool: extracts all Lattes profiles
        2. _judge_committee() - Judge Tool: analyzes COI (no browser)
        
        Analyzes COI only between student and non-advisor committee members.
        Advisor-student COI is expected and excluded from analysis.
        Member-member COI is not relevant for committee validation.
        """
        try:
            coi_config = coi_rules_config or {"R1": True, "R2": True, "R3": True, "R4": True, "R5": True, "R6": True, "R7": True}
            cutoff_date = datetime.now() - timedelta(days=time_window * 365)
            
            results = {
                'status': 'valid',
                'execution_metadata': {
                    'execution_date': datetime.now().isoformat(),
                    'time_window_years': time_window,
                    'cutoff_date': cutoff_date.isoformat(),
                    'coi_rules_active': coi_config,
                    'browser_use_available': self.browser_available
                },
                'student': None,
                'advisor': None,
                'members_analysis': [],
                'conflicts': [],
                'collection_log': [],
                'summary': ''
            }
            
            # STEP 1: Browser Tool - Collect all profiles
            collected = self._collect_all_profiles(student, advisor, committee_members, cutoff_date)
            
            results['student'] = collected['student_data']
            results['advisor'] = collected.get('advisor_data')
            results['collection_log'] = collected['collection_log']
            
            # Check if collection was aborted (student extraction failed)
            if collected.get('aborted'):
                results['status'] = 'error'
                results['summary'] = f"Collection aborted: {collected.get('abort_reason', 'Student extraction failed')}"
                return json.dumps(results, ensure_ascii=False, indent=2)
            
            # STEP 2: Judge Tool - Analyze COI (no browser operations)
            judgment = self._judge_committee(
                collected['student_data'],
                collected['members_data'],
                coi_config,
                cutoff_date
            )
            
            results['members_analysis'] = judgment['members_analysis']
            results['conflicts'] = judgment['conflicts']
            
            if judgment['has_conflicts']:
                results['status'] = 'invalid'
            
            # Generate summary
            num_members = len(results['members_analysis'])
            num_conflicts = len(results['conflicts'])
            
            if num_conflicts == 0:
                results['summary'] = f"Committee valid. Analyzed {num_members} members against student. No conflicts detected."
            else:
                conflict_names = [c['member_name'] for c in results['conflicts']]
                results['summary'] = f"Committee INVALID. {num_conflicts} conflict(s) detected with: {', '.join(conflict_names)}."
            
            return json.dumps(results, ensure_ascii=False, indent=2)
            
        except Exception as e:
            return self._error_response('unexpected_error', str(e))
    
    def _analyze_coi_pair(self, a: Dict, b: Dict, config: Dict[str, bool], cutoff: datetime) -> Dict[str, Any]:
        """Analyze COI between two researchers (student vs member)."""
        checks = {
            'R1': self._check_r1,
            'R2': self._check_r2,
            'R3': self._check_r3,
            'R4': self._check_r4,
            'R5': self._check_r5,
            'R6': self._check_r6,
            'R7': self._check_r7
        }
        rule_descriptions = {
            'R1': 'Co-authorship (shared publication)',
            'R2': 'Advisor-advisee relationship',
            'R3': 'Institutional overlap',
            'R4': 'Project overlap',
            'R5': 'Committee/event overlap',
            'R6': 'Frequent co-authorship (3+ publications)',
            'R7': 'Same lab/research group'
        }
        
        details = []
        all_evidence = []
        rules_triggered = []
        levels = []
        
        for rule, fn in checks.items():
            if config.get(rule, True):
                triggered, conf, ev = fn(a, b, cutoff)
                if triggered:
                    rules_triggered.append(rule)
                    details.append({
                        'rule': rule,
                        'description': rule_descriptions[rule],
                        'confidence': conf,
                        'evidence': ev
                    })
                    all_evidence.extend(ev)
                    levels.append(conf)
        
        has_coi = len(rules_triggered) > 0
        confidence = 'high' if 'high' in levels else ('medium' if 'medium' in levels else 'low')
        
        return {
            'has_coi': has_coi,
            'rules_triggered': rules_triggered,
            'confidence': confidence if has_coi else None,
            'evidence': all_evidence,
            'details': details
        }