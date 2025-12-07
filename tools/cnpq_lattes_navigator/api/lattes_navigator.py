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
    from browser_use import Agent, ChatOpenAI
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
    
    def _extract_researcher_profile(self, name: str, lattes_id: str, cutoff_date: datetime) -> Dict[str, Any]:
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
            extracted_data = self._run_browser_extraction(profile_url, name, lattes_id, cutoff_date)
            
            if extracted_data is None:
                warnings.append("Extraction failed")
                return self._mock_profile(name, lattes_id, profile_url, warnings)
            
            return {
                'person': {
                    'name': name,
                    'lattes_id': lattes_id,
                    'profile_url': profile_url,
                    'last_update': extracted_data.get('last_update')
                },
                'production_5y': self._process_production(extracted_data, cutoff_date),
                'affiliations_5y': extracted_data.get('affiliations', []),
                'coauthors_5y': extracted_data.get('coauthors', []),
                'warnings': warnings + extracted_data.get('warnings', []),
                'evidence_urls': [profile_url]
            }
        except Exception as e:
            warnings.append(f"Error: {str(e)}")
            return self._mock_profile(name, lattes_id, profile_url, warnings)
    
    def _mock_profile(self, name: str, lattes_id: str, profile_url: str, warnings: List[str]) -> Dict[str, Any]:
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
            'evidence_urls': [profile_url]
        }
    
    def _run_browser_extraction(self, profile_url: str, name: str, lattes_id: str, cutoff_date: datetime) -> Optional[Dict[str, Any]]:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                return loop.run_until_complete(self._async_extraction(profile_url, name, lattes_id, cutoff_date))
            finally:
                loop.close()
        except Exception as e:
            return {'warnings': [str(e)], 'publications': [], 'projects': [], 'advising': [], 'affiliations': [], 'coauthors': [], 'last_update': None}
    
    async def _async_extraction(self, profile_url: str, name: str, lattes_id: str, cutoff_date: datetime) -> Dict[str, Any]:
        cutoff_year = cutoff_date.year
        current_year = datetime.now().year
        
        llm = ChatOpenAI(model=self.openai_model)
        
        task = f"""
TASK: Extract academic data from Brazilian Lattes CV for "{name}".

NAVIGATION (try in order):
1. Go to https://buscatextual.cnpq.br/buscatextual/busca.do?metodo=apresentar
2. In the search form, enter name: "{name}"
3. Click search button ("Buscar")
4. Find and click on the researcher matching Lattes ID: {lattes_id}
5. If search fails, try direct URL: {profile_url}

ON PROFILE PAGE:
- Wait for page to load completely
- Look for researcher name "{name}" 
- If page shows "Currículo não encontrado" or error, profile doesn't exist

EXTRACT (only years {cutoff_year}-{current_year}):
- "Artigos completos publicados em periódicos" = journal publications
- "Trabalhos em eventos" = conference papers
- "Projetos de pesquisa" = research projects  
- "Orientações" = supervisions (PhD, Masters, etc)
- Current affiliation (institution, department)

RETURN ONLY THIS JSON:
```json
{{
  "last_update": null,
  "affiliations": [{{"institution": "USP", "department": "ICMC"}}],
  "publications": [{{"title": "Paper Title", "year": 2024, "type": "journal", "venue": "Journal Name"}}],
  "projects": [{{"title": "Project Name", "start_year": 2022, "status": "active"}}],
  "advising": [{{"name": "Student Name", "level": "PhD", "year": 2023}}],
  "coauthors": [{{"name": "Coauthor Name", "count": 2}}],
  "warnings": []
}}
```

ERROR RESPONSES:
- If captcha/blocked: {{"warnings": ["captcha_blocked"], "publications": [], "projects": [], "advising": [], "affiliations": [], "coauthors": [], "last_update": null}}
- If profile not found: {{"warnings": ["profile_not_found"], "publications": [], "projects": [], "advising": [], "affiliations": [], "coauthors": [], "last_update": null}}
- If page error: {{"warnings": ["page_error"], "publications": [], "projects": [], "advising": [], "affiliations": [], "coauthors": [], "last_update": null}}
"""
        
        agent = Agent(task=task, llm=llm)
        
        try:
            history = await agent.run(max_steps=25)
            
            # Extract content from all results in history
            all_content = []
            if hasattr(history, 'all_results'):
                for r in history.all_results:
                    if hasattr(r, 'extracted_content') and r.extracted_content:
                        all_content.append(str(r.extracted_content))
            
            # Also check final_result if available
            if hasattr(history, 'final_result') and history.final_result:
                all_content.append(str(history.final_result))
            
            # Combine all content
            full_text = '\n'.join(all_content)
            
            # Try to find JSON block
            json_block = re.search(r'```json\s*([\s\S]*?)\s*```', full_text)
            if json_block:
                try:
                    return json.loads(json_block.group(1))
                except json.JSONDecodeError:
                    pass
            
            # Try to find raw JSON object
            json_match = re.search(r'\{[^{}]*"warnings"[^{}]*\}', full_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Try any JSON object
            json_match = re.search(r'\{[\s\S]*\}', full_text)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # Return debug info
            return {'warnings': [f'No JSON in response. Content: {full_text[:500]}'], 'publications': [], 'projects': [], 'advising': [], 'affiliations': [], 'coauthors': [], 'last_update': None}
        except Exception as e:
            return {'warnings': [f'Error: {str(e)}'], 'publications': [], 'projects': [], 'advising': [], 'affiliations': [], 'coauthors': [], 'last_update': None}
    
    def _process_production(self, data: Dict[str, Any], cutoff_date: datetime) -> Dict[str, Any]:
        pub_by_type = defaultdict(int)
        filtered_pubs = []
        
        for pub in data.get('publications', []):
            year = self._parse_year(pub.get('year'))
            if self._in_window(year, cutoff_date):
                filtered_pubs.append(pub)
                pub_by_type[pub.get('type', 'other')] += 1
        
        active_proj, concluded_proj = [], []
        for proj in data.get('projects', []):
            if self._in_window(self._parse_year(proj.get('start_year')), cutoff_date):
                (active_proj if proj.get('status') == 'active' else concluded_proj).append(proj)
        
        ongoing_adv, concluded_adv = [], []
        for adv in data.get('advising', []):
            if self._in_window(self._parse_year(adv.get('year')), cutoff_date):
                (ongoing_adv if adv.get('status') == 'ongoing' else concluded_adv).append(adv)
        
        return {
            'publications': {'total': len(filtered_pubs), 'by_type': dict(pub_by_type), 'top_items': filtered_pubs[:10]},
            'projects': {'total': len(active_proj) + len(concluded_proj), 'active': active_proj, 'concluded': concluded_proj},
            'advising': {'total': len(ongoing_adv) + len(concluded_adv), 'ongoing': ongoing_adv, 'concluded': concluded_adv},
            'activities': []
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
        
        for i in range(len(data)):
            for j in range(i + 1, len(data)):
                a, b = data[i], data[j]
                rules, evidence, levels = [], [], []
                
                for rule, fn in checks.items():
                    if config.get(rule, True):
                        triggered, conf, ev = fn(a, b, cutoff)
                        if triggered:
                            rules.append(rule)
                            evidence.extend(ev)
                            levels.append(conf)
                
                if rules:
                    pairs.append({
                        'a_lattes_id': a.get('person', {}).get('lattes_id'),
                        'b_lattes_id': b.get('person', {}).get('lattes_id'),
                        'a_name': a.get('person', {}).get('name'),
                        'b_name': b.get('person', {}).get('name'),
                        'rules_triggered': rules,
                        'confidence': 'high' if 'high' in levels else ('medium' if 'medium' in levels else 'low'),
                        'evidence': evidence
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