# CNPq/Lattes Navigator

Detects Conflicts of Interest (COI) and summarizes academic production from public CNPq/Lattes profiles using browser automation.

## Structure

```
cnpq_lattes_navigator/
├── api/                 # FastAPI service
│   ├── Dockerfile
│   ├── main.py
│   ├── lattes_navigator.py
│   └── requirements.txt
├── tool/                # Open WebUI tool module
│   ├── Dockerfile
│   ├── lattes_navigator.py
│   └── requirements.txt
├── schema.json
├── TESTING.md
└── examples/
```

## Railway Deployment

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| OPENAI_API_KEY | Yes | - | OpenAI API key for LLM |
| OPENAI_MODEL | No | gpt-4o-mini | Model to use |
| PORT | No | 8000 | Server port (auto-set by Railway) |
| BROWSER_USE_API_KEY | Yes | - | Browser-Use Cloud API key (get from cloud.browser-use.com) |
| BROWSER_USE_CLOUD | No | true | Use cloud browser for stealth mode |

### Deploy

Point Railway to `tools/cnpq_lattes_navigator/api/` directory.

## API Endpoints

### GET /health

Health check with system status.

```bash
curl https://lattes-navigator-api-production.up.railway.app/health
```

Response:
```json
{
  "status": "ok",
  "browser_available": true,
  "api_key_set": true,
  "import_error": null
}
```

### GET /debug

Import diagnostics.

```bash
curl https://lattes-navigator-api-production.up.railway.app/debug
```

### POST /analyze

Analyze researchers for COI (pairwise analysis).

```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "researchers": [
      {"name": "Ricardo Marcacini", "lattes_id": "3272611282260295"},
      {"name": "Matheus Yasuo", "lattes_id": "6191612710855387"}
    ],
    "time_window": 5,
    "coi_rules": {"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}
  }'
```

### POST /validate-committee

Validate academic committee for conflicts of interest. Analyzes COI only between student and non-advisor committee members.

**Request Body:**
```json
{
  "student": {
    "name": "Matheus Yasuo Ribeiro Utino",
    "lattes_id": "6191612710855387"
  },
  "advisor": {
    "name": "Ricardo Marcondes Marcacini",
    "lattes_id": "3272611282260295"
  },
  "committee_members": [
    {
      "name": "Solange Oliveira Rezende",
      "lattes_id": "8526960535874806",
      "email": "solange@icmc.usp.br",
      "institution": "ICMC-USP",
      "role": "internal",
      "is_president": false
    },
    {
      "name": "Paulo Roberto Mann Marques Júnior",
      "lattes_id": "3571577377652346",
      "email": "paulomann@ufrj.br",
      "institution": "UFRJ",
      "role": "external",
      "is_president": false
    }
  ],
  "thesis_title": "Unstructured Text Mining in the Era of Large Language Models",
  "committee_type": "qualification",
  "time_window": 5
}
```

**Test Valid Committee:**
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/validate-committee \
  -H "Content-Type: application/json" \
  -d @tools/cnpq_lattes_navigator/examples/valid_committee.json
```

**Test Invalid Committee (with COI):**
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/validate-committee \
  -H "Content-Type: application/json" \
  -d @tools/cnpq_lattes_navigator/examples/invalid_committee.json
```

**Response (Valid Committee):**
```json
{
  "status": "valid",
  "student": {...},
  "advisor": {...},
  "members_analysis": [
    {
      "member": {...},
      "coi_detected": false,
      "coi_details": []
    }
  ],
  "conflicts": [],
  "collection_log": [
    "Extracting 1/5: Matheus Yasuo Ribeiro Utino (student)",
    "Extracting 2/5: Ricardo Marcondes Marcacini (advisor)",
    ...
  ],
  "summary": "Committee valid. Analyzed 4 members against student. No conflicts detected."
}
```

**Response (Invalid Committee):**
```json
{
  "status": "invalid",
  "conflicts": [
    {
      "student_name": "Matheus Yasuo Ribeiro Utino",
      "member_name": "Paulo Roberto Mann Marques Júnior",
      "member_role": "external",
      "rules_triggered": ["R1"],
      "confidence": "high",
      "evidence": ["Shared: Paper Title (2024)"]
    }
  ],
  "summary": "Committee INVALID. 1 conflict(s) detected with: Paulo Roberto Mann Marques Júnior."
}
```

## Test Procedures

### 1. Verify Deployment

```bash
# Health check
curl https://lattes-navigator-api-production.up.railway.app/health

# Expected: browser_available: true, api_key_set: true
```

### 2. Check Imports

```bash
# Debug imports
curl https://lattes-navigator-api-production.up.railway.app/debug

# Expected: browser_use.Agent: OK, browser_use.ChatOpenAI: OK
```

### 3. Single Researcher Test

```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"researchers": [{"name": "Test Name", "lattes_id": "0000000000000000"}], "time_window": 5}'
```

### 4. COI Detection Test

```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{"researchers": [{"name": "Researcher A", "lattes_id": "ID_A"}, {"name": "Researcher B", "lattes_id": "ID_B"}], "time_window": 5}'
```

### 5. Committee Validation Test

**Test Valid Committee (no conflicts expected):**
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/validate-committee \
  -H "Content-Type: application/json" \
  -d @tools/cnpq_lattes_navigator/examples/valid_committee.json
```

**Test Invalid Committee (conflict expected with Paulo Mann):**
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/validate-committee \
  -H "Content-Type: application/json" \
  -d @tools/cnpq_lattes_navigator/examples/invalid_committee.json
```

**Expected Results:**
- Valid committee: `"status": "valid"`, `"conflicts": []`
- Invalid committee: `"status": "invalid"`, conflicts with Paulo Roberto Mann Marques Júnior

## COI Rules

| Rule | Description |
|------|-------------|
| R1 | Co-authorship (1+ shared publication) |
| R2 | Advisor-advisee relationship |
| R3 | Institutional overlap |
| R4 | Project overlap |
| R5 | Committee/event overlap |
| R6 | Frequent co-authorship (3+ publications) |
| R7 | Same lab/group |

## Open WebUI Integration

Copy `tool/lattes_navigator.py` content to Open WebUI Tools interface.

## Test Results

See [TESTING.md](TESTING.md) for detailed test documentation and results.
