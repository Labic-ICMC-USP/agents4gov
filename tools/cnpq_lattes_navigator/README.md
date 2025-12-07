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

| Variable | Required | Default |
|----------|----------|---------|
| OPENAI_API_KEY | Yes | - |
| OPENAI_MODEL | No | gpt-4o-mini |
| PORT | No | 8000 (auto-set by Railway) |

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

Analyze researchers for COI.

```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "researchers": [
      {"name": "Ricardo Marcacini", "lattes_id": "4003190744770195"},
      {"name": "Solange Rezende", "lattes_id": "1458324546544936"}
    ],
    "time_window": 5,
    "coi_rules": {"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}
  }'
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
