# CNPq/Lattes Navigator

Detects Conflicts of Interest (COI) and summarizes academic production from public CNPq/Lattes profiles.

## Structure

```
cnpq_lattes_navigator/
├── api/                 # FastAPI service (Railway deployable)
│   ├── Dockerfile
│   ├── main.py
│   ├── lattes_navigator.py
│   └── requirements.txt
├── tool/                # Open WebUI tool module
│   ├── Dockerfile
│   ├── lattes_navigator.py
│   └── requirements.txt
├── schema.json
└── examples/
```

## Railway Deployment

### API Service

```bash
cd api
# Set environment variable in Railway:
# OPENAI_API_KEY=sk-...

# Railway will auto-detect Dockerfile
```

### Environment Variables

| Variable | Required | Default |
|----------|----------|---------|
| OPENAI_API_KEY | Yes | - |
| OPENAI_MODEL | No | gpt-4o-mini |
| PORT | No | 8000 |

## API Endpoints

### GET /health

```json
{"status": "ok", "browser_available": true, "api_key_set": true}
```

### POST /analyze

Request:
```json
{
  "researchers": [
    {"name": "Ricardo Marcacini", "lattes_id": "4003190744770195"}
  ],
  "time_window": 5,
  "coi_rules": {"R1": true, "R2": true, "R3": true, "R4": true, "R5": true, "R6": true, "R7": true}
}
```

Response:
```json
{
  "status": "success",
  "execution_metadata": {...},
  "researchers": [...],
  "coi_matrix": {"pairs": [...]},
  "summary_text": "..."
}
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
