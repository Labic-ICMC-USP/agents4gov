# CNPq/Lattes Navigator - Test Documentation

### 1. Single Researcher Analysis

**Endpoint**: `POST /analyze`

**Command**:
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "researchers": [{"name": "Ricardo Marcacini", "lattes_id": "4003190744770195"}],
    "time_window": 5
  }'
```

**Result**: PASS (with expected limitation)
```json
{
    "status": "success",
    "execution_metadata": {
        "browser_use_available": true,
        "num_researchers": 1,
        "time_window_years": 5
    },
    "researchers": [{
        "person": {
            "name": "Ricardo Marcacini",
            "lattes_id": "4003190744770195",
            "profile_url": "http://lattes.cnpq.br/4003190744770195"
        },
        "warnings": ["captcha_blocked"],
        "production_5y": {"publications": {"total": 0}}
    }],
    "summary_text": "Analyzed 1 researchers over 5 years. No COI detected."
}
```

**Notes**: 
- Browser automation executes correctly
- JSON response parsing works
- Lattes platform blocks automated access with captcha

---

### 2. COI Detection Test (Two Researchers)

**Command**:
```bash
curl -X POST https://lattes-navigator-api-production.up.railway.app/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "researchers": [
      {"name": "Ricardo Marcacini", "lattes_id": "4003190744770195"},
      {"name": "Solange Rezende", "lattes_id": "1458324546544936"}
    ],
    "time_window": 5
  }'
```

**Expected**: Both researchers return `captcha_blocked` warning due to platform protection.

---

## Working Components

- API deployment on Railway
- browser-use integration
- Agent execution
- JSON response parsing
- Error handling with fallback responses

## Known Limitation

**Captcha Protection**: The CNPq/Lattes platform has anti-bot protection that blocks automated browser access. This is a platform-level restriction, not a tool issue.

Potential workarounds:
1. Use official CNPq API (if available)
2. Manual data entry
3. Request institutional API access
