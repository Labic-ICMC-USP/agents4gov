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

**Result**: PARTIAL PASS
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
        "warnings": ["JSON parse error"],
        "production_5y": {"publications": {"total": 0}}
    }],
    "summary_text": "Analyzed 1 researchers over 5 years. No COI detected."
}
```

**Notes**: Browser automation executed but LLM response was not valid JSON. May need prompt refinement.

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

**Result**: PARTIAL PASS
```json
{
    "status": "success",
    "execution_metadata": {
        "browser_use_available": true,
        "num_researchers": 2,
        "time_window_years": 5
    },
    "researchers": [
        {"name": "Ricardo Marcacini", "warnings": ["JSON parse error"]},
        {"name": "Solange Rezende", "warnings": ["JSON parse error"]}
    ],
    "coi_matrix": {"pairs": []},
    "summary_text": "Analyzed 2 researchers over 5 years. No COI detected."
}
```

## Issues

1. **JSON Parse Error**: LLM response from browser-use not returning valid JSON
   - Cause: Task prompt may need refinement for Lattes page structure
   - Impact: Data extraction returns empty results
   - Workaround: None currently

