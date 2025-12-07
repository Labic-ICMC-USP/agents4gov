import os
import sys
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
import json

app = FastAPI(title="CNPq/Lattes Navigator API", version="1.0.0")

# Capture import error for diagnostics
browser_import_error = None
try:
    from lattes_navigator import Tools, BROWSER_USE_AVAILABLE, BROWSER_IMPORT_ERROR
    tool = Tools()
    browser_import_error = BROWSER_IMPORT_ERROR
except Exception as e:
    browser_import_error = str(e)
    BROWSER_USE_AVAILABLE = False
    tool = None


class Researcher(BaseModel):
    name: str
    lattes_id: str


class AnalysisRequest(BaseModel):
    researchers: List[Researcher]
    time_window: int = 5
    coi_rules: Optional[dict] = None


class HealthResponse(BaseModel):
    status: str
    browser_available: bool
    api_key_set: bool
    import_error: Optional[str] = None
    python_version: str


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok" if tool else "error",
        browser_available=BROWSER_USE_AVAILABLE if tool else False,
        api_key_set=bool(os.getenv("OPENAI_API_KEY")),
        import_error=browser_import_error,
        python_version=sys.version
    )


@app.get("/debug")
def debug():
    errors = []
    
    # Test browser-use import
    try:
        from browser_use import Agent
        errors.append({"browser_use.Agent": "OK"})
    except Exception as e:
        errors.append({"browser_use.Agent": str(e)})
    
    try:
        from browser_use import Browser, BrowserConfig
        errors.append({"browser_use.Browser": "OK"})
    except Exception as e:
        errors.append({"browser_use.Browser": str(e)})
    
    # Test langchain import
    try:
        from langchain_openai import ChatOpenAI
        errors.append({"langchain_openai": "OK"})
    except Exception as e:
        errors.append({"langchain_openai": str(e)})
    
    # Test playwright
    try:
        import playwright
        errors.append({"playwright": "OK", "version": playwright.__version__})
    except Exception as e:
        errors.append({"playwright": str(e)})
    
    return {"imports": errors, "python": sys.version}


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    if not tool:
        return {"status": "error", "message": "Tool not initialized", "import_error": browser_import_error}
    
    researchers_json = json.dumps([r.model_dump() for r in request.researchers])
    coi_config = json.dumps(request.coi_rules or {"R1": True, "R2": True, "R3": True, "R4": True, "R5": True, "R6": True, "R7": True})
    
    result = tool.analyze_researchers_coi(
        researchers_json=researchers_json,
        time_window=request.time_window,
        coi_rules_config=coi_config
    )
    
    return json.loads(result)


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
