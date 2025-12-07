import os
from typing import List, Optional
from fastapi import FastAPI
from pydantic import BaseModel
from lattes_navigator import Tools
import json

app = FastAPI(title="CNPq/Lattes Navigator API", version="1.0.0")
tool = Tools()


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


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        browser_available=tool.browser_available,
        api_key_set=bool(tool.openai_api_key)
    )


@app.post("/analyze")
def analyze(request: AnalysisRequest):
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
