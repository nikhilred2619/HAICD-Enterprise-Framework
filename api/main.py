"""
api/main.py — HAICD Framework REST API (FastAPI)
Run: uvicorn api.main:app --reload --port 8000
Docs: http://localhost:8000/docs
"""
import sys, time
sys.path.insert(0, '/home/claude/haicd')
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import uvicorn
from haicd_core import FeatureVector, Decision
from decision_arbitration.engine import HAICDPipeline

app = FastAPI(
    title="HAICD Framework — Hybrid AI-CRM Decisioning API",
    description="""
## HAICD: Hybrid AI-CRM Decisioning Framework

**Validated Performance (Paper Results)**:
- Synthetic benchmark (N=120): **90.8% accuracy**, F1=0.908, FAR=2.5%
- German Credit (N=1,000): **−50% FRR** vs rule-based, McNemar χ²=27.44, p<0.001
- LLM stability: **98.9% consistency** across 3 runs
- DBR: Adaptive (≈0 fast path, ≈0.86 AI path)

**Decision Boundary Rigidity (DBR)**: Novel metric formalizing when AI augmentation 
improves enterprise decisions — DBR = Jₜ/N where Jₜ = jointly evaluated attributes.

**Reference**: Donapati, N.R. (2025). HAICD: A Hybrid AI-CRM Decisioning Framework. 
IEEE Access (Under Review).
    """,
    version="1.0.0",
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

pipeline = HAICDPipeline(delta=10.0, mock_mode=True)
_start = time.time(); _count = 0

class LoanRequest(BaseModel):
    customer_id:         str  = Field("CUST-001", description="Customer/application ID")
    credit_score:        float = Field(680.0, ge=300, le=850, description="Credit score (FICO-style)")
    dti_ratio:           float = Field(0.35, ge=0, le=1, description="Debt-to-income ratio [0-1]")
    tenure_years:        float = Field(5.0, ge=0, description="Employment/customer tenure in years")
    asset_value:         float = Field(150000.0, ge=0, description="Total asset value USD")
    employment_status:   str  = Field("Stable", description="Stable / Unstable / Unknown")
    fraud_flag:          bool = Field(False, description="Formal fraud indicator")
    transaction_behavior: str = Field("Normal", description="Normal / Suspicious")
    behavioral_signals:  List[str] = Field([], description="Additional risk signals")
    compliance_flags:    List[str] = Field([], description="ECOA / GDPR / SOX flags")
    domain:              str = Field("Loan", description="Loan / Mortgage / Insurance / CRM")

    class Config:
        json_schema_extra = {"example": {
            "customer_id": "CS-041", "credit_score": 718.0, "dti_ratio": 0.25,
            "tenure_years": 15.0, "asset_value": 280000.0, "employment_status": "Stable",
            "fraud_flag": False, "transaction_behavior": "Normal",
            "behavioral_signals": [], "compliance_flags": [], "domain": "Loan"
        }}

class HAICDResponse(BaseModel):
    application_id:       str
    final_decision:       str
    confidence_score:     float
    routing_path:         str
    reasoning_trace:      str
    compliance_status:    str
    adverse_action_reason: str
    processing_time_ms:   float
    manual_review_triggered: bool
    audit_events:         int
    dbr_characterization: Dict[str, Any]

@app.get("/health")
async def health():
    return {"status":"healthy","version":"1.0.0","framework":"HAICD","uptime_s":round(time.time()-_start,1),"requests":_count}

@app.post("/loan/decide", response_model=HAICDResponse, tags=["Decisions"])
async def decide_loan(req: LoanRequest):
    """
    Execute HAICD five-layer loan decision pipeline.
    
    **Layer Architecture:**
    - L1: Input normalization
    - L2: Rule-based confidence classification (DBR≈0, fast path R=0)
    - L3: AI contextual reasoning (DBR≈0.86, AI path R=1)  
    - L4: Decision arbitration + ECOA audit trail
    - L5: Feedback (future deployment)
    """
    global _count; _count += 1
    try:
        fv = FeatureVector(
            credit_score=req.credit_score, dti_ratio=req.dti_ratio,
            tenure_years=req.tenure_years, asset_value=req.asset_value,
            employment_status=req.employment_status, fraud_flag=req.fraud_flag,
            transaction_behavior=req.transaction_behavior,
            behavioral_signals=req.behavioral_signals,
            compliance_flags=req.compliance_flags,
            application_id=req.customer_id, domain=req.domain,
        )
        rec = pipeline.decide(fv)
        d = rec.to_dict()
        d["dbr_characterization"] = {
            "routing_path": rec.routing_path.value,
            "dbr_value": "≈0 (fast path)" if rec.routing_path.value=="R0_FastPath" else "≈0.86 (AI joint inference)",
            "attributes_jointly_evaluated": (rec.ai_output.attributes_used if rec.ai_output else ["score","dti"]),
            "compensation_detected": (rec.ai_output.compensation_detected if rec.ai_output else False),
            "contradiction_detected": (rec.ai_output.contradiction_detected if rec.ai_output else False),
        }
        return HAICDResponse(**d)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/framework/metrics", tags=["Framework"])
async def metrics():
    return {
        "synthetic_benchmark": {"n":120,"accuracy":"90.8%","f1":"0.908","far":"2.5%","frr":"6.7%","mcnemar_chi2":19.31,"p_value":"<0.001"},
        "german_credit":       {"n":1000,"frr_reduction":"−50%","f1_improvement":"0.377→0.528","mcnemar_chi2":27.44,"statistical_power":"100%"},
        "llm_stability":       {"cross_run_consistency":"98.9%","runs":3,"temperature":0.0},
        "dbr":                 {"rule_based":"≈0","llm_only":"≈0.86","haicd":"Adaptive (R=0:≈0, R=1:≈0.86)"},
        "latency":             {"fast_path_s":0.35,"ai_path_s":1.92,"hybrid_effective_s":0.87},
        "cost_per_100k_day":   "$140 (hybrid) vs $420 (AI-only)",
    }

@app.get("/framework/dbr", tags=["Framework"])
async def dbr():
    return {
        "definition": "DBR = Jₜ/N — Decision Boundary Rigidity",
        "systems": [
            {"name":"Rule-Based (Salesforce Flow)","dbr":"≈0","description":"Independent Boolean predicates"},
            {"name":"LLM (Agentforce)","dbr":"≈0.86","description":"Self-attention joint inference; 6/7=0.857"},
            {"name":"HAICD (Proposed)","dbr":"Adaptive","description":"R=0→≈0, R=1→≈0.86"},
        ]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
