"""tests/test_haicd.py — HAICD Framework test suite"""
import sys; sys.path.insert(0,'/home/claude/haicd')
from haicd_core import FeatureVector, Decision, RoutingPath
from rule_engine.engine import HAICDRuleEngine, SalesforceFlowBaseline
from ai_reasoning.engine import HAICDAIReasoningEngine
from decision_arbitration.engine import HAICDPipeline

def make_fv(**kwargs):
    defaults = dict(credit_score=700,dti_ratio=0.30,tenure_years=5,asset_value=100000,
                    employment_status="Stable",fraud_flag=False,transaction_behavior="Normal",
                    behavioral_signals=[],compliance_flags=[],application_id="TEST")
    defaults.update(kwargs)
    return FeatureVector(**defaults)

tests_pass = 0; tests_total = 0

def test(name, cond):
    global tests_pass, tests_total
    tests_total += 1
    if cond: tests_pass += 1; print(f"  ✅ {name}")
    else: print(f"  ❌ {name}")

# Layer 2 tests
re = HAICDRuleEngine(delta=10)
test("Clear approval → fast path", re.classify(make_fv(credit_score=750,dti_ratio=0.28)).routing==RoutingPath.FAST_PATH)
test("Borderline score → AI path", re.classify(make_fv(credit_score=718)).routing==RoutingPath.AI_PATH)
test("Behavioral anomaly → AI path", re.classify(make_fv(transaction_behavior="Suspicious")).routing==RoutingPath.AI_PATH)
test("Fraud flag → AI path", re.classify(make_fv(fraud_flag=True)).routing==RoutingPath.AI_PATH)
test("High DTI → AI path", re.classify(make_fv(credit_score=800,dti_ratio=0.65)).routing==RoutingPath.AI_PATH)

# Layer 3 tests
ai = HAICDAIReasoningEngine(mock_mode=True)
# CS-041: borderline + long tenure → compensatory → approve
r041 = ai.reason(make_fv(credit_score=718,dti_ratio=0.25,tenure_years=15,asset_value=280000))
test("Compensation: long tenure offsets borderline score", r041.compensation_detected)
test("CS-041 decision: approve", r041.decision==Decision.APPROVE)

# CS-082: high score + extreme DTI → contradiction → review
r082 = ai.reason(make_fv(credit_score=810,dti_ratio=0.65))
test("Contradiction: high score + extreme DTI detected", r082.contradiction_detected)

# CS-095: formal fraud=No + suspicious behavior → reject
r095 = ai.reason(make_fv(credit_score=740,dti_ratio=0.28,transaction_behavior="Suspicious",
                          behavioral_signals=["50+ intl transfers/48h"]))
test("Contradiction: behavioral fraud overrides formal flag", r095.contradiction_detected)
test("CS-095: reject on behavioral evidence", r095.decision==Decision.REJECT)

# Layer 3: DBR estimate
test("DBR estimate > 0.5", r041.dbr_estimate > 0.5)
test("Multiple attributes evaluated", len(r041.attributes_used) >= 4)

# Full pipeline
pipe = HAICDPipeline(delta=10,mock_mode=True)
rec1 = pipe.decide(make_fv(credit_score=750,dti_ratio=0.28))
test("Clear approval: has decision", rec1.final_decision != None)
test("Audit log populated", len(rec1.audit_log) > 0)
test("Processing time tracked", rec1.processing_time_ms > 0)
test("Reasoning trace present", len(rec1.reasoning_trace) > 10)
test("to_dict works", "final_decision" in rec1.to_dict())

# Compliance
rec_comp = pipe.decide(make_fv(compliance_flags=["HIPAA","GDPR"]))
test("Compliance flags → manual review", rec_comp.final_decision==Decision.MANUAL_REVIEW)
test("Compliance flagged", rec_comp.compliance_status.value in ["Flagged","Escalated"])

# Adverse action
rec_rej = pipe.decide(make_fv(credit_score=580,dti_ratio=0.55))
test("Rejection has adverse action reason", rec_rej.final_decision==Decision.REJECT and len(rec_rej.adverse_action_reason)>5)

# Salesforce integration
from crm_integration.salesforce.connector import SalesforceHAICDConnector
conn = SalesforceHAICDConnector()
sf_result = conn.full_flow(rec1, "LOAN_001")
test("Salesforce integration success", sf_result["success"])
test("Platform event published", sf_result["steps"]["1_event"]["success"])
test("Agentforce triggered", sf_result["steps"]["3_agentforce"]["success"])

# Flow baseline
flow = SalesforceFlowBaseline()
test("Flow: clear approval", flow.decide(make_fv(credit_score=750,dti_ratio=0.28))==Decision.APPROVE)
test("Flow: clear rejection", flow.decide(make_fv(credit_score=580))==Decision.REJECT)
test("Flow misses behavioral fraud (DBR=0)", 
     flow.decide(make_fv(credit_score=740,transaction_behavior="Suspicious"))==Decision.APPROVE)

print(f"\n{'='*50}")
print(f"Tests: {tests_pass}/{tests_total} passed")

if __name__ == "__main__":
    pass
