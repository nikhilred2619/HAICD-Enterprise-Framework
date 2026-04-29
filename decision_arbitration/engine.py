"""decision_arbitration/engine.py — HAICD Layer 4: Decision Arbitration + Audit Trail"""
import time, sys
sys.path.insert(0, '/home/claude/haicd')
from haicd_core import (FeatureVector, HAICDDecisionRecord, Decision,
                         RoutingPath, ComplianceStatus)
from rule_engine.engine import HAICDRuleEngine, SalesforceFlowBaseline
from ai_reasoning.engine import HAICDAIReasoningEngine

class HAICDArbitrationEngine:
    THETA = 0.60

    def __init__(self, delta=10.0, mock_mode=True):
        self.rule_engine = HAICDRuleEngine(delta=delta)
        self.ai_engine   = HAICDAIReasoningEngine(mock_mode=mock_mode)

    def process(self, fv: FeatureVector, ground_truth=None) -> HAICDDecisionRecord:
        t0 = time.time()
        rec = HAICDDecisionRecord(application_id=fv.application_id)
        rec.ground_truth = ground_truth
        rec.log_event("L1","INPUT_NORMALIZED",f"Score={fv.credit_score:.0f},DTI={fv.dti_ratio:.1%}")

        t2 = time.time()
        rule_out = self.rule_engine.classify(fv)
        rec.layer2_time_ms = (time.time()-t2)*1000
        rec.rule_output = rule_out
        rec.log_event("L2","RULE_CLASSIFICATION",f"Routing={rule_out.routing.value}")

        if rule_out.routing == RoutingPath.FAST_PATH:
            rec.routing_path    = RoutingPath.FAST_PATH
            rec.final_decision  = rule_out.fast_path_decision
            rec.confidence_score = rule_out.confidence
            rec.reasoning_trace = (f"Fast-path R=0 (DBR≈0). Rule:{rule_out.rule_triggered}. "
                                   f"Δ={rule_out.threshold_delta:.0f}pts. C={rule_out.confidence:.2f}.")
            rec.log_event("L4","FAST_PATH",f"Decision={rec.final_decision.value}")
        else:
            t3 = time.time()
            ai_out = self.ai_engine.reason(fv)
            rec.layer3_time_ms = (time.time()-t3)*1000
            rec.ai_output = ai_out
            rec.log_event("L3","AI_REASONING",
                          f"Decision={ai_out.decision.value},C={ai_out.confidence:.2f},DBR≈{ai_out.dbr_estimate:.2f}")

            if ai_out.confidence < self.THETA:
                rec.final_decision         = Decision.MANUAL_REVIEW
                rec.manual_review_triggered = True
                rec.confidence_gate_applied = True
                rec.log_event("L4","CONFIDENCE_GATE",f"C={ai_out.confidence:.2f}<θ={self.THETA}→Review")
            else:
                rec.final_decision = ai_out.decision

            rec.routing_path     = RoutingPath.AI_PATH
            rec.confidence_score = ai_out.confidence
            rec.reasoning_trace  = ai_out.reasoning_trace

        self._compliance(fv, rec)

        if rec.final_decision == Decision.REJECT:
            reasons = []
            if fv.credit_score < 660: reasons.append(f"Score {fv.credit_score:.0f} below minimum")
            if fv.dti_ratio > 0.45:   reasons.append(f"DTI {fv.dti_ratio:.0%} exceeds maximum")
            if fv.fraud_flag:          reasons.append("Fraud indicators present")
            if fv.transaction_behavior=="Suspicious": reasons.append("Suspicious transactions")
            rec.adverse_action_reason = "; ".join(reasons) or "Does not meet lending criteria"
            rec.log_event("L4","ADVERSE_ACTION",rec.adverse_action_reason)

        if ground_truth:
            rec.is_correct = (rec.final_decision == ground_truth)

        rec.processing_time_ms = (time.time()-t0)*1000
        rec.log_event("L4","COMPLETE",f"Final={rec.final_decision.value},{rec.processing_time_ms:.1f}ms")
        return rec

    def _compliance(self, fv, rec):
        if fv.compliance_flags:
            rec.compliance_status = ComplianceStatus.FLAGGED
            if rec.final_decision == Decision.APPROVE:
                rec.final_decision = Decision.MANUAL_REVIEW
                rec.manual_review_triggered = True
                rec.compliance_status = ComplianceStatus.ESCALATED
                rec.log_event("L4","COMPLIANCE_OVERRIDE",str(fv.compliance_flags))

class HAICDPipeline:
    """Complete HAICD five-layer pipeline — main production entry point."""
    def __init__(self, delta=10.0, mock_mode=True):
        self.arbitration   = HAICDArbitrationEngine(delta=delta, mock_mode=mock_mode)
        self.flow_baseline = SalesforceFlowBaseline()

    def decide(self, fv: FeatureVector, ground_truth=None) -> HAICDDecisionRecord:
        return self.arbitration.process(fv, ground_truth)

    def decide_flow(self, fv: FeatureVector) -> Decision:
        return self.flow_baseline.decide(fv)
