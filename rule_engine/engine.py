"""rule_engine/engine.py — HAICD Layer 2: Rule-Based Confidence Classification (DBR≈0)"""
import time, sys, os
sys.path.insert(0, '/home/claude/haicd')
from haicd_core import FeatureVector, RuleEngineOutput, Decision, RoutingPath

class HAICDRuleEngine:
    APPROVE_SCORE = 720; REJECT_SCORE = 640
    APPROVE_DTI = 0.35;  REJECT_DTI  = 0.50

    def __init__(self, delta=10.0, theta=0.70):
        self.delta = delta; self.theta = theta

    def classify(self, fv: FeatureVector) -> RuleEngineOutput:
        if fv.has_behavioral_anomaly:
            return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,"R1_behavioral_anomaly")
        if fv.compliance_flags:
            return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,"R1_compliance_flags")
        if fv.fraud_flag:
            return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,"R1_fraud_flag")

        da = abs(fv.credit_score - self.APPROVE_SCORE)
        dr = abs(fv.credit_score - self.REJECT_SCORE)
        mn = min(da, dr)

        if mn <= self.delta:
            return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,f"R1_borderline(d={mn:.0f})",mn)
        if fv.dti_ratio > 0.55 and fv.credit_score < 780:
            return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,"R1_high_dti",mn)
        if fv.credit_score >= self.APPROVE_SCORE+self.delta and fv.dti_ratio<=self.APPROVE_DTI:
            c = self._conf(fv, Decision.APPROVE)
            return RuleEngineOutput(RoutingPath.FAST_PATH,Decision.APPROVE,c,"R0_clear_approval",da)
        if fv.credit_score < self.REJECT_SCORE-self.delta or fv.dti_ratio>self.REJECT_DTI:
            c = self._conf(fv, Decision.REJECT)
            return RuleEngineOutput(RoutingPath.FAST_PATH,Decision.REJECT,c,"R0_clear_rejection",dr)
        return RuleEngineOutput(RoutingPath.AI_PATH,None,0.0,"R1_borderline_default",mn)

    def _conf(self, fv, dec):
        if dec == Decision.APPROVE:
            s = min(1.0,(fv.credit_score-self.APPROVE_SCORE)/100)
            d = min(1.0,(self.APPROVE_DTI-fv.dti_ratio)/0.15)
        else:
            s = min(1.0,(self.REJECT_SCORE-fv.credit_score)/100)
            d = min(1.0,(fv.dti_ratio-self.REJECT_DTI)/0.15)
        c = 0.40*s + 0.35*d + 0.15
        if fv.employment_status=="Stable": c+=0.10
        return round(min(0.98,max(0.55,c)),3)

class SalesforceFlowBaseline:
    """System A — Salesforce Flow (DBR≈0). Paper Table II: 68.3% acc, FAR=14.2%, F1=0.641"""
    def decide(self, fv: FeatureVector) -> Decision:
        if fv.fraud_flag: return Decision.REJECT
        if fv.credit_score >= 720 and fv.dti_ratio <= 0.35: return Decision.APPROVE
        if fv.credit_score < 640 or fv.dti_ratio > 0.50:   return Decision.REJECT
        return Decision.MANUAL_REVIEW
