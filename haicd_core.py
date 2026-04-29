"""
haicd_core.py — HAICD Framework Core Data Structures
HAICD: Hybrid AI-CRM Decisioning Framework
DBR = Jt/N (Decision Boundary Rigidity)
Paper: Donapati, N.R. (2025). IEEE Access (Under Review).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional
import time, uuid, numpy as np

class Decision(str, Enum):
    APPROVE = "Approve"
    REJECT  = "Reject"
    MANUAL_REVIEW = "Review"

class RoutingPath(str, Enum):
    FAST_PATH = "R0_FastPath"
    AI_PATH   = "R1_AIPath"

class ComplianceStatus(str, Enum):
    COMPLIANT  = "Compliant"
    FLAGGED    = "Flagged"
    ESCALATED  = "Escalated"

@dataclass
class FeatureVector:
    """F = {score, DTI, tenure, behavior, assets, employment, fraud_flags}"""
    credit_score:        float
    dti_ratio:           float
    tenure_years:        float
    asset_value:         float
    employment_status:   str
    fraud_flag:          bool
    transaction_behavior: str
    behavioral_signals:  List[str]
    compliance_flags:    List[str]
    application_id:      str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    domain:              str = "Loan"
    score_available:     bool = True
    tenure_available:    bool = True
    assets_available:    bool = True
    employment_available: bool = True

    @property
    def is_borderline_score(self):
        return any(abs(self.credit_score - t) <= 10 for t in [660, 720])

    @property
    def has_behavioral_anomaly(self):
        return self.transaction_behavior == "Suspicious" or bool(self.behavioral_signals)

    @property
    def has_compensatory_factors(self):
        return self.tenure_years >= 10 or self.asset_value >= 200000

    @property
    def dbr_routing_needed(self):
        return self.is_borderline_score or self.has_behavioral_anomaly

    @classmethod
    def from_german_credit(cls, row: dict) -> "FeatureVector":
        checking   = row.get('checking_account', 1)
        cr_history = row.get('credit_history', 3)
        score_raw  = (checking * 60) + ((5 - cr_history) * 30) + 560
        score      = min(850, max(400, float(score_raw)))
        install_rate = row.get('installment_rate', 2)
        dti_map = {1: 0.12, 2: 0.22, 3: 0.33, 4: 0.45}
        dti = dti_map.get(install_rate, 0.28)
        employ = row.get('employment_duration', 2)
        tenure_map = {1: 0, 2: 2, 3: 5, 4: 8, 5: 12}
        tenure = float(tenure_map.get(employ, 2))
        savings  = row.get('savings_balance', 1)
        property_ = row.get('property', 1)
        asset_val = float((savings * 15000) + (property_ * 25000))
        employ_status = "Stable" if employ >= 3 else "Unstable"
        other_debtors  = row.get('other_debtors', 1)
        other_installs = row.get('other_installments', 3)
        behavior = "Suspicious" if other_debtors == 2 or other_installs != 3 else "Normal"
        return cls(
            credit_score=score, dti_ratio=dti, tenure_years=tenure,
            asset_value=asset_val, employment_status=employ_status,
            fraud_flag=False, transaction_behavior=behavior,
            behavioral_signals=[], compliance_flags=[],
            application_id=str(row.get('id', uuid.uuid4()))[:8],
        )

@dataclass
class RuleEngineOutput:
    routing:             RoutingPath
    fast_path_decision:  Optional[Decision]
    confidence:          float
    rule_triggered:      str
    threshold_delta:     float = 0.0

@dataclass
class AIReasoningOutput:
    decision:             Decision
    confidence:           float
    reasoning_trace:      str
    attributes_used:      List[str]
    compensation_detected: bool
    contradiction_detected: bool
    dbr_estimate:         float

@dataclass
class HAICDDecisionRecord:
    decision_id:          str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    application_id:       str = ""
    timestamp:            float = field(default_factory=time.time)
    final_decision:       Decision = Decision.MANUAL_REVIEW
    confidence_score:     float = 0.0
    routing_path:         RoutingPath = RoutingPath.AI_PATH
    rule_output:          Optional[RuleEngineOutput] = None
    ai_output:            Optional[AIReasoningOutput] = None
    confidence_gate_applied: bool = False
    manual_review_triggered: bool = False
    reasoning_trace:      str = ""
    audit_log:            List[Dict[str, Any]] = field(default_factory=list)
    compliance_status:    ComplianceStatus = ComplianceStatus.COMPLIANT
    adverse_action_reason: str = ""
    processing_time_ms:   float = 0.0
    layer2_time_ms:       float = 0.0
    layer3_time_ms:       float = 0.0
    ground_truth:         Optional[Decision] = None
    is_correct:           Optional[bool] = None

    def log_event(self, layer, event, detail=""):
        self.audit_log.append({"timestamp":time.time(),"layer":layer,"event":event,"detail":detail})

    def to_dict(self):
        return {
            "decision_id": self.decision_id,
            "application_id": self.application_id,
            "final_decision": self.final_decision.value,
            "confidence_score": round(self.confidence_score, 3),
            "routing_path": self.routing_path.value,
            "reasoning_trace": self.reasoning_trace,
            "compliance_status": self.compliance_status.value,
            "adverse_action_reason": self.adverse_action_reason,
            "processing_time_ms": round(self.processing_time_ms, 2),
            "manual_review_triggered": self.manual_review_triggered,
            "audit_events": len(self.audit_log),
        }
