"""
crm_integration/salesforce/connector.py
─────────────────────────────────────────
Salesforce CRM + Agentforce Integration for HAICD Framework.

Integration chain:
  HAICD Decision → Salesforce Platform Event → Agentforce Topic → Case/Loan Resolution

Maps HAICD routing decisions to Salesforce operations:
  APPROVE       → Auto-approve Loan_Application__c, update Stage
  REJECT        → Generate adverse action notice (ECOA compliance)
  MANUAL_REVIEW → Assign to underwriter queue + trigger Agentforce review agent
"""
import sys; sys.path.insert(0,'/home/claude/haicd')
from dataclasses import dataclass
from typing import Any, Dict, Optional
from haicd_core import HAICDDecisionRecord, Decision

@dataclass
class SalesforceConfig:
    instance_url: str; access_token: str; api_version: str = "v60.0"

class SalesforceHAICDConnector:
    QUEUE_MAP = {
        Decision.APPROVE:       "SF_AutoApproval_Queue",
        Decision.REJECT:        "SF_AdverseAction_Queue",
        Decision.MANUAL_REVIEW: "SF_Underwriter_Queue",
    }
    AGENT_TOPIC_MAP = {
        Decision.APPROVE:       "Loan_AutoApproval_Agent",
        Decision.REJECT:        "Loan_AdverseAction_Agent",
        Decision.MANUAL_REVIEW: "Loan_Underwriter_Review_Agent",
    }

    def __init__(self, config: Optional[SalesforceConfig] = None):
        self.config = config; self._mock = config is None
        if self._mock: print("[SalesforceHAICDConnector] Mock mode — no live SF connection")

    def publish_decision_event(self, rec: HAICDDecisionRecord, app_id: str) -> Dict[str,Any]:
        """Publish HAICD decision as Salesforce Platform Event."""
        payload = {
            "HAICD_Decision_ID__c": rec.decision_id,
            "Loan_Application_ID__c": app_id,
            "Final_Decision__c": rec.final_decision.value,
            "Confidence_Score__c": rec.confidence_score,
            "Routing_Path__c": rec.routing_path.value,
            "Reasoning_Trace__c": rec.reasoning_trace[:255],
            "Adverse_Action_Required__c": bool(rec.adverse_action_reason),
            "ECOA_Compliant__c": True,
        }
        if self._mock:
            return {"success":True,"mock":True,"event":"HAICD_Decision__e","payload":payload}
        try:
            import requests
            url=f"{self.config.instance_url}/services/data/{self.config.api_version}/sobjects/HAICD_Decision__e/"
            hdrs={"Authorization":f"Bearer {self.config.access_token}","Content-Type":"application/json"}
            r=requests.post(url,json=payload,headers=hdrs)
            return {"success":r.status_code==201,"response":r.json()}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def update_loan_application(self, rec: HAICDDecisionRecord, record_id: str) -> Dict[str,Any]:
        """Update Salesforce Loan_Application__c record with HAICD decision."""
        update = {
            "HAICD_Decision__c":     rec.final_decision.value,
            "HAICD_Confidence__c":   rec.confidence_score,
            "HAICD_Routing_Path__c": rec.routing_path.value,
            "HAICD_Reasoning__c":    rec.reasoning_trace[:1000],
            "HAICD_Processed__c":    True,
            "Adverse_Action_Text__c": rec.adverse_action_reason,
            "Underwriter_Queue__c":   self.QUEUE_MAP[rec.final_decision],
        }
        if self._mock:
            return {"success":True,"mock":True,"record_id":record_id,"updates":update}
        try:
            import requests
            url=f"{self.config.instance_url}/services/data/{self.config.api_version}/sobjects/Loan_Application__c/{record_id}"
            hdrs={"Authorization":f"Bearer {self.config.access_token}","Content-Type":"application/json"}
            r=requests.patch(url,json=update,headers=hdrs)
            return {"success":r.status_code==204}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def trigger_agentforce(self, rec: HAICDDecisionRecord, record_id: str) -> Dict[str,Any]:
        """Trigger Agentforce Agent Topic with HAICD reasoning context."""
        topic = self.AGENT_TOPIC_MAP[rec.final_decision]
        ctx = {
            "loan_application_id": record_id,
            "haicd_decision":      rec.final_decision.value,
            "confidence":          rec.confidence_score,
            "haicd_reasoning":     rec.reasoning_trace,
            "routing_path":        rec.routing_path.value,
            "adverse_action":      rec.adverse_action_reason,
        }
        if self._mock:
            return {"success":True,"mock":True,"agent_topic":topic,"context":ctx}
        try:
            import requests
            url=f"{self.config.instance_url}/services/data/{self.config.api_version}/einstein/copilot/actions/invoke"
            hdrs={"Authorization":f"Bearer {self.config.access_token}","Content-Type":"application/json"}
            r=requests.post(url,json={"agentTopicName":topic,"context":ctx},headers=hdrs)
            return {"success":r.status_code==200,"response":r.json()}
        except Exception as e:
            return {"success":False,"error":str(e)}

    def full_flow(self, rec: HAICDDecisionRecord, app_id: str) -> Dict[str,Any]:
        """Execute full HAICD → Salesforce integration chain."""
        s1=self.publish_decision_event(rec,app_id)
        s2=self.update_loan_application(rec,app_id)
        s3=self.trigger_agentforce(rec,app_id)
        return {"chain":"HAICD→PlatformEvent→Loan_Application→Agentforce",
                "success":all(x["success"] for x in [s1,s2,s3]),
                "steps":{"1_event":s1,"2_record":s2,"3_agentforce":s3}}
