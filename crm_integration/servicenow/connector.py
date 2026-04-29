"""crm_integration/servicenow/connector.py — HAICD servicenow Integration"""
import sys; sys.path.insert(0,'/home/claude/haicd')
from haicd_core import HAICDDecisionRecord
from typing import Dict, Any

class HAICDConnector:
    def __init__(self): self._mock = True
    def push_decision(self, rec: HAICDDecisionRecord, record_id: str) -> Dict[str,Any]:
        return {"success":True,"mock":True,"platform":"servicenow",
                "decision":rec.final_decision.value,"record_id":record_id}
