"""crm_integration/oracle/connector.py — HAICD oracle Integration"""
import sys; sys.path.insert(0,'/home/claude/haicd')
from haicd_core import HAICDDecisionRecord
from typing import Dict, Any

class HAICDConnector:
    def __init__(self): self._mock = True
    def push_decision(self, rec: HAICDDecisionRecord, record_id: str) -> Dict[str,Any]:
        return {"success":True,"mock":True,"platform":"oracle",
                "decision":rec.final_decision.value,"record_id":record_id}
