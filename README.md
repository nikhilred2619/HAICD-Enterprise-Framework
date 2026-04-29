# 🔵 HAICD Framework
## Hybrid AI-CRM Decisioning Framework for Context-Aware Loan Risk Evaluation

> **"From Rule Execution to Judgment Augmentation — DBR = 0 to DBR ≈ 0.86"**

<div align="center">

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-3776ab?style=for-the-badge&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Paper: IEEE Access](https://img.shields.io/badge/Paper-IEEE_Access-blue?style=for-the-badge)](https://orcid.org/0009-0006-7699-3928)

**Author:** Nikhil Reddy Donapati · Agentforce AI Specialist · Texas, USA  
**ORCID:** [0009-0006-7699-3928](https://orcid.org/0009-0006-7699-3928)

</div>

---

## The Problem: Why Rule Engines Fail in Financial Decisioning

Enterprise CRM systems evaluate loan applications through sequential IF/THEN rules — each condition evaluated independently as Boolean predicates. This is Decision Boundary Rigidity DBR = 0.

**Real cases rule engines get wrong:**

```
CS-041: Score=718, DTI=25%, Tenure=15yr, Assets=$280k
→ Rule Engine: MANUAL REVIEW (score 2pts below threshold)
→ HAICD:       APPROVE (15yr tenure + $280k assets materially offset borderline score)
→ Ground Truth: APPROVE  ✅

CS-082: Score=810, DTI=65%, Income=$400k  
→ Rule Engine: APPROVE (sees high score only)
→ HAICD:       MANUAL REVIEW (contradiction: strong score + over-leverage detected)
→ Ground Truth: MANUAL REVIEW  ✅

CS-095: Score=740, DTI=28%, Fraud_Flag=No, 50 intl. transfers/48h
→ Rule Engine: APPROVE (fraud flag negative)
→ HAICD:       REJECT (behavioral evidence overrides formal flag)
→ Ground Truth: REJECT  ✅
```

HAICD solves this through **joint attribute inference** — the theoretical property measured by Decision Boundary Rigidity.

---

## Validated Performance

```
EXPERIMENT 1: Synthetic Adversarial Benchmark (N=120)
  System A (Salesforce Flow, DBR≈0): 68.3%  accuracy, F1=0.641, FAR=14.2%, FRR=17.5%
  System B (HAICD, DBR≈0.86):        90.8%  accuracy, F1=0.908, FAR=2.5%,  FRR=6.7%
  Improvement: +22.5pp accuracy, −82% FAR
  McNemar: χ²=19.31, p<0.001

EXPERIMENT 2: German Credit Public Dataset (N=1,000)
  System A (Salesforce Flow): FRR=34.5%, F1=0.377
  System B (HAICD):           FRR=17.2%, F1=0.528
  Improvement: −50% False Rejection Rate (identical FAR=4.1%)
  McNemar: χ²=27.44, p<0.001, Power=100%, Cohen's h=0.40

LLM STABILITY: 98.9% cross-run consistency (temperature=0.0)
LATENCY:       0.87s effective hybrid | $140/day per 100k applications
```

---

## Novel Theoretical Contribution: Decision Boundary Rigidity

**DBR = Jₜ/N** — the fraction of input attributes evaluated jointly via a unified inference function.

| System | DBR | Inference | Implication |
|--------|-----|-----------|-------------|
| Rule-Based (Salesforce Flow) | ≈ 0 | None | Independent Boolean predicates — cannot detect compensatory factors |
| LLM Only (Agentforce) | ≈ 0.86 | Full | Self-attention; 6/7=0.857 attributes jointly evaluated |
| **HAICD (Proposed)** | **Adaptive** | **Selective** | DBR≈0 for R=0 (fast path); DBR≈0.86 for R=1 (AI path) |

**Proposition 1:** For a dataset with ε=33% adversarial edge cases, a DBR=0 classifier is bounded above by 1−0.5ε=83.5%. Observed Flow accuracy: 68.3% (below bound due to threshold-boundary failures). HAICD: 90.8% (above bound via joint inference).

---

## Framework Architecture

```
HAICD Five-Layer Architecture
═══════════════════════════════════════════════════════════════
│  L1  INPUT NORMALIZATION                                     │
│       F = {score, DTI, tenure, assets, employment,           │
│            fraud_flag, transaction_behavior}                 │
├───────────────────────────────────────────────────────────────┤
│  L2  RULE-BASED CONFIDENCE CLASSIFICATION  (DBR ≈ 0)        │
│       |score − threshold| > δ=10 AND no anomaly → R=0       │
│       Otherwise → R=1 (AI path)                              │
├───────────────────────────────────────────────────────────────┤
│  L3  AI CONTEXTUAL REASONING               (DBR ≈ 0.86)     │
│       • Simultaneous multi-attribute evaluation               │
│       • Compensation logic: tenure/assets offset score       │
│       • Contradiction detection: behavioral vs formal         │
│       • Structured JSON output: {decision, confidence, T}    │
├───────────────────────────────────────────────────────────────┤
│  L4  DECISION ARBITRATION + AUDIT TRAIL                      │
│       Confidence gate θ=0.60 → Manual Review if C<θ         │
│       ECOA / Fair Lending / SR 11-7 audit trail generation   │
│       Adverse action reason generation (ECOA compliance)      │
├───────────────────────────────────────────────────────────────┤
│  L5  FEEDBACK & ADAPTATION   (future deployment)             │
│       δ and prompt weight updates from outcome data           │
═══════════════════════════════════════════════════════════════
```

---

## Quick Start

```bash
git clone https://github.com/nikhildonapati/haicd-framework.git
cd haicd-framework
pip install -r requirements.txt

# REST API
uvicorn api.main:app --reload --port 8000
# → http://localhost:8000/docs

# Run full experiment (paper replication)
python3 evaluation/experiment.py

# Docker
docker compose up --build
```

---

## REST API Example

```python
import requests
response = requests.post("http://localhost:8000/loan/decide", json={
    "customer_id": "CS-041",
    "credit_score": 718.0,
    "dti_ratio": 0.25,
    "tenure_years": 15.0,
    "asset_value": 280000.0,
    "employment_status": "Stable",
    "fraud_flag": False,
    "transaction_behavior": "Normal"
})
# → {"final_decision":"Approve","confidence_score":0.847,"routing_path":"R1_AIPath",
#    "reasoning_trace":"CS-041: Score 718; DTI 25%; Tenure 15.0yr; Assets $280,000; 
#     COMPENSATION: 15yr tenure offsets borderline score. Confidence: 0.85."}
```

---

## Enterprise Integrations

### Salesforce CRM + Agentforce
```python
from crm_integration.salesforce.connector import SalesforceHAICDConnector
connector = SalesforceHAICDConnector()
result = connector.full_flow(haicd_record, loan_application_id="LA-001")
# → Platform Event → Loan_Application__c update → Agentforce Agent Topic trigger
```

### Cross-Platform Mappings (Table XVII)
| Layer | Salesforce (Validated) | MS Dynamics 365 | Generic CRM |
|-------|----------------------|-----------------|-------------|
| L2 Rule Engine | Record-Triggered Flow | Business Rules Engine | Workflow engine |
| L3 AI Reasoning | Agentforce + Atlas | Azure OpenAI + Power Automate | Any LLM REST API |
| L4 Audit Trail | Apex @InvocableMethod | Dataverse + Power Automate | CRM audit log |

---

## Enterprise Use Cases

| Use Case | HAICD Benefit | Integration |
|----------|---------------|-------------|
| Loan approval automation | −82% FAR, −50% FRR | Salesforce + Agentforce |
| Financial risk assessment | DBR edge case detection | SAP BTP + HAICD API |
| Mortgage decisioning | Compensatory factor recognition | Oracle + HAICD |
| Insurance claim approval | Contradiction detection | ServiceNow + HAICD |
| Banking compliance | ECOA/SR 11-7 audit trail | Any CRM |
| CRM underwriting | 98.9% consistent decisions | Dynamics 365 + Azure OAI |

---

## Prompt Ablation Study (Table XIV)

| Configuration | Accuracy | F1-Score | Drop |
|---------------|----------|----------|------|
| Full Prompt | **90.8%** | **0.908** | — |
| Without Compensation Logic | 82.5% | 0.831 | −8.3pp |
| Without Contradiction Detection | 84.2% | 0.849 | −6.6pp |
| Without Structured Output | 88.3% | 0.876 | −2.5pp |
| No Prompt (Raw LLM) | 74.2% | 0.748 | **−16.6pp** |

Structured prompt design drives the majority of HAICD's performance advantage over raw LLM.

---

## Project Structure

```
haicd-framework/
├── haicd_core.py                    ← Core data structures (FeatureVector, DBR, etc.)
├── rule_engine/engine.py            ← L2: Rule-based classification (DBR≈0)
├── ai_reasoning/engine.py           ← L3: AI contextual reasoning (DBR≈0.86)
├── decision_arbitration/engine.py   ← L4: Arbitration + ECOA audit trail
├── api/main.py                      ← FastAPI REST API (OpenAPI 3.0)
├── crm_integration/
│   ├── salesforce/connector.py      ← Salesforce + Agentforce
│   ├── dynamics365/connector.py     ← Microsoft Dynamics 365 + Azure OAI
│   ├── servicenow/connector.py      ← ServiceNow ITSM
│   ├── sap/connector.py             ← SAP BTP + S/4HANA
│   └── oracle/connector.py          ← Oracle Financial Cloud
├── evaluation/experiment.py         ← Full paper experiment replication
├── dashboard/app.py                 ← Streamlit interactive demo
├── tests/test_haicd.py              ← Test suite (24/26 passing)
├── documentation/
│   ├── METHODOLOGY.md
│   └── PROMPT_ENGINEERING.md        ← Appendix A specification
├── results/haicd_results.json       ← Experimental results
├── Dockerfile + docker-compose.yml
└── requirements.txt
```

---

## Citation

```bibtex
@article{donapati2025haicd,
  title   = {HAICD: A Hybrid AI-CRM Decisioning Framework for 
             Context-Aware Loan Risk Evaluation Using LLM-Based Reasoning},
  author  = {Donapati, Nikhil Reddy},
  journal = {IEEE Access},
  year    = {2025},
  note    = {Under Review},
  url     = {https://github.com/nikhildonapati/haicd-framework}
}
```

---

## License

MIT License — see [LICENSE](LICENSE).

---

<div align="center">

**Nikhil Reddy Donapati · Texas, USA**  
*Agentforce AI Specialist & Senior Salesforce Developer*  
[ORCID 0009-0006-7699-3928](https://orcid.org/0009-0006-7699-3928)

</div>
