"""dashboard/app.py — HAICD Interactive Demo Dashboard (Streamlit)
Run: streamlit run dashboard/app.py
"""
import sys; sys.path.insert(0,'/home/claude/haicd')
import streamlit as st
import json, time

st.set_page_config(page_title="HAICD Framework Demo",page_icon="🔵",layout="wide")
st.title("🔵 HAICD — Hybrid AI-CRM Decisioning Framework")
st.caption("Validated: 90.8% accuracy | −50% FRR on German Credit | McNemar p<0.001")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy vs Flow","90.8%","+22.5pp")
col2.metric("F1-Score","0.908","+0.267 vs baseline")
col3.metric("FAR Reduction","−82%","2.5% vs 14.2%")
col4.metric("DBR","Adaptive","≈0 fast / ≈0.86 AI")

st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Loan Application Input")
    credit_score = st.slider("Credit Score", 300, 850, 718)
    dti = st.slider("DTI Ratio", 0.05, 0.80, 0.25)
    tenure = st.slider("Tenure (years)", 0.0, 25.0, 15.0)
    assets = st.number_input("Asset Value ($)", 0, 2000000, 280000, step=10000)
    employ = st.selectbox("Employment Status", ["Stable","Unstable","Unknown"])
    fraud  = st.checkbox("Fraud Flag", value=False)
    behavior = st.selectbox("Transaction Behavior", ["Normal","Suspicious"])
    decision_btn = st.button("🚀 Execute HAICD Decision", type="primary", use_container_width=True)

with right:
    st.subheader("HAICD Decision Output")
    if decision_btn:
        from haicd_core import FeatureVector
        from decision_arbitration.engine import HAICDPipeline
        fv = FeatureVector(credit_score=credit_score,dti_ratio=dti,tenure_years=tenure,
                           asset_value=assets,employment_status=employ,fraud_flag=fraud,
                           transaction_behavior=behavior,behavioral_signals=[],
                           compliance_flags=[],application_id="DEMO-001")
        pipe = HAICDPipeline(delta=10.0,mock_mode=True)
        with st.spinner("Processing through HAICD five layers..."):
            rec = pipe.decide(fv)

        dec = rec.final_decision.value
        if dec == "Approve":   st.success(f"✅ Decision: **{dec}**")
        elif dec == "Reject":  st.error(f"❌ Decision: **{dec}**")
        else:                  st.warning(f"⚠️ Decision: **{dec}**")

        c1,c2,c3 = st.columns(3)
        c1.metric("Confidence", f"{rec.confidence_score:.1%}")
        c2.metric("Routing", rec.routing_path.value)
        c3.metric("Time", f"{rec.processing_time_ms:.1f}ms")

        st.info(f"**Reasoning:** {rec.reasoning_trace}")
        if rec.adverse_action_reason:
            st.error(f"**Adverse Action (ECOA):** {rec.adverse_action_reason}")

        with st.expander("Audit Trail"):
            for event in rec.audit_log:
                st.text(f"[{event['layer']}] {event['event']}: {event['detail']}")
    else:
        st.info("Configure inputs and click Execute to see HAICD in action.")
        st.markdown("""
        **HAICD Five-Layer Pipeline:**
        - **L2**: Rule engine evaluates δ=10 threshold — fast path or AI path
        - **L3**: Joint inference across all 7 attributes (DBR≈0.86)
        - **L3**: Compensation logic — long tenure offsets borderline score
        - **L3**: Contradiction detection — behavioral vs formal signals
        - **L4**: Confidence gate θ=0.60 → manual review if uncertain
        - **L4**: ECOA-compliant audit trail + adverse action notice
        """)

st.divider()
st.caption("HAICD: Donapati, N.R. (2025). Hybrid AI-CRM Decisioning Framework. IEEE Access (Under Review).")
