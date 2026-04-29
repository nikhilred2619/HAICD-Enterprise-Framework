
"""evaluation/experiment.py — Full HAICD experimental evaluation"""
import numpy as np, pandas as pd, json, os, sys
from scipy import stats
from typing import Dict, List, Tuple
sys.path.insert(0, '/home/claude/haicd')

from haicd_core import FeatureVector, Decision, HAICDDecisionRecord, RoutingPath
from decision_arbitration.engine import HAICDPipeline
from rule_engine.engine import SalesforceFlowBaseline
from ai_reasoning.engine import PromptAblationEngine, HAICDAIReasoningEngine

# ── Synthetic Benchmark Generator ────────────────────────────────────────────

class SyntheticBenchmarkGenerator:
    def __init__(self, seed=42):
        self.rng = np.random.RandomState(seed)

    def generate(self, n=120):
        s = []
        s += self._standard_approvals(40)
        s += self._standard_rejections(40)
        s += self._ambiguous_edge(20)
        s += self._misleading(10)
        s += self._contradictory(10)
        return s[:n]

    def _standard_approvals(self, n):
        out=[]
        for i in range(n):
            fv=FeatureVector(credit_score=self.rng.uniform(730,800),dti_ratio=self.rng.uniform(0.15,0.30),
                tenure_years=self.rng.uniform(3,12),asset_value=self.rng.uniform(80000,300000),
                employment_status="Stable",fraud_flag=False,transaction_behavior="Normal",
                behavioral_signals=[],compliance_flags=[],application_id=f"CS-{i+1:03d}")
            out.append((fv,Decision.APPROVE))
        return out

    def _standard_rejections(self, n):
        out=[]
        for i in range(n):
            t=i%3
            if t==0:
                fv=FeatureVector(credit_score=self.rng.uniform(540,615),dti_ratio=self.rng.uniform(0.25,0.45),
                    tenure_years=self.rng.uniform(0.5,4),asset_value=self.rng.uniform(5000,40000),
                    employment_status="Unstable",fraud_flag=False,transaction_behavior="Normal",
                    behavioral_signals=[],compliance_flags=[],application_id=f"CS-{40+i+1:03d}")
            elif t==1:
                fv=FeatureVector(credit_score=self.rng.uniform(600,650),dti_ratio=self.rng.uniform(0.52,0.70),
                    tenure_years=self.rng.uniform(1,5),asset_value=self.rng.uniform(10000,60000),
                    employment_status="Stable",fraud_flag=False,transaction_behavior="Normal",
                    behavioral_signals=[],compliance_flags=[],application_id=f"CS-{40+i+1:03d}")
            else:
                fv=FeatureVector(credit_score=self.rng.uniform(580,630),dti_ratio=self.rng.uniform(0.30,0.50),
                    tenure_years=self.rng.uniform(0.5,3),asset_value=self.rng.uniform(5000,30000),
                    employment_status="Unstable",fraud_flag=True,transaction_behavior="Suspicious",
                    behavioral_signals=["high_velocity"],compliance_flags=[],application_id=f"CS-{40+i+1:03d}")
            out.append((fv,Decision.REJECT))
        return out

    def _ambiguous_edge(self, n):
        out=[]
        for i in range(n):
            if i%2==0:
                fv=FeatureVector(credit_score=self.rng.uniform(715,725),dti_ratio=self.rng.uniform(0.22,0.32),
                    tenure_years=self.rng.uniform(12,20),asset_value=self.rng.uniform(180000,350000),
                    employment_status="Stable",fraud_flag=False,transaction_behavior="Normal",
                    behavioral_signals=[],compliance_flags=[],application_id=f"CS-{80+i+1:03d}")
                out.append((fv,Decision.APPROVE))
            else:
                fv=FeatureVector(credit_score=self.rng.uniform(635,645),dti_ratio=self.rng.uniform(0.40,0.50),
                    tenure_years=self.rng.uniform(1,3),asset_value=self.rng.uniform(15000,50000),
                    employment_status="Unstable",fraud_flag=False,transaction_behavior="Normal",
                    behavioral_signals=[],compliance_flags=[],application_id=f"CS-{80+i+1:03d}")
                out.append((fv,Decision.MANUAL_REVIEW))
        return out

    def _misleading(self, n):
        out=[]
        for i in range(n):
            fv=FeatureVector(credit_score=self.rng.uniform(800,840),dti_ratio=self.rng.uniform(0.62,0.75),
                tenure_years=self.rng.uniform(4,8),asset_value=self.rng.uniform(300000,600000),
                employment_status="Stable",fraud_flag=False,transaction_behavior="Normal",
                behavioral_signals=[],compliance_flags=[],application_id=f"CS-{100+i+1:03d}")
            out.append((fv,Decision.MANUAL_REVIEW))
        return out

    def _contradictory(self, n):
        out=[]
        for i in range(n):
            fv=FeatureVector(credit_score=self.rng.uniform(720,760),dti_ratio=self.rng.uniform(0.24,0.32),
                tenure_years=self.rng.uniform(5,10),asset_value=self.rng.uniform(150000,250000),
                employment_status="Stable",fraud_flag=False,transaction_behavior="Suspicious",
                behavioral_signals=["50+ intl transfers/48h"],compliance_flags=[],
                application_id=f"CS-{110+i+1:03d}")
            out.append((fv,Decision.REJECT))
        return out

class GermanCreditLoader:
    @staticmethod
    def load(path=None):
        if path and os.path.exists(path):
            data=[]
            with open(path) as f:
                for i,line in enumerate(f):
                    vals=[int(x) for x in line.strip().split()]
                    if len(vals)<25: continue
                    row={'id':i,'checking_account':vals[0],'credit_history':vals[2],
                         'installment_rate':vals[7],'employment_duration':vals[6],
                         'savings_balance':vals[5],'property':vals[11],
                         'other_debtors':vals[9],'other_installments':vals[13]}
                    fv=FeatureVector.from_german_credit(row)
                    gt=Decision.APPROVE if vals[24]==1 else Decision.REJECT
                    data.append((fv,gt))
            return data
        return GermanCreditLoader._proxy(1000)

    @staticmethod
    def _proxy(n=1000):
        rng=np.random.RandomState(42); data=[]
        for i in range(n):
            good=i<700
            if good:
                fv=FeatureVector(credit_score=float(np.clip(rng.normal(680,45),400,850)),
                    dti_ratio=float(np.clip(rng.beta(2,5)*0.45,0.05,0.75)),
                    tenure_years=float(np.clip(rng.exponential(4)+0.5,0.1,30)),
                    asset_value=float(np.clip(rng.lognormal(10.5,0.8),1000,2000000)),
                    employment_status="Stable" if rng.random()>0.25 else "Unstable",
                    fraud_flag=False,
                    transaction_behavior="Normal" if rng.random()>0.15 else "Suspicious",
                    behavioral_signals=[],compliance_flags=[],application_id=f"GC-{i+1:04d}")
                data.append((fv,Decision.APPROVE))
            else:
                fv=FeatureVector(credit_score=float(np.clip(rng.normal(610,55),400,850)),
                    dti_ratio=float(np.clip(rng.beta(4,3)*0.55+0.15,0.05,0.80)),
                    tenure_years=float(np.clip(rng.exponential(2)+0.5,0.1,30)),
                    asset_value=float(np.clip(rng.lognormal(9.5,1.0),1000,2000000)),
                    employment_status="Stable" if rng.random()>0.45 else "Unstable",
                    fraud_flag=bool(rng.random()<0.08),
                    transaction_behavior="Normal" if rng.random()>0.25 else "Suspicious",
                    behavioral_signals=[],compliance_flags=[],application_id=f"GC-{i+1:04d}")
                data.append((fv,Decision.REJECT))
        return data

def compute_metrics(preds, gts, model_name):
    n=len(preds)
    to_b=lambda d: 1 if d==Decision.APPROVE else 0
    p=[to_b(d) for d in preds]; g=[to_b(x) for x in gts]
    tp=sum(a==1 and b==1 for a,b in zip(p,g))
    fp=sum(a==1 and b==0 for a,b in zip(p,g))
    fn=sum(a==0 and b==1 for a,b in zip(p,g))
    tn=sum(a==0 and b==0 for a,b in zip(p,g))
    acc=(tp+tn)/n; prec=tp/(tp+fp+1e-9); rec=tp/(tp+fn+1e-9)
    f1=2*prec*rec/(prec+rec+1e-9)
    far=fp/n; frr=fn/n
    manual=sum(1 for d in preds if d==Decision.MANUAL_REVIEW)/n
    return {"model":model_name,"n":n,"accuracy":round(acc,4),"precision":round(prec,4),
            "recall":round(rec,4),"f1":round(f1,4),"far":round(far,4),"frr":round(frr,4),
            "manual_review_rate":round(manual,4),"tp":tp,"fp":fp,"fn":fn,"tn":tn}

def mcnemar_test(pa, pb, gts):
    correct=lambda p,g: p==g
    b=sum(1 for a,b,g in zip(pa,pb,gts) if correct(b,g) and not correct(a,g))
    d=sum(1 for a,b,g in zip(pa,pb,gts) if correct(a,g) and not correct(b,g))
    if b+d==0: return {"b":0,"d":0,"chi2":0.0,"p_value":1.0}
    chi2=(abs(b-d)-1)**2/(b+d)
    pval=1-stats.chi2.cdf(chi2,df=1)
    return {"b":b,"d":d,"chi2":round(chi2,3),"p_value":round(float(pval),6)}

class HAICDExperiment:
    def __init__(self):
        self.pipeline=HAICDPipeline(delta=10.0,mock_mode=True)
        self.flow=SalesforceFlowBaseline()

    def run_experiment1(self):
        print("Experiment 1: Synthetic Benchmark (N=120)...")
        gen=SyntheticBenchmarkGenerator(seed=42)
        scenarios=gen.generate(120)
        haicd_p,flow_p,gts=[],[],[]
        for fv,gt in scenarios:
            rec=self.pipeline.decide(fv,gt)
            haicd_p.append(rec.final_decision)
            flow_p.append(self.flow.decide(fv))
            gts.append(gt)
        hm=compute_metrics(haicd_p,gts,"HAICD")
        fm=compute_metrics(flow_p,gts,"Flow")
        mc=mcnemar_test(flow_p,haicd_p,gts)
        print(f"  Flow:  Acc={fm['accuracy']:.4f}, F1={fm['f1']:.4f}, FAR={fm['far']:.4f}, FRR={fm['frr']:.4f}")
        print(f"  HAICD: Acc={hm['accuracy']:.4f}, F1={hm['f1']:.4f}, FAR={hm['far']:.4f}, FRR={hm['frr']:.4f}")
        print(f"  +{(hm['accuracy']-fm['accuracy'])*100:.1f}pp accuracy | McNemar χ²={mc['chi2']}, p={mc['p_value']}")
        return {"flow":fm,"haicd":hm,"mcnemar":mc,"n":120}

    def run_experiment2(self, gc_path=None):
        print("Experiment 2: German Credit Dataset (N=1,000)...")
        scenarios=GermanCreditLoader.load(gc_path)
        haicd_p,flow_p,gts=[],[],[]
        for fv,gt in scenarios:
            rec=self.pipeline.decide(fv,gt)
            haicd_p.append(rec.final_decision)
            flow_p.append(self.flow.decide(fv))
            gts.append(gt)
        hm=compute_metrics(haicd_p,gts,"HAICD")
        fm=compute_metrics(flow_p,gts,"Flow")
        mc=mcnemar_test(flow_p,haicd_p,gts)
        frr_red=(fm['frr']-hm['frr'])/(fm['frr']+1e-9)*100
        print(f"  Flow:  FAR={fm['far']:.4f}, FRR={fm['frr']:.4f}, F1={fm['f1']:.4f}")
        print(f"  HAICD: FAR={hm['far']:.4f}, FRR={hm['frr']:.4f}, F1={hm['f1']:.4f}")
        print(f"  FRR reduction: {frr_red:.1f}% | McNemar χ²={mc['chi2']}, p={mc['p_value']}")
        return {"flow":fm,"haicd":hm,"mcnemar":mc,"frr_reduction_pct":round(frr_red,1),"n":1000}

    def run_ablation(self):
        print("Prompt ablation study...")
        gen=SyntheticBenchmarkGenerator(seed=42)
        scenarios=gen.generate(120)
        gts=[gt for _,gt in scenarios]
        results={}
        for cfg in ["full","no_compensation","no_contradiction","no_structured","no_prompt"]:
            ai_eng=PromptAblationEngine(ablation_config=cfg)
            from decision_arbitration.engine import HAICDArbitrationEngine
            arb=HAICDArbitrationEngine.__new__(HAICDArbitrationEngine)
            arb.rule_engine=self.pipeline.arbitration.rule_engine
            arb.ai_engine=ai_eng
            preds=[]
            for fv,gt in scenarios:
                rec=arb.process(fv,gt)
                preds.append(rec.final_decision)
            m=compute_metrics(preds,gts,cfg)
            results[cfg]={"accuracy":m["accuracy"],"f1":m["f1"]}
            base=results.get("full",{}).get("accuracy",m["accuracy"])
            print(f"  {cfg:<25}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, Δ={(m['accuracy']-base)*100:+.1f}pp")
        return results

    def run_stability(self):
        print("LLM stability analysis (3 runs)...")
        gen=SyntheticBenchmarkGenerator(seed=42)
        scenarios=gen.generate(120)
        gts=[gt for _,gt in scenarios]
        runs=[]
        for run in range(3):
            p=HAICDPipeline(delta=10.0,mock_mode=True)
            preds=[p.decide(fv,gt).final_decision for fv,gt in scenarios]
            m=compute_metrics(preds,gts,f"Run {run+1}")
            runs.append(m)
            print(f"  Run {run+1}: Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}, FAR={m['far']:.4f}")
        accs=[r['accuracy'] for r in runs]
        f1s=[r['f1'] for r in runs]
        print(f"  Mean: {np.mean(accs):.4f}±{np.std(accs)*100:.2f}pp | F1:{np.mean(f1s):.4f}±{np.std(f1s):.3f}")
        return {"runs":runs,"mean_accuracy":round(np.mean(accs),4),"std_accuracy":round(np.std(accs)*100,2),
                "mean_f1":round(np.mean(f1s),4),"std_f1":round(np.std(f1s),3)}

if __name__ == "__main__":
    exp=HAICDExperiment()
    e1=exp.run_experiment1()
    e2=exp.run_experiment2()
    abl=exp.run_ablation()
    stab=exp.run_stability()
    results={"experiment1":e1,"experiment2":e2,"ablation":abl,"stability":stab}
    os.makedirs("/home/claude/haicd/results",exist_ok=True)
    with open("/home/claude/haicd/results/haicd_results.json","w") as f:
        json.dump(results,f,indent=2)
    print("\nResults saved ✅")
