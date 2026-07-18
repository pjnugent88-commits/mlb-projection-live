from __future__ import annotations
import math
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score,brier_score_loss,log_loss,roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .odds import expected_value_per_unit,probability_to_american

HR_FEATURES=[
"expected_pa","batting_order","is_home","batter_hr_pa_long","batter_barrel_pa_long","batter_barrel_bbe_long","batter_hard_air_pa_long","batter_sweet_pa_long","batter_k_rate_long","batter_bb_rate_long","batter_xwoba_long","batter_ev90_long","batter_hr_pa_recent","batter_barrel_pa_recent","batter_hard_air_pa_recent","batter_xwoba_recent","batter_ev90_recent","pitcher_hr_bf","pitcher_barrel_bf","pitcher_hard_air_bf","pitcher_fly_rate","pitcher_k_rate","pitcher_bb_rate","pitcher_xwoba","pitcher_ev90","pitcher_outs","same_hand","park_hr_factor","temperature_f","humidity_pct","wind_speed_mph","precipitation_in","roof_control_factor","bvp_hr_shrunk","bvp_barrel_shrunk","bvp_reliability","bvp_pa"]

def _logit():return Pipeline([("imputer",SimpleImputer(strategy="median")),("scale",StandardScaler()),("model",LogisticRegression(C=.35,max_iter=1500))])
def _tree(seed):return Pipeline([("imputer",SimpleImputer(strategy="median")),("model",ExtraTreesClassifier(n_estimators=180,max_depth=9,min_samples_leaf=24,max_features=.72,n_jobs=-1,random_state=seed))])

@dataclass
class HomeRunModel:
    logistic:Pipeline;tree:Pipeline;calibrator:LogisticRegression;tree_weight:float;metrics:dict
    def raw(self,x):
        a=self.logistic.predict_proba(x[HR_FEATURES])[:,1];b=self.tree.predict_proba(x[HR_FEATURES])[:,1]
        return np.clip((1-self.tree_weight)*a+self.tree_weight*b,1e-5,1-1e-5)
    def predict(self,x):
        p=self.raw(x);z=np.log(p/(1-p)).reshape(-1,1)
        return np.clip(self.calibrator.predict_proba(z)[:,1],.002,.65)

def train_home_run_model(frame:pd.DataFrame,tree_weight=.35,test_fraction=.15,calibration_fraction=.15,minimum_rows=4000,random_state=42)->HomeRunModel:
    o=frame.sort_values(["game_date","game_pk","player_id"]).dropna(subset=["target_hr"]).reset_index(drop=True)
    if len(o)<minimum_rows:raise ValueError(f"Home-run model requires at least {minimum_rows} rows; received {len(o)}")
    t=int(len(o)*(1-test_fraction));c=int(len(o)*(1-test_fraction-calibration_fraction));c=max(1,min(c,t-1));t=max(c+1,min(t,len(o)-1))
    tr,ca,te=o.iloc[:c],o.iloc[c:t],o.iloc[t:]
    lg,trm=_logit(),_tree(random_state);lg.fit(tr[HR_FEATURES],tr.target_hr);trm.fit(tr[HR_FEATURES],tr.target_hr)
    pc=np.clip((1-tree_weight)*lg.predict_proba(ca[HR_FEATURES])[:,1]+tree_weight*trm.predict_proba(ca[HR_FEATURES])[:,1],1e-5,1-1e-5)
    cal=LogisticRegression(C=1e6,max_iter=1000).fit(np.log(pc/(1-pc)).reshape(-1,1),ca.target_hr)
    pt=np.clip((1-tree_weight)*lg.predict_proba(te[HR_FEATURES])[:,1]+tree_weight*trm.predict_proba(te[HR_FEATURES])[:,1],1e-5,1-1e-5)
    pt=np.clip(cal.predict_proba(np.log(pt/(1-pt)).reshape(-1,1))[:,1],.002,.65);y=te.target_hr.to_numpy(int);q=float(np.quantile(pt,.9))
    m={"model_version":"stage4-home-run-probability","train_rows":len(tr),"calibration_rows":len(ca),"test_rows":len(te),"test_start_date":str(pd.Timestamp(te.game_date.min()).date()),"test_end_date":str(pd.Timestamp(te.game_date.max()).date()),"event_rate":float(y.mean()),"prediction_mean":float(pt.mean()),"brier_score":float(brier_score_loss(y,pt)),"log_loss":float(log_loss(y,pt,labels=[0,1])),"roc_auc":float(roc_auc_score(y,pt)) if len(np.unique(y))>1 else math.nan,"average_precision":float(average_precision_score(y,pt)),"top_decile_cutoff":q,"top_decile_hr_rate":float(y[pt>=q].mean()),"feature_count":len(HR_FEATURES),"bvp_prior_pa":40}
    fit=o.iloc[:t];lg.fit(fit[HR_FEATURES],fit.target_hr);trm.fit(fit[HR_FEATURES],fit.target_hr)
    return HomeRunModel(lg,trm,cal,tree_weight,m)

def score_home_runs(frame,model,odds=None,minimum_edge=.04,minimum_ev=.04,watch_probability=.16):
    if frame.empty:return pd.DataFrame()
    x=frame.copy();x["home_run_probability"]=model.predict(x);x["fair_odds"]=x.home_run_probability.map(lambda p:probability_to_american(float(np.clip(p,.002,.998))))
    o=pd.DataFrame() if odds is None else odds.copy()
    if not o.empty:
        o=o[o.market_key.eq("batter_home_runs")];keep=[c for c in ["game_pk","player_name","point","over_odds","under_odds","over_book","under_book","over_no_vig_probability","under_no_vig_probability"] if c in o]
        x=x.merge(o[keep],on=["game_pk","player_name"],how="left")
    for c in ["point","over_odds","under_odds","over_no_vig_probability","under_no_vig_probability"]:
        if c not in x:x[c]=np.nan
    for c in ["over_book","under_book"]:
        if c not in x:x[c]=None
    x["line"]=pd.to_numeric(x.point,errors="coerce").fillna(.5);x["market_edge"]=x.home_run_probability-pd.to_numeric(x.over_no_vig_probability,errors="coerce")
    x["ev_per_unit"]=[expected_value_per_unit(p,o) if pd.notna(o) else math.nan for p,o in zip(x.home_run_probability,x.over_odds)]
    x["signal"]=["HR VALUE" if pd.notna(r.over_odds) and pd.notna(r.market_edge) and r.market_edge>=minimum_edge and r.ev_per_unit>=minimum_ev else ("HR WATCH" if pd.isna(r.over_odds) and r.home_run_probability>=watch_probability else "PASS") for _,r in x.iterrows()]
    rank=x.signal.map({"HR VALUE":2,"HR WATCH":1,"PASS":0}).fillna(0)
    return x.assign(_rank=rank).sort_values(["_rank","home_run_probability","ev_per_unit"],ascending=[False,False,False]).drop(columns="_rank").reset_index(drop=True)

def save_home_run_model(model:HomeRunModel,model_dir:str|Path)->Path:
    p=Path(model_dir)/"home_run_probability_model.joblib";p.parent.mkdir(parents=True,exist_ok=True);joblib.dump(model,p);return p
