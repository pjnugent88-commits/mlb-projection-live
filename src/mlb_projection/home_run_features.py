from __future__ import annotations

import math
import numpy as np
import pandas as pd

from .prop_constants import EVENT_OUTS, TEAM_CODE_MAP

BVP_PRIOR_PA = 40.0
BVP_PRIOR_HR_RATE = 0.03
BVP_PRIOR_BARREL_RATE = 0.06
EXPECTED_PA = {1:4.75,2:4.65,3:4.55,4:4.45,5:4.30,6:4.15,7:4.00,8:3.85,9:3.70}


def _n(df: pd.DataFrame, col: str) -> pd.Series:
    return pd.to_numeric(df[col], errors="coerce") if col in df else pd.Series(np.nan, index=df.index)


def _q90(s: pd.Series) -> float:
    s = pd.to_numeric(s, errors="coerce").dropna()
    return float(s.quantile(.9)) if len(s) else math.nan


def prepare_hr_pitches(pitches: pd.DataFrame) -> pd.DataFrame:
    f = pitches.copy()
    if f.empty: return f
    for c in ("home_team","away_team"):
        if c in f: f[c] = f[c].replace(TEAM_CODE_MAP)
    f["game_date"] = pd.to_datetime(f["game_date"], utc=True, errors="coerce")
    if "game_type" in f: f = f[f.game_type.isin(["R","F","D","L","W"])].copy()
    f["bat_team"] = np.where(f.inning_topbot.eq("Top"), f.away_team, f.home_team)
    f["field_team"] = np.where(f.inning_topbot.eq("Top"), f.home_team, f.away_team)
    f["event"] = f.get("events", pd.Series(index=f.index,dtype=object)).fillna("")
    f["pa"] = f.event.ne(""); f["hr"] = f.event.eq("home_run")
    f["k"] = f.event.isin(["strikeout","strikeout_double_play"])
    f["bb"] = f.event.isin(["walk","intent_walk"])
    f["outs"] = f.event.map(EVENT_OUTS).fillna(0)
    f["ev"] = _n(f,"launch_speed"); f["la"] = _n(f,"launch_angle")
    f["bbe"] = f.ev.notna(); f["barrel"] = _n(f,"launch_speed_angle").eq(6)
    f["hard_air"] = f.ev.ge(95) & f.la.ge(10)
    f["sweet"] = f.la.between(20,35,inclusive="both")
    f["fly"] = f.la.ge(20)
    ex = _n(f,"estimated_woba_using_speedangle"); ac = _n(f,"woba_value")
    f["xwoba"] = ex.where(ex.notna(), ac)
    sort = [c for c in ["game_pk","field_team","inning","at_bat_number","pitch_number"] if c in f]
    st = f.sort_values(sort).groupby(["game_pk","field_team"],as_index=False).first()[["game_pk","field_team","pitcher"]]
    return f.merge(st.rename(columns={"pitcher":"starter_id"}),on=["game_pk","field_team"],how="left")


def game_tables(pitches: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    f = prepare_hr_pitches(pitches)
    if f.empty: return pd.DataFrame(),pd.DataFrame(),pd.DataFrame()
    order = f[f.pa].groupby(["game_pk","bat_team","batter"],as_index=False).at_bat_number.min()
    order["batting_order"] = order.groupby(["game_pk","bat_team"]).at_bat_number.rank(method="first")
    b = f.groupby(["game_pk","game_date","bat_team","field_team","home_team","away_team","batter"],as_index=False).agg(
        home_runs=("hr","sum"),pa=("pa","sum"),strikeouts=("k","sum"),walks=("bb","sum"),bbe=("bbe","sum"),
        barrels=("barrel","sum"),hard_air=("hard_air","sum"),sweet=("sweet","sum"),xwoba=("xwoba","mean"),
        ev90=("ev",_q90),batter_hand=("stand","last"),opposing_starter_id=("starter_id","first"),pitcher_hand=("p_throws","first"))
    b = b.rename(columns={"bat_team":"team","field_team":"opponent","batter":"player_id"})
    b = b.merge(order.rename(columns={"bat_team":"team","batter":"player_id"})[["game_pk","team","player_id","batting_order"]],on=["game_pk","team","player_id"],how="left")
    b["target_hr"] = b.home_runs.gt(0).astype(int); b["is_home"] = b.team.eq(b.home_team).astype(int)
    b = b[b.batting_order.between(1,9,inclusive="both")].copy()
    s = f[f.pitcher.eq(f.starter_id)].groupby(["game_pk","game_date","field_team","pitcher"],as_index=False).agg(
        hr_allowed=("hr","sum"),bf=("pa","sum"),strikeouts=("k","sum"),walks=("bb","sum"),bbe=("bbe","sum"),
        barrels=("barrel","sum"),hard_air=("hard_air","sum"),fly=("fly","sum"),xwoba=("xwoba","mean"),
        ev90=("ev",_q90),outs=("outs","sum"),pitcher_hand=("p_throws","last"))
    s = s.rename(columns={"field_team":"team","pitcher":"player_id"}); s = s[s.bf.ge(3)].copy()
    pair = b[["game_pk","game_date","player_id","opposing_starter_id","pa","home_runs","barrels"]].copy()
    return b,s,pair


def _roll(df: pd.DataFrame, group: str, col: str, window: int, how: str="sum") -> pd.Series:
    o = df.sort_values([group,"game_date","game_pk"])
    def calc(s):
        r = pd.to_numeric(s,errors="coerce").shift(1).rolling(window,min_periods=1)
        return r.sum() if how=="sum" else r.mean()
    return o.groupby(group,group_keys=False)[col].transform(calc).reindex(df.index)


def _rate(num: pd.Series, den: pd.Series) -> pd.Series: return num/den.clip(lower=1)


def _batter_roll(b: pd.DataFrame, window: int, suffix: str) -> pd.DataFrame:
    pa=_roll(b,"player_id","pa",window); bbe=_roll(b,"player_id","bbe",window)
    b[f"batter_hr_pa_{suffix}"]=_rate(_roll(b,"player_id","home_runs",window),pa)
    b[f"batter_barrel_pa_{suffix}"]=_rate(_roll(b,"player_id","barrels",window),pa)
    b[f"batter_barrel_bbe_{suffix}"]=_rate(_roll(b,"player_id","barrels",window),bbe)
    b[f"batter_hard_air_pa_{suffix}"]=_rate(_roll(b,"player_id","hard_air",window),pa)
    b[f"batter_sweet_pa_{suffix}"]=_rate(_roll(b,"player_id","sweet",window),pa)
    b[f"batter_k_rate_{suffix}"]=_rate(_roll(b,"player_id","strikeouts",window),pa)
    b[f"batter_bb_rate_{suffix}"]=_rate(_roll(b,"player_id","walks",window),pa)
    b[f"batter_xwoba_{suffix}"]=_roll(b,"player_id","xwoba",window,"mean")
    b[f"batter_ev90_{suffix}"]=_roll(b,"player_id","ev90",window,"mean")
    return b


def _pitcher_roll(p: pd.DataFrame, window: int) -> pd.DataFrame:
    bf=_roll(p,"player_id","bf",window); bbe=_roll(p,"player_id","bbe",window)
    p["pitcher_hr_bf"]=_rate(_roll(p,"player_id","hr_allowed",window),bf)
    p["pitcher_barrel_bf"]=_rate(_roll(p,"player_id","barrels",window),bf)
    p["pitcher_hard_air_bf"]=_rate(_roll(p,"player_id","hard_air",window),bf)
    p["pitcher_fly_rate"]=_rate(_roll(p,"player_id","fly",window),bbe)
    p["pitcher_k_rate"]=_rate(_roll(p,"player_id","strikeouts",window),bf)
    p["pitcher_bb_rate"]=_rate(_roll(p,"player_id","walks",window),bf)
    p["pitcher_xwoba"]=_roll(p,"player_id","xwoba",window,"mean")
    p["pitcher_ev90"]=_roll(p,"player_id","ev90",window,"mean")
    p["pitcher_outs"]=_roll(p,"player_id","outs",window,"mean")
    return p


def _bvp_shift(b: pd.DataFrame) -> pd.DataFrame:
    o=b.sort_values(["player_id","opposing_starter_id","game_date","game_pk"]).copy(); g=o.groupby(["player_id","opposing_starter_id"],group_keys=False)
    o["bvp_pa"]=g.pa.transform(lambda s:s.shift(1).fillna(0).cumsum())
    o["bvp_hr"]=g.home_runs.transform(lambda s:s.shift(1).fillna(0).cumsum())
    o["bvp_barrels"]=g.barrels.transform(lambda s:s.shift(1).fillna(0).cumsum())
    o["bvp_hr_shrunk"]=(o.bvp_hr+BVP_PRIOR_PA*BVP_PRIOR_HR_RATE)/(o.bvp_pa+BVP_PRIOR_PA)
    o["bvp_barrel_shrunk"]=(o.bvp_barrels+BVP_PRIOR_PA*BVP_PRIOR_BARREL_RATE)/(o.bvp_pa+BVP_PRIOR_PA)
    o["bvp_reliability"]=o.bvp_pa/(o.bvp_pa+BVP_PRIOR_PA)
    return o.sort_index()


def context_frame(context: pd.DataFrame|None) -> pd.DataFrame:
    cols=["game_pk","park_hr_factor","temperature_f","humidity_pct","wind_speed_mph","precipitation_in","roof_control_factor"]
    d={"park_hr_factor":1.0,"temperature_f":72.0,"humidity_pct":55.0,"wind_speed_mph":7.0,"precipitation_in":0.0,"roof_control_factor":0.0}
    if context is None or context.empty: return pd.DataFrame(columns=cols)
    x=context.copy()
    for c in cols:
        if c not in x: x[c]=d.get(c,np.nan)
    return x[cols].drop_duplicates("game_pk",keep="last")


def build_training(pitches: pd.DataFrame, context: pd.DataFrame|None, long_window=60, recent_window=15, pitcher_window=15):
    b,p,pair=game_tables(pitches)
    if b.empty or p.empty: return pd.DataFrame(),b,p,pair
    b=_batter_roll(b,long_window,"long"); b=_batter_roll(b,recent_window,"recent"); b=_bvp_shift(b)
    b["prior_batter_games"]=b.sort_values(["player_id","game_date","game_pk"]).groupby("player_id").cumcount().reindex(b.index)
    p=_pitcher_roll(p,pitcher_window); p["prior_pitcher_starts"]=p.sort_values(["player_id","game_date","game_pk"]).groupby("player_id").cumcount().reindex(p.index)
    pc=["game_pk","player_id","pitcher_hand","prior_pitcher_starts","pitcher_hr_bf","pitcher_barrel_bf","pitcher_hard_air_bf","pitcher_fly_rate","pitcher_k_rate","pitcher_bb_rate","pitcher_xwoba","pitcher_ev90","pitcher_outs"]
    pj=p[pc].rename(columns={"player_id":"opposing_starter_id","pitcher_hand":"opposing_pitcher_hand"})
    b["opposing_starter_id"]=pd.to_numeric(b.opposing_starter_id,errors="coerce").astype("Int64"); pj["opposing_starter_id"]=pd.to_numeric(pj.opposing_starter_id,errors="coerce").astype("Int64")
    x=b.merge(pj,on=["game_pk","opposing_starter_id"],how="left")
    x["expected_pa"]=x.batting_order.round().astype("Int64").map(EXPECTED_PA)
    x["same_hand"]=((x.batter_hand.isin(["L","R"]))&(x.opposing_pitcher_hand.isin(["L","R"]))&x.batter_hand.eq(x.opposing_pitcher_hand)).astype(int)
    x=x.merge(context_frame(context),on="game_pk",how="left")
    for c,v in {"park_hr_factor":1.0,"temperature_f":72.0,"humidity_pct":55.0,"wind_speed_mph":7.0,"precipitation_in":0.0,"roof_control_factor":0.0}.items(): x[c]=pd.to_numeric(x.get(c),errors="coerce").fillna(v)
    return x[(x.prior_batter_games>=10)&(x.prior_pitcher_starts>=3)].copy(),b,p,pair


def _live_batter(b: pd.DataFrame,pid:int,long:int,recent:int)->dict:
    h=b[pd.to_numeric(b.player_id,errors="coerce").eq(pid)].sort_values(["game_date","game_pk"])
    def f(w,s):
        z=h.tail(w); pa=max(float(z.pa.sum()),1); bbe=max(float(z.bbe.sum()),1)
        return {f"batter_hr_pa_{s}":float(z.home_runs.sum()/pa),f"batter_barrel_pa_{s}":float(z.barrels.sum()/pa),f"batter_barrel_bbe_{s}":float(z.barrels.sum()/bbe),f"batter_hard_air_pa_{s}":float(z.hard_air.sum()/pa),f"batter_sweet_pa_{s}":float(z.sweet.sum()/pa),f"batter_k_rate_{s}":float(z.strikeouts.sum()/pa),f"batter_bb_rate_{s}":float(z.walks.sum()/pa),f"batter_xwoba_{s}":float(z.xwoba.mean()),f"batter_ev90_{s}":float(z.ev90.mean())}
    out=f(long,"long");out.update(f(recent,"recent"));out["batter_hand"]=h.batter_hand.dropna().iloc[-1] if h.batter_hand.notna().any() else None;return out


def _live_pitcher(p:pd.DataFrame,pid:int,w:int)->dict:
    h=p[pd.to_numeric(p.player_id,errors="coerce").eq(pid)].sort_values(["game_date","game_pk"]).tail(w)
    if h.empty:return {}
    bf=max(float(h.bf.sum()),1);bbe=max(float(h.bbe.sum()),1)
    return {"pitcher_hr_bf":float(h.hr_allowed.sum()/bf),"pitcher_barrel_bf":float(h.barrels.sum()/bf),"pitcher_hard_air_bf":float(h.hard_air.sum()/bf),"pitcher_fly_rate":float(h.fly.sum()/bbe),"pitcher_k_rate":float(h.strikeouts.sum()/bf),"pitcher_bb_rate":float(h.walks.sum()/bf),"pitcher_xwoba":float(h.xwoba.mean()),"pitcher_ev90":float(h.ev90.mean()),"pitcher_outs":float(h.outs.mean()),"opposing_pitcher_hand":h.pitcher_hand.dropna().iloc[-1] if h.pitcher_hand.notna().any() else None}


def _live_bvp(pair:pd.DataFrame,bid:int,pid:int)->dict:
    h=pair[pd.to_numeric(pair.player_id,errors="coerce").eq(bid)&pd.to_numeric(pair.opposing_starter_id,errors="coerce").eq(pid)]
    pa=float(h.pa.sum()) if len(h) else 0;hr=float(h.home_runs.sum()) if len(h) else 0;br=float(h.barrels.sum()) if len(h) else 0
    return {"bvp_pa":pa,"bvp_hr":hr,"bvp_barrels":br,"bvp_hr_shrunk":(hr+BVP_PRIOR_PA*BVP_PRIOR_HR_RATE)/(pa+BVP_PRIOR_PA),"bvp_barrel_shrunk":(br+BVP_PRIOR_PA*BVP_PRIOR_BARREL_RATE)/(pa+BVP_PRIOR_PA),"bvp_reliability":pa/(pa+BVP_PRIOR_PA)}


def build_live(lineups,slate,batter_history,pitcher_history,pair_history,weather,venues,long_window=60,recent_window=15,pitcher_window=15):
    if lineups is None or lineups.empty or slate.empty:return pd.DataFrame()
    ctx=slate[["game_pk","venue"]].copy()
    if weather is not None and not weather.empty:
        cc=[c for c in ["game_pk","temperature_f","humidity_pct","wind_speed_mph","precipitation_in","roof_control_factor"] if c in weather];ctx=ctx.merge(weather[cc].drop_duplicates("game_pk"),on="game_pk",how="left")
    if "park_hr_factor" in slate:ctx=ctx.merge(slate[["game_pk","park_hr_factor"]].drop_duplicates("game_pk"),on="game_pk",how="left")
    for c,v in {"park_hr_factor":1.0,"temperature_f":72.0,"humidity_pct":55.0,"wind_speed_mph":7.0,"precipitation_in":0.0,"roof_control_factor":0.0}.items():
        if c not in ctx:ctx[c]=v
        ctx[c]=pd.to_numeric(ctx[c],errors="coerce").fillna(v)
    ctx=ctx.set_index("game_pk");games=slate.set_index("game_pk");rows=[]
    for _,l in lineups.iterrows():
        gpk=int(l.game_pk)
        if gpk not in games.index:continue
        g=games.loc[gpk];side=str(l.side);raw=g.get("home_probable_pitcher_id" if side=="away" else "away_probable_pitcher_id")
        if pd.isna(raw):continue
        bid=int(l.player_id);pid=int(float(raw));order=int(float(l.get("batting_order",9)))
        r={"game_pk":gpk,"game_datetime":g.get("game_datetime"),"away_team":g.get("away_team"),"home_team":g.get("home_team"),"team":l.get("team"),"opponent":g.get("home_team" if side=="away" else "away_team"),"is_home":int(side=="home"),"player_id":bid,"player_name":l.get("player_name") or str(bid),"lineup_status":l.get("lineup_status","confirmed"),"batting_order":float(order),"expected_pa":EXPECTED_PA.get(order,3.7),"opposing_starter_id":pid}
        r.update(_live_batter(batter_history,bid,long_window,recent_window));r.update(_live_pitcher(pitcher_history,pid,pitcher_window));r.update(_live_bvp(pair_history,bid,pid));r["same_hand"]=int(r.get("batter_hand") in ["L","R"] and r.get("opposing_pitcher_hand") in ["L","R"] and r.get("batter_hand")==r.get("opposing_pitcher_hand"));r.update(ctx.loc[gpk].to_dict());rows.append(r)
    return pd.DataFrame(rows)
