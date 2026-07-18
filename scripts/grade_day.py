from __future__ import annotations
import argparse, io, json, math, os, zipfile
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np, pandas as pd, requests

GH='https://api.github.com'; SCHED='https://statsapi.mlb.com/api/v1/schedule'; BOX='https://statsapi.mlb.com/api/v1/game/{}/boxscore'
FINAL={'Final','Game Over','Completed Early'}

def get(url,**kw):
 r=requests.get(url,timeout=90,**kw); r.raise_for_status(); return r

def ghh():
 h={'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'}
 if os.getenv('GITHUB_TOKEN'): h['Authorization']='Bearer '+os.environ['GITHUB_TOKEN']
 return h

def datearg(v): return str(pd.Timestamp(v).date()) if v else str(datetime.now(ZoneInfo('America/New_York')).date()-timedelta(days=1))

def artifact(repo,workflow,date,run_id=None):
 ids=[run_id] if run_id else [x['id'] for x in get(f'{GH}/repos/{repo}/actions/workflows/{workflow}/runs',headers=ghh(),params={'status':'success','per_page':100}).json()['workflow_runs']]
 name=f'mlb-projections-{date}'
 for rid in ids:
  for a in get(f'{GH}/repos/{repo}/actions/runs/{rid}/artifacts',headers=ghh(),params={'per_page':100}).json()['artifacts']:
   if a['name']==name and not a['expired']:
    return int(rid),get(a['archive_download_url'],headers=ghh()).content
 raise RuntimeError(f'No {name} artifact found')

def actual_games(date):
 p=get(SCHED,params={'sportId':1,'date':date,'hydrate':'linescore,team'}).json(); rows=[]
 for d in p.get('dates',[]):
  for g in d.get('games',[]):
   st=g['status']; final=st.get('abstractGameState')=='Final' or st.get('detailedState') in FINAL
   rows.append({'game_pk':g['gamePk'],'is_final':final,'actual_status':st.get('detailedState'),'actual_away_runs':g['teams']['away'].get('score') if final else np.nan,'actual_home_runs':g['teams']['home'].get('score') if final else np.nan})
 return pd.DataFrame(rows)

def aprofit(o): return float(o)/100 if float(o)>0 else 100/abs(float(o))

def grade_games(p,r):
 p=p.copy(); p['best_ev_per_unit']=pd.to_numeric(p['best_ev_per_unit'],errors='coerce'); p=p.sort_values(['game_pk','best_ev_per_unit'],ascending=[1,0]).drop_duplicates('game_pk')
 x=p.merge(r,on='game_pk',how='left'); x=x[x.is_final.fillna(False)].copy(); x['actual_home_win']=(x.actual_home_runs>x.actual_away_runs).astype(int); x['pick_home']=(x.home_win_probability>=.5).astype(int); x['pick_correct']=x.pick_home.eq(x.actual_home_win)
 x['actual_total_runs']=x.actual_home_runs+x.actual_away_runs; x['total_error']=(x.expected_total_runs-x.actual_total_runs).abs(); x['home_error']=(x.expected_home_runs-x.actual_home_runs).abs(); x['away_error']=(x.expected_away_runs-x.actual_away_runs).abs()
 x['bet_side']=x.model_signal.astype(str).str.extract(r'^(HOME|AWAY)'); x['bet_result']='NO BET'; x['profit_units']=np.nan
 for i,z in x.iterrows():
  side=z.bet_side
  if side not in {'HOME','AWAY'}: continue
  win=(z.actual_home_win==1) if side=='HOME' else (z.actual_home_win==0); o=z['best_home_odds' if side=='HOME' else 'best_away_odds']; x.at[i,'bet_result']='WIN' if win else 'LOSS'; x.at[i,'profit_units']=aprofit(o) if win and pd.notna(o) else (-1 if pd.notna(o) else np.nan)
 b=x[x.bet_side.notna()]; bo=b[b.profit_units.notna()]; y=x.actual_home_win.to_numpy(); pr=np.clip(x.home_win_probability.to_numpy(),1e-6,1-1e-6)
 return x,{'graded_games':len(x),'winner_correct':int(x.pick_correct.sum()),'winner_incorrect':int((~x.pick_correct).sum()),'winner_accuracy':float(x.pick_correct.mean()),'brier_score':float(np.mean((pr-y)**2)),'log_loss':float(-np.mean(y*np.log(pr)+(1-y)*np.log(1-pr))),'home_runs_mae':float(x.home_error.mean()),'away_runs_mae':float(x.away_error.mean()),'total_runs_mae':float(x.total_error.mean()),'value_bets':len(b),'value_wins':int(b.bet_result.eq('WIN').sum()),'value_losses':int(b.bet_result.eq('LOSS').sum()),'profit_units':float(bo.profit_units.sum()),'roi':float(bo.profit_units.sum()/len(bo)) if len(bo) else None}

def outs(ip):
 try:
  a,b=str(ip).split('.'); return int(a)*3+int(b)
 except: return np.nan

def player_actuals(game_ids):
 rows=[]
 for gid in sorted(set(game_ids)):
  p=get(BOX.format(gid)).json()
  for side in ('away','home'):
   for z in p.get('teams',{}).get(side,{}).get('players',{}).values():
    pid=z.get('person',{}).get('id'); s=z.get('stats',{}); bat=s.get('batting') or {}; pit=s.get('pitching') or {}
    if pid is None: continue
    ab=bool(bat) and any(k in bat for k in ('plateAppearances','atBats','hits')); ap=bool(pit) and any(k in pit for k in ('inningsPitched','battersFaced','strikeOuts'))
    rows.append({'game_pk':gid,'player_id':pid,'actual_batter_hits':bat.get('hits',0) if ab else np.nan,'actual_batter_total_bases':bat.get('totalBases',0) if ab else np.nan,'actual_batter_home_runs':bat.get('homeRuns',0) if ab else np.nan,'actual_pitcher_strikeouts':pit.get('strikeOuts',0) if ap else np.nan,'actual_pitcher_outs':pit.get('outs',outs(pit.get('inningsPitched'))) if ap else np.nan,'actual_pitcher_hits_allowed':pit.get('hits',0) if ap else np.nan})
 return pd.DataFrame(rows).groupby(['game_pk','player_id'],as_index=False).max() if rows else pd.DataFrame(columns=['game_pk','player_id'])

def sm(g):
 d=g[g.pick_result.isin(['WIN','LOSS'])]; o=g[g.profit_units.notna()]
 return {'picks':len(g),'wins':int(d.pick_result.eq('WIN').sum()),'losses':int(d.pick_result.eq('LOSS').sum()),'pushes':int(g.pick_result.eq('PUSH').sum()),'hit_rate':float(d.pick_result.eq('WIN').mean()) if len(d) else None,'profit_units':float(o.profit_units.sum()) if len(o) else None,'roi':float(o.profit_units.sum()/len(o)) if len(o) else None}

def grade_props(p,r,a):
 finals=set(r.loc[r.is_final.fillna(False),'game_pk']); x=p[p.game_pk.isin(finals)].copy(); x['player_id']=pd.to_numeric(x.player_id).astype('Int64'); x=x.merge(a,on=['game_pk','player_id'],how='left')
 m={'batter_hits':'actual_batter_hits','batter_total_bases':'actual_batter_total_bases','batter_home_runs':'actual_batter_home_runs','pitcher_strikeouts':'actual_pitcher_strikeouts','pitcher_outs':'actual_pitcher_outs','pitcher_hits_allowed':'actual_pitcher_hits_allowed'}
 x['actual']=[z.get(m[z.market_key],np.nan) for _,z in x.iterrows()]; x['graded']=x.actual.notna(); x['projection_error']=(x.projection-x.actual).abs(); x['pick_side']=x.signal.astype(str).str.extract(r'^(OVER|UNDER)'); x['pick_result']='NO PICK'; x['profit_units']=np.nan
 for i,z in x.iterrows():
  if not z.graded: x.at[i,'pick_result']='VOID'; continue
  if z.pick_side not in {'OVER','UNDER'}: continue
  if z.actual==z.line: res='PUSH'
  else: res='WIN' if (z.pick_side=='OVER')==(z.actual>z.line) else 'LOSS'
  x.at[i,'pick_result']=res; o=z['over_odds' if z.pick_side=='OVER' else 'under_odds']; x.at[i,'profit_units']=aprofit(o) if res=='WIN' and pd.notna(o) else (-1 if res=='LOSS' and pd.notna(o) else (0 if res=='PUSH' and pd.notna(o) else np.nan))
 g=x[x.graded]; v=g[g.signal.astype(str).str.contains('VALUE')]; by={}
 for k,q in g.groupby('market_key'):
  by[k]={'graded_rows':len(q),'projection_mae':float(q.projection_error.mean()),'value':sm(q[q.signal.astype(str).str.contains('VALUE')])}
 return x,{'published_rows':len(p),'graded_rows':len(g),'void_rows':int((~x.graded).sum()),'projection_mae':float(g.projection_error.mean()),'value':sm(v),'by_market':by}

def pc(v): return '—' if v is None or pd.isna(v) else f'{100*v:.1f}%'
def nu(v): return '—' if v is None or pd.isna(v) else f'{v:.2f}'
def report(date,rid,g,p):
 v=p['value']; L=[f'# MLB model grading — {date}','',f'Projection workflow run: `{rid}`','','## Game model','',f"- Winner picks: **{g['winner_correct']}-{g['winner_incorrect']}** ({pc(g['winner_accuracy'])})",f"- Brier / log loss: **{g['brier_score']:.3f} / {g['log_loss']:.3f}**",f"- Total-runs MAE: **{g['total_runs_mae']:.2f}**",f"- Moneyline values: **{g['value_wins']}-{g['value_losses']}**, **{g['profit_units']:.2f}u**, ROI **{pc(g['roi'])}**",'','## Player props','',f"- VALUE selections: **{v['wins']}-{v['losses']}-{v['pushes']}** ({pc(v['hit_rate'])})",f"- VALUE profit: **{nu(v['profit_units'])}u**, ROI **{pc(v['roi'])}**",'','| Market | MAE | W-L-P | Hit | Profit | ROI |','|---|---:|---:|---:|---:|---:|']
 for k,z in p['by_market'].items():
  s=z['value']; L.append(f"| {k.replace('_',' ').title()} | {z['projection_mae']:.2f} | {s['wins']}-{s['losses']}-{s['pushes']} | {pc(s['hit_rate'])} | {nu(s['profit_units'])}u | {pc(s['roi'])} |")
 L+=['','One slate is a small sample. One unit is risked per published VALUE row at its recorded price; correlated props are counted separately.']; return '\n'.join(L)

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--date'); ap.add_argument('--repo',default=os.getenv('GITHUB_REPOSITORY','pjnugent88-commits/mlb-projection-live')); ap.add_argument('--workflow',default='daily.yml'); ap.add_argument('--run-id',type=int); ap.add_argument('--output-dir',default='grading_output'); a=ap.parse_args(); date=datearg(a.date); rid,data=artifact(a.repo,a.workflow,date,a.run_id); out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); art=out/'artifact'; art.mkdir(exist_ok=True); zipfile.ZipFile(io.BytesIO(data)).extractall(art)
 p=pd.read_csv(art/'projections.csv'); pp=pd.read_csv(art/'player_props.csv'); r=actual_games(date); gd,gm=grade_games(p,r); ids=set(pp.game_pk)&set(r.loc[r.is_final.fillna(False),'game_pk']); act=player_actuals(ids); pd_,pm=grade_props(pp,r,act); summary={'date':date,'generated_at_utc':pd.Timestamp.now(tz='UTC').isoformat(),'projection_run_id':rid,'game_model':gm,'player_props':pm}; (out/'grading_summary.json').write_text(json.dumps(summary,indent=2)); gd.to_csv(out/'graded_games.csv',index=False); pd_.to_csv(out/'graded_player_props.csv',index=False); act.to_csv(out/'official_player_actuals.csv',index=False); text=report(date,rid,gm,pm); (out/'grading_report.md').write_text(text); print(text)
if __name__=='__main__': main()
