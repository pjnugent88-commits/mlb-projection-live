from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _pct(value, signed: bool = False) -> str:
    if pd.isna(value):
        return "—"
    number = 100.0 * float(value)
    return f"{number:+.1f}%" if signed else f"{number:.1f}%"


def _num(value, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _odds(value) -> str:
    if pd.isna(value):
        return "—"
    number = int(round(float(value)))
    return f"+{number}" if number > 0 else str(number)


def _time(value) -> str:
    if pd.isna(value):
        return "Time TBD"
    stamp = pd.Timestamp(value)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    return stamp.tz_convert("America/New_York").strftime("%-I:%M %p ET")


def render_mobile_dashboard(projections: pd.DataFrame, metrics: dict, output_path: str | Path, title: str = "Daily MLB Projection Board") -> Path:
    output_path = Path(output_path); output_path.parent.mkdir(parents=True, exist_ok=True)
    date_value = metrics.get("projection_date") or (projections["game_date"].iloc[0] if not projections.empty else None)
    date_label = pd.Timestamp(date_value).strftime("%B %-d, %Y") if date_value is not None else "Current slate"
    generated = metrics.get("generated_at_utc")
    generated_label = pd.Timestamp(generated).tz_convert("America/New_York").strftime("%B %-d, %-I:%M %p ET") if generated else "Not generated"
    market_available = bool(metrics.get("market_odds_available", False))
    cards = []
    for _, row in projections.iterrows():
        signal = str(row.get("model_signal", "PASS"))
        css = "value" if "VALUE" in signal else ("lean" if "LEAN" in signal else "pass")
        weather = "Weather unavailable"
        if pd.notna(row.get("temperature_f")):
            weather = f"{_num(row.get('temperature_f'),0)}°F · wind {_num(row.get('wind_speed_mph'),0)} mph · rain {_num(row.get('precipitation_in'),2)} in"
        market = ""
        if market_available and pd.notna(row.get("best_home_odds")):
            market = f'''<div class="market">Market: away {_odds(row.get('best_away_odds'))} · home {_odds(row.get('best_home_odds'))}<br>Edge: away {_pct(row.get('away_edge'),True)} · home {_pct(row.get('home_edge'),True)} · EV: away {_pct(row.get('away_ev_per_unit'),True)} · home {_pct(row.get('home_ev_per_unit'),True)}</div>'''
        coverage = "inputs complete" if bool(row.get("starter_data_complete", False)) and bool(row.get("weather_data_complete", False)) else "some inputs pending"
        cards.append(f'''<article class="card"><div class="top"><div><h2>{escape(str(row['away_team']))} @ {escape(str(row['home_team']))}</h2><span>{_time(row.get('game_datetime'))}</span></div><b class="{css}">{escape(signal)}</b></div><p>{escape(str(row.get('away_probable_pitcher') or 'TBD'))} vs {escape(str(row.get('home_probable_pitcher') or 'TBD'))}</p><div class="grid"><div><small>Away win</small><strong>{_pct(row.get('away_win_probability'))}</strong><i>{_odds(row.get('away_fair_odds'))} fair</i></div><div><small>Home win</small><strong>{_pct(row.get('home_win_probability'))}</strong><i>{_odds(row.get('home_fair_odds'))} fair</i></div><div><small>Projected score</small><strong>{_num(row.get('expected_away_runs'))}–{_num(row.get('expected_home_runs'))}</strong><i>{_num(row.get('expected_total_runs'))} total</i></div></div><div class="context">{escape(str(row.get('venue') or 'Venue TBD'))}<br>{escape(weather)} · {escape(coverage)}</div>{market}</article>''')
    if not cards:
        cards.append('<section class="empty"><h2>No uncompleted MLB games</h2><p>The pipeline ran successfully, but this date has no eligible games remaining.</p></section>')
    metric_line = "No model fit was required for this slate."
    if metrics.get("home_win_log_loss") is not None:
        metric_line = f"Holdout log loss {_num(metrics.get('home_win_log_loss'),3)} · Brier {_num(metrics.get('home_win_brier_score'),3)} · total MAE {_num(metrics.get('total_runs_mae'),2)}"
    market_note = "Sportsbook prices connected; VALUE requires no-vig edge and positive EV." if market_available else "No sportsbook key configured; LEAN labels reflect model probability only, not market value."
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#090b0f"><title>{escape(title)}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"><style>:root{{--bg:#090b0f;--panel:#171e29;--text:#f5f7fa;--muted:#9da8b7;--line:#2a3442;--green:#66e08d;--amber:#f3bd54;--blue:#78b8ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0%,#15241d,var(--bg) 35rem);color:var(--text);font-family:'IBM Plex Sans',sans-serif}}main{{max-width:850px;margin:auto;padding:24px 15px 50px}}.badge{{color:var(--green);font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}}h1{{font:800 clamp(2.3rem,10vw,4.8rem)/.9 'Barlow Condensed';text-transform:uppercase;margin:15px 0 9px}}.sub,.notice,footer{{color:var(--muted);font-size:.78rem;line-height:1.5}}.notice{{border:1px solid #5b4b2d;background:#241f15;padding:11px;border-radius:12px;margin:16px 0}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a{{color:var(--text);text-decoration:none;background:#151c26;border:1px solid var(--line);padding:10px 12px;border-radius:10px;font-size:.76rem;font-weight:700}}.card,.empty{{background:linear-gradient(145deg,#1b232f,#121720);border:1px solid var(--line);border-radius:16px;padding:15px;margin:12px 0}}.top{{display:flex;justify-content:space-between;gap:10px}}h2{{font:700 1.45rem/1 'Barlow Condensed';margin:0}}.top span,.context,.market,p{{color:var(--muted);font-size:.72rem}}.top b{{font-size:.68rem;border:1px solid;padding:6px 8px;border-radius:999px;height:max-content}}.value{{color:var(--green)}}.lean{{color:var(--blue)}}.pass{{color:#bdc4ce}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}.grid div{{background:#0f141c;border:1px solid #202a37;border-radius:11px;padding:10px}}small,i{{display:block;color:var(--muted);font-size:.63rem;font-style:normal}}strong{{display:block;font:700 1.28rem 'Barlow Condensed';margin:3px 0}}.context,.market{{margin-top:10px}}.market{{padding-top:9px;border-top:1px solid var(--line)}}footer{{border-top:1px solid var(--line);padding-top:16px;margin-top:24px}}@media(max-width:540px){{.grid{{grid-template-columns:repeat(2,1fr)}}.grid div:last-child{{grid-column:1/-1}}}}</style></head><body><main><span class="badge">Production · real-source pipeline</span><h1>{escape(title)}</h1><div class="sub">{escape(date_label)} · generated {escape(generated_label)}<br>{escape(metric_line)}</div><div class="notice">{escape(market_note)}</div><nav><a href="projections.csv">Projection CSV</a><a href="metrics.json">Metrics</a><a href="metadata.json">Sources</a></nav>{''.join(cards)}<footer>Inputs can change with scratches, lineups, roofs, weather, and market movement. Holdout accuracy does not guarantee results or profitability.</footer></main></body></html>'''
    output_path.write_text(html, encoding="utf-8")
    return output_path
