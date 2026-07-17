from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


def _num(value, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _pct(value) -> str:
    return "—" if pd.isna(value) else f"{100 * float(value):.1f}%"


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


def render_player_props_dashboard(props: pd.DataFrame, metrics: dict, output_path: str | Path, title: str = "MLB Player Props") -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = metrics.get("generated_at_utc")
    generated_label = pd.Timestamp(generated).tz_convert("America/New_York").strftime("%B %-d, %-I:%M %p ET") if generated else "Not generated"
    projection_date = metrics.get("projection_date")
    date_label = pd.Timestamp(projection_date).strftime("%B %-d, %Y") if projection_date else "Current slate"
    market_available = bool(metrics.get("market_odds_available", False))
    cards: list[str] = []
    for _, row in props.iterrows():
        signal = str(row.get("signal", "PASS"))
        css = "value" if "VALUE" in signal else ("lean" if ("LEAN" in signal or "WATCH" in signal) else "pass")
        market = ""
        if pd.notna(row.get("over_odds")) or pd.notna(row.get("under_odds")):
            market = f'''<div class="market">Market: O {_odds(row.get('over_odds'))} ({escape(str(row.get('over_book') or '—'))}) · U {_odds(row.get('under_odds'))} ({escape(str(row.get('under_book') or '—'))})<br>EV: O {_pct(row.get('over_ev_per_unit'))} · U {_pct(row.get('under_ev_per_unit'))}</div>'''
        order = ""
        if row.get("player_type") == "batter" and pd.notna(row.get("batting_order")):
            order = f" · batting {int(float(row['batting_order']))}"
        cards.append(f'''<article class="card"><div class="top"><div><h2>{escape(str(row.get('player_name')))}</h2><span>{escape(str(row.get('away_team')))} @ {escape(str(row.get('home_team')))} · {_time(row.get('game_datetime'))}</span></div><b class="{css}">{escape(signal)}</b></div><p>{escape(str(row.get('market_label')))} · {escape(str(row.get('lineup_status') or ''))}{escape(order)}</p><div class="grid"><div><small>Projection</small><strong>{_num(row.get('projection'),2)}</strong><i>line {_num(row.get('line'),1)}</i></div><div><small>Over</small><strong>{_pct(row.get('over_probability'))}</strong><i>{_odds(row.get('over_fair_odds'))} fair</i></div><div><small>Under</small><strong>{_pct(row.get('under_probability'))}</strong><i>{_odds(row.get('under_fair_odds'))} fair</i></div></div>{market}</article>''')
    if not cards:
        cards.append('<section class="empty"><h2>No player props available</h2><p>Pitcher props require probable starters. Batter props appear after MLB posts a batting order.</p></section>')
    market_note = "Sportsbook prop prices connected; VALUE labels require edge and positive EV." if market_available else "Fair lines are active. Sportsbook EV requires a player-prop feed; LEAN labels are model-only."
    market_metrics = metrics.get("markets", {})
    metric_text = " · ".join(f"{key.replace('_',' ')} MAE {_num(value.get('mae'),2)}" for key, value in market_metrics.items())
    html = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="theme-color" content="#090b0f"><title>{escape(title)}</title><link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet"><style>:root{{--bg:#090b0f;--panel:#171e29;--text:#f5f7fa;--muted:#9da8b7;--line:#2a3442;--green:#66e08d;--amber:#f3bd54;--blue:#78b8ff}}*{{box-sizing:border-box}}body{{margin:0;background:radial-gradient(circle at 15% 0%,#17231d,var(--bg) 35rem);color:var(--text);font-family:'IBM Plex Sans',sans-serif}}main{{max-width:900px;margin:auto;padding:24px 15px 50px}}.badge{{color:var(--green);font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}}h1{{font:800 clamp(2.3rem,10vw,4.8rem)/.9 'Barlow Condensed';text-transform:uppercase;margin:15px 0 9px}}.sub,.notice,footer{{color:var(--muted);font-size:.78rem;line-height:1.5}}.notice{{border:1px solid #5b4b2d;background:#241f15;padding:11px;border-radius:12px;margin:16px 0}}nav{{display:flex;gap:8px;flex-wrap:wrap}}nav a{{color:var(--text);text-decoration:none;background:#151c26;border:1px solid var(--line);padding:10px 12px;border-radius:10px;font-size:.76rem;font-weight:700}}.card,.empty{{background:linear-gradient(145deg,#1b232f,#121720);border:1px solid var(--line);border-radius:16px;padding:15px;margin:12px 0}}.top{{display:flex;justify-content:space-between;gap:10px}}h2{{font:700 1.45rem/1 'Barlow Condensed';margin:0}}.top span,p,.market{{color:var(--muted);font-size:.72rem}}.top b{{font-size:.68rem;border:1px solid;padding:6px 8px;border-radius:999px;height:max-content}}.value{{color:var(--green)}}.lean{{color:var(--blue)}}.pass{{color:#bdc4ce}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}.grid div{{background:#0f141c;border:1px solid #202a37;border-radius:11px;padding:10px}}small,i{{display:block;color:var(--muted);font-size:.63rem;font-style:normal}}strong{{display:block;font:700 1.28rem 'Barlow Condensed';margin:3px 0}}.market{{margin-top:10px;padding-top:9px;border-top:1px solid var(--line)}}footer{{border-top:1px solid var(--line);padding-top:16px;margin-top:24px}}@media(max-width:540px){{.grid{{grid-template-columns:repeat(2,1fr)}}.grid div:last-child{{grid-column:1/-1}}}}</style></head><body><main><span class="badge">Stage 3 · real Statcast player models</span><h1>{escape(title)}</h1><div class="sub">{escape(date_label)} · generated {escape(generated_label)}<br>{escape(metric_text)}</div><div class="notice">{escape(market_note)}</div><nav><a href="index.html">Game Board</a><a href="player_props.csv">Props CSV</a><a href="prop_metrics.json">Prop Metrics</a><a href="prop_metadata.json">Sources</a></nav>{''.join(cards)}<footer>Lineups, scratches, opener usage, pitch limits, weather, and prices can change. A fair probability is not proof of a profitable wager.</footer></main></body></html>'''
    output_path.write_text(html, encoding="utf-8")
    return output_path
