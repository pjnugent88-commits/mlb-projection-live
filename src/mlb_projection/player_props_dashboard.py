from __future__ import annotations

from html import escape
from pathlib import Path

import pandas as pd


MARKET_SECTIONS = [
    ("pitcher_strikeouts", "Pitcher Strikeouts", "pitcher"),
    ("pitcher_outs", "Pitcher Outs Recorded", "pitcher"),
    ("pitcher_hits_allowed", "Pitcher Hits Allowed", "pitcher"),
    ("batter_hits", "Batter Hits", "batter"),
    ("batter_total_bases", "Batter Total Bases", "batter"),
    ("batter_home_runs", "Batter Home Runs", "batter"),
]


def _num(value, digits: int = 1) -> str:
    return "—" if pd.isna(value) else f"{float(value):.{digits}f}"


def _pct(value, signed: bool = False) -> str:
    if pd.isna(value):
        return "—"
    number = 100 * float(value)
    return f"{number:+.1f}%" if signed else f"{number:.1f}%"


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


def _market_detail(row: pd.Series) -> str:
    if pd.isna(row.get("over_odds")) and pd.isna(row.get("under_odds")):
        return ""
    return f'''<div class="market">
        <div><small>Sportsbook</small><strong>O {_odds(row.get("over_odds"))}</strong><i>{escape(str(row.get("over_book") or "—"))}</i></div>
        <div><small>Sportsbook</small><strong>U {_odds(row.get("under_odds"))}</strong><i>{escape(str(row.get("under_book") or "—"))}</i></div>
        <div><small>No-vig / edge</small><strong>{_pct(row.get("over_no_vig_probability"))}</strong><i>O edge {_pct(row.get("over_edge"), True)}</i></div>
        <div><small>No-vig / edge</small><strong>{_pct(row.get("under_no_vig_probability"))}</strong><i>U edge {_pct(row.get("under_edge"), True)}</i></div>
        <div><small>Expected value</small><strong>{_pct(row.get("over_ev_per_unit"), True)}</strong><i>Over per unit</i></div>
        <div><small>Expected value</small><strong>{_pct(row.get("under_ev_per_unit"), True)}</strong><i>Under per unit</i></div>
    </div>'''


def _render_card(row: pd.Series) -> str:
    signal = str(row.get("signal", "PASS"))
    css = "value" if "VALUE" in signal else ("lean" if ("LEAN" in signal or "WATCH" in signal) else "pass")
    order = ""
    if row.get("player_type") == "batter" and pd.notna(row.get("batting_order")):
        order = f" · batting {int(float(row['batting_order']))}"
    confirmed = str(row.get("lineup_status") or "")
    player = escape(str(row.get("player_name") or "Unknown player"))
    away = escape(str(row.get("away_team") or "—"))
    home = escape(str(row.get("home_team") or "—"))
    market_key = escape(str(row.get("market_key") or ""))
    player_type = escape(str(row.get("player_type") or ""))
    is_value = "true" if "VALUE" in signal else "false"
    search_text = escape(f"{row.get('player_name','')} {row.get('away_team','')} {row.get('home_team','')} {row.get('market_label','')}".lower())
    return f'''<article class="card" data-market="{market_key}" data-player-type="{player_type}" data-value="{is_value}" data-search="{search_text}">
        <div class="top">
            <div><h2>{player}</h2><span>{away} @ {home} · {_time(row.get("game_datetime"))}</span></div>
            <b class="{css}">{escape(signal)}</b>
        </div>
        <p>{escape(str(row.get("market_label") or ""))} · {escape(confirmed)}{escape(order)}</p>
        <div class="grid">
            <div><small>Projection</small><strong>{_num(row.get("projection"), 2)}</strong><i>line {_num(row.get("line"), 1)}</i></div>
            <div><small>Over</small><strong>{_pct(row.get("over_probability"))}</strong><i>{_odds(row.get("over_fair_odds"))} fair</i></div>
            <div><small>Under</small><strong>{_pct(row.get("under_probability"))}</strong><i>{_odds(row.get("under_fair_odds"))} fair</i></div>
        </div>
        {_market_detail(row)}
    </article>'''


def render_player_props_dashboard(
    props: pd.DataFrame,
    metrics: dict,
    output_path: str | Path,
    title: str = "MLB Player Props",
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    generated = metrics.get("generated_at_utc")
    generated_label = (
        pd.Timestamp(generated).tz_convert("America/New_York").strftime("%B %-d, %-I:%M %p ET")
        if generated
        else "Not generated"
    )
    projection_date = metrics.get("projection_date")
    date_label = pd.Timestamp(projection_date).strftime("%B %-d, %Y") if projection_date else "Current slate"
    market_available = bool(metrics.get("market_odds_available", False))

    total_displayed = int(len(props))
    pitcher_count = int(props["player_type"].eq("pitcher").sum()) if not props.empty else 0
    batter_count = int(props["player_type"].eq("batter").sum()) if not props.empty else 0
    value_count = int(props["signal"].astype(str).str.contains("VALUE").sum()) if not props.empty else 0

    filter_specs = [
        ("all", "All", total_displayed),
        ("value", "Best Values", value_count),
        ("pitcher", "Pitcher Props", pitcher_count),
        ("batter", "Batter Props", batter_count),
    ]
    for market_key, label, _ in MARKET_SECTIONS:
        count = int(props["market_key"].eq(market_key).sum()) if not props.empty else 0
        filter_specs.append((market_key, label, count))
    filter_buttons = "".join(
        f'<button type="button" data-filter="{escape(key)}"><span>{escape(label)}</span><b>{count}</b></button>'
        for key, label, count in filter_specs
    )

    sections: list[str] = []
    for market_key, heading, player_type in MARKET_SECTIONS:
        market_rows = props[props["market_key"].eq(market_key)] if not props.empty else pd.DataFrame()
        cards = "".join(_render_card(row) for _, row in market_rows.iterrows())
        if cards:
            sections.append(
                f'''<section class="market-section" data-section-market="{market_key}" data-section-type="{player_type}">
                    <div class="section-head"><div><span>{escape(player_type)} market</span><h2>{escape(heading)}</h2></div><b>{len(market_rows)}</b></div>
                    <div class="cards">{cards}</div>
                </section>'''
            )

    if not sections:
        sections.append(
            '<section class="empty"><h2>No player props available</h2><p>Pitcher props require probable starters. Batter props appear after MLB posts a batting order.</p></section>'
        )

    if market_available:
        market_note = "Sportsbook player-prop prices are connected. VALUE labels require no-vig edge and positive expected value."
        connect_link = ""
    else:
        market_note = "Fair lines are active. Add the repository secret THE_ODDS_API_KEY to activate sportsbook prices, no-vig edge, and expected value."
        connect_link = '<a class="connect" href="https://github.com/pjnugent88-commits/mlb-projection-live/settings/secrets/actions/new" target="_blank" rel="noopener">Connect Odds Feed</a>'

    market_metrics = metrics.get("markets", {})
    metric_text = " · ".join(
        f"{key.replace('_', ' ')} MAE {_num(value.get('mae'), 2)}"
        for key, value in market_metrics.items()
    )
    summary = f'''<div class="summary">
        <div><small>Total modeled</small><strong>{int(metrics.get("projected_props", total_displayed))}</strong></div>
        <div><small>Displayed</small><strong id="shown-count">{total_displayed}</strong></div>
        <div><small>Pitcher</small><strong>{pitcher_count}</strong></div>
        <div><small>Batter</small><strong>{batter_count}</strong></div>
        <div><small>Market values</small><strong>{value_count}</strong></div>
    </div>'''

    html = f'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="theme-color" content="#090b0f">
<title>{escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700;800&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
:root{{--bg:#090b0f;--panel:#171e29;--text:#f5f7fa;--muted:#9da8b7;--line:#2a3442;--green:#66e08d;--amber:#f3bd54;--blue:#78b8ff}}
*{{box-sizing:border-box}}
body{{margin:0;background:radial-gradient(circle at 15% 0%,#17231d,var(--bg) 35rem);color:var(--text);font-family:'IBM Plex Sans',sans-serif}}
main{{max-width:980px;margin:auto;padding:24px 15px 50px}}
.badge{{color:var(--green);font-size:.72rem;font-weight:700;letter-spacing:.08em;text-transform:uppercase}}
h1{{font:800 clamp(2.3rem,10vw,4.8rem)/.9 'Barlow Condensed';text-transform:uppercase;margin:15px 0 9px}}
.sub,.notice,footer{{color:var(--muted);font-size:.78rem;line-height:1.5}}
.notice{{display:flex;justify-content:space-between;gap:12px;align-items:center;border:1px solid #5b4b2d;background:#241f15;padding:11px;border-radius:12px;margin:16px 0}}
.connect{{color:var(--text);white-space:nowrap;font-weight:700}}
nav{{display:flex;gap:8px;flex-wrap:wrap}}
nav a{{color:var(--text);text-decoration:none;background:#151c26;border:1px solid var(--line);padding:10px 12px;border-radius:10px;font-size:.76rem;font-weight:700}}
.summary{{display:grid;grid-template-columns:repeat(5,1fr);gap:8px;margin:16px 0}}
.summary div{{background:#10161e;border:1px solid var(--line);border-radius:12px;padding:10px}}
.summary strong{{font-size:1.5rem}}
.controls{{position:sticky;top:0;z-index:4;background:rgba(9,11,15,.94);backdrop-filter:blur(12px);padding:10px 0;border-bottom:1px solid var(--line);margin-bottom:16px}}
.filters{{display:flex;gap:7px;overflow-x:auto;padding-bottom:5px;scrollbar-width:none}}
.filters::-webkit-scrollbar{{display:none}}
.filters button{{display:flex;gap:7px;align-items:center;white-space:nowrap;color:var(--text);background:#151c26;border:1px solid var(--line);padding:9px 11px;border-radius:999px;font:700 .72rem 'IBM Plex Sans';cursor:pointer}}
.filters button.active{{border-color:var(--green);color:var(--green);background:#112019}}
.filters button b{{font-size:.64rem;background:#0b1016;padding:2px 6px;border-radius:999px}}
.search{{width:100%;margin-top:8px;background:#0e141c;border:1px solid var(--line);color:var(--text);border-radius:10px;padding:10px 12px;font:500 .8rem 'IBM Plex Sans'}}
.market-section{{margin:24px 0}}
.section-head{{display:flex;align-items:end;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:9px;margin-bottom:10px}}
.section-head span{{color:var(--green);font-size:.66rem;text-transform:uppercase;letter-spacing:.08em;font-weight:700}}
.section-head h2{{font:800 1.8rem 'Barlow Condensed';margin:1px 0 0;text-transform:uppercase}}
.section-head>b{{color:var(--muted);font-size:.8rem}}
.card,.empty{{background:linear-gradient(145deg,#1b232f,#121720);border:1px solid var(--line);border-radius:16px;padding:15px;margin:12px 0}}
.top{{display:flex;justify-content:space-between;gap:10px}}
.card h2{{font:700 1.45rem/1 'Barlow Condensed';margin:0}}
.top span,p,.market{{color:var(--muted);font-size:.72rem}}
.top b{{font-size:.68rem;border:1px solid;padding:6px 8px;border-radius:999px;height:max-content}}
.value{{color:var(--green)}}.lean{{color:var(--blue)}}.pass{{color:#bdc4ce}}
.grid,.market{{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin-top:12px}}
.grid div,.market div{{background:#0f141c;border:1px solid #202a37;border-radius:11px;padding:10px}}
.market{{padding-top:10px;border-top:1px solid var(--line)}}
small,i{{display:block;color:var(--muted);font-size:.63rem;font-style:normal}}
strong{{display:block;font:700 1.28rem 'Barlow Condensed';margin:3px 0}}
.no-results{{display:none;background:#10161e;border:1px dashed var(--line);border-radius:14px;padding:20px;color:var(--muted);text-align:center}}
footer{{border-top:1px solid var(--line);padding-top:16px;margin-top:24px}}
[hidden]{{display:none!important}}
@media(max-width:720px){{.summary{{grid-template-columns:repeat(2,1fr)}}.summary div:first-child{{grid-column:1/-1}}}}
@media(max-width:540px){{.grid,.market{{grid-template-columns:repeat(2,1fr)}}.grid div:last-child{{grid-column:1/-1}}.notice{{align-items:flex-start;flex-direction:column}}}}
</style>
</head>
<body>
<main>
<span class="badge">Stage 3 · real Statcast player models</span>
<h1>{escape(title)}</h1>
<div class="sub">{escape(date_label)} · generated {escape(generated_label)}<br>{escape(metric_text)}</div>
<div class="notice"><span>{escape(market_note)}</span>{connect_link}</div>
<nav><a href="index.html">Game Board</a><a href="player_props.csv">All Props CSV</a><a href="prop_metrics.json">Prop Metrics</a><a href="prop_metadata.json">Sources</a></nav>
{summary}
<div class="controls">
    <div class="filters">{filter_buttons}</div>
    <input class="search" id="prop-search" type="search" placeholder="Search player, team, or market" autocomplete="off">
</div>
<div id="prop-sections">{''.join(sections)}</div>
<div class="no-results" id="no-results">No props match this category and search.</div>
<footer>Lineups, scratches, opener usage, pitch limits, weather, and prices can change. A fair probability is not proof of a profitable wager.</footer>
</main>
<script>
(() => {{
  const buttons = [...document.querySelectorAll('[data-filter]')];
  const cards = [...document.querySelectorAll('.card')];
  const sections = [...document.querySelectorAll('.market-section')];
  const search = document.getElementById('prop-search');
  const shown = document.getElementById('shown-count');
  const noResults = document.getElementById('no-results');
  let activeFilter = 'all';

  function filterMatches(card) {{
    if (activeFilter === 'all') return true;
    if (activeFilter === 'value') return card.dataset.value === 'true';
    if (activeFilter === 'pitcher' || activeFilter === 'batter') return card.dataset.playerType === activeFilter;
    return card.dataset.market === activeFilter;
  }}

  function applyFilters() {{
    const term = search.value.trim().toLowerCase();
    let visible = 0;
    cards.forEach(card => {{
      const matches = filterMatches(card) && (!term || card.dataset.search.includes(term));
      card.hidden = !matches;
      if (matches) visible += 1;
    }});
    sections.forEach(section => {{
      section.hidden = ![...section.querySelectorAll('.card')].some(card => !card.hidden);
    }});
    shown.textContent = String(visible);
    noResults.style.display = visible ? 'none' : 'block';
  }}

  buttons.forEach(button => button.addEventListener('click', () => {{
    activeFilter = button.dataset.filter;
    buttons.forEach(item => item.classList.toggle('active', item === button));
    applyFilters();
  }}));
  search.addEventListener('input', applyFilters);
  if (buttons.length) buttons[0].classList.add('active');
  applyFilters();
}})();
</script>
</body>
</html>'''
    output_path.write_text(html, encoding="utf-8")
    return output_path
