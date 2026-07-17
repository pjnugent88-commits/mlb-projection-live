from pathlib import Path

import pandas as pd

from mlb_projection.player_props_dashboard import render_player_props_dashboard
from scripts.run_player_props import _select_dashboard_props


def _props() -> pd.DataFrame:
    rows = []
    specs = [
        ("pitcher_strikeouts", "Pitcher strikeouts", "pitcher"),
        ("pitcher_outs", "Pitcher outs recorded", "pitcher"),
        ("pitcher_hits_allowed", "Pitcher hits allowed", "pitcher"),
        ("batter_hits", "Batter hits", "batter"),
        ("batter_total_bases", "Batter total bases", "batter"),
        ("batter_home_runs", "Batter home runs", "batter"),
    ]
    for market_index, (market_key, label, player_type) in enumerate(specs):
        for rank in range(3):
            rows.append({
                "game_pk": 100 + market_index,
                "game_datetime": "2026-07-17T23:00:00Z",
                "away_team": "BOS",
                "home_team": "NYY",
                "player_name": f"Player {market_index}-{rank}",
                "player_type": player_type,
                "lineup_status": "confirmed" if player_type == "batter" else "probable",
                "batting_order": rank + 1 if player_type == "batter" else None,
                "market_key": market_key,
                "market_label": label,
                "projection": 1.0 + rank,
                "line": 0.5 + rank,
                "over_probability": 0.61,
                "under_probability": 0.39,
                "over_fair_odds": -156,
                "under_fair_odds": 156,
                "over_odds": -110 if rank == 0 else None,
                "under_odds": -110 if rank == 0 else None,
                "over_book": "Test Book" if rank == 0 else None,
                "under_book": "Test Book" if rank == 0 else None,
                "over_no_vig_probability": 0.50 if rank == 0 else None,
                "under_no_vig_probability": 0.50 if rank == 0 else None,
                "over_edge": 0.11 if rank == 0 else None,
                "under_edge": -0.11 if rank == 0 else None,
                "over_ev_per_unit": 0.16 if rank == 0 else None,
                "under_ev_per_unit": -0.26 if rank == 0 else None,
                "signal": "OVER VALUE" if rank == 0 else "OVER LEAN",
            })
    return pd.DataFrame(rows)


def test_balanced_dashboard_selection_keeps_every_market_and_values():
    selected = _select_dashboard_props(_props(), per_market=1)
    assert selected["market_key"].nunique() == 6
    assert selected["signal"].str.contains("VALUE").sum() == 6


def test_dashboard_renders_categories_filters_and_market_details(tmp_path: Path):
    props = _select_dashboard_props(_props(), per_market=1)
    metrics = {
        "projection_date": "2026-07-17",
        "generated_at_utc": "2026-07-17T19:00:00Z",
        "projected_props": 18,
        "market_odds_available": True,
        "markets": {},
    }
    output = render_player_props_dashboard(props, metrics, tmp_path / "props.html")
    html = output.read_text(encoding="utf-8")
    assert "Best Values" in html
    assert "Pitcher Props" in html
    assert "Batter Props" in html
    assert 'data-filter="batter_home_runs"' in html
    assert 'data-section-market="pitcher_strikeouts"' in html
    assert "No-vig / edge" in html
    assert "Expected value" in html
