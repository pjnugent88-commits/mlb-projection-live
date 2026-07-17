import numpy as np
import pandas as pd

from mlb_projection.lineups import fetch_game_lineups
from mlb_projection.player_props import PropMarketModel, default_prop_line
from mlb_projection.prop_odds import attach_game_pks, normalize_player_name


def test_default_prop_lines():
    assert default_prop_line("batter_hits", 1.2) == 0.5
    assert default_prop_line("batter_total_bases", 1.2) == 1.5
    assert default_prop_line("batter_home_runs", 0.3) == 0.5
    assert default_prop_line("pitcher_strikeouts", 5.8) == 5.5


def test_home_run_probability_is_poisson():
    model = PropMarketModel("batter_home_runs", None, None, 0, [], 0.5, np.array([0.0]), {})
    probability = model.over_probability([0.2], [0.5])[0]
    assert 0.17 < probability < 0.19


def test_lineup_parser_requires_nine(monkeypatch):
    payload = {
        "teams": {
            "away": {"battingOrder": list(range(1, 10)), "players": {f"ID{i}": {"person": {"fullName": f"A{i}"}} for i in range(1, 10)}},
            "home": {"battingOrder": list(range(11, 20)), "players": {f"ID{i}": {"person": {"fullName": f"H{i}"}} for i in range(11, 20)}},
        }
    }
    monkeypatch.setattr("mlb_projection.lineups._fetch_boxscore", lambda game_pk: payload)
    slate = pd.DataFrame([{"game_pk": 1, "away_team": "BOS", "home_team": "NYY"}])
    lineups = fetch_game_lineups(slate)
    assert len(lineups) == 18
    assert lineups["lineup_status"].eq("confirmed").all()


def test_prop_name_and_game_matching():
    odds = pd.DataFrame([{"away_team": "BOS", "home_team": "NYY", "player_name": "José Ramírez"}])
    slate = pd.DataFrame([{"game_pk": 7, "away_team": "BOS", "home_team": "NYY"}])
    matched = attach_game_pks(odds, slate)
    assert matched.loc[0, "game_pk"] == 7
    assert normalize_player_name("José Ramírez") == "jose ramirez"
