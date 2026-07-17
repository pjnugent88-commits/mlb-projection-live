import pandas as pd

from mlb_projection.odds import fetch_moneyline_odds


def test_placeholder_odds_key_returns_empty_frame_without_request(monkeypatch):
    def fail_request(*args, **kwargs):
        raise AssertionError("placeholder keys must not reach the network")

    monkeypatch.setattr("mlb_projection.odds.requests.get", fail_request)
    frame = fetch_moneyline_odds("your The Odds API key")

    assert isinstance(frame, pd.DataFrame)
    assert frame.empty
    assert list(frame.columns) == [
        "away_team", "home_team", "best_away_odds", "best_home_odds", "away_book", "home_book",
    ]


def test_unauthorized_odds_response_does_not_break_projection(monkeypatch):
    class UnauthorizedResponse:
        status_code = 401

        def raise_for_status(self):
            import requests

            response = requests.Response()
            response.status_code = 401
            raise requests.HTTPError("unauthorized", response=response)

        def json(self):
            return {}

    monkeypatch.setattr("mlb_projection.odds.requests.get", lambda *args, **kwargs: UnauthorizedResponse())
    frame = fetch_moneyline_odds("not-a-valid-key")

    assert frame.empty
