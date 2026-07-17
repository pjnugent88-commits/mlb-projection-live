from __future__ import annotations

TEAM_CODE_MAP = {"CHW": "CWS", "KCR": "KC", "SDP": "SD", "SFG": "SF", "TBR": "TB", "WSN": "WSH", "OAK": "ATH"}

HIT_BASES = {"single": 1, "double": 2, "triple": 3, "home_run": 4}
EVENT_OUTS = {
    "strikeout": 1, "field_out": 1, "force_out": 1, "fielders_choice_out": 1,
    "sac_fly": 1, "sac_bunt": 1, "caught_stealing_2b": 1, "caught_stealing_3b": 1,
    "caught_stealing_home": 1, "pickoff_caught_stealing_2b": 1,
    "pickoff_caught_stealing_3b": 1, "pickoff_caught_stealing_home": 1,
    "other_out": 1, "sac_bunt_double_play": 2, "grounded_into_double_play": 2,
    "double_play": 2, "strikeout_double_play": 2, "triple_play": 3,
}
SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip", "foul_bunt",
    "missed_bunt", "hit_into_play", "hit_into_play_no_out", "hit_into_play_score",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}

PITCHER_FEATURES = [
    "pitcher_k_per_start", "pitcher_outs_per_start", "pitcher_hits_allowed_per_start",
    "pitcher_pitches_per_start", "pitcher_bf_per_start", "pitcher_k_rate",
    "pitcher_whiff_rate", "pitcher_called_strike_rate", "pitcher_xwoba_allowed",
    "pitcher_hits_allowed_rate", "pitcher_walk_rate", "pitcher_velocity",
    "opponent_k_rate", "opponent_hits_per_game", "opponent_tb_per_game",
    "opponent_pa_per_game", "is_home", "park_hr_factor", "temperature_f",
    "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor",
]
BATTER_FEATURES = [
    "batter_hits_per_game", "batter_tb_per_game", "batter_hr_per_game",
    "batter_pa_per_game", "batter_hit_rate", "batter_tb_rate", "batter_hr_rate",
    "batter_k_rate", "batter_bb_rate", "batter_xwoba", "batter_hard_hit_rate",
    "batter_barrel_rate", "opposing_pitcher_k_rate", "opposing_pitcher_xwoba_allowed",
    "opposing_pitcher_hits_allowed_rate", "opposing_pitcher_outs_per_start",
    "batting_order", "is_home", "park_hr_factor", "temperature_f",
    "humidity_pct", "wind_speed_mph", "precipitation_in", "roof_control_factor",
]
MARKET_LABELS = {
    "pitcher_strikeouts": "Pitcher strikeouts", "pitcher_outs": "Pitcher outs",
    "pitcher_hits_allowed": "Pitcher hits allowed", "batter_hits": "Batter hits",
    "batter_total_bases": "Batter total bases", "batter_home_runs": "Batter home runs",
}
TARGETS = {
    "pitcher_strikeouts": ("pitcher", "strikeouts"),
    "pitcher_outs": ("pitcher", "outs"),
    "pitcher_hits_allowed": ("pitcher", "hits_allowed"),
    "batter_hits": ("batter", "hits"),
    "batter_total_bases": ("batter", "total_bases"),
    "batter_home_runs": ("batter", "home_runs"),
}
