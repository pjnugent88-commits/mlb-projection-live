from __future__ import annotations

import pandas as pd


def select_dashboard_props(props: pd.DataFrame, per_market: int) -> pd.DataFrame:
    """Keep every VALUE row plus a balanced top set from each market."""
    if props.empty:
        return props.copy()
    per_market = max(int(per_market), 1)
    selected = set(props.index[props["signal"].astype(str).str.contains("VALUE")].tolist())
    for _, market_rows in props.groupby("market_key", sort=False):
        selected.update(market_rows.head(per_market).index.tolist())
    return props.loc[sorted(selected)].copy().reset_index(drop=True)
