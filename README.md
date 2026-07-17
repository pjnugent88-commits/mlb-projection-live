# Live Daily MLB Projection and Player-Prop Model

A production GitHub Actions pipeline that retrains and publishes mobile MLB game and player-prop projection boards from current public data.

## Permanent dashboards

- Game projections: **https://pjnugent88-commits.github.io/mlb-projection-live/**
- Player props: **https://pjnugent88-commits.github.io/mlb-projection-live/props.html**

Downloads and audit files:

- `/projections.csv` — current game projections
- `/metrics.json` — game-model chronological holdout metrics
- `/metadata.json` — game-model sources and freshness
- `/player_props.csv` — current pitcher and batter prop projections
- `/prop_metrics.json` — market-by-market chronological holdout metrics
- `/prop_metadata.json` — prop sources, lineup status, and odds status

## What runs automatically

The workflow runs at approximately 7 AM, noon, and 4 PM Eastern during daylight-saving time. It can also be started manually from the repository's **Actions** tab.

1. Pulls completed games, probable starters, and the current slate from MLB's Stats API.
2. Pulls Baseball Savant Statcast pitches through `pybaseball`.
3. Builds leakage-safe rolling offense, starter, bullpen, park, Elo, form, rest, pitcher-prop, and batter-prop features.
4. Retrieves archived 24-hour-ahead forecasts for historical training and current Open-Meteo forecasts.
5. Trains chronological game and player-level ensembles.
6. Checks MLB boxscores for confirmed batting orders.
7. Publishes game probabilities, projected scores, fair odds, prop projections, fair prop lines, input coverage, and downloadable files to GitHub Pages.

## Stage 3 player props

The live prop page models:

- Pitcher strikeouts
- Pitcher outs recorded
- Pitcher hits allowed
- Batter hits
- Batter total bases
- Batter home runs

Pitcher props appear when MLB lists a probable starter. Batter props appear only after MLB publishes the batting order for that game. This avoids presenting a bench player or scratched hitter as a confirmed prop candidate.

Without sportsbook prices, the system publishes model projections, standard reference lines, over/under probabilities, and fair American odds. With a qualifying The Odds API plan and the `THE_ODDS_API_KEY` repository secret, it requests event-level player props, removes two-way vig, and calculates edge and expected value.

## Production guarantees

- Production mode contains no synthetic-data fallback.
- A failed real-data pull fails the workflow visibly.
- Historical rolling features use only information available before each game.
- Historical player outcomes are derived from pitch-level Statcast events.
- Player rolling form is shifted so the current game's results never enter its own features.
- Historical park and weather joins are pregame-only.
- Train/test splits are chronological rather than random.
- Model leans are not described as market value unless sportsbook prices are available.

## Optional sportsbook layer

1. Obtain an API key from The Odds API on a plan that includes MLB player props.
2. Open repository **Settings → Secrets and variables → Actions**.
3. Create a repository secret named `THE_ODDS_API_KEY`.
4. Run the workflow again.

Player props are queried one event at a time. API usage therefore depends on the number of games, markets, and bookmaker regions requested.

## Manual run

Open **Actions → Live MLB projections and player props → Run workflow**. Leave the date blank for today's New York date or enter a date in `YYYY-MM-DD` format.

## Limitations

Probable pitchers, batting orders, opener decisions, pitch limits, injuries, weather, roofs, and sportsbook prices can change after publication. Pitcher outs reconstructed from Statcast event outcomes can differ slightly from official scoring on unusual baserunning outs. Holdout metrics measure prediction error; they do not prove profitability. Historical timestamped prop odds and walk-forward closing-line evaluation are required before claiming a durable market edge.
