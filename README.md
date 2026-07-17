# Live Daily MLB Projection and Player-Prop Model

A production GitHub Actions pipeline that retrains and publishes mobile MLB game and player-prop projection boards from current public data.

## Permanent dashboards

- Game projections: **https://pjnugent88-commits.github.io/mlb-projection-live/**
- Player props: **https://pjnugent88-commits.github.io/mlb-projection-live/props.html**

Downloads and audit files:

- `/projections.csv` — current game projections
- `/metrics.json` — game-model chronological holdout metrics
- `/metadata.json` — game-model sources and freshness
- `/player_props.csv` — every current pitcher and batter prop projection
- `/prop_metrics.json` — market-by-market chronological holdout metrics
- `/prop_metadata.json` — prop sources, lineup status, odds status, and display policy

## Categorized player props

The mobile props page separates and filters:

- Best sportsbook values
- All pitcher props
- Pitcher strikeouts
- Pitcher outs recorded
- Pitcher hits allowed
- All batter props
- Batter hits
- Batter total bases
- Batter home runs

It also supports player/team search. All qualifying VALUE rows are displayed, plus a balanced top set from every market so one category cannot crowd out the others. The CSV contains every projection.

## Sportsbook prices and expected value

The statistical models work without a paid API key and publish projections, standard reference lines, over/under probabilities, and fair American odds.

To activate actual sportsbook player-prop prices, no-vig probabilities, model edge, expected value, and best available books:

1. Obtain a The Odds API key on a plan that includes MLB event-level player props.
2. Open repository **Settings → Secrets and variables → Actions**.
3. Create a repository secret named exactly `THE_ODDS_API_KEY`.
4. Paste the key as the secret value and save it.
5. Run **Actions → Live MLB projections and player props → Run workflow**.

Direct secret page: **https://github.com/pjnugent88-commits/mlb-projection-live/settings/secrets/actions/new**

The key remains inside GitHub Actions and is never published to GitHub Pages or the CSV files.

## What runs automatically

The workflow runs at approximately 7 AM, noon, and 4 PM Eastern during daylight-saving time. It can also be started manually from the repository's **Actions** tab.

1. Pulls completed games, probable starters, and the current slate from MLB's Stats API.
2. Pulls Baseball Savant Statcast pitches through `pybaseball`.
3. Builds leakage-safe rolling offense, starter, bullpen, park, Elo, form, rest, pitcher-prop, and batter-prop features.
4. Retrieves archived 24-hour-ahead forecasts for historical training and current Open-Meteo forecasts.
5. Trains chronological game and six-market player-level ensembles.
6. Checks MLB boxscores for confirmed batting orders.
7. Optionally retrieves event-level sportsbook player-prop prices.
8. Publishes categorized mobile dashboards and downloadable audit files to GitHub Pages.

## Production guarantees

- Production mode contains no synthetic-data fallback.
- A failed real-data pull fails the workflow visibly.
- Historical rolling features use only information available before each game.
- Historical player outcomes are derived from pitch-level Statcast events.
- Player rolling form is shifted so the current game's results never enter its own features.
- Historical park and weather joins are pregame-only.
- Train/test splits are chronological rather than random.
- Batter props appear only for batting orders returned by MLB.
- Started and completed games are excluded from the live prop board.
- VALUE labels require both positive model edge and positive expected value against connected sportsbook prices.
- Model-only projections are labeled as leans or watches, not market value.

## Manual run

Open **Actions → Live MLB projections and player props → Run workflow**. Leave the date blank for today's New York date or enter a date in `YYYY-MM-DD` format.

## Limitations

Probable pitchers, batting orders, opener decisions, pitch limits, injuries, weather, roofs, and sportsbook prices can change after publication. Pitcher outs reconstructed from Statcast event outcomes can differ slightly from official scoring on unusual baserunning outs. Holdout metrics measure prediction error; they do not prove profitability. Historical timestamped prop odds and walk-forward closing-line evaluation are required before claiming a durable market edge.
