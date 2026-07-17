# Live Daily MLB Projection Model

A production GitHub Actions pipeline that retrains and publishes a mobile MLB projection board from current public data.

## Permanent dashboard

After the first successful workflow:

**https://pjnugent88-commits.github.io/mlb-projection-live/**

The same site also exposes:

- `/projections.csv` — current slate projections
- `/metrics.json` — chronological holdout metrics
- `/metadata.json` — source and freshness information

## What runs automatically

The workflow runs twice daily and can also be started manually from the repository's **Actions** tab.

1. Pulls completed games and the current slate from MLB's Stats API.
2. Pulls Baseball Savant Statcast pitches through `pybaseball`.
3. Builds leakage-safe rolling offense, starter, bullpen, park, Elo, form, and rest features.
4. Retrieves archived 24-hour-ahead forecasts for historical training and a current Open-Meteo forecast for today's games.
5. Trains a chronological linear/Extra Trees ensemble.
6. Publishes win probabilities, projected runs, expected totals, fair odds, input coverage, metrics, and downloadable CSV output to GitHub Pages.

## Production guarantees

- Production mode contains no synthetic-data fallback.
- A failed real-data pull fails the workflow visibly.
- Every historical rolling feature uses only information available before the game.
- The train/test split is chronological rather than random.
- Historical weather uses a forecast created 24 hours before the game's valid time, not observed postgame weather.
- Historical starting-pitcher features use the pitcher who actually started, joined only to prior pitcher performance.
- Model leans are not described as market value unless sportsbook prices are available.

## Optional market-value layer

The statistical model works without a paid API key. To add current sportsbook moneylines:

1. Obtain a key from The Odds API.
2. Open repository **Settings → Secrets and variables → Actions**.
3. Create a repository secret named `THE_ODDS_API_KEY`.
4. Run the workflow again.

When odds are absent, the board reports model probabilities, projected scores, fair odds, and probability-based leans. When odds are present, it removes the market vig, calculates edge and expected value, and labels only signals that pass the configured thresholds.

## Manual run

Open **Actions → Live MLB projections → Run workflow**. Leave the date blank for today's New York date or enter a date in `YYYY-MM-DD` format.

## Limitations

Probable pitchers, lineups, roof decisions, forecasts, injuries, and sportsbook prices can change after publication. The model's holdout metrics measure prediction error; they do not prove betting profitability. Historical timestamped odds and a walk-forward closing-line evaluation are required before claiming a durable market edge.
