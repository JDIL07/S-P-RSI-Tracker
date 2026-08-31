# S&P 500 RSI Monitor — GitHub Actions Edition

Runs entirely in the cloud on GitHub's free servers. No machine of yours needs
to stay powered on. Every 15 minutes during market hours, a scheduled workflow
scans all ~500 S&P 500 stocks, computes RSI(14), and pushes a notification to
your phone the moment any stock's RSI first drops below 30.

## How it works

- **GitHub Actions** triggers `rsi_monitor.py` on a cron schedule (free tier
  gives you 2,000 build-minutes/month — this uses only a few minutes/day).
- The script pulls intraday prices via `yfinance`, computes Wilder's RSI(14),
  and calls **ntfy.sh** to push a notification straight to your phone.
- Alert state (which tickers have already fired, so you don't get spammed) is
  saved to `rsi_state.json` and committed back into the repo automatically by
  the workflow after each run, so it persists between runs.

## One-time setup (10 minutes)

### 1. Create the GitHub repo
1. Go to [github.com/new](https://github.com/new).
2. Name it something like `sp500-rsi-monitor`. Choose **Private** (recommended,
   since the repo will contain your alert history).
3. Upload the 5 files/folders included here, preserving the folder structure:
   ```
   .github/workflows/rsi_monitor.yml
   rsi_monitor.py
   requirements.txt
   .gitignore
   README.md
   ```
   Easiest way: on the repo's main page, use **Add file → Upload files** and
   drag in everything (GitHub preserves the `.github/workflows/` path as long
   as you drag the whole folder structure, or use `git push` from your machine
   if you're comfortable with git).

### 2. Set up phone notifications (free, no account)
1. Install the **ntfy** app (iOS App Store or Google Play / F-Droid).
2. Open the app → **Subscribe to topic**.
3. Choose a **unique, hard-to-guess** topic name — anyone who knows it can see
   your alerts (e.g. `jack-dilauro-rsi-8837`).

### 3. Add the topic name as a GitHub Secret
1. In your repo, go to **Settings → Secrets and variables → Actions**.
2. Click **New repository secret**.
3. Name: `NTFY_TOPIC`, Value: the topic name you picked above.
4. Save.

### 4. Enable and test the workflow
1. Go to the **Actions** tab in your repo → you should see "S&P 500 RSI Monitor."
2. Click it, then click **Run workflow** (this is the manual trigger — no need
   to wait for the schedule) to confirm everything works end-to-end.
3. Check the run logs — you should see it scan tickers and log any alerts.
4. If the market is closed when you test, it'll log "Market is closed" and
   exit immediately, which is expected — the RSI logic won't run.

That's it. From here on, it runs itself every 15 minutes during market hours,
Monday–Friday, with zero maintenance.

## Tuning knobs (top of `rsi_monitor.py`)

| Setting | What it does |
|---|---|
| `RSI_THRESHOLD` | Alert level (default 30) |
| `RSI_PERIOD` | RSI lookback bars (default 14, the standard) |
| `BAR_INTERVAL` | `"5m"`, `"15m"`, `"30m"`, `"60m"` — smaller = more granular but more prone to Yahoo rate-limiting across 500 tickers |
| `RESET_BUFFER` | RSI must climb this many points above 30 before it can re-trigger (prevents spam if a stock hovers right at 30) |
| `CHUNK_SIZE` | Tickers per batch download call |

To change the scan frequency, edit the `cron` line in
`.github/workflows/rsi_monitor.yml`. The current schedule
(`*/15 13-21 * * 1-5`) runs every 15 minutes from 13:00–21:45 UTC on
weekdays — a window wide enough to cover 9:30am–4:00pm ET across both
Eastern Standard and Eastern Daylight Time. The script's own market-hours
check (using proper `America/New_York` timezone handling) filters out the
runs outside actual trading hours, so no config changes are needed for
daylight saving transitions.

## Important notes / limitations

- **Data source**: `yfinance` is a free, unofficial wrapper around Yahoo
  Finance's public endpoints. It requires no API key, but Yahoo can change or
  rate-limit it without notice. If you need guaranteed reliability, a paid
  feed (Polygon.io, Finnhub, IEX Cloud, Alpha Vantage) would be more robust —
  the script can be adapted to any of those with a different `fetch_batch_closes`
  function.
- **"Real-time" caveat**: genuine tick-by-tick data across 500 names
  typically requires a paid feed. 15-minute bars is a solid, free middle
  ground for RSI, which isn't designed as a second-by-second signal anyway.
- **GitHub Actions timing**: scheduled workflows are "best effort" — GitHub
  states cron jobs can be delayed a few minutes during periods of high load
  on their infrastructure. This is generally fine for RSI monitoring but
  worth knowing if you need precise timing.
- **Privacy**: keep your ntfy topic name private (don't commit it into the
  repo itself — that's why it's stored as a GitHub Secret). Anyone who knows
  your topic name can subscribe and see your alerts.
