"""
S&P 500 RSI Monitor — GitHub Actions edition
----------------------------------------------
Designed to run as a single "one scan and exit" invocation, triggered on a
schedule by a GitHub Actions workflow (see .github/workflows/rsi_monitor.yml).

Each run:
  1. Checks whether the US market is currently open (exits immediately if not,
     so scheduled runs outside market hours are nearly instant/free).
  2. Loads the S&P 500 ticker list (cached locally, refreshed periodically).
  3. Loads persisted alert state from rsi_state.json (committed to the repo
     by the workflow after every run, so state survives between runs).
  4. Computes Wilder's RSI(14) for every ticker on 15-minute bars.
  5. Sends a push notification via ntfy.sh for any ticker whose RSI just
     crossed below the threshold (and hasn't already been alerted).
  6. Saves updated state back to rsi_state.json for the workflow to commit.

Author: Built for Jack Dilauro
"""

import json
import os
import sys
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import numpy as np
import requests
import yfinance as yf

# ============================== CONFIG ==============================
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")   # set via GitHub Actions secret
RSI_PERIOD = 14
RSI_THRESHOLD = 30.0
RESET_BUFFER = 5.0            # must climb back above 35 before re-alerting
BAR_INTERVAL = "15m"
LOOKBACK_PERIOD = "5d"
CHUNK_SIZE = 100
STATE_FILE = "rsi_state.json"
TICKERS_FILE = "sp500_tickers.json"
TICKERS_REFRESH_HOURS = 24     # re-scrape Wikipedia at most once a day
ONLY_DURING_MARKET_HOURS = True
# ======================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def is_market_open_now() -> bool:
    """US equity market hours check (9:30-16:00 ET, Mon-Fri). Ignores holidays."""
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False
    open_t = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_t = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return open_t <= now <= close_t


def get_sp500_tickers() -> list[str]:
    """Load S&P 500 tickers, using a local cache refreshed at most once a day."""
    if os.path.exists(TICKERS_FILE):
        with open(TICKERS_FILE) as f:
            cached = json.load(f)
        cached_time = datetime.fromisoformat(cached["updated"])
        if datetime.utcnow() - cached_time < timedelta(hours=TICKERS_REFRESH_HOURS):
            log.info(f"Using cached ticker list ({len(cached['tickers'])} tickers).")
            return cached["tickers"]

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    with open(TICKERS_FILE, "w") as f:
        json.dump({"updated": datetime.utcnow().isoformat(), "tickers": tickers}, f, indent=2)
    log.info(f"Refreshed ticker list from Wikipedia: {len(tickers)} tickers.")
    return tickers


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    """Wilder's RSI, returns the most recent value."""
    if closes is None or len(closes) < period + 1:
        return np.nan
    delta = closes.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = losses.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.fillna(100)
    return float(rsi.iloc[-1])


def send_notification(title: str, message: str, priority: str = "high"):
    if not NTFY_TOPIC:
        log.error("NTFY_TOPIC is not set - cannot send notification.")
        return
    try:
        requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={"Title": title, "Priority": priority, "Tags": "chart_with_downwards_trend,warning"},
            timeout=10,
        )
        log.info(f"Notification sent: {title} | {message}")
    except Exception as e:
        log.error(f"Failed to send notification: {e}")


def load_state() -> dict:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def save_state(state: dict):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)


def fetch_batch_closes(tickers: list[str]) -> dict:
    result = {}
    try:
        data = yf.download(
            tickers=tickers,
            period=LOOKBACK_PERIOD,
            interval=BAR_INTERVAL,
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=True,
        )
    except Exception as e:
        log.error(f"Batch download failed: {e}")
        return result

    for t in tickers:
        try:
            closes = data["Close"].dropna() if len(tickers) == 1 else data[t]["Close"].dropna()
            if len(closes) >= RSI_PERIOD + 1:
                result[t] = closes
        except Exception:
            continue
    return result


def run_one_scan(tickers: list[str], state: dict) -> list[str]:
    alerts_this_scan = []
    for i in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[i : i + CHUNK_SIZE]
        closes_map = fetch_batch_closes(chunk)
        for ticker, closes in closes_map.items():
            rsi = compute_rsi(closes)
            if np.isnan(rsi):
                continue

            already_alerted = state.get(ticker, {}).get("alerted", False)

            if rsi < RSI_THRESHOLD and not already_alerted:
                last_price = closes.iloc[-1]
                title = f"RSI Oversold: {ticker}"
                message = f"{ticker} RSI({RSI_PERIOD}) = {rsi:.1f} | Last price: ${last_price:.2f}"
                send_notification(title, message)
                alerts_this_scan.append(ticker)
                state[ticker] = {"alerted": True, "rsi": rsi, "time": datetime.utcnow().isoformat()}

            elif rsi >= RSI_THRESHOLD + RESET_BUFFER and already_alerted:
                state[ticker] = {"alerted": False, "rsi": rsi, "time": datetime.utcnow().isoformat()}

            else:
                state.setdefault(ticker, {})
                state[ticker]["rsi"] = rsi
                state[ticker]["time"] = datetime.utcnow().isoformat()

    return alerts_this_scan


def main():
    if ONLY_DURING_MARKET_HOURS and not is_market_open_now():
        log.info("Market is closed right now (America/New_York). Skipping this run.")
        return

    log.info(f"Market open. Threshold: RSI < {RSI_THRESHOLD} | Interval: {BAR_INTERVAL}")
    tickers = get_sp500_tickers()
    state = load_state()

    log.info(f"Scanning {len(tickers)} tickers...")
    alerts = run_one_scan(tickers, state)
    save_state(state)

    if alerts:
        log.info(f"Alerts fired this run: {alerts}")
    else:
        log.info("No new oversold signals this run.")


if __name__ == "__main__":
    main()
