"""
S&P 500 RSI Monitor - GitHub Actions edition
--------------------------------------------
Designed to run as a single "one scan and exit" invocation, triggered on a
schedule by a GitHub Actions workflow (see .github/workflows/rsi_monitor.yml).

Each run:
  1. Checks whether the US market is currently open.
  2. Loads the S&P 500 ticker list.
  3. Loads persisted alert state from rsi_state.json.
  4. Computes Wilder's RSI(14) for every ticker on 5-minute bars.
  5. Calculates SPY's RSI on the same 5-minute timeframe.
  6. Calculates each stock's 200-day simple moving average using daily data.
  7. Alerts only when:
       - stock RSI is below 30,
       - SPY RSI is at or above 40,
       - stock price is above its 200-day moving average, and
       - the stock has not already triggered an active alert.
  8. Sends one combined push notification through ntfy.sh.
  9. Saves updated alert state back to rsi_state.json.

Author: Built for Jack Dilauro
"""

import json
import logging
import os
import sys
from datetime import datetime, timedelta
from io import StringIO
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import yfinance as yf

# ============================== CONFIG ==============================
NTFY_TOPIC = os.environ.get("NTFY_TOPIC", "")

RSI_PERIOD = 14
RSI_THRESHOLD = 30.0
RESET_BUFFER = 5.0

BAR_INTERVAL = "5m"
LOOKBACK_PERIOD = "1d"

# Signal-quality filters
USE_SPY_RSI_FILTER = True
SPY_RSI_MINIMUM = 40.0

USE_200_DAY_TREND_FILTER = True
TREND_MA_PERIOD = 200
DAILY_LOOKBACK_PERIOD = "1y"

CHUNK_SIZE = 100
STATE_FILE = "rsi_state.json"
TICKERS_FILE = "sp500_tickers.json"
TICKERS_REFRESH_HOURS = 24
ONLY_DURING_MARKET_HOURS = True
MAX_TICKERS_PER_NOTIFICATION = 25

HTTP_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

FALLBACK_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "GOOG", "META", "NVDA", "TSLA",
    "BRK-B", "JPM", "V", "UNH", "HD", "PG", "MA", "XOM", "JNJ", "MRK",
]
# ====================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def is_market_open_now() -> bool:
    """Return True during regular US equity hours, Mon-Fri, 9:30-16:00 ET."""
    # This lightweight check does not account for exchange holidays or early closes.
    now = datetime.now(ZoneInfo("America/New_York"))
    if now.weekday() >= 5:
        return False

    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close


def get_sp500_tickers() -> list[str]:
    """Load S&P 500 tickers from cache or refresh them from Wikipedia."""
    if os.path.exists(TICKERS_FILE):
        try:
            with open(TICKERS_FILE, encoding="utf-8") as f:
                cached = json.load(f)

            cached_time = datetime.fromisoformat(cached["updated"])
            cache_age = (
                datetime.now(ZoneInfo("UTC")).replace(tzinfo=None) - cached_time
            )

            if cache_age < timedelta(hours=TICKERS_REFRESH_HOURS):
                tickers = cached["tickers"]
                log.info("Using cached ticker list (%s tickers).", len(tickers))
                return tickers
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            log.warning("Ticker cache could not be read: %s", exc)

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    try:
        response = requests.get(url, headers=HTTP_HEADERS, timeout=15)
        response.raise_for_status()
        table = pd.read_html(StringIO(response.text))[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()

        with open(TICKERS_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "updated": datetime.now(ZoneInfo("UTC"))
                    .replace(tzinfo=None)
                    .isoformat(),
                    "tickers": tickers,
                },
                f,
                indent=2,
            )

        log.info("Refreshed ticker list from Wikipedia: %s tickers.", len(tickers))
        return tickers

    except Exception as exc:
        log.error("Failed to refresh ticker list from Wikipedia: %s", exc)

        if os.path.exists(TICKERS_FILE):
            try:
                with open(TICKERS_FILE, encoding="utf-8") as f:
                    cached = json.load(f)
                tickers = cached["tickers"]
                log.warning(
                    "Falling back to stale cached ticker list (%s tickers).",
                    len(tickers),
                )
                return tickers
            except Exception as cache_exc:
                log.error("Stale ticker cache could not be read: %s", cache_exc)

        log.warning(
            "No usable cache is available. Using built-in fallback list (%s tickers).",
            len(FALLBACK_TICKERS),
        )
        return FALLBACK_TICKERS


def compute_rsi(closes: pd.Series, period: int = RSI_PERIOD) -> float:
    """Calculate the most recent Wilder-style RSI value."""
    if closes is None or len(closes) < period + 1:
        return np.nan

    closes = pd.to_numeric(closes, errors="coerce").dropna()
    if len(closes) < period + 1:
        return np.nan

    delta = closes.diff().dropna()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gain = gains.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()
    avg_loss = losses.ewm(
        alpha=1 / period,
        min_periods=period,
        adjust=False,
    ).mean()

    latest_gain = avg_gain.iloc[-1]
    latest_loss = avg_loss.iloc[-1]

    if pd.isna(latest_gain) or pd.isna(latest_loss):
        return np.nan
    if latest_loss == 0:
        return 100.0 if latest_gain > 0 else 50.0

    relative_strength = latest_gain / latest_loss
    return float(100 - (100 / (1 + relative_strength)))


def extract_single_close(data: pd.DataFrame) -> pd.Series:
    """Extract a single ticker's Close series from a yfinance result."""
    close_data = data["Close"]
    if isinstance(close_data, pd.DataFrame):
        close_data = close_data.iloc[:, 0]
    return pd.to_numeric(close_data, errors="coerce").dropna()


def get_spy_intraday_rsi() -> float:
    """Download SPY intraday data and return its latest RSI."""
    try:
        data = yf.download(
            tickers="SPY",
            period=LOOKBACK_PERIOD,
            interval=BAR_INTERVAL,
            progress=False,
            auto_adjust=True,
            threads=False,
        )

        if data is None or data.empty:
            log.error("SPY intraday download returned no data.")
            return np.nan

        closes = extract_single_close(data)
        spy_rsi = compute_rsi(closes)

        if np.isnan(spy_rsi):
            log.error("Not enough valid SPY bars to calculate RSI.")
            return np.nan

        log.info("SPY intraday RSI: %.2f", spy_rsi)
        return spy_rsi

    except Exception as exc:
        log.error("Failed to calculate SPY intraday RSI: %s", exc)
        return np.nan


def send_notification(
    title: str,
    message: str,
    priority: str = "default",
) -> bool:
    """Send an ntfy notification and return True on confirmed success."""
    if not NTFY_TOPIC:
        log.error("NTFY_TOPIC is not set. Cannot send notification.")
        return False

    try:
        response = requests.post(
            f"https://ntfy.sh/{NTFY_TOPIC}",
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": "chart_with_downwards_trend,warning",
            },
            timeout=10,
        )

        if response.status_code == 200:
            log.info("Notification sent successfully: %s", title)
            return True

        if response.status_code == 429:
            log.error(
                "Notification rejected by ntfy.sh due to rate limiting "
                "(HTTP 429). Body: %s",
                response.text[:200],
            )
            return False

        log.error(
            "Notification failed with HTTP %s: %s",
            response.status_code,
            response.text[:200],
        )
        return False

    except Exception as exc:
        log.error("Failed to send notification: %s", exc)
        return False


def load_state() -> dict:
    """Load persisted alert state. Return an empty state if unavailable."""
    if not os.path.exists(STATE_FILE):
        return {}

    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        log.error("Could not load %s: %s", STATE_FILE, exc)
        return {}


def save_state(state: dict) -> None:
    """Write alert state to disk for the workflow to commit."""
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def fetch_batch_closes(tickers: list[str]) -> dict[str, pd.Series]:
    """Download 5-minute closes for a batch of tickers."""
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
    except Exception as exc:
        log.error("Intraday batch download failed: %s", exc)
        return result

    if data is None or data.empty:
        log.warning("Intraday batch download returned no data.")
        return result

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                closes = extract_single_close(data)
            else:
                closes = pd.to_numeric(
                    data[ticker]["Close"], errors="coerce"
                ).dropna()

            if len(closes) >= RSI_PERIOD + 1:
                result[ticker] = closes
        except (KeyError, TypeError, IndexError):
            continue

    return result


def fetch_batch_trend_data(tickers: list[str]) -> dict[str, dict]:
    """Download daily prices and calculate each ticker's 200-day SMA."""
    result = {}

    try:
        data = yf.download(
            tickers=tickers,
            period=DAILY_LOOKBACK_PERIOD,
            interval="1d",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=True,
        )
    except Exception as exc:
        log.error("Daily-price batch download failed: %s", exc)
        return result

    if data is None or data.empty:
        log.warning("Daily-price batch download returned no data.")
        return result

    for ticker in tickers:
        try:
            if len(tickers) == 1:
                daily_closes = extract_single_close(data)
            else:
                daily_closes = pd.to_numeric(
                    data[ticker]["Close"], errors="coerce"
                ).dropna()

            if len(daily_closes) < TREND_MA_PERIOD:
                log.warning(
                    "%s: only %s daily observations; %s are required.",
                    ticker,
                    len(daily_closes),
                    TREND_MA_PERIOD,
                )
                continue

            ma_200 = float(daily_closes.tail(TREND_MA_PERIOD).mean())
            daily_close = float(daily_closes.iloc[-1])

            result[ticker] = {
                "daily_close": daily_close,
                "ma_200": ma_200,
                "above_200dma": daily_close > ma_200,
            }

        except (KeyError, TypeError, IndexError, ValueError):
            continue

    return result


def update_ticker_state(
    state: dict,
    ticker: str,
    rsi: float,
    now_iso: str,
    alerted: bool | None = None,
) -> None:
    """Update a ticker's state without discarding its prior alert flag."""
    prior_alerted = state.get(ticker, {}).get("alerted", False)
    state[ticker] = {
        "alerted": prior_alerted if alerted is None else alerted,
        "rsi": rsi,
        "time": now_iso,
    }


def run_one_scan(
    tickers: list[str],
    state: dict,
    spy_rsi: float,
) -> list[dict]:
    """Return new oversold signals that pass all enabled filters."""
    new_oversold = []
    now_iso = datetime.now(ZoneInfo("UTC")).replace(tzinfo=None).isoformat()

    spy_filter_passed = True
    if USE_SPY_RSI_FILTER:
        if np.isnan(spy_rsi):
            log.error(
                "SPY RSI is unavailable. New alerts are blocked for this run."
            )
            spy_filter_passed = False
        else:
            spy_filter_passed = spy_rsi >= SPY_RSI_MINIMUM
            if not spy_filter_passed:
                log.info(
                    "SPY filter blocked new alerts: SPY RSI %.2f is below %.2f.",
                    spy_rsi,
                    SPY_RSI_MINIMUM,
                )

    for start in range(0, len(tickers), CHUNK_SIZE):
        chunk = tickers[start : start + CHUNK_SIZE]
        intraday_closes_map = fetch_batch_closes(chunk)

        trend_map = (
            fetch_batch_trend_data(chunk)
            if USE_200_DAY_TREND_FILTER
            else {}
        )

        for ticker, closes in intraday_closes_map.items():
            rsi = compute_rsi(closes)
            if np.isnan(rsi):
                continue

            already_alerted = state.get(ticker, {}).get("alerted", False)

            # Always allow the RSI reset even if a quality filter is currently failing.
            if rsi >= RSI_THRESHOLD + RESET_BUFFER and already_alerted:
                update_ticker_state(
                    state,
                    ticker,
                    rsi,
                    now_iso,
                    alerted=False,
                )
                continue

            if rsi < RSI_THRESHOLD and not already_alerted:
                if not spy_filter_passed:
                    log.info(
                        "%s: RSI %.2f is below %.2f, but the SPY filter failed.",
                        ticker,
                        rsi,
                        RSI_THRESHOLD,
                    )
                    update_ticker_state(state, ticker, rsi, now_iso)
                    continue

                trend_data = trend_map.get(ticker)
                if USE_200_DAY_TREND_FILTER and trend_data is None:
                    log.info(
                        "%s: RSI %.2f is below %.2f, but 200-day data is unavailable.",
                        ticker,
                        rsi,
                        RSI_THRESHOLD,
                    )
                    update_ticker_state(state, ticker, rsi, now_iso)
                    continue

                if (
                    USE_200_DAY_TREND_FILTER
                    and not trend_data["above_200dma"]
                ):
                    log.info(
                        "%s: RSI %.2f is below %.2f, but daily close %.2f "
                        "is not above the 200-day SMA %.2f.",
                        ticker,
                        rsi,
                        RSI_THRESHOLD,
                        trend_data["daily_close"],
                        trend_data["ma_200"],
                    )
                    update_ticker_state(state, ticker, rsi, now_iso)
                    continue

                last_price = float(closes.iloc[-1])
                ma_200 = (
                    trend_data["ma_200"]
                    if USE_200_DAY_TREND_FILTER
                    else np.nan
                )

                new_oversold.append(
                    {
                        "ticker": ticker,
                        "rsi": rsi,
                        "price": last_price,
                        "spy_rsi": spy_rsi,
                        "ma_200": ma_200,
                    }
                )

                update_ticker_state(
                    state,
                    ticker,
                    rsi,
                    now_iso,
                    alerted=True,
                )
                continue

            update_ticker_state(state, ticker, rsi, now_iso)

    return new_oversold


def send_digest_notification(new_oversold: list[dict]) -> None:
    """Send one combined notification containing all new qualifying signals."""
    if not new_oversold:
        return

    sorted_signals = sorted(new_oversold, key=lambda item: item["rsi"])
    lines = []

    for signal in sorted_signals:
        line = (
            f"{signal['ticker']}: RSI {signal['rsi']:.1f} | "
            f"${signal['price']:.2f}"
        )

        if USE_200_DAY_TREND_FILTER and not np.isnan(signal["ma_200"]):
            line += f" | 200DMA ${signal['ma_200']:.2f}"

        lines.append(line)

    count = len(lines)
    title = (
        f"{count} S&P 500 Stock{'s' if count != 1 else ''} "
        f"Oversold (RSI < {int(RSI_THRESHOLD)})"
    )

    shown = lines[:MAX_TICKERS_PER_NOTIFICATION]
    body_parts = []

    if USE_SPY_RSI_FILTER:
        body_parts.append(f"SPY RSI: {sorted_signals[0]['spy_rsi']:.1f}")

    body_parts.extend(shown)

    if count > MAX_TICKERS_PER_NOTIFICATION:
        body_parts.append(
            f"...and {count - MAX_TICKERS_PER_NOTIFICATION} more "
            "(see workflow log for the full list)"
        )

    body = "\n".join(body_parts)
    success = send_notification(title, body)

    if not success:
        log.error(
            "Digest notification failed. Qualifying tickers: %s",
            [signal["ticker"] for signal in sorted_signals],
        )


def main() -> None:
    if ONLY_DURING_MARKET_HOURS and not is_market_open_now():
        log.info(
            "Market is closed right now (America/New_York). Skipping this run."
        )
        return

    log.info(
        "Market open. Stock threshold: RSI < %.1f | Interval: %s",
        RSI_THRESHOLD,
        BAR_INTERVAL,
    )

    tickers = get_sp500_tickers()
    state = load_state()

    spy_rsi = get_spy_intraday_rsi() if USE_SPY_RSI_FILTER else np.nan

    log.info("Scanning %s tickers...", len(tickers))
    new_oversold = run_one_scan(tickers, state, spy_rsi)
    save_state(state)

    if new_oversold:
        log.info(
            "New qualifying oversold tickers this run: %s",
            [signal["ticker"] for signal in new_oversold],
        )
        send_digest_notification(new_oversold)
    else:
        log.info("No new oversold signals passed all enabled filters.")


if __name__ == "__main__":
    main()
