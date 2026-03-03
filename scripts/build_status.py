from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import requests

TZ = "America/Mexico_City"
EMA_N = 21

SYMBOLS = {
    "QQQE": "QQQE",
    "VIX": "^VIX",
}

OUT_PATH = "docs/data/status.json"


def ema(series: pd.Series, n: int) -> pd.Series:
    return series.ewm(span=n, adjust=False).mean()


def slope_up(series: pd.Series, lookback_days: int = 5) -> bool:
    if len(series) < lookback_days + 2:
        return False
    return series.iloc[-1] > series.iloc[-(lookback_days + 1)]


def slope_down(series: pd.Series, lookback_days: int = 5) -> bool:
    if len(series) < lookback_days + 2:
        return False
    return series.iloc[-1] < series.iloc[-(lookback_days + 1)]


def qqqe_points(qqqe_close: pd.Series) -> int:
    qqqe_ema = ema(qqqe_close, EMA_N)
    above = qqqe_close.iloc[-1] > qqqe_ema.iloc[-1]
    up = slope_up(qqqe_ema, 5)

    if above and up:
        return 3
    if above:
        return 1
    return 0


def vix_points(vix_close: pd.Series) -> int:
    vix_ema = ema(vix_close, EMA_N)
    below = vix_close.iloc[-1] < vix_ema.iloc[-1]
    down = slope_down(vix_ema, 5)

    if below and down:
        return 2
    if below or down:
        return 1
    return 0


def label_from_score(score: int) -> str:
    if score <= 1:
        return "Risk Off"
    if score <= 3:
        return "Neutral"
    return "Risk On"


def write_json(score: int, label: str, max_score: int = 5, note: str | None = None):
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(TZ))

    out = {
        "asof_utc": now_utc.isoformat(timespec="seconds"),
        "asof_local": now_local.isoformat(timespec="seconds"),
        "market_status_score": int(score),
        "market_status_text": f"{int(score)}/{int(max_score)}",
        "market_status_label": label,
    }
    if note:
        out["note"] = note

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}: {out['market_status_text']} {out['market_status_label']}")


def fetch_yahoo_close(symbol: str, range_: str = "6mo", interval: str = "1d") -> pd.Series:
    url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
    params = {"range": range_, "interval": interval}
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BAMAH-DASHBOARD/1.0)",
        "Accept": "application/json,text/plain,*/*",
    }

    r = requests.get(url, params=params, headers=headers, timeout=20)
    r.raise_for_status()
    data = r.json()

    chart = data.get("chart", {})
    error = chart.get("error")
    if error:
        raise RuntimeError(f"Yahoo error for {symbol}: {error}")

    result = chart.get("result")
    if not result or not isinstance(result, list):
        raise RuntimeError(f"No result for {symbol}")

    res0 = result[0]
    timestamps = res0.get("timestamp") or []
    quote = ((res0.get("indicators", {}).get("quote")) or [{}])[0]
    closes = quote.get("close") or []

    if not timestamps or not closes:
        return pd.Series(dtype="float64")

    idx = pd.to_datetime(timestamps, unit="s", utc=True)
    return pd.Series(closes, index=idx, dtype="float64").dropna()


def fetch_with_retry(symbol: str, tries: int = 5) -> pd.Series:
    waits = [2, 5, 10, 20, 40]
    last_err = None

    for i in range(tries):
        try:
            s = fetch_yahoo_close(symbol)
            if not s.empty:
                return s
        except Exception as e:
            last_err = e
        time.sleep(waits[min(i, len(waits) - 1)])

    print(f"Failed to fetch {symbol}. Last error: {last_err}")
    return pd.Series(dtype="float64")


def main():
    qqqe = fetch_with_retry(SYMBOLS["QQQE"])
    vix = fetch_with_retry(SYMBOLS["VIX"])

    if qqqe.empty or vix.empty:
        write_json(
            score=0,
            label="Risk Off",
            note="Data unavailable for one or more symbols (Yahoo rate limit or symbol not supported).",
        )
        return

    score = int(max(0, min(5, qqqe_points(qqqe) + vix_points(vix))))
    write_json(score=score, label=label_from_score(score))


if __name__ == "__main__":
    main()
