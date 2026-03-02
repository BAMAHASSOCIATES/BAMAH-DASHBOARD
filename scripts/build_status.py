from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

# =========================
# Config (lo puedes cambiar)
# =========================
TZ = "America/Mexico_City"
EMA_N = 21

SYMBOLS = {
    "NYAD": "^NYAD",   # NYSE Cumulative Advance-Decline Line
    "QQQE": "QQQE",    # Nasdaq 100 Equal Weight
    "VIX": "^VIX",     # Volatility Index
}

OUT_PATH = "docs/data/status.json"


# =========================
# Helpers matemáticos
# =========================
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


def higher_low(close: pd.Series, window: int = 10) -> bool:
    if len(close) < window * 2 + 2:
        return False
    last = close.iloc[-window:].min()
    prev = close.iloc[-2 * window : -window].min()
    return last > prev


# =========================
# Puntos por componente
# =========================
def nyad_points(nyad_close: pd.Series) -> int:
    nyad_ema = ema(nyad_close, EMA_N)
    above = nyad_close.iloc[-1] > nyad_ema.iloc[-1]
    up = slope_up(nyad_ema, 5)
    hl = higher_low(nyad_close, 10)

    if above and up and hl:
        return 4
    if above and up:
        return 2
    if above:
        return 1
    return 0


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
    if score <= 3:
        return "Risk Off"
    if score <= 6:
        return "Neutral"
    return "Risk On"


def write_fallback(note: str):
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(TZ))

    out = {
        "asof_utc": now_utc.isoformat(timespec="seconds"),
        "asof_local": now_local.isoformat(timespec="seconds"),
        "market_status_score": 0,
        "market_status_text": "0/9",
        "market_status_label": "Risk Off",
        "note": note,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("Wrote fallback status.json:", note)


# =========================
# Main
# =========================
def main():
    tickers = list(SYMBOLS.values())

    df = None
    last_err = None

    # Reintentos para evitar rate limits temporales
    waits = [5, 10, 20, 40, 60]
    for attempt in range(1, 6):
        try:
            df = yf.download(
                tickers=tickers,
                period="6mo",
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=False,  # menos agresivo
            )
            if df is not None and len(df) > 0:
                break
        except Exception as e:
            last_err = e

        time.sleep(waits[attempt - 1])

    if df is None or len(df) == 0:
        # Si de plano no bajó nada, escribimos fallback y salimos “bien”
        write_fallback(f"Download failed (rate-limited or unavailable). Last error: {last_err}")
        return

    def get_close(ticker: str) -> pd.Series:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                s = df[(ticker, "Close")].dropna()
            else:
                s = df["Close"].dropna()
            return s
        except Exception:
            return pd.Series(dtype="float64")

    nyad = get_close(SYMBOLS["NYAD"])
    qqqe = get_close(SYMBOLS["QQQE"])
    vix = get_close(SYMBOLS["VIX"])

    # Si algún ticker vino vacío, fallback
    if nyad.empty or qqqe.empty or vix.empty:
        write_fallback("Data unavailable for one or more symbols (likely rate-limited). Try again soon.")
        return

    # Calculamos puntos
    p_nyad = nyad_points(nyad)
    p_qqqe = qqqe_points(qqqe)
    p_vix = vix_points(vix)

    score = int(max(0, min(9, p_nyad + p_qqqe + p_vix)))
    label = label_from_score(score)

    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(TZ))

    out = {
        "asof_utc": now_utc.isoformat(timespec="seconds"),
        "asof_local": now_local.isoformat(timespec="seconds"),
        "market_status_score": score,
        "market_status_text": f"{score}/9",
        "market_status_label": label,
    }

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}: {out['market_status_text']} {out['market_status_label']}")


if __name__ == "__main__":
    main()
