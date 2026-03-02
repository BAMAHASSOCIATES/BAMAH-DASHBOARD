from __future__ import annotations

import json
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
    """EMA estándar."""
    return series.ewm(span=n, adjust=False).mean()


def slope_up(series: pd.Series, lookback_days: int = 5) -> bool:
    """
    Pendiente positiva simple:
    EMA(hoy) > EMA(hace 5 días de trading aprox)
    """
    if len(series) < lookback_days + 2:
        return False
    return series.iloc[-1] > series.iloc[-(lookback_days + 1)]


def slope_down(series: pd.Series, lookback_days: int = 5) -> bool:
    """Pendiente negativa simple."""
    if len(series) < lookback_days + 2:
        return False
    return series.iloc[-1] < series.iloc[-(lookback_days + 1)]


def higher_low(close: pd.Series, window: int = 10) -> bool:
    """
    Estructura objetiva de higher low (muy útil para NYAD):
    Min(últimos 10 días) > Min(10 días anteriores)
    """
    if len(close) < window * 2 + 2:
        return False
    last = close.iloc[-window:].min()
    prev = close.iloc[-2 * window : -window].min()
    return last > prev


# =========================
# Puntos por componente
# =========================
def nyad_points(nyad_close: pd.Series) -> int:
    """
    NYAD (0–4):
    +4 si: close > EMA21 AND slope up AND higher low
    +2 si: close > EMA21 AND slope up
    +1 si: close > EMA21 (aunque slope no esté claro)
    +0 si: close < EMA21
    """
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
    """
    QQQE (0–3):
    +3 si: close > EMA21 AND slope up
    +1 si: close > EMA21 (pero slope no está claro)
    +0 si: close < EMA21
    """
    qqqe_ema = ema(qqqe_close, EMA_N)
    above = qqqe_close.iloc[-1] > qqqe_ema.iloc[-1]
    up = slope_up(qqqe_ema, 5)

    if above and up:
        return 3
    if above:
        return 1
    return 0


def vix_points(vix_close: pd.Series) -> int:
    """
    VIX (0–2):
    +2 si: close < EMA21 AND slope down
    +1 si: "neutral" (solo 1 de las dos condiciones)
    +0 si: close > EMA21 AND slope up (hostil)
    """
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


# =========================
# Main
# =========================
def main():
    # 1) Descargamos datos (gratis) desde Yahoo
    tickers = list(SYMBOLS.values())

    df = yf.download(
        tickers=tickers,
        period="6mo",
        interval="1d",
        group_by="ticker",
        auto_adjust=False,
        progress=False,
        threads=True,
    )

    def get_close(ticker: str) -> pd.Series:
        # yfinance puede devolver multiindex cuando hay varios tickers
        if isinstance(df.columns, pd.MultiIndex):
            s = df[(ticker, "Close")].dropna()
        else:
            s = df["Close"].dropna()
        return s

    nyad = get_close(SYMBOLS["NYAD"])
    qqqe = get_close(SYMBOLS["QQQE"])
    vix = get_close(SYMBOLS["VIX"])

    # 2) Calculamos puntos
    p_nyad = nyad_points(nyad)
    p_qqqe = qqqe_points(qqqe)
    p_vix = vix_points(vix)

    score = p_nyad + p_qqqe + p_vix
    score = max(0, min(9, int(score)))  # clamp 0..9

    label = label_from_score(score)

    # 3) Guardamos salida para el dashboard (JSON)
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(ZoneInfo(TZ))

    out = {
        "asof_utc": now_utc.isoformat(timespec="seconds"),
        "asof_local": now_local.isoformat(timespec="seconds"),
        "market_status_score": score,      # 0..9
        "market_status_text": f"{score}/9",
        "market_status_label": label,      # Risk Off / Neutral / Risk On
    }

    # Asegúrate de que exista docs/data en el repo (lo haremos en un paso aparte)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print(f"Wrote {OUT_PATH}: {out['market_status_text']} {out['market_status_label']}")


if __name__ == "__main__":
    main()
