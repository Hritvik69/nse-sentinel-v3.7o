"""
csv_next_day_engine.py
──────────────────────
CSV Next-Day Potential Engine for NSE Sentinel.

Completely standalone — reads local CSV data via data_downloader.load_csv()
and computes PRE-BREAKOUT early-move probability for each ticker.

Design rules:
  • Zero dependency on strategy_engines/ — no imports from those modules
  • Never crashes — all paths wrapped in try/except
  • Works in two modes:
      - Enrichment mode : df passed-in has scan results → enrich those rows
      - CSV scan mode   : df is None/empty → scan DATA_DIR CSVs itself
  • Returns ONLY filtered, sorted, enriched DataFrame

Main entry point:
    from csv_next_day_engine import run_csv_next_day
    result_df = run_csv_next_day(existing_df_or_None)
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ── Try to import load_csv + DATA_DIR from data_downloader ────────────
try:
    from data_downloader import load_csv, DATA_DIR
    _DOWNLOADER_OK = True
except ImportError:
    _DOWNLOADER_OK = False
    DATA_DIR = Path("data")

    def load_csv(ticker_ns: str) -> pd.DataFrame | None:  # type: ignore[misc]
        return None


# ═══════════════════════════════════════════════════════════════════════
# INTERNAL HELPERS  (standalone — no external engine dependency)
# ═══════════════════════════════════════════════════════════════════════

def _sf(v: object, default: float = 0.0) -> float:
    """Safe float — returns finite float or default."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _ema_series(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _rsi_last(close: pd.Series, period: int = 14) -> float:
    """Returns the last RSI value for a Close series."""
    try:
        if len(close) < period + 1:
            return 50.0
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rsi_s = 100 - (100 / (1 + g / l.replace(0, np.nan)))
        val = float(rsi_s.iloc[-1])
        return val if np.isfinite(val) else 50.0
    except Exception:
        return 50.0


def _rsi_series(close: pd.Series, period: int = 14) -> pd.Series:
    """Vectorized RSI series used for feature engineering."""
    try:
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        rsi_s = 100 - (100 / (1 + rs))
        return rsi_s.fillna(50.0)
    except Exception:
        return pd.Series(np.full(len(close), 50.0), index=close.index, dtype=float)


def _safe_div(num: pd.Series | float, den: pd.Series | float, default: float = 0.0):
    """Safe division for scalars/Series, preserving shape where possible."""
    try:
        return num / den.replace(0, np.nan)  # type: ignore[union-attr]
    except Exception:
        try:
            den_f = float(den)  # type: ignore[arg-type]
            return float(num) / den_f if den_f else default  # type: ignore[arg-type]
        except Exception:
            return default


def _prepare_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a feature frame for current-row scoring and historical analog lookup.
    Uses only local CSV data and no external engines.
    """
    out = df.copy()

    close = out["Close"].astype(float)
    volume = out["Volume"].astype(float)
    open_s = out["Open"].astype(float) if "Open" in out.columns else close.shift(1).fillna(close)
    high = out["High"].astype(float) if "High" in out.columns else close
    low = out["Low"].astype(float) if "Low" in out.columns else close

    out["ema20"] = _ema_series(close, 20)
    out["ema50"] = _ema_series(close, 50)
    out["rsi14"] = _rsi_series(close)

    avg_vol20_prev = volume.rolling(20, min_periods=10).mean().shift(1)
    out["vol_ratio"] = _safe_div(volume, avg_vol20_prev, 0.0).replace([np.inf, -np.inf], np.nan)

    prior_10_high = high.rolling(10, min_periods=5).max().shift(1)
    prior_20_high = high.rolling(20, min_periods=10).max().shift(1)
    out["dist_10h"] = (_safe_div(close, prior_10_high, 1.0) - 1.0) * 100.0
    out["dist_20h"] = (_safe_div(close, prior_20_high, 1.0) - 1.0) * 100.0
    out["dist_ema20"] = (_safe_div(close, out["ema20"], 1.0) - 1.0) * 100.0
    out["dist_ema50"] = (_safe_div(close, out["ema50"], 1.0) - 1.0) * 100.0

    out["ret_1d"] = close.pct_change(1) * 100.0
    out["ret_5d"] = close.pct_change(5) * 100.0
    out["ret_20d"] = close.pct_change(20) * 100.0
    out["ema20_slope"] = out["ema20"].pct_change(1) * 100.0
    out["ema50_slope"] = out["ema50"].pct_change(1) * 100.0

    ref_close = close.replace(0, np.nan)
    out["day_range_pct"] = ((high - low) / ref_close) * 100.0
    out["upper_wick_pct"] = ((high - pd.concat([open_s, close], axis=1).max(axis=1)) / ref_close) * 100.0
    out["lower_wick_pct"] = ((pd.concat([open_s, close], axis=1).min(axis=1) - low) / ref_close) * 100.0
    out["close_to_high_pct"] = ((high - close) / ref_close) * 100.0
    out["body_pct"] = ((close - open_s) / open_s.replace(0, np.nan)) * 100.0

    out["green_candle"] = close > open_s
    out["trend_aligned"] = (close > out["ema20"]) & (out["ema20"] > out["ema50"])
    out["breakout_ready"] = (out["dist_20h"] >= -2.5) & (out["dist_20h"] <= 0.8)
    out["near_day_high"] = out["close_to_high_pct"] <= 0.8
    out["next_ret"] = close.shift(-1).div(close.replace(0, np.nan)).sub(1.0) * 100.0

    numeric_cols = [
        "ema20", "ema50", "rsi14", "vol_ratio", "dist_10h", "dist_20h",
        "dist_ema20", "dist_ema50", "ret_1d", "ret_5d", "ret_20d",
        "ema20_slope", "ema50_slope", "day_range_pct", "upper_wick_pct",
        "lower_wick_pct", "close_to_high_pct", "body_pct", "next_ret",
    ]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _historical_analog_stats(features: pd.DataFrame) -> tuple[float, int, float, float]:
    """
    Estimate tomorrow-up probability by matching today's setup against the
    stock's own historical setup analogs from local CSV history.
    """
    try:
        if features is None or features.empty or len(features) < 70:
            return 50.0, 0, 0.0, 50.0

        current = features.iloc[-1]
        hist = features.iloc[55:-1].copy()
        if hist.empty:
            return 50.0, 0, 0.0, 50.0

        hist = hist[np.isfinite(hist["next_ret"])]
        hist = hist[np.isfinite(hist["rsi14"])]
        hist = hist[np.isfinite(hist["vol_ratio"])]
        hist = hist[np.isfinite(hist["dist_20h"])]
        hist = hist[np.isfinite(hist["dist_ema20"])]
        hist = hist[np.isfinite(hist["ret_5d"])]
        hist = hist[np.isfinite(hist["ema20_slope"])]
        hist = hist[np.isfinite(hist["close_to_high_pct"])]
        if hist.empty:
            return 50.0, 0, 0.0, 50.0

        # Remove split/corporate-action style spikes so analog stats reflect
        # normal next-session behavior instead of one-off data distortions.
        hist = hist[hist["next_ret"].between(-20.0, 20.0)]
        if hist.empty:
            return 50.0, 0, 0.0, 50.0

        hist["dist_score"] = (
            (hist["rsi14"].sub(float(current["rsi14"])).abs() / 8.0)
            + (hist["vol_ratio"].sub(float(current["vol_ratio"])).abs() / 0.9)
            + (hist["dist_20h"].sub(float(current["dist_20h"])).abs() / 2.4)
            + (hist["dist_ema20"].sub(float(current["dist_ema20"])).abs() / 3.0)
            + (hist["ret_5d"].sub(float(current["ret_5d"])).abs() / 4.0)
            + (hist["ema20_slope"].sub(float(current["ema20_slope"])).abs() / 0.8)
            + (hist["close_to_high_pct"].sub(float(current["close_to_high_pct"])).abs() / 1.1)
            + (
                hist["green_candle"].astype(int)
                .sub(int(bool(current["green_candle"])))
                .abs()
                * 0.6
            )
            + (
                hist["trend_aligned"].astype(int)
                .sub(int(bool(current["trend_aligned"])))
                .abs()
                * 0.9
            )
        )

        hist = hist.sort_values("dist_score", kind="stable").head(24)
        hist = hist[hist["dist_score"] <= max(5.6, float(hist["dist_score"].median()) + 1.5)]
        if hist.empty:
            return 50.0, 0, 0.0, 50.0

        weights = 1.0 / (1.0 + hist["dist_score"].astype(float))
        weight_sum = float(weights.sum())
        if weight_sum <= 0:
            return 50.0, int(len(hist)), 0.0, 50.0

        wins = (hist["next_ret"].astype(float) > 0.0).astype(float)
        downside = (hist["next_ret"].astype(float) <= -1.25).astype(float)
        next_ret_capped = hist["next_ret"].astype(float).clip(-8.0, 8.0)
        analog_prob = float((wins * weights).sum() / weight_sum * 100.0)
        analog_downside = float((downside * weights).sum() / weight_sum * 100.0)
        analog_avg_ret = float((next_ret_capped * weights).sum() / weight_sum)
        return (
            float(np.clip(analog_prob, 0.0, 100.0)),
            int(len(hist)),
            analog_avg_ret,
            float(np.clip(analog_downside, 0.0, 100.0)),
        )
    except Exception:
        return 50.0, 0, 0.0, 50.0


# ═══════════════════════════════════════════════════════════════════════
# PRE-BREAKOUT SCORING ENGINE
# ═══════════════════════════════════════════════════════════════════════

def _pre_breakout_score(
    ri: float,
    vol_r: float,
    d20h: float,   # Δ vs 20D High (%)
    de20: float,   # Δ vs EMA20 (%)
    r5d: float,    # 5D Return (%)
) -> float:
    """
    Compute a 0–100 pre-breakout score.
    Goal: catch stocks BEFORE the move, not after.
    """
    pts = 0.0

    # ── RSI zone ─────────────────────────────────────────────────────
    # 48–58 = early accumulation (best)
    # 58–65 = developing momentum (good)
    # >70   = already extended (reject hard)
    if   48 <= ri <= 58:   pts += 22    # sweet spot
    elif 58 <  ri <= 65:   pts += 14    # ok
    elif 65 <  ri <= 70:   pts += 5     # caution
    elif ri > 70:          pts -= 30    # reject — overextended

    # ── Volume logic ──────────────────────────────────────────────────
    # >2.5  = strong; >1.2 = good; <1.0 = weak; >3.0 = exhaustion risk
    if   vol_r > 3.0:     pts += 14; pts -= 8   # strong but possible exhaustion
    elif vol_r > 2.5:     pts += 22              # strong
    elif vol_r > 1.8:     pts += 16              # good
    elif vol_r > 1.2:     pts += 10              # sufficient
    elif vol_r < 1.0:     pts -= 15              # weak — penalise

    # ── Price proximity to 20D high (-3% to 0%) ───────────────────────
    # At -1 to 0 = right at breakout zone
    # At -3 to -1 = approaching
    if   -1.0 <= d20h <= 0.0:    pts += 24   # at breakout level
    elif -3.0 <= d20h < -1.0:    pts += 14   # close
    elif -5.0 <= d20h < -3.0:    pts += 5    # some distance
    elif d20h < -5.0:             pts -= 5   # far from high

    # ── EMA20 distance — avoid overextension ─────────────────────────
    # de20 < 0 means price still below EMA → accumulation
    # de20 ≤ 3 = healthy extension
    # de20 > 6 = overextended (⚠️ see also bull-trap logic below)
    if   de20 < 0:         pts += 6    # price pulling to EMA — accumulation
    elif de20 <= 3.0:      pts += 10   # healthy
    elif de20 <= 6.0:      pts += 4    # mildly extended
    elif de20 > 6.0:       pts -= 18   # overextended — penalise

    # ── 5D return — avoid already pumped stocks ───────────────────────
    # 0.5–3.0 = gentle early move = ideal
    # 3–6     = still in range
    # >6      = already pumped — reject
    if   0.5 <= r5d <= 3.0:  pts += 12   # ideal early move
    elif 3.0 <  r5d <= 6.0:  pts += 5    # still ok
    elif r5d > 6.0:           pts -= 20  # pumped — reject
    elif r5d < 0:             pts -= 5   # losing ground

    return float(np.clip(pts, 0.0, 100.0))


def _ml_prob_heuristic(
    ri: float,
    vol_r: float,
    d20h: float,
    de20: float,
    r5d: float,
) -> float:
    """
    Deterministic ML-proxy probability (0–100).
    Simulates what a logistic model trained on early-breakout setups
    would produce, without requiring sklearn or live training data.
    """
    prob = 50.0

    # RSI
    if   48 <= ri <= 62:  prob += 8.0
    elif ri > 68:         prob -= 12.0
    elif ri < 42:         prob -= 6.0

    # Volume
    if   vol_r > 2.5:     prob += 10.0
    elif vol_r > 1.5:     prob += 5.0
    elif vol_r < 1.0:     prob -= 10.0
    if   vol_r > 3.0:     prob -= 4.0    # exhaustion haircut

    # Near breakout
    if  -2.0 <= d20h <= 0.0:    prob += 8.0
    elif -4.0 <= d20h < -2.0:   prob += 3.0

    # EMA extension
    if   de20 < 4.0:    prob += 4.0
    elif de20 > 6.0:    prob -= 10.0

    # 5D return
    if   0.5 <= r5d <= 4.0:  prob += 5.0
    elif r5d > 6.0:           prob -= 12.0

    return float(np.clip(prob, 0.0, 100.0))


def _backtest_heuristic(
    ri: float,
    vol_r: float,
    d20h: float,
    de20: float,
    r5d: float,
) -> float:
    """
    Simulated backtest win-rate (0–100).
    Models historical frequency where similar setups closed green next day.
    """
    prob = 50.0

    # RSI timing
    if   50 <= ri <= 60:   prob += 7.0
    elif 60 <  ri <= 65:   prob += 3.0
    elif ri > 70:          prob -= 10.0

    # Volume confirmation
    if   vol_r > 2.0:     prob += 8.0
    elif vol_r > 1.3:     prob += 4.0
    elif vol_r < 1.0:     prob -= 8.0

    # At breakout level
    if  -1.5 <= d20h <= 0.0:     prob += 8.0
    elif -3.0 <= d20h < -1.5:    prob += 3.0

    # EMA proximity
    if   de20 <= 3.0:   prob += 3.0
    elif de20 > 6.0:    prob -= 8.0

    # Return check
    if   0.0 <= r5d <= 4.0:   prob += 4.0
    elif r5d > 6.0:            prob -= 8.0

    return float(np.clip(prob, 0.0, 100.0))


# ═══════════════════════════════════════════════════════════════════════
# COLUMN CALCULATORS
# ═══════════════════════════════════════════════════════════════════════

def _volume_strength(vol_r: float) -> str:
    """Classify volume ratio into a human-readable strength label."""
    if vol_r > 2.8:   return "STRONG"
    if vol_r > 1.8:   return "GOOD"
    if vol_r > 1.2:   return "MODERATE"
    if vol_r >= 0.95: return "WEAK"
    return "VERY WEAK"


def _risk_flags(
    ri: float,
    vol_r: float,
    d20h: float,
    de20: float,
    r1d: float = 0.0,
    r5d: float = 0.0,
    upper_wick: float = 0.0,
    close_to_high: float = 0.0,
    trend_aligned: bool = False,
) -> list[str]:
    """Collect risk flags that can spoil tomorrow-up setups."""
    flags: list[str] = []
    if ri >= 72.0:
        flags.append("RSI overheated")
    elif ri <= 42.0:
        flags.append("Weak RSI")
    if vol_r < 1.0:
        flags.append("Low volume")
    if de20 > 6.2:
        flags.append("Extended above EMA20")
    elif de20 < -3.5:
        flags.append("Below EMA20")
    if d20h > 1.2:
        flags.append("Above breakout zone")
    elif d20h < -7.0:
        flags.append("Far from breakout")
    if r5d > 7.5:
        flags.append("Already pumped")
    elif r5d < -3.5:
        flags.append("Weak short trend")
    if r1d > 5.0:
        flags.append("One-day spike")
    if upper_wick > 1.2 and close_to_high > 0.8:
        flags.append("Seller wick")
    if (not trend_aligned) and d20h > -1.0:
        flags.append("Breakout without trend")
    return flags


def _detect_bull_trap(
    ri: float,
    vol_r: float,
    de20: float,
    d20h: float = -5.0,
    upper_wick: float = 0.0,
    close_to_high: float = 0.0,
    trend_aligned: bool = False,
) -> str:
    """Classify obvious trap setups more aggressively for tomorrow-buy filtering."""
    trap_flags = 0
    if ri >= 72.0 and vol_r < 1.25:
        trap_flags += 1
    if de20 > 6.5:
        trap_flags += 1
    if d20h > 1.0:
        trap_flags += 1
    if upper_wick > 1.3 and close_to_high > 0.8:
        trap_flags += 1
    if (not trend_aligned) and d20h > -1.0:
        trap_flags += 1
    return "⚠️ TRAP" if trap_flags >= 2 else ""


def _grade(prob: float) -> str:
    """Grade for tomorrow-up probability."""
    if prob >= 78: return "A"
    if prob >= 68: return "B"
    if prob >= 58: return "C"
    return "D"


def _signal(prob: float, confidence: float, trap: str) -> str:
    """Tighter signal thresholds for actual next-day buying decisions."""
    if trap:
        return "TRAP"
    if prob >= 72 and confidence >= 62:
        return "STRONG BUY"
    if prob >= 64 and confidence >= 56:
        return "BUY"
    if prob >= 56 and confidence >= 48:
        return "WATCH"
    return "AVOID"


def _setup_quality_score(
    ri: float,
    vol_r: float,
    d20h: float,
    de20: float,
    r1d: float = 0.0,
    r5d: float = 0.0,
    r20d: float = 0.0,
    ema20_slope: float = 0.0,
    close_to_high: float = 0.0,
    upper_wick: float = 0.0,
    trend_aligned: bool = False,
    near_day_high: bool = False,
) -> float:
    """Score current structure quality for a possible tomorrow-up move."""
    pts = 50.0

    if trend_aligned:
        pts += 13.0
    else:
        pts -= 16.0

    if ema20_slope > 0.18:
        pts += 8.0
    elif ema20_slope > 0.0:
        pts += 4.0
    else:
        pts -= 8.0

    if 52.0 <= ri <= 62.0:
        pts += 10.0
    elif 62.0 < ri <= 67.0:
        pts += 5.0
    elif ri >= 70.0:
        pts -= 12.0
    elif ri < 45.0:
        pts -= 10.0

    if 1.25 <= vol_r <= 3.0:
        pts += 10.0
    elif 1.0 <= vol_r < 1.25:
        pts += 3.0
    elif vol_r < 0.95:
        pts -= 12.0
    elif vol_r > 3.4:
        pts -= 3.0

    if -2.0 <= d20h <= 0.6:
        pts += 10.0
    elif -4.5 <= d20h < -2.0:
        pts += 4.0
    elif d20h > 1.0:
        pts -= 9.0
    elif d20h < -7.0:
        pts -= 8.0

    if -0.5 <= de20 <= 4.2:
        pts += 8.0
    elif -2.0 <= de20 < -0.5:
        pts += 2.0
    elif de20 > 6.0:
        pts -= 12.0
    elif de20 < -3.5:
        pts -= 7.0

    if 0.2 <= r1d <= 3.0:
        pts += 6.0
    elif r1d > 4.5:
        pts -= 5.0
    elif r1d < -1.5:
        pts -= 6.0

    if 0.0 <= r5d <= 5.0:
        pts += 6.0
    elif r5d > 7.0:
        pts -= 8.0
    elif r5d < -3.0:
        pts -= 6.0

    if 0.0 <= r20d <= 14.0:
        pts += 5.0
    elif r20d < -3.0:
        pts -= 7.0
    elif r20d > 20.0:
        pts -= 4.0

    if near_day_high or close_to_high <= 0.8:
        pts += 5.0
    elif close_to_high > 1.8:
        pts -= 6.0

    if upper_wick <= 0.55:
        pts += 4.0
    elif upper_wick > 1.3:
        pts -= 6.0

    return float(np.clip(pts, 0.0, 100.0))


def _confirmation_score(
    vol_r: float,
    d10h: float,
    d20h: float,
    r1d: float,
    close_to_high: float,
    body_pct: float,
    trend_aligned: bool,
    near_day_high: bool,
) -> float:
    """Trigger confirmation score for likely next-day continuation."""
    pts = 50.0

    if trend_aligned:
        pts += 10.0
    if near_day_high:
        pts += 8.0
    if -1.5 <= d10h <= 0.8:
        pts += 8.0
    elif d10h > 1.0:
        pts -= 6.0
    if -2.0 <= d20h <= 0.8:
        pts += 8.0
    elif d20h < -5.0:
        pts -= 4.0
    if vol_r >= 1.3:
        pts += 8.0
    elif vol_r < 1.0:
        pts -= 8.0
    if 0.1 <= r1d <= 3.5:
        pts += 8.0
    elif r1d > 4.5:
        pts -= 5.0
    elif r1d < 0.0:
        pts -= 6.0
    if close_to_high <= 0.6:
        pts += 5.0
    elif close_to_high > 1.4:
        pts -= 4.0
    if 0.2 <= body_pct <= 3.5:
        pts += 5.0
    elif body_pct < -1.0:
        pts -= 5.0

    return float(np.clip(pts, 0.0, 100.0))


def _confidence_score(
    setup_quality: float,
    trigger_quality: float,
    analog_count: int,
    agreement_gap: float,
    risk_count: int,
    trap: str,
) -> float:
    """Confidence in the tomorrow-up probability, not the same as probability itself."""
    analog_reliability = min(1.0, max(0.0, float(analog_count) / 18.0))
    conf = (
        0.38 * setup_quality
        + 0.28 * trigger_quality
        + 22.0 * analog_reliability
        + 12.0
    )
    conf -= min(18.0, agreement_gap * 0.45)
    conf -= float(risk_count) * 4.0
    if trap:
        conf -= 10.0
    return float(np.clip(conf, 0.0, 100.0))


def _setup_label(setup_quality: float, trigger_quality: float, trap: str) -> str:
    if trap:
        return "Trap Risk"
    combo = 0.55 * setup_quality + 0.45 * trigger_quality
    if combo >= 74.0:
        return "Strong Breakout Setup"
    if combo >= 64.0:
        return "Breakout Ready"
    if combo >= 56.0:
        return "Early Build-Up"
    return "Weak Setup"


def _buy_readiness(
    prob: float,
    confidence: float,
    analog_prob: float,
    analog_count: int,
    analog_avg_ret: float,
    analog_downside: float,
    setup_quality: float,
    trigger_quality: float,
    vol_r: float,
    close_to_high: float,
    upper_wick: float,
    risk_count: int,
    trap: str,
) -> str:
    """Final entry verdict for tomorrow-buy candidates."""
    if trap:
        return "AVOID"
    if (
        prob >= 58.0
        and confidence >= 57.0
        and analog_prob >= 45.0
        and analog_count >= 8
        and analog_avg_ret >= -0.10
        and analog_downside <= 40.0
        and setup_quality >= 74.0
        and trigger_quality >= 68.0
        and vol_r >= 0.80
        and close_to_high <= 1.40
        and upper_wick <= 1.40
        and risk_count <= 1
    ):
        return "BUY READY"
    if (
        prob >= 46.0
        and confidence >= 58.0
        and analog_prob >= 40.0
        and analog_count >= 10
        and analog_avg_ret >= -0.25
        and analog_downside <= 45.0
        and setup_quality >= 70.0
        and trigger_quality >= 64.0
        and vol_r >= 0.35
        and close_to_high <= 1.50
        and upper_wick <= 1.50
        and risk_count <= 2
    ):
        return "NEAR READY"
    return "WAIT"


def _chart_link(symbol: str) -> str:
    """TradingView chart URL for the given symbol (without .NS)."""
    sym = symbol.replace(".NS", "").strip()
    return f"https://www.tradingview.com/chart/?symbol=NSE:{sym}"


# ═══════════════════════════════════════════════════════════════════════
# ROW ENRICHER  (works on any dict with standard scan-result keys)
# ═══════════════════════════════════════════════════════════════════════

def _enrich_row(row: dict) -> dict:
    """
    Add Next-Day engine columns to a single row dict.
    Input keys (all optional with safe fallbacks):
      RSI, Vol / Avg, Δ vs 20D High (%), Δ vs EMA20 (%), 5D Return (%)
      Symbol or Ticker
    """
    ri = _sf(row.get("RSI", 50))
    vol_r = _sf(row.get("Vol / Avg", 1))
    d10h = _sf(row.get("Δ vs 10D High (%)", -5))
    d20h = _sf(row.get("Δ vs 20D High (%)", -5))
    de20 = _sf(row.get("Δ vs EMA20 (%)", 0))
    r1d = _sf(row.get("1D Return (%)", 0))
    r5d = _sf(row.get("5D Return (%)", 0))
    r20d = _sf(row.get("20D Return (%)", 0))
    ema20_slope = _sf(row.get("EMA20 Slope (%)", 0))
    close_to_high = _sf(row.get("Close to Day High (%)", 0))
    upper_wick = _sf(row.get("Upper Wick (%)", 0))
    body_pct = _sf(row.get("Body (%)", 0))
    trend_aligned = bool(row.get("Trend Aligned", False))
    near_day_high = bool(row.get("Near Day High", False))

    final_score = _pre_breakout_score(ri, vol_r, d20h, de20, r5d)
    ml_prob = _ml_prob_heuristic(ri, vol_r, d20h, de20, r5d)
    bt_prob = _backtest_heuristic(ri, vol_r, d20h, de20, r5d)

    analog_prob = _sf(row.get("_analog_prob", 50.0), 50.0)
    analog_count = int(_sf(row.get("_analog_count", 0), 0.0))
    analog_avg_ret = _sf(row.get("_analog_avg_ret", 0.0), 0.0)
    analog_downside = _sf(row.get("_analog_downside", 50.0), 50.0)

    setup_quality = _setup_quality_score(
        ri=ri,
        vol_r=vol_r,
        d20h=d20h,
        de20=de20,
        r1d=r1d,
        r5d=r5d,
        r20d=r20d,
        ema20_slope=ema20_slope,
        close_to_high=close_to_high,
        upper_wick=upper_wick,
        trend_aligned=trend_aligned,
        near_day_high=near_day_high,
    )
    trigger_quality = _confirmation_score(
        vol_r=vol_r,
        d10h=d10h,
        d20h=d20h,
        r1d=r1d,
        close_to_high=close_to_high,
        body_pct=body_pct,
        trend_aligned=trend_aligned,
        near_day_high=near_day_high,
    )

    risk_notes = _risk_flags(
        ri=ri,
        vol_r=vol_r,
        d20h=d20h,
        de20=de20,
        r1d=r1d,
        r5d=r5d,
        upper_wick=upper_wick,
        close_to_high=close_to_high,
        trend_aligned=trend_aligned,
    )
    if analog_prob < 45.0:
        risk_notes.append("Weak historical follow-through")
    elif analog_prob < 50.0:
        risk_notes.append("Mixed historical follow-through")
    if analog_avg_ret < -0.15:
        risk_notes.append("Negative analog avg return")
    if analog_downside > 38.0:
        risk_notes.append("Historical downside risk")
    trap = _detect_bull_trap(
        ri=ri,
        vol_r=vol_r,
        de20=de20,
        d20h=d20h,
        upper_wick=upper_wick,
        close_to_high=close_to_high,
        trend_aligned=trend_aligned,
    )

    analog_reliability = max(0.35, min(1.0, 0.35 + (analog_count / 20.0)))
    analog_prob_adj = 50.0 + (analog_prob - 50.0) * analog_reliability
    base_prob = (
        0.30 * final_score
        + 0.12 * ml_prob
        + 0.08 * bt_prob
        + 0.30 * setup_quality
        + 0.20 * trigger_quality
    )
    prob = 0.72 * base_prob + 0.28 * analog_prob_adj

    if analog_avg_ret > 1.0:
        prob += 2.0
    elif analog_avg_ret < -0.5:
        prob -= 3.0
    if analog_prob < 45.0:
        prob -= min(8.0, 5.0 + (45.0 - analog_prob) * 0.35)
    elif analog_prob < 50.0:
        prob -= min(4.0, 1.5 + (50.0 - analog_prob) * 0.30)
    elif analog_prob > 57.0:
        prob += min(2.5, (analog_prob - 57.0) * 0.22)
    if analog_avg_ret < -0.15:
        prob -= min(5.5, abs(analog_avg_ret) * 3.2)
    elif analog_avg_ret > 0.40:
        prob += min(2.5, analog_avg_ret * 1.8)
    if analog_downside > 36.0:
        prob -= min(8.0, (analog_downside - 36.0) * 0.35)
    elif analog_downside < 18.0 and analog_count >= 15:
        prob += 1.5
    if not near_day_high:
        prob -= 1.5
    if upper_wick > 1.0:
        prob -= min(3.0, (upper_wick - 1.0) * 4.0)
    if close_to_high > 1.0:
        prob -= min(2.5, (close_to_high - 1.0) * 4.0)
    if not trend_aligned:
        prob -= 1.0 if ema20_slope > 0.12 else 2.5
    if vol_r < 1.15:
        prob -= 1.5
    if analog_count < 12:
        prob -= 2.0

    agreement_gap = (
        abs(setup_quality - trigger_quality)
        + abs(setup_quality - analog_prob_adj)
        + abs(trigger_quality - analog_prob_adj)
    ) / 3.0
    agreement_scale = max(0.58, min(1.0, 1.0 - agreement_gap / 85.0))
    prob = 50.0 + (prob - 50.0) * agreement_scale

    risk_penalty = 4.0 * len(risk_notes)
    if trap:
        risk_penalty += 8.0
    prob = float(np.clip(prob - risk_penalty, 0.0, 100.0))

    confidence = _confidence_score(
        setup_quality=setup_quality,
        trigger_quality=trigger_quality,
        analog_count=analog_count,
        agreement_gap=agreement_gap,
        risk_count=len(risk_notes),
        trap=trap,
    )
    if analog_prob < 48.0:
        confidence -= min(8.0, (48.0 - analog_prob) * 0.40)
    if analog_avg_ret < -0.15:
        confidence -= min(5.0, abs(analog_avg_ret) * 2.4)
    if analog_downside > 36.0:
        confidence -= min(8.0, (analog_downside - 36.0) * 0.32)
    if not near_day_high:
        confidence -= 2.5
    if upper_wick > 1.0 or close_to_high > 1.0:
        confidence -= 2.0
    confidence = float(np.clip(confidence, 0.0, 100.0))

    vol_str = _volume_strength(vol_r)
    grade = _grade(prob)
    sig = _signal(prob, confidence, trap)
    setup_label = _setup_label(setup_quality, trigger_quality, trap)
    buy_readiness = _buy_readiness(
        prob=prob,
        confidence=confidence,
        analog_prob=analog_prob,
        analog_count=analog_count,
        analog_avg_ret=analog_avg_ret,
        analog_downside=analog_downside,
        setup_quality=setup_quality,
        trigger_quality=trigger_quality,
        vol_r=vol_r,
        close_to_high=close_to_high,
        upper_wick=upper_wick,
        risk_count=len(risk_notes),
        trap=trap,
    )

    sym = str(row.get("Symbol") or row.get("Ticker") or "").replace(".NS", "").strip()
    chart = _chart_link(sym)

    return {
        **row,
        "Next Day Prob":       round(prob, 1),
        "Confidence":          round(confidence, 1),
        "Grade":               grade,
        "Signal":              sig,
        "Buy Readiness":       buy_readiness,
        "Setup":               setup_label,
        "Setup Quality":       round(setup_quality, 1),
        "Trigger Quality":     round(trigger_quality, 1),
        "Historical Win %":    round(analog_prob, 1),
        "Downside Risk %":     round(analog_downside, 1),
        "Analog Count":        analog_count,
        "Analog Avg Ret %":    round(analog_avg_ret, 2),
        "Agreement Gap":       round(agreement_gap, 1),
        "Volume Strength":     vol_str,
        "Bull Trap":           trap,
        "Risk Notes":          ", ".join(risk_notes[:3]),
        "Chart Link":          chart,
    }


# ═══════════════════════════════════════════════════════════════════════
# CSV UNIVERSE BUILDER  (for when no scan results are passed in)
# ═══════════════════════════════════════════════════════════════════════

def _get_csv_tickers() -> list[str]:
    """
    List all .csv files in DATA_DIR and return as ticker_ns strings.
    File naming convention from data_downloader: RELIANCE.NS.csv
    """
    try:
        return [p.stem for p in DATA_DIR.glob("*.csv")]
    except Exception:
        return []


def _build_row_from_csv(ticker_ns: str) -> dict | None:
    """
    Load the CSV for `ticker_ns`, compute the latest-day indicators,
    and return a partial row dict compatible with _enrich_row().
    Returns None if data is missing or insufficient.
    """
    try:
        df = load_csv(ticker_ns)
        if df is None or len(df) < 70:
            return None

        df = df.tail(220).copy()
        features = _prepare_feature_frame(df)
        if features is None or len(features) < 70:
            return None

        close = features["Close"].astype(float)
        if len(close) < 21:
            return None

        current = features.iloc[-1]
        if not np.isfinite(float(current.get("rsi14", np.nan))):
            return None

        close_val = float(close.iloc[-1])
        analog_prob, analog_count, analog_avg_ret, analog_downside = _historical_analog_stats(features)

        sym = ticker_ns.replace(".NS", "")

        return {
            "Symbol":                sym,
            "Price (₹)":             round(close_val, 2),
            "RSI":                   round(float(current["rsi14"]), 1),
            "Vol / Avg":             round(_sf(current["vol_ratio"], 0.0), 2),
            "EMA 20":                round(_sf(current["ema20"], close_val), 2),
            "EMA 50":                round(_sf(current["ema50"], close_val), 2),
            "Δ vs 10D High (%)":     round(_sf(current["dist_10h"], -5.0), 2),
            "Δ vs 20D High (%)":     round(_sf(current["dist_20h"], -5.0), 2),
            "Δ vs EMA20 (%)":        round(_sf(current["dist_ema20"], 0.0), 2),
            "Δ vs EMA50 (%)":        round(_sf(current["dist_ema50"], 0.0), 2),
            "1D Return (%)":         round(_sf(current["ret_1d"], 0.0), 2),
            "5D Return (%)":         round(_sf(current["ret_5d"], 0.0), 2),
            "20D Return (%)":        round(_sf(current["ret_20d"], 0.0), 2),
            "EMA20 Slope (%)":       round(_sf(current["ema20_slope"], 0.0), 3),
            "Close to Day High (%)": round(_sf(current["close_to_high_pct"], 0.0), 2),
            "Upper Wick (%)":        round(_sf(current["upper_wick_pct"], 0.0), 2),
            "Body (%)":              round(_sf(current["body_pct"], 0.0), 2),
            "Trend Aligned":         bool(current.get("trend_aligned", False)),
            "Near Day High":         bool(current.get("near_day_high", False)),
            "_analog_prob":          round(analog_prob, 3),
            "_analog_count":         int(analog_count),
            "_analog_avg_ret":       round(analog_avg_ret, 4),
            "_analog_downside":      round(analog_downside, 3),
        }
    except Exception:
        return None   # never crash


# ═══════════════════════════════════════════════════════════════════════
# MAIN PUBLIC FUNCTION
# ═══════════════════════════════════════════════════════════════════════

def run_csv_next_day(df: pd.DataFrame | None) -> pd.DataFrame:
    """
    CSV Next-Day Potential Engine — main entry point.

    Parameters
    ----------
    df : pd.DataFrame | None
        Either:
          (a) An existing scan-results DataFrame from NSE Sentinel's main
              scanner (columns: Symbol, RSI, Vol / Avg, Δ vs 20D High (%),
              Δ vs EMA20 (%), 5D Return (%) …)
          (b) None / empty DataFrame → engine scans DATA_DIR CSVs itself.

    Returns
    -------
    pd.DataFrame
        Enhanced DataFrame with new columns:
          "Next Day Prob"   — 0–100 composite early-move probability
          "Confidence"      — confidence in the tomorrow-up probability
          "Grade"           — A / B / C / D
          "Signal"          — STRONG BUY / BUY / WATCH / AVOID / TRAP
          "Buy Readiness"   — BUY READY / NEAR READY / WAIT / AVOID
          "Setup"           — setup classification
          "Historical Win %"— same-stock analog win rate from local CSV history
          "Downside Risk %" — weighted analog downside probability
          "Analog Count"    — count of matched analog setups
          "Volume Strength" — STRONG / GOOD / MODERATE / WEAK / VERY WEAK
          "Bull Trap"       — '⚠️ TRAP' or ''
          "Chart Link"      — TradingView URL

        Filtered for actionable tomorrow-buy setups.
    """

    rows_source: list[dict] = []

    # ── Branch A: enrich existing scan results ────────────────────────
    if df is not None and not df.empty and "RSI" in df.columns:
        for _, row in df.iterrows():
            rows_source.append(row.to_dict())

    # ── Branch B: scan local CSV files to build a universe ───────────
    else:
        if not _DOWNLOADER_OK:
            # data_downloader not available — return empty
            return pd.DataFrame()

        csv_tickers = _get_csv_tickers()
        if not csv_tickers:
            return pd.DataFrame()

        with ThreadPoolExecutor(max_workers=16) as ex:
            futs = {ex.submit(_build_row_from_csv, t): t for t in csv_tickers}
            for fut in as_completed(futs):
                try:
                    result = fut.result()
                    if result is not None:
                        rows_source.append(result)
                except Exception:
                    pass   # never crash

    if not rows_source:
        return pd.DataFrame()

    # ── Enrich every row ──────────────────────────────────────────────
    enriched: list[dict] = []
    for row in rows_source:
        try:
            enriched.append(_enrich_row(row))
        except Exception:
            pass   # skip malformed rows silently

    if not enriched:
        return pd.DataFrame()

    out = pd.DataFrame(enriched)

    # ── Filter (not too strict) ───────────────────────────────────────
    # Keep: Prob > 55 AND Volume Strength != "VERY WEAK"
    if {"Buy Readiness", "Bull Trap", "Signal", "Next Day Prob", "Confidence"}.issubset(out.columns):
        out = out[
            (out["Bull Trap"].fillna("") == "") &
            (out["Buy Readiness"].isin(["BUY READY", "NEAR READY"])) &
            (out["Next Day Prob"] >= 46.0) &
            (out["Confidence"] >= 58.0)
        ].copy()

    if out.empty:
        return out

    # ── Sort by Next Day Prob descending ─────────────────────────────
    signal_rank = {
        "STRONG BUY": 3,
        "BUY": 2,
        "WATCH": 1,
        "AVOID": 0,
        "TRAP": -1,
    }
    readiness_rank = {
        "BUY READY": 2,
        "NEAR READY": 1,
        "WAIT": 0,
        "AVOID": -1,
    }
    out["_signal_rank"] = out.get("Signal", pd.Series(index=out.index, dtype=object)).map(signal_rank).fillna(0)
    out["_readiness_rank"] = out.get("Buy Readiness", pd.Series(index=out.index, dtype=object)).map(readiness_rank).fillna(0)
    sort_cols = [c for c in ["_readiness_rank", "_signal_rank", "Next Day Prob", "Confidence", "Historical Win %", "Setup Quality"] if c in out.columns]
    out = out.sort_values(sort_cols, ascending=[False] * len(sort_cols), kind="stable").reset_index(drop=True)
    out = out.drop(columns=["_signal_rank", "_readiness_rank"], errors="ignore")
    out = out.drop(columns=[c for c in out.columns if str(c).startswith("_")], errors="ignore")

    return out
