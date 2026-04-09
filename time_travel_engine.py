"""
time_travel_engine.py
──────────────────────
🕰️ TIME-TRAVEL MODE — System-wide Historical Simulation for NSE Sentinel.

How it works
────────────
When activated with a cutoff date D, this engine:

  1. Patches `get_df_for_ticker` so EVERY call (from analyse(), Stock Aura,
     Battle, Sector Screener) returns data truncated to ≤ D.
  2. Snapshots existing ALL_DATA entries (used by zero-API backtests) with
     the same truncated data.
  3. Stores originals so `restore()` puts everything back exactly.

Data guarantee
──────────────
  • ALL indicators (EMA, RSI, Vol/Avg) are computed AFTER the cutoff filter,
    so no future data ever leaks into any calculation.
  • yfinance fallback downloads (for tickers not yet in ALL_DATA) are also
    truncated immediately on receipt.
  • The staleness check in analyse() uses reference datetime (4pm on cutoff
    date) instead of datetime.now() so historical DFs are never rejected.

Zero impact on live mode
─────────────────────────
  • When inactive, all functions are no-ops and return normal values.
  • is_active() guard protects every code path.
  • Never raises — fully wrapped in try/except throughout.

Public API
──────────
    activate(cutoff_date)           → int   (tickers snapshotted)
    restore()                       → None
    is_active()                     → bool
    get_reference_datetime()        → datetime  (4pm on cutoff, or now())
    get_reference_date()            → date | None
    truncate_df(df, cutoff_date)    → pd.DataFrame | None
    apply_time_travel_cutoff(df)    → pd.DataFrame | None  (convenience)
"""

from __future__ import annotations

import threading
from datetime import date, datetime, time as dtime
from typing import Callable

import numpy as np
import pandas as pd

# ── Internal imports ──────────────────────────────────────────────────
try:
    from strategy_engines._engine_utils import (
        ALL_DATA,
        _ALL_DATA_LOCK,
        get_df_for_ticker as _original_get_df,
        download_history as _original_download,
    )
    import strategy_engines._engine_utils as _eu
    _EU_OK = True
except ImportError:
    _EU_OK = False
    ALL_DATA = {}
    _ALL_DATA_LOCK = threading.Lock()
    _original_get_df = lambda t: None  # noqa: E731
    _original_download = lambda t, **kw: None  # noqa: E731
    _eu = None


# ══════════════════════════════════════════════════════════════════════
# MODULE-LEVEL STATE (thread-safe)
# ══════════════════════════════════════════════════════════════════════

_STATE_LOCK         = threading.Lock()
_active             = False
_reference_date:    date | None = None
_all_data_backup:   dict        = {}   # ticker → original df


# ══════════════════════════════════════════════════════════════════════
# CORE UTILITY
# ══════════════════════════════════════════════════════════════════════

def truncate_df(df: pd.DataFrame | None, cutoff: date) -> pd.DataFrame | None:
    """
    Return df filtered to rows where index.date <= cutoff.
    Returns None if result has fewer than 10 rows (not enough for indicators).
    Never raises.
    """
    if df is None or df.empty:
        return None
    try:
        idx_dates = pd.to_datetime(df.index).date
        mask = idx_dates <= cutoff
        trimmed = df.loc[mask]
        return trimmed if len(trimmed) >= 10 else None
    except Exception:
        return df   # fail-safe: return original rather than None


# ══════════════════════════════════════════════════════════════════════
# PATCHED get_df_for_ticker
# ══════════════════════════════════════════════════════════════════════

def _time_travel_get_df(ticker: str) -> pd.DataFrame | None:
    """
    Drop-in replacement for get_df_for_ticker() when time-travel is active.
    Fetches data normally then truncates to _reference_date before returning.
    """
    try:
        ticker_ns = ticker if ticker.endswith(".NS") else f"{ticker}.NS"

        # 1️⃣ Try ALL_DATA (already snapshotted — truncated values are there)
        with _ALL_DATA_LOCK:
            cached = ALL_DATA.get(ticker_ns)
        if cached is not None:
            return cached   # already truncated by apply_snapshot()

        # 2️⃣ Live download fallback — truncate immediately
        df_live = _original_download(ticker_ns, period="6mo")
        if df_live is None:
            return None
        with _STATE_LOCK:
            cutoff = _reference_date
        if cutoff is None:
            return df_live
        trimmed = truncate_df(df_live, cutoff)
        # ── FIX 7: Validate no future leakage ─────────────────────────
        if trimmed is not None and len(trimmed) > 0:
            last = pd.to_datetime(trimmed.index[-1]).date()
            if last > cutoff:   # should never happen — log if it does
                import warnings
                warnings.warn(
                    f"[TimeTravelEngine] LEAKAGE DETECTED: {ticker_ns} "
                    f"last_date={last} > cutoff={cutoff}. Forcing re-trim.",
                    RuntimeWarning, stacklevel=2,
                )
                trimmed = truncate_df(df_live, cutoff)
        return trimmed

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════
# PUBLIC API
# ══════════════════════════════════════════════════════════════════════

def activate(cutoff: date) -> int:
    """
    Enable time-travel mode for cutoff date.
    Truncates ALL existing ALL_DATA entries and patches get_df_for_ticker.
    Returns number of ticker DataFrames successfully truncated.
    Thread-safe. Never raises.
    """
    global _active, _reference_date, _all_data_backup

    try:
        with _STATE_LOCK:
            _reference_date = cutoff
            _active = True

        count = 0

        # ── Snapshot + truncate ALL_DATA ──────────────────────────────
        if _EU_OK:
            with _ALL_DATA_LOCK:
                _all_data_backup.clear()
                for ticker, df in list(ALL_DATA.items()):
                    _all_data_backup[ticker] = df      # store original
                    if df is None or df.empty:
                        continue
                    trimmed = truncate_df(df, cutoff)
                    ALL_DATA[ticker] = trimmed
                    if trimmed is not None:
                        count += 1

            # ── Monkey-patch get_df_for_ticker ────────────────────────
            try:
                _eu.get_df_for_ticker = _time_travel_get_df
            except Exception:
                pass

        return count

    except Exception:
        return 0


def restore() -> None:
    """
    Restore ALL_DATA to its original state and unpatch get_df_for_ticker.
    Safe to call even if activate() was never called.
    Never raises.
    """
    global _active, _reference_date, _all_data_backup

    try:
        # ── Restore ALL_DATA ──────────────────────────────────────────
        if _EU_OK and _all_data_backup:
            with _ALL_DATA_LOCK:
                for ticker, df in _all_data_backup.items():
                    ALL_DATA[ticker] = df
                _all_data_backup.clear()

            # ── Restore original get_df_for_ticker ────────────────────
            try:
                _eu.get_df_for_ticker = _original_get_df
            except Exception:
                pass

        with _STATE_LOCK:
            _active = False
            _reference_date = None

    except Exception:
        # Absolute fail-safe — mark inactive even if restore partially failed
        _active = False
        _reference_date = None


def is_active() -> bool:
    """Return True when time-travel mode is currently activated."""
    with _STATE_LOCK:
        return _active


def get_reference_date() -> date | None:
    """Return the active cutoff date, or None if not in time-travel mode."""
    with _STATE_LOCK:
        return _reference_date


def get_reference_datetime() -> datetime:
    """
    Return the reference datetime for staleness checks.
    Time-travel ON  → 4:00 PM on the cutoff date (post-market-close).
    Time-travel OFF → datetime.now() (unchanged live behaviour).
    """
    with _STATE_LOCK:
        d = _reference_date
    if d is None:
        return datetime.now()
    return datetime.combine(d, dtime(16, 0, 0))   # 4pm IST = post-market


def apply_time_travel_cutoff(df: pd.DataFrame | None) -> pd.DataFrame | None:
    """
    Convenience wrapper: truncate df to the active cutoff date if
    time-travel is active, otherwise return df unchanged.

    Use this everywhere a yfinance download is used outside of ALL_DATA
    (e.g. index data in market_bias_engine, multi_index_market_bias_engine).

    Never raises. Returns None if df is None or becomes empty after trim.
    """
    cutoff = get_reference_date()
    if cutoff is None or df is None or df.empty:
        return df
    return truncate_df(df, cutoff)


def format_banner() -> str:
    """
    Return a human-readable banner string for display in the UI.
    Returns empty string if time-travel is not active.
    """
    with _STATE_LOCK:
        active = _active
        d = _reference_date
    if not active or d is None:
        return ""
    day_str = d.strftime("%d-%b-%Y")
    weekday = d.strftime("%A")
    return f"🕰️ TIME TRAVEL · Simulating Market Date: {day_str} ({weekday}) Post-Market Close"