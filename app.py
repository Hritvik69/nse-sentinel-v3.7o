"""
NSE Sentinel — Production-Ready Streamlit App  (Enhanced Edition)
Dark terminal aesthetic | Multi-strategy scanner | 1000+ NSE stocks
Run: streamlit run app.py

CHANGES vs original:
  • Scoring layer  (compute_score)         — added AFTER scan, never touches filters
  • Backtest prob  (compute_backtest_prob)  — added AFTER scan
  • ML probability (train_model_once /
                    predict_ml_probability) — added AFTER scan
  • Final rank     (enhance_results)        — combines the three
  • Bull-trap warning                       — display only
  • Top Picks section                       — display only
  • All existing analyse() / run_scan()
    logic is 100 % untouched.
"""
#nse-sentinel-v3o-jmasdj9sgpde4gertzb3do
#.\.venv\Scripts\python.exe -m streamlit run app.py
from __future__ import annotations
# At the top of app.py

import io
import threading
import time
import warnings
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf
from strategy_engines import (
    get_engine_functions,
    preload_all,
    backtest_with_preloaded,
    get_df_for_ticker,
)
from strategy_engines._engine_utils import preload_history_batch

try:
    from strategy_engines.app_sector_screener_dashboard import (
        render_sector_screener_dashboard,
    )
    _SECTOR_SCREENER_UI_OK = True
except ImportError:
    _SECTOR_SCREENER_UI_OK = False

try:
    from strategy_engines.csv_next_day_engine import run_csv_next_day
    _CSV_NEXT_DAY_ENGINE_OK = True
except ImportError:
    _CSV_NEXT_DAY_ENGINE_OK = False

    def run_csv_next_day(df=None):  # type: ignore[misc]
        return pd.DataFrame()

warnings.filterwarnings("ignore")

# ── TradingView symbol helper ─────────────────────────────────────────
def tv_symbol(ticker: str) -> str:
    """
    Convert yfinance NSE ticker to TradingView symbol.
    e.g. RELIANCE.NS → NSE:RELIANCE
    """
    return "NSE:" + ticker.replace(".NS", "")


def tv_chart_url(symbol_no_ns: str) -> str:
    """Return TradingView chart URL for a bare symbol (no .NS suffix)."""
    return f"https://www.tradingview.com/chart/?symbol=NSE:{symbol_no_ns}"


# ── Data downloader (optional, graceful if missing) ───────────────────
try:
    from data_downloader import (
        update_data_if_old,
        update_all_data,
        data_status_summary,
        bulk_download,
        load_csv,
    )
    _DATA_DOWNLOADER_OK = True
except ImportError:
    _DATA_DOWNLOADER_OK = False

    def update_data_if_old(tickers, **kw):  # type: ignore[misc]
        return 0

    def update_all_data(tickers, **kw):  # type: ignore[misc]
        return {"updated": 0, "skipped": 0, "failed": 0}

    def data_status_summary(tickers):  # type: ignore[misc]
        return {}

    def bulk_download(tickers, **kw):  # type: ignore[misc]
        return {}

    def load_csv(ticker):  # type: ignore[misc]
        return None

# ── Grading engine (optional, graceful if missing) ────────────────────
try:
    from grading_engine import apply_universal_grading
    _GRADING_OK = True
except ImportError:
    _GRADING_OK = False

    def apply_universal_grading(df, market_bias=None):  # type: ignore[misc]
        return df

# ── Enhanced logic engine (optional, graceful if missing) ─────────────
try:
    from enhanced_logic_engine import apply_enhanced_logic
    _ENHANCED_LOGIC_OK = True
except ImportError:
    _ENHANCED_LOGIC_OK = False

    def apply_enhanced_logic(df):  # type: ignore[misc]
        return df

# ── Phase 4 logic engine (optional, graceful if missing) ──────────────
try:
    from phase4_logic_engine import apply_phase4_logic, apply_phase42_logic
    _PHASE4_LOGIC_OK = True
except ImportError:
    _PHASE4_LOGIC_OK = False

    def apply_phase4_logic(df, market_bias=None):  # type: ignore[misc]
        return df

    def apply_phase42_logic(df):  # type: ignore[misc]
        return df

# ── Time Travel engine (optional, graceful if missing) ────────────────
try:
    import time_travel_engine as _tt
    _TIME_TRAVEL_OK = True
except ImportError:
    _TIME_TRAVEL_OK = False

    class _tt:  # type: ignore[no-redef]
        """Silent stub — all calls are no-ops when engine file is missing."""
        @staticmethod
        def is_active() -> bool:          return False
        @staticmethod
        def get_reference_datetime():     return datetime.now()
        @staticmethod
        def get_reference_date():         return None
        @staticmethod
        def format_banner() -> str:       return ""
        @staticmethod
        def activate(d) -> int:           return 0
        @staticmethod
        def restore() -> None:            pass

# ── Stock Aura — fully inlined (no external file dependency) ─────────

def _aura_ema(s: "pd.Series", n: int) -> "pd.Series":
    return s.ewm(span=n, adjust=False).mean()

def _aura_rsi_last(close: "pd.Series", period: int = 14) -> float:
    try:
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])
    except Exception:
        return 50.0

def _aura_fetch(symbol: str) -> "pd.DataFrame | None":
    """Fetch OHLCV; always truncates to TT cutoff if active — no leakage."""
    ticker_ns = symbol.upper().strip()
    if not ticker_ns.endswith(".NS"):
        ticker_ns += ".NS"

    # Use the engine's authoritative cutoff (works even in worker threads)
    cutoff = _tt.get_reference_date()  # None when TT is off

    def _cut(df):
        if df is None or df.empty or cutoff is None:
            return df
        try:
            mask = pd.to_datetime(df.index).date <= cutoff
            t    = df.loc[mask]
            return t if len(t) >= 10 else None
        except Exception:
            return df

    # 1️⃣ Try ALL_DATA cache — _cut enforces cutoff even if not pre-truncated
    try:
        df = get_df_for_ticker(ticker_ns)
        if df is not None and len(df) >= 10:
            return _cut(df)
    except Exception:
        pass

    # 2️⃣ Live yfinance fallback
    try:
        df = yf.download(
            ticker_ns, period="6mo", interval="1d",
            auto_adjust=True, progress=False, timeout=15, threads=False,
        )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.columns = [c.strip().title() for c in df.columns]
        df = df.dropna(subset=["Close", "Volume"])
        return _cut(df) if len(df) >= 10 else None
    except Exception:
        return None


def _aura_engine(df: "pd.DataFrame", symbol: str, market_bias: dict) -> dict:
    """
    Run all 8 Aura checks; return a result dict.
    Never raises — returns AVOID on any error.
    """
    r = dict(
        symbol=symbol.upper().replace(".NS", ""),
        price=0.0, rsi=50.0, ema20=0.0, ema50=0.0,
        vol_ratio=1.0, delta_ema20=0.0, delta_20h=0.0,
        ret_5d=0.0, ret_20d=0.0, rr_ratio=0.0,
        verdict="❌ AVOID", timing="NO TRADE", verdict_color="#ff4d6d",
        setup_type="None", trend_ok=False, volume_ok=False,
        momentum_ok=False, entry_ok=False, sl_quality="Poor",
        rr_ok=False, market_note="",
        pos=[], warn=[], rej=[],
    )
    try:
        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        high_s = df["High"].dropna()  if "High"  in df.columns else close
        low_s  = df["Low"].dropna()   if "Low"   in df.columns else close

        if len(close) < 30:
            r["rej"].append("Insufficient price history")
            return r

        lc       = float(close.iloc[-1])
        e20      = float(_aura_ema(close, 20).iloc[-1])
        e50      = float(_aura_ema(close, 50).iloc[-1])
        prev_e20 = float(_aura_ema(close, 20).iloc[-2]) if len(close) >= 2 else e20
        rsi_v    = _aura_rsi_last(close)

        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        lv      = float(volume.iloc[-1])
        vol_r   = round(lv / avg_vol, 2) if avg_vol > 0 else 1.0

        h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
        ret_5d  = (lc / float(close.iloc[-6])  - 1) * 100 if len(close) >= 6  else 0.0
        ret_20d = (lc / float(close.iloc[-21]) - 1) * 100 if len(close) >= 21 else 0.0
        d_ema20 = (lc / e20  - 1) * 100 if e20  > 0 else 0.0
        d_20h   = (lc / h20  - 1) * 100 if h20  > 0 else 0.0

        # Risk-reward
        if d_20h >= -1.5:
            target = lc * 1.06          # breakout: project 6% continuation
        else:
            target = h20                # pullback: prior high
        downside = max(lc - e20, 0.01) if lc > e20 > 0 else 0.01
        rr       = max(target - lc, 0.0) / downside

        r.update(price=round(lc,2), rsi=round(rsi_v,1), ema20=round(e20,2),
                 ema50=round(e50,2), vol_ratio=round(vol_r,2),
                 delta_ema20=round(d_ema20,2), delta_20h=round(d_20h,2),
                 ret_5d=round(ret_5d,2), ret_20d=round(ret_20d,2),
                 rr_ratio=round(rr,2))

        # 1 — Trend
        if lc > e20 > e50:
            r["trend_ok"] = True
            r["pos"].append("Strong uptrend (Price > EMA20 > EMA50)")
            if e20 > prev_e20:
                r["pos"].append("EMA20 slope rising — momentum intact")
        elif lc > e20:
            r["warn"].append("Price above EMA20 but EMA20 < EMA50 — weak structure")
        else:
            r["rej"].append("Downtrend — price below EMA20")

        # 2 — Setup
        at_zone   = -1.5 <= d_20h <= 0.5
        pb_zone   = -6.0 <= d_20h <  -1.5
        if at_zone and vol_r >= 1.5:
            r["setup_type"] = "Breakout"
            r["pos"].append("Breakout setup — price at 20D high with volume")
        elif at_zone:
            r["setup_type"] = "Pullback"
            r["warn"].append("Near 20D high but volume not confirming — wait for vol")
        elif pb_zone and e20 > prev_e20:
            r["setup_type"] = "Pullback"
            r["pos"].append("Healthy pullback to EMA support — potential re-entry")
        elif d_20h < -6.0:
            r["setup_type"] = "None"
            r["rej"].append(f"Too far from 20D high ({d_20h:.1f}%) — no valid entry")
        else:
            r["setup_type"] = "Pullback"
            r["warn"].append("Setup not fully formed — borderline zone")

        # 3 — Volume
        if vol_r >= 1.5:
            r["volume_ok"] = True
            r["pos"].append(f"Volume strong ({vol_r:.1f}× avg) — institutional participation")
        elif vol_r >= 1.3:
            r["volume_ok"] = True
            r["pos"].append(f"Volume valid ({vol_r:.1f}× avg) — acceptable participation")
        elif vol_r >= 1.0:
            r["warn"].append(f"Volume weak ({vol_r:.1f}× avg) — no conviction")
        else:
            r["rej"].append(f"Volume below average ({vol_r:.1f}×) — distribution risk")

        # 4 — Momentum
        if rsi_v > 75:
            r["rej"].append(f"RSI overbought ({rsi_v:.0f}) — late-stage entry risk")
        elif ret_5d > 12.0:
            r["rej"].append(f"5D return {ret_5d:.1f}% — short-term exhaustion risk")
        elif 50 <= rsi_v <= 70 and 2 <= ret_5d <= 10:
            r["momentum_ok"] = True
            r["pos"].append(f"RSI healthy ({rsi_v:.0f}) with strong 5D return ({ret_5d:.1f}%)")
        elif 50 <= rsi_v <= 70:
            r["momentum_ok"] = True
            r["pos"].append(f"RSI healthy ({rsi_v:.0f}) — momentum not stretched")
        elif 70 < rsi_v <= 75:
            r["warn"].append(f"RSI elevated ({rsi_v:.0f}) — caution zone")
        else:
            r["momentum_ok"] = True
            r["warn"].append(f"RSI low ({rsi_v:.0f}) — early accumulation stage")

        # 5 — Entry quality
        if d_ema20 <= 3.0:
            r["entry_ok"] = True
            r["pos"].append(f"Close to EMA20 ({d_ema20:.1f}%) — tight entry")
        elif d_ema20 <= 6.0:
            r["entry_ok"] = True
            r["warn"].append(f"Moderately extended from EMA20 ({d_ema20:.1f}%)")
        else:
            r["rej"].append(f"Overextended from EMA20 ({d_ema20:.1f}%) — late entry")

        # 6 — Stop quality
        if d_ema20 <= 3.0:
            r["sl_quality"] = "Tight"
            r["pos"].append(f"Tight stop ({d_ema20:.1f}% to EMA20)")
        elif d_ema20 <= 6.0:
            r["sl_quality"] = "Medium"
            r["warn"].append(f"Medium stop distance ({d_ema20:.1f}% to EMA20)")
        else:
            r["sl_quality"] = "Poor"
            r["rej"].append(f"Wide stop ({d_ema20:.1f}% to EMA20) — poor structure")

        # 7 — Risk-reward
        if rr >= 2.0:
            r["rr_ok"] = True
            r["pos"].append(f"Risk-reward {rr:.1f}:1 — excellent setup")
        elif rr >= 1.5:
            r["rr_ok"] = True
            r["pos"].append(f"Risk-reward {rr:.1f}:1 — acceptable")
        elif rr >= 1.0:
            r["warn"].append(f"Risk-reward {rr:.1f}:1 — marginal, prefer ≥2:1")
        else:
            r["rej"].append(f"Risk-reward {rr:.1f}:1 — unfavorable")

        # 8 — Market context
        mb  = market_bias if isinstance(market_bias, dict) else {}
        lbl = str(mb.get("bias", mb.get("regime", ""))).strip()
        if lbl:
            if any(w in lbl.lower() for w in ("bearish","weak","caution")):
                r["market_note"] = f"Market: {lbl} ⚠️"
                r["warn"].append(f"Market is {lbl} — reduce position size")
            elif any(w in lbl.lower() for w in ("bullish","trending up","strong")):
                r["market_note"] = f"Market: {lbl} ✅"
                r["pos"].append(f"Favorable market backdrop ({lbl})")
            else:
                r["market_note"] = f"Market: {lbl}"
        else:
            r["market_note"] = "Market context unavailable — run Market Bias first"

        # Verdict
        rej_n  = len(r["rej"])
        warn_n = len(r["warn"])
        rb = r

        is_buy_now = (
            rb["setup_type"] == "Breakout" and rb["trend_ok"] and
            rb["volume_ok"] and rb["momentum_ok"] and
            rb["entry_ok"] and rb["rr_ok"] and rej_n == 0
        )
        is_buy_conf = (
            rb["setup_type"] in ("Breakout","Pullback") and
            rb["trend_ok"] and rb["momentum_ok"] and
            (rb["rr_ok"] or rr >= 1.0) and rej_n == 0
        )
        is_watch = (
            rb["trend_ok"] and rb["setup_type"] != "None" and
            rej_n <= 1 and rsi_v <= 75
        )

        if is_buy_now:
            r["verdict"]       = "🔥 BUY RIGHT NOW"
            r["timing"]        = "BUY NOW"
            r["verdict_color"] = "#00d4a8"
        elif is_buy_conf:
            r["verdict"]       = "✅ BUY (ON CONFIRMATION)"
            r["timing"]        = "BUY TOMORROW"
            r["verdict_color"] = "#0094ff"
        elif is_watch:
            r["verdict"]       = "👀 WATCH"
            r["timing"]        = "WAIT"
            r["verdict_color"] = "#f0b429"
        else:
            r["verdict"]       = "❌ AVOID"
            r["timing"]        = "NO TRADE"
            r["verdict_color"] = "#ff4d6d"

    except Exception as exc:
        r["rej"].append(f"Engine error: {exc}")
    return r


def _aura_timing_badge(timing: str, vc: str) -> str:
    return (
        f'<span style="background:{vc}20;border:1px solid {vc};border-radius:6px;'
        f'padding:3px 10px;font-size:12px;font-weight:700;color:{vc};">{timing}</span>'
    )

def _aura_factor_row(label: str, value: str, color: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480;">{label}</span>'
        f'<span style="font-size:12px;font-weight:700;color:{color};">{value}</span></div>'
    )

def render_stock_aura_panel() -> None:
    """Fully self-contained Stock Aura panel — no external file needed."""
    if not st.session_state.get("aura_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)

    # Header
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:22px;'
        'color:#ccd9e8;margin-bottom:4px;">🔮 Stock Aura</h2>'
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        "Single stock decision engine — a trader's brain, not a screener</div>",
        unsafe_allow_html=True,
    )

    # TT banner inside Aura
    _aura_tt = st.session_state.get("aura_tt_date")
    if _aura_tt is not None:
        try:
            _aura_tt_str = _aura_tt.strftime("%d %b %Y")
        except Exception:
            _aura_tt_str = str(_aura_tt)
        st.markdown(
            f'<div style="background:#1a0a00;border:1.5px solid #f0b429;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:14px;font-size:12px;color:#f0b429;">'
            f'🕰️ <b>TIME TRAVEL ACTIVE</b> — Evaluating {_aura_tt_str} post-market · '
            f'No future data used</div>',
            unsafe_allow_html=True,
        )

    # Input row
    c_in, c_btn, c_cls = st.columns([3, 1, 1])
    with c_in:
        ticker_raw = st.text_input(
            "Stock Symbol", placeholder="e.g.  RELIANCE  or  TCS  or  HDFCBANK",
            key="aura_ticker_input", label_visibility="collapsed",
        )
    with c_btn:
        analyze_clicked = st.button("🧠 Analyze Aura", key="aura_analyze_btn")
    with c_cls:
        if st.button("✕ Close", key="aura_close_btn"):
            st.session_state["aura_show_panel"] = False
            st.rerun()

    if not (analyze_clicked and ticker_raw.strip()):
        return

    sym = ticker_raw.strip().upper().replace(".NS", "")
    _spin = (
        f"🕰️ Historical aura for {sym} ({_aura_tt_str})…"
        if _aura_tt else f"🔮 Reading aura for {sym}…"
    )
    with st.spinner(_spin):
        df = _aura_fetch(sym)

    if df is None or df.empty:
        st.error(
            f"❌ No data for **{sym}**. Check the symbol "
            "(e.g. RELIANCE, not RELIANCE.NS) and try again."
        )
        return

    mb = dict(st.session_state.get("market_bias_result") or {})
    if _aura_tt and not mb.get("bias"):
        mb["bias"] = f"Historical ({_aura_tt_str}) — run Market Bias for that date"

    res = _aura_engine(df, sym, mb)
    vc  = res["verdict_color"]

    # ── Verdict card ─────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#0b1017;border:2px solid {vc};border-radius:14px;'
        f'padding:20px 24px;margin:12px 0 20px;">'
        f'<div style="font-size:13px;color:#4a6480;letter-spacing:1px;'
        f'text-transform:uppercase;margin-bottom:4px;">🔮 STOCK AURA RESULT</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;'
        f'color:#ccd9e8;margin-bottom:2px;">{res["symbol"]}</div>'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:14px;">'
        f'₹{res["price"]:.2f} &nbsp;|&nbsp; RSI {res["rsi"]:.0f} &nbsp;|&nbsp; '
        f'Vol {res["vol_ratio"]:.1f}× &nbsp;|&nbsp; EMA20 {res["delta_ema20"]:+.1f}% '
        f'&nbsp;|&nbsp; 5D {res["ret_5d"]:+.1f}%</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:22px;font-weight:900;'
        f'color:{vc};margin-bottom:10px;">{res["verdict"]}</div>'
        f'Timing: {_aura_timing_badge(res["timing"], vc)}'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_l, col_r = st.columns([3, 2])

    # ── Why / Warnings ────────────────────────────────────────────────
    with col_l:
        if res["pos"]:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#00d4a8;'
                'letter-spacing:.5px;margin-bottom:8px;">WHY ✔</div>' +
                "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:#00d4a8;font-weight:700;">✔</span> &nbsp;{t}</div>'
                    for t in res["pos"]
                ) + '</div>',
                unsafe_allow_html=True,
            )

        issues = [(t, "#f0b429") for t in res["warn"]] + [(t, "#ff4d6d") for t in res["rej"]]
        if issues:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #3a1e1e;'
                'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
                '<div style="font-size:11px;font-weight:700;color:#ff4d6d;'
                'letter-spacing:.5px;margin-bottom:8px;">WARNINGS / REJECTIONS ✖</div>' +
                "".join(
                    f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                    f'<span style="color:{c};font-weight:700;">✖</span> &nbsp;{t}</div>'
                    for t, c in issues
                ) + '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;'
                'border-radius:10px;padding:14px 16px;font-size:12px;color:#00d4a8;">'
                'Warnings: None ✔</div>',
                unsafe_allow_html=True,
            )

    # ── Factor scorecard ──────────────────────────────────────────────
    with col_r:
        def _g(ok, lt="PASS", lf="FAIL"):
            return (lt, "#00d4a8") if ok else (lf, "#ff4d6d")

        tl, tc = _g(res["trend_ok"],     "ALIGNED",   "WEAK")
        sc2    = "#00d4a8" if res["setup_type"] != "None" else "#ff4d6d"
        vl, vc2 = _g(res["volume_ok"],   "STRONG",    "WEAK")
        ml, mc2 = _g(res["momentum_ok"], "HEALTHY",   "STRETCHED")
        el, ec  = _g(res["entry_ok"],    "GOOD",      "EXTENDED")
        slc     = {"Tight":"#00d4a8","Medium":"#f0b429","Poor":"#ff4d6d"}.get(res["sl_quality"],"#4a6480")
        rrl, rrc = _g(res["rr_ok"],
                      f'{res["rr_ratio"]:.1f}:1 ✔',
                      f'{res["rr_ratio"]:.1f}:1 ✖')

        factors = (
            _aura_factor_row("Trend",        tl,   tc)  +
            _aura_factor_row("Setup",        res["setup_type"], sc2) +
            _aura_factor_row("Volume",       f'{res["vol_ratio"]:.1f}× — {vl}', vc2) +
            _aura_factor_row("Momentum RSI", f'{res["rsi"]:.0f} — {ml}', mc2) +
            _aura_factor_row("Entry Quality",f'{res["delta_ema20"]:+.1f}% — {el}', ec) +
            _aura_factor_row("Stop Quality", res["sl_quality"], slc) +
            _aura_factor_row("Risk-Reward",  rrl,  rrc)
        )
        st.markdown(
            '<div style="background:#0f1923;border:1px solid #1e3a5f;'
            'border-radius:10px;padding:14px 16px;margin-bottom:12px;">'
            '<div style="font-size:11px;font-weight:700;color:#8ab4d8;'
            'letter-spacing:.5px;margin-bottom:8px;">FACTOR SCORECARD</div>'
            + factors + '</div>',
            unsafe_allow_html=True,
        )
        if res["market_note"]:
            nc = "#f0b429" if "caution" in res["market_note"].lower() else "#4a6480"
            st.markdown(
                f'<div style="font-size:11px;color:{nc};padding:4px 0;">'
                f'🌐 {res["market_note"]}</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        '<div style="font-size:10px;color:#2a3f58;margin-top:12px;text-align:center;">'
        '⚠️ Stock Aura is for educational purposes only. Not financial advice.</div>',
        unsafe_allow_html=True,
    )

_STOCK_AURA_OK = True   # always True — no external dependency

# ── optional sklearn (graceful fallback if missing) ───────────────────
try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

# NOTE:
# External module imports (scoring_engine/backtest_engine/ml_engine/ui_components)
# intentionally removed. Mode-specific strategy engines are used directly.

# ─────────────────────────────────────────────────────────────────────
# YFINANCE THROTTLING  (unchanged from original)
# ─────────────────────────────────────────────────────────────────────
MAX_YF_CONCURRENCY = 12
_YF_SEM = threading.BoundedSemaphore(MAX_YF_CONCURRENCY)

_MKT_LOCK   = threading.Lock()
_MKT_CACHE: dict[str, float] = {}
_NIFTY_LOCK = threading.Lock()
_NIFTY_20D_RET: float | None = None

# SPEED FIX — restore mktcap cache from session_state on each rerun so
# mode 1/2 re-scans don't re-fetch tickers already looked up this session.
try:
    _ss_mkt = st.session_state.get("_mkt_cache_store", {})
    if _ss_mkt:
        _MKT_CACHE.update(_ss_mkt)
except Exception:
    pass


def get_mktcap_cr(ticker: str) -> float:
    """DO NOT MODIFY — strategy rule"""
    with _MKT_LOCK:
        if ticker in _MKT_CACHE:
            return _MKT_CACHE[ticker]
    try:
        with _YF_SEM:
            info = yf.Ticker(ticker).fast_info
            raw  = getattr(info, "market_cap", 0) or 0
    except Exception:
        raw = 0
    mc_cr = float(raw) / 1e7 if raw else 0.0
    with _MKT_LOCK:
        _MKT_CACHE[ticker] = mc_cr
    # SPEED FIX — persist new entry to session_state so next rerun skips API call
    try:
        if "_mkt_cache_store" not in st.session_state:
            st.session_state["_mkt_cache_store"] = {}
        st.session_state["_mkt_cache_store"][ticker] = mc_cr
    except Exception:
        pass
    return mc_cr


def get_nifty_20d_return() -> float | None:
    """20-day return for Nifty (^NSEI), shared across all stocks."""
    global _NIFTY_20D_RET
    with _NIFTY_LOCK:
        if _NIFTY_20D_RET is not None:
            return _NIFTY_20D_RET
    try:
        with _YF_SEM:
            df_n = yf.download(
                "^NSEI", period="2mo", interval="1d",
                auto_adjust=True, progress=False, timeout=10, threads=False,
            )
        if df_n is None or len(df_n) < 21:
            return None
        close_n = df_n["Close"].dropna()
        if len(close_n) < 21:
            return None
        n_today = float(close_n.iloc[-1])
        n_ago20 = float(close_n.iloc[-21])
        if n_ago20 <= 0:
            return None
        ret = (n_today - n_ago20) / n_ago20
    except Exception:
        return None
    with _NIFTY_LOCK:
        _NIFTY_20D_RET = ret
    return ret


# ─────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NSE Sentinel — Stock Scanner",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────
# DESIGN SYSTEM  (Space Mono + Syne, terminal/Bloomberg aesthetic)
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;700;800&display=swap" rel="stylesheet">

<style>
:root {
  --bg: #060a0f; --bg2: #0b1017; --bg3: #0f1823;
  --border: #1a2840; --border2: #243550;
  --accent: #00d4a8; --accent2: #0094ff; --accent3: #f0b429;
  --red: #ff4d6d; --text: #ccd9e8; --muted: #4a6480;
  --mono: 'Space Mono', monospace; --sans: 'Syne', sans-serif;
}
html, body, .stApp { background-color: var(--bg) !important; color: var(--text) !important; font-family: var(--mono) !important; }
.stApp::before { content:''; position:fixed; inset:0; pointer-events:none; z-index:9999;
  background:repeating-linear-gradient(0deg,transparent,transparent 2px,rgba(0,0,0,0.07) 2px,rgba(0,0,0,0.07) 4px); }
.stApp::after { content:''; position:fixed; inset:0; pointer-events:none; z-index:0;
  background-image:linear-gradient(rgba(0,212,168,0.018) 1px,transparent 1px),linear-gradient(90deg,rgba(0,212,168,0.018) 1px,transparent 1px);
  background-size:40px 40px; }
[data-testid="stSidebar"] { background-color:var(--bg2) !important; border-right:1px solid var(--border) !important; font-family:var(--mono) !important; }
[data-testid="stSidebar"] * { color:var(--text) !important; }
section[data-testid="stSidebar"] > div { padding-top:20px !important; }
[data-testid="stSidebar"] .stSelectbox [data-baseweb="select"] > div { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:8px !important; color:var(--text) !important; font-family:var(--mono) !important; }
[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] { background:var(--accent) !important; box-shadow:0 0 8px var(--accent) !important; }
[data-testid="stMetric"] { background:var(--bg2) !important; border:1px solid var(--border) !important; border-radius:12px !important; padding:18px 20px !important; }
[data-testid="stMetricValue"] { font-family:var(--sans) !important; font-weight:800 !important; font-size:2rem !important; color:var(--accent) !important; }
[data-testid="stMetricLabel"] { font-family:var(--sans) !important; font-size:10px !important; letter-spacing:2px !important; text-transform:uppercase !important; color:var(--muted) !important; }
[data-testid="stMetricDelta"] { color:var(--accent3) !important; }
.stButton > button { background:transparent !important; border:1px solid var(--accent) !important; color:var(--accent) !important; font-family:var(--mono) !important; font-size:13px !important; font-weight:700 !important; letter-spacing:1px !important; border-radius:10px !important; padding:14px 28px !important; width:100% !important; transition:all 0.2s !important; }
.stButton > button:hover { background:var(--accent) !important; color:var(--bg) !important; }
.stButton > button:disabled { border-color:var(--muted) !important; color:var(--muted) !important; }
.stDownloadButton > button { background:rgba(0,148,255,0.1) !important; border:1px solid var(--accent2) !important; color:var(--accent2) !important; font-family:var(--mono) !important; font-weight:700 !important; border-radius:8px !important; width:100% !important; }
.stDownloadButton > button:hover { background:var(--accent2) !important; color:var(--bg) !important; }
.stProgress > div > div { background:linear-gradient(90deg,var(--accent),var(--accent2)) !important; box-shadow:0 0 10px var(--accent) !important; }
.stProgress > div { background:var(--border) !important; border-radius:3px !important; height:6px !important; }
.stDataFrame { border:1px solid var(--border) !important; border-radius:12px !important; overflow:hidden !important; font-family:var(--mono) !important; }
.stDataFrame thead tr th { background:var(--bg3) !important; color:var(--muted) !important; font-family:var(--sans) !important; font-size:10px !important; letter-spacing:1.5px !important; text-transform:uppercase !important; }
.stDataFrame tbody tr:hover td { background:rgba(0,212,168,0.04) !important; }
.stAlert { background:var(--bg3) !important; border:1px solid var(--border2) !important; border-radius:10px !important; font-family:var(--mono) !important; }
h1,h2,h3,h4 { font-family:var(--sans) !important; }
h1 { color:var(--accent) !important; font-weight:800 !important; }
h2,h3 { color:#79c0ff !important; font-weight:700 !important; }
hr { border-color:var(--border) !important; }

@keyframes pdot { 0%,100%{opacity:1;transform:scale(1)} 50%{opacity:0.4;transform:scale(0.75)} }
.live-dot { width:9px;height:9px;border-radius:50%;background:var(--accent);box-shadow:0 0 10px var(--accent);animation:pdot 2s ease infinite;display:inline-block;margin-right:8px; }
.section-lbl { font-family:var(--sans);font-size:10px;font-weight:700;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:12px;padding-bottom:6px;border-bottom:1px solid var(--border); }
.mode-pill { display:inline-flex;align-items:center;gap:6px;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:700;border:1px solid currentColor;font-family:var(--mono); }
.pill-m1 { color:#00d4a8;background:rgba(0,212,168,0.08); }
.pill-m2 { color:#0094ff;background:rgba(0,148,255,0.08); }
.pill-m3 { color:#f0b429;background:rgba(240,180,41,0.08); }
.pill-m5 { color:#ff8c00;background:rgba(255,140,0,0.08); }
.pill-m6 { color:#ff4d6d;background:rgba(255,77,109,0.08); }
.top-banner { display:flex;align-items:center;gap:16px;padding:20px 0 8px 0; }
.banner-logo { font-family:var(--sans);font-weight:800;font-size:26px;color:var(--accent);letter-spacing:-0.5px; }
.count-pill { background:rgba(0,212,168,0.1);border:1px solid var(--accent);color:var(--accent);border-radius:20px;padding:2px 12px;font-size:13px;font-weight:700;font-family:var(--mono); }
.result-hdr { display:flex;align-items:center;gap:12px;padding:14px 0; }
.result-hdr h3 { margin:0 !important; }
.status-line { display:flex;align-items:center;gap:8px;padding:8px 14px;background:var(--bg3);border:1px solid var(--border);border-radius:8px;font-size:12px;color:var(--muted); }
.sdot { width:7px;height:7px;border-radius:50%;display:inline-block; }
.sdot-green { background:var(--accent);box-shadow:0 0 6px var(--accent); }

/* ── NEW: score badge + top-pick cards ─────────────────────────── */
.score-green  { color:#00d4a8;font-weight:700; }
.score-blue   { color:#0094ff;font-weight:700; }
.score-yellow { color:#f0b429;font-weight:700; }
.score-red    { color:#ff4d6d;font-weight:700; }
.pick-card {
  background:#0b1017;border:1px solid #1a2840;border-radius:14px;
  padding:18px 20px;transition:border 0.2s;
}
.pick-card:hover { border-color:#243550; }
.pick-rank  { font-family:'Syne',sans-serif;font-size:28px;font-weight:800;color:#00d4a8;line-height:1; }
.pick-sym   { font-family:'Syne',sans-serif;font-size:18px;font-weight:700;color:#ccd9e8; }
.pick-score { font-size:11px;color:#4a6480;margin-top:4px; }
.trap-badge { color:#ff4d6d;font-weight:700;font-size:12px; }
.breakdown-box {
  background:#060a0f;border:1px solid #1a2840;border-radius:8px;
  padding:10px 14px;font-size:11px;line-height:1.9;color:#4a6480;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# NSE TICKER LOADER  (unchanged)
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def fetch_nse_tickers() -> list:
    # ─────────────────────────────────────────────────────────────────
    # NSE MAINBOARD UNIVERSE  (~2100 stocks, cloud-safe hardcoded list)
    # Covers Nifty 50 / Next50 / Midcap150 / Smallcap250 / Microcap250
    # + all major sector stocks as of 2024-25. ALWAYS loaded first.
    # Live NSE fetch supplements this if accessible.
    # ─────────────────────────────────────────────────────────────────
    _NSE_MAINBOARD = [
        # ══ LARGE CAP / NIFTY 50 ══════════════════════════════════════
        "RELIANCE","TCS","HDFCBANK","INFY","ICICIBANK","HINDUNILVR","SBIN",
        "BHARTIARTL","ITC","KOTAKBANK","LT","AXISBANK","ASIANPAINT","MARUTI",
        "BAJFINANCE","HCLTECH","SUNPHARMA","TITAN","ULTRACEMCO","ONGC",
        "NESTLEIND","WIPRO","POWERGRID","NTPC","TECHM","INDUSINDBK","ADANIPORTS",
        "TATAMOTORS","JSWSTEEL","BAJAJFINSV","HINDALCO","GRASIM","DIVISLAB",
        "CIPLA","DRREDDY","BPCL","EICHERMOT","APOLLOHOSP","TATACONSUM","BRITANNIA",
        "COALINDIA","HEROMOTOCO","SHREECEM","SBILIFE","HDFCLIFE","ADANIENT",
        "BAJAJ-AUTO","TATASTEEL","UPL","M&M",
        # ══ NIFTY NEXT 50 ═════════════════════════════════════════════
        "ADANIGREEN","ADANITRANS","ATGL","AWL","BAJAJHFL","BANKBARODA",
        "BERGEPAINT","BEL","BHEL","BOSCHLTD","CANBK","CGPOWER","CHOLAFIN",
        "COLPAL","CUMMINSIND","DABUR","DLF","DMART","GODREJCP","GODREJPROP",
        "HAL","HAVELLS","HDFCAMC","ICICIGI","ICICIPRULI","IOC","IRCTC",
        "IRFC","LODHA","LTIM","LTTS","MARICO","MOTHERSON","MUTHOOTFIN",
        "NAUKRI","NHPC","OFSS","PAGEIND","PERSISTENT","PFC","PIDILITIND",
        "POLYCAB","PNB","RECLTD","SHRIRAMFIN","SRF","TORNTPHARM","TRENT",
        "TVSMOTOR","UNIONBANK","VEDL","ZOMATO","ZYDUSLIFE",
        # ══ NIFTY MIDCAP 150 ══════════════════════════════════════════
        "AAPL","AARTIIND","ABCAPITAL","ABFRL","ACC","AFFLE","AJANTPHARM",
        "ALKEM","ALKYLAMINE","ANGELONE","APLAPOLLO","APOLLOTYRE","APTUS",
        "ASTRAL","AUBANK","AUROPHARMA","BALKRISHIND","BANDHANBNK","BATAINDIA",
        "BIKAJI","BIOCON","BIRLASOFT","BLUESTARCO","BRIGADE","CARBORUNIV",
        "CASTROLIND","CDSL","CESC","CHENNPETRO","CLEAN","COFORGE","CROMPTON",
        "CYIENT","DALBHARAT","DATAMATICS","DCBBANK","DEEPAKNTR","DELHIVERY",
        "DEVYANI","DIXON","DOMS","DRREDDYS","ECLERX","EIDPARRY","ELGIEQUIP",
        "EMAMILTD","ENDURANCE","EPL","EQUITASBNK","ESCORTS","EXIDEIND",
        "FINEORG","FLUOROCHEM","FORTIS","GABRIEL","GALAXYSURF","GLAND",
        "GLAXO","GLENMARK","GNFC","GRINDWELL","GSFC","GUJGASLTD",
        "HAPPSTMNDS","HATSUN","HFCL","HLEGLAS","HOMEFIRST","HPCL",
        "HUDCO","IEX","IGL","INDIAMART","INDIGO","INOXWIND","IOB",
        "IPCALAB","IREDA","ISEC","JBCHEPHARM","JKCEMENT","JKLAKSHMI",
        "JMFINANCIL","JSL","JUBILANT","JUBLFOOD","KAJARIACER","KALPATPOWR",
        "KALYANKJIL","KANSAINER","KEC","KIMS","KPITTECH","KRSNAA",
        "KSCL","LAURUSLABS","LAXMIMACH","LICHSGFIN","LLOYDSME","LUXIND",
        "MAHINDRA","MANAPPURAM","MANKIND","MAPMYINDIA","MASTEK","MAXHEALTH",
        "METROPOLIS","MFSL","MIDHANI","MKPL","MMTC","MOIL","MOTILALOFS",
        "MPHASIS","MRPL","NAVA","NAVINFLUOR","NBCC","NIACL","NILKAMAL",
        "NMDC","NOCIL","NYKAA","OBEROIRLTY","OIL","ORIENTELEC","PATELENG",
        "PAYTM","PCBL","PGHH","PHOENIXLTD","PNBHOUSING","POLICYBZR",
        "POLYMED","PRESTIGE","PRINCEPIPE","RITES","RVNL","SAFARI",
        "SAIL","SCHAEFFLER","SEQUENT","SJVN","SKF","SOBHA","SONACOMS",
        "SPANDEX","STAR","STARHEALTH","SUDARSCHEM","SUNDARMFIN","SUNPHARMA",
        "SUNTECK","SUNTV","SUPREME","SUZLON","SWARAJENG","SYMPHONY",
        "TANLA","TATACHEM","TATACOMM","TATAMTRDVR","TATAPOWER","TEAMLEASE",
        "THYROCARE","TIINDIA","TIMKEN","TITAGARH","TORNTPOWER","TRIDENT",
        "TRIVENI","UFLEX","UJJIVANSFB","UTIAMC","VGUARD","VIJAYA",
        "VIPIND","VMART","VOLTAMP","VOLTAS","VSTIND","WELSPUNLIV","WHIRLPOOL",
        "ZEEL","ZENSAR","ZFCVINDIA",
        # ══ NIFTY SMALLCAP 250 ════════════════════════════════════════
        "3MINDIA","AAVAS","ABAN","ABGSHIP","ACCELYA","ACE","ACRYSIL",
        "ADANIPORTS","ADANITRANS","ADFFOODS","AEGISLOG","AETHER","AGARIND",
        "AGROPHOS","AHMEDABADSTEEL","AIAENG","AIRAN","AJAXENG","AKZOINDIA",
        "ALEMBICLTD","ALICON","ALLCARGO","ALMONDZ","ALOKINDS","AMBIKCO",
        "AMBUJACEM","AMBER","AMJUMBO","ANANTRAJ","ANDHRSUGAR","ANUP",
        "APARINDS","APCL","APOLLOPIPE","APTECHT","ARCHIDPLY","ARCOTECH",
        "ARVINDFASHN","ARVSMART","ASAHIINDIA","ASALCBR","ASHAPURMIN",
        "ASHIANA","ASHOKLEY","ASKAUTOLTD","ASTERDM","ASTRAZEN","ATGL",
        "ATUL","AVANTIFEED","AVTNPL","AXISCADES","AYMSYNTEX","AZAD",
        "BAFNAPH","BAJAJCON","BAJAJHHL","BAJAJHIND","BAJAJSFL",
        "BALKRISHIND","BALMLAWRIE","BALRAMCHIN","BANARISUG","BANKBARODA",
        "BANSWRAS","BASF","BBTC","BCG","BECTORFOOD","BEDMUTHA","BHAGERIA",
        "BHANDARI","BHARATFORG","BHARATGEAR","BHARATRAS","BHARATWIRE",
        "BHARTIHEXA","BHORUKA","BIKAJI","BIRLACABLE","BIRLAMONEY",
        "BMAXAUTO","BOROLTD","BPCL","BPL","BSEINDIA","BSL","BURGERKING",
        "BUTTERFLY","BYKE","CADILAHC","CAMLINFINE","CAMLIN","CAMPUS",
        "CANFINHOME","CANTABIL","CAPACITE","CAPITALSFB","CAPTRUST",
        "CARERATING","CARTRADE","CASTEXTECH","CAVALRYIND","CCL","CERA",
        "CEREBRAINT","CESC","CGPOWER","CHALET","CHAMBLFERT","CHEMFAB",
        "CHEMPLASTS","CHENNPETRO","CHOLAHLDNG","CIEINDIA","CINELINE",
        "CMSINFO","COCHINSHIP","COFFEDAY","COFORGE","CONCENTRIX","CONFIPET",
        "CONTROLPR","COROMANDEL","COSMOFILMS","CRAFTSMAN","CREDITACC",
        "CRISIL","CUBEXTUB","CUMMINSIND","CYIENT","DALMIASUG","DAMODARIND",
        "DATAMATICS","DBCORP","DBFSL","DCB","DCBBANK","DECCANCE","DEEPAKFERT",
        "DELTACORP","DEVIT","DHANI","DHARMAJ","DHUNSERI","DISHTV",
        "DIVGIITTS","DOLLAR","DPWIRES","DREDGECORP","DTIL","DUCON",
        "EASEMYTRIP","ECLERX","EIDPARRY","EIL","ELECTCAST","ELGIEQUIP",
        "ELIN","EMKAY","EMMBI","ESABINDIA","ESAFSFB","ESCORTS",
        "ESTER","ETHOSLTD","EXLSERVICE","FAZE3Q","FCL","FEDERALBNK",
        "FINEORG","FINPIPE","FINOLEX","FMGOETZE","FOODSIN","FORCEMOT",
        "FORTIS","FSC","FUELCELL","FUTURETAIL","G2FTECH","GAEL","GALAXYSURF",
        "GANDHAR","GANESHBE","GARFIBRES","GARWARE","GATI","GATEWAY",
        "GAYAPROJ","GEECEE","GESHIP","GHCL","GILLETTE","GINNIFILA",
        "GLAXO","GLOBALVECT","GLOBUSSPR","GMBREW","GMMPFAUDLR","GMRAIRPORT",
        "GNFC","GOACARBON","GOKALDAS","GOLDIAM","GOODLUCK","GPIL","GPPL",
        "GPTINFRA","GRANULES","GREAVESCOT","GREENPANEL","GREENPLY","GRINDWELL",
        "GRSE","GUFICBIO","GUJALKALI","GUJGASLTD","GULFOILLUB","GULFPETRO",
        "GULSHAN","GULSHPOLY","HCG","HDFCAMC","HDIL","HECL","HEMIPROP",
        "HERITGFOOD","HERCULES","HFCL","HGINFRA","HIKAL","HIRECT",
        "HLEGLAS","HLVLTD","HMVL","HONAUT","HPAL","HUBTOWN",
        "HUHTAMAKI","IBREALEST","ICICIBANK","ICICINXT50","IDBI","IDFC",
        "IDFCFIRSTB","IGPL","IINOX","INDABELL","INDHOTEL","INDIABULL",
        "INDIAMART","INDIANB","INDIGO","INDORAMA","INDOSTAR","INDSWFTLAB",
        "INDTERRAIN","INFIBEAM","INFOBEAN","INGERRAND","INTELLECT",
        "INVENTURE","IOB","IPCALAB","IRCON","ISEC","ITD","ITDCEM",
        "ITIVITI","IVP","JAGRAN","JAIBALAJI","JAICORPLTD","JAMNAAUTO",
        "JASH","JAYAGROGN","JAYBPHARMA","JBMA","JCHAC","JHS","JIOFIN",
        "JKIL","JKTYRE","JLHL","JMFINANCIL","JNKINDIA","JOHNSONCONT",
        "JPPOWER","JSFB","JSHL","JSWHL","JTEKTINDIA","JUBLINGREA",
        "JUSTDIAL","JYOTHYLAB","KAJARIACER","KALPATPOWR","KALYANKJIL",
        "KAMDHENU","KANSAINER","KARURVYSYA","KCP","KDDL","KECL","KHADIM",
        "KHAITANLTD","KIRIINDUS","KNR","KOLTEPATIL","KOPRAN","KPIGLOBAL",
        "KPRMILL","KRBL","KSCL","KUBOTA","KUKREJA","LAMBODHARA","LATTEYS",
        "LAXMIMACH","LEMONTREE","LIQWAAUTO","LLOYDSME","LOKESHMACH","LORDSCHLO",
        "LUPIN","LUXIND","LXCHEM","MAHASTEEL","MAHEPC","MAHINDCIE",
        "MAHINDLOG","MAHSCOOTER","MAHSEAMLES","MAITHANALL","MALUPAPER",
        "MANAPPURAM","MANGALAM","MANGCHEM","MANINDS","MARATHON","MARKSANS",
        "MASTEK","MAWANASUG","MBECL","MBLINFRA","MCDOWELL-N","MDHINDUS",
        "MEDPLUS","MEGH","MEISEI","MFSL","MGEL","MHRIL","MIDHANI",
        "MINDAIND","MINDACORP","MINDSPACE","MIRC","MIRCELECTR","MITCON",
        "MMTC","MOLDTEK","MONTECARLO","MORARJEE","MOTHERSUMI","MPDL",
        "MPSLTD","MRPL","MSTCLTD","MUTHOOTCAP","MUTHOOTFIN","MYND",
        "NACLIND","NAHARPOLY","NAHARSPINN","NAUKRI","NAVNETEDUL","NBCC",
        "NCLIND","NEULANDLAB","NEWGEN","NGLFINE","NIITLTD","NILKAMAL",
        "NLCINDIA","NMC","NOCIL","NRAIL","NSIL","NUCLEUS","NURECA",
        "OBEROIRLTY","OCCL","OFSS","OMAXE","ONEPOINT","ONGC","OPTIEMUS",
        "ORCHPHARMA","ORIENTBELL","ORIENTCEM","ORIENTELEC","ORISSAMINE",
        "OSWAL","PAISALO","PALASHSECU","PANAMAPET","PARADEEP","PARAGONFL",
        "PARAS","PARENTERAL","PARKHOTELS","PATANJALI","PATELENG","PBAINFRA",
        "PFIZER","PHOENIXLTD","PILANIINVS","PITTIENG","PKTEA","PLASMAGEN",
        "PLASTIBLEN","POLYMED","POOJA","POWERMECH","PPAP","PRAJIND",
        "PRAKASH","PRECWIRE","PREMIER","PREMIEREXP","PRICOLLTD","PRINCE",
        "PRISM","PROCTER","PTL","PUNJABCHEM","PVRINOX","PYRAMIDTECH",
        "QUESS","QUICKHEAL","RADICO","RAJRATAN","RAJSREESUG","RALLIS",
        "RAMCOIND","RAMCOCEM","RANEHOLDIN","RASHTRIYA","RAYMOND","RBA",
        "RBLBANK","RECLTD","REDTAPE","REFEX","RELAXO","RENUKA","REPCOHOME",
        "REPRO","RESBEES","RFCL","RITES","ROLEXRINGS","ROSSARI","RPGLIFE",
        "RPOWER","RPSGVENT","RTNPOWER","RUDRA","RUPA","RUSHIL","RVNL",
        "SAFARI","SAGCEM","SAKAR","SAKSOFT","SANOFI","SAPPHIRE","SAURASHTRA",
        "SEASOFTS","SECURKLOUD","SEPC","SEQUENT","SERVTECH","SESHAPAPER",
        "SEYA","SGFL","SHANKARA","SHAREINDIA","SHILPAMED","SHIVALIK",
        "SHIVAM","SHIVNAT","SHIVTEXTIL","SHOPERSTOP","SHREEPUSHK","SHREERAMA",
        "SHRIRAM","SHYAM","SIGACHI","SIGMAIND","SILLYMONKS","SINTERCAST",
        "SKFINDIA","SKIPPER","SMLISUZU","SNOWMAN","SOBHA","SOFTTECH",
        "SOLARA","SOLARINDS","SOMAIND","SONACOMS","SOTL","SPANDEX",
        "SPARC","SPECIALITY","SPENTEX","SPIC","SPORTKING","SPSLIFE",
        "SRMTRANS","SRTRANSFIN","STCINDIA","STERTOOLS","STEWARTS",
        "STLTECH","SUBROS","SUCCINPHAR","SUKHJITS","SUMEDHA","SUNTECK",
        "SUPRIYA","SUPRAJIT","SURAJEST","SURANASOL","SURYALAXMI","SURYAROSNI",
        "SUTLEJTEX","SUVENPHAR","SVPGLOB","SYMPHONY","SYNCOMF","TAINWALCHM",
        "TALBROAUTO","TANFAC","TATAINVEST","TATAMETALI","TATASP","TATTEL",
        "TCPLPACK","TDPOWERSYS","TEAMLEASE","TECHNOE","TEXINFRA","TEXRAIL",
        "THANGAMAYL","THEMIS","THERMAX","TIMEX","TIRUMALCHM","TIRUPATI",
        "TKTFSL","TORNTPOWER","TOYOTO","TPLPLASTEH","TRANSWIND","TREEHOUSE",
        "TRIDENT","TRIGYN","TRIL","TRIPATHI","TRITURBINE","TTKHLTCARE",
        "TTKPRESTIG","TTML","TULSYAN","TVSSCS","TVTL","UGARSUGAR",
        "UNICHEMLAB","UNIENTER","UNIONBANK","UNITECH","UNIVASTU","UNOMINDA",
        "URJA","USHAMART","UTTAMDVR","V2RETAIL","VADILALIND","VAKRANGSFT",
        "VAMSIFOOD","VARROC","VASWANI","VBHCL","VEDANT","VENKEYS",
        "VESUVIUS","VGUARD","VINATIORGA","VINDHYATEL","VISHNU","VISHNUCHEM",
        "VIVIDHA","VLSFINANCE","WABAG","WALCHANNAG","WEIZMANIND","WELCORP",
        "WELSUNINT","WINDMACHINES","WIPL","WONDER","WPIL","XCHANGING",
        "XPROINDIA","YASHO","YESBANK","ZENSARTECH","ZICOM","ZODIAC",
        # ══ ADDITIONAL SECTOR LEADERS & MID / SMALL CAPS ═════════════
        # Banking
        "HDFCBANK","ICICIBANK","KOTAKBANK","AXISBANK","SBIN","INDUSINDBK",
        "AUBANK","BANDHANBNK","CSBBANK","CITYUNIONB","DCBBANK","EQUITASBNK",
        "ESAFSFB","FEDERALBNK","IDFCFIRSTB","KARURVYSYA","PNB","RBLBANK",
        "SOUTHBANK","TMVTNL","UJJIVANSFB","UTKARSHBNK","JANA","SBFC",
        # NBFC
        "BAJFINANCE","BAJAJFINSV","CHOLAFIN","MUTHOOTFIN","MANAPPURAM",
        "SHRIRAMFIN","ABCAPITAL","APTUS","AAVAS","CANFINHOME","LICHSGFIN",
        "PNBHOUSING","REPCO","HOMEFIRST","CREDITACC","LICI","SBICARD",
        "MASFIN","UGROCAP","POONAWALLA","MOTILALOFS","IIFL","ANGELONE",
        "HDFCAMC","NIPPONLIFE","UTIAMC","SBIMF","ICICIPRULI","HDFCLIFE",
        "SBILIFE","MFSL","ICICIGI","HDFCGI","STARHEALTH","NIACL","GICRE",
        # IT & Tech
        "TCS","INFY","WIPRO","HCLTECH","TECHM","LTIM","LTTS","PERSISTENT",
        "COFORGE","MPHASIS","TATAELXSI","KPITTECH","ZENSAR","RATEGAIN",
        "NAUKRI","IRCTC","ZOMATO","DELHIVERY","PAYTM","POLICYBZR","NYKAA",
        "CARTRADE","ANGELONE","HAPPSTMNDS","TANLA","MASTEK","INTELLECT",
        "NEWGEN","OFSS","CYIENT","BIRLASOFT","HEXAWARE","SONACOMS",
        "DATAMATICS","ECLERX","FIRSTSOURCE","NIITLTD","RATEGAIN","ZENSARTECH",
        "ONMOBILE","FSL","SASKEN","NELCAST","CMSINFO","CAMPUS","3MINDIA",
        # Pharma
        "SUNPHARMA","DRREDDY","CIPLA","DIVISLAB","AUROPHARMA","LUPIN",
        "TORNTPHARM","ALKEM","IPCALAB","NATCOPHARM","GRANULES","GLAND",
        "LAURUSLABS","BIOCON","ZYDUSLIFE","MANKIND","JBCHEPHARM","SYNGENE",
        "PIRAMALPHARM","GLENMARK","WOCKHARDT","SOLARA","STRIDES","LAURUS",
        "AJANTPHARM","GLAXO","PFIZER","ABBOTINDIA","SANOFI","NOVARTIS",
        "MARKSANS","SEQUENT","AARTI","ORCHPHARMA","SUPRIYA","NEULANDLAB",
        "POLYMED","BLISSGVS","SMRUTHI","HIKAL","KOPRAN","SPARC","RFCL",
        "PARENTERAL","THEMIS","JUBLPHARMA","CAPLIPOINT","NUVOCO","TIRUPATI",
        # Hospitals
        "APOLLOHOSP","FORTIS","MAXHEALTH","MEDANTA","RAINBOW","KIMS",
        "ASTERDM","METROPOLIS","THYROCARE","KRSNAA","HEALTHSPRING","VIJAYA",
        "HCGONCOLOGY","YATHARTH","SURYAHC",
        # Chemicals
        "PIDILITIND","ALKYLAMINE","DEEPAKNTR","NAVINFLUOR","TATACHEM",
        "GNFC","GHCL","VINATI","AARTIIND","CLEAN","FINEORG","GALAXYSURF",
        "IGPL","NOCIL","PCBL","SUDARSCHEM","CAMLIN","ROSSARI","TATACHEML",
        "ANUPAM","NEOGEN","ATUL","BASF","AKZOINDIA","KANSAINER","BERGER",
        "ASIANPAINT","BERGEPAINT","INDIGO","HUHTAMAKI","MOLD-TEK","UFLEX",
        "SRF","NATIONAL","FINOLEX","GSFC","COROMANDEL","UPL","BAYER",
        "RALLIS","DHANUKA","DHARAMSI","TATACHEM","SPIC","CHAMBAL",
        # Metals
        "TATASTEEL","JSWSTEEL","HINDALCO","VEDL","SAIL","NMDC","COALINDIA",
        "MOIL","RATNAMANI","APLAPOLLO","JINDALSAW","WELSPUNLIV","SHYAMMETL",
        "HINDZINC","HINDCOPPER","NALCO","GMRINFRA","NBCC","IRCON","TITAGARH",
        "MANGLMCEM","SEPC","MAHASTEEL","GESHIP","MIDHANI","JSL","JKLAKSHMI",
        "JSHL","JSWHL","APL","NRAIL","PENIND","LLOYDSME","GPIL","SKIPPER",
        # Auto
        "TATAMOTORS","MARUTI","M&M","HEROMOTOCO","BAJAJ-AUTO","EICHERMOT",
        "TVSMOTOR","ASHOKLEY","BOSCHLTD","MOTHERSON","EXIDEIND","AMARARAJA",
        "LUMAXTECH","CRAFTSMAN","SUPRAJIT","FIEM","ENDURANCE","GABRIEL",
        "JAMNA","SWARAJENG","ESCORTS","FORCEMOT","TIINDIA","ZFCVINDIA",
        "SCHAEFFLER","BALKRISHIND","MRF","APOLLOTYRE","MINDAIND","SUBROS",
        "SUNDRMFAST","PRICOLLTD","JAMNAAUTO","MAHINDCIE","VARROC","WABCO",
        "CEAT","GOODYEAR","JTEKTINDIA","SETCO","SMLISUZU","MAHSCOOTER",
        "HMVL","MAHINDLOG","CIEINDIA","KALYANI","INDSIL","MINDA",
        # Cement
        "ULTRACEMCO","SHREECEM","AMBUJACEM","ACC","RAMCOCEM","DALMIACEM",
        "HEIDELBERG","JKCEMENT","JKLAKSHMI","STARCEMENT","BIRLACORPN",
        "INDIACEM","SANGHI","PRISM","SAGCEM","NUVOCO","ORIENTCEM","BURNPUR",
        # Energy / Power / Oil
        "RELIANCE","ONGC","BPCL","IOC","HPCL","GAIL","PETRONET","IGL","MGL",
        "GUJGASLTD","TATAPOWER","NTPC","POWERGRID","ADANIGREEN","ADANITRANS",
        "CESC","TORNTPOWER","NHPC","SJVN","IREDA","RPOWER","JSWENERGY",
        "SUZLON","INOXWIND","ORIENTGREEN","ACME","GREENKO","KALPATPOWR",
        "KEC","PFC","RECLTD","IRFC","BPCL","MRPL","CHENNPETRO","OIL",
        # FMCG
        "HINDUNILVR","ITC","NESTLEIND","BRITANNIA","TATACONSUM","MARICO",
        "GODREJCP","DABUR","EMAMILTD","COLPAL","PGHH","JYOTHYLAB","BAJAJCON",
        "GILLETTE","HENKEL","BIKAJI","PRATAAP","VARUN","HATSUN","HERITGFOOD",
        "VSTIND","RADICO","UNITDSPR","MCDOWELL-N","TILAKNAGAR","ABINBEV",
        "AVANTIFEED","KRBL","LTTL","DMMART","CCLPROD","BAJAJHHL",
        # Retail / Consumer
        "TRENT","DMART","WESTLIFE","JUBLFOOD","DEVYANI","DIXON","AMBER",
        "SHOPERSTOP","VMART","BATAINDIA","RELAXO","METRO","KPRMILL",
        "PAGEIND","VEDANT","MANYAVAR","GOKALDAS","RAYMOND","SIYARAM",
        "CANTABIL","VIPIND","LUXIND","RUPA","DOLLAR","CAMPUS","KEWAL",
        "NYKAA","V2RETAIL","SAFARI","DOMS","REDTAPE","ETHNOTES",
        # Real Estate / Infra
        "DLF","OBEROIRLTY","GODREJPROP","PHOENIXLTD","BRIGADE","SOBHA",
        "KOLTEPATIL","SUNTECK","LODHA","PRESTIGE","MAHINDRALIFE","PURVA",
        "AHLUCONT","NESCO","IBREALEST","ANANTRAJ","OMAXE","HEMIPROP",
        "MHRIL","CHALET","LEMONTREE","INDHOTEL","EIHLTD","TAJGVK",
        "ITCHOTELS","VENTIVE","SHIVALIKB","TREEHOUSE","AJMERA","PARSVNATH",
        # Capital Goods / Engineering
        "LT","SIEMENS","ABB","BHEL","THERMAX","CUMMINSIND","ELGIEQUIP",
        "KEC","KALPATPOWR","GRINDWELL","TIMKEN","SKF","SCHAEFFLER","HAL",
        "BEL","PARAS","MTAR","ASTRA","GRSE","COCHINSHIP","MAZAGON",
        "PATELENG","NBCC","IRCON","HGINFRA","PNC","KNR","ASHOKA",
        "ITD","CAPACITE","GPPL","CONCOR","ALLCARGO","AEGISLOG","BLUEDART",
        "MAHINDRA","GATI","TCI","DREDGECORP","RVNL","RAILTEL","IRFC",
        "TITAGARH","PENIND","POWERINDIA","INOX","IOCL","HARDWYN",
        "AIAENG","APARINDS","GREAVESCOT","TDPOWERSYS","VOLTAMP","POWERMECH",
        # Telecom / Media
        "BHARTIARTL","VODAFONEIDEA","TATACOMM","RAILTEL","HFCL","STLTECH",
        "INDUSTOWER","GTLINFRA","OPTIEMUS","VINDHYATEL","DISHTV","SUNTV",
        "ZEEMEDIA","PVRINOX","DBCORP","JAGRAN","HMVL","TVTODAY","NDTV",
        # Agri / Fertilisers
        "UPL","DHANUKA","BAYER","RALLIS","PARADEEP","COROMANDEL","GSFC",
        "GNFC","CHAMBAL","KSCL","INSECTICID","DHARMAJ","EIDPARRY",
        "BAJAJHIND","BALRAMCHIN","RENUKA","DWARIKESH","TRIVENI","MAWANASUG",
        # Logistics
        "DELHIVERY","ALLCARGO","GATI","TCI","BLUEDART","CONCOR","SNOWMAN",
        "MAPMYINDIA","INTERGLOBE","SPICEJET","GMRAIRPORT","AIRPORT",
        # Textiles
        "WELSPUNLIV","TRIDENT","VARDHACRLC","NITIN","NANDAN","SPORTKING",
        "RAYMOND","SIYARAM","VIPIND","GOKALDAS","ARVINDFASHN","WELCORP",
        "SUTLEJTEX","AYMSYNTEX","SPANDEX","FILATEX","BSLLTD","DONEAR",
        "GARFIBRES","GARWARE","TEXINFRA","MPDL","MORARJEE","BINANIIND",
        # Paper / Packaging
        "TNPL","WESTERNINDIA","SHRIRAMPOLY","UFLEX","MOLDTEK","HUHTAMAKI",
        "EPL","COSMOFILMS","POLYPLEX","SETCO","PARAGONFL","PLASTIBLEN",
        "TCPLPACK","TPLPLASTEH","EMMBI","PPAP","PRINCEPIPE","ASTRAL",
        # Defence
        "HAL","BEL","BHEL","PARAS","MTAR","ASTRA","GRSE","COCHINSHIP",
        "MAZAGON","MIDHANI","DYNAMATECH","ZENTEC","ABSLAMC","IDEAFORGE",
        # Gems & Jewellery
        "TITAN","KALYANKJIL","SENCO","THANGAMAYL","PCJEWELLER","RAJESHEXPO",
        "GOLDIAM","IIFLSEC","TRIBHOVANDAS","GITANJALI",
        # Misc
        "GREENPLY","GREENLAM","BAJAJELEC","CROMPTON","ORIENTELEC","HAVELLS",
        "BLUESTAR","WHIRLPOOL","VOLTAS","DAIKIN","SYMPHONY","VGUARD",
        "HINDWAREAP","CERA","KAJARIA","SOMANYCER","ORIENTCEREM","NITCO",
        "SUNFLAG","SELAN","TRIL","SNOWMAN","QUESS","TEAMLEASE","XCHANGING",
        "JUSTDIAL","INDIABULL","INFIBEAM","AFFLE","RATEGAIN","NAZARA",
        "LATENTVIEW","MAPMYINDIA","ZAGGLE","EASEMYTRIP","IXIGO","YATRA",
        "MMTC","STCINDIA","MSTCLTD","IRCTC","DFMFOODS","CCLPROD",
        "ABBOTINDIA","3MINDIA","HONAUT","GLAXO","PFIZER","SANOFI",
        "GRINDWELL","CARBORUNIV","WENDT","DYNAMATIC","ELGIEQUIP",
    ]

    tickers: set[str] = set(f"{s}.NS" for s in _NSE_MAINBOARD)

    # ── LIVE SUPPLEMENT: try NSE EQUITY_L.csv (adds 1000+ if accessible) ──
    _NSE_HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Referer": "https://www.nseindia.com/",
        "Connection": "keep-alive",
    }
    try:
        session = requests.Session()
        session.headers.update(_NSE_HEADERS)
        session.get("https://www.nseindia.com/", timeout=8)
        r = session.get(
            "https://archives.nseindia.com/content/equities/EQUITY_L.csv",
            timeout=15,
        )
        if r.status_code == 200 and len(r.content) > 5000:
            df_eq = pd.read_csv(io.StringIO(r.text))
            col = "SYMBOL" if "SYMBOL" in df_eq.columns else df_eq.columns[0]
            for s in df_eq[col].dropna().unique():
                tickers.add(f"{str(s).strip()}.NS")
    except Exception:
        pass

    # ── LIVE SUPPLEMENT: bhav copy (full daily list ~2200 symbols) ────
    if len(tickers) < 1000:
        for days_back in range(0, 7):
            try:
                dt = datetime.now() - timedelta(days=days_back)
                if dt.weekday() >= 5:
                    continue
                date_str = dt.strftime("%d%b%Y").upper()
                url = (
                    f"https://archives.nseindia.com/content/historical/EQUITIES/"
                    f"{dt.year}/{dt.strftime('%b').upper()}/cm{date_str}bhav.csv.zip"
                )
                r = requests.get(url, headers=_NSE_HEADERS, timeout=15)
                if r.status_code == 200 and len(r.content) > 1000:
                    z  = zipfile.ZipFile(io.BytesIO(r.content))
                    df_bh = pd.read_csv(z.open(z.namelist()[0]))
                    for s in df_bh["SYMBOL"].dropna().unique():
                        tickers.add(f"{s.strip()}.NS")
                    break
            except Exception:
                continue

    return sorted(list(tickers))


# ─────────────────────────────────────────────────────────────────────
# TECHNICAL HELPERS  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def ema(series, period):
    return series.ewm(span=period, adjust=False).mean()

def rsi(series, period=14):
    delta = series.diff().dropna()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    r     = 100 - (100 / (1 + rs))
    return float(r.iloc[-1]) if not r.empty else np.nan


# ─────────────────────────────────────────────────────────────────────
# MARKET BIAS ENGINE (add-on; does not affect scan/mode engines)
# ─────────────────────────────────────────────────────────────────────
def _safe_float(v, default: float = 0.0) -> float:
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


@st.cache_data(ttl=900, show_spinner=False)
def compute_market_bias(include_bank: bool = True) -> dict:
    """
    Compute probabilistic market bias for next day using only free yfinance data.
    Fail-safe: returns conservative "Sideways / no edge" output if index data missing.
    """
    try:
        def _fetch_index(symbol: str) -> pd.DataFrame | None:
            try:
                df_i = yf.download(
                    symbol, period="3mo", interval="1d",
                    auto_adjust=True, progress=False, timeout=12, threads=False,
                )
                if df_i is None or df_i.empty:
                    return None
                if isinstance(df_i.columns, pd.MultiIndex):
                    df_i.columns = df_i.columns.get_level_values(0)
                # ── Time-Travel: truncate to historical cutoff ─────────
                if _TIME_TRAVEL_OK:
                    df_i = _tt.apply_time_travel_cutoff(df_i) if hasattr(_tt, "apply_time_travel_cutoff") else df_i
                    if df_i is None or df_i.empty:
                        return None
                return df_i
            except Exception:
                return None

        def _index_features(df_i: pd.DataFrame) -> dict:
            close = df_i["Close"].dropna() if "Close" in df_i.columns else pd.Series(dtype=float)
            if len(close) < 40:
                return {}

            ema20_s = ema(close, 20)
            ema50_s = ema(close, 50)
            rsi14_v = rsi(close, 14)
            c_last  = _safe_float(close.iloc[-1], 0.0)
            e20_last = _safe_float(ema20_s.iloc[-1], 0.0)
            e50_last = _safe_float(ema50_s.iloc[-1], 0.0)

            ret5d = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 and float(close.iloc[-6]) != 0 else np.nan
            ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 and float(close.iloc[-21]) != 0 else np.nan

            # Vol relative to 20D average (if volume exists; indices may lack Volume)
            vol_ratio = None
            if "Volume" in df_i.columns:
                vol = df_i["Volume"].dropna()
                if len(vol) >= 21:
                    avg20 = float(vol.iloc[-21:-1].mean()) if len(vol.iloc[-21:-1]) >= 20 else float(vol.mean())
                    lastv = float(vol.iloc[-1])
                    vol_ratio = (lastv / avg20) if avg20 > 0 else None

            # Realized volatility proxy for expected move range
            ret_1d = close.pct_change().dropna().tail(20) * 100.0
            sigma_pct = float(ret_1d.std()) if not ret_1d.empty else 0.0

            features = {
                "close": c_last,
                "ema20": e20_last,
                "ema50": e50_last,
                "rsi14": _safe_float(rsi14_v, 50.0),
                "ret5d": _safe_float(ret5d, 0.0),
                "ret20d": _safe_float(ret20d, 0.0),
                "vol_ratio": (float(vol_ratio) if vol_ratio is not None else None),
                "sigma_pct": max(0.0, sigma_pct),
            }
            return features

        df_nifty = _fetch_index("^NSEI")
        if df_nifty is None:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "\u00b10.30% to \u00b10.70%",
                "breakdown": ["Nifty data unavailable (fallback)."],
                "regime": "Ranging",
            }

        nifty_feat = _index_features(df_nifty)
        if not nifty_feat:
            return {
                "bias": "Sideways / no edge",
                "confidence": 50,
                "expected_range": "\u00b10.30% to \u00b10.70%",
                "breakdown": ["Nifty indicators insufficient (fallback)."],
                "regime": "Ranging",
            }

        bank_feat = None
        if include_bank:
            df_bn = _fetch_index("^NSEBANK")
            if df_bn is not None:
                bf = _index_features(df_bn)
                bank_feat = bf if bf else None

        # Score / interpret (nifty dominates; bank is confirmation)
        return interpret_market_bias(nifty_feat, bank_feat)
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "\u00b10.30% to \u00b10.70%",
            "breakdown": ["Market bias computation failed (fallback)."],
            "regime": "Ranging",
        }




def _classify_regime_nifty(feat: dict) -> str:
    """
    Regime label aligned with grading_engine._REGIME_ADJ keys (soft context only).
    """
    try:
        if not feat:
            return "Ranging"
        c = _safe_float(feat.get("close", 0.0), 0.0)
        e20 = _safe_float(feat.get("ema20", 0.0), 0.0)
        e50 = _safe_float(feat.get("ema50", 0.0), 0.0)
        rsi14_v = _safe_float(feat.get("rsi14", 50.0), 50.0)
        r5 = _safe_float(feat.get("ret5d", 0.0), 0.0)
        sig = _safe_float(feat.get("sigma_pct", 0.5), 0.5)
        if rsi14_v > 72 and r5 > 0.15:
            return "Overbought Pullback Risk"
        if rsi14_v < 32:
            return "Oversold Bounce Zone"
        if c > e20 > e50 and r5 > 0:
            return "Trending Up"
        if c < e20 < e50 and r5 < 0:
            return "Trending Down"
        if sig > 1.15:
            return "High Volatility / Choppy"
        return "Ranging"
    except Exception:
        return "Ranging"


def interpret_market_bias(nifty_feat: dict, bank_feat: dict | None = None) -> dict:
    """
    Convert index features into conservative bias/confidence/expected-range.
    """
    try:
        def _signal(feat: dict) -> tuple[float, dict]:
            close = _safe_float(feat.get("close", 0.0), 0.0)
            ema20_v = _safe_float(feat.get("ema20", 0.0), 0.0)
            ema50_v = _safe_float(feat.get("ema50", 0.0), 0.0)
            rsi14_v = _safe_float(feat.get("rsi14", 50.0), 50.0)
            ret5d_v = _safe_float(feat.get("ret5d", 0.0), 0.0)
            ret20d_v = _safe_float(feat.get("ret20d", 0.0), 0.0)
            vol_ratio = feat.get("vol_ratio", None)

            bull_trend = close > ema20_v > ema50_v
            bear_trend = close < ema20_v < ema50_v
            trend_sig = 1.0 if bull_trend else (-1.0 if bear_trend else 0.0)

            momentum_sig = 1.0 if ret5d_v > 0.30 else (-1.0 if ret5d_v < -0.30 else 0.0)
            rsi_sig = 1.0 if rsi14_v >= 55.0 else (-1.0 if rsi14_v <= 45.0 else 0.0)

            if vol_ratio is None:
                volume_sig = 0.0
            else:
                volume_sig = 1.0 if vol_ratio >= 1.10 else (-1.0 if vol_ratio <= 0.90 else 0.0)

            breakdown = (close < ema20_v and ret20d_v < 0.0)
            support = (close > ema20_v and ret20d_v > 0.0)

            base_score = 0.35 * trend_sig + 0.25 * momentum_sig + 0.20 * rsi_sig + 0.20 * volume_sig

            details = {
                "trend_sig": trend_sig,
                "momentum_sig": momentum_sig,
                "rsi_sig": rsi_sig,
                "volume_sig": volume_sig,
                "breakdown": breakdown,
                "support": support,
                "close": close,
                "ema20": ema20_v,
                "ema50": ema50_v,
                "rsi14": rsi14_v,
                "ret5d": ret5d_v,
                "ret20d": ret20d_v,
                "vol_ratio": vol_ratio,
            }
            return base_score, details

        nifty_score, nf = _signal(nifty_feat)

        bank_score = 0.0
        bf = None
        if bank_feat:
            bank_score, bf = _signal(bank_feat)

        combined = nifty_score
        bank_used = bf is not None
        if bank_used:
            combined = 0.80 * nifty_score + 0.20 * bank_score

        trend_pos = nf["trend_sig"] > 0
        trend_neg = nf["trend_sig"] < 0
        mom_pos = nf["momentum_sig"] > 0
        mom_neg = nf["momentum_sig"] < 0
        rsi_pos = nf["rsi_sig"] > 0
        rsi_neg = nf["rsi_sig"] < 0
        vol_pos = nf["volume_sig"] > 0 or nf["vol_ratio"] is None
        vol_neg = nf["volume_sig"] < 0 or nf["vol_ratio"] is None

        bullish_strict = trend_pos and mom_pos and vol_pos and rsi_pos
        bearish_strict = trend_neg and mom_neg and vol_neg and rsi_neg

        bullish_relaxed = (bullish_strict or ((trend_pos or nf["support"]) and mom_pos and rsi_pos and vol_pos))
        bearish_relaxed = (bearish_strict or ((trend_neg or nf["breakdown"]) and mom_neg and rsi_neg and vol_neg))

        if bank_used and bf is not None:
            bank_trend_pos = bf["trend_sig"] > 0
            bank_trend_neg = bf["trend_sig"] < 0
            # Only veto if bank is meaningfully negative (score < -0.25), not just mildly soft
            if bullish_relaxed and bank_trend_neg and bf["momentum_sig"] < 0 and bank_score < -0.25:
                bullish_relaxed = False
            if bearish_relaxed and bank_trend_pos and bf["momentum_sig"] > 0 and bank_score > 0.25:
                bearish_relaxed = False

        if bullish_strict:
            bias = "Bullish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bearish_strict:
            bias = "Bearish bias"
            conf = min(80, int(round(62 + abs(combined) * 35)))
        elif bullish_relaxed and not bearish_relaxed:
            bias = "Bullish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        elif bearish_relaxed and not bullish_relaxed:
            bias = "Bearish bias"
            conf = min(65, int(round(52 + abs(combined) * 25)))
        else:
            bias = "Sideways / no edge"
            conf = min(58, int(round(45 - abs(combined) * 18)))
            conf = max(conf, 25)

        sigma_pct = _safe_float(nifty_feat.get("sigma_pct", 0.0), 0.0)
        sigma_pct = max(sigma_pct, 0.10)
        if bias == "Bullish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"+{low_mag:.2f}% to +{high_mag:.2f}%"
        elif bias == "Bearish bias":
            low_mag = sigma_pct * 0.45 * (0.85 + conf / 200.0)
            high_mag = sigma_pct * 0.95 * (0.85 + conf / 200.0)
            expected_range = f"-{low_mag:.2f}% to -{high_mag:.2f}%"
        else:
            side = sigma_pct * 0.55 * (0.80 + conf / 220.0)
            expected_range = f"±{side:.2f}% to ±{(side * 1.2):.2f}%"

        breakdown = []
        trend_txt = "Bullish trend (Close > EMA20 > EMA50)" if nf["trend_sig"] > 0 else (
            "Bearish trend (Close < EMA20 < EMA50)" if nf["trend_sig"] < 0 else "Neutral trend (EMA alignment mixed)"
        )
        breakdown.append(f"Trend: {trend_txt}.")
        breakdown.append(f"Momentum: 5D return {nf['ret5d']:+.2f}%.")
        breakdown.append(f"RSI(14): {nf['rsi14']:.1f}.")
        if nf["vol_ratio"] is None:
            breakdown.append("Volume: not available for index (confidence kept conservative).")
        else:
            breakdown.append(f"Volume: Vol/20D avg {nf['vol_ratio']:.2f}x.")
        breakdown.append(f"20D return: {nf['ret20d']:+.2f}%.")

        if bank_used and bf is not None:
            bn_txt = "BankNifty confirms bullishness" if bf["trend_sig"] > 0 and bf["momentum_sig"] > 0 else (
                "BankNifty confirms bearishness" if bf["trend_sig"] < 0 and bf["momentum_sig"] < 0 else
                "BankNifty mixed / neutral."
            )
            breakdown.append(bn_txt)

        return {
            "bias": bias,
            "confidence": conf,
            "expected_range": expected_range,
            "breakdown": breakdown[:6],
            "regime": _classify_regime_nifty(nifty_feat),
        }
    except Exception:
        return {
            "bias": "Sideways / no edge",
            "confidence": 50,
            "expected_range": "\u00b10.30% to \u00b10.70%",
            "breakdown": ["Interpretation failed (fallback)."],
            "regime": "Ranging",
        }


# ── FRESH ISOLATED MARKET BIAS ENGINE (Task 4.3 — UI Version) ─────────
@st.cache_data(ttl=600, show_spinner=False)
def compute_market_bias_ui(_tt_cache_key: str = "live") -> dict:
    """
    Independent function for the 'Market Bias Engine' UI button.
    Does not touch scanner/strategy mode logic.
    _tt_cache_key is passed as a cache-buster — callers pass the active
    TT date string (or "live") so Streamlit re-runs when the date changes.
    """
    try:
        # 1. Fetch data
        df = yf.download("^NSEI", period="4mo", interval="1d", progress=False, auto_adjust=True)
        if df is None or df.empty or len(df) < 50:
            return {
                "bias": "Sideways / No Edge",
                "confidence": 50,
                "expected_move": "±0.5% (fallback)",
                "reasons": ["Insufficient data from yfinance for Nifty (^NSEI)."]
            }
        
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # ── Time-Travel: truncate to historical cutoff ─────────────────
        if _TIME_TRAVEL_OK and hasattr(_tt, "apply_time_travel_cutoff"):
            df = _tt.apply_time_travel_cutoff(df)
            if df is None or df.empty or len(df) < 50:
                return {
                    "bias": "Sideways / No Edge",
                    "confidence": 50,
                    "expected_move": "±0.5% (fallback)",
                    "reasons": ["Insufficient historical data for selected Time Travel date."]
                }

        close = df["Close"].dropna()
        vol   = df["Volume"].dropna()

        # 2. Indicators
        e20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        e50 = close.ewm(span=50, adjust=False).mean().iloc[-1]
        
        # RSI manual calc (pure dependency-free)
        delta = close.diff()
        gain  = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss  = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs    = gain / loss.replace(0, np.nan)
        rsi_val = 100 - (100 / (1 + rs.iloc[-1])) if not np.isnan(rs.iloc[-1]) else 50.0

        c_last = float(close.iloc[-1])
        ret5d  = (c_last / float(close.iloc[-6]) - 1.0) * 100.0 if len(close) >= 6 else 0.0
        ret20d = (c_last / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else 0.0
        
        avg_vol = vol.iloc[-21:-1].mean() if len(vol) >= 21 else 1.0
        vol_r   = vol.iloc[-1] / avg_vol if avg_vol > 0 else 1.0

        # 3. Bias Logic (Strictly Isolated)
        bullish = (c_last > e20 > e50) and (rsi_val > 52) and (ret5d > 0.1)
        bearish = (c_last < e20 < e50) and (rsi_val < 48) and (ret5d < -0.1)
        
        score = 50
        if bullish:
            score += 15
            if ret20d > 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bullish / Strong Bias" if score > 70 else "Bullish"
        elif bearish:
            score += 15
            if ret20d < 0: score += 10
            if vol_r > 1.2: score += 5
            bias = "Bearish / Negative Bias" if score > 70 else "Bearish"
        else:
            bias = "Sideways / No Clear Edge"
        
        score = min(95, max(5, score))

        # 4. Volatility based move range
        daily_returns = close.pct_change().tail(20)
        volatility = daily_returns.std() * 100.0 if not daily_returns.empty else 0.5
        move_pct = round(volatility * 0.8, 2)
        expected_move = f"±{move_pct}% to ±{round(move_pct*1.5, 2)}%"

        # 5. Reason Breakdown
        reasons = []
        trend_txt = "Bullish stack" if c_last > e20 > e50 else ("Bearish stack" if c_last < e20 < e50 else "Neutral trend")
        reasons.append(f"Trend: {trend_txt} (Close={c_last:.0f}, EMA20={e20:.0f}).")
        reasons.append(f"RSI(14): {rsi_val:.1f} ({'Overbought' if rsi_val > 70 else ('Oversold' if rsi_val < 30 else 'Neutral')}).")
        reasons.append(f"Momentum: 5-day return {ret5d:+.2f}%.")
        reasons.append(f"Volume: Recent volume is {vol_r:.2f}x of 20-day average.")
        reasons.append(f"Structure: 20-day return is {ret20d:+.2f}%.")

        return {
            "bias": bias,
            "confidence": int(score),
            "expected_move": expected_move,
            "reasons": reasons
        }
    except Exception as e:
        return {
            "bias": "Sideways / Unknown",
            "confidence": 50,
            "expected_move": "±0.5% (error)",
            "reasons": [f"Market analysis error: {str(e)}"]
        }


# ─────────────────────────────────────────────────────────────────────
# STOCK ANALYSER  (Zero-API Refactored)
# ─────────────────────────────────────────────────────────────────────
def analyse(ticker, mode, retries=2):
    try:
        ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
        df = get_df_for_ticker(ticker_ns)

        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Open", "Close", "Volume"])
        if df.empty or len(df) < 30:
            return None

        # ── 🕰️ TIME TRAVEL: truncate to cutoff date (guaranteed no leakage) ──
        # This runs REGARDLESS of whether the ticker was in ALL_DATA or came
        # from a live yfinance fallback — the explicit slice here is the true
        # data-leakage guard. The monkey-patch in time_travel_engine only covers
        # ALL_DATA hits; this covers everything else.
        try:
            _tt_cut = _tt.get_reference_date()
            if _tt_cut is not None:
                _tt_mask = pd.to_datetime(df.index).date <= _tt_cut
                df = df.loc[_tt_mask]
                if df.empty or len(df) < 30:
                    return None
        except Exception:
            pass  # fail-safe: continue with untruncated data rather than crash

        try:
            last_idx = df.index[-1]
            last_dt  = pd.to_datetime(last_idx).to_pydatetime()
        except Exception:
            return None
        if (_tt.get_reference_datetime() - last_dt).days > 7:
            return None

        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        open_p = df["Open"].dropna()
        if len(close) < 25:
            return None

        lc  = float(close.iloc[-1])
        lo  = float(open_p.iloc[-1])
        lv  = float(volume.iloc[-1])
        e20 = float(ema(close, 20).iloc[-1])
        e50 = float(ema(close, 50).iloc[-1])
        avg_vol = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
        ri  = rsi(close)

        if not (1 < lc <= 100000):
            return None
        if lv <= 0:
            return None
        if any(np.isnan(v) for v in (ri, e20, e50)):
            return None

        # SPEED FIX — mktcap only fetched for modes that actually use it (1 & 2).
        # Modes 3-6 never reference mktcap_cr so skipping saves ~1000 API calls/scan.
        ok = False
        if mode == 1:
            mktcap_cr = get_mktcap_cr(ticker)
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            ok  = (lc > e20 and e20 > e50 and lv > 1.5 * avg_vol
                   and 52 <= ri <= 74 and lc >= 0.99 * h10 and lc > lo
                   and lc > 30 and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 2:
            mktcap_cr = get_mktcap_cr(ticker)
            h15 = float(close.iloc[-16:-1].max()) if len(close) >= 16 else float(close.max())
            ok  = (lc > 30 and lc > e20 and e20 > e50
                   and lv > 1.3 * avg_vol and 50 <= ri <= 72
                   and lc >= 0.96 * h15 and lc > lo
                   and (mktcap_cr > 500 or mktcap_cr == 0))
        elif mode == 3:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            ok  = (lc > e20 and lv > 1.1 * avg_vol
                   and 48 <= ri <= 74 and lc >= 0.90 * h20 and lc > 20)
        elif mode == 4:
            h20 = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            if len(close) < 21:
                return None
            base_20 = float(close.iloc[-21])
            if base_20 <= 0:
                return None
            stock_ret_20d = (lc - base_20) / base_20
            nifty_ret_20d = get_nifty_20d_return()
            if nifty_ret_20d is None:
                nifty_ret_20d = 0.0  # fallback: compare vs flat market instead of blocking all results
            ok = (
                lc > e20 and e20 > e50 and
                lv > 1.3 * avg_vol and 52 <= ri <= 72 and
                lc >= 0.97 * h20 and stock_ret_20d > nifty_ret_20d and lc > lo
            )
        elif mode == 5:
            h10 = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and lv > 1.1 * avg_vol_sma and
                lc >= 0.99 * h10 and 50 <= ri <= 65 and lc > lo and lc > 20
            )
        elif mode == 6:
            if len(close) < 2:
                return None
            prev_e20     = float(ema(close, 20).iloc[-2])
            h10          = float(close.iloc[-11:-1].max()) if len(close) >= 11 else float(close.max())
            avg_vol_sma  = float(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else float(volume.mean())
            ok = (
                lc > e20 and e20 > e50 and e20 > prev_e20 and
                lv > 1.1 * avg_vol_sma and
                lc >= 0.97 * h10 and 50 <= ri <= 68 and lc > lo and lc > 40
            )

        if ok:
            h20_full      = float(close.iloc[-21:-1].max()) if len(close) >= 21 else float(close.max())
            dist_20d_high = (lc / h20_full - 1.0) * 100.0 if h20_full > 0 else 0.0
            dist_ema20    = (lc / e20 - 1.0) * 100.0 if e20 > 0 else 0.0
            ret_5d  = (lc / float(close.iloc[-6]) - 1.0) * 100.0  if len(close) >= 6  else np.nan
            ret_20d = (lc / float(close.iloc[-21]) - 1.0) * 100.0 if len(close) >= 21 else np.nan

            return {
                "Symbol":            ticker.replace(".NS", ""),
                "Price (₹)":         round(lc, 2),
                "Volume":            int(lv),
                "RSI":               round(ri, 2),
                "EMA 20":            round(e20, 2),
                "EMA 50":            round(e50, 2),
                "Vol / Avg":         round(lv / avg_vol, 2) if avg_vol > 0 else 0,
                "Mode":              {1:"🟢 Momentum",2:"🔵 Balanced",3:"🟡 Relaxed",
                                      4:"🟣 Institutional",5:"🟠 Intraday",6:"🔴 Swing"}[mode],
                "Δ vs 20D High (%)": round(dist_20d_high, 2),
                "Δ vs EMA20 (%)":    round(dist_ema20, 2),
                "5D Return (%)":     round(ret_5d, 2)  if not np.isnan(ret_5d)  else np.nan,
                "20D Return (%)":    round(ret_20d, 2) if not np.isnan(ret_20d) else np.nan,
            }
        return None
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────
# PARALLEL SCANNER  (unchanged)
# ─────────────────────────────────────────────────────────────────────
def run_scan(tickers, mode, workers=20):
    results = []
    total   = len(tickers)
    done    = 0

    progress_bar = st.progress(0.0)
    col_a, col_b = st.columns([3, 1])
    with col_a: status  = st.empty()
    with col_b: eta_box = st.empty()
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(analyse, t, mode): t for t in tickers}
        for fut in as_completed(futures):
            done += 1
            r = fut.result()
            if r:
                results.append(r)
            pct     = done / total
            elapsed = time.time() - t0
            rate    = done / elapsed if elapsed > 0 else 1
            remaining = (total - done) / rate if rate > 0 else 0
            progress_bar.progress(pct)
            status.markdown(
                f'<div class="status-line"><span class="sdot sdot-green"></span>'
                f'&nbsp;Scanned <b style="color:#ccd9e8">{done:,}</b> / {total:,}'
                f' &nbsp;·&nbsp; Found <b style="color:#00d4a8">{len(results)}</b> candidates</div>',
                unsafe_allow_html=True)
            eta_box.markdown(
                f'<div class="status-line" style="justify-content:center">'
                f'â± ETA <b style="color:#f0b429">{remaining:.0f}s</b></div>',
                unsafe_allow_html=True)

    progress_bar.progress(1.0)
    elapsed_total = time.time() - t0
    status.markdown(
        f'<div class="status-line"><span class="sdot sdot-green"></span>'
        f'&nbsp;✅ Complete &nbsp;·&nbsp; {total:,} stocks in'
        f' <b style="color:#f0b429">{elapsed_total:.1f}s</b>'
        f' &nbsp;·&nbsp; <b style="color:#00d4a8">{len(results)}</b> found</div>',
        unsafe_allow_html=True)
    eta_box.empty()
    return results, elapsed_total


# ═════════════════════════════════════════════════════════════════════
# ▼▼▼  NEW LAYER — SCORING / BACKTEST / ML  (added AFTER scan) ▼▼▼
# ═════════════════════════════════════════════════════════════════════

# ── mode-specific weight configs ─────────────────────────────────────
_MODE_WEIGHTS = {
    # (vol_bonus, breakout_bonus, ema_bonus, rsi_bonus, penalty_scale)
    1: dict(vol=1.4,  breakout=1.5, ema=1.0, rsi=1.0, pen=1.0),   # Momentum
    2: dict(vol=1.0,  breakout=1.0, ema=1.0, rsi=1.0, pen=1.0),   # Balanced
    3: dict(vol=0.8,  breakout=0.8, ema=0.8, rsi=0.8, pen=0.5),   # Relaxed
    4: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Institutional
    5: dict(vol=1.5,  breakout=1.2, ema=0.7, rsi=0.8, pen=0.9),   # Intraday
    6: dict(vol=1.0,  breakout=1.0, ema=1.5, rsi=1.2, pen=1.0),   # Swing
}


def _safe(v, default=0.0):
    """Return v if finite, else default."""
    try:
        f = float(v)
        return f if np.isfinite(f) else default
    except Exception:
        return default


def compute_score(row: dict, mode: int = 2) -> tuple[float, dict]:
    """
    Returns (score_0_100, breakdown_dict).
    breakdown_dict is used for the tooltip.
    """
    w   = _MODE_WEIGHTS.get(mode, _MODE_WEIGHTS[2])
    pts = {}

    ri       = _safe(row.get("RSI",            50))
    vol_r    = _safe(row.get("Vol / Avg",       1))
    d20h     = _safe(row.get("Δ vs 20D High (%)", -5))
    d_ema20  = _safe(row.get("Δ vs EMA20 (%)",   0))
    r5d      = _safe(row.get("5D Return (%)",    0))
    price    = _safe(row.get("Price (₹)",        0))
    e20      = _safe(row.get("EMA 20",           0))
    e50      = _safe(row.get("EMA 50",           0))

    # ── RSI zone ─────────────────────────────────────────────────────
    if 55 <= ri <= 65:
        pts["RSI 55-65"] = round(15 * w["rsi"])
    elif 65 < ri <= 70:
        pts["RSI 65-70"] = round(10 * w["rsi"])

    # ── Volume ratio ──────────────────────────────────────────────────
    if vol_r > 2.0:
        pts["Vol >2×"] = round(25 * w["vol"])
    elif vol_r > 1.5:
        pts["Vol >1.5×"] = round(20 * w["vol"])

    # ── Near 20D breakout ─────────────────────────────────────────────
    if -2.0 <= d20h <= 0.0:
        pts["Near 20D High"] = round(15 * w["breakout"])

    # ── Above EMA20 ───────────────────────────────────────────────────
    if price > e20 > 0:
        pts["Price > EMA20"] = round(10 * w["ema"])

    # ── EMA stack ─────────────────────────────────────────────────────
    if e20 > e50 > 0:
        pts["EMA20 > EMA50"] = round(10 * w["ema"])

    # ── 5D return zone ────────────────────────────────────────────────
    if 1.0 <= r5d <= 5.0:
        pts["5D Return 1-5%"] = round(10 * w["rsi"])

    # ── PENALTIES ─────────────────────────────────────────────────────
    if ri > 72:
        pts["RSI Overbought"] = round(-20 * w["pen"])
    if d_ema20 > 6.0:
        pts["Overextended EMA"] = round(-15 * w["pen"])
    if r5d > 8.0:
        pts["5D Return >8%"] = round(-10 * w["pen"])
    if vol_r < 1.2:
        pts["Low Volume"] = round(-15 * w["pen"])

    raw = sum(pts.values())
    score = float(np.clip(raw, 0, 100))
    return score, pts


# ── Backtest cache: ticker → float ───────────────────────────────────
_BT_CACHE: dict[str, float] = {}
_BT_LOCK = threading.Lock()


def _download_history(ticker_ns: str, period: str = "6mo") -> pd.DataFrame | None:
    """Download history; returns None on failure."""
    try:
        with _YF_SEM:
            df = yf.download(
                ticker_ns, period=period, interval="1d",
                auto_adjust=True, progress=False, timeout=12, threads=False,
            )
        if df is None or df.empty:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.dropna(subset=["Close", "Volume"])
        return df if len(df) >= 30 else None
    except Exception:
        return None


def compute_backtest_probability(row: dict, ticker: str, mode: int = 2) -> float:
    """
    Mode-aware backtest probability (Task 7).
    Each mode uses different RSI/vol tolerances + extra conditions that
    mirror what the live scanner filter actually checks.
    Cache key is ticker+mode so each strategy gets its own probability.
    Returns 50 (neutral) if fewer than 20 matching samples found.
    """
    # per-mode matching config: rsi_tol, vol_tol, ema_trend required,
    # near-high required, rolling-high window, near-high threshold
    _MBTCFG: dict[int, dict] = {
        1: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=10, hp=0.02),
        2: dict(rsi_tol=4,  vol_tol=0.25, ema_trend=True,  near_high=False, hw=15, hp=0.03),
        3: dict(rsi_tol=5,  vol_tol=0.30, ema_trend=False, near_high=False, hw=20, hp=0.05),
        4: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=True,  hw=20, hp=0.02),
        5: dict(rsi_tol=2,  vol_tol=0.15, ema_trend=True,  near_high=True,  hw=10, hp=0.01),
        6: dict(rsi_tol=3,  vol_tol=0.20, ema_trend=True,  near_high=False, hw=10, hp=0.02),
    }
    cfg = _MBTCFG.get(mode, _MBTCFG[2])
    ticker_ns = ticker if ticker.endswith(".NS") else ticker + ".NS"
    cache_key = f"{ticker_ns}|m{mode}"

    with _BT_LOCK:
        if cache_key in _BT_CACHE:
            return _BT_CACHE[cache_key]

    result = 50.0
    try:
        df = _download_history(ticker_ns)
        if df is None or len(df) < 40:
            raise ValueError("insufficient data")

        close  = df["Close"].copy()
        volume = df["Volume"].copy()

        e20s = ema(close, 20)
        e50s = ema(close, 50)
        # vectorised RSI — no per-row loop, avoids pandas 2.x FutureWarning
        _d  = close.diff()
        _g  = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l  = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_series = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))

        avg_vol   = volume.rolling(20, min_periods=10).mean().shift(1)
        vol_ratio = volume / avg_vol.replace(0, np.nan)
        roll_high = close.rolling(cfg["hw"], min_periods=max(1, cfg["hw"] // 2)).max().shift(1)

        target_rsi  = _safe(row.get("RSI",      50))
        target_volr = _safe(row.get("Vol / Avg", 1))

        mask = (
            rsi_series.notna() &
            (rsi_series >= target_rsi  - cfg["rsi_tol"]) &
            (rsi_series <= target_rsi  + cfg["rsi_tol"]) &
            (vol_ratio  >= target_volr * (1 - cfg["vol_tol"])) &
            (vol_ratio  <= target_volr * (1 + cfg["vol_tol"]))
        )
        if cfg["ema_trend"]:
            mask &= (e20s > e50s)
        if cfg["near_high"]:
            mask &= (roll_high.notna() & (close >= roll_high * (1 - cfg["hp"])))

        # Mode-specific extra matching conditions (Task 7)
        if mode == 4:
            mask &= (close.pct_change(20) > 0)          # positive 20D return (institutional)
        elif mode == 5:
            mask &= (vol_ratio > 1.5)                    # strong vol spike (intraday)
        elif mode == 6:
            mask &= (e20s > e20s.shift(1))               # rising EMA20 slope (swing)

        idx = np.where(mask.values)[0]
        idx = idx[idx < len(close) - 1]         # exclude last row

        if len(idx) < 20:
            raise ValueError(f"too few samples: {len(idx)}")

        close_vals  = close.values
        green_count = int(sum(close_vals[i + 1] > close_vals[i] for i in idx))
        result = round((green_count / len(idx)) * 100, 1)

    except Exception:
        result = 50.0

    with _BT_LOCK:
        _BT_CACHE[cache_key] = result
    return result


# ── ML model cache ────────────────────────────────────────────────────
_ML_MODEL: "LogisticRegression | None" = None
_ML_SCALER: "StandardScaler | None"   = None
_ML_LOCK = threading.Lock()
_ML_TICKERS_TRAINED: list[str] = []


def _build_ml_features(close: pd.Series, volume: pd.Series) -> pd.DataFrame | None:
    """
    Build training feature matrix for one ticker.
    All computations vectorised — no per-row Python loop (BUG 3 fix).
    """
    try:
        if len(close) < 30:
            return None
        e20s      = ema(close, 20)
        e50s      = ema(close, 50)
        avg_vol   = volume.rolling(20, min_periods=5).mean().shift(1)
        vol_r     = volume / avg_vol.replace(0, np.nan)
        ema_dist  = (close / e20s.replace(0, np.nan) - 1.0) * 100
        # vectorised RSI — no per-row loop
        _d        = close.diff()
        _g        = _d.clip(lower=0).ewm(com=13, adjust=False).mean()
        _l        = (-_d.clip(upper=0)).ewm(com=13, adjust=False).mean()
        rsi_col   = 100 - (100 / (1 + _g / _l.replace(0, np.nan)))
        ret5      = close.pct_change(5)  * 100
        ret20     = close.pct_change(20) * 100
        target    = (close.shift(-1) > close).astype(int)
        ema_trend = (e20s > e50s).astype(int)

        df = pd.DataFrame({
            "rsi":       rsi_col,
            "vol_ratio": vol_r,
            "ema_dist":  ema_dist,
            "ret_5d":    ret5,
            "ret_20d":   ret20,
            "ema_trend": ema_trend,
            "target":    target,
        }).dropna()
        return df if len(df) >= 10 else None
    except Exception:
        return None


# 50-stock training universe (Task 4 — was only 15 stocks)
_ML_TRAIN_UNIVERSE: list[str] = [
    "RELIANCE.NS","TCS.NS","INFY.NS","HDFCBANK.NS","ICICIBANK.NS",
    "SBIN.NS","BAJFINANCE.NS","HCLTECH.NS","WIPRO.NS","AXISBANK.NS",
    "TATAMOTORS.NS","MARUTI.NS","LT.NS","NTPC.NS","ADANIPORTS.NS",
    "HINDALCO.NS","JSWSTEEL.NS","COALINDIA.NS","ONGC.NS","POWERGRID.NS",
    "BHARTIARTL.NS","TITAN.NS","NESTLEIND.NS","ULTRACEMCO.NS","HEROMOTOCO.NS",
    "BAJAJ-AUTO.NS","EICHERMOT.NS","M&M.NS","TATACONSUM.NS","BRITANNIA.NS",
    "TECHM.NS","INDUSINDBK.NS","KOTAKBANK.NS","ASIANPAINT.NS","GRASIM.NS",
    "DIVISLAB.NS","CIPLA.NS","DRREDDY.NS","SUNPHARMA.NS","APOLLOHOSP.NS",
    "ITC.NS","BPCL.NS","IOC.NS","GAIL.NS","VEDL.NS",
    "ZOMATO.NS","NAUKRI.NS","IRCTC.NS","DMART.NS","TRENT.NS",
]


def train_model_once(tickers_sample: list[str] | None = None) -> bool:
    """
    Train LogisticRegression on up to 50 NSE stocks (1-year history).
    Task 4 upgrades:
      • 80/20 stratified train/test split
      • Prints test accuracy to stdout
      • class_weight='balanced' to reduce overfitting
      • C=0.5 (moderate regularisation)
      • 50-stock training universe (was 15)
    Model + scaler cached in module globals; re-entrant safe.
    """
    global _ML_MODEL, _ML_SCALER, _ML_TICKERS_TRAINED

    if not _SKLEARN_OK:
        return False

    with _ML_LOCK:
        if _ML_MODEL is not None:
            return True

    sample = (list(tickers_sample)[:50]
              if (tickers_sample and len(tickers_sample) >= 5)
              else _ML_TRAIN_UNIVERSE[:50])

    all_rows: list[pd.DataFrame] = []
    for t in sample:
        df_h = _download_history(t, period="1y")
        if df_h is None:
            continue
        rows = _build_ml_features(df_h["Close"], df_h["Volume"])
        if rows is not None:
            all_rows.append(rows)

    if not all_rows:
        return False

    data = pd.concat(all_rows, ignore_index=True)
    if len(data) < 100:
        return False

    FEAT = ["rsi", "vol_ratio", "ema_dist", "ret_5d", "ret_20d", "ema_trend"]
    X = data[FEAT].values
    y = data["target"].values

    try:
        # 80/20 stratified split (Task 4)
        try:
            from sklearn.model_selection import train_test_split
            X_tr, X_te, y_tr, y_te = train_test_split(
                X, y, test_size=0.20, random_state=42, stratify=y
            )
        except Exception:
            split = int(len(data) * 0.8)
            X_tr, X_te = X[:split], X[split:]
            y_tr, y_te = y[:split], y[split:]

        scaler    = StandardScaler()
        X_tr_sc   = scaler.fit_transform(X_tr)
        X_te_sc   = scaler.transform(X_te)

        model = LogisticRegression(
            max_iter=500, C=0.5, solver="lbfgs",
            class_weight="balanced", random_state=42,
        )
        model.fit(X_tr_sc, y_tr)

        acc = model.score(X_te_sc, y_te)
        print(
            f"[ML] Model trained on {len(data)} samples "
            f"({len(sample)} tickers) — test accuracy: {acc:.3f}"
        )

        with _ML_LOCK:
            _ML_MODEL            = model
            _ML_SCALER           = scaler
            _ML_TICKERS_TRAINED  = sample
        return True
    except Exception as exc:
        print(f"[ML] Training failed: {exc}")
        return False


def predict_ml_probability(row: dict, mode: int = 2) -> float:
    """
    Returns next-day-green probability (0-100) from the trained LR model.
    Task 7: base probability is adjusted by mode-specific signal weights
    so each strategy context influences the final ML score differently:
      Mode 1 (Momentum) → boosts high-vol + near-breakout
      Mode 3 (Relaxed)  → slight confidence haircut
      Mode 4 (Institutional) → rewards RSI + 20D return
      Mode 5 (Intraday)  → rewards vol spike, penalises high RSI
      Mode 6 (Swing)     → rewards EMA distance + 5D return
    Falls back to 50 if model not ready or features invalid.
    """
    if not _SKLEARN_OK:
        return 50.0

    with _ML_LOCK:
        model  = _ML_MODEL
        scaler = _ML_SCALER

    if model is None or scaler is None:
        return 50.0

    try:
        ri    = _safe(row.get("RSI",               50))
        vol_r = _safe(row.get("Vol / Avg",           1))
        de20  = _safe(row.get("Δ vs EMA20 (%)",      0))
        r5d   = _safe(row.get("5D Return (%)",        0))
        r20d  = _safe(row.get("20D Return (%)",       0))
        d20h  = _safe(row.get("Δ vs 20D High (%)", -10))

        feat    = np.array([[ri, vol_r, de20, r5d, r20d, 1.0]])
        feat_sc = scaler.transform(feat)
        base_p  = float(model.predict_proba(feat_sc)[0][1])

        # ── mode-specific probability adjustment (Task 7) ─────────────
        adj = 0.0
        if mode == 1:                              # Momentum
            if vol_r > 1.7:              adj += 0.05
            if -2.0 <= d20h <= 0.0:      adj += 0.03
        elif mode == 2:                            # Balanced — no adjustment
            pass
        elif mode == 3:                            # Relaxed — confidence haircut
            adj -= 0.03
        elif mode == 4:                            # Institutional
            if ri > 58 and r20d > 3.0:   adj += 0.04
            if vol_r > 1.5:              adj += 0.02
        elif mode == 5:                            # Intraday
            if vol_r > 1.5:              adj += 0.06
            if ri > 60:                  adj -= 0.03
        elif mode == 6:                            # Swing
            if r5d > 1.5:               adj += 0.04
            if de20 < 3.0:              adj += 0.02   # not overextended

        final_p = float(np.clip(base_p + adj, 0.01, 0.99))
        return round(final_p * 100, 1)
    except Exception:
        return 50.0


def apply_phase43_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4.3 — Dynamic intelligence (additive only).
    Adds:
      - "Dynamic Score"
      - "Confidence Level"
    No API calls; never filters/removes rows; safe fallbacks on missing columns.
    """
    try:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return df
        except Exception:
            return df

        def _sf(row: pd.Series, keys: list[str], default: float = 0.0) -> float:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        f = float(v)
                        return f if np.isfinite(f) else default
                except Exception:
                    continue
            return default

        def _ss(row: pd.Series, keys: list[str], default: str = "") -> str:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        return str(v).strip()
                except Exception:
                    continue
            return default

        out = df.copy()

        dyn_scores: list[float] = []
        conf_levels: list[str] = []

        for idx in out.index:
            try:
                row = out.loc[idx]

                score = _sf(row, ["Score"], 0.0)
                ml_p  = _sf(row, ["ML %", "ML"], 0.0)
                bt_p  = _sf(row, ["Backtest %", "Backtest"], 0.0)
                vol_r  = _sf(row, ["Vol / Avg", "Vol/Avg", "Volume"], 1.0)

                # Market bias is optional input; we read it safely but do not invent new effects.
                _mb_raw = _ss(row, ["Market Bias"], default="")
                _ = _mb_raw  # keep read-only; intentionally not used in formula

                if vol_r > 2.0:
                    weight_score, weight_ml, weight_bt = 0.6, 0.2, 0.2
                elif 1.2 <= vol_r <= 2.0:
                    weight_score, weight_ml, weight_bt = 0.5, 0.25, 0.25
                else:
                    weight_score, weight_ml, weight_bt = 0.3, 0.3, 0.4

                dynamic_score = score * weight_score + ml_p * weight_ml + bt_p * weight_bt
                dynamic_score = float(np.clip(dynamic_score, 0.0, 100.0))

                avg_prob = (bt_p + ml_p) / 2.0
                if avg_prob > 60.0:
                    conf = "HIGH"
                elif avg_prob > 52.0:
                    conf = "MEDIUM"
                else:
                    conf = "LOW"

            except Exception:
                dynamic_score = 0.0
                conf = "LOW"

            dyn_scores.append(round(dynamic_score, 2))
            conf_levels.append(conf)

        out["Dynamic Score"] = dyn_scores
        out["Confidence Level"] = conf_levels
        return out
    except Exception:
        return df


def apply_phase44_logic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Phase 4.4 — Feedback tracking (additive only).
    Adds:
      - "Next Day Return (%)"
      - "Signal Outcome"
      - "System Accuracy"
      - "Weight Suggestion" (optional, based on System Accuracy)
    No API calls; safe fallbacks on missing outcome/price columns.
    """
    try:
        try:
            if df is None or not isinstance(df, pd.DataFrame) or df.empty:
                return df
        except Exception:
            return df

        def _sf(row: pd.Series, keys: list[str], default: float | None = None) -> float | None:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        f = float(v)
                        if np.isfinite(f):
                            return f
                except Exception:
                    continue
            return default

        def _ss(row: pd.Series, keys: list[str], default: str = "") -> str:
            for k in keys:
                try:
                    v = row.get(k)
                    if v is not None and pd.notna(v):
                        return str(v).strip()
                except Exception:
                    continue
            return default

        out = df.copy()

        next_returns: list[float] = []
        outcomes: list[str] = []
        accuracies: list[float] = []
        weight_suggestions: list[str] = []

        running_win = 0
        running_total = 0

        for idx in out.index:
            try:
                row = out.loc[idx]

                current_close = _sf(row, ["Price (₹)", "Close", "Current Close", "Price"], default=None)
                next_close = _sf(row, ["Next Close", "Next Close (₹)", "Next Close (Rs)"], default=None)

                next_ret = np.nan
                if current_close is not None and next_close is not None:
                    if current_close != 0:
                        next_ret = float(((next_close - current_close) / current_close) * 100.0)

                final_signal = _ss(row, ["Final Signal"], default="") or "AVOID"

                if final_signal in ["STRONG BUY", "BUY"]:
                    if pd.notna(next_ret):
                        if next_ret > 0:
                            outcome = "WIN"
                        elif next_ret < 0:
                            outcome = "LOSS"
                        else:
                            outcome = "NEUTRAL"
                    else:
                        outcome = "NEUTRAL"
                else:
                    outcome = "NEUTRAL"

                # Running accuracy (only count rows where we have an outcome)
                if final_signal in ["STRONG BUY", "BUY"] and pd.notna(next_ret):
                    running_total += 1
                    if outcome == "WIN":
                        running_win += 1

                if running_total > 0:
                    acc = (running_win / running_total) * 100.0
                else:
                    acc = 50.0

                if acc < 50.0:
                    w_s = "Reduce ML weight"
                elif acc > 65.0:
                    w_s = "Increase Score weight"
                else:
                    w_s = "Balanced"

            except Exception:
                next_ret = np.nan
                outcome = "NEUTRAL"
                acc = 50.0
                w_s = "Balanced"

            next_returns.append(next_ret)
            outcomes.append(outcome)
            accuracies.append(round(float(acc), 2))
            weight_suggestions.append(w_s)

        out["Next Day Return (%)"] = next_returns
        out["Signal Outcome"] = outcomes
        out["System Accuracy"] = accuracies
        out["Weight Suggestion"] = weight_suggestions
        return out
    except Exception:
        return df


def compute_next_day_signal(row: dict, df: pd.DataFrame | None) -> str:
    """Compute short-term confirmational signal using last 10 days geometry."""
    if df is None or len(df) < 10:
        return "❌ No Data"
    try:
        last_10 = df.tail(10)
        closes = last_10["Close"].values
        vols = last_10["Volume"].values
        highs = last_10["High"].values if "High" in last_10.columns else closes

        if len(closes) < 10:
            return "❌ No Data"

        last_3 = closes[-3:]
        vol_last = vols[-1]
        vol_avg10 = vols.mean()
        high_10 = highs.max()
        close_today = closes[-1]

        momentum = (last_3[0] < last_3[1] < last_3[2])
        vol_spike = vol_last > 1.3 * vol_avg10
        near_breakout = close_today >= high_10 * 0.98
        overextended = _safe(row.get("Δ vs EMA20 (%)", 0)) > 7.0

        if momentum and vol_spike and near_breakout and not overextended:
            return "🔥 Strong Green"
        elif momentum and near_breakout:
            return "🟢 Possible Up"
        elif overextended:
            return "⚠️ Risky (Late Entry)"
        else:
            return "❌ Weak Setup"
    except Exception:
        return "❌ Error"


def check_bull_trap(row: dict) -> str:
    """Return warning string or empty string."""
    ri    = _safe(row.get("RSI",         50))
    vol_r = _safe(row.get("Vol / Avg",    1))
    de20  = _safe(row.get("Δ vs EMA20 (%)", 0))

    traps = []
    if ri > 72:
        traps.append("RSI overbought")
    if vol_r < 1.0:
        traps.append("vol declining")
    if de20 > 6.5:
        traps.append("far from EMA20")

    return "⚠️ Bull Trap" if len(traps) >= 2 else ""


def enhance_results(results: list[dict], mode: int) -> pd.DataFrame:
    """
    Given raw scan results, attach Score / Backtest% / ML% / FinalRank.
    Uses the central ALL_DATA store for zero-API backtest computation.
    Returns a DataFrame sorted by FinalScore DESC.
    """
    if not results:
        return pd.DataFrame()

    _eng_score, _eng_bt, _eng_ml, _eng_trap = get_engine_functions(mode)
    max_workers = 10

    tickers = [
        (r.get("Ticker") or r.get("Symbol") or "") + ".NS"
        if not (r.get("Ticker") or r.get("Symbol") or "").endswith(".NS")
        else (r.get("Ticker") or r.get("Symbol") or "")
        for r in results if isinstance(r, dict)
    ]
    # Data is already preloaded before run_scan, so we remove duplicate preload_all

    # ── Step 1: score all rows (fast, no I/O) ─────────────────────────
    pre_rows: list[dict] = []
    for r in results:
        sym = r.get("Ticker") or r.get("Symbol") or ""
        try:
            score, breakdown = _eng_score(r)
        except Exception:
            score, breakdown = 0.0, {}
        score = _safe(score, 0.0)
        pre_rows.append({
            "row":       r,
            "sym":       sym,
            "score":     round(score, 2),
            "breakdown": breakdown if isinstance(breakdown, dict) else {},
        })

    # ── Step 2: backtest top 50 only (zero-API via ALL_DATA) ──────────
    top_bt = {
        id(x["row"])
        for x in sorted(pre_rows, key=lambda x: x["score"], reverse=True)[:50]
    }

    def _process_enriched(pr: dict) -> dict:
        r         = pr["row"]
        sym       = pr["sym"]
        score     = pr["score"]
        breakdown = pr["breakdown"]

        # Backtest: zero-API — reads from ALL_DATA via backtest_with_preloaded
        if id(r) in top_bt:
            try:
                bt_prob = float(backtest_with_preloaded(mode, r, sym))
            except Exception:
                bt_prob = 50.0
        else:
            bt_prob = 50.0

        try:
            ml_prob = float(_eng_ml(r))
        except Exception:
            ml_prob = 50.0
        try:
            trap = _eng_trap(r)
        except Exception:
            trap = ""

        try:
            df_for_signal = get_df_for_ticker(sym)
            # TT guard: truncate signal df to cutoff so next-day signal
            # is computed on historical data only
            try:
                _tt_sig_cut = _tt.get_reference_date()
                if _tt_sig_cut is not None and df_for_signal is not None:
                    _sig_mask = pd.to_datetime(df_for_signal.index).date <= _tt_sig_cut
                    df_for_signal = df_for_signal.loc[_sig_mask]
                    if df_for_signal.empty:
                        df_for_signal = None
            except Exception:
                pass
            nd_signal = compute_next_day_signal(r, df_for_signal)
        except Exception:
            nd_signal = "❌ Error"

        bt_prob = _safe(bt_prob, 50.0)
        ml_prob = _safe(ml_prob, 50.0)
        final   = round(0.5 * score + 0.3 * bt_prob + 0.2 * ml_prob, 2)
        return {
            **r,
            "Score":       score,
            "_breakdown":  breakdown,
            "Backtest %":  round(bt_prob, 2),
            "ML %":        round(ml_prob, 2),
            "Final Score": final,
            "Trap":        trap,
            "Next-Day Signal": nd_signal,
        }

    rows: list[dict] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(_process_enriched, pr): pr for pr in pre_rows}
        for fut in as_completed(futs):
            try:
                rows.append(fut.result())
            except Exception:
                pr    = futs[fut]
                r     = pr["row"]
                score = _safe(pr.get("score", 0.0), 0.0)
                rows.append({
                    **r,
                    "Score":       round(score, 2),
                    "_breakdown":  pr.get("breakdown", {}),
                    "Backtest %":  50.0,
                    "ML %":        50.0,
                    "Final Score": round(0.5 * score + 0.3 * 50.0 + 0.2 * 50.0, 2),
                    "Trap":        "",
                    "Next-Day Signal": "❌ Error",
                })

    df = pd.DataFrame(rows).sort_values("Final Score", ascending=False).reset_index(drop=True)
    df.index += 1
    return df


def _score_color(v: float) -> str:
    if v > 75:   return "#00d4a8"
    if v > 60:   return "#0094ff"
    if v > 40:   return "#f0b429"
    return "#ff4d6d"


def _score_label(v: float) -> str:
    if v > 75:   return "score-green"
    if v > 60:   return "score-blue"
    if v > 40:   return "score-yellow"
    return "score-red"


def render_top_picks(df: pd.DataFrame, n: int = 5) -> None:
    """Render the Top N pick cards in a horizontal strip."""
    st.markdown('<div class="section-lbl">🏅 Top Picks</div>', unsafe_allow_html=True)
    cols = st.columns(min(n, len(df)))
    for i, (col, (_, row)) in enumerate(zip(cols, df.head(n).iterrows())):
        sc   = row.get("Score",      0)
        bt   = row.get("Backtest %", 50)
        ml   = row.get("ML %",       50)
        fin  = row.get("Final Score", 0)
        trap = row.get("Trap",        "")
        sym  = row.get("Symbol",     "—")
        nd_sig= row.get("Next-Day Signal", "❌ No Data")
        c    = _score_color(sc)
        bd   = row.get("_breakdown", {})
        bd_html = "".join(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span>{k}</span>'
            f'<span style="color:{"#00d4a8" if v>=0 else "#ff4d6d"}">{v:+d}</span></div>'
            for k, v in bd.items()
        )
        with col:
            st.markdown(
                f'<div class="pick-card">'
                f'<div class="pick-rank">#{i+1}</div>'
                f'<div class="pick-sym">{sym}</div>'
                f'<div class="pick-score">'
                f'Score <span style="color:{c};font-weight:700">{sc:.0f}</span> &nbsp;|&nbsp; '
                f'BT <span style="color:#0094ff">{bt:.0f}%</span> &nbsp;|&nbsp; '
                f'ML <span style="color:#b08cff">{ml:.0f}%</span><br>'
                f'<b style="color:{c}">Final {fin:.1f}</b>'
                f'<br><span style="font-size:12px;color:#ccd9e8;font-weight:bold;">{nd_sig}</span>'
                f'{"&nbsp;&nbsp;<span class=trap-badge>" + trap + "</span>" if trap else ""}'
                f'</div>'
                f'<details style="margin-top:8px">'
                f'<summary style="font-size:11px;color:#4a6480;cursor:pointer">Score breakdown ▾</summary>'
                f'<div class="breakdown-box" style="margin-top:6px">{bd_html}</div>'
                f'</details>'
                f'</div>',
                unsafe_allow_html=True,
            )


# ═════════════════════════════════════════════════════════════════════
# SIDEBAR  (unchanged logic, unchanged options)
# ═════════════════════════════════════════════════════════════════════
mode_names  = {1:"Momentum", 2:"Balanced", 3:"Relaxed", 4:"Institutional", 5:"Intraday", 6:"Swing"}
mode_colors = {1:"#00d4a8",  2:"#0094ff",  3:"#f0b429", 4:"#b08cff", 5:"#ff8c00", 6:"#ff4d6d"}
pill_cls    = {1:"pill-m1",  2:"pill-m2",  3:"pill-m3", 4:"pill-m3", 5:"pill-m5", 6:"pill-m6"}

with st.sidebar:
    st.markdown(
        '<div style="font-family:\'Syne\',sans-serif;font-weight:800;font-size:20px;'
        'color:#00d4a8;letter-spacing:-0.5px;padding:4px 0 16px 0;">'
        '<span class="live-dot"></span>NSE SENTINEL</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Strategy Mode</div>', unsafe_allow_html=True)

    # ── FIX 8 — Mode hint from cached market regime ───────────────────
    try:
        _hint_bias = st.session_state.get("market_bias_result") or {}
        _hint_regime = str(_hint_bias.get("regime", "")).strip()
        _REGIME_HINTS = {
            "Trending Up": "💡 Mode 4 (Institutional) or Mode 1 (Momentum) recommended",
            "Ranging":     "💡 Mode 3 (Relaxed) or Mode 6 (Swing) recommended",
            "High Vol":    "💡 Mode 5 (Intraday) recommended — tight stops",
            "Bearish":     "💡 Caution — all modes show elevated risk",
        }
        _hint_text = _REGIME_HINTS.get(_hint_regime, "")
        if _hint_text:
            st.info(_hint_text)
    except Exception:
        pass

    mode_map = {
        "🟢  Mode 1 — Momentum (Breakout)":       1,
        "🔵  Mode 2 — Balanced (Swing)":          2,
        "🟡  Mode 3 — Relaxed (Early Setup)":     3,
        "🟣  Mode 4 — Institutional Strength":    4,
        "🟠  Mode 5 — Intraday":                  5,
        "🔴  Mode 6 — Swing":                     6,
    }
    mode_label = st.selectbox(
        "Strategy mode",
        list(mode_map.keys()),
        label_visibility="collapsed",
        key="strategy_mode",
    )
    mode = mode_map[mode_label]

    filter_data = {
        1: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.7× avg"),
            ("RSI","55 – 72"),("Price Floor","₹70"),
            ("10D High","Break above"),("Mkt Cap","> ₹500 Cr")],
        2: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.5× avg"),
            ("RSI","52 – 70"),("Price","₹50 – 1500"),
            ("15D High","Within 3%"),("Mkt Cap","> ₹500 Cr")],
        3: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.3× avg"),
            ("RSI","50 – 72"),("Price Floor","₹50"),
            ("20D High","Within 5%"),("Mkt Cap","—")],
        4: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.5× avg"),
            ("RSI","55 – 70"),("20D High","Within 2%"),
            ("Rel Strength","Stock > Nifty 20D"),("Price Action","Close > Open")],
        5: [("EMA Trend","Close > EMA20 > EMA50"),("Volume","> 1.1× avg"),
            ("RSI","52 – 60"),("Price Floor","₹20"),
            ("10D High","Break above"),("Price Action","Close > Open")],
        6: [("EMA Trend","Close > EMA20 > EMA50"),("EMA20 Slope","Rising"),
            ("Volume","> 1.3× avg & > prev"),("RSI","53 – 59"),
            ("Price Floor","₹40"),("10D High","Break above")],
    }
    colors = {1:"#00d4a8",2:"#0094ff",3:"#f0b429",4:"#b08cff",5:"#ff8c00",6:"#ff4d6d"}
    mc = colors[mode]

    params_html = "".join([
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:7px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480">{k}</span>'
        f'<span style="font-size:11px;color:{mc};font-weight:700">{v}</span></div>'
        for k, v in filter_data[mode]
    ])
    st.markdown(
        f'<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;'
        f'padding:12px 14px;margin-bottom:16px;">{params_html}</div>',
        unsafe_allow_html=True)

    st.markdown('<div class="section-lbl">Parallel Workers</div>', unsafe_allow_html=True)
    workers = st.slider(
        "Parallel workers",
        min_value=4,
        max_value=MAX_YF_CONCURRENCY,
        value=MAX_YF_CONCURRENCY,
        step=2,
        label_visibility="collapsed",
        key="workers",
    )
    st.markdown(
        f'<div style="text-align:center;font-size:11px;color:#4a6480;margin-top:-8px;">'
        f'<b style="color:{mc};font-size:18px">{workers}</b> worker threads '
        f'(Yahoo requests internally capped at {MAX_YF_CONCURRENCY})</div>',
        unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    scan_clicked = st.button("▶  SCAN MARKET NOW")
    sector_screener_clicked = st.button("🔭 Sector Screener Dashboard", key="sector_screener_dashboard_btn")
    battle_compare_clicked = st.button("⚔️ Compare Stocks", key="battle_compare_btn")
    csv_scan_clicked = st.button("📂 CSV Next-Day Potential", key="csv_next_day_btn")
    aura_clicked = st.button("🔮 Stock Aura", key="stock_aura_btn")

    if battle_compare_clicked:
        st.session_state["battle_show_panel"] = True
    if csv_scan_clicked:
        st.session_state["csv_next_day_show_panel"] = True
    if aura_clicked:
        st.session_state["aura_show_panel"] = True

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── 🕰️ Time Travel Mode ───────────────────────────────────────
    st.markdown('<div class="section-lbl">🕰️ Time Travel Mode</div>', unsafe_allow_html=True)
    _tt_toggle = st.toggle(
        "Simulate a past market date",
        value=st.session_state.get("tt_toggle_val", False),
        key="tt_toggle",
    )
    st.session_state["tt_toggle_val"] = _tt_toggle

    if _tt_toggle:
        _tt_min     = datetime(2023, 1, 1).date()
        _tt_max     = (datetime.now() - timedelta(days=1)).date()
        _tt_default = st.session_state.get("tt_date_val", _tt_max)
        _tt_selected = st.date_input(
            "Market date to simulate",
            value=_tt_default,
            min_value=_tt_min,
            max_value=_tt_max,
            key="tt_date_picker",
            label_visibility="collapsed",
        )
        if _tt_selected is None:
            _tt_selected = _tt_max
        st.session_state["tt_date_val"] = _tt_selected
        st.markdown(
            f'<div style="background:#1a0a00;border:1px solid #f0b429;border-radius:8px;'
            f'padding:8px 12px;font-size:11px;color:#f0b429;margin-top:4px;line-height:1.6;">'
            f'🕰️ <b>SIMULATING</b><br>'
            f'{_tt_selected.strftime("%d %b %Y")} (Post-Market Close)<br>'
            f'<span style="color:#4a6480;">All scans use data up to this date only</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.session_state["tt_date_val"] = None
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;">Live mode — using current market data</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<hr>", unsafe_allow_html=True)

    # ── Data Management Panel ─────────────────────────────────────
    st.markdown('<div class="section-lbl">📦 Local Data Cache</div>', unsafe_allow_html=True)
    if _DATA_DOWNLOADER_OK:
        if st.button("🔄 Download / Refresh Data Now", key="refresh_data_btn"):
            with st.spinner("Updating data..."):
                try:
                    _tickers_for_dl = fetch_nse_tickers()
                    update_all_data(_tickers_for_dl)
                    st.success("Data updated")
                except Exception as _e:
                    st.error(f"Data update failed: {_e}")
        # Show cache status
        try:
            _status = data_status_summary(fetch_nse_tickers())
            st.markdown(
                f'<div style="font-size:11px;color:#4a6480;line-height:1.9;">'
                f'Fresh: <b style="color:#00d4a8">{_status.get("fresh", "?")}</b> &nbsp;'
                f'Stale: <b style="color:#f0b429">{_status.get("stale", "?")}</b> &nbsp;'
                f'Missing: <b style="color:#ff4d6d">{_status.get("missing", "?")}</b></div>',
                unsafe_allow_html=True
            )
        except Exception:
            pass
    else:
        st.markdown(
            '<div style="font-size:11px;color:#4a6480;">'
            'data_downloader.py not found — using live yfinance.</div>',
            unsafe_allow_html=True
        )

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:11px;color:#4a6480;line-height:1.7;">'
        'Data: Yahoo Finance (NSE)<br>Indicators: EMA · RSI · Volume<br>'
        'Universe: NSE equity archive<br><br>'
        '⚠️ Educational use only.<br>Not financial advice.</div>',
        unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# FIX 1 — Startup background ML training (fires once on app boot)
# Non-blocking daemon thread — UI never waits for this.
# The scan handler still has its own fallback call if model isn't ready.
# ─────────────────────────────────────────────────────────────────────
if _SKLEARN_OK and _ML_MODEL is None and not st.session_state.get("_startup_ml_fired"):
    st.session_state["_startup_ml_fired"] = True
    def _startup_bg_train():
        try:
            train_model_once()
        except Exception:
            pass
    threading.Thread(target=_startup_bg_train, daemon=True).start()

# ─────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────
mc = mode_colors[mode]

st.markdown(
    f'<div class="top-banner">'
    f'<div class="banner-logo"><span class="live-dot"></span>NSE SENTINEL</div>'
    f'<div style="margin-left:auto">'
    f'<span class="mode-pill {pill_cls[mode]}">MODE {mode} · {mode_names[mode].upper()}</span>'
    f'</div></div>',
    unsafe_allow_html=True)
st.markdown(
    f'<p style="color:#4a6480;font-size:12px;font-family:\'Space Mono\',monospace;'
    f'margin-top:-8px;margin-bottom:20px;">'
    f'Automated multi-strategy scanner for NSE equities · '
    f'{(_tt.get_reference_datetime()).strftime("%d %b %Y, %H:%M")}'
    f'{"  🕰️ TIME TRAVEL" if _tt.is_active() else ""}</p>',
    unsafe_allow_html=True)

with st.spinner("Loading NSE ticker list..."):
    all_tickers = fetch_nse_tickers()
n = len(all_tickers)

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("📋 NSE Tickers Loaded", f"{n:,}")
with c2: st.metric("🎯 Active Mode", f"M{mode} · {mode_names[mode]}")
with c3: st.metric("⚡ Workers", workers)
with c4:
    found_val  = len(st.session_state.get("results", [])) if "results" in st.session_state else None
    elapsed_v  = st.session_state.get("elapsed", None)
    if found_val is not None:
        st.metric("✅ Last Scan Found", found_val,
                  delta=f"{elapsed_v:.1f}s" if elapsed_v else None)
    else:
        st.metric("✅ Last Scan Found", "—")

st.markdown("<hr>", unsafe_allow_html=True)

# ── SCAN ──────────────────────────────────────────────────────────────

# ── 🕰️ Time-travel banner (shown whenever TT is active) ───────────────
_tt_banner = _tt.format_banner()
if _tt_banner:
    st.markdown(
        f'<div style="background:#1a0a00;border:2px solid #f0b429;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-family:\'Space Mono\',monospace;'
        f'font-size:13px;font-weight:700;color:#f0b429;letter-spacing:0.3px;">'
        f'{_tt_banner}</div>',
        unsafe_allow_html=True,
    )

if scan_clicked:
    st.markdown(
        f'<div class="section-lbl">⏳ Scanning {n:,} NSE Equities — Mode {mode}: {mode_names[mode]}</div>',
        unsafe_allow_html=True)

    # Background ML pre-train (non-blocking daemon thread)
    if _SKLEARN_OK and _ML_MODEL is None:
        def _bg_train():
            try:
                from strategy_engines import get_train_function
                get_train_function(mode)()
            except Exception:
                train_model_once(all_tickers[:50])

        threading.Thread(target=_bg_train, daemon=True).start()

    # ── 🕰️ Activate time-travel BEFORE scan if toggle is on ───────────
    _tt_active_date = st.session_state.get("tt_date_val")
    if _tt_active_date is not None and _TIME_TRAVEL_OK:
        with st.spinner(f"🕰️ Preparing historical snapshot for {_tt_active_date.strftime('%d %b %Y')}…"):
            _snapped = _tt.activate(_tt_active_date)
        st.caption(f"🕰️ Time-travel active — {_snapped} ticker(s) snapshotted to {_tt_active_date}")

    try:
        results, elapsed = run_scan(all_tickers, mode, workers=workers)
    finally:
        # Always restore — even if scan raised an exception
        if _tt_active_date is not None and _TIME_TRAVEL_OK:
            _tt.restore()

    _scan_time_label = (
        _tt_active_date.strftime("%d %b %Y (TT)") if _tt_active_date
        else datetime.now().strftime("%H:%M:%S")
    )
    st.session_state.update({
        "results":       results,
        "scan_time":     _scan_time_label,
        "elapsed":       elapsed,
        "mode":          mode,
        "tt_was_active": _tt_active_date is not None,
        "tt_scan_date":  str(_tt_active_date) if _tt_active_date else "",
    })

    # FIX 6 — Auto-backfill actual returns for past predictions (background)
    try:
        from prediction_feedback_store import backfill_actual_returns
        from strategy_engines._engine_utils import ALL_DATA
        def _bg_backfill():
            try:
                backfill_actual_returns(ALL_DATA)
            except Exception:
                pass
        threading.Thread(target=_bg_backfill, daemon=True).start()
    except Exception:
        pass

    st.rerun()

# ── SECTOR SCREENER DASHBOARD ─────────────────────────────────────
if sector_screener_clicked:
    st.session_state["show_sector_screener"] = True

if st.session_state.get("show_sector_screener", False):
    if _SECTOR_SCREENER_UI_OK:
        render_sector_screener_dashboard(
            mode=mode,
            enhance_results_fn=enhance_results,
            apply_enhanced_logic_fn=apply_enhanced_logic,
            apply_universal_grading_fn=apply_universal_grading,
            apply_phase4_logic_fn=apply_phase4_logic,
            apply_phase42_logic_fn=apply_phase42_logic,
            compute_market_bias_fn=compute_market_bias,
        )
    else:
        st.warning("⚠️ Sector Screener UI not available. Restart after ensuring the file exists.")

# ── NEW: MARKET BIAS UI PANEL (Isolated) ──────────────────────────────
if st.session_state.get("battle_show_panel", False):
    st.markdown("<hr>", unsafe_allow_html=True)
    _battle_hdr_col, _battle_close_col = st.columns([6, 1])
    with _battle_hdr_col:
        st.markdown('<h2>⚔️ Compare Stocks</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
            'This panel now opens in the main UI. Enter up to 10 stocks and run the full comparison here.</div>',
            unsafe_allow_html=True,
        )
    with _battle_close_col:
        st.markdown("<br>", unsafe_allow_html=True)
        _battle_close_panel = st.button("Close", key="battle_close_panel_btn", use_container_width=True)

    if _battle_close_panel:
        st.session_state["battle_show_panel"] = False
        st.rerun()

    _battle_input_col1, _battle_input_col2 = st.columns(2)
    with _battle_input_col1:
        _t1  = st.text_input("Stock 1",  key="battle_t1",  placeholder="e.g. RELIANCE")
        _t2  = st.text_input("Stock 2",  key="battle_t2",  placeholder="e.g. TCS")
        _t3  = st.text_input("Stock 3",  key="battle_t3",  placeholder="e.g. INFY")
        _t4  = st.text_input("Stock 4",  key="battle_t4",  placeholder="e.g. HDFCBANK")
        _t5  = st.text_input("Stock 5",  key="battle_t5",  placeholder="e.g. SBIN")
    with _battle_input_col2:
        _t6  = st.text_input("Stock 6",  key="battle_t6",  placeholder="e.g. ICICIBANK")
        _t7  = st.text_input("Stock 7",  key="battle_t7",  placeholder="e.g. AXISBANK")
        _t8  = st.text_input("Stock 8",  key="battle_t8",  placeholder="e.g. BAJFINANCE")
        _t9  = st.text_input("Stock 9",  key="battle_t9",  placeholder="e.g. TATAMOTORS")
        _t10 = st.text_input("Stock 10", key="battle_t10", placeholder="e.g. MARUTI")

    _battle_main_run = st.button("Run Battle Analysis", key="battle_run_btn", use_container_width=True)
    if _battle_main_run:
        _all_inputs = [_t1, _t2, _t3, _t4, _t5, _t6, _t7, _t8, _t9, _t10]
        _battle_tickers = [t.strip() for t in _all_inputs if t and t.strip()][:10]
        if not _battle_tickers:
            st.warning("Please enter at least 1 stock.")
        else:
            st.session_state["battle_mode_request"] = mode
            st.session_state["battle_tickers_request"] = _battle_tickers

if st.session_state.get("show_bias_engine"):
    st.markdown('<div class="section-lbl">📊 Market Bias Engine (Analytics)</div>', unsafe_allow_html=True)
    with st.spinner("Crunching latest Nifty (^NSEI) indicators..."):
        _ui_tt_key = str(st.session_state.get("tt_date_val") or "live")
        _bias_data = compute_market_bias_ui(_ui_tt_key)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Market Bias", _bias_data["bias"])
    with col2:
        st.metric("Confidence", f"{_bias_data['confidence']}%")
    with col3:
        st.metric("Expected Move", _bias_data["expected_move"])

    with st.expander("Short Reason Breakdown", expanded=True):
        for r in _bias_data["reasons"]:
            st.markdown(f"- {r}")

    if st.button("Close Bias Panel"):
        st.session_state["show_bias_engine"] = False
        st.rerun()

    st.markdown("<hr>", unsafe_allow_html=True)

# ── RESULTS ───────────────────────────────────────────────────────────
if "results" in st.session_state:
    results     = st.session_state["results"]
    stored_mode = st.session_state.get("mode", mode)
    elapsed_d   = st.session_state.get("elapsed", 0)
    scan_time_d = st.session_state.get("scan_time", "—")
    mc2         = mode_colors.get(stored_mode, "#00d4a8")

    st.markdown(
        f'<div class="result-hdr">'
        f'<h3>🏆 Bullish Candidates</h3>'
        f'<span class="mode-pill {pill_cls[stored_mode]}">M{stored_mode} · {mode_names[stored_mode].upper()}</span>'
        f'<span class="count-pill">{len(results)}</span>'
        f'<span style="margin-left:auto;font-size:11px;color:#4a6480;">'
        f'Scanned at {scan_time_d} · {elapsed_d:.1f}s</span></div>',
        unsafe_allow_html=True)

    if results:
        # ── Summary metrics (unchanged) ──────────────────────────────
        df_raw = pd.DataFrame(results)
        i1, i2, i3 = st.columns(3)
        with i1: st.metric("📊 Avg RSI",      f"{df_raw['RSI'].mean():.1f}")
        with i2: st.metric("💰 Avg Price",    f"₹{df_raw['Price (₹)'].mean():,.0f}")
        with i3: st.metric("⚡ Avg Vol / Avg", f"{df_raw['Vol / Avg'].mean():.2f}×")

        # ── Enhance with scoring / backtest / ML ─────────────────────
        with st.spinner("🔢 Computing Smart Scores, Backtest & ML probabilities …"):
            df = enhance_results(results, stored_mode)

        # ── Enhanced Logic Engine — runs FIRST so Setup Quality /
        # ── Volume Trend / Trap Risk are available for Prediction Score
        try:
            df = apply_enhanced_logic(df)
        except Exception:
            pass

        # ── Universal Grading Engine ──────────────────────────────────
        try:
            # FIX 2: reuse cached bias if younger than 30 min
            _now_ts = time.time()
            _cached_mb   = st.session_state.get("market_bias_result")
            _cached_ts   = st.session_state.get("market_bias_ts", 0.0)
            _cached_ttkey = st.session_state.get("market_bias_tt_key", "live")
            _cur_ttkey   = str(st.session_state.get("tt_date_val") or "live")
            _cache_valid = (
                _cached_mb
                and (_now_ts - float(_cached_ts)) < 1800
                and _cached_ttkey == _cur_ttkey   # bust cache on TT date change
            )
            if _cache_valid:
                _mb = _cached_mb
            else:
                _mb = compute_market_bias()
                st.session_state["market_bias_result"]  = _mb
                st.session_state["market_bias_ts"]      = _now_ts
                st.session_state["market_bias_tt_key"]  = _cur_ttkey
            df = apply_universal_grading(df, _mb)
        except Exception:
            pass

        # ── Phase 4 Logic Engine (Setup Type, Reason, Risk, Final Signal)
        try:
            df = apply_phase4_logic(df, _mb)
        except Exception:
            pass

        # ── Phase 4.2 Logic Engine (Advanced Trap, Expected Move, Adjusted Signal)
        try:
            df = apply_phase42_logic(df)
        except Exception:
            pass

        # ── Phase 4.3/4.4 (Dynamic Intelligence + Feedback Tracking) ─
        try:
            df = apply_phase43_logic(df)
        except Exception:
            pass
        try:
            df = apply_phase44_logic(df)
        except Exception:
            pass

        st.session_state["last_scan_df"] = df.copy()

        try:
            from prediction_feedback_store import feedback_summary, log_scan_predictions

            _log_key = f"{stored_mode}|{scan_time_d}|{len(df)}"
            if st.session_state.get("_prediction_log_key") != _log_key:
                log_scan_predictions(df, stored_mode, st.session_state.get("market_bias_result"))
                st.session_state["_prediction_log_key"] = _log_key
            _fs = feedback_summary()
            if _fs.get("total_logged", 0):
                _cap = f"📒 Prediction log: {_fs['total_logged']} row(s) stored"
                if _fs.get("rows_with_outcome"):
                    _cap += f"; {_fs.get('rows_with_outcome', 0)} with outcomes"
                if _fs.get("accuracy_pct") is not None and _fs.get("rows_with_outcome", 0):
                    _cap += f". Recent accuracy: {_fs.get('accuracy_pct')}%"
                _cap += "."
                st.caption(_cap)
        except Exception:
            pass

        if "Next-Day Signal" in df.columns:
            with st.sidebar:
                counts = df["Next-Day Signal"].value_counts().to_dict()
                sg = counts.get("🔥 Strong Green", 0)
                pu = counts.get("🟢 Possible Up", 0)
                ri = counts.get("⚠️ Risky (Late Entry)", 0)
                we = counts.get("❌ Weak Setup", 0)

                st.markdown('<hr><div class="section-lbl">📊 Next-Day Signals Summary</div>', unsafe_allow_html=True)
                _nd_summary = (
                    '<div style="background:#0f1823;border:1px solid #1a2840;border-radius:10px;padding:12px 14px;margin-bottom:16px;">'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">🔥 Strong Green</span>'
                    '<span style="font-size:13px;color:#00d4a8;font-weight:700">{sg}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">🟢 Possible Up</span>'
                    '<span style="font-size:13px;color:#0094ff;font-weight:700">{pu}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">⚠️ Risky (Late Entry)</span>'
                    '<span style="font-size:13px;color:#f0b429;font-weight:700">{ri}</span></div>'
                    '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;">'
                    '<span style="font-size:12px;color:#ccd9e8;">❌ Weak Setup</span>'
                    '<span style="font-size:13px;color:#ff4d6d;font-weight:700">{we}</span></div>'
                    '</div>'
                ).format(sg=sg, pu=pu, ri=ri, we=we)
                st.markdown(_nd_summary, unsafe_allow_html=True)

        # ── TASK 4: Section Headers & Spacing ─────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<h2>🔥 Top Picks</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 1: Top Picks Cards ───────────────────────────────────
        top_n = min(5, len(df))
        cols = st.columns(top_n)
        for i, (col, (_, row)) in enumerate(zip(cols, df.head(top_n).iterrows())):
            sym = row.get("Symbol", "—")
            fin   = row.get("Final Score", 0)
            bt    = row.get("Backtest %", 50)
            ml    = row.get("ML %", 50)
            rsi_v = float(row.get("RSI", 0))
            vol   = row.get("Vol / Avg", 0)
            trap  = row.get("Trap", "")
            tv_link = tv_chart_url(sym)
            nd_sig  = row.get("Next-Day Signal", "❌ No Data")

            # Score colour
            if fin > 75:     color = "#00d4a8"
            elif fin >= 60:  color = "#0094ff"
            elif fin >= 40:  color = "#f0b429"
            else:            color = "#ff4d6d"

            # FIX 7 — Stop-loss and target (display-only, not added to df)
            # SL  = Price × (1 − (Δ vs EMA20 % / 100) × 0.5)  half EMA distance below
            # Tgt = Price × (1 + (Δ vs EMA20 % / 100) × 1.5)  1.5× EMA distance above
            try:
                _price   = float(row.get("Price (₹)", 0) or 0)
                _de20    = float(row.get("Δ vs EMA20 (%)", 0) or 0)
                if _price > 0 and _de20 != 0:
                    _sl  = round(_price * (1 - (_de20 / 100) * 0.5), 2)
                    _tgt = round(_price * (1 + (_de20 / 100) * 1.5), 2)
                    _sl_tgt_html = (
                        f'<div style="font-size:11px;color:#4a6480;margin-top:8px;">'
                        f'SL ₹{_sl:,.2f}&nbsp;&nbsp;|&nbsp;&nbsp;Tgt ₹{_tgt:,.2f}</div>'
                    )
                else:
                    _sl_tgt_html = ""
            except Exception:
                _sl_tgt_html = ""

            with col:
                trap_html = (
                    f'<div style="margin-top:8px;color:#ff4d6d;font-size:12px;'
                    f'font-weight:bold;background:rgba(255,77,109,0.1);'
                    f'padding:4px 8px;border-radius:4px;display:inline-block;">⚠️ {trap}</div>'
                    if trap else ""
                )
                st.markdown(
                    f'<div style="border:1px solid #1a2840;padding:16px;border-radius:8px;'
                    f'background:#0f1823;position:relative;">'
                    f'<div style="font-size:20px;font-weight:bold;margin-bottom:8px;">{sym}</div>'
                    f'<div style="font-size:32px;font-weight:bold;color:{color};margin-bottom:12px;">{fin:.1f}</div>'
                    f'<div style="font-size:14px;font-weight:bold;color:#ccd9e8;margin-bottom:8px;">{nd_sig}</div>'
                    f'<div style="font-size:14px;color:#ccd9e8;line-height:1.6;">'
                    f'<b>BT:</b> {bt:.1f}%&nbsp; '
                    f'<b>ML:</b> {ml:.1f}%&nbsp; '
                    f'<b>RSI:</b> {rsi_v:.1f}&nbsp; '
                    f'<b>Vol:</b> {vol:.1f}x'
                    f'</div>{_sl_tgt_html}{trap_html}</div>',
                    unsafe_allow_html=True,
                )
                st.link_button("📈 TradingView", tv_link, use_container_width=True)

        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown('<h2>📊 Full Rankings</h2>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # ── TASK 2 & 5: Clean Table with TradingView Link ─────────────
        table_df = df.copy()
        try:
            from strategy_engines._engine_utils import add_rank_score_columns
            table_df = add_rank_score_columns(table_df)
            if "rank_score" in table_df.columns:
                table_df = table_df.sort_values("rank_score", ascending=False).reset_index(drop=True)
                table_df["Rank Score"] = table_df["rank_score"]
        except Exception:
            pass
        if "Rank Score" not in table_df.columns:
            table_df["Rank Score"] = 0.0
        table_df.insert(0, "Rank", range(1, len(table_df) + 1))
        table_df["Ticker"] = table_df["Symbol"]
        table_df["TradingView"] = table_df["Symbol"].apply(lambda s: tv_chart_url(s))

        display_cols = [
            "Rank", "Rank Score", "Ticker", "Score", "Backtest %", "ML %",
            "Final Score", "Prediction Score", "Conviction Tier", "Trap", "Next-Day Signal", "TradingView",
        ]
        display_cols = [c for c in display_cols if c in table_df.columns]

        st.dataframe(
            table_df[display_cols],
            column_config={
                "Rank": st.column_config.NumberColumn("Rank"),
                "Rank Score": st.column_config.NumberColumn("Rank Score", format="%.2f"),
                "Ticker": st.column_config.TextColumn("Ticker", width="medium"),
                "Score": st.column_config.NumberColumn("Score", format="%.0f"),
                "Backtest %": st.column_config.NumberColumn("Backtest %", format="%.1f%%"),
                "ML %": st.column_config.NumberColumn("ML %", format="%.1f%%"),
                "Final Score": st.column_config.NumberColumn("Final Score", format="%.2f"),
                "Prediction Score": st.column_config.NumberColumn("Pred Score", format="%.1f"),
                "Conviction Tier": st.column_config.TextColumn("Conviction"),
                "Trap": st.column_config.TextColumn("Trap"),
                "Next-Day Signal": st.column_config.TextColumn("Signal"),
                "TradingView": st.column_config.LinkColumn("TradingView Link", display_text="📈 Open Chart"),
            },
            use_container_width=True,
            hide_index=True,
        )

        # ── TASK 3: Expandable Details ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        for _, row in df.iterrows():
            sym = row.get("Symbol", "—")
            fin = row.get("Final Score", 0)
            with st.expander(f"🔍 {sym} Details (Final Score: {fin:.1f})"):
                brk_col, ind_col = st.columns([1, 2])
                with brk_col:
                    st.markdown("**Score Breakdown**")
                    st.json(row.get("_breakdown", {}))
                with ind_col:
                    st.markdown("**Key Indicators**")
                    ic1, ic2, ic3, ic4 = st.columns(4)
                    ic1.metric("RSI", f"{row.get('RSI', 0):.1f}")
                    ic2.metric("EMA 20", f"₹{row.get('EMA 20', 0):.2f}")
                    ic3.metric("EMA 50", f"₹{row.get('EMA 50', 0):.2f}")
                    ic4.metric("Vol / Avg", f"{row.get('Vol / Avg', 0):.2f}x")
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    rc1, rc2, rc3 = st.columns(3)
                    rc1.metric("5D Return", f"{row.get('5D Return (%)', 0):+.2f}%")
                    rc2.metric("20D Return", f"{row.get('20D Return (%)', 0):+.2f}%")
                    
                    # Handle possible missing columns gracefully
                    h_val = row.get('Δ vs 20D High (%)', 0)
                    rc3.metric("Δ vs 20D Hi", f"{h_val:+.2f}%" if pd.notna(h_val) else "—")

        # ── NEW: Clean Export Helper ──────────────────────────────────
        def get_clean_export_df(df):
            """Return a clean, emoji-free copy of df for CSV/Excel export.
            Only includes columns shown in the UI table. Safe if any column
            is missing. Does NOT modify the original df.
            """
            import re

            _EXPORT_COLS = [
                "Rank", "Rank Score", "Ticker", "Score",
                "Backtest %", "ML %", "Final Score",
                "Prediction Score", "Conviction Tier", "Trap", "Next-Day Signal",
            ]

            def _strip_emoji(val):
                """Remove emoji / icon characters from a string value."""
                if not isinstance(val, str):
                    return val
                # Remove all emoji and non-ASCII symbols
                cleaned = re.sub(
                    r"[\U00010000-\U0010ffff"
                    r"\U00002600-\U000027BF"
                    r"\U0001F300-\U0001F9FF"
                    r"\u2700-\u27BF"
                    r"\u2300-\u23FF"
                    r"\u2B50-\u2B55"
                    r"\u231A-\u231B"
                    r"\u25AA-\u25FE"
                    r"\u2614-\u2615"
                    r"\u2648-\u2653"
                    r"\u26AA-\u26AB"
                    r"\u26BD-\u26BE"
                    r"\u26C4-\u26C5"
                    r"\u26CE-\u26CE"
                    r"\u26D4-\u26D4"
                    r"\u26EA-\u26EA"
                    r"\u26F2-\u26F3"
                    r"\u26F5-\u26F5"
                    r"\u26FA-\u26FA"
                    r"\u26FD-\u26FD]",
                    "",
                    val,
                    flags=re.UNICODE,
                )
                return cleaned.strip()

            _copy = df.copy()
            # Keep only columns that exist in this df
            _cols = [c for c in _EXPORT_COLS if c in _copy.columns]
            _copy = _copy[_cols]

            # Round numeric percentage columns to 1 decimal place
            for _pct_col in ["Backtest %", "ML %", "Final Score", "Prediction Score", "Rank Score"]:
                if _pct_col in _copy.columns:
                    _copy[_pct_col] = pd.to_numeric(_copy[_pct_col], errors="coerce").round(1)

            # Round Score to 0 decimal places
            if "Score" in _copy.columns:
                _copy["Score"] = pd.to_numeric(_copy["Score"], errors="coerce").round(0)

            # Strip emojis from all string columns
            for _col in _copy.select_dtypes(include="object").columns:
                _copy[_col] = _copy[_col].apply(_strip_emoji)

            return _copy

        # ── CSV download (uses clean export layer) ────────────────────
        st.markdown("<br><br>", unsafe_allow_html=True)
        _clean_export = get_clean_export_df(table_df)
        _csv_buf = io.StringIO()
        _clean_export.to_csv(_csv_buf, index=False)
        dl_col, _ = st.columns([1, 3])
        with dl_col:
            st.download_button(
                label="⬇ Download Results CSV",
                data=_csv_buf.getvalue().encode("utf-8-sig"),
                file_name=f"nse_scan_{datetime.now().strftime('%Y%m%d_%H%M')}_mode{stored_mode}.csv",
                mime="text/csv",
                use_container_width=True,
                key="main_scan_csv_download",
            )

        # ── ML status note ───────────────────────────────────────────
        if not _SKLEARN_OK:
            st.info("ℹ️  scikit-learn not installed — ML % column shows neutral 50. "
                    "Run `pip install scikit-learn` to enable ML probability.")

    else:
        if not st.session_state.get("show_sector_screener", False):
            st.markdown(
                f'<div style="text-align:center;padding:60px 24px;background:#0f1823;'
                f'border:1px solid #1a2840;border-radius:12px;">'
                f'<div style="font-size:48px;opacity:0.3;margin-bottom:16px;">📡</div>'
                f'<div style="color:#4a6480;font-size:13px;line-height:1.7;">'
                f'No stocks matched Mode {mode} criteria.<br>'
                f'Try <b style="color:#ccd9e8">Mode 3 (Relaxed)</b> for a broader scan.</div></div>',
                unsafe_allow_html=True)
else:
    if not st.session_state.get("show_sector_screener", False):
        st.markdown(
            f'<div style="text-align:center;padding:64px 24px;background:#0f1823;'
            f'border:1px solid #1a2840;border-radius:12px;">'
            f'<div style="font-size:52px;opacity:0.25;margin-bottom:18px;">📡</div>'
            f'<div style="color:#4a6480;font-size:14px;line-height:1.8;">'
            f'Select a <b style="color:#ccd9e8">strategy mode</b> in the sidebar<br>'
            f'then click <b style="color:#00d4a8">▶ SCAN MARKET NOW</b> to begin.'
            f'</div></div>',
            unsafe_allow_html=True)


# ── CSV NEXT-DAY POTENTIAL (Completely standalone — no analyse/run_scan) ──────
_csv_panel_open = bool(st.session_state.get("csv_next_day_show_panel", False))
if csv_scan_clicked or _csv_panel_open:
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown('<h2>📂 CSV Next-Day Potential</h2>', unsafe_allow_html=True)
    st.markdown(
        '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
        'Fast local scan using cached CSV data · Focused on tomorrow-up probability and stricter buy quality</div>',
        unsafe_allow_html=True
    )

    if not _DATA_DOWNLOADER_OK or not _CSV_NEXT_DAY_ENGINE_OK:
        st.warning("CSV next-day engine is not available. Check `data_downloader.py` and `csv_next_day_engine.py`.")
    else:
        if csv_scan_clicked:
            # 🕰️ Pass TT cutoff so CSV engine slices data before computing indicators
            _csv_tt_cut = st.session_state.get("tt_date_val")  # None in live mode
            with st.spinner("📂 Scanning local CSVs …"):
                try:
                    _csv_fresh_df = run_csv_next_day(None, cutoff_date=_csv_tt_cut)
                    st.session_state["csv_next_day_results_df"] = (
                        _csv_fresh_df.copy() if isinstance(_csv_fresh_df, pd.DataFrame) else pd.DataFrame()
                    )
                    st.session_state["csv_next_day_last_error"] = ""
                    _ts_label = (
                        _csv_tt_cut.strftime("%d %b %Y (TT)")
                        if _csv_tt_cut else datetime.now().strftime("%d %b %Y, %H:%M")
                    )
                    st.session_state["csv_next_day_last_scan_at"] = _ts_label
                except Exception as _csv_err:
                    st.session_state["csv_next_day_last_error"] = str(_csv_err)

        csv_df = st.session_state.get("csv_next_day_results_df", pd.DataFrame())
        csv_last_error = str(st.session_state.get("csv_next_day_last_error", "") or "").strip()
        csv_last_scan_at = str(st.session_state.get("csv_next_day_last_scan_at", "") or "").strip()

        if csv_last_scan_at:
            st.caption(f"Last CSV scan: {csv_last_scan_at}")

        if csv_last_error:
            st.error(f"CSV scan failed: {csv_last_error}")

        if isinstance(csv_df, pd.DataFrame) and not csv_df.empty:
            st.success(f"✅ {len(csv_df)} buy-ready setups matched the stricter tomorrow-up CSV criteria")
            _m1, _m2, _m3, _m4, _m5 = st.columns(5)
            with _m1:
                st.metric("Matches", f"{len(csv_df):,}")
            with _m2:
                st.metric("Avg Prob", f"{csv_df['Next Day Prob'].mean():.1f}%")
            with _m3:
                st.metric("Avg Conf", f"{csv_df['Confidence'].mean():.1f}%")
            with _m4:
                _ready_count = int((csv_df["Buy Readiness"] == "BUY READY").sum()) if "Buy Readiness" in csv_df.columns else 0
                st.metric("Buy Ready", f"{_ready_count:,}")
            with _m5:
                _best_grade = "-"
                if "Grade" in csv_df.columns:
                    _grade_order = ["A", "B", "C", "D"]
                    _grade_values = csv_df["Grade"].astype(str).tolist()
                    _best_grade = next((g for g in _grade_order if g in _grade_values), _grade_values[0] if _grade_values else "-")
                st.metric("Best Grade", _best_grade)

            _download_col, _grade_col = st.columns([0.32, 0.68])
            with _download_col:
                _csv_download_data = csv_df.to_csv(index=False).encode("utf-8-sig")
                _scan_stamp = csv_last_scan_at.replace(" ", "_").replace(",", "").replace(":", "-") if csv_last_scan_at else datetime.now().strftime("%Y-%m-%d_%H-%M")
                st.download_button(
                    "⬇️ Download CSV Results",
                    data=_csv_download_data,
                    file_name=f"csv_next_day_results_{_scan_stamp}.csv",
                    mime="text/csv",
                    key="csv_next_day_download_btn",
                )
            with _grade_col:
                if "Grade" in csv_df.columns:
                    _grade_counts = csv_df["Grade"].fillna("-").astype(str).value_counts()
                    _grade_summary = " | ".join(
                        f"{_grade}: {_grade_counts.get(_grade, 0)}" for _grade in ["A", "B", "C", "D"]
                    )
                    st.markdown(
                        '<div style="font-size:12px;color:#4a6480;padding-top:8px;">'
                        f'Grading System: <b>A</b> strongest setup · <b>B</b> good setup · '
                        f'<b>C</b> watchlist quality · <b>D</b> weak setup'
                        f'<br>Grade Distribution: {_grade_summary}</div>',
                        unsafe_allow_html=True,
                    )

            st.dataframe(
                csv_df,
                column_config={
                    "Symbol":           st.column_config.TextColumn("Ticker"),
                    "Price (₹)":        st.column_config.NumberColumn("Close (₹)", format="₹%.2f"),
                    "Next Day Prob":    st.column_config.NumberColumn("Tomorrow Up %", format="%.1f%%"),
                    "Confidence":       st.column_config.NumberColumn("Confidence %", format="%.1f%%"),
                    "Grade":            st.column_config.TextColumn("Grade"),
                    "Buy Readiness":    st.column_config.TextColumn("Buy Verdict"),
                    "Signal":           st.column_config.TextColumn("Signal"),
                    "Setup":            st.column_config.TextColumn("Setup"),
                    "Historical Win %": st.column_config.NumberColumn("Hist Win %", format="%.1f%%"),
                    "Downside Risk %":  st.column_config.NumberColumn("Downside Risk %", format="%.1f%%"),
                    "Analog Count":     st.column_config.NumberColumn("Analogs", format="%d"),
                    "Analog Avg Ret %": st.column_config.NumberColumn("Analog Avg %", format="%.2f%%"),
                    "Setup Quality":    st.column_config.NumberColumn("Setup Q", format="%.1f"),
                    "Trigger Quality":  st.column_config.NumberColumn("Trigger Q", format="%.1f"),
                    "RSI":              st.column_config.NumberColumn("RSI", format="%.1f"),
                    "Vol / Avg":        st.column_config.NumberColumn("Vol/Avg", format="%.2fx"),
                    "Volume Strength":  st.column_config.TextColumn("Volume"),
                    "Bull Trap":        st.column_config.TextColumn("Trap"),
                    "Risk Notes":       st.column_config.TextColumn("Risk Notes", width="large"),
                    "Chart Link":       st.column_config.LinkColumn("Chart", display_text="📈 Open"),
                },
                use_container_width=True,
                hide_index=True,
            )
        elif not csv_last_error:
            st.info("No clean buy-ready setups were found for tomorrow in the current CSV universe. That usually means wait for better structure instead of forcing a trade.")

# ══════════════════════════════════════════════════════════
# ⚔️  MULTI-STOCK BATTLE MODE
# ══════════════════════════════════════════════════════════
try:
    from battle_mode_engine import run_battle_mode, compute_battle_scores
    _BATTLE_OK = True
except ImportError:
    _BATTLE_OK = False

if not _BATTLE_OK:
    st.warning("⚠️ battle_mode_engine.py not found. Place it in the same folder as app.py.")
else:
    _battle_request_tickers = st.session_state.get("battle_tickers_request", None)
    _battle_mode = st.session_state.get("battle_mode_request", mode)

    # Execute the battle pipeline only when the sidebar requested it.
    if isinstance(_battle_request_tickers, list) and _battle_request_tickers:
        with st.spinner(f"⚔️ Analysing {len(_battle_request_tickers)} stock(s)…"):
            # ── 🕰️ Activate time-travel for battle if toggle is on ─────
            _tt_battle_date = st.session_state.get("tt_date_val")
            if _tt_battle_date is not None and _TIME_TRAVEL_OK:
                _tt.activate(_tt_battle_date)
            try:
                _battle_raw = run_battle_mode(_battle_request_tickers, _battle_mode)
                if not _battle_raw:
                    st.error("No valid data found. Check symbols and try again.")
                    st.session_state["battle_results_df"] = pd.DataFrame()
                else:
                    _battle_df = enhance_results(_battle_raw, _battle_mode)
                    try:
                        _battle_df = apply_enhanced_logic(_battle_df)
                    except Exception:
                        pass
                    try:
                        _mb = st.session_state.get("market_bias_result", None)
                        _mb_ttkey = st.session_state.get("market_bias_tt_key", "live")
                        _battle_ttkey = str(st.session_state.get("tt_date_val") or "live")
                        if _mb is None or _mb_ttkey != _battle_ttkey:
                            _mb = compute_market_bias()
                            st.session_state["market_bias_result"]  = _mb
                            st.session_state["market_bias_tt_key"]  = _battle_ttkey
                        _battle_df = apply_universal_grading(_battle_df, _mb)
                    except Exception:
                        pass
                    try:
                        _mb2 = st.session_state.get("market_bias_result", None)
                        _battle_df = apply_phase4_logic(_battle_df, _mb2)
                        _battle_df = apply_phase42_logic(_battle_df)
                    except Exception:
                        pass
                    _battle_df = compute_battle_scores(_battle_df)
                    st.session_state["battle_results_df"] = _battle_df
            except Exception as _battle_err:
                st.error(f"Battle Mode error: {_battle_err}. Check your tickers and try again.")
                st.session_state["battle_results_df"] = pd.DataFrame()
            finally:
                st.session_state["battle_tickers_request"] = None
                if _tt_battle_date is not None and _TIME_TRAVEL_OK:
                    _tt.restore()

    _battle_df = st.session_state.get("battle_results_df", None)
    if isinstance(_battle_df, pd.DataFrame) and not _battle_df.empty:
        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown('<h2>⚔️ Multi-Stock Battle Mode</h2>', unsafe_allow_html=True)
        st.markdown(
            '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
            'Compare up to 10 stocks head-to-head · Full pipeline per ticker · Ranks by battle probability, quality and risk-adjusted strength</div>',
            unsafe_allow_html=True,
        )

        # ── 🥇 Winner Card ────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">🥇 Battle Winner</div>', unsafe_allow_html=True)
        _w = _battle_df.iloc[0]
        _w_sym    = _w.get("Symbol", "—")
        _w_score  = _w.get("Final Score", 0)
        _w_conf   = _w.get("Confidence", 50)
        _w_signal = _w.get("Signal", _w.get("Final Signal", "—"))
        _w_setup  = _w.get("Setup Type", _w.get("Volume Trend", "—"))
        _w_bat    = _w.get("Battle Score", 0)
        _w_prob   = _w.get("Battle Probability", _w_bat)
        _w_bconf  = _w.get("Battle Confidence", _w_conf)
        _w_bqual  = _w.get("Battle Quality", _w_score)
        _w_verdict = _w.get("Battle Verdict", "BETTER PICK")
        _w_edge   = _w.get("Battle Edge", 0)
        _w_notes  = _w.get("Battle Notes", "")
        _w_grade  = _w.get("Grade", "—")
        _wc       = "#00d4a8" if _w_bat >= 65 else ("#f0b429" if _w_bat >= 45 else "#ff4d6d")
        st.markdown(
            f'<div style="background:#0b1017;border:2px solid {_wc};border-radius:16px;padding:24px 28px;">'
            f'<div style="display:flex;align-items:center;gap:20px;flex-wrap:wrap;">'
            f'<div style="font-size:42px;">🥇</div>'
            f'<div>'
            f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;color:#ccd9e8;">{_w_sym}</div>'
            f'<div style="font-size:12px;color:#4a6480;margin-top:4px;">Battle Winner · Grade: <b style="color:{_wc}">{_w_grade}</b></div>'
            f'</div>'
            f'<div style="margin-left:auto;text-align:right;">'
            f'<div style="font-size:32px;font-weight:800;color:{_wc};">{_w_bat:.1f}</div>'
            f'<div style="font-size:11px;color:#4a6480;">Battle Score</div>'
            f'</div></div>'
            f'<div style="display:flex;gap:32px;margin-top:18px;flex-wrap:wrap;">'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Final Score</div>'
            f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_score:.1f}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Battle Probability</div>'
            f'<div style="font-size:18px;font-weight:700;color:{_wc};">{_w_prob:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Confidence</div>'
            f'<div style="font-size:18px;font-weight:700;color:#0094ff;">{_w_conf:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Compare Confidence</div>'
            f'<div style="font-size:18px;font-weight:700;color:#7fd1ff;">{_w_bconf:.0f}%</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Battle Quality</div>'
            f'<div style="font-size:18px;font-weight:700;color:#8cf08c;">{_w_bqual:.1f}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Signal</div>'
            f'<div style="font-size:18px;font-weight:700;color:#f0b429;">{_w_signal}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Setup Type</div>'
            f'<div style="font-size:18px;font-weight:700;color:#b08cff;">{_w_setup}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Verdict</div>'
            f'<div style="font-size:18px;font-weight:700;color:{_wc};">{_w_verdict}</div></div>'
            f'<div><div style="font-size:10px;color:#4a6480;text-transform:uppercase;letter-spacing:1px;">Lead Margin</div>'
            f'<div style="font-size:18px;font-weight:700;color:#ccd9e8;">{_w_edge:.1f}</div></div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )
        if _w_notes:
            st.caption(f"Winner notes: {_w_notes}")

        # ── 📊 Comparison Table ───────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-lbl">📊 Head-to-Head Comparison</div>', unsafe_allow_html=True)
        _table_rows = []
        for _, _br in _battle_df.iterrows():
            _trap_r = str(_br.get("Trap Risk", "")).strip()
            _trap_w = str(_br.get("Trap", "")).strip()
            _trap_flag = "⚠️ Potential Bull Trap" if (_trap_r == "HIGH" or "Bull Trap" in _trap_w) else "✅ Clean"
            _table_rows.append({
                "Rank":         int(_br.get("Battle Rank", 0)),
                "Stock":        _br.get("Symbol", "—"),
                "Verdict":      _br.get("Battle Verdict", "WATCHLIST"),
                "Battle Score": round(float(_br.get("Battle Score", 0)), 1),
                "Probability %": round(float(_br.get("Battle Probability", _br.get("Battle Score", 0))), 1),
                "Compare Conf %": round(float(_br.get("Battle Confidence", _br.get("Confidence", 50))), 1),
                "Quality":      round(float(_br.get("Battle Quality", _br.get("Final Score", 0))), 1),
                "Signal":       _br.get("Signal", _br.get("Final Signal", "—")),
                "Grade":        _br.get("Grade", "—"),
                "Risk Score":   round(float(_br.get("Risk Score", 50)), 1),
                "Edge":         round(float(_br.get("Battle Edge", 0)), 1),
                "⚠️ Trap Check": _trap_flag,
                "Notes":        _br.get("Battle Notes", ""),
            })
        st.dataframe(
            pd.DataFrame(_table_rows),
            column_config={
                "Rank":         st.column_config.NumberColumn("Rank", format="%d"),
                "Stock":        st.column_config.TextColumn("Stock"),
                "Verdict":      st.column_config.TextColumn("Verdict"),
                "Battle Score": st.column_config.NumberColumn("Battle Score", format="%.1f"),
                "Probability %": st.column_config.NumberColumn("Probability %", format="%.1f%%"),
                "Compare Conf %": st.column_config.NumberColumn("Compare Conf %", format="%.1f%%"),
                "Quality":      st.column_config.NumberColumn("Quality", format="%.1f"),
                "Signal":       st.column_config.TextColumn("Signal"),
                "Grade":        st.column_config.TextColumn("Grade"),
                "Risk Score":   st.column_config.NumberColumn("Risk Score", format="%.1f"),
                "Edge":         st.column_config.NumberColumn("Edge", format="%.1f"),
                "⚠️ Trap Check": st.column_config.TextColumn("⚠️ Trap Check"),
                "Notes":        st.column_config.TextColumn("Notes", width="large"),
            },
            use_container_width=True,
            hide_index=True,
        )
        with st.expander("🧾 Full Battle Diagnostics", expanded=False):
            st.dataframe(_battle_df, use_container_width=True, hide_index=True)

        # ── ⚠️ Trap Warnings ─────────────────────────────
        _trap_stocks = [
            str(_r.get("Symbol", "?"))
            for _, _r in _battle_df.iterrows()
            if (str(_r.get("Trap Risk", "")).strip() == "HIGH"
                or "Bull Trap" in str(_r.get("Trap", "")))
        ]
        if _trap_stocks:
            st.warning(
                f"⚠️ **Potential Bull Trap** detected in: {', '.join(_trap_stocks)}  —  "
                "RSI overbought and/or volume declining. Proceed with caution."
            )

# ── 🔮 Stock Aura Panel ───────────────────────────────────────────────
if _STOCK_AURA_OK:
    # Pass TT date into session_state so Aura's _fetch_data() truncates correctly
    _aura_tt_date = st.session_state.get("tt_date_val")
    if _aura_tt_date is not None and _TIME_TRAVEL_OK:
        st.session_state["aura_tt_date"] = _aura_tt_date
    else:
        st.session_state["aura_tt_date"] = None
    render_stock_aura_panel()
elif st.session_state.get("aura_show_panel", False):
    st.warning("⚠️ app_stock_aura_section.py not found — place it next to app.py and restart.")

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-family:\'Space Mono\',monospace;'
    'font-size:11px;color:#2a3f58;padding:8px 0 16px;">'
    'NSE SENTINEL · Python + Streamlit + yFinance &nbsp;|&nbsp; '
    'For educational purposes only — not financial advice</div>',
    unsafe_allow_html=True)
