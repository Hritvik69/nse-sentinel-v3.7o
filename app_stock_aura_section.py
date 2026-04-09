"""
app_stock_aura_section.py
──────────────────────────
🧠 STOCK AURA — Single Stock Decision Engine for NSE Sentinel

A trader's brain, not a screener.
Renders a full verdict card when triggered from the sidebar.

HOW TO INTEGRATE INTO app.py
─────────────────────────────
1. Near the top of app.py, add:
       from app_stock_aura_section import render_stock_aura_panel

2. Inside `with st.sidebar:`, after the csv_scan_clicked button, add:
       aura_clicked = st.button("🔮 Stock Aura", key="stock_aura_btn")
       if aura_clicked:
           st.session_state["aura_show_panel"] = True

3. In the main body (anywhere after the battle/csv sections), add:
       render_stock_aura_panel()
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

# ── Internal pipeline helpers (already in app.py's scope) ─────────────
try:
    from strategy_engines._engine_utils import ema as _ema, rsi_vec as _rsi_vec
    _UTILS_OK = True
except ImportError:
    _UTILS_OK = False

try:
    from strategy_engines import get_df_for_ticker as _get_df
    _GETDF_OK = True
except ImportError:
    _GETDF_OK = False

try:
    import yfinance as yf
    _YF_OK = True
except ImportError:
    _YF_OK = False


# ══════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════

def _sf(v: object, default: float = 0.0) -> float:
    """safe float — never raises."""
    try:
        f = float(v)  # type: ignore[arg-type]
        return f if np.isfinite(f) else default
    except Exception:
        return default


def _ema_series(s: pd.Series, n: int) -> pd.Series:
    if _UTILS_OK:
        return _ema(s, n)
    return s.ewm(span=n, adjust=False).mean()


def _rsi_last(close: pd.Series, period: int = 14) -> float:
    try:
        if _UTILS_OK:
            return float(_rsi_vec(close).iloc[-1])
        d = close.diff()
        g = d.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
        l = (-d.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
        rs = g / l.replace(0, np.nan)
        return float((100 - 100 / (1 + rs)).iloc[-1])
    except Exception:
        return 50.0


def _fetch_data(symbol: str) -> pd.DataFrame | None:
    """Fetch OHLCV for symbol; respects Time-Travel cutoff date if active."""
    ticker_ns = symbol.upper().strip()
    if not ticker_ns.endswith(".NS"):
        ticker_ns += ".NS"

    # Determine cutoff date from session_state (set by app.py TT logic)
    cutoff_date = None
    try:
        _tt_raw = st.session_state.get("aura_tt_date")
        if _tt_raw is not None:
            from datetime import date as _date_cls
            if isinstance(_tt_raw, _date_cls):
                cutoff_date = _tt_raw
    except Exception:
        cutoff_date = None

    def _truncate(df: pd.DataFrame | None) -> pd.DataFrame | None:
        if df is None or df.empty or cutoff_date is None:
            return df
        try:
            idx_dates = pd.to_datetime(df.index).date
            mask = idx_dates <= cutoff_date
            trimmed = df.loc[mask]
            return trimmed if len(trimmed) >= 10 else None
        except Exception:
            return df

    # 1️⃣ Prefer preloaded ALL_DATA (already truncated by time_travel_engine)
    if _GETDF_OK:
        try:
            df = _get_df(ticker_ns)
            if df is not None and len(df) >= 10:
                # If TT is active but engine hasn't patched ALL_DATA yet, truncate here too
                return _truncate(df) if cutoff_date else df
        except Exception:
            pass

    # 2️⃣ Live yfinance fallback — always truncate if TT active
    if _YF_OK:
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
            if len(df) < 10:
                return None
            return _truncate(df)
        except Exception:
            return None
    return None


# ══════════════════════════════════════════════════════════════════════
# CORE AURA ENGINE
# ══════════════════════════════════════════════════════════════════════

class AuraResult:
    """Holds all computed data and the final verdict."""

    # Verdict constants
    BUY_NOW    = "🔥 BUY RIGHT NOW"
    BUY_CONF   = "✅ BUY (ON CONFIRMATION)"
    WATCH      = "👀 WATCH"
    AVOID      = "❌ AVOID"

    # Timing constants
    T_BUY_NOW  = "BUY NOW"
    T_TOMORROW = "BUY TOMORROW"
    T_WAIT     = "WAIT"
    T_NO_TRADE = "NO TRADE"

    def __init__(self) -> None:
        self.symbol       = ""
        self.price        = 0.0
        self.rsi          = 50.0
        self.ema20        = 0.0
        self.ema50        = 0.0
        self.vol_ratio    = 1.0
        self.delta_ema20  = 0.0   # % price above EMA20
        self.delta_20h    = 0.0   # % from 20D high (negative = below)
        self.ret_5d       = 0.0
        self.ret_20d      = 0.0
        self.rr_ratio     = 0.0   # risk-reward ratio

        # Verdict
        self.verdict      = self.AVOID
        self.timing       = self.T_NO_TRADE
        self.verdict_color = "#ff4d6d"

        # Reason bullets
        self.reasons_positive: list[str] = []
        self.reasons_warning:  list[str] = []
        self.reasons_reject:   list[str] = []

        # Factor grades (True=pass, False=fail, None=partial)
        self.trend_ok     = False
        self.setup_type   = "None"   # "Breakout" | "Pullback" | "None"
        self.volume_ok    = False
        self.momentum_ok  = False
        self.entry_ok     = False
        self.sl_quality   = "Poor"   # "Tight" | "Medium" | "Poor"
        self.rr_ok        = False
        self.market_note  = ""


def _run_aura_engine(df: pd.DataFrame, symbol: str, market_bias: dict | None) -> AuraResult:
    """
    Core decision engine — converts OHLCV into a structured AuraResult.
    Never raises; returns a default AVOID result on any error.
    """
    result = AuraResult()
    result.symbol = symbol.upper().replace(".NS", "")

    try:
        close  = df["Close"].dropna()
        volume = df["Volume"].dropna()
        high_s = df.get("High", close).dropna() if "High" in df.columns else close
        low_s  = df.get("Low",  close).dropna() if "Low"  in df.columns else close

        if len(close) < 30:
            result.reasons_reject.append("Insufficient price history")
            return result

        # ── Raw values ─────────────────────────────────────────────────
        lc       = _sf(close.iloc[-1])
        e20      = _sf(_ema_series(close, 20).iloc[-1])
        e50      = _sf(_ema_series(close, 50).iloc[-1])
        prev_e20 = _sf(_ema_series(close, 20).iloc[-2]) if len(close) >= 2 else e20
        rsi_val  = _rsi_last(close)

        avg_vol  = _sf(volume.iloc[-21:-1].mean()) if len(volume) >= 21 else _sf(volume.mean())
        lv       = _sf(volume.iloc[-1])
        vol_r    = round(lv / avg_vol, 2) if avg_vol > 0 else 1.0

        h20   = _sf(close.iloc[-21:-1].max()) if len(close) >= 21 else _sf(close.max())
        h10   = _sf(close.iloc[-11:-1].max()) if len(close) >= 11 else h20
        l20_s = _sf(low_s.iloc[-21:-1].min()) if len(low_s) >= 21 else _sf(low_s.min())

        ret_5d  = (_sf(close.iloc[-1]) / _sf(close.iloc[-6])  - 1.0) * 100 if len(close) >= 6  else 0.0
        ret_20d = (_sf(close.iloc[-1]) / _sf(close.iloc[-21]) - 1.0) * 100 if len(close) >= 21 else 0.0
        delta_ema20 = (lc / e20 - 1.0) * 100 if e20 > 0 else 0.0
        delta_20h   = (lc / h20 - 1.0) * 100  if h20 > 0 else 0.0

        # Store for display
        result.price       = round(lc, 2)
        result.rsi         = round(rsi_val, 1)
        result.ema20       = round(e20, 2)
        result.ema50       = round(e50, 2)
        result.vol_ratio   = round(vol_r, 2)
        result.delta_ema20 = round(delta_ema20, 2)
        result.delta_20h   = round(delta_20h, 2)
        result.ret_5d      = round(ret_5d, 2)
        result.ret_20d     = round(ret_20d, 2)

        # ── FACTOR 1 — TREND ───────────────────────────────────────────
        price_above_e20 = lc > e20 > 0
        ema_stack       = e20 > e50 > 0
        ema_slope_up    = e20 > prev_e20  # EMA20 rising

        if price_above_e20 and ema_stack:
            result.trend_ok = True
            result.reasons_positive.append("Strong uptrend (Price > EMA20 > EMA50)")
        elif price_above_e20 and not ema_stack:
            result.trend_ok = False
            result.reasons_warning.append("Price above EMA20 but EMA20 < EMA50 — weak structure")
        else:
            result.trend_ok = False
            result.reasons_reject.append("Downtrend — price below EMA20")

        if ema_slope_up and result.trend_ok:
            result.reasons_positive.append("EMA20 slope rising — momentum intact")

        # ── FACTOR 2 — SETUP ───────────────────────────────────────────
        breakout_vol_ok  = vol_r >= 1.5
        at_breakout_zone = -1.5 <= delta_20h <= 0.5   # within 1.5% of 20D high
        pullback_zone    = -6.0 <= delta_20h <= -1.5   # 1.5–6% below high (normal pullback)
        vol_normalizing  = 0.8 <= vol_r <= 1.5         # volume calming on pullback

        if at_breakout_zone and breakout_vol_ok:
            result.setup_type = "Breakout"
            result.reasons_positive.append("Breakout setup — price at 20D high with volume surge")
        elif at_breakout_zone and not breakout_vol_ok:
            result.setup_type = "Pullback"
            result.reasons_warning.append("Near 20D high but volume not confirming — wait for vol")
        elif pullback_zone and ema_slope_up:
            result.setup_type = "Pullback"
            result.reasons_positive.append("Healthy pullback to EMA support — potential re-entry")
        elif delta_20h < -6.0:
            result.setup_type = "None"
            result.reasons_reject.append(f"Too far from 20D high ({delta_20h:.1f}%) — no valid entry")
        else:
            result.setup_type = "Pullback"
            result.reasons_warning.append("Setup not fully formed — borderline zone")

        # ── FACTOR 3 — VOLUME ──────────────────────────────────────────
        if vol_r >= 1.5:
            result.volume_ok = True
            result.reasons_positive.append(f"Volume strong ({vol_r:.1f}× avg) — institutional participation")
        elif vol_r >= 1.3:
            result.volume_ok = True
            result.reasons_positive.append(f"Volume valid ({vol_r:.1f}× avg) — acceptable participation")
        elif vol_r >= 1.0:
            result.volume_ok = False
            result.reasons_warning.append(f"Volume weak ({vol_r:.1f}× avg) — no conviction signal")
        else:
            result.volume_ok = False
            result.reasons_reject.append(f"Volume below average ({vol_r:.1f}×) — distribution risk")

        # ── FACTOR 4 — MOMENTUM (RSI + 5D return) ─────────────────────
        rsi_overbought  = rsi_val > 75
        rsi_healthy     = 50.0 <= rsi_val <= 70.0
        rsi_caution     = 70.0 < rsi_val <= 75.0
        exhaustion_5d   = ret_5d > 12.0
        strong_5d       = 2.0 <= ret_5d <= 10.0

        if rsi_overbought:
            result.momentum_ok = False
            result.reasons_reject.append(f"RSI overbought ({rsi_val:.0f}) — late-stage entry risk")
        elif exhaustion_5d:
            result.momentum_ok = False
            result.reasons_reject.append(f"5D return {ret_5d:.1f}% — possible short-term exhaustion")
        elif rsi_healthy and strong_5d:
            result.momentum_ok = True
            result.reasons_positive.append(f"RSI in healthy zone ({rsi_val:.0f}) with strong 5D return ({ret_5d:.1f}%)")
        elif rsi_healthy:
            result.momentum_ok = True
            result.reasons_positive.append(f"RSI healthy ({rsi_val:.0f}) — momentum not stretched")
        elif rsi_caution:
            result.momentum_ok = False
            result.reasons_warning.append(f"RSI elevated ({rsi_val:.0f}) — caution zone, risk of reversal")
        else:
            result.momentum_ok = True
            result.reasons_warning.append(f"RSI low ({rsi_val:.0f}) — accumulation zone, early stage")

        # ── FACTOR 5 — ENTRY QUALITY (EMA distance) ───────────────────
        if delta_ema20 <= 3.0:
            result.entry_ok = True
            result.reasons_positive.append(f"Close to EMA20 ({delta_ema20:.1f}%) — tight entry quality")
        elif delta_ema20 <= 6.0:
            result.entry_ok = True
            result.reasons_warning.append(f"Moderately extended from EMA20 ({delta_ema20:.1f}%) — acceptable")
        else:
            result.entry_ok = False
            result.reasons_reject.append(f"Overextended from EMA20 ({delta_ema20:.1f}%) — late entry, high risk")

        # ── FACTOR 6 — STOP-LOSS QUALITY ──────────────────────────────
        sl_dist = delta_ema20  # EMA20 is the natural stop
        if sl_dist <= 3.0:
            result.sl_quality = "Tight"
            result.reasons_positive.append(f"Tight stop (EMA20 is {sl_dist:.1f}% below) — good risk control")
        elif sl_dist <= 6.0:
            result.sl_quality = "Medium"
            result.reasons_warning.append(f"Medium stop distance ({sl_dist:.1f}% to EMA20) — manageable")
        else:
            result.sl_quality = "Poor"
            result.reasons_reject.append(f"Wide stop required ({sl_dist:.1f}% to EMA20) — poor structure")

        # ── FACTOR 7 — RISK-REWARD ─────────────────────────────────────
        # Downside stop = EMA20 (always)
        # Upside target:
        #   Breakout (price at/above 20D high) → next extension = h20 × 1.06
        #   Pullback (price below 20D high)    → prior high (h20)
        downside  = max(lc - e20, 0.01) if lc > e20 > 0 else 0.01
        if delta_20h >= -1.5:
            # Breakout scenario: price is at the high, project 6% continuation
            target = lc * 1.06
        else:
            # Pullback scenario: prior 20D high is the target
            target = h20
        upside = max(target - lc, 0.0)
        rr     = upside / downside if downside > 0 else 0.0
        result.rr_ratio = round(rr, 2)

        # For pullback entries: target = prior high, stop = EMA20
        if rr >= 2.0:
            result.rr_ok = True
            result.reasons_positive.append(f"Risk-reward {rr:.1f}:1 — excellent setup")
        elif rr >= 1.5:
            result.rr_ok = True
            result.reasons_positive.append(f"Risk-reward {rr:.1f}:1 — acceptable trade")
        elif rr >= 1.0:
            result.rr_ok = False
            result.reasons_warning.append(f"Risk-reward only {rr:.1f}:1 — marginal, prefer ≥2:1")
        else:
            result.rr_ok = False
            result.reasons_reject.append(f"Risk-reward {rr:.1f}:1 — unfavorable, skip trade")

        # ── FACTOR 8 — MARKET + SECTOR CONTEXT ────────────────────────
        mb = market_bias if isinstance(market_bias, dict) else {}
        bias_str   = str(mb.get("bias",   "")).strip()
        regime_str = str(mb.get("regime", "")).strip()
        if bias_str or regime_str:
            label = bias_str or regime_str
            if any(w in label.lower() for w in ("bearish", "weak", "caution")):
                result.market_note = f"Market context: {label} — reduces conviction"
                result.reasons_warning.append(f"Market is {label} — trade with extra caution")
            elif any(w in label.lower() for w in ("bullish", "trending up", "strong")):
                result.market_note = f"Market: {label} ✅"
                result.reasons_positive.append(f"Favorable market backdrop ({label})")
            else:
                result.market_note = f"Market: {label}"
        else:
            result.market_note = "Market context unavailable — run Market Bias scan first"

        # ══ VERDICT ENGINE ════════════════════════════════════════════
        #
        # BUY RIGHT NOW  = breakout setup + strong vol + trend + RSI ok + rr ok + tight entry
        # BUY CONFIRM    = pullback or near-breakout + trend + RSI ok + at least rr ok
        # WATCH          = 1-2 factors weak, trend mostly OK
        # AVOID          = trend broken OR no setup OR overbought OR rr bad + entry bad
        #
        reject_count  = len(result.reasons_reject)
        warning_count = len(result.reasons_warning)

        is_breakout_confirmed = (
            result.setup_type == "Breakout" and
            result.trend_ok and
            result.volume_ok and
            result.momentum_ok and
            result.entry_ok and
            result.rr_ok
        )

        is_buy_on_confirm = (
            result.setup_type in ("Breakout", "Pullback") and
            result.trend_ok and
            result.momentum_ok and
            (result.rr_ok or rr >= 1.0) and
            reject_count == 0
        )

        is_watch = (
            result.trend_ok and
            result.setup_type != "None" and
            reject_count <= 1 and
            not rsi_overbought
        )

        # TIMING logic
        if is_breakout_confirmed:
            result.verdict       = AuraResult.BUY_NOW
            result.timing        = AuraResult.T_BUY_NOW
            result.verdict_color = "#00d4a8"
        elif is_buy_on_confirm:
            result.verdict       = AuraResult.BUY_CONF
            result.timing        = AuraResult.T_TOMORROW
            result.verdict_color = "#0094ff"
        elif is_watch:
            result.verdict       = AuraResult.WATCH
            result.timing        = AuraResult.T_WAIT
            result.verdict_color = "#f0b429"
        else:
            result.verdict       = AuraResult.AVOID
            result.timing        = AuraResult.T_NO_TRADE
            result.verdict_color = "#ff4d6d"

        # Override timing if near-breakout but vol weak
        if result.verdict == AuraResult.BUY_CONF and at_breakout_zone and not breakout_vol_ok:
            result.timing = AuraResult.T_TOMORROW

        # Override to WAIT if warnings dominant
        if result.verdict == AuraResult.WATCH and warning_count >= 3:
            result.timing = AuraResult.T_WAIT

        return result

    except Exception as exc:
        result.reasons_reject.append(f"Engine error: {exc}")
        return result


# ══════════════════════════════════════════════════════════════════════
# UI RENDERER
# ══════════════════════════════════════════════════════════════════════

def _timing_badge(timing: str) -> str:
    color_map = {
        "BUY NOW":       "#00d4a8",
        "BUY TOMORROW":  "#0094ff",
        "WAIT":          "#f0b429",
        "NO TRADE":      "#ff4d6d",
    }
    c = color_map.get(timing, "#4a6480")
    return (
        f'<span style="background:{c}20;border:1px solid {c};border-radius:6px;'
        f'padding:3px 10px;font-size:12px;font-weight:700;color:{c};">{timing}</span>'
    )


def _factor_row(label: str, value: str, color: str) -> str:
    return (
        f'<div style="display:flex;justify-content:space-between;align-items:center;'
        f'padding:6px 0;border-bottom:1px solid #1a2840;">'
        f'<span style="font-size:11px;color:#4a6480;">{label}</span>'
        f'<span style="font-size:12px;font-weight:700;color:{color};">{value}</span></div>'
    )


def _render_aura_card(r: AuraResult) -> None:
    """Render the full Aura verdict card using Streamlit HTML components."""

    # ── Verdict header ────────────────────────────────────────────────
    st.markdown(
        f'<div style="background:#0b1017;border:2px solid {r.verdict_color};border-radius:14px;'
        f'padding:20px 24px;margin:12px 0 20px;">'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:13px;color:#4a6480;'
        f'letter-spacing:1px;text-transform:uppercase;margin-bottom:4px;">🔮 STOCK AURA RESULT</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:26px;font-weight:800;'
        f'color:#ccd9e8;margin-bottom:2px;">{r.symbol}</div>'
        f'<div style="font-size:11px;color:#4a6480;margin-bottom:14px;">'
        f'₹{r.price:.2f} &nbsp;|&nbsp; RSI {r.rsi:.0f} &nbsp;|&nbsp; '
        f'Vol {r.vol_ratio:.1f}× &nbsp;|&nbsp; EMA20 {r.delta_ema20:+.1f}% &nbsp;|&nbsp; '
        f'5D {r.ret_5d:+.1f}%</div>'
        f'<div style="font-family:\'Syne\',sans-serif;font-size:22px;font-weight:900;'
        f'color:{r.verdict_color};margin-bottom:10px;">{r.verdict}</div>'
        f'<div>Timing: {_timing_badge(r.timing)}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([3, 2])

    # ── Why / Warnings ────────────────────────────────────────────────
    with col_left:
        if r.reasons_positive:
            positives_html = "".join(
                f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                f'<span style="color:#00d4a8;font-weight:700;">✔</span> &nbsp;{reason}</div>'
                for reason in r.reasons_positive
            )
            st.markdown(
                f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;'
                f'padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-size:11px;font-weight:700;color:#00d4a8;'
                f'letter-spacing:0.5px;margin-bottom:8px;">WHY ✔</div>'
                f'{positives_html}</div>',
                unsafe_allow_html=True,
            )

        all_issues = [
            (w, "#f0b429") for w in r.reasons_warning
        ] + [
            (e, "#ff4d6d") for e in r.reasons_reject
        ]
        if all_issues:
            issues_html = "".join(
                f'<div style="padding:5px 0;font-size:12px;color:#ccd9e8;">'
                f'<span style="color:{c};font-weight:700;">✖</span> &nbsp;{txt}</div>'
                for txt, c in all_issues
            )
            st.markdown(
                f'<div style="background:#0f1923;border:1px solid #3a1e1e;border-radius:10px;'
                f'padding:14px 16px;margin-bottom:12px;">'
                f'<div style="font-size:11px;font-weight:700;color:#ff4d6d;'
                f'letter-spacing:0.5px;margin-bottom:8px;">WARNINGS / REJECTIONS ✖</div>'
                f'{issues_html}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;'
                'padding:14px 16px;margin-bottom:12px;font-size:12px;color:#00d4a8;">'
                'Warnings: None ✔</div>',
                unsafe_allow_html=True,
            )

    # ── Factor scorecard ──────────────────────────────────────────────
    with col_right:
        def _grade(ok: bool | None, label_t: str = "PASS", label_f: str = "FAIL") -> tuple[str, str]:
            if ok is True:
                return label_t, "#00d4a8"
            if ok is False:
                return label_f, "#ff4d6d"
            return "PARTIAL", "#f0b429"

        trend_lbl, trend_c   = _grade(r.trend_ok, "ALIGNED", "WEAK")
        setup_c              = "#00d4a8" if r.setup_type != "None" else "#ff4d6d"
        vol_lbl, vol_c       = _grade(r.volume_ok, "STRONG", "WEAK")
        mom_lbl, mom_c       = _grade(r.momentum_ok, "HEALTHY", "STRETCHED")
        entry_lbl, entry_c   = _grade(r.entry_ok, "GOOD", "EXTENDED")
        sl_c                 = {"Tight":"#00d4a8","Medium":"#f0b429","Poor":"#ff4d6d"}.get(r.sl_quality, "#4a6480")
        rr_lbl, rr_c         = _grade(r.rr_ok, f"{r.rr_ratio:.1f}:1 ✔", f"{r.rr_ratio:.1f}:1 ✖")

        factors_html = (
            _factor_row("Trend",         trend_lbl,         trend_c)  +
            _factor_row("Setup",         r.setup_type,      setup_c)  +
            _factor_row("Volume",        f"{r.vol_ratio:.1f}× — {vol_lbl}", vol_c)  +
            _factor_row("Momentum RSI",  f"{r.rsi:.0f} — {mom_lbl}", mom_c) +
            _factor_row("Entry Quality", f"{r.delta_ema20:+.1f}% — {entry_lbl}", entry_c) +
            _factor_row("Stop Quality",  r.sl_quality,      sl_c)     +
            _factor_row("Risk-Reward",   rr_lbl,            rr_c)
        )
        st.markdown(
            f'<div style="background:#0f1923;border:1px solid #1e3a5f;border-radius:10px;'
            f'padding:14px 16px;margin-bottom:12px;">'
            f'<div style="font-size:11px;font-weight:700;color:#8ab4d8;'
            f'letter-spacing:0.5px;margin-bottom:8px;">FACTOR SCORECARD</div>'
            f'{factors_html}</div>',
            unsafe_allow_html=True,
        )

        # Market note
        if r.market_note:
            note_c = "#f0b429" if "caution" in r.market_note.lower() else "#4a6480"
            st.markdown(
                f'<div style="font-size:11px;color:{note_c};padding:4px 0;">'
                f'🌐 {r.market_note}</div>',
                unsafe_allow_html=True,
            )


# ══════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT — called from app.py
# ══════════════════════════════════════════════════════════════════════

def render_stock_aura_panel() -> None:
    """
    Render the full Stock Aura panel in the main app body.
    Call this unconditionally — it guards itself via session_state.
    """
    if not st.session_state.get("aura_show_panel", False):
        return

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown(
        '<h2 style="font-family:\'Syne\',sans-serif;font-weight:900;font-size:22px;'
        'color:#ccd9e8;margin-bottom:4px;">🔮 Stock Aura</h2>'
        '<div style="font-size:12px;color:#4a6480;margin-bottom:18px;">'
        'Single stock decision engine — a trader\'s brain, not a screener</div>',
        unsafe_allow_html=True,
    )

    # ── 🕰️ Time-travel banner inside Aura ────────────────────────────
    _aura_tt = st.session_state.get("aura_tt_date")
    if _aura_tt is not None:
        try:
            _aura_tt_str = _aura_tt.strftime("%d %b %Y")
        except Exception:
            _aura_tt_str = str(_aura_tt)
        st.markdown(
            f'<div style="background:#1a0a00;border:1.5px solid #f0b429;border-radius:8px;'
            f'padding:8px 14px;margin-bottom:14px;font-size:12px;color:#f0b429;">'
            f'🕰️ <b>TIME TRAVEL ACTIVE</b> — Evaluating {_aura_tt_str} post-market close · '
            f'No future data used</div>',
            unsafe_allow_html=True,
        )

    # ── Input row ─────────────────────────────────────────────────────
    col_inp, col_btn, col_close = st.columns([3, 1, 1])
    with col_inp:
        ticker_raw = st.text_input(
            "Enter Stock Symbol",
            placeholder="e.g.  RELIANCE  or  TCS  or  HDFCBANK",
            key="aura_ticker_input",
            label_visibility="collapsed",
        )
    with col_btn:
        analyze_clicked = st.button("🧠 Analyze Aura", key="aura_analyze_btn")
    with col_close:
        if st.button("✕ Close", key="aura_close_btn"):
            st.session_state["aura_show_panel"] = False
            st.rerun()

    if analyze_clicked and ticker_raw.strip():
        sym = ticker_raw.strip().upper().replace(".NS", "")
        _spinner_msg = (
            f"🕰️ Reading historical aura for {sym} ({_aura_tt_str})…"
            if _aura_tt is not None else f"🔮 Reading aura for {sym}…"
        )
        with st.spinner(_spinner_msg):
            df = _fetch_data(sym)

        if df is None or df.empty:
            st.error(
                f"❌ Could not fetch data for **{sym}**.  "
                "Check the symbol (e.g. RELIANCE, not RELIANCE.NS) and try again."
            )
            return

        # Pull market bias from session state if available
        mb = st.session_state.get("market_bias_result", None)

        # In TT mode, annotate market bias so the Aura card shows a meaningful note
        if _aura_tt is not None:
            if mb is None:
                mb = {}
            mb = dict(mb)  # don't mutate the shared cache
            mb["_tt_date"] = str(_aura_tt)
            if not mb.get("bias"):
                mb["bias"] = f"Historical ({_aura_tt_str}) — run Market Bias for that date"

        result = _run_aura_engine(df, sym, mb)
        _render_aura_card(result)

        # ── Disclaimer ────────────────────────────────────────────────
        st.markdown(
            '<div style="font-size:10px;color:#2a3f58;margin-top:12px;text-align:center;">'
            '⚠️ Stock Aura is for educational purposes only. Not financial advice. '
            'Always do your own research before trading.</div>',
            unsafe_allow_html=True,
        )

    elif analyze_clicked and not ticker_raw.strip():
        st.warning("Enter a stock symbol first (e.g. RELIANCE, TCS, INFY)")