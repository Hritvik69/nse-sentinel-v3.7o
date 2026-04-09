# 🛡️ NSE Sentinel

**Production-grade NSE stock screener and decision engine built with Python + Streamlit.**

NSE Sentinel scans 1000+ NSE-listed equities using six independent strategy modes, layers multiple intelligence engines on top of raw scan results, and converts raw indicator data into clear, actionable trading decisions — all without a paid data subscription.

> ⚠️ **Educational use only. Not financial advice. Always do your own research before trading.**

---

## 📸 Features at a Glance

| Feature | Description |
|---|---|
| 🔍 **6 Strategy Modes** | Momentum, Balanced, Relaxed, Institutional, Intraday, Swing |
| 🧠 **Intelligence Pipeline** | Scoring → Backtest → ML → Grading → Phase Logic |
| 🔮 **Stock Aura** | Single-stock decision engine — verdict, timing, risk-reward |
| ⚔️ **Battle Mode** | Side-by-side comparison of up to 10 stocks |
| 🌐 **Market Bias Engine** | Nifty + BankNifty regime detection |
| 📊 **Sector Intelligence** | 17-sector mapping, sector screener, sector dominance |
| 📂 **CSV Next-Day Engine** | Pre-breakout probability from local cache |
| 📦 **Local Data Cache** | Parallel CSV download, incremental updates, zero API waste |
| 📈 **Prediction Feedback** | Auto-backfill actual returns, accuracy tracking |

---

## 🗂️ Project Structure

```
nse-sentinel/
│
├── app.py                          # Main Streamlit application entry point
│
├── strategy_engines/               # Core strategy engine package
│   ├── __init__.py                 # Engine dispatcher (modes 1–6)
│   ├── _engine_utils.py            # Shared: EMA, RSI, ALL_DATA cache, preload
│   ├── _df_extensions.py           # Zero-API backtest variants (all 6 modes)
│   ├── mode1_engine.py             # Momentum / Breakout
│   ├── mode2_engine.py             # Balanced / Swing
│   ├── mode3_engine.py             # Relaxed / Early Accumulation
│   ├── mode4_engine.py             # Institutional Strength
│   ├── mode5_engine.py             # Intraday
│   └── mode6_engine.py             # Swing (EMA Slope)
│
├── enhanced_logic_engine.py        # Phase 3: Volume Trend, Setup Quality, Entry Timing, Trap Risk
├── grading_engine.py               # Universal grading: Grade, Signal, Conviction Tier
├── phase4_logic_engine.py          # Phase 4: Setup Type, Reason, Risk Score, Final Signal
├── prediction_feedback_store.py    # Append-only prediction log + accuracy tracker
│
├── market_bias_engine.py           # Standalone Nifty/BankNifty regime engine
├── multi_index_market_bias_engine.py # Multi-index + sector prediction layer
│
├── sector_master.py                # Static 17-sector stock mapping (1000+ stocks)
├── sector_intelligence_engine.py   # Dynamic sector dominance + smart filtering
│
├── battle_mode_engine.py           # Multi-stock comparison engine
├── csv_next_day_engine.py          # Pre-breakout potential from local CSV cache
├── data_downloader.py              # Parallel CSV downloader (yfinance → local)
│
├── app_stock_aura_section.py       # 🔮 Stock Aura UI + decision engine
├── app_battle_section.py           # ⚔️ Battle Mode UI section
├── app_sector_screener_section.py  # Sector screener UI section
├── app_sector_screener_dashboard.py# Sector screener full dashboard
├── app_sector_intelligence_section.py # Sector intelligence UI
├── app_sector_explorer_section.py  # Sector explorer / browse UI
│
├── data/                           # Auto-created — local CSV price cache
│   └── *.csv                       # One file per ticker (e.g. RELIANCE.NS.csv)
│
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

---

## ⚙️ Strategy Modes

Each mode has its own independent scoring engine, backtest simulator, ML model, and bull-trap detector.

| Mode | Name | Philosophy | RSI Zone | Volume | ML Target |
|------|------|------------|----------|--------|-----------|
| 1 | 🟢 Momentum | Explosive breakout | 55–72 | > 1.7× | 3-day hold |
| 2 | 🔵 Balanced | Equal weight all signals | 52–70 | > 1.5× | Next day |
| 3 | 🟡 Relaxed | Early accumulation | 50–72 | > 1.3× | Next day |
| 4 | 🟣 Institutional | Relative strength dominant | 55–70 | > 1.5× | 5-day hold |
| 5 | 🟠 Intraday | Vol spike + 5D high | 52–62 | > 1.1× | Next day |
| 6 | 🔴 Swing | Rising EMA slope | 53–59 | > 1.3× | 3-day hold |

---

## 🧠 Intelligence Pipeline

Every scan result passes through this sequential enrichment stack:

```
run_scan()
    └─ Raw OHLCV indicators (RSI, EMA, Vol/Avg, returns)
         │
         ▼
enhance_results()
    └─ Score + Backtest% + ML% + Final Score + Bull Trap + Next-Day Signal
         │
         ▼
apply_enhanced_logic()           [enhanced_logic_engine.py]
    └─ Volume Trend / Setup Quality / Entry Timing / Trap Risk
         │
         ▼
apply_universal_grading()        [grading_engine.py]
    └─ Grade (A+→D) / Signal / Confidence / Conviction Tier / Prediction Score
         │
         ▼
apply_phase4_logic()             [phase4_logic_engine.py]
    └─ Setup Type / Reason / Risk Score / Final Signal
```

---

## 🔮 Stock Aura — Single Stock Decision Engine

Unlike the screener (which finds stocks), **Stock Aura** evaluates a single stock and outputs a verdict a trader can act on immediately.

**Checks performed:**

1. **Trend** — Price > EMA20 > EMA50, EMA slope direction
2. **Setup** — Breakout (at 20D high + volume) or Pullback (healthy retracement)
3. **Volume** — Participation strength vs 20-day average
4. **Momentum** — RSI zone + 5D return exhaustion check
5. **Entry Quality** — Distance from EMA20 (overextension check)
6. **Stop Quality** — EMA20 distance = natural stop width
7. **Risk-Reward** — Breakout: 6% extension target; Pullback: prior high target
8. **Market Context** — Reads live market bias from session state

**Possible verdicts:**

| Verdict | Meaning |
|---|---|
| 🔥 BUY RIGHT NOW | All factors aligned, breakout confirmed with volume |
| ✅ BUY (ON CONFIRMATION) | Setup forming, needs next-session confirmation |
| 👀 WATCH | Partial alignment, not a clean entry yet |
| ❌ AVOID | Trend broken, overbought, bad risk-reward, or no valid setup |

---

## 🚀 Quick Start

### 1. Clone & install

```bash
git clone https://github.com/yourusername/nse-sentinel.git
cd nse-sentinel
python -m venv .venv

# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. (Optional) Pre-download data cache

Significantly speeds up scan time by avoiding live API calls during scan:

```bash
python data_downloader.py
```

This downloads ~6 months of daily OHLCV for 50 Nifty stocks into `/data/`. The app also downloads on first use — this step is optional but recommended.

### 3. Run the app

```bash
# Windows
.\.venv\Scripts\python.exe -m streamlit run app.py

# macOS / Linux
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📦 Data Layer

NSE Sentinel uses a local CSV cache to minimise yfinance API calls:

- `data_downloader.py` saves per-ticker CSVs in `/data/`
- On each scan, `_engine_utils.py` loads from CSV first, falls back to live yfinance
- Incremental updates: only fetches last 5 days if CSV exists and is recent
- Market-aware skip: won't re-download on weekends if Friday data is fresh
- Thread-safe writes with file locking

**Refresh cache manually** from the sidebar → `📦 Local Data Cache` → `🔄 Download / Refresh Data Now`

---

## 🌐 Market Bias Engine

Analyses Nifty 50 (`^NSEI`) and Bank Nifty (`^NSEBANK`) to determine market regime before scoring stocks:

- **Bias:** Bullish / Bearish / Sideways
- **Regime:** Trending Up / Ranging / High Vol / Bearish
- **Confidence:** 0–100 score
- Influences scoring via `apply_universal_grading()` market adjustment

---

## 📊 Sector Coverage

`sector_master.py` maps 1000+ NSE stocks into 17 sectors:

Banking & Finance · Technology · Pharmaceuticals · FMCG · Automobiles · Energy & Oil · Metals & Mining · Infrastructure · Telecom · Real Estate · Chemicals · Consumer Durables · Textiles · Media & Entertainment · Aviation & Logistics · Retail · Diversified

---

## 📈 Prediction Feedback & Accuracy Tracking

Every scan is logged to `data/prediction_feedback_log.csv`. Actual next-day returns are auto-backfilled using `backfill_actual_returns()`. Accuracy metrics (bullish precision, false signal rate) are computed from filled rows.

---

## 🛠️ Dependencies

| Package | Purpose |
|---|---|
| `streamlit` | Web UI framework |
| `yfinance` | NSE price data via Yahoo Finance |
| `pandas` | Data manipulation |
| `numpy` | Numerical computation |
| `ta` | Technical analysis indicators |
| `plotly` | Interactive charts |
| `scikit-learn` | Logistic regression ML models |
| `scipy` | Statistical utilities |
| `requests` | HTTP for NSE ticker list fetch |
| `tqdm` | Progress bars (CLI) |

---

## ⚠️ Disclaimer

NSE Sentinel is built for **educational and research purposes only**.

- It does not constitute financial advice
- Past backtest results do not guarantee future performance
- All trading decisions are solely your own responsibility
- The authors are not liable for any financial losses

Always conduct independent research and consult a qualified financial advisor before making investment decisions.

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for full terms.
