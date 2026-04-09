# ═══════════════════════════════════════════════════════════════════════
# 📊 SECTOR EXPLORER  (sector_master.py UI section)
# ═══════════════════════════════════════════════════════════════════════
# HOW TO ADD:
#   Paste this block into app.py just ABOVE the Market Bias Dashboard
#   section (or anywhere after the main scan section).
#
# ALSO ADD at the top of app.py:
#   from sector_master import (
#       SECTOR_STOCKS, get_sector, get_stocks_in_sector,
#       get_all_sectors, get_sector_count, search_stock,
#       get_sector_peers, get_sector_description,
#       SECTOR_DESCRIPTIONS,
#   )
# ═══════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd

# ── Safe stubs for lint/static analysis ──────────────────────────────
try:
    from sector_master import (
        SECTOR_STOCKS, get_sector, get_stocks_in_sector,
        get_all_sectors, get_sector_count, search_stock,
        get_sector_peers, get_sector_description,
        SECTOR_DESCRIPTIONS,
    )
    _SM_OK = True
except ImportError:
    _SM_OK = False
    SECTOR_STOCKS        = {}
    SECTOR_DESCRIPTIONS  = {}
    def get_all_sectors():                    return []
    def get_sector_count():                   return {}
    def get_stocks_in_sector(s):              return []
    def get_sector(sym):                      return None
    def search_stock(q):                      return []
    def get_sector_peers(sym):                return []
    def get_sector_description(s):            return s

st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    '<h2 style="margin-bottom:4px;">📊 Sector Explorer</h2>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div style="font-size:12px;color:#4a6480;margin-bottom:16px;">'
    'Static curated sector database · 17 NSE sectors · Sector lookup, peers & coverage</div>',
    unsafe_allow_html=True,
)

if not _SM_OK:
    st.warning("⚠️ sector_master.py not found. Place it next to app.py and restart.")
else:
    _se_tab1, _se_tab2, _se_tab3 = st.tabs([
        "🔍 Browse Sectors",
        "🔎 Stock Lookup",
        "📋 Coverage Overview",
    ])

    # ── TAB 1 : Browse Sectors ─────────────────────────────────────────
    with _se_tab1:
        _se_all_sectors   = get_all_sectors()
        _se_sector_counts = get_sector_count()

        # Dropdown with count
        _se_options = [
            f"{s}  ({_se_sector_counts.get(s, 0)} stocks)"
            for s in _se_all_sectors
        ]
        _se_choice = st.selectbox(
            "Select Sector",
            options=_se_options,
            key="se_sector_select",
        )

        if _se_choice:
            # Extract the sector name (strip the count suffix)
            _se_sector_name = _se_choice.split("  (")[0].strip()
            _se_stocks      = get_stocks_in_sector(_se_sector_name)
            _se_desc        = get_sector_description(_se_sector_name)

            st.markdown(
                f'<div style="background:#0b1017;border:1.5px solid #1e3a5f;'
                f'border-radius:12px;padding:14px 18px;margin:8px 0 16px;">'
                f'<span style="font-family:\'Syne\',sans-serif;font-size:16px;'
                f'font-weight:800;color:#ccd9e8;">{_se_sector_name}</span>'
                f'<span style="font-size:11px;color:#4a6480;margin-left:12px;">'
                f'{_se_desc}</span><br>'
                f'<span style="font-size:12px;color:#8ab4d8;margin-top:6px;display:block;">'
                f'📦 {len(_se_stocks)} stocks</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

            if _se_stocks:
                # Show as a grid of pill-style badges
                _se_col_n = 6
                _se_rows  = [
                    _se_stocks[i: i + _se_col_n]
                    for i in range(0, len(_se_stocks), _se_col_n)
                ]
                for _row in _se_rows:
                    _se_cols = st.columns(len(_row))
                    for _ci, (_col, _sym) in enumerate(zip(_se_cols, _row)):
                        _col.markdown(
                            f'<div style="background:#0f1923;border:1px solid #1e3a5f;'
                            f'border-radius:8px;padding:6px 10px;text-align:center;'
                            f'font-size:11px;font-weight:700;color:#ccd9e8;">'
                            f'{_sym}</div>',
                            unsafe_allow_html=True,
                        )

                # Also show as a flat copyable list
                with st.expander("📋 Plain list (copy-paste friendly)", expanded=False):
                    st.code(", ".join(_se_stocks), language=None)

            else:
                st.info("No stocks in this sector.")

    # ── TAB 2 : Stock Lookup ──────────────────────────────────────────
    with _se_tab2:
        _se_lookup_col1, _se_lookup_col2 = st.columns([3, 2])

        with _se_lookup_col1:
            _se_sym_input = st.text_input(
                "Enter stock symbol",
                placeholder="e.g. HDFCBANK or partial: HDFC",
                key="se_symbol_input",
            ).strip().upper()

        with _se_lookup_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            _se_search_btn = st.button("🔍 Find Sector", key="se_search_btn")

        if _se_search_btn and _se_sym_input:
            # Exact match first
            _se_exact = get_sector(_se_sym_input)
            if _se_exact:
                _se_peers = get_sector_peers(_se_sym_input)
                st.success(
                    f"**{_se_sym_input}** → Primary Sector: **{_se_exact}**  "
                    f"({get_sector_description(_se_exact)})"
                )
                if _se_peers:
                    st.markdown(
                        f'<div style="font-size:12px;color:#8ab4d8;margin:8px 0 4px;">'
                        f'🤝 Sector Peers ({len(_se_peers)} stocks):</div>',
                        unsafe_allow_html=True,
                    )
                    st.write(", ".join(_se_peers))
            else:
                # Partial search
                _se_matches = search_stock(_se_sym_input)
                if _se_matches:
                    st.info(
                        f"No exact match for '{_se_sym_input}'. "
                        f"Found {len(_se_matches)} partial match(es):"
                    )
                    _se_match_df = pd.DataFrame(
                        _se_matches, columns=["Symbol", "Sector"]
                    )
                    _se_match_df["Description"] = _se_match_df["Sector"].apply(
                        get_sector_description
                    )
                    st.dataframe(
                        _se_match_df,
                        use_container_width=True,
                        hide_index=True,
                    )
                else:
                    st.warning(
                        f"'{_se_sym_input}' not found in the sector database. "
                        "Check the symbol or add it to sector_master.py."
                    )

    # ── TAB 3 : Coverage Overview ─────────────────────────────────────
    with _se_tab3:
        _se_counts = get_sector_count()
        _se_total  = sum(_se_counts.values())

        st.markdown(
            f'<div style="font-size:13px;color:#8ab4d8;margin-bottom:12px;">'
            f'Total coverage: <b style="color:#00d4a8;">{_se_total} stocks</b> '
            f'across <b style="color:#00d4a8;">{len(_se_counts)} sectors</b></div>',
            unsafe_allow_html=True,
        )

        _se_cov_rows = []
        for _sec in get_all_sectors():
            _se_cov_rows.append({
                "Sector":      _sec,
                "Description": get_sector_description(_sec),
                "Stocks":      _se_counts.get(_sec, 0),
            })

        _se_cov_df = pd.DataFrame(_se_cov_rows)

        st.dataframe(
            _se_cov_df,
            column_config={
                "Sector":      st.column_config.TextColumn("Sector"),
                "Description": st.column_config.TextColumn("Description"),
                "Stocks":      st.column_config.NumberColumn("# Stocks", format="%d"),
            },
            use_container_width=True,
            hide_index=True,
        )