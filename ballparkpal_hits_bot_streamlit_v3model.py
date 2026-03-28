"""
A1PICKS MLB Hit Predictor  —  V4.1
====================================
Ground-up rebuild using the new single Matchups.csv from BallPark Pal.

Scores
------
  🎯  Hit Score    — any base hit prop
  1️⃣  Single Score — single specifically (penalises high XB%/HR%)
  🔥  XB Score     — double or triple
  💣  HR Score     — home run (light K% penalty)

Pitcher Layer
-------------
  Mild game-level multiplier (±5% max) derived from:
    pitcher_hits.csv   → Hit_8Plus_Probability
    pitcher_hrs.csv    → HR_2Plus_Probability
    pitcher_walks.csv  → Walk_3Plus_Probability
  Applied multiplicatively to each score after base calculation.
  Displayed in a dedicated "Pitcher Landscape" panel.

Park Mode
---------
  ON  → blend of with-park and no-park probabilities
  OFF → no-park only (pure player vs pitcher)
  Park Δ column shows exact score impact per player.

League Averages (4-year stable baselines)
-----------------------------------------
  K%   22.8%   BB%  8.6%   HR%  3.15%   AVG  .2445
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor",
    layout="wide",
    page_icon="⚾",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────

CONFIG = {
    'matchups_url':       'https://github.com/a1faded/a1picks-hits-bot/raw/main/Matchups.csv',
    'pitcher_hits_url':   'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv',
    'pitcher_hrs_url':    'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
    'pitcher_walks_url':  'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
    'cache_ttl':          900,
    'hist_min_pa':        10,
    'hist_bonus_max':     3.0,
    'pitcher_hit_neutral': 2.8,
    'pitcher_hr_neutral':  12.0,
    'pitcher_walk_neutral':18.0,
    'pitcher_max_mult':    0.05,
    'league_k_avg':        22.8,
    'league_bb_avg':        8.6,
    'league_hr_avg':        3.15,
    'league_avg':           0.2445,
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
.block-container { padding-top: 1.4rem; }
.card {
    padding: .85rem 1rem; border-radius: 10px; color: white;
    margin: .3rem 0; box-shadow: 0 3px 8px rgba(0,0,0,.2);
}
.card h4 { margin: 0 0 .25rem 0; font-size: .82rem; opacity: .88; }
.card h2 { margin: 0; font-size: 1.45rem; font-weight: 800; line-height:1.1; }
.card small { opacity: .72; font-size: .73rem; }
.card-blue   { background: linear-gradient(135deg,#2980b9,#6dd5fa); }
.card-green  { background: linear-gradient(135deg,#11998e,#38ef7d); }
.card-amber  { background: linear-gradient(135deg,#f7971e,#ffd200); color:#1a1a1a; }
.card-red    { background: linear-gradient(135deg,#c0392b,#e74c3c); }
.card-purple { background: linear-gradient(135deg,#667eea,#764ba2); }
.card-teal   { background: linear-gradient(135deg,#43b89c,#7bd9c8); }
.badge { display:inline-block; padding:.28rem .7rem; border-radius:20px;
         font-size:.78rem; font-weight:700; color:white; }
.badge-green  { background:#27ae60; }
.badge-yellow { background:#f39c12; color:#1a1a1a; }
.badge-red    { background:#e74c3c; }
.park-notice {
    background:#1a1a2e; border-left:4px solid #f7971e;
    padding:.55rem 1rem; border-radius:0 6px 6px 0;
    color:#ffd200; font-size:.83rem; margin:.45rem 0;
}
.pitcher-notice {
    background:#0d1117; border-left:4px solid #38ef7d;
    padding:.55rem 1rem; border-radius:0 6px 6px 0;
    color:#c8ffd4; font-size:.83rem; margin:.45rem 0;
}
.legend {
    background:#0d0d0d; border-radius:8px; padding:.9rem 1.1rem;
    color:#e0e0e0; font-size:.82rem; margin:.9rem 0; line-height:1.85;
}
.grade-ap { background:#1a9641; color:white; padding:2px 7px; border-radius:4px; font-weight:700; }
.grade-a  { background:#a6d96a; color:#111;  padding:2px 7px; border-radius:4px; font-weight:700; }
.grade-b  { background:#ffffbf; color:#111;  padding:2px 7px; border-radius:4px; }
.grade-c  { background:#fdae61; color:#111;  padding:2px 7px; border-radius:4px; }
.grade-d  { background:#d7191c; color:white; padding:2px 7px; border-radius:4px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def card(css: str, icon: str, title: str, value: str, sub: str = "") -> str:
    sub_html = f"<small>{sub}</small>" if sub else ""
    return (f'<div class="card {css}"><h4>{icon} {title}</h4>'
            f'<h2>{value}</h2>{sub_html}</div>')


def normalize_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(1)


def staleness_badge() -> str:
    if 'data_loaded_at' not in st.session_state:
        return ""
    mins = int((datetime.now() - st.session_state.data_loaded_at).total_seconds() / 60)
    if mins < 5:
        return f'<span class="badge badge-green">🟢 Fresh ({mins}m ago)</span>'
    elif mins < 12:
        return f'<span class="badge badge-yellow">🟡 {mins}m old</span>'
    return f'<span class="badge badge-red">🔴 {mins}m old — refresh</span>'


def grade_span(grade: str) -> str:
    css = {'A+': 'grade-ap', 'A': 'grade-a', 'B': 'grade-b',
           'C': 'grade-c',   'D': 'grade-d'}.get(grade, 'grade-b')
    return f'<span class="{css}">{grade}</span>'


def style_grade_cell(val):
    return {
        'A+': 'background-color:#1a9641;color:white;font-weight:700',
        'A':  'background-color:#a6d96a;color:#111;font-weight:700',
        'B':  'background-color:#ffffbf;color:#111',
        'C':  'background-color:#fdae61;color:#111',
        'D':  'background-color:#d7191c;color:white;font-weight:700',
    }.get(str(val), '')


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=CONFIG['cache_ttl'])
def _fetch_csv(url: str, label: str):
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        return pd.read_csv(StringIO(r.text))
    except Exception as e:
        st.warning(f"⚠️ Could not load {label}: {e}")
        return None


@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_matchups():
    df = _fetch_csv(CONFIG['matchups_url'], "Matchups")
    if df is None:
        return None
    required = ['Game','Team','Batter','Pitcher',
                'HR Prob','XB Prob','1B Prob','BB Prob','K Prob',
                'HR Prob (no park)','XB Prob (no park)','1B Prob (no park)',
                'BB Prob (no park)','K Prob (no park)','RC','vs Grade']
    missing = [c for c in required if c not in df.columns]
    if missing:
        st.error(f"❌ Matchups CSV missing columns: {missing}")
        return None
    for col in ['Team','Batter','Pitcher','Game']:
        df[col] = df[col].astype(str).str.strip()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PITCHER DATA
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_pitcher_data():
    hits_df  = _fetch_csv(CONFIG['pitcher_hits_url'],  "Pitcher Hits")
    hrs_df   = _fetch_csv(CONFIG['pitcher_hrs_url'],   "Pitcher HRs")
    walks_df = _fetch_csv(CONFIG['pitcher_walks_url'], "Pitcher Walks")

    if hits_df is None and hrs_df is None and walks_df is None:
        return None

    def clean(df, col_name):
        if df is None:
            return pd.DataFrame(columns=['last_name','full_name','team','park',col_name])
        d = df.copy()
        d.columns = [c.strip() for c in d.columns]
        d['prob_val'] = (pd.to_numeric(
            d['Prob'].astype(str).str.replace('%','',regex=False).str.strip(),
            errors='coerce'
        ).fillna(0))
        d['last_name'] = d['Name'].astype(str).str.split().str[-1]
        d['full_name'] = d['Name'].astype(str).str.strip()
        d['team']      = d['Team'].astype(str).str.strip()
        d['park']      = d['Park'].astype(str).str.strip() if 'Park' in d.columns else ''
        return d[['last_name','full_name','team','park','prob_val']].rename(
            columns={'prob_val': col_name})

    hits_c  = clean(hits_df,  'hit8_prob')
    hrs_c   = clean(hrs_df,   'hr2_prob')
    walks_c = clean(walks_df, 'walk3_prob')

    merged = hits_c.merge(hrs_c,   on=['last_name','full_name','team','park'], how='outer')
    merged = merged.merge(walks_c, on=['last_name','full_name','team','park'], how='outer')

    merged['hit8_prob']  = merged['hit8_prob'].fillna(CONFIG['pitcher_hit_neutral'])
    merged['hr2_prob']   = merged['hr2_prob'].fillna(CONFIG['pitcher_hr_neutral'])
    merged['walk3_prob'] = merged['walk3_prob'].fillna(CONFIG['pitcher_walk_neutral'])

    M = CONFIG['pitcher_max_mult']

    # Hit multiplier — higher hit8_prob = pitcher gives up more hits = batter-friendly
    hit_anchor = CONFIG['pitcher_hit_neutral']
    merged['pitch_hit_mult'] = (
        1.0 + np.clip((merged['hit8_prob'] - hit_anchor) / 4.0 * M, -M, M)
    ).round(4)

    # HR multiplier — higher hr2_prob = pitcher gives up more HRs
    hr_anchor = CONFIG['pitcher_hr_neutral']
    merged['pitch_hr_mult'] = (
        1.0 + np.clip((merged['hr2_prob'] - hr_anchor) / 8.0 * M, -M, M)
    ).round(4)

    # Walk penalty — higher walk3_prob = fewer ABs resolved as hits = mild penalty
    walk_anchor = CONFIG['pitcher_walk_neutral']
    merged['pitch_walk_pen'] = (
        0.0 - np.clip((merged['walk3_prob'] - walk_anchor) / 10.0 * (M * 0.5),
                      -(M * 0.5), (M * 0.5))
    ).round(4)

    # Overall grade for display (hit_mult + walk_pen composite)
    composite = merged['pitch_hit_mult'] + merged['pitch_walk_pen']
    merged['pitch_grade'] = np.select(
        [composite >= 1.04, composite >= 1.01, composite >= 0.98, composite >= 0.95],
        ['A+', 'A', 'B', 'C'], default='D'
    )

    return merged.drop_duplicates(subset='last_name').reset_index(drop=True)


def merge_pitcher_data(df: pd.DataFrame, pitcher_df) -> pd.DataFrame:
    if pitcher_df is None or pitcher_df.empty:
        df['pitch_hit_mult']  = 1.0
        df['pitch_hr_mult']   = 1.0
        df['pitch_walk_pen']  = 0.0
        df['pitch_grade']     = 'B'
        df['hit8_prob']       = CONFIG['pitcher_hit_neutral']
        df['hr2_prob']        = CONFIG['pitcher_hr_neutral']
        df['walk3_prob']      = CONFIG['pitcher_walk_neutral']
        return df

    pm = pitcher_df.set_index('last_name')
    def _get(p, col, default):
        return pm.at[p, col] if p in pm.index else default

    df['pitch_hit_mult']  = df['Pitcher'].apply(lambda p: _get(p, 'pitch_hit_mult',  1.0))
    df['pitch_hr_mult']   = df['Pitcher'].apply(lambda p: _get(p, 'pitch_hr_mult',   1.0))
    df['pitch_walk_pen']  = df['Pitcher'].apply(lambda p: _get(p, 'pitch_walk_pen',  0.0))
    df['pitch_grade']     = df['Pitcher'].apply(lambda p: _get(p, 'pitch_grade',     'B'))
    df['hit8_prob']       = df['Pitcher'].apply(
        lambda p: _get(p, 'hit8_prob',  CONFIG['pitcher_hit_neutral']))
    df['hr2_prob']        = df['Pitcher'].apply(
        lambda p: _get(p, 'hr2_prob',   CONFIG['pitcher_hr_neutral']))
    df['walk3_prob']      = df['Pitcher'].apply(
        lambda p: _get(p, 'walk3_prob', CONFIG['pitcher_walk_neutral']))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, use_park: bool) -> pd.DataFrame:
    df = df.copy()
    for s in ['HR','XB','1B','BB','K']:
        pk = f'{s} Prob'
        pb = f'{s} Prob (no park)'
        df[f'p_{s.lower()}_park'] = pd.to_numeric(df[pk], errors='coerce').fillna(0)
        df[f'p_{s.lower()}_base'] = pd.to_numeric(df[pb], errors='coerce').fillna(0)
        df[f'p_{s.lower()}'] = (
            (df[f'p_{s.lower()}_park'] + df[f'p_{s.lower()}_base']) / 2
            if use_park else df[f'p_{s.lower()}_base']
        )

    df['vs_mod']    = pd.to_numeric(df['vs Grade'], errors='coerce').fillna(0).clip(-10,10) / 10
    df['vs_contrib'] = df['vs_mod'] * 2.0

    rc_col = 'RC' if use_park else 'RC (no park)'
    if rc_col not in df.columns:
        rc_col = 'RC'
    df['rc_norm'] = pd.to_numeric(df[rc_col], errors='coerce').fillna(0)
    rc_min, rc_max = df['rc_norm'].min(), df['rc_norm'].max()
    df['rc_contrib'] = (
        ((df['rc_norm'] - rc_min) / (rc_max - rc_min) * 2 - 1)
        if rc_max > rc_min else pd.Series(0.0, index=df.index)
    )

    df['PA']  = pd.to_numeric(df['PA'],  errors='coerce').fillna(0)
    df['AVG'] = pd.to_numeric(df['AVG'], errors='coerce').fillna(0)
    df['hist_bonus'] = np.where(
        df['PA'] >= CONFIG['hist_min_pa'],
        (df['AVG'] * CONFIG['hist_bonus_max']).round(3), 0.0
    )
    df['Starter']       = pd.to_numeric(df.get('Starter', 0), errors='coerce').fillna(0).astype(int)
    df['total_hit_prob'] = (df['p_1b'] + df['p_xb'] + df['p_hr']).clip(upper=100).round(1)

    # ── XB Boost — blended for HR Score supporting signal ─────────────────────
    # XB Boost is the matchup-specific adjustment factor for extra base hits.
    # Used only in HR Score as a small secondary signal: a positive XB Boost
    # means this batter is projected to make harder contact in this specific
    # matchup, even when HR Prob alone doesn't fully reflect that power ceiling.
    xb_boost_park = pd.to_numeric(df['XB Boost'],            errors='coerce').fillna(0) \
                    if 'XB Boost' in df.columns else pd.Series(0.0, index=df.index)
    xb_boost_base = pd.to_numeric(df['XB Boost (no park)'],  errors='coerce').fillna(0) \
                    if 'XB Boost (no park)' in df.columns else pd.Series(0.0, index=df.index)
    df['xb_boost'] = (xb_boost_park + xb_boost_base) / 2 if use_park else xb_boost_base

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    vc = df['vs_contrib']
    rc = df['rc_contrib']
    hb = df['hist_bonus']

    # Pitcher composite multipliers per score type
    hit_mult    = (df['pitch_hit_mult']  + df['pitch_walk_pen']).clip(0.90, 1.10)
    xb_mult     = ((df['pitch_hit_mult'] + df['pitch_hr_mult']) / 2 + df['pitch_walk_pen']).clip(0.90, 1.10)
    hr_mult     = (df['pitch_hr_mult']   + df['pitch_walk_pen']).clip(0.90, 1.10)
    single_mult = (df['pitch_hit_mult']  + df['pitch_walk_pen']).clip(0.90, 1.10)

    # ── 🎯 HIT SCORE ──────────────────────────────────────────────────────────
    hit_raw = (
          df['p_1b'] * 3.0 + df['p_xb'] * 2.0 + df['p_hr'] * 1.0
        - df['p_k']  * 2.5 - df['p_bb'] * 1.0
        + vc * 1.0 + rc * 0.5 + hb
    ) * hit_mult
    df['Hit_Score'] = normalize_0_100(hit_raw)

    # ── 1️⃣ SINGLE SCORE ───────────────────────────────────────────────────────
    single_raw = (
          df['p_1b'] * 5.0
        - df['p_k']  * 2.5 - df['p_bb'] * 1.0
        - df['p_xb'] * 0.8 - df['p_hr'] * 0.5   # power swing ≠ single
        + vc * 0.8 + rc * 0.4 + hb
    ) * single_mult
    df['Single_Score'] = normalize_0_100(single_raw)

    # ── 🔥 XB SCORE ───────────────────────────────────────────────────────────
    xb_raw = (
          df['p_xb'] * 5.0 + df['p_hr'] * 0.8
        - df['p_k']  * 1.5 - df['p_bb'] * 1.0
        + vc * 1.2 + rc * 0.6 + hb
    ) * xb_mult
    df['XB_Score'] = normalize_0_100(xb_raw)

    # ── 💣 HR SCORE ───────────────────────────────────────────────────────────
    # vs Grade weight reduced 1.5 → 0.5:
    #   Analysis of HR hitters showed 7/16 had negative vs Grades yet still
    #   hit home runs. vs Grade is a general matchup quality indicator, not
    #   a power indicator. Still present as a minor signal, just not a gatekeeper.
    #
    # XB Boost added at 0.03 weight:
    #   Players like Mookie Betts (XB Prob 5.25%, K% 10.6%) hit HRs despite
    #   below-average HR Prob. High XB Boost signals hard contact in this
    #   specific matchup — a legitimate secondary power indicator.
    #   Scale: XB Boost ranges ~-37 to +59, so 0.03 = max ±1.8 raw pts.
    hr_raw = (
          df['p_hr'] * 6.0 + df['p_xb'] * 0.8
        - df['p_k']  * 0.8 - df['p_bb'] * 1.0   # light K penalty
        + df['xb_boost'] * 0.03                   # hard contact signal
        + vc * 0.5 + rc * 0.5 + hb               # vs Grade reduced: 1.5 → 0.5
    ) * hr_mult
    df['HR_Score'] = normalize_0_100(hr_raw)

    # ── Base (no-park) versions for Park Δ ────────────────────────────────────
    df['Hit_Score_base']    = normalize_0_100((
          df['p_1b_base'] * 3.0 + df['p_xb_base'] * 2.0 + df['p_hr_base'] * 1.0
        - df['p_k_base']  * 2.5 - df['p_bb_base'] * 1.0
        + vc * 1.0 + rc * 0.5 + hb
    ) * hit_mult)

    df['Single_Score_base'] = normalize_0_100((
          df['p_1b_base'] * 5.0
        - df['p_k_base']  * 2.5 - df['p_bb_base'] * 1.0
        - df['p_xb_base'] * 0.8 - df['p_hr_base'] * 0.5
        + vc * 0.8 + rc * 0.4 + hb
    ) * single_mult)

    df['XB_Score_base']     = normalize_0_100((
          df['p_xb_base'] * 5.0 + df['p_hr_base'] * 0.8
        - df['p_k_base']  * 1.5 - df['p_bb_base'] * 1.0
        + vc * 1.2 + rc * 0.6 + hb
    ) * xb_mult)

    df['HR_Score_base']     = normalize_0_100((
          df['p_hr_base'] * 6.0 + df['p_xb_base'] * 0.8
        - df['p_k_base']  * 0.8 - df['p_bb_base'] * 1.0
        + df['xb_boost']  * 0.03                   # same XB Boost signal (park-neutral)
        + vc * 0.5 + rc * 0.5 + hb               # vs Grade reduced: 1.5 → 0.5
    ) * hr_mult)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def build_filters(df: pd.DataFrame) -> dict:
    st.sidebar.title("🏟️ A1PICKS Filters")
    st.sidebar.markdown("---")
    filters = {}

    st.sidebar.markdown("### 🎯 Betting Target")
    target_map = {
        "🎯 Hit Score  — Any Base Hit":          "hit",
        "1️⃣ Single Score — Single Specifically":  "single",
        "🔥 XB Score  — Double / Triple":         "xb",
        "💣 HR Score  — Home Run":                "hr",
    }
    label = st.sidebar.selectbox("Choose Your Betting Target", list(target_map.keys()))
    filters['target']    = target_map[label]
    score_col_map        = {'hit':'Hit_Score','single':'Single_Score','xb':'XB_Score','hr':'HR_Score'}
    filters['score_col'] = score_col_map[filters['target']]

    st.sidebar.markdown("### 🏟️ Park Adjustment")
    filters['use_park'] = st.sidebar.toggle(
        "Include Park Factors", value=True,
        help="ON = blends park-adjusted + base probabilities.\nOFF = pure player vs pitcher only."
    )

    st.sidebar.markdown("### ⚾ Pitcher Filter")
    filters['starters_only'] = st.sidebar.checkbox("Starters only", value=False)

    st.sidebar.markdown("### 📊 Stat Filters")
    filters['max_k']  = st.sidebar.slider("Max K Prob %",  10.0, 50.0, 35.0, 0.5)
    filters['max_bb'] = st.sidebar.slider("Max BB Prob %",  2.0, 20.0, 15.0, 0.5)

    min_cfg = {
        'hit':    ("Min Hit Prob % (1B+XB+HR)", "total_hit_prob", 0.0, 50.0, 20.0),
        'single': ("Min 1B Prob %",              "p_1b",           0.0, 30.0, 10.0),
        'xb':     ("Min XB Prob %",              "p_xb",           0.0, 12.0,  4.0),
        'hr':     ("Min HR Prob %",              "p_hr",           0.0,  8.0,  2.0),
    }
    pl, pc, mn, mx, dv = min_cfg[filters['target']]
    filters['min_prob']     = st.sidebar.slider(pl, mn, mx, dv, 0.5)
    filters['min_prob_col'] = pc

    filters['min_vs'] = st.sidebar.slider("Min vs Grade", -10, 10, -10, 1)

    st.sidebar.markdown("### 🏟️ Team Filters")
    all_teams = sorted(df['Team'].unique().tolist()) if df is not None else []
    filters['include_teams'] = st.sidebar.multiselect("Include Only Teams", options=all_teams)
    filters['exclude_teams'] = st.sidebar.multiselect("Exclude Teams",       options=all_teams)

    st.sidebar.markdown("### 🚫 Lineup Status")
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []
    all_players = sorted(df['Batter'].unique().tolist()) if df is not None else []
    excl = st.sidebar.multiselect(
        "Players NOT Playing Today",
        options=all_players,
        default=st.session_state.excluded_players,
        key="lineup_exclusions"
    )
    st.session_state.excluded_players = excl
    filters['excluded_players'] = excl
    if st.sidebar.button("🔄 Clear All Exclusions"):
        st.session_state.excluded_players = []
        st.rerun()

    st.sidebar.markdown("### 🔢 Display")
    sort_options = {
        "Score (High→Low)":      (filters['score_col'], False),
        "Hit Prob % (High→Low)": ("total_hit_prob",     False),
        "1B Prob % (High→Low)":  ("p_1b",               False),
        "XB Prob % (High→Low)":  ("p_xb",               False),
        "HR Prob % (High→Low)":  ("p_hr",               False),
        "K Prob % (Low→High)":   ("p_k",                True),
        "BB Prob % (Low→High)":  ("p_bb",               True),
        "vs Grade (High→Low)":   ("vs Grade",           False),
        "Pitcher Grade (A+→D)":  ("pitch_grade",        True),
    }
    filters['sort_label'] = st.sidebar.selectbox("Sort By", list(sort_options.keys()))
    filters['sort_col'], filters['sort_asc'] = sort_options[filters['sort_label']]
    filters['result_count'] = st.sidebar.selectbox(
        "Show Top N", [5,10,15,20,25,30,"All"], index=2
    )
    filters['best_per_team'] = st.sidebar.checkbox(
        "🏟️ Best player per team only", value=False
    )
    return filters


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    out = df.copy()

    if filters.get('starters_only'):
        out = out[out['Starter'] == 1]

    excl = filters.get('excluded_players', [])
    if excl:
        n = len(out)
        out = out[~out['Batter'].isin(excl)]
        if n - len(out):
            st.info(f"🚫 Excluded {n - len(out)} player(s) from lineups")

    if filters.get('include_teams'):
        out = out[out['Team'].isin(filters['include_teams'])]
    if filters.get('exclude_teams'):
        n = len(out)
        out = out[~out['Team'].isin(filters['exclude_teams'])]
        if n - len(out):
            st.info(f"🚫 Excluded players from {', '.join(filters['exclude_teams'])}")

    out = out[out['p_k']  <= filters['max_k']]
    out = out[out['p_bb'] <= filters['max_bb']]

    mc = filters.get('min_prob_col', 'total_hit_prob')
    if mc in out.columns:
        out = out[out[mc] >= filters['min_prob']]

    if filters['min_vs'] > -10:
        out = out[pd.to_numeric(out['vs Grade'], errors='coerce').fillna(-10) >= filters['min_vs']]

    sc = filters['score_col']
    if filters.get('best_per_team') and not out.empty:
        out = out.loc[out.groupby('Team')[sc].idxmax()].copy()
        st.info(f"🏟️ Showing best player from each of {len(out)} teams")

    sc_s = filters['sort_col']
    if sc_s in out.columns:
        out[sc_s] = pd.to_numeric(out[sc_s], errors='coerce')
        out = out.sort_values(sc_s, ascending=filters['sort_asc'], na_position='last')

    n = filters['result_count']
    if n != "All":
        out = out.head(int(n))
    return out


def get_slate_df(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Full slate with player exclusions only — used for persistent summary sections."""
    if df is None or df.empty:
        return df
    out = df.copy()
    excl = filters.get('excluded_players', [])
    if excl:
        out = out[~out['Batter'].isin(excl)]
    return out


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(
            'https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true',
            width=175
        )
    with c2:
        st.title("⚾ A1PICKS MLB Hit Predictor")
        st.markdown("*Extract the best betting targets from BallPark Pal simulation data*")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — DATA DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────

def render_data_dashboard(df: pd.DataFrame):
    st.markdown("---")
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card("card-blue",   "📈", "Matchups",     str(len(df))),                         unsafe_allow_html=True)
    with c2:
        st.markdown(card("card-blue",   "👤", "Batters",      str(df['Batter'].nunique())),           unsafe_allow_html=True)
    with c3:
        st.markdown(card("card-blue",   "🏟️", "Teams",        str(df['Team'].nunique())),             unsafe_allow_html=True)
    with c4:
        st.markdown(card("card-green",  "🎯", "Avg Hit Prob", f"{df['total_hit_prob'].mean():.1f}%"), unsafe_allow_html=True)
    with c5:
        sb   = staleness_badge()
        html = sb if sb else '<span class="badge badge-green">🟢 Just loaded</span>'
        st.markdown(card("card-purple", "🕐", "Data Age", html, "Refreshes every 15 min"),           unsafe_allow_html=True)
    st.markdown("---")


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — PITCHER LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

def render_pitcher_landscape(pitcher_df, df: pd.DataFrame):
    with st.expander("⚾ Today's Pitcher Landscape", expanded=True):
        if pitcher_df is None or pitcher_df.empty:
            st.info(
                "ℹ️ Pitcher CSV data not available. "
                "Scores are based on BallPark Pal matchup probabilities only."
            )
            return

        today_pitchers = df['Pitcher'].unique()
        rows = []
        pm = pitcher_df.set_index('last_name')

        for p in sorted(today_pitchers):
            if p in pm.index:
                r = pm.loc[p]
                rows.append({
                    'Pitcher':      r['full_name'],
                    'Team':         r['team'],
                    'Hit 8+ Prob':  f"{r['hit8_prob']:.1f}%",
                    'HR 2+ Prob':   f"{r['hr2_prob']:.1f}%",
                    'Walk 3+ Prob': f"{r['walk3_prob']:.1f}%",
                    'Grade':        r['pitch_grade'],
                    'Hit Mult':     f"{r['pitch_hit_mult']:.3f}×",
                    'HR Mult':      f"{r['pitch_hr_mult']:.3f}×",
                    'Walk Pen':     f"{r['pitch_walk_pen']:+.3f}",
                })
            else:
                rows.append({
                    'Pitcher':      p,
                    'Team':         '—',
                    'Hit 8+ Prob':  f"{CONFIG['pitcher_hit_neutral']:.1f}% (est.)",
                    'HR 2+ Prob':   f"{CONFIG['pitcher_hr_neutral']:.1f}% (est.)",
                    'Walk 3+ Prob': f"{CONFIG['pitcher_walk_neutral']:.1f}% (est.)",
                    'Grade':        'B',
                    'Hit Mult':     '1.000×',
                    'HR Mult':      '1.000×',
                    'Walk Pen':     '+0.000',
                })

        if not rows:
            st.info("No pitchers to display.")
            return

        disp = pd.DataFrame(rows)
        styled = (
            disp.style.apply(
                lambda x: [style_grade_cell(v) if x.name == 'Grade' else '' for v in x],
                axis=0
            )
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="pitcher-notice">'
            '📊 <b>How pitcher data affects scores:</b> '
            '<b>Hit 8+ Prob</b> drives Hit/Single/XB score multiplier — '
            'higher = pitcher gives up more hits = small batter boost. '
            '<b>HR 2+ Prob</b> drives HR score multiplier. '
            '<b>Walk 3+ Prob</b> applies a mild penalty to all scores — '
            'a walk-heavy pitcher wastes at-bats without producing the outcome you\'re betting on. '
            'Max effect is ±5% — batter-level probabilities always dominate.'
            '</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — PARK NOTICE
# ─────────────────────────────────────────────────────────────────────────────

def render_park_notice(slate_df: pd.DataFrame, filters: dict):
    sc, base_sc, use_park = filters['score_col'], filters['score_col'] + '_base', filters['use_park']
    if not use_park:
        st.markdown(
            '<div class="park-notice">🏟️ <b>Park Adjustment OFF</b> — '
            'Scores use pure player vs pitcher probabilities only.</div>',
            unsafe_allow_html=True
        )
        return
    if base_sc not in slate_df.columns or slate_df.empty:
        return
    ab, bl = slate_df[base_sc].mean(), slate_df[sc].mean()
    delta = bl - ab
    pct   = (delta / ab * 100) if ab != 0 else 0
    dir_  = "boosted" if delta >= 0 else "reduced"
    col_  = "#38ef7d" if delta >= 0 else "#e74c3c"
    st.markdown(
        f'<div class="park-notice">🏟️ <b>Park Adjustment ON</b> — '
        f'Park factors have <span style="color:{col_};font-weight:700">'
        f'{dir_} scores by ~{abs(pct):.1f}%</span> on average. '
        f'Toggle OFF to see pure base scores. '
        f'<b>Park Δ</b> column shows each player\'s individual park impact.</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — SCORE SUMMARY CARDS  (full slate — never changes with filter)
# ─────────────────────────────────────────────────────────────────────────────

def render_score_summary_cards(slate_df: pd.DataFrame, filters: dict):
    if slate_df.empty:
        return
    st.markdown("#### 🏆 Today's Top Player per Betting Target (Full Slate)")
    score_defs = [
        ('Hit_Score',    '🎯 Hit',    'card-green', 'Any Base Hit'),
        ('Single_Score', '1️⃣ Single', 'card-teal',  'Single Specifically'),
        ('XB_Score',     '🔥 XB',     'card-amber', 'Double / Triple'),
        ('HR_Score',     '💣 HR',     'card-red',   'Home Run'),
    ]
    cols = st.columns(4)
    for i, (sc, lbl, css, desc) in enumerate(score_defs):
        if sc not in slate_df.columns:
            continue
        row      = slate_df.loc[slate_df[sc].idxmax()]
        base_col = sc + '_base'
        park_str = ""
        if filters['use_park'] and base_col in slate_df.columns and row[base_col] != 0:
            delta = row[sc] - row[base_col]
            pct   = delta / row[base_col] * 100
            park_str = f" | Park: {'+' if delta>=0 else ''}{pct:.1f}%"
        with cols[i]:
            st.markdown(
                card(css, lbl, desc, row['Batter'],
                     f"Score {row[sc]:.1f}{park_str} · {row['Team']}"),
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def render_results_table(filtered_df: pd.DataFrame, filters: dict):
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters. Try relaxing the thresholds.")
        return

    sc, base_sc = filters['score_col'], filters['score_col'] + '_base'
    use_park    = filters['use_park']
    disp        = filtered_df.copy()

    disp['K% vs Lg']    = (CONFIG['league_k_avg']  - disp['p_k']).round(1)
    disp['BB% vs Lg']   = (CONFIG['league_bb_avg'] - disp['p_bb']).round(1)
    disp['HR% vs Lg']   = (disp['p_hr'] - CONFIG['league_hr_avg']).round(2)
    disp['Hit Prob %']  = disp['total_hit_prob'].round(1)
    disp['Hist PA']     = disp['PA'].astype(int)
    disp['AVG vs P']    = disp['AVG'].round(3)
    disp['vs Grade']    = pd.to_numeric(disp['vs Grade'], errors='coerce').round(0).astype(int)
    disp['Park Δ']      = (disp[sc] - disp[base_sc]).round(1) if (use_park and base_sc in disp.columns) else 0.0

    lbl_map = {'Hit_Score':'🎯 Hit','Single_Score':'1️⃣ Single','XB_Score':'🔥 XB','HR_Score':'💣 HR'}
    active  = lbl_map.get(sc, 'Score')

    cols_map = {'Batter':'Batter','Team':'Team','Pitcher':'Pitcher','pitch_grade':'Pitcher Grade', sc: active}
    if use_park and base_sc in disp.columns:
        cols_map[base_sc]  = f'{active} (no park)'
        cols_map['Park Δ'] = 'Park Δ'
    for sc2, lb2 in lbl_map.items():
        if sc2 != sc and sc2 in disp.columns:
            cols_map[sc2] = lb2
    cols_map.update({
        'Hit Prob %':'Hit Prob %','p_1b':'1B Prob %','p_xb':'XB Prob %',
        'p_hr':'HR Prob %','p_k':'K Prob %','p_bb':'BB Prob %',
        'K% vs Lg':'K% vs Lg','BB% vs Lg':'BB% vs Lg','HR% vs Lg':'HR% vs Lg',
        'vs Grade':'vs Grade','Hist PA':'Hist PA','AVG vs P':'AVG vs P',
    })

    existing = [c for c in cols_map if c in disp.columns]
    out_df   = disp[existing].rename(columns=cols_map)

    fmt = {}
    for cn in out_df.columns:
        if 'Prob %' in cn or cn in ['1B Prob %','XB Prob %','HR Prob %','K Prob %','BB Prob %']:
            fmt[cn] = "{:.1f}%"
        elif cn in ['K% vs Lg','BB% vs Lg']:
            fmt[cn] = "{:+.1f}%"
        elif cn == 'HR% vs Lg':
            fmt[cn] = "{:+.2f}%"
        elif cn == 'Park Δ':
            fmt[cn] = "{:+.1f}"
        elif cn == 'AVG vs P':
            fmt[cn] = "{:.3f}"
        elif any(e in cn for e in ['🎯','1️⃣','🔥','💣']) and 'Prob' not in cn and 'Park' not in cn:
            fmt[cn] = "{:.1f}"

    styled = out_df.style.format(fmt, na_rep="—")

    cmap_map = {'🎯 Hit':'Greens','1️⃣ Single':'YlGn','🔥 XB':'YlOrBr','💣 HR':'YlOrRd'}
    for sn, cm in cmap_map.items():
        if sn in out_df.columns:
            try:
                styled = styled.background_gradient(subset=[sn], cmap=cm, vmin=0, vmax=100)
            except Exception:
                pass
    if 'Park Δ' in out_df.columns:
        styled = styled.background_gradient(subset=['Park Δ'], cmap='RdYlGn', vmin=-10, vmax=10)
    if 'K% vs Lg' in out_df.columns:
        styled = styled.background_gradient(subset=['K% vs Lg'], cmap='RdYlGn', vmin=-8, vmax=12)
    if 'HR% vs Lg' in out_df.columns:
        styled = styled.background_gradient(subset=['HR% vs Lg'], cmap='RdYlGn', vmin=-2, vmax=3)
    if 'vs Grade' in out_df.columns:
        styled = styled.background_gradient(subset=['vs Grade'], cmap='RdYlGn', vmin=-10, vmax=10)
    if 'Pitcher Grade' in out_df.columns:
        styled = styled.apply(
            lambda x: [style_grade_cell(v) if x.name == 'Pitcher Grade' else '' for v in x],
            axis=0
        )

    st.dataframe(styled, use_container_width=True)

    LG        = CONFIG
    park_note = (
        "Park Δ = score impact of park factors (+ = park helped, − = park hurt). "
        "Toggle park OFF to see pure base scores."
        if use_park else "Park Adjustment OFF — scores use pure player vs pitcher probabilities."
    )
    st.markdown(f"""
<div class="legend">
<b>📊 Score Guide</b><br>
🎯 <b>Hit Score</b> — Any base hit. Heavy K% penalty. 1B Prob drives it.<br>
1️⃣ <b>Single Score</b> — Singles specifically. High XB%/HR% penalised (power swing ≠ single).<br>
🔥 <b>XB Score</b> — Double or triple. XB Prob dominant. Moderate K% tolerance.<br>
💣 <b>HR Score</b> — Home run. HR Prob dominant. Light K% penalty (power hitters K more — that's fine).<br><br>
<b>Pitcher Grade</b> — A+ to D from today's game-level pitcher stats. Applied as mild ±5% multiplier.<br>
<b>K% vs Lg</b> — Positive = better contact than league avg ({LG['league_k_avg']}%).<br>
<b>BB% vs Lg</b> — Positive = more aggressive than league avg ({LG['league_bb_avg']}%).<br>
<b>HR% vs Lg</b> — Positive = HR rate above league avg ({LG['league_hr_avg']}%).<br>
<b>vs Grade</b> — BallPark Pal batter vs pitcher on-paper rating (−10 to +10).<br>
<b>Hist PA / AVG vs P</b> — History vs this pitcher. Bonus activates at ≥{LG['hist_min_pa']} PA.<br><br>
{park_note}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — BEST PER TARGET  (full slate — never changes with filter)
# ─────────────────────────────────────────────────────────────────────────────

def render_best_per_target(slate_df: pd.DataFrame, filters: dict):
    if len(slate_df) < 3:
        st.info("💡 Need at least 3 players for this analysis.")
        return

    with st.expander("🔍 Best Player per Betting Target (Full Slate)", expanded=True):
        st.caption(
            "Drawn from the full day's slate (player exclusions only). "
            "Does not change when you switch betting targets."
        )
        score_defs = [
            ('Hit_Score',    '🎯 Hit',    'Any Base Hit'),
            ('Single_Score', '1️⃣ Single', 'Single'),
            ('XB_Score',     '🔥 XB',     'Double / Triple'),
            ('HR_Score',     '💣 HR',     'Home Run'),
        ]
        cols = st.columns(4)
        for i, (sc, lbl, desc) in enumerate(score_defs):
            if sc not in slate_df.columns:
                continue
            row      = slate_df.loc[slate_df[sc].idxmax()]
            base_col = sc + '_base'
            park_str = ""
            if filters['use_park'] and base_col in slate_df.columns and row[base_col] != 0:
                delta    = row[sc] - row[base_col]
                pct      = delta / row[base_col] * 100
                park_str = f"Park: {'+' if delta>=0 else ''}{pct:.1f}%"

            k_lg  = CONFIG['league_k_avg']  - row['p_k']
            bb_lg = CONFIG['league_bb_avg'] - row['p_bb']
            hr_lg = row['p_hr'] - CONFIG['league_hr_avg']
            hist  = (f"{int(row['PA'])} PA / {row['AVG']:.3f} avg vs this pitcher"
                     if row['PA'] >= CONFIG['hist_min_pa']
                     else "No significant history vs this pitcher")
            grade_h = grade_span(str(row.get('pitch_grade', 'B')))
            pk_row  = f"| Park Impact | {park_str} |" if park_str else ""

            with cols[i]:
                st.markdown(f"**{lbl}** — {desc}")
                st.success(f"**{row['Batter']}** ({row['Team']})")
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Score | **{row[sc]:.1f}** / 100 |
| Pitcher Grade | {grade_h} — {row['Pitcher']} |
| Hit Prob | {row['total_hit_prob']:.1f}% |
| 1B% / XB% / HR% | {row['p_1b']:.1f}% / {row['p_xb']:.1f}% / {row['p_hr']:.1f}% |
| K% | {row['p_k']:.1f}% (vs Lg: {k_lg:+.1f}%) |
| BB% | {row['p_bb']:.1f}% (vs Lg: {bb_lg:+.1f}%) |
| HR% vs Lg | {hr_lg:+.2f}% |
| vs Grade | {int(row['vs Grade'])} |
{pk_row}
| History | {hist} |
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — VISUALIZATIONS
# ─────────────────────────────────────────────────────────────────────────────

def render_visualizations(df: pd.DataFrame, filtered_df: pd.DataFrame, score_col: str):
    st.subheader("📈 Analysis Charts")
    c1, c2 = st.columns(2)
    with c1:
        ch = alt.Chart(df).mark_bar(color='#2980b9', opacity=0.75).encode(
            alt.X(f'{score_col}:Q', bin=alt.Bin(maxbins=15), title='Score'),
            alt.Y('count()', title='Players'),
            tooltip=['count()']
        ).properties(title=f'{score_col} — Full Slate Distribution', width=350, height=270)
        st.altair_chart(ch, use_container_width=True)
    with c2:
        if not filtered_df.empty:
            ch2 = alt.Chart(filtered_df).mark_circle(size=90, opacity=0.75).encode(
                alt.X('total_hit_prob:Q', title='Hit Prob %'),
                alt.Y('p_k:Q',           title='K Prob %'),
                alt.Color(f'{score_col}:Q', scale=alt.Scale(scheme='viridis')),
                alt.Size('p_hr:Q',       title='HR Prob %'),
                tooltip=['Batter','Team',score_col,'total_hit_prob','p_k','p_hr','pitch_grade']
            ).properties(title='Hit Prob vs K Risk (filtered)', width=350, height=270)
            st.altair_chart(ch2, use_container_width=True)

    if not filtered_df.empty and len(filtered_df) <= 30:
        melt = filtered_df[['Batter','Hit_Score','Single_Score','XB_Score','HR_Score']].melt(
            id_vars='Batter', var_name='Score Type', value_name='Score'
        )
        melt['Score Type'] = melt['Score Type'].map({
            'Hit_Score':'🎯 Hit','Single_Score':'1️⃣ Single',
            'XB_Score':'🔥 XB','HR_Score':'💣 HR',
        })
        ch3 = alt.Chart(melt).mark_bar().encode(
            alt.X('Batter:N', sort='-y'),
            alt.Y('Score:Q'),
            alt.Color('Score Type:N', scale=alt.Scale(
                domain=['🎯 Hit','1️⃣ Single','🔥 XB','💣 HR'],
                range=['#27ae60','#43b89c','#f39c12','#e74c3c']
            )),
            alt.Column('Score Type:N', title=None),
            tooltip=['Batter','Score Type', alt.Tooltip('Score:Q', format='.1f')]
        ).properties(width=180, height=250, title='All Four Scores per Player')
        st.altair_chart(ch3, use_container_width=True)

    if not filtered_df.empty:
        ts = filtered_df.groupby('Team').agg(
            Players     =('Batter',        'count'),
            AvgHitProb  =('total_hit_prob', 'mean'),
            AvgHitScore =('Hit_Score',      'mean'),
            AvgXBScore  =('XB_Score',       'mean'),
            AvgHRScore  =('HR_Score',       'mean'),
        ).round(1).sort_values('AvgHitProb', ascending=False).reset_index()
        ts.columns = ['Team','Players','Avg Hit Prob%','🎯 Hit Score','🔥 XB Score','💣 HR Score']
        st.subheader("🏟️ Team Summary")
        st.dataframe(ts, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def main_page():
    render_header()
    if 'data_loaded_at' not in st.session_state:
        st.session_state.data_loaded_at = datetime.now()

    with st.spinner("⚾ Loading today's matchups from BallPark Pal..."):
        raw_df     = load_matchups()
        pitcher_df = load_pitcher_data()

    if raw_df is None:
        st.error("❌ Could not load Matchups data. Check connection or refresh.")
        return

    filters   = build_filters(raw_df)
    df        = compute_metrics(raw_df, use_park=filters['use_park'])
    df        = merge_pitcher_data(df, pitcher_df)
    df        = compute_scores(df)
    slate_df  = get_slate_df(df, filters)

    render_data_dashboard(df)
    render_pitcher_landscape(pitcher_df, df)
    render_park_notice(slate_df, filters)
    render_score_summary_cards(slate_df, filters)
    st.markdown("---")

    filtered_df = apply_filters(df, filters)
    target_labels = {
        'Hit_Score':    '🎯 Any Base Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 Extra Base Hit',
        'HR_Score':     '💣 Home Run',
    }
    st.subheader(f"Top {len(filtered_df)} {target_labels.get(filters['score_col'], 'Hit')} Candidates")
    render_results_table(filtered_df, filters)
    render_best_per_target(slate_df, filters)

    if not filtered_df.empty:
        render_visualizations(df, filtered_df, filters['score_col'])

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.data_loaded_at = datetime.now()
            st.rerun()
    with c2:
        if not filtered_df.empty:
            st.download_button(
                "💾 Export Results (CSV)",
                filtered_df.to_csv(index=False),
                f"a1picks_mlb_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    with c3:
        sb = staleness_badge()
        if sb:
            st.markdown(sb, unsafe_allow_html=True)

    if not filtered_df.empty:
        with st.expander("⚡ Quick Player Exclusions"):
            available = [p for p in filtered_df['Batter'].tolist()
                         if p not in st.session_state.excluded_players]
            if available:
                cl, cr = st.columns(2)
                for i, player in enumerate(available[:5]):
                    with cl:
                        if st.button(f"❌ {player}", key=f"qx_{i}"):
                            st.session_state.excluded_players.append(player)
                            st.rerun()
                for i, player in enumerate(available[5:10]):
                    with cr:
                        if st.button(f"❌ {player}", key=f"qx2_{i}"):
                            st.session_state.excluded_players.append(player)
                            st.rerun()
            else:
                st.success("✅ All current results are confirmed playing.")
            if st.session_state.excluded_players:
                st.markdown(f"**Excluded:** {', '.join(st.session_state.excluded_players)}")
                if st.button("🔄 Re-include All", key="main_clear"):
                    st.session_state.excluded_players = []
                    st.rerun()

    st.markdown("---")
    LG = CONFIG
    st.markdown(f"""
### 🎯 V4.1 Betting Strategy Guide

#### Score → Bet Type Mapping
| Score | Bet | K% Tolerance | Primary Driver | Pitcher Stat |
|-------|-----|-------------|----------------|--------------|
| 🎯 Hit | Any base hit | Low | 1B Prob × 3.0 | Hit 8+ Prob |
| 1️⃣ Single | Single specifically | Low + XB/HR penalised | 1B Prob × 5.0 | Hit 8+ Prob |
| 🔥 XB | Double or Triple | Moderate | XB Prob × 5.0 | Blend Hit + HR |
| 💣 HR | Home run | High | HR Prob × 6.0 | HR 2+ Prob |

#### Pitcher Grade Context
- Grades reflect **game-level** likelihood: will this pitcher allow 8+ hits, 2+ HRs, 3+ walks today?
- Applied as mild ±5% multiplier — batter-level probabilities always dominate.
- A+ = tailwind. D = headwind. Won't flip a bad matchup good, but adds real context.

#### Park Toggle
- **ON** (default): Blends park-adjusted + base probabilities.
- **OFF**: Pure player vs pitcher — no park/weather adjustment.
- **Park Δ** column shows exactly how much park shifted each score.

#### League Baselines (4-year stable)
K%: **{LG['league_k_avg']}%** | BB%: **{LG['league_bb_avg']}%** | HR%: **{LG['league_hr_avg']}%** | AVG: **{LG['league_avg']}**

**V4.1 — Pitcher Game Layer | Stable Summary Cards | Updated League Averages**
    """)


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def info_page():
    st.title("📚 A1PICKS MLB Hit Predictor — Reference Manual")
    with st.expander("📖 System Overview", expanded=True):
        LG = CONFIG
        st.markdown(f"""
## Purpose
Extract the best betting targets from BallPark Pal's simulation data (3,000 runs per game).

## Data Sources
| File | Purpose |
|------|---------|
| `Matchups.csv` | Main batter vs pitcher probabilities (park + no-park versions) |
| `pitcher_hits.csv` | Game-level probability pitcher allows 8+ hits |
| `pitcher_hrs.csv` | Game-level probability pitcher allows 2+ HRs |
| `pitcher_walks.csv` | Game-level probability pitcher issues 3+ walks |

## Four Scores
| Score | Formula | K% Penalty |
|-------|---------|-----------|
| 🎯 Hit | 1B×3 + XB×2 + HR×1 − K×2.5 − BB×1 | Heavy |
| 1️⃣ Single | 1B×5 − K×2.5 − BB×1 − XB×0.8 − HR×0.5 | Heavy |
| 🔥 XB | XB×5 + HR×0.8 − K×1.5 − BB×1 | Moderate |
| 💣 HR | HR×6 + XB×0.8 − K×0.8 − BB×1 | Light |

All scores multiplied by pitcher game-level modifier before normalisation to 0–100.

## Pitcher Multiplier
- Max ±5% effect per score
- Hit/Single/XB: driven by Hit_8Plus_Probability (neutral anchor {LG['pitcher_hit_neutral']}%)
- HR: driven by HR_2Plus_Probability (neutral anchor {LG['pitcher_hr_neutral']}%)
- All scores: mild walk penalty from Walk_3Plus_Probability (neutral anchor {LG['pitcher_walk_neutral']}%)

## Park Mode
Blends with-park and no-park columns. Park Δ shows exact impact per player.

## Historical Tiebreaker
≥{LG['hist_min_pa']} PA vs this pitcher → up to {LG['hist_bonus_max']} bonus raw points before normalisation.

## League Baselines
K%: **{LG['league_k_avg']}%** | BB%: **{LG['league_bb_avg']}%** | HR%: **{LG['league_hr_avg']}%** | AVG: **{LG['league_avg']}**
        """)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("🎵 Background Music"):
        audio_url = (
            "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/"
            "Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        )
        components.html(
            f'<audio controls autoplay loop style="width:100%;">'
            f'<source src="{audio_url}" type="audio/mpeg"></audio>',
            height=60
        )
    page = st.sidebar.radio("Navigate", ["⚾ Hit Predictor", "📚 Reference Manual"], index=0)
    if page == "⚾ Hit Predictor":
        main_page()
    else:
        info_page()
    st.sidebar.markdown("---")
    st.sidebar.markdown("**V4.1** | Pitcher Layer | Park Toggle | 4 Scores")


if __name__ == "__main__":
    main()
