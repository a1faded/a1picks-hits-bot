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


# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  Dark Precision Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root {
  --bg:#080c14; --surf:#0f1623; --surf2:#161e2e; --border:#1e2d3d;
  --text:#e2e8f0; --muted:#64748b;
  --hit:#10b981; --single:#06b6d4; --xb:#f59e0b; --hr:#ef4444;
  --accent:#3b82f6; --pos:#22c55e; --neg:#ef4444;
}
html,body,[class*="css"]{ font-family:'Outfit',sans-serif !important; }
.block-container{ padding:.75rem 1rem 2rem !important; max-width:1400px; }
[data-testid="stSidebar"]{ background:var(--surf) !important; border-right:1px solid var(--border); }
[data-testid="stSidebar"] *{ font-size:.85rem !important; }
h1,h2,h3,h4,h5{ font-family:'Outfit',sans-serif !important; }
/* stat bar */
.stat-bar{ display:flex; flex-wrap:wrap; gap:.4rem; background:var(--surf);
  border:1px solid var(--border); border-radius:10px; padding:.6rem 1rem;
  margin-bottom:.75rem; align-items:center; }
.stat-item{ display:flex; flex-direction:column; align-items:center;
  padding:.3rem .75rem; border-right:1px solid var(--border); min-width:80px; }
.stat-item:last-child{ border-right:none; }
.stat-item .val{ font-family:'JetBrains Mono',monospace; font-size:1.15rem;
  font-weight:600; color:var(--text); line-height:1; }
.stat-item .lbl{ font-size:.65rem; color:var(--muted); text-transform:uppercase;
  letter-spacing:.05em; margin-top:.2rem; }
/* score cards */
.score-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin:.5rem 0; }
@media(max-width:768px){
  .score-grid{ grid-template-columns:repeat(2,1fr); }
  .pcard-grid{ grid-template-columns:1fr 1fr; }
  .stat-item{ min-width:60px; padding:.3rem .4rem; }
  .block-container{ padding:.4rem .5rem 2rem !important; }
}
.scard{ background:var(--surf); border:1px solid var(--border); border-radius:10px;
  padding:.7rem .85rem; position:relative; overflow:hidden; transition:border-color .2s; }
.scard:hover{ border-color:var(--accent); }
.scard::before{ content:''; position:absolute; top:0; left:0; right:0; height:3px; }
.scard-hit::before{ background:var(--hit); }
.scard-single::before{ background:var(--single); }
.scard-xb::before{ background:var(--xb); }
.scard-hr::before{ background:var(--hr); }
.scard .sc-type{ font-size:.65rem; text-transform:uppercase; letter-spacing:.08em;
  color:var(--muted); margin-bottom:.25rem; }
.scard .sc-type span{ font-weight:700; }
.scard-hit .sc-type span{ color:var(--hit); }
.scard-single .sc-type span{ color:var(--single); }
.scard-xb .sc-type span{ color:var(--xb); }
.scard-hr .sc-type span{ color:var(--hr); }
.scard .sc-name{ font-size:1rem; font-weight:700; color:var(--text);
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.scard .sc-meta{ font-family:'JetBrains Mono',monospace; font-size:.7rem;
  color:var(--muted); margin-top:.2rem; }
.scard .sc-score{ font-family:'JetBrains Mono',monospace; font-size:.8rem;
  font-weight:600; position:absolute; top:.6rem; right:.7rem; }
.scard-hit .sc-score{ color:var(--hit); }
.scard-single .sc-score{ color:var(--single); }
.scard-xb .sc-score{ color:var(--xb); }
.scard-hr .sc-score{ color:var(--hr); }
/* player detail cards */
.pcard-grid{ display:grid; grid-template-columns:repeat(4,1fr); gap:.5rem; margin:.5rem 0; }
.pcard{ background:var(--surf); border:1px solid var(--border); border-radius:10px;
  padding:.75rem; font-size:.8rem; }
.pcard-header{ display:flex; justify-content:space-between; align-items:flex-start;
  margin-bottom:.5rem; padding-bottom:.4rem; border-bottom:1px solid var(--border); }
.pcard-name{ font-weight:700; color:var(--text); font-size:.9rem; }
.pcard-team{ font-size:.65rem; color:var(--muted); margin-top:.1rem; }
.pcard-score{ font-family:'JetBrains Mono',monospace; font-size:1.1rem; font-weight:700; }
.pcard-hit .pcard-score{ color:var(--hit); }
.pcard-single .pcard-score{ color:var(--single); }
.pcard-xb .pcard-score{ color:var(--xb); }
.pcard-hr .pcard-score{ color:var(--hr); }
.pcard-row{ display:flex; justify-content:space-between; padding:.18rem 0;
  border-bottom:1px solid var(--border); color:var(--muted); }
.pcard-row:last-child{ border-bottom:none; }
.pcard-row .pk{ font-size:.7rem; }
.pcard-row .pv{ font-family:'JetBrains Mono',monospace; font-size:.72rem;
  color:var(--text); font-weight:600; }
.pos-val{ color:var(--pos) !important; }
.neg-val{ color:var(--neg) !important; }
/* notices */
.notice{ border-left:3px solid; padding:.4rem .8rem; border-radius:0 6px 6px 0;
  font-size:.78rem; margin:.4rem 0; line-height:1.5; }
.notice-park{ background:#1a1a2e; border-color:var(--xb); color:#fde68a; }
.notice-pitcher{ background:#0d1a14; border-color:var(--hit); color:#a7f3d0; }
.notice-info{ background:#0f1623; border-color:var(--accent); color:#bfdbfe; }
/* grade pills */
.gp{ padding:1px 7px; border-radius:20px; font-size:.7rem; font-weight:700; display:inline-block; }
.gp-ap{ background:#1a9641; color:white; }
.gp-a{ background:#a6d96a; color:#111; }
.gp-b{ background:#fef08a; color:#111; }
.gp-c{ background:#fdae61; color:#111; }
.gp-d{ background:#ef4444; color:white; }
/* legend */
.legend-compact{ background:var(--surf); border:1px solid var(--border); border-radius:8px;
  padding:.6rem 1rem; font-size:.76rem; color:var(--muted); line-height:1.7; margin:.5rem 0; }
.legend-compact b{ color:var(--text); }
.legend-compact .hit-c{ color:var(--hit); }
.legend-compact .xb-c{ color:var(--xb); }
.legend-compact .hr-c{ color:var(--hr); }
.legend-compact .sl-c{ color:var(--single); }
/* staleness */
.sbadge{ display:inline-flex; align-items:center; gap:.3rem; padding:.25rem .65rem;
  border-radius:20px; font-size:.73rem; font-weight:600; font-family:'JetBrains Mono',monospace; }
.sbadge-green{ background:#052e16; color:#4ade80; border:1px solid #166534; }
.sbadge-yellow{ background:#1c1400; color:#fbbf24; border:1px solid #713f12; }
.sbadge-red{ background:#1c0000; color:#f87171; border:1px solid #7f1d1d; }
/* header */
.app-header{ display:flex; align-items:center; gap:1rem; padding:.4rem 0 .6rem;
  border-bottom:1px solid var(--border); margin-bottom:.75rem; flex-wrap:wrap; }
.app-header .title-wrap h1{ font-size:1.15rem !important; font-weight:700;
  color:var(--text); margin:0 !important; line-height:1.2; }
.app-header .title-wrap p{ font-size:.68rem; color:var(--muted); margin:.1rem 0 0; }
.app-header .meta{ margin-left:auto; display:flex; align-items:center; gap:.5rem; }
/* section head */
.section-head{ font-size:.7rem; text-transform:uppercase; letter-spacing:.1em;
  color:var(--muted); margin:.9rem 0 .4rem; display:flex; align-items:center; gap:.4rem; }
.section-head::after{ content:''; flex:1; height:1px; background:var(--border); }
/* result head */
.result-head{ display:flex; align-items:center; gap:.5rem; margin:.5rem 0 .3rem; }
.result-head .rh-label{ font-size:.85rem; font-weight:600; color:var(--text); }
.result-head .rh-count{ background:var(--surf2); border:1px solid var(--border);
  border-radius:20px; padding:.1rem .55rem; font-family:'JetBrains Mono',monospace;
  font-size:.72rem; color:var(--muted); }
/* table */
[data-testid="stDataFrame"] th{ font-size:.72rem !important;
  font-family:'Outfit',sans-serif !important; text-transform:uppercase; letter-spacing:.04em; }
[data-testid="stDataFrame"] td{ font-family:'JetBrains Mono',monospace !important;
  font-size:.78rem !important; }
/* pitcher table */
.pt-wrap{ overflow-x:auto; }
.pt-table{ width:100%; border-collapse:collapse; font-size:.78rem; }
.pt-table th{ background:var(--surf2); color:var(--muted); text-transform:uppercase;
  letter-spacing:.05em; font-size:.65rem; padding:.45rem .7rem; border-bottom:1px solid var(--border); white-space:nowrap; }
.pt-table td{ padding:.4rem .7rem; border-bottom:1px solid var(--border); color:var(--text);
  font-family:'JetBrains Mono',monospace; white-space:nowrap; }
.pt-table tr:last-child td{ border-bottom:none; }
.pt-table tr:hover td{ background:var(--surf2); }
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



# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def staleness_badge() -> str:
    if 'data_loaded_at' not in st.session_state:
        return ""
    mins = int((datetime.now() - st.session_state.data_loaded_at).total_seconds() / 60)
    if mins < 5:
        return f'<span class="sbadge sbadge-green">🟢 {mins}m ago</span>'
    elif mins < 12:
        return f'<span class="sbadge sbadge-yellow">🟡 {mins}m old</span>'
    return f'<span class="sbadge sbadge-red">🔴 {mins}m — refresh</span>'


def grade_pill(grade: str) -> str:
    css = {'A+':'gp-ap','A':'gp-a','B':'gp-b','C':'gp-c','D':'gp-d'}.get(grade,'gp-b')
    return f'<span class="gp {css}">{grade}</span>'


def style_grade_cell(val):
    return {
        'A+': 'background-color:#1a9641;color:white;font-weight:700',
        'A':  'background-color:#a6d96a;color:#111;font-weight:700',
        'B':  'background-color:#fef08a;color:#111',
        'C':  'background-color:#fdae61;color:#111',
        'D':  'background-color:#ef4444;color:white;font-weight:700',
    }.get(str(val), '')


def normalize_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(1)


def _pv(val, pos_good=True) -> str:
    """Coloured value span for player detail cards."""
    try:
        v = float(val)
        css = ('pos-val' if v >= 0 else 'neg-val') if pos_good else ('neg-val' if v >= 0 else 'pos-val')
        return f'<span class="pv {css}">{val}</span>'
    except Exception:
        return f'<span class="pv">{val}</span>'


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    sb = staleness_badge()
    sb_html = sb if sb else '<span class="sbadge sbadge-green">🟢 Ready</span>'
    st.markdown(f"""
<div class="app-header">
  <div class="logo-wrap">
    <img src="https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true"
         style="height:38px;width:auto;" />
  </div>
  <div class="title-wrap">
    <h1>A1PICKS MLB Hit Predictor</h1>
    <p>BallPark Pal simulation data → betting targets · V4.1</p>
  </div>
  <div class="meta">{sb_html}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — STAT BAR  (replaces big 5-card dashboard)
# ─────────────────────────────────────────────────────────────────────────────

def render_stat_bar(df: pd.DataFrame):
    if df is None or df.empty:
        return
    matchups   = len(df)
    batters    = df['Batter'].nunique()
    teams      = df['Team'].nunique()
    avg_hp     = df['total_hit_prob'].mean()
    avg_hr     = df['p_hr'].mean() if 'p_hr' in df.columns else 0

    st.markdown(f"""
<div class="stat-bar">
  <div class="stat-item">
    <span class="val">{matchups}</span>
    <span class="lbl">Matchups</span>
  </div>
  <div class="stat-item">
    <span class="val">{batters}</span>
    <span class="lbl">Batters</span>
  </div>
  <div class="stat-item">
    <span class="val">{teams}</span>
    <span class="lbl">Teams</span>
  </div>
  <div class="stat-item">
    <span class="val" style="color:var(--hit)">{avg_hp:.1f}%</span>
    <span class="lbl">Avg Hit Prob</span>
  </div>
  <div class="stat-item">
    <span class="val" style="color:var(--hr)">{avg_hr:.2f}%</span>
    <span class="lbl">Avg HR Prob</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — SCORE SUMMARY CARDS  (full slate — never changes with filter)
# ─────────────────────────────────────────────────────────────────────────────

def render_score_summary_cards(slate_df: pd.DataFrame, filters: dict):
    if slate_df.empty:
        return

    st.markdown('<div class="section-head">🏆 Today\'s Best — Full Slate</div>',
                unsafe_allow_html=True)

    score_defs = [
        ('Hit_Score',    'scard-hit',    '🎯',  'HIT',    'Any Base Hit'),
        ('Single_Score', 'scard-single', '1️⃣',  'SINGLE', 'Single Specifically'),
        ('XB_Score',     'scard-xb',     '🔥',  'XB',     'Double / Triple'),
        ('HR_Score',     'scard-hr',     '💣',  'HR',     'Home Run'),
    ]

    cards_html = '<div class="score-grid">'
    for sc, css, icon, short, desc in score_defs:
        if sc not in slate_df.columns:
            continue
        row      = slate_df.loc[slate_df[sc].idxmax()]
        base_col = sc + '_base'
        park_str = ""
        if filters['use_park'] and base_col in slate_df.columns and row.get(base_col, 0) != 0:
            delta    = row[sc] - row[base_col]
            pct      = delta / row[base_col] * 100
            sign     = "+" if delta >= 0 else ""
            park_str = f" · park {sign}{pct:.0f}%"

        cards_html += f"""
<div class="scard {css}">
  <div class="sc-type"><span>{icon} {short}</span> — {desc}</div>
  <div class="sc-name">{row['Batter']}</div>
  <div class="sc-meta">{row['Team']} vs {row['Pitcher']}{park_str}</div>
  <div class="sc-score">{row[sc]:.1f}</div>
</div>"""

    cards_html += '</div>'
    st.markdown(cards_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — PITCHER LANDSCAPE  (compact)
# ─────────────────────────────────────────────────────────────────────────────

def render_pitcher_landscape(pitcher_df, df: pd.DataFrame):
    with st.expander("⚾ Pitcher Landscape", expanded=False):
        if pitcher_df is None or pitcher_df.empty:
            st.markdown(
                '<div class="notice notice-info">ℹ️ Pitcher CSV data unavailable — '
                'scores use BallPark Pal matchup probabilities only.</div>',
                unsafe_allow_html=True
            )
            return

        today_pitchers = df['Pitcher'].unique()
        pm = pitcher_df.set_index('last_name')
        rows_html = ""

        for p in sorted(today_pitchers):
            if p in pm.index:
                r = pm.loc[p]
                grade_h = grade_pill(str(r['pitch_grade']))
                hit_val = f"{r['hit8_prob']:.1f}%"
                hr_val  = f"{r['hr2_prob']:.1f}%"
                wk_val  = f"{r['walk3_prob']:.1f}%"
                hm_val  = f"{r['pitch_hit_mult']:.3f}×"
                hrm_val = f"{r['pitch_hr_mult']:.3f}×"
                name    = r['full_name']
                team    = r['team']
            else:
                grade_h = grade_pill('B')
                hit_val = f"{CONFIG['pitcher_hit_neutral']:.1f}% *"
                hr_val  = f"{CONFIG['pitcher_hr_neutral']:.1f}% *"
                wk_val  = f"{CONFIG['pitcher_walk_neutral']:.1f}% *"
                hm_val  = "1.000×"
                hrm_val = "1.000×"
                name    = p
                team    = "—"

            rows_html += f"""<tr>
<td style="color:var(--text);font-family:Outfit,sans-serif">{name}</td>
<td style="color:var(--muted);font-family:Outfit,sans-serif">{team}</td>
<td>{grade_h}</td>
<td style="color:var(--hit)">{hit_val}</td>
<td style="color:var(--hr)">{hr_val}</td>
<td style="color:var(--xb)">{wk_val}</td>
<td style="color:var(--muted)">{hm_val}</td>
<td style="color:var(--muted)">{hrm_val}</td>
</tr>"""

        table_html = f"""
<div class="pt-wrap">
<table class="pt-table">
<thead><tr>
  <th>Pitcher</th><th>Team</th><th>Grade</th>
  <th>Hit 8+</th><th>HR 2+</th><th>Walk 3+</th>
  <th>Hit Mult</th><th>HR Mult</th>
</tr></thead>
<tbody>{rows_html}</tbody>
</table>
</div>
<div class="notice notice-pitcher" style="margin-top:.5rem">
  📊 <b>Hit 8+</b> drives Hit/Single/XB multiplier · <b>HR 2+</b> drives HR multiplier ·
  <b>Walk 3+</b> mild penalty all scores · Max effect ±5%
</div>"""
        st.markdown(table_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — PARK NOTICE  (compact inline)
# ─────────────────────────────────────────────────────────────────────────────

def render_park_notice(slate_df: pd.DataFrame, filters: dict):
    sc       = filters['score_col']
    base_sc  = sc + '_base'
    use_park = filters['use_park']

    if not use_park:
        st.markdown(
            '<div class="notice notice-park">🏟️ <b>Park OFF</b> — pure player vs pitcher, no environment.</div>',
            unsafe_allow_html=True
        )
        return

    if base_sc not in slate_df.columns or slate_df.empty:
        return

    ab, bl  = slate_df[base_sc].mean(), slate_df[sc].mean()
    delta   = bl - ab
    pct     = (delta / ab * 100) if ab != 0 else 0
    dir_    = "boosted" if delta >= 0 else "reduced"
    col_    = "var(--pos)" if delta >= 0 else "var(--neg)"

    st.markdown(
        f'<div class="notice notice-park">🏟️ <b>Park ON</b> — park factors '
        f'<span style="color:{col_};font-weight:700">{dir_} scores ~{abs(pct):.1f}%</span> avg · '
        f'Park Δ column shows per-player impact</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — RESULTS TABLE
# ─────────────────────────────────────────────────────────────────────────────

def render_results_table(filtered_df: pd.DataFrame, filters: dict):
    if filtered_df.empty:
        st.warning("⚠️ No players match your filters — try relaxing the thresholds.")
        return

    sc, base_sc = filters['score_col'], filters['score_col'] + '_base'
    use_park    = filters['use_park']
    disp        = filtered_df.copy()

    disp['K% vs Lg']   = (CONFIG['league_k_avg']  - disp['p_k']).round(1)
    disp['BB% vs Lg']  = (CONFIG['league_bb_avg'] - disp['p_bb']).round(1)
    disp['HR% vs Lg']  = (disp['p_hr'] - CONFIG['league_hr_avg']).round(2)
    disp['Hit Prob %'] = disp['total_hit_prob'].round(1)
    disp['Hist PA']    = disp['PA'].astype(int)
    disp['AVG vs P']   = disp['AVG'].round(3)
    disp['vs Grade']   = pd.to_numeric(disp['vs Grade'], errors='coerce').round(0).astype(int)
    disp['Park Δ']     = (
        (disp[sc] - disp[base_sc]).round(1)
        if (use_park and base_sc in disp.columns) else 0.0
    )

    lbl_map = {'Hit_Score':'🎯 Hit','Single_Score':'1️⃣ Single','XB_Score':'🔥 XB','HR_Score':'💣 HR'}
    active  = lbl_map.get(sc, 'Score')

    cols_map = {
        'Batter': 'Batter', 'Team': 'Team', 'Pitcher': 'Pitcher',
        'pitch_grade': 'P.Grade', sc: active,
    }
    if use_park and base_sc in disp.columns:
        cols_map[base_sc]  = f'Base'
        cols_map['Park Δ'] = 'Park Δ'

    for sc2, lb2 in lbl_map.items():
        if sc2 != sc and sc2 in disp.columns:
            cols_map[sc2] = lb2

    cols_map.update({
        'Hit Prob %': 'Hit%',
        'p_1b':       '1B%',
        'p_xb':       'XB%',
        'p_hr':       'HR%',
        'p_k':        'K%',
        'p_bb':       'BB%',
        'K% vs Lg':   'K Δ',
        'BB% vs Lg':  'BB Δ',
        'HR% vs Lg':  'HR Δ',
        'vs Grade':   'vsPit',
        'Hist PA':    'PA',
        'AVG vs P':   'AVG',
    })

    existing = [c for c in cols_map if c in disp.columns]
    out_df   = disp[existing].rename(columns=cols_map)

    fmt = {}
    for cn in out_df.columns:
        if cn in ['Hit%','1B%','XB%','HR%','K%','BB%']:
            fmt[cn] = "{:.1f}%"
        elif cn in ['K Δ','BB Δ']:
            fmt[cn] = "{:+.1f}%"
        elif cn == 'HR Δ':
            fmt[cn] = "{:+.2f}%"
        elif cn in ['Park Δ']:
            fmt[cn] = "{:+.1f}"
        elif cn == 'AVG':
            fmt[cn] = "{:.3f}"
        elif any(e in cn for e in ['🎯','1️⃣','🔥','💣','Base']) and 'Prob' not in cn:
            fmt[cn] = "{:.1f}"

    styled = out_df.style.format(fmt, na_rep="—")

    cmap_map = {'🎯 Hit':'Greens','1️⃣ Single':'GnBu','🔥 XB':'YlOrBr','💣 HR':'YlOrRd'}
    for sn, cm in cmap_map.items():
        if sn in out_df.columns:
            try:
                styled = styled.background_gradient(subset=[sn], cmap=cm, vmin=0, vmax=100)
            except Exception:
                pass
    for col_name, cmap, vmin, vmax in [
        ('Park Δ', 'RdYlGn', -10, 10),
        ('K Δ',   'RdYlGn',  -8, 12),
        ('HR Δ',  'RdYlGn',  -2,  3),
        ('vsPit', 'RdYlGn', -10, 10),
    ]:
        if col_name in out_df.columns:
            try:
                styled = styled.background_gradient(subset=[col_name], cmap=cmap, vmin=vmin, vmax=vmax)
            except Exception:
                pass

    if 'P.Grade' in out_df.columns:
        styled = styled.apply(
            lambda x: [style_grade_cell(v) if x.name == 'P.Grade' else '' for v in x],
            axis=0
        )

    st.dataframe(styled, use_container_width=True)

    LG = CONFIG
    park_note = (
        f"<b>Park Δ</b> = score impact of park factors (+= park helped). "
        f"Toggle park OFF for pure base scores."
        if use_park
        else "Park OFF — scores use pure player vs pitcher probabilities."
    )
    st.markdown(f"""
<div class="legend-compact">
  <span class="hit-c">🎯 Hit</span> any hit · 1B×3 · heavy K pen ·&nbsp;&nbsp;
  <span class="sl-c">1️⃣ Single</span> 1B×5 · XB/HR penalised ·&nbsp;&nbsp;
  <span class="xb-c">🔥 XB</span> XB×5 · mod K pen ·&nbsp;&nbsp;
  <span class="hr-c">💣 HR</span> HR×6 · light K pen<br>
  <b>P.Grade</b> pitcher grade A+→D (±5% multiplier) ·
  <b>K Δ / BB Δ</b> vs league avg (+ = better) ·
  <b>vsPit</b> BallPark Pal batter vs pitcher rating ·
  <b>PA/AVG</b> history vs this pitcher (bonus at ≥{LG['hist_min_pa']} PA)<br>
  {park_note}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — BEST PER TARGET  (compact player cards — full slate)
# ─────────────────────────────────────────────────────────────────────────────

def render_best_per_target(slate_df: pd.DataFrame, filters: dict):
    if len(slate_df) < 3:
        return

    with st.expander("🔍 Best Per Target — Full Slate", expanded=True):
        st.markdown(
            '<div class="notice notice-info" style="margin-bottom:.5rem">'
            'ℹ️ Drawn from full slate (player exclusions only). '
            'Does not change when you switch betting targets.</div>',
            unsafe_allow_html=True
        )

        score_defs = [
            ('Hit_Score',    'pcard-hit',    '🎯', 'HIT',    'Any Base Hit'),
            ('Single_Score', 'pcard-single', '1️⃣', 'SINGLE', 'Single'),
            ('XB_Score',     'pcard-xb',     '🔥', 'XB',     'Double / Triple'),
            ('HR_Score',     'pcard-hr',     '💣', 'HR',     'Home Run'),
        ]

        cards_html = '<div class="pcard-grid">'
        LG = CONFIG

        for sc, css, icon, short, desc in score_defs:
            if sc not in slate_df.columns:
                continue
            row      = slate_df.loc[slate_df[sc].idxmax()]
            base_col = sc + '_base'

            park_row = ""
            if filters['use_park'] and base_col in slate_df.columns and row.get(base_col, 0) != 0:
                delta    = row[sc] - row[base_col]
                pct      = delta / row[base_col] * 100
                sign     = "+" if delta >= 0 else ""
                col      = "var(--pos)" if delta >= 0 else "var(--neg)"
                park_row = f'<div class="pcard-row"><span class="pk">Park Δ</span><span class="pv" style="color:{col}">{sign}{pct:.1f}%</span></div>'

            k_lg  = LG['league_k_avg']  - row['p_k']
            bb_lg = LG['league_bb_avg'] - row['p_bb']
            hr_lg = row['p_hr'] - LG['league_hr_avg']
            grade_h = grade_pill(str(row.get('pitch_grade', 'B')))

            hist_row = ""
            if row['PA'] >= LG['hist_min_pa']:
                hist_row = f'<div class="pcard-row"><span class="pk">Hist</span><span class="pv">{int(row["PA"])}PA · {row["AVG"]:.3f}</span></div>'

            k_col  = "pos-val" if k_lg  >= 0 else "neg-val"
            bb_col = "pos-val" if bb_lg >= 0 else "neg-val"
            hr_col = "pos-val" if hr_lg >= 0 else "neg-val"

            cards_html += f"""
<div class="pcard {css}">
  <div class="pcard-header">
    <div>
      <div class="pcard-name">{row['Batter']}</div>
      <div class="pcard-team">{row['Team']} · {icon} {desc}</div>
    </div>
    <div class="pcard-score">{row[sc]:.1f}</div>
  </div>
  <div class="pcard-row"><span class="pk">Pitcher</span><span class="pv">{row['Pitcher']} {grade_h}</span></div>
  <div class="pcard-row"><span class="pk">Hit Prob</span><span class="pv">{row['total_hit_prob']:.1f}%</span></div>
  <div class="pcard-row"><span class="pk">1B / XB / HR</span><span class="pv">{row['p_1b']:.1f} / {row['p_xb']:.1f} / {row['p_hr']:.1f}%</span></div>
  <div class="pcard-row"><span class="pk">K%</span><span class="pv">{row['p_k']:.1f}% <span class="{k_col}">({k_lg:+.1f})</span></span></div>
  <div class="pcard-row"><span class="pk">BB%</span><span class="pv">{row['p_bb']:.1f}% <span class="{bb_col}">({bb_lg:+.1f})</span></span></div>
  <div class="pcard-row"><span class="pk">HR vs Lg</span><span class="pv {hr_col}">{hr_lg:+.2f}%</span></div>
  <div class="pcard-row"><span class="pk">vs Grade</span><span class="pv">{int(row['vs Grade'])}</span></div>
  {park_row}
  {hist_row}
</div>"""

        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — VISUALIZATIONS  (collapsed by default)
# ─────────────────────────────────────────────────────────────────────────────

def render_visualizations(df: pd.DataFrame, filtered_df: pd.DataFrame, score_col: str):
    with st.expander("📈 Charts & Team Summary", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            ch = alt.Chart(df).mark_bar(color='#3b82f6', opacity=0.75).encode(
                alt.X(f'{score_col}:Q', bin=alt.Bin(maxbins=15), title='Score'),
                alt.Y('count()', title='Players'),
                tooltip=['count()']
            ).properties(
                title=alt.TitleParams(f'{score_col} Distribution', color='#94a3b8', fontSize=11),
                width=300, height=220
            ).configure_view(strokeWidth=0).configure_axis(
                gridColor='#1e2d3d', domainColor='#1e2d3d',
                labelColor='#64748b', titleColor='#64748b', labelFontSize=10
            )
            st.altair_chart(ch, use_container_width=True)

        with c2:
            if not filtered_df.empty:
                ch2 = alt.Chart(filtered_df).mark_circle(size=80, opacity=0.8).encode(
                    alt.X('total_hit_prob:Q', title='Hit Prob %'),
                    alt.Y('p_k:Q',           title='K Prob %'),
                    alt.Color(f'{score_col}:Q', scale=alt.Scale(scheme='viridis')),
                    alt.Size('p_hr:Q',         legend=None),
                    tooltip=['Batter','Team', alt.Tooltip(score_col, format='.1f'),
                             'total_hit_prob','p_k','p_hr','pitch_grade']
                ).properties(
                    title=alt.TitleParams('Hit Prob vs K Risk', color='#94a3b8', fontSize=11),
                    width=300, height=220
                ).configure_view(strokeWidth=0).configure_axis(
                    gridColor='#1e2d3d', domainColor='#1e2d3d',
                    labelColor='#64748b', titleColor='#64748b', labelFontSize=10
                )
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
                alt.X('Batter:N', sort='-y', axis=alt.Axis(labelAngle=-45, labelFontSize=9)),
                alt.Y('Score:Q', axis=alt.Axis(labelFontSize=9)),
                alt.Color('Score Type:N', scale=alt.Scale(
                    domain=['🎯 Hit','1️⃣ Single','🔥 XB','💣 HR'],
                    range=['#10b981','#06b6d4','#f59e0b','#ef4444']
                )),
                alt.Column('Score Type:N', title=None,
                           header=alt.Header(labelFontSize=9, labelColor='#94a3b8')),
                tooltip=['Batter','Score Type', alt.Tooltip('Score:Q', format='.1f')]
            ).properties(width=160, height=200).configure_view(strokeWidth=0)
            st.altair_chart(ch3, use_container_width=True)

        if not filtered_df.empty:
            ts = filtered_df.groupby('Team').agg(
                Players     =('Batter',        'count'),
                AvgHitProb  =('total_hit_prob', 'mean'),
                AvgHitScore =('Hit_Score',      'mean'),
                AvgXBScore  =('XB_Score',       'mean'),
                AvgHRScore  =('HR_Score',       'mean'),
            ).round(1).sort_values('AvgHitProb', ascending=False).reset_index()
            ts.columns = ['Team','Players','Avg Hit%','🎯 Hit','🔥 XB','💣 HR']
            st.dataframe(ts, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def main_page():
    render_header()

    if 'data_loaded_at' not in st.session_state:
        st.session_state.data_loaded_at = datetime.now()

    with st.spinner("⚾ Loading today's matchups…"):
        raw_df     = load_matchups()
        pitcher_df = load_pitcher_data()

    if raw_df is None:
        st.error("❌ Could not load Matchups data. Check connection or refresh.")
        return

    filters  = build_filters(raw_df)
    df       = compute_metrics(raw_df, use_park=filters['use_park'])
    df       = merge_pitcher_data(df, pitcher_df)
    df       = compute_scores(df)
    slate_df = get_slate_df(df, filters)

    render_stat_bar(df)
    render_pitcher_landscape(pitcher_df, df)
    render_park_notice(slate_df, filters)
    render_score_summary_cards(slate_df, filters)

    # ── Filtered results ─────────────────────────────────────────────────────
    filtered_df = apply_filters(df, filters)
    sc = filters['score_col']
    target_labels = {
        'Hit_Score':    '🎯 Any Base Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 Extra Base Hit',
        'HR_Score':     '💣 Home Run',
    }
    st.markdown(f"""
<div class="result-head">
  <span class="rh-label">{target_labels.get(sc,'Hit')} Candidates</span>
  <span class="rh-count">{len(filtered_df)} results</span>
</div>
""", unsafe_allow_html=True)

    render_results_table(filtered_df, filters)
    render_best_per_target(slate_df, filters)

    if not filtered_df.empty:
        render_visualizations(df, filtered_df, sc)

    # ── Controls ─────────────────────────────────────────────────────────────
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
                "💾 Export CSV",
                filtered_df.to_csv(index=False),
                f"a1picks_mlb_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )
    with c3:
        sb = staleness_badge()
        if sb:
            st.markdown(sb, unsafe_allow_html=True)

    # ── Quick exclusions ──────────────────────────────────────────────────────
    if not filtered_df.empty:
        with st.expander("⚡ Quick Exclusions"):
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
                st.success("✅ All current results confirmed playing.")
            if st.session_state.excluded_players:
                st.caption(f"Excluded: {', '.join(st.session_state.excluded_players)}")
                if st.button("🔄 Re-include All", key="main_clear"):
                    st.session_state.excluded_players = []
                    st.rerun()

    # ── Strategy guide (collapsed) ────────────────────────────────────────────
    LG = CONFIG
    with st.expander("📖 Scoring & Strategy Guide"):
        st.markdown(f"""
**Score → Bet mapping**

| Score | Bet | K% | Primary Driver | Pitcher Signal |
|-------|-----|----|----------------|----------------|
| 🎯 Hit | Any base hit | Heavy | 1B×3 + XB×2 + HR×1 | Hit 8+ Prob |
| 1️⃣ Single | Single (XB/HR penalised) | Heavy | 1B×5 | Hit 8+ Prob |
| 🔥 XB | Double / Triple | Moderate | XB×5 | Hit+HR blend |
| 💣 HR | Home run | **Light** | HR×6 + XB Boost | HR 2+ Prob |

**Pitcher grades** — game-level stats, ±5% max multiplier. A+ = tailwind. D = headwind.

**Park toggle** — ON blends park+base probabilities. Park Δ shows per-player impact.

**vs Grade** — on-paper batter vs pitcher rating. Weighted normally for Hit/Single/XB.
For HR: weight is reduced (−10 grade players still hit HRs frequently).

**League baselines (4yr)** — K% {LG['league_k_avg']}% · BB% {LG['league_bb_avg']}% · HR% {LG['league_hr_avg']}% · AVG {LG['league_avg']}

**V4.1** · Pitcher game layer · HR Score tuned · Stable summary cards
""")


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def info_page():
    st.title("📚 A1PICKS MLB Hit Predictor — Reference")
    with st.expander("System Overview", expanded=True):
        LG = CONFIG
        st.markdown(f"""
## Purpose
Extract the best betting targets from BallPark Pal's 3,000-simulation-per-game data.

## Data Sources
| File | Purpose |
|------|---------|
| `Matchups.csv` | Main batter vs pitcher probabilities (with-park + no-park) |
| `pitcher_hits.csv` | Pitcher allows 8+ hits probability |
| `pitcher_hrs.csv` | Pitcher allows 2+ HRs probability |
| `pitcher_walks.csv` | Pitcher issues 3+ walks probability |

## Four Scores
| Score | Formula summary | K% Tolerance |
|-------|----------------|-------------|
| 🎯 Hit | 1B×3 + XB×2 + HR×1 − K×2.5 − BB×1 | Low |
| 1️⃣ Single | 1B×5 − K×2.5 − BB×1 − XB×0.8 − HR×0.5 | Low |
| 🔥 XB | XB×5 + HR×0.8 − K×1.5 − BB×1 | Moderate |
| 💣 HR | HR×6 + XB×0.8 − K×0.8 − BB×1 + XBBoost×0.03 | Light |

Each score multiplied by pitcher game-level modifier before 0–100 normalisation.

## HR Score Notes
- vs Grade weight reduced to 0.5 (analysis showed 7/16 HR hitters had negative grades)
- XB Boost added as soft power signal (max ±1.8 raw pts before normalisation)

## League Baselines
K% **{LG['league_k_avg']}%** · BB% **{LG['league_bb_avg']}%** · HR% **{LG['league_hr_avg']}%** · AVG **{LG['league_avg']}**
""")


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.markdown("---")
    if st.sidebar.checkbox("🎵 Music"):
        audio_url = (
            "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/"
            "Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        )
        components.html(
            f'<audio controls autoplay loop style="width:100%;">'
            f'<source src="{audio_url}" type="audio/mpeg"></audio>',
            height=55
        )

    page = st.sidebar.radio("Navigate", ["⚾ Predictor", "📚 Reference"], index=0)

    if page == "⚾ Predictor":
        main_page()
    else:
        info_page()

    st.sidebar.markdown("---")
    st.sidebar.caption("V4.1 · Dark Precision · 4 Scores")


if __name__ == "__main__":
    main()
