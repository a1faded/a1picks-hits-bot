"""
A1PICKS MLB Hit Predictor  —  V4.2
====================================
Scores  : Hit · Single · XB · HR  (each 0–100, normalised)
Data    : BallPark Pal Matchups.csv (single file, park + no-park)
Pitcher : game-level multiplier ±5% from pitcher_hits/hrs/walks CSVs
Park    : blend mode (toggle) with per-player Park Δ column
Parlay  : 2/3/4-leg + SGP builder with correlation awareness
"""

import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime, timezone
import itertools

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

REPO          = "a1faded/a1picks-hits-bot"
REPO_RAW      = f"https://github.com/{REPO}/raw/main"
REPO_API      = f"https://api.github.com/repos/{REPO}/commits"

CONFIG = {
    # ── Main data ──────────────────────────────────────────────────────────────
    'matchups_url':        f"{REPO_RAW}/Matchups.csv",
    # ── Pitcher CSVs ──────────────────────────────────────────────────────────
    'pitcher_hits_url':    f"{REPO_RAW}/pitcher_hits.csv",
    'pitcher_hrs_url':     f"{REPO_RAW}/pitcher_hrs.csv",
    'pitcher_walks_url':   f"{REPO_RAW}/pitcher_walks.csv",
    # ── Game & pitcher context CSVs ───────────────────────────────────────────
    'pitcher_qs_url':       f"{REPO_RAW}/pitcher_quality_start.csv",
    'game_4hr_url':         f"{REPO_RAW}/game_4plusHR.csv",
    'game_20hits_url':      f"{REPO_RAW}/game_20plushits.csv",
    'game_20k_url':         f"{REPO_RAW}/game_20plusK.csv",
    'game_8walks_url':      f"{REPO_RAW}/game_8pluswalks.csv",
    'game_10runs_url':      f"{REPO_RAW}/game_10plusruns.csv",
    # ── Game conditions multiplier anchors (median across all games) ──────────
    'gc_hr4_anchor':        12.2,
    'gc_hits20_anchor':     18.6,
    'gc_k20_anchor':        23.3,
    'gc_walks8_anchor':     46.5,
    'gc_runs10_anchor':     28.4,
    'gc_qs_anchor':         21.5,
    'gc_max_mult':           0.07,
    'gc_cap':                0.12,
    # ── Cache ─────────────────────────────────────────────────────────────────
    'cache_ttl':           900,     # 15 min
    # ── Historical tiebreaker ─────────────────────────────────────────────────
    'hist_min_pa':         10,
    'hist_bonus_max':      3.0,
    # ── Pitcher multiplier anchors ────────────────────────────────────────────
    'pitcher_hit_neutral': 2.8,
    'pitcher_hr_neutral':  12.0,
    'pitcher_walk_neutral':18.0,
    'pitcher_max_mult':    0.05,
    # ── League averages (4-year stable) ──────────────────────────────────────
    'league_k_avg':        22.8,
    'league_bb_avg':        8.6,
    'league_hr_avg':        3.15,
    'league_avg':           0.2445,
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS  —  Dark Precision Theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;600&display=swap');
:root{
  --bg:#080c14;--surf:#0f1623;--surf2:#161e2e;--border:#1e2d3d;
  --text:#e2e8f0;--muted:#64748b;
  --hit:#10b981;--single:#06b6d4;--xb:#f59e0b;--hr:#ef4444;
  --accent:#3b82f6;--pos:#22c55e;--neg:#ef4444;
}
html,body,[class*="css"]{font-family:'Outfit',sans-serif !important;}
.block-container{padding:.75rem 1rem 2rem !important;max-width:1400px;}
[data-testid="stSidebar"]{background:var(--surf) !important;border-right:1px solid var(--border);}
[data-testid="stSidebar"] *{font-size:.85rem !important;}
h1,h2,h3,h4,h5{font-family:'Outfit',sans-serif !important;}
/* stat bar */
.stat-bar{display:flex;flex-wrap:wrap;gap:.4rem;background:var(--surf);
  border:1px solid var(--border);border-radius:10px;padding:.6rem 1rem;
  margin-bottom:.75rem;align-items:center;}
.stat-item{display:flex;flex-direction:column;align-items:center;
  padding:.3rem .75rem;border-right:1px solid var(--border);min-width:80px;}
.stat-item:last-child{border-right:none;}
.stat-item .val{font-family:'JetBrains Mono',monospace;font-size:1.15rem;
  font-weight:600;color:var(--text);line-height:1;}
.stat-item .lbl{font-size:.65rem;color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;margin-top:.2rem;}
/* score summary cards */
.score-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin:.5rem 0;}
@media(max-width:768px){
  .score-grid{grid-template-columns:repeat(2,1fr);}
  .pcard-grid{grid-template-columns:1fr 1fr;}
  .stat-item{min-width:60px;padding:.3rem .4rem;}
  .block-container{padding:.4rem .5rem 2rem !important;}
  .parlay-grid{grid-template-columns:1fr !important;}
}
.scard{background:var(--surf);border:1px solid var(--border);border-radius:10px;
  padding:.7rem .85rem;position:relative;overflow:hidden;transition:border-color .2s;}
.scard:hover{border-color:var(--accent);}
.scard::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;}
.scard-hit::before{background:var(--hit);}
.scard-single::before{background:var(--single);}
.scard-xb::before{background:var(--xb);}
.scard-hr::before{background:var(--hr);}
.scard .sc-type{font-size:.65rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:.25rem;}
.scard .sc-type span{font-weight:700;}
.scard-hit .sc-type span{color:var(--hit);}
.scard-single .sc-type span{color:var(--single);}
.scard-xb .sc-type span{color:var(--xb);}
.scard-hr .sc-type span{color:var(--hr);}
.scard .sc-name{font-size:1rem;font-weight:700;color:var(--text);
  white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.scard .sc-meta{font-family:'JetBrains Mono',monospace;font-size:.7rem;
  color:var(--muted);margin-top:.2rem;}
.scard .sc-score{font-family:'JetBrains Mono',monospace;font-size:.8rem;
  font-weight:600;position:absolute;top:.6rem;right:.7rem;}
.scard-hit .sc-score{color:var(--hit);}
.scard-single .sc-score{color:var(--single);}
.scard-xb .sc-score{color:var(--xb);}
.scard-hr .sc-score{color:var(--hr);}
/* player detail cards */
.pcard-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.5rem;margin:.5rem 0;}
.pcard{background:var(--surf);border:1px solid var(--border);border-radius:10px;
  padding:.75rem;font-size:.8rem;}
.pcard-header{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:.5rem;padding-bottom:.4rem;border-bottom:1px solid var(--border);}
.pcard-name{font-weight:700;color:var(--text);font-size:.9rem;}
.pcard-team{font-size:.65rem;color:var(--muted);margin-top:.1rem;}
.pcard-score{font-family:'JetBrains Mono',monospace;font-size:1.1rem;font-weight:700;}
.pcard-hit .pcard-score{color:var(--hit);}
.pcard-single .pcard-score{color:var(--single);}
.pcard-xb .pcard-score{color:var(--xb);}
.pcard-hr .pcard-score{color:var(--hr);}
.pcard-row{display:flex;justify-content:space-between;padding:.18rem 0;
  border-bottom:1px solid var(--border);color:var(--muted);}
.pcard-row:last-child{border-bottom:none;}
.pcard-row .pk{font-size:.7rem;}
.pcard-row .pv{font-family:'JetBrains Mono',monospace;font-size:.72rem;
  color:var(--text);font-weight:600;}
.pos-val{color:var(--pos) !important;}
.neg-val{color:var(--neg) !important;}
/* notices */
.notice{border-left:3px solid;padding:.4rem .8rem;border-radius:0 6px 6px 0;
  font-size:.78rem;margin:.4rem 0;line-height:1.5;}
.notice-park{background:#1a1a2e;border-color:var(--xb);color:#fde68a;}
.notice-pitcher{background:#0d1a14;border-color:var(--hit);color:#a7f3d0;}
.notice-info{background:#0f1623;border-color:var(--accent);color:#bfdbfe;}
.notice-warn{background:#1c1400;border-color:var(--xb);color:#fde68a;}
/* grade pills */
.gp{padding:1px 7px;border-radius:20px;font-size:.7rem;font-weight:700;display:inline-block;}
.gp-ap{background:#1a9641;color:white;}
.gp-a{background:#a6d96a;color:#111;}
.gp-b{background:#fef08a;color:#111;}
.gp-c{background:#fdae61;color:#111;}
.gp-d{background:#ef4444;color:white;}
/* legend */
.legend-compact{background:var(--surf);border:1px solid var(--border);border-radius:8px;
  padding:.6rem 1rem;font-size:.76rem;color:var(--muted);line-height:1.7;margin:.5rem 0;}
.legend-compact b{color:var(--text);}
.legend-compact .hit-c{color:var(--hit);}
.legend-compact .xb-c{color:var(--xb);}
.legend-compact .hr-c{color:var(--hr);}
.legend-compact .sl-c{color:var(--single);}
/* staleness */
.sbadge{display:inline-flex;align-items:center;gap:.3rem;padding:.25rem .65rem;
  border-radius:20px;font-size:.73rem;font-weight:600;font-family:'JetBrains Mono',monospace;}
.sbadge-green{background:#052e16;color:#4ade80;border:1px solid #166534;}
.sbadge-yellow{background:#1c1400;color:#fbbf24;border:1px solid #713f12;}
.sbadge-red{background:#1c0000;color:#f87171;border:1px solid #7f1d1d;}
/* header */
.app-header{display:flex;align-items:center;gap:1rem;padding:.4rem 0 .6rem;
  border-bottom:1px solid var(--border);margin-bottom:.75rem;flex-wrap:wrap;}
.app-header .title-wrap h1{font-size:1.15rem !important;font-weight:700;
  color:var(--text);margin:0 !important;line-height:1.2;}
.app-header .title-wrap p{font-size:.68rem;color:var(--muted);margin:.1rem 0 0;}
.app-header .meta{margin-left:auto;display:flex;align-items:center;gap:.5rem;}
/* section head */
.section-head{font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;
  color:var(--muted);margin:.9rem 0 .4rem;display:flex;align-items:center;gap:.4rem;}
.section-head::after{content:'';flex:1;height:1px;background:var(--border);}
/* result head */
.result-head{display:flex;align-items:center;gap:.5rem;margin:.5rem 0 .3rem;}
.result-head .rh-label{font-size:.85rem;font-weight:600;color:var(--text);}
.result-head .rh-count{background:var(--surf2);border:1px solid var(--border);
  border-radius:20px;padding:.1rem .55rem;font-family:'JetBrains Mono',monospace;
  font-size:.72rem;color:var(--muted);}
/* table */
[data-testid="stDataFrame"] th{font-size:.72rem !important;
  font-family:'Outfit',sans-serif !important;text-transform:uppercase;letter-spacing:.04em;}
[data-testid="stDataFrame"] td{font-family:'JetBrains Mono',monospace !important;
  font-size:.78rem !important;}
/* pitcher table */
.pt-wrap{overflow-x:auto;}
.pt-table{width:100%;border-collapse:collapse;font-size:.78rem;}
.pt-table th{background:var(--surf2);color:var(--muted);text-transform:uppercase;
  letter-spacing:.05em;font-size:.65rem;padding:.45rem .7rem;
  border-bottom:1px solid var(--border);white-space:nowrap;}
.pt-table td{padding:.4rem .7rem;border-bottom:1px solid var(--border);color:var(--text);
  font-family:'JetBrains Mono',monospace;white-space:nowrap;}
.pt-table tr:last-child td{border-bottom:none;}
.pt-table tr:hover td{background:var(--surf2);}
/* parlay builder */
.parlay-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:.75rem;margin:.75rem 0;}
.parlay-leg{background:var(--surf);border:1px solid var(--border);border-radius:10px;
  padding:.75rem;position:relative;}
.parlay-leg .leg-num{position:absolute;top:.5rem;right:.6rem;font-family:'JetBrains Mono',monospace;
  font-size:.65rem;color:var(--muted);}
.parlay-leg .leg-batter{font-weight:700;color:var(--text);font-size:.9rem;}
.parlay-leg .leg-meta{font-size:.7rem;color:var(--muted);margin-top:.15rem;}
.parlay-leg .leg-score{font-family:'JetBrains Mono',monospace;font-size:1.05rem;
  font-weight:700;margin-top:.3rem;}
.parlay-summary{background:var(--surf2);border:1px solid var(--border);border-radius:10px;
  padding:.9rem 1.1rem;margin:.75rem 0;}
.parlay-summary .ps-title{font-size:.7rem;text-transform:uppercase;letter-spacing:.08em;
  color:var(--muted);margin-bottom:.5rem;}
.parlay-summary .ps-conf{font-family:'JetBrains Mono',monospace;font-size:1.6rem;
  font-weight:700;color:var(--accent);}
.parlay-summary .ps-sub{font-size:.72rem;color:var(--muted);margin-top:.2rem;}
/* ref guide sections */
.ref-section{background:var(--surf);border:1px solid var(--border);border-radius:10px;
  padding:1rem 1.2rem;margin:.6rem 0;}
.ref-section h3{font-size:.9rem;font-weight:700;color:var(--text);margin:0 0 .5rem;}
.ref-section p,.ref-section li{font-size:.82rem;color:var(--muted);line-height:1.7;}
.ref-section b{color:var(--text);}
.ref-section .col-pill{display:inline-block;padding:1px 8px;border-radius:20px;
  font-family:'JetBrains Mono',monospace;font-size:.7rem;font-weight:600;
  margin:1px 2px;}
.pill-hit{background:#052e16;color:#4ade80;}
.pill-single{background:#083344;color:#67e8f9;}
.pill-xb{background:#1c1400;color:#fbbf24;}
.pill-hr{background:#1c0000;color:#fca5a5;}
.pill-neutral{background:#1e2d3d;color:#94a3b8;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def normalize_0_100(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(1)


def grade_pill(grade: str) -> str:
    css = {'A+':'gp-ap','A':'gp-a','B':'gp-b','C':'gp-c','D':'gp-d'}.get(str(grade),'gp-b')
    return f'<span class="gp {css}">{grade}</span>'


def style_grade_cell(val):
    return {
        'A+': 'background-color:#1a9641;color:white;font-weight:700',
        'A':  'background-color:#a6d96a;color:#111;font-weight:700',
        'B':  'background-color:#fef08a;color:#111',
        'C':  'background-color:#fdae61;color:#111',
        'D':  'background-color:#ef4444;color:white;font-weight:700',
    }.get(str(val), '')


def _pv(val: str, cls: str = '') -> str:
    return f'<span class="pv {cls}">{val}</span>'


@st.cache_data(ttl=300)
def get_last_commit_time(path: str) -> str:
    """
    Fetch the timestamp of the last GitHub commit that touched `path`.
    Returns a human-readable string like '7 minutes ago' or 'today at 14:32'.
    Falls back gracefully if the API is unavailable or rate-limited.
    """
    try:
        resp = requests.get(
            REPO_API,
            params={'path': path, 'per_page': 1},
            headers={'Accept': 'application/vnd.github+json'},
            timeout=8
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        commit_time_str = data[0]['commit']['committer']['date']  # ISO 8601 UTC
        commit_dt = datetime.fromisoformat(commit_time_str.replace('Z', '+00:00'))
        now_utc   = datetime.now(timezone.utc)
        diff_sec  = int((now_utc - commit_dt).total_seconds())
        if diff_sec < 60:
            return "just now"
        elif diff_sec < 3600:
            mins = diff_sec // 60
            return f"{mins}m ago"
        elif diff_sec < 86400:
            hrs = diff_sec // 3600
            return f"{hrs}h ago"
        else:
            days = diff_sec // 86400
            return f"{days}d ago"
    except Exception:
        return None


def data_freshness_badge() -> str:
    age = get_last_commit_time('Matchups.csv')
    if age is None:
        return '<span class="sbadge sbadge-yellow">⏱ Updated: unknown</span>'
    # colour by recency
    if 'just' in age or (age.endswith('m ago') and int(age.replace('m ago','')) < 60):
        css = 'sbadge-green'
        icon = '🟢'
    elif age.endswith('h ago') and int(age.replace('h ago','')) < 6:
        css = 'sbadge-yellow'
        icon = '🟡'
    else:
        css = 'sbadge-red'
        icon = '🔴'
    return f'<span class="sbadge {css}">{icon} Data: {age}</span>'


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
            errors='coerce').fillna(0))
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
    merged['pitch_hit_mult']  = (1.0 + np.clip((merged['hit8_prob']  - CONFIG['pitcher_hit_neutral'])  / 4.0 * M, -M, M)).round(4)
    merged['pitch_hr_mult']   = (1.0 + np.clip((merged['hr2_prob']   - CONFIG['pitcher_hr_neutral'])   / 8.0 * M, -M, M)).round(4)
    merged['pitch_walk_pen']  = (0.0 - np.clip((merged['walk3_prob'] - CONFIG['pitcher_walk_neutral']) / 10.0 * (M*0.5), -(M*0.5), (M*0.5))).round(4)

    composite = merged['pitch_hit_mult'] + merged['pitch_walk_pen']
    merged['pitch_grade'] = np.select(
        [composite>=1.04, composite>=1.01, composite>=0.98, composite>=0.95],
        ['A+','A','B','C'], default='D'
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
    def _g(p, col, default):
        return pm.at[p, col] if p in pm.index else default
    df['pitch_hit_mult']  = df['Pitcher'].apply(lambda p: _g(p,'pitch_hit_mult',1.0))
    df['pitch_hr_mult']   = df['Pitcher'].apply(lambda p: _g(p,'pitch_hr_mult',1.0))
    df['pitch_walk_pen']  = df['Pitcher'].apply(lambda p: _g(p,'pitch_walk_pen',0.0))
    df['pitch_grade']     = df['Pitcher'].apply(lambda p: _g(p,'pitch_grade','B'))
    df['hit8_prob']       = df['Pitcher'].apply(lambda p: _g(p,'hit8_prob',CONFIG['pitcher_hit_neutral']))
    df['hr2_prob']        = df['Pitcher'].apply(lambda p: _g(p,'hr2_prob',CONFIG['pitcher_hr_neutral']))
    df['walk3_prob']      = df['Pitcher'].apply(lambda p: _g(p,'walk3_prob',CONFIG['pitcher_walk_neutral']))
    return df


# ─────────────────────────────────────────────────────────────────────────────
# GAME CONDITIONS DATA
# ─────────────────────────────────────────────────────────────────────────────

# ── Park name → home team abbreviation (all 30 MLB parks) ─────────────────────
PARK_TO_TEAM = {
    # AL East
    'Fenway Park':          'BOS', 'Camden Yards':        'BAL',
    'Oriole Park':          'BAL', 'Yankee Stadium':      'NYY',
    'Citi Field':           'NYM', 'Rogers Centre':       'TOR',
    'Tropicana Field':      'TB',
    # AL Central
    'Guaranteed Rate Fld':  'CWS', 'Progressive Field':   'CLE',
    'Comerica Park':        'DET', 'Kauffman Stadium':    'KC',
    'Target Field':         'MIN',
    # AL West
    'Daikin Park':          'HOU', 'Minute Maid Park':    'HOU',
    'Angel Stadium':        'LAA', 'Oakland Coliseum':    'ATH',
    'T-Mobile Park':        'SEA', 'Globe Life Field':    'TEX',
    # NL East
    'Truist Park':          'ATL', 'LoanDepot Park':      'MIA',
    'Marlins Park':         'MIA', 'Citizens Bank Park':  'PHI',
    'Nationals Park':       'WSH',
    # NL Central
    'Wrigley Field':        'CHC', 'Great American BP':   'CIN',
    'American Family Fld':  'MIL', 'PNC Park':            'PIT',
    'Busch Stadium':        'STL',
    # NL West
    'Chase Field':          'ARI', 'Coors Field':         'COL',
    'Dodger Stadium':       'LAD', 'Petco Park':          'SD',
    'Oracle Park':          'SF',
}

# ── Team nickname (as it appears in Matchups 'Game' column) → abbreviation ────
NICK_TO_ABBR = {
    'Red Sox':'BOS','Yankees':'NYY','Rays':'TB','Orioles':'BAL','Blue Jays':'TOR',
    'White Sox':'CWS','Guardians':'CLE','Tigers':'DET','Royals':'KC','Twins':'MIN',
    'Astros':'HOU','Angels':'LAA','Athletics':'ATH','Mariners':'SEA','Rangers':'TEX',
    'Braves':'ATL','Marlins':'MIA','Mets':'NYM','Phillies':'PHI','Nationals':'WSH',
    'Cubs':'CHC','Reds':'CIN','Brewers':'MIL','Pirates':'PIT','Cardinals':'STL',
    'Diamondbacks':'ARI','Rockies':'COL','Dodgers':'LAD','Padres':'SD','Giants':'SF',
}


def _clean_prob_col(series: pd.Series) -> pd.Series:
    """Strip % and convert to float."""
    return pd.to_numeric(
        series.astype(str).str.replace('%', '', regex=False).str.strip(),
        errors='coerce'
    ).fillna(0)


@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_game_conditions():
    """
    Load all five game-level prop CSVs.
    Each CSV has columns: Time, Park, Prob
    Park = full stadium name (e.g. 'T-Mobile Park')
    Returns a merged DataFrame with one row per home-team abbreviation,
    or None if all files are missing.
    """
    files = {
        'hr4_prob':    CONFIG['game_4hr_url'],
        'hits20_prob': CONFIG['game_20hits_url'],
        'k20_prob':    CONFIG['game_20k_url'],
        'walks8_prob': CONFIG['game_8walks_url'],
        'runs10_prob': CONFIG['game_10runs_url'],
    }

    frames = {}
    for col_name, url in files.items():
        df = _fetch_csv(url, col_name)
        if df is not None and 'Park' in df.columns and 'Prob' in df.columns:
            df = df.copy()
            df['home_team'] = df['Park'].astype(str).str.strip().map(PARK_TO_TEAM)
            df[col_name]    = _clean_prob_col(df['Prob'])
            frames[col_name] = df[['home_team', col_name]].dropna(subset=['home_team'])

    if not frames:
        return None

    # Merge all on home_team
    merged = None
    for col_name, frame in frames.items():
        if merged is None:
            merged = frame
        else:
            merged = merged.merge(frame, on='home_team', how='outer')

    # Fill any gaps with neutral anchors
    defaults = {
        'hr4_prob':    CONFIG['gc_hr4_anchor'],
        'hits20_prob': CONFIG['gc_hits20_anchor'],
        'k20_prob':    CONFIG['gc_k20_anchor'],
        'walks8_prob': CONFIG['gc_walks8_anchor'],
        'runs10_prob': CONFIG['gc_runs10_anchor'],
    }
    for col, default in defaults.items():
        if col in merged.columns:
            merged[col] = merged[col].fillna(default)
        else:
            merged[col] = default

    return merged.reset_index(drop=True)


@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_pitcher_qs():
    """
    Load pitcher quality start CSV.
    Columns: Team, Name, Park (home team 3-letter abbr), Prob
    Returns merged frame with last_name + home_team + qs_prob, or None.
    """
    df = _fetch_csv(CONFIG['pitcher_qs_url'], "Pitcher QS")
    if df is None:
        return None
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    if 'Prob' not in df.columns or 'Name' not in df.columns:
        return None
    df['qs_prob']   = _clean_prob_col(df['Prob'])
    df['last_name'] = df['Name'].astype(str).str.split().str[-1]
    df['home_team'] = df['Park'].astype(str).str.strip() if 'Park' in df.columns else ''
    return df[['last_name','home_team','qs_prob']].reset_index(drop=True)


def merge_game_conditions(df: pd.DataFrame, game_cond, pitcher_qs) -> pd.DataFrame:
    """
    Join game-level conditions and pitcher QS data onto the matchups frame.

    Matching logic:
      - Parse home team from Matchups 'Game' column (team after '@')
      - Map home team nickname → 3-letter abbreviation (NICK_TO_ABBR)
      - Join game conditions on home_team abbreviation
      - Both visiting AND home batters get the same game conditions
      - Join QS on pitcher last name

    All columns default to neutral anchor values when no match found,
    so scores are unaffected if data is missing or from a different slate.
    """
    df = df.copy()

    # Parse home team abbreviation from Game column
    df['_home_nick'] = df['Game'].astype(str).str.split(' @ ').str[-1].str.strip()
    df['_home_abbr'] = df['_home_nick'].map(NICK_TO_ABBR).fillna('')

    # Default game condition columns (neutral — no effect)
    defaults = {
        'gc_hr4':    CONFIG['gc_hr4_anchor'],
        'gc_hits20': CONFIG['gc_hits20_anchor'],
        'gc_k20':    CONFIG['gc_k20_anchor'],
        'gc_walks8': CONFIG['gc_walks8_anchor'],
        'gc_runs10': CONFIG['gc_runs10_anchor'],
    }
    for col, default in defaults.items():
        df[col] = default

    if game_cond is not None and not game_cond.empty:
        gmap = game_cond.set_index('home_team')
        col_map = {
            'hr4_prob':    'gc_hr4',
            'hits20_prob': 'gc_hits20',
            'k20_prob':    'gc_k20',
            'walks8_prob': 'gc_walks8',
            'runs10_prob': 'gc_runs10',
        }
        for src_col, dst_col in col_map.items():
            if src_col in gmap.columns:
                df[dst_col] = df['_home_abbr'].map(gmap[src_col]).fillna(defaults[dst_col])

    # QS: join on pitcher last name
    df['gc_qs'] = CONFIG['gc_qs_anchor']
    if pitcher_qs is not None and not pitcher_qs.empty:
        qs_map = pitcher_qs.set_index('last_name')['qs_prob']
        df['gc_qs'] = df['Pitcher'].map(qs_map).fillna(CONFIG['gc_qs_anchor'])

    df.drop(columns=['_home_nick', '_home_abbr'], inplace=True, errors='ignore')
    return df


def compute_game_condition_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute game-conditions-adjusted score variants (*_gc columns).

    Hybrid approach — each condition affects only the scores it's relevant to:
      gc_hr4    → HR Score only    (power-friendly game)
      gc_hits20 → Hit/Single/XB   (contact-friendly game)
      gc_runs10 → all four scores  (high-run game = everyone benefits)
      gc_k20    → all four (drag)  (K-heavy = pitching dominant)
      gc_walks8 → all four (drag)  (walk-heavy = fewer hit ABs)
      gc_qs     → all four (drag)  (dominant starting pitcher)

    All multipliers: linear interpolation from neutral anchor, capped ±gc_max_mult each,
    combined total capped ±gc_cap so stacking can't run wild.
    """
    df = df.copy()
    M   = CONFIG['gc_max_mult']   # ±7% per condition
    CAP = CONFIG['gc_cap']        # ±12% combined cap

    def _mult(series, anchor, rng, direction=1):
        """Linear multiplier: neutral at anchor, ±M at anchor±rng."""
        return np.clip((series - anchor) / rng * M * direction, -M, M)

    # ── Individual condition deltas ────────────────────────────────────────────
    # Positive delta = boost, negative = drag
    d_hr4    = _mult(df['gc_hr4'],    CONFIG['gc_hr4_anchor'],    15.0,  +1)  # high = good for HR
    d_hits20 = _mult(df['gc_hits20'], CONFIG['gc_hits20_anchor'], 12.0,  +1)  # high = good for hits
    d_runs10 = _mult(df['gc_runs10'], CONFIG['gc_runs10_anchor'], 20.0,  +1)  # high = good all
    d_k20    = _mult(df['gc_k20'],    CONFIG['gc_k20_anchor'],    20.0,  -1)  # high = bad all
    d_walks8 = _mult(df['gc_walks8'], CONFIG['gc_walks8_anchor'], 15.0,  -1)  # high = bad all
    d_qs     = _mult(df['gc_qs'],     CONFIG['gc_qs_anchor'],     15.0,  -1)  # high = bad all

    # ── Compose per-score multipliers and apply combined cap ───────────────────
    def _apply_gc(base_scores_raw, score_col, conditions_delta):
        """
        Apply combined game conditions delta to already-normalised scores.
        We re-scale: score_gc = score * (1 + combined_delta)
        Then renormalise to 0–100.
        """
        combined = np.clip(conditions_delta.sum(axis=1), -CAP, CAP)
        adjusted = df[score_col] * (1 + combined)
        return normalize_0_100(adjusted)

    # Hit Score: hits20 + runs10 + k20 + walks8 + qs (not hr4)
    hit_cond = pd.DataFrame({'h': d_hits20, 'r': d_runs10, 'k': d_k20, 'w': d_walks8, 'q': d_qs})
    df['Hit_Score_gc'] = _apply_gc(None, 'Hit_Score', hit_cond)

    # Single Score: same as Hit (hits20 is the key signal; power-game conditions don't apply)
    df['Single_Score_gc'] = _apply_gc(None, 'Single_Score', hit_cond)

    # XB Score: hits20 + runs10 + k20 + walks8 + qs
    df['XB_Score_gc'] = _apply_gc(None, 'XB_Score', hit_cond)

    # HR Score: hr4 + runs10 + k20 + walks8 + qs (hits20 NOT included — contact env ≠ HR env)
    hr_cond = pd.DataFrame({'h': d_hr4, 'r': d_runs10, 'k': d_k20, 'w': d_walks8, 'q': d_qs})
    df['HR_Score_gc'] = _apply_gc(None, 'HR_Score', hr_cond)

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

    df['vs_mod']     = pd.to_numeric(df['vs Grade'], errors='coerce').fillna(0).clip(-10,10) / 10
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

    df['PA']   = pd.to_numeric(df['PA'],          errors='coerce').fillna(0)
    df['H']    = pd.to_numeric(df['H'],           errors='coerce').fillna(0)
    df['AVG']  = pd.to_numeric(df['AVG'],         errors='coerce').fillna(0)

    # ── Historical matchup adjustment ──────────────────────────────────────────
    # Three cases:
    #   PA = 0          → zero effect (no data, completely neutral)
    #   PA ≥ 3, H = 0   → graduated penalty (faced pitcher multiple times, never got a hit)
    #                      A walk in those PAs doesn't help us — we're betting on hits.
    #                      Scales with PA so Keibert Ruiz (5 PA, 0 H) ≠ Mike Yastrzemski (11 PA, 0 H).
    #                      Capped at -5 raw pts so it informs without dominating.
    #   PA ≥ hist_min_pa, H > 0 → positive bonus proportional to batting average vs this pitcher

    # Penalty: -min(PA/10 * 5, 5) raw pts when PA ≥ 3 and H == 0
    zero_hit_penalty = np.where(
        (df['PA'] >= 3) & (df['H'] == 0),
        -np.clip(df['PA'] / 10.0 * 5.0, 1.5, 5.0),
        0.0
    )

    # Positive bonus: AVG × 3.0, only when sample is meaningful AND batter has hits
    pos_bonus = np.where(
        (df['PA'] >= CONFIG['hist_min_pa']) & (df['H'] > 0),
        (df['AVG'] * CONFIG['hist_bonus_max']).round(3),
        0.0
    )

    # Combined: penalty and bonus are mutually exclusive (can't have H=0 and H>0 simultaneously)
    df['hist_bonus'] = (zero_hit_penalty + pos_bonus).round(3)
    df['Starter']        = pd.to_numeric(df.get('Starter', 0), errors='coerce').fillna(0).astype(int)
    df['total_hit_prob'] = (df['p_1b'] + df['p_xb'] + df['p_hr']).clip(upper=100).round(1)

    # XB Boost — used in HR Score as hard-contact signal
    xb_boost_park = pd.to_numeric(df['XB Boost'],           errors='coerce').fillna(0) \
                    if 'XB Boost' in df.columns else pd.Series(0.0, index=df.index)
    xb_boost_base = pd.to_numeric(df['XB Boost (no park)'], errors='coerce').fillna(0) \
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

    hit_mult    = (df['pitch_hit_mult']  + df['pitch_walk_pen']).clip(0.90, 1.10)
    xb_mult     = ((df['pitch_hit_mult'] + df['pitch_hr_mult']) / 2 + df['pitch_walk_pen']).clip(0.90, 1.10)
    hr_mult     = (df['pitch_hr_mult']   + df['pitch_walk_pen']).clip(0.90, 1.10)
    single_mult = (df['pitch_hit_mult']  + df['pitch_walk_pen']).clip(0.90, 1.10)

    # 🎯 Hit
    hit_raw    = (df['p_1b']*3.0 + df['p_xb']*2.0 + df['p_hr']*1.0 - df['p_k']*2.5 - df['p_bb']*1.0 + vc*1.0 + rc*0.5 + hb) * hit_mult
    df['Hit_Score'] = normalize_0_100(hit_raw)

    # 1️⃣ Single
    single_raw = (df['p_1b']*5.0 - df['p_k']*2.5 - df['p_bb']*1.0 - df['p_xb']*0.8 - df['p_hr']*0.5 + vc*0.8 + rc*0.4 + hb) * single_mult
    df['Single_Score'] = normalize_0_100(single_raw)

    # 🔥 XB
    xb_raw     = (df['p_xb']*5.0 + df['p_hr']*0.8 - df['p_k']*1.5 - df['p_bb']*1.0 + vc*1.2 + rc*0.6 + hb) * xb_mult
    df['XB_Score'] = normalize_0_100(xb_raw)

    # 💣 HR  (vs Grade weight reduced 1.5→0.5; XB Boost added as hard-contact signal)
    hr_raw     = (df['p_hr']*6.0 + df['p_xb']*0.8 - df['p_k']*0.8 - df['p_bb']*1.0 + df['xb_boost']*0.03 + vc*0.5 + rc*0.5 + hb) * hr_mult
    df['HR_Score'] = normalize_0_100(hr_raw)

    # Base (no-park) versions for Park Δ
    df['Hit_Score_base']    = normalize_0_100((df['p_1b_base']*3.0 + df['p_xb_base']*2.0 + df['p_hr_base']*1.0 - df['p_k_base']*2.5 - df['p_bb_base']*1.0 + vc*1.0 + rc*0.5 + hb) * hit_mult)
    df['Single_Score_base'] = normalize_0_100((df['p_1b_base']*5.0 - df['p_k_base']*2.5 - df['p_bb_base']*1.0 - df['p_xb_base']*0.8 - df['p_hr_base']*0.5 + vc*0.8 + rc*0.4 + hb) * single_mult)
    df['XB_Score_base']     = normalize_0_100((df['p_xb_base']*5.0 + df['p_hr_base']*0.8 - df['p_k_base']*1.5 - df['p_bb_base']*1.0 + vc*1.2 + rc*0.6 + hb) * xb_mult)
    df['HR_Score_base']     = normalize_0_100((df['p_hr_base']*6.0 + df['p_xb_base']*0.8 - df['p_k_base']*0.8 - df['p_bb_base']*1.0 + df['xb_boost']*0.03 + vc*0.5 + rc*0.5 + hb) * hr_mult)

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
    filters['score_col_base'] = filters['score_col']  # always the non-gc base

    st.sidebar.markdown("### 🏟️ Park Adjustment")
    filters['use_park'] = st.sidebar.toggle(
        "Include Park Factors", value=True,
        help="ON = blends park-adjusted + base probabilities.\nOFF = pure player vs pitcher only."
    )

    st.sidebar.markdown("### 🌦️ Game Conditions")
    filters['use_gc'] = st.sidebar.toggle(
        "Apply Game Conditions",
        value=False,
        help=(
            "ON → Applies game-level multipliers from BallPark Pal game props:\n"
            "• 4+ HR Prob boosts HR Score\n"
            "• 20+ Hits Prob boosts Hit/Single/XB Scores\n"
            "• 10+ Runs Prob mild boost all scores\n"
            "• 20+ Ks Prob mild drag all scores\n"
            "• 8+ Walks Prob mild drag all scores\n"
            "• QS% mild drag all scores\n"
            "Max effect ±12% combined. Requires game CSVs in repo."
        )
    )
    if filters['use_gc']:
        st.sidebar.markdown(
            '<small style="color:#64748b">Cond Δ column shows per-player impact</small>',
            unsafe_allow_html=True
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
    filters['min_vs']       = st.sidebar.slider("Min vs Grade", -10, 10, -10, 1)

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
        "Score (High→Low)":         (filters['score_col'], False),
        "Hit Prob % (High→Low)":    ("total_hit_prob",     False),
        "1B Prob % (High→Low)":     ("p_1b",               False),
        "XB Prob % (High→Low)":     ("p_xb",               False),
        "HR Prob % (High→Low)":     ("p_hr",               False),
        "K Prob % (Low→High)":      ("p_k",                True),
        "BB Prob % (Low→High)":     ("p_bb",               True),
        "vs Grade (High→Low)":      ("vs Grade",           False),
        "Pitcher Grade (A+→D)":     ("pitch_grade",        True),
    }
    filters['sort_label'] = st.sidebar.selectbox("Sort By", list(sort_options.keys()))
    filters['sort_col'], filters['sort_asc'] = sort_options[filters['sort_label']]
    filters['result_count'] = st.sidebar.selectbox("Show Top N", [5,10,15,20,25,30,"All"], index=2)
    filters['best_per_team'] = st.sidebar.checkbox("🏟️ Best player per team only", value=False)
    return filters


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
        st.info(f"🏟️ Best player from each of {len(out)} teams")

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
    badge = data_freshness_badge()
    st.markdown(f"""
<div class="app-header">
  <div>
    <img src="https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true"
         style="height:38px;width:auto;" />
  </div>
  <div class="title-wrap">
    <h1>A1PICKS MLB Hit Predictor</h1>
    <p>BallPark Pal simulation data → betting targets · V4.2</p>
  </div>
  <div class="meta">{badge}</div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — STAT BAR
# ─────────────────────────────────────────────────────────────────────────────

def render_stat_bar(df: pd.DataFrame):
    if df is None or df.empty:
        return
    avg_hp = df['total_hit_prob'].mean()
    avg_hr = df['p_hr'].mean() if 'p_hr' in df.columns else 0
    st.markdown(f"""
<div class="stat-bar">
  <div class="stat-item"><span class="val">{len(df)}</span><span class="lbl">Matchups</span></div>
  <div class="stat-item"><span class="val">{df['Batter'].nunique()}</span><span class="lbl">Batters</span></div>
  <div class="stat-item"><span class="val">{df['Team'].nunique()}</span><span class="lbl">Teams</span></div>
  <div class="stat-item"><span class="val" style="color:var(--hit)">{avg_hp:.1f}%</span><span class="lbl">Avg Hit Prob</span></div>
  <div class="stat-item"><span class="val" style="color:var(--hr)">{avg_hr:.2f}%</span><span class="lbl">Avg HR Prob</span></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — SCORE SUMMARY CARDS  (full slate, never changes with filter)
# ─────────────────────────────────────────────────────────────────────────────

def render_score_summary_cards(slate_df: pd.DataFrame, filters: dict):
    if slate_df.empty:
        return
    st.markdown('<div class="section-head">🏆 Today\'s Best — Full Slate</div>',
                unsafe_allow_html=True)
    defs = [
        ('Hit_Score',    'scard-hit',    '🎯', 'HIT',    'Any Base Hit'),
        ('Single_Score', 'scard-single', '1️⃣', 'SINGLE', 'Single Specifically'),
        ('XB_Score',     'scard-xb',     '🔥', 'XB',     'Double / Triple'),
        ('HR_Score',     'scard-hr',     '💣', 'HR',     'Home Run'),
    ]
    cards_html = '<div class="score-grid">'
    for sc, css, icon, short, desc in defs:
        if sc not in slate_df.columns:
            continue
        row      = slate_df.loc[slate_df[sc].idxmax()]
        base_col = sc + '_base'
        park_str = ""
        if filters['use_park'] and base_col in slate_df.columns and row.get(base_col, 0) != 0:
            delta    = row[sc] - row[base_col]
            pct      = delta / row[base_col] * 100
            park_str = f" · park {'+' if delta>=0 else ''}{pct:.0f}%"
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
# RENDER — PITCHER LANDSCAPE
# ─────────────────────────────────────────────────────────────────────────────

def render_pitcher_landscape(pitcher_df, df: pd.DataFrame):
    with st.expander("⚾ Pitcher Landscape", expanded=False):
        if pitcher_df is None or pitcher_df.empty:
            st.markdown('<div class="notice notice-info">ℹ️ Pitcher CSV data unavailable — scores use BallPark Pal matchup probabilities only.</div>', unsafe_allow_html=True)
            return

        today_pitchers = df['Pitcher'].unique()
        pm = pitcher_df.set_index('last_name')
        rows_html = ""

        for p in sorted(today_pitchers):
            if p in pm.index:
                r = pm.loc[p]
                grade_h  = grade_pill(str(r['pitch_grade']))
                hit_val  = f"{r['hit8_prob']:.1f}%"
                hr_val   = f"{r['hr2_prob']:.1f}%"
                wk_val   = f"{r['walk3_prob']:.1f}%"
                hm_val   = f"{r['pitch_hit_mult']:.3f}×"
                hrm_val  = f"{r['pitch_hr_mult']:.3f}×"
                name, team = r['full_name'], r['team']
            else:
                grade_h  = grade_pill('B')
                hit_val  = f"{CONFIG['pitcher_hit_neutral']:.1f}% *"
                hr_val   = f"{CONFIG['pitcher_hr_neutral']:.1f}% *"
                wk_val   = f"{CONFIG['pitcher_walk_neutral']:.1f}% *"
                hm_val, hrm_val = "1.000×", "1.000×"
                name, team = p, "—"

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

        st.markdown(f"""
<div class="pt-wrap"><table class="pt-table"><thead><tr>
<th>Pitcher</th><th>Team</th><th>Grade</th>
<th>Hit 8+</th><th>HR 2+</th><th>Walk 3+</th>
<th>Hit Mult</th><th>HR Mult</th>
</tr></thead><tbody>{rows_html}</tbody></table></div>
<div class="notice notice-pitcher" style="margin-top:.5rem">
📊 <b>Hit 8+</b> drives Hit/Single/XB multiplier · <b>HR 2+</b> drives HR multiplier ·
<b>Walk 3+</b> mild penalty all scores · Max effect ±5%
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — PARK NOTICE
# ─────────────────────────────────────────────────────────────────────────────

def render_park_notice(slate_df: pd.DataFrame, filters: dict):
    sc, base_sc, use_park = filters['score_col'], filters['score_col']+'_base', filters['use_park']
    if not use_park:
        st.markdown('<div class="notice notice-park">🏟️ <b>Park OFF</b> — pure player vs pitcher, no environment.</div>', unsafe_allow_html=True)
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
        f'<b>Park Δ</b> column shows per-player impact</div>',
        unsafe_allow_html=True
    )


def render_game_conditions_panel(slate_df: pd.DataFrame, filters: dict, game_cond, pitcher_qs):
    """
    Show a compact game conditions table when the toggle is ON.
    Displays each game's environment stats with a visual indicator
    of whether conditions favor or suppress the active bet type.
    """
    use_gc = filters.get('use_gc', False)
    sc     = filters['score_col']

    if not use_gc:
        return

    gc_cols = ['gc_hr4','gc_hits20','gc_k20','gc_walks8','gc_runs10','gc_qs']
    if slate_df.empty or not all(c in slate_df.columns for c in gc_cols):
        st.markdown(
            '<div class="notice notice-warn">🌦️ <b>Game Conditions ON</b> — '
            'No game condition CSVs found in repo yet. Upload '
            '<code>game_4plusHR.csv</code>, <code>game_20plushits.csv</code>, '
            '<code>game_20plusK.csv</code>, <code>game_8pluswalks.csv</code>, '
            '<code>game_10plusruns.csv</code>, <code>pitcher_quality_start.csv</code> '
            'to enable. Scores currently unaffected.</div>',
            unsafe_allow_html=True
        )
        return

    gc_sc_col = sc + '_gc'
    if gc_sc_col not in slate_df.columns:
        return

    # Build per-game summary
    game_rows = []
    for game in sorted(slate_df['Game'].unique()):
        gdf = slate_df[slate_df['Game'] == game].iloc[:1]
        if gdf.empty:
            continue
        row = gdf.iloc[0]
        avg_base = slate_df[slate_df['Game'] == game][sc].mean()
        avg_gc   = slate_df[slate_df['Game'] == game][gc_sc_col].mean()
        delta    = avg_gc - avg_base

        game_rows.append({
            'Game':        game,
            '4+ HR %':     f"{row['gc_hr4']:.1f}%",
            '20+ Hits %':  f"{row['gc_hits20']:.1f}%",
            '20+ Ks %':    f"{row['gc_k20']:.1f}%",
            '8+ Walks %':  f"{row['gc_walks8']:.1f}%",
            '10+ Runs %':  f"{row['gc_runs10']:.1f}%",
            'QS %':        f"{row['gc_qs']:.1f}%",
            'Cond Δ (avg)': delta,
        })

    if not game_rows:
        return

    # Active score label for the impact description
    sc_lbl = {'Hit_Score':'Hit','Single_Score':'Single','XB_Score':'XB','HR_Score':'HR'}.get(sc,'Score')

    with st.expander(f"🌦️ Game Conditions — {sc_lbl} Score Impact", expanded=True):
        gdf_disp = pd.DataFrame(game_rows)

        styled = gdf_disp.style.format({'Cond Δ (avg)': '{:+.1f}'})
        styled = styled.background_gradient(
            subset=['Cond Δ (avg)'], cmap='RdYlGn', vmin=-8, vmax=8
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="notice notice-info" style="font-size:.74rem">'
            '🌦️ <b>Cond Δ</b> = avg score shift for that game after applying game conditions. '
            'Positive (green) = game environment favours your bet type today. '
            'Negative (red) = tighter conditions. Max combined effect ±12%.</div>',
            unsafe_allow_html=True
        )

def render_results_table(filtered_df: pd.DataFrame, filters: dict):
    if filtered_df.empty:
        st.warning("⚠️ No players match your filters — try relaxing the thresholds.")
        return

    use_gc   = filters.get('use_gc', False)
    use_park = filters['use_park']
    sc_base  = filters['score_col']                    # e.g. 'Hit_Score'
    sc_gc    = sc_base + '_gc'                         # e.g. 'Hit_Score_gc'
    base_sc  = sc_base + '_base'                       # e.g. 'Hit_Score_base'

    # Active score: use _gc version when toggle ON and column exists
    sc = sc_gc if (use_gc and sc_gc in filtered_df.columns) else sc_base

    disp = filtered_df.copy()

    disp['K% ↓Lg']   = (CONFIG['league_k_avg']  - disp['p_k']).round(1)
    disp['BB% ↓Lg']  = (CONFIG['league_bb_avg'] - disp['p_bb']).round(1)
    disp['HR% ↑Lg']  = (disp['p_hr'] - CONFIG['league_hr_avg']).round(2)
    disp['Hit%']    = disp['total_hit_prob'].round(1)
    disp['PA']      = disp['PA'].astype(int)
    disp['AVG']     = disp['AVG'].round(3)
    disp['vs Grade']= pd.to_numeric(disp['vs Grade'], errors='coerce').round(0).astype(int)
    disp['Park Δ']  = (disp[sc_base] - disp[base_sc]).round(1) if (use_park and base_sc in disp.columns) else 0.0

    # Cond Δ: difference between GC score and base score
    if use_gc and sc_gc in disp.columns:
        disp['Cond Δ'] = (disp[sc_gc] - disp[sc_base]).round(1)
    else:
        disp['Cond Δ'] = 0.0

    lbl    = {'Hit_Score':'🎯 Hit','Single_Score':'1️⃣ Single','XB_Score':'🔥 XB','HR_Score':'💣 HR'}
    active = lbl.get(sc_base, 'Score')
    active_label = (active + ' ⛅') if (use_gc and sc_gc in filtered_df.columns) else active

    cols = {'Batter':'Batter','Team':'Team','Pitcher':'Pitcher','pitch_grade':'P.Grd', sc: active_label}
    if use_park and base_sc in disp.columns:
        cols[base_sc]  = 'Base'
        cols['Park Δ'] = 'Park Δ'
    if use_gc and sc_gc in disp.columns:
        cols['Cond Δ'] = 'Cond Δ'
    for sc2, lb2 in lbl.items():
        if sc2 != sc_base and sc2 != sc_gc and sc2 in disp.columns:
            cols[sc2] = lb2
    cols.update({'Hit%':'Hit%','p_1b':'1B%','p_xb':'XB%','p_hr':'HR%',
                 'p_k':'K%','p_bb':'BB%','K% ↓Lg':'K% ↓Lg','BB% ↓Lg':'BB% ↓Lg',
                 'HR% ↑Lg':'HR% ↑Lg','vs Grade':'vsPit','PA':'PA','AVG':'AVG'})

    existing = [c for c in cols if c in disp.columns]
    out_df   = disp[existing].rename(columns=cols)

    fmt = {}
    for cn in out_df.columns:
        if cn in ['Hit%','1B%','XB%','HR%','K%','BB%']:
            fmt[cn] = "{:.1f}%"
        elif cn in ['K% ↓Lg','BB% ↓Lg']:
            fmt[cn] = "{:+.1f}%"
        elif cn == 'HR% ↑Lg':
            fmt[cn] = "{:+.2f}%"
        elif cn in ['Park Δ','Cond Δ']:
            fmt[cn] = "{:+.1f}"
        elif cn == 'AVG':
            fmt[cn] = "{:.3f}"
        elif any(e in cn for e in ['🎯','1️⃣','🔥','💣','Base']) and 'Prob' not in cn:
            fmt[cn] = "{:.1f}"

    styled = out_df.style.format(fmt, na_rep="—")

    for sn, cm in {'🎯 Hit':'Greens','1️⃣ Single':'GnBu','🔥 XB':'YlOrBr','💣 HR':'YlOrRd'}.items():
        # Match both normal and ⛅ labelled columns
        target_col = sn + ' ⛅' if (sn + ' ⛅') in out_df.columns else sn
        if target_col in out_df.columns:
            try:
                styled = styled.background_gradient(subset=[target_col], cmap=cm, vmin=0, vmax=100)
            except Exception:
                pass
    for cn, cm, v0, v1 in [('Park Δ','RdYlGn',-10,10),('Cond Δ','RdYlGn',-8,8),
                            ('K% ↓Lg','RdYlGn',-8,12),('HR% ↑Lg','RdYlGn',-2,3),
                            ('vsPit','RdYlGn',-10,10)]:
        if cn in out_df.columns:
            try:
                styled = styled.background_gradient(subset=[cn], cmap=cm, vmin=v0, vmax=v1)
            except Exception:
                pass
    if 'P.Grd' in out_df.columns:
        styled = styled.apply(lambda x: [style_grade_cell(v) if x.name=='P.Grd' else '' for v in x], axis=0)

    st.dataframe(styled, use_container_width=True)

    LG = CONFIG
    park_note = (
        "<b>Park Δ</b> = score impact of park factors (+= park helped). Toggle park OFF for pure base."
        if use_park else "Park OFF — pure player vs pitcher probabilities."
    )
    gc_note = (
        " · <b>Cond Δ ⛅</b> = game conditions score shift (+= game favours this bet type today)"
        if use_gc else ""
    )
    st.markdown(f"""
<div class="legend-compact">
  <span class="hit-c">🎯 Hit</span> any hit · 1B×3 · heavy K pen &nbsp;
  <span class="sl-c">1️⃣ Single</span> 1B×5 · XB/HR penalised &nbsp;
  <span class="xb-c">🔥 XB</span> XB×5 · mod K pen &nbsp;
  <span class="hr-c">💣 HR</span> HR×6 · light K pen<br>
  <b>P.Grd</b> pitcher grade A+→D (±5%) ·
  <b>K% ↓Lg</b> how far this batter's K% is <b>below</b> league ({LG['league_k_avg']}%) — <span style="color:var(--pos)">+ve = strikes out less = better contact</span> ·
  <b>BB% ↓Lg</b> how far BB% is <b>below</b> league ({LG['league_bb_avg']}%) — <span style="color:var(--pos)">+ve = walks less = more aggressive</span> ·
  <b>HR% ↑Lg</b> how far HR% is <b>above</b> league ({LG['league_hr_avg']}%) — <span style="color:var(--pos)">+ve = above avg HR rate</span> ·
  <b>vsPit</b> batter vs pitcher rating (−10 to +10) · <b>PA/AVG</b> history vs this pitcher<br>
  <b>⚠️ hist penalty:</b> batters with ≥3 PA and zero hits vs this pitcher receive a score penalty — they've seen this pitcher before and couldn't get a hit.<br>
  {park_note}{gc_note}
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — BEST PER TARGET  (full slate)
# ─────────────────────────────────────────────────────────────────────────────

def render_best_per_target(slate_df: pd.DataFrame, filters: dict):
    if len(slate_df) < 3:
        return
    with st.expander("🔍 Best Per Target — Full Slate", expanded=True):
        st.markdown('<div class="notice notice-info">ℹ️ Full slate (player exclusions only). Does not change when you switch betting targets.</div>', unsafe_allow_html=True)

        defs = [
            ('Hit_Score',    'pcard-hit',    '🎯', 'HIT',    'Any Base Hit'),
            ('Single_Score', 'pcard-single', '1️⃣', 'SINGLE', 'Single'),
            ('XB_Score',     'pcard-xb',     '🔥', 'XB',     'Double / Triple'),
            ('HR_Score',     'pcard-hr',     '💣', 'HR',     'Home Run'),
        ]
        cards_html = '<div class="pcard-grid">'
        LG = CONFIG

        for sc, css, icon, short, desc in defs:
            if sc not in slate_df.columns:
                continue
            row      = slate_df.loc[slate_df[sc].idxmax()]
            base_col = sc + '_base'
            park_row = ""
            if filters['use_park'] and base_col in slate_df.columns and row.get(base_col, 0) != 0:
                delta  = row[sc] - row[base_col]
                pct    = delta / row[base_col] * 100
                col_   = "var(--pos)" if delta >= 0 else "var(--neg)"
                park_row = f'<div class="pcard-row"><span class="pk">Park Δ</span><span class="pv" style="color:{col_}">{("+" if delta>=0 else "")}{pct:.1f}%</span></div>'

            k_lg  = LG['league_k_avg']  - row['p_k']
            bb_lg = LG['league_bb_avg'] - row['p_bb']
            hr_lg = row['p_hr'] - LG['league_hr_avg']
            k_cls  = "pos-val" if k_lg  >= 0 else "neg-val"
            bb_cls = "pos-val" if bb_lg >= 0 else "neg-val"
            hr_cls = "pos-val" if hr_lg >= 0 else "neg-val"
            gph    = grade_pill(str(row.get('pitch_grade','B')))
            hist_row = ""
            if row['PA'] >= LG['hist_min_pa']:
                hist_row = f'<div class="pcard-row"><span class="pk">Hist PA</span><span class="pv">{int(row["PA"])} PA · {row["AVG"]:.3f}</span></div>'

            cards_html += f"""
<div class="pcard {css}">
  <div class="pcard-header">
    <div><div class="pcard-name">{row['Batter']}</div>
      <div class="pcard-team">{row['Team']} · {icon} {desc}</div></div>
    <div class="pcard-score">{row[sc]:.1f}</div>
  </div>
  <div class="pcard-row"><span class="pk">Pitcher</span><span class="pv">{row['Pitcher']} {gph}</span></div>
  <div class="pcard-row"><span class="pk">Hit Prob</span><span class="pv">{row['total_hit_prob']:.1f}%</span></div>
  <div class="pcard-row"><span class="pk">1B/XB/HR</span><span class="pv">{row['p_1b']:.1f}/{row['p_xb']:.1f}/{row['p_hr']:.1f}%</span></div>
  <div class="pcard-row"><span class="pk">K%</span><span class="pv">{row['p_k']:.1f}% <span class="{k_cls}">({k_lg:+.1f})</span></span></div>
  <div class="pcard-row"><span class="pk">BB%</span><span class="pv">{row['p_bb']:.1f}% <span class="{bb_cls}">({bb_lg:+.1f})</span></span></div>
  <div class="pcard-row"><span class="pk">HR vs Lg</span><span class="pv {hr_cls}">{hr_lg:+.2f}%</span></div>
  <div class="pcard-row"><span class="pk">vs Grade</span><span class="pv">{int(row['vs Grade'])}</span></div>
  {park_row}{hist_row}
</div>"""
        cards_html += '</div>'
        st.markdown(cards_html, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# RENDER — VISUALIZATIONS  (collapsed, 2×2 stacked grid instead of wide Column)
# ─────────────────────────────────────────────────────────────────────────────

def render_visualizations(df: pd.DataFrame, filtered_df: pd.DataFrame, score_col: str):
    with st.expander("📈 Charts & Team Summary", expanded=False):

        # ── Chart theme config ──────────────────────────────────────────────
        axis_cfg = alt.Axis(gridColor='#1e2d3d', domainColor='#1e2d3d',
                            labelColor='#64748b', titleColor='#64748b', labelFontSize=9)
        title_cfg = lambda t: alt.TitleParams(t, color='#94a3b8', fontSize=11)

        # ── Row 1: Score distribution + Hit Prob vs K Risk ──────────────────
        c1, c2 = st.columns(2)
        with c1:
            ch = alt.Chart(df).mark_bar(color='#3b82f6', opacity=0.75).encode(
                alt.X(f'{score_col}:Q', bin=alt.Bin(maxbins=15), title='Score', axis=axis_cfg),
                alt.Y('count()', title='Players', axis=axis_cfg),
                tooltip=['count()']
            ).properties(title=title_cfg(f'{score_col} Distribution'), width=300, height=200)
            st.altair_chart(ch.configure_view(strokeWidth=0), use_container_width=True)

        with c2:
            if not filtered_df.empty:
                ch2 = alt.Chart(filtered_df).mark_circle(size=80, opacity=0.8).encode(
                    alt.X('total_hit_prob:Q', title='Hit Prob %', axis=axis_cfg),
                    alt.Y('p_k:Q',           title='K Prob %',   axis=axis_cfg),
                    alt.Color(f'{score_col}:Q', scale=alt.Scale(scheme='viridis'), legend=None),
                    alt.Size('p_hr:Q', legend=None),
                    tooltip=['Batter','Team', alt.Tooltip(score_col, format='.1f'),
                             'total_hit_prob','p_k','p_hr','pitch_grade']
                ).properties(title=title_cfg('Hit Prob vs K Risk'), width=300, height=200)
                st.altair_chart(ch2.configure_view(strokeWidth=0), use_container_width=True)

        # ── Row 2–3: Per-score bar charts stacked vertically (not Column facet) ──
        if not filtered_df.empty and len(filtered_df) <= 30:
            st.markdown("**Individual Score Breakdowns**")
            score_defs = [
                ('Hit_Score',    '#10b981', '🎯 Hit Score'),
                ('Single_Score', '#06b6d4', '1️⃣ Single Score'),
                ('XB_Score',     '#f59e0b', '🔥 XB Score'),
                ('HR_Score',     '#ef4444', '💣 HR Score'),
            ]
            r1c1, r1c2 = st.columns(2)
            r2c1, r2c2 = st.columns(2)
            target_cols = [r1c1, r1c2, r2c1, r2c2]

            for i, (sc, colour, label) in enumerate(score_defs):
                if sc not in filtered_df.columns:
                    continue
                chart_df = filtered_df[['Batter', sc]].sort_values(sc, ascending=False)
                ch_s = alt.Chart(chart_df).mark_bar(color=colour, opacity=0.85).encode(
                    alt.X('Batter:N', sort='-y',
                          axis=alt.Axis(labelAngle=-45, labelFontSize=8,
                                        labelColor='#64748b', domainColor='#1e2d3d')),
                    alt.Y(f'{sc}:Q', scale=alt.Scale(domain=[0,100]),
                          axis=alt.Axis(labelFontSize=8, labelColor='#64748b',
                                        domainColor='#1e2d3d', gridColor='#1e2d3d',
                                        title='Score')),
                    tooltip=['Batter', alt.Tooltip(f'{sc}:Q', format='.1f', title='Score')]
                ).properties(title=title_cfg(label), width=250, height=180)
                with target_cols[i]:
                    st.altair_chart(ch_s.configure_view(strokeWidth=0), use_container_width=True)

        # ── Team summary ────────────────────────────────────────────────────
        if not filtered_df.empty:
            ts = filtered_df.groupby('Team').agg(
                Players    =('Batter',        'count'),
                AvgHitProb =('total_hit_prob', 'mean'),
                AvgHit     =('Hit_Score',      'mean'),
                AvgXB      =('XB_Score',       'mean'),
                AvgHR      =('HR_Score',       'mean'),
            ).round(1).sort_values('AvgHitProb', ascending=False).reset_index()
            ts.columns = ['Team','Players','Avg Hit%','🎯 Hit','🔥 XB','💣 HR']
            st.markdown("**Team Summary**")
            st.dataframe(ts, use_container_width=True, hide_index=True)



# ─────────────────────────────────────────────────────────────────────────────
# PARLAY BUILDER  —  V2
# ─────────────────────────────────────────────────────────────────────────────

_SCORE_MAP  = {'🎯 Hit':'Hit_Score','1️⃣ Single':'Single_Score',
               '🔥 XB (Double/Triple)':'XB_Score','💣 HR':'HR_Score'}
_LBL_MAP    = {'Hit_Score':'🎯 Hit','Single_Score':'1️⃣ Single',
               'XB_Score':'🔥 XB','HR_Score':'💣 HR'}
_SCORE_CSS  = {'Hit_Score':'var(--hit)','Single_Score':'var(--single)',
               'XB_Score':'var(--xb)','HR_Score':'var(--hr)'}

# Bet-type-aware game condition weights: (hits20, hr4, runs10, k20, walks8)
_GC_WEIGHTS = {
    'Hit_Score':    (1.8, 0.4, 1.0, 1.5, 1.0),
    'Single_Score': (1.8, 0.2, 1.0, 1.5, 1.0),
    'XB_Score':     (1.2, 1.0, 1.0, 1.2, 1.0),
    'HR_Score':     (0.4, 1.8, 1.0, 1.2, 1.0),
}


def _gc_adjusted_score(pool: pd.DataFrame, sc: str) -> pd.Series:
    """Re-compute game-conditions adjustment with bet-type-aware weights."""
    if sc not in pool.columns:
        return pd.Series(50.0, index=pool.index)
    gc_cols = ['gc_hr4','gc_hits20','gc_k20','gc_walks8','gc_runs10','gc_qs']
    if not all(c in pool.columns for c in gc_cols):
        return pool[sc]
    base_sc = sc.replace('_gc','')
    if base_sc not in pool.columns:
        base_sc = sc
    M, CAP = CONFIG['gc_max_mult'], CONFIG['gc_cap']
    hits_w, hr4_w, runs_w, k_w, walk_w = _GC_WEIGHTS.get(base_sc, (1.0,1.0,1.0,1.0,1.0))
    def _d(s, anchor, rng, direction=1):
        return np.clip((s - anchor) / rng * M * direction, -M, M)
    d_hits = _d(pool['gc_hits20'], CONFIG['gc_hits20_anchor'], 12.0, +1) * hits_w
    d_hr4  = _d(pool['gc_hr4'],    CONFIG['gc_hr4_anchor'],   15.0, +1) * hr4_w
    d_runs = _d(pool['gc_runs10'], CONFIG['gc_runs10_anchor'],20.0, +1) * runs_w
    d_k    = _d(pool['gc_k20'],    CONFIG['gc_k20_anchor'],   20.0, -1) * k_w
    d_walk = _d(pool['gc_walks8'], CONFIG['gc_walks8_anchor'],15.0, -1) * walk_w
    d_qs   = _d(pool['gc_qs'],     CONFIG['gc_qs_anchor'],    15.0, -1)
    combined = (d_hits + d_hr4 + d_runs + d_k + d_walk + d_qs).clip(-CAP, CAP)
    raw = pool[base_sc] * (1 + combined)
    mn, mx = raw.min(), raw.max()
    if mx == mn:
        return pd.Series(50.0, index=pool.index)
    return ((raw - mn) / (mx - mn) * 100).round(1)


def _build_all_combos(pool: pd.DataFrame, leg_bets: list, sgp: bool,
                      locked: list, env_filter: bool) -> list:
    """Build ALL valid combos ranked by harmonic mean confidence."""
    legs = len(leg_bets)
    if sgp:
        primary_sc = leg_bets[0]
        ranked     = pool.sort_values(primary_sc, ascending=False)
        candidates = ranked['Batter'].unique().tolist()
        if locked:
            candidates = [p for p in locked if p in candidates] + [p for p in candidates if p not in locked]
        all_combos_raw = list(itertools.combinations(candidates[:min(10, len(candidates))], legs))
        leg_candidates = None
    else:
        leg_candidates = []
        for sc in leg_bets:
            if sc not in pool.columns:
                return []
            gc_sc = _gc_adjusted_score(pool, sc)
            ps = pool.copy()
            ps['_gc_adj'] = gc_sc.values
            if locked:
                ps['_locked'] = ps['Batter'].isin(locked).astype(int)
                ps = ps.sort_values(['_locked','_gc_adj'], ascending=[False,False])
            else:
                ps = ps.sort_values('_gc_adj', ascending=False)
            per_game = ps.drop_duplicates(subset='Game').head(12)
            leg_candidates.append(per_game)
        all_combos_raw = list(itertools.product(*[lc['Batter'].tolist() for lc in leg_candidates]))

    ranked_combos = []
    for combo in all_combos_raw:
        if len(set(combo)) < legs:
            continue
        if not sgp:
            games_in_combo = []
            valid = True
            for batter, cand in zip(combo, leg_candidates):
                row = cand[cand['Batter'] == batter]
                if row.empty:
                    valid = False
                    break
                games_in_combo.append(row.iloc[0]['Game'])
            if not valid or len(set(games_in_combo)) < legs:
                continue
        scores = []
        valid  = True
        for batter, sc in zip(combo, leg_bets):
            row = pool[pool['Batter'] == batter]
            if row.empty or sc not in row.columns:
                valid = False
                break
            scores.append(float(row[sc].values[0]))
        if not valid:
            continue
        conf = (len(scores) / sum(1/s for s in scores if s > 0)) if all(s > 0 for s in scores) else 0
        ranked_combos.append((combo, scores, conf))

    ranked_combos.sort(key=lambda x: x[2], reverse=True)
    return ranked_combos


def parlay_page(df: pd.DataFrame):
    st.title("⚡ Parlay Builder")
    st.markdown(
        '<div class="notice notice-info">ℹ️ Scores are a statistical foundation — not a guarantee. '
        'Parlay risk compounds with each leg. Use as research, not a tip sheet.</div>',
        unsafe_allow_html=True
    )
    if df is None or df.empty:
        st.error("No data loaded.")
        return

    all_batters = sorted(df['Batter'].unique().tolist())
    global_excl = st.session_state.get('excluded_players', [])

    with st.expander("🚫 Exclude Players (Parlay Builder)", expanded=False):
        parlay_excl = st.multiselect(
            "Exclude from parlay candidates",
            options=all_batters,
            default=global_excl,
            help="These exclusions apply only inside the Parlay Builder.",
            key="parlay_exclusions"
        )
    pool = df[~df['Batter'].isin(parlay_excl)].copy()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        parlay_type = st.selectbox("Parlay Type",
            ["Cross-Game", "SGP — Stack (same team)", "SGP — Split (both teams)"])
    with c2:
        legs = st.selectbox("Number of Legs", [2, 3, 4], index=1)
    with c3:
        bet_mode = st.selectbox("Bet Mode",
            ["Same bet on all legs", "Mixed — I'll choose per leg"])
    with c4:
        env_filter = st.toggle("🌦️ Weight Game Conditions", value=True,
            help="Hit bets: 20+ Hits signal weighted 1.8×. HR bets: 4+ HR signal weighted 1.8×.")

    if bet_mode == "Same bet on all legs":
        all_bet  = st.selectbox("Bet Type (all legs)", list(_SCORE_MAP.keys()))
        leg_bets = [_SCORE_MAP[all_bet]] * legs
    else:
        leg_cols = st.columns(legs)
        leg_bets = []
        for i, lc in enumerate(leg_cols):
            with lc:
                choice = st.selectbox(f"Leg {i+1}", list(_SCORE_MAP.keys()), key=f"lb_{i}")
                leg_bets.append(_SCORE_MAP[choice])

    with st.expander("🔒 Lock Players (anchor specific players)", expanded=False):
        st.caption("Locked players are prioritised. Leave empty for fully automatic.")
        max_lock = min(legs - 1, 2)
        locked = st.multiselect(f"Lock up to {max_lock} player(s)", options=all_batters,
                                max_selections=max_lock, key="parlay_locked") if max_lock > 0 else []

    st.markdown("---")

    sgp = parlay_type.startswith("SGP")
    chosen_game = None
    if sgp:
        games = sorted(pool['Game'].unique().tolist())
        if not games:
            st.warning("No games available.")
            return
        chosen_game = st.selectbox("Select Game for SGP", games)
        game_pool = pool[pool['Game'] == chosen_game].copy()
        if parlay_type == "SGP — Stack (same team)":
            primary_sc = leg_bets[0]
            team_avg   = game_pool.groupby('Team')[primary_sc].mean()
            sgp_pool   = game_pool[game_pool['Team'] == team_avg.idxmax()].copy()
        else:
            sgp_pool = game_pool.copy()
        build_pool = sgp_pool
    else:
        build_pool = pool

    cache_key = (f"parlay_{parlay_type}_{legs}_{'-'.join(leg_bets)}_{env_filter}_"
                 f"{'-'.join(sorted(locked))}_{chosen_game or 'cg'}_{'-'.join(sorted(parlay_excl))}")

    if st.session_state.get('parlay_cache_key') != cache_key:
        combos = _build_all_combos(build_pool, leg_bets, sgp, locked, env_filter)
        st.session_state['parlay_combos']    = combos
        st.session_state['parlay_combo_idx'] = 0
        st.session_state['parlay_cache_key'] = cache_key
    else:
        combos = st.session_state.get('parlay_combos', [])

    if not combos:
        st.warning("⚠️ Could not build any valid combinations. Try relaxing exclusions or adding more games.")
        return

    total_combos = min(len(combos), 50)
    idx = min(st.session_state.get('parlay_combo_idx', 0), total_combos - 1)

    nav_c1, nav_c2, nav_c3, nav_c4 = st.columns([2,1,1,2])
    with nav_c1:
        st.markdown(
            f'<div style="font-family:JetBrains Mono,monospace;font-size:.85rem;'
            f'color:var(--muted);padding:.4rem 0">Combo {idx+1} of {total_combos}</div>',
            unsafe_allow_html=True)
    with nav_c2:
        if st.button("◀ Prev", disabled=(idx == 0)):
            st.session_state['parlay_combo_idx'] = max(0, idx - 1)
            st.rerun()
    with nav_c3:
        if st.button("Next ▶", disabled=(idx >= total_combos - 1)):
            st.session_state['parlay_combo_idx'] = min(total_combos - 1, idx + 1)
            st.rerun()
    with nav_c4:
        if st.button("🎲 Random"):
            import random
            st.session_state['parlay_combo_idx'] = random.randint(0, total_combos - 1)
            st.rerun()

    combo_batters, combo_scores, conf = combos[idx]
    _show_parlay_card(combo_batters, combo_scores, leg_bets, conf,
                      _LBL_MAP, _SCORE_CSS, parlay_type, chosen_game, pool, sgp, env_filter)


def _show_parlay_card(combo_batters, combo_scores, leg_bets, conf,
                      lbl_map, score_css, parlay_type, game_label, pool, sgp, env_filter):
    """Render the parlay combo card."""
    LG   = CONFIG
    legs = len(combo_batters)

    if not sgp:
        player_games = [pool[pool['Batter']==b].iloc[0]['Game']
                        for b in combo_batters if not pool[pool['Batter']==b].empty]
        if len(player_games) != len(set(player_games)):
            st.markdown(
                '<div class="notice notice-warn">⚠️ <b>Correlation Warning</b> — '
                'Two or more legs are from the same game. Outcomes are not fully independent.</div>',
                unsafe_allow_html=True)

    conf_lbl, conf_note = (
        ("🟢 Strong", "All legs have solid backing.") if conf >= 70 else
        ("🟡 Moderate", "Most legs solid — check flagged legs.") if conf >= 50 else
        ("🔴 Weak", "One or more legs have limited support.")
    )
    env_note = " · 🌦️ bet-type conditions weighted" if env_filter else ""
    st.markdown(f"""
<div class="parlay-summary">
  <div class="ps-title">{parlay_type} · {legs}-Leg{(' · ' + game_label) if game_label else ''}{env_note}</div>
  <div class="ps-conf">{conf:.1f} <span style="font-size:.8rem;color:var(--muted)">/ 100</span></div>
  <div class="ps-sub">{conf_lbl} — {conf_note}</div>
  <div class="ps-sub" style="font-size:.7rem;margin-top:.2rem">Harmonic mean of leg scores. Weak legs penalised heavily. Not a win probability.</div>
</div>""", unsafe_allow_html=True)

    leg_htmls = ""
    clip_lines = []

    for i, (batter, sc, score) in enumerate(zip(combo_batters, leg_bets, combo_scores)):
        m2 = pool[pool['Batter'] == batter]
        if m2.empty:
            leg_htmls += f'<div class="parlay-leg"><div class="leg-num">Leg {i+1}</div><div class="leg-batter">{batter}</div><div class="leg-meta">Data unavailable</div></div>'
            continue
        row = m2.iloc[0]

        def _s(col, default=0.0):
            v = row.get(col, default)
            try:
                return float(v)
            except Exception:
                return default

        k_lg = LG['league_k_avg'] - _s('p_k')
        hr_lg = _s('p_hr') - LG['league_hr_avg']
        k_cls = "pos-val" if k_lg >= 0 else "neg-val"
        hr_cls = "pos-val" if hr_lg >= 0 else "neg-val"
        col_css = score_css.get(sc, 'var(--accent)')
        lbl = lbl_map.get(sc, sc)
        gph = grade_pill(str(row.get('pitch_grade','B')))
        pa_val = _s('PA')
        hist_row = (f'<div class="pcard-row"><span class="pk">Hist</span>' +
                    f'<span class="pv">{int(pa_val)} PA · {_s("AVG"):.3f}</span></div>') if pa_val >= LG['hist_min_pa'] else ""

        if score >= 70:
            sbadge = '<span style="background:#052e16;color:#4ade80;padding:1px 6px;border-radius:10px;font-size:.65rem;font-weight:700">STRONG</span>'
        elif score >= 50:
            sbadge = '<span style="background:#1c1400;color:#fbbf24;padding:1px 6px;border-radius:10px;font-size:.65rem;font-weight:700">OK</span>'
        else:
            sbadge = '<span style="background:#1c0000;color:#f87171;padding:1px 6px;border-radius:10px;font-size:.65rem;font-weight:700">⚠️ WEAK</span>'

        gc_adj = float(_gc_adjusted_score(pool, sc).loc[m2.index[0]])
        cond_delta = gc_adj - _s(sc)
        cond_str = ""
        if env_filter and abs(cond_delta) >= 0.5:
            cc = "var(--pos)" if cond_delta > 0 else "var(--neg)"
            cond_str = f'<div class="pcard-row"><span class="pk">🌦️ Cond Δ</span><span class="pv" style="color:{cc}">{cond_delta:+.1f}</span></div>'

        leg_htmls += f"""
<div class="parlay-leg">
  <div class="leg-num">Leg {i+1} {sbadge}</div>
  <div class="leg-batter">{batter}</div>
  <div class="leg-meta">{row.get('Team','?')}) vs {row.get('Pitcher','?')} {gph}</div>
  <div class="leg-score" style="color:{col_css}">{lbl} &nbsp; {score:.1f}</div>
  <div style="margin-top:.45rem">
    <div class="pcard-row"><span class="pk">Hit Prob</span><span class="pv">{_s('total_hit_prob'):.1f}%</span></div>
    <div class="pcard-row"><span class="pk">1B/XB/HR</span><span class="pv">{_s('p_1b'):.1f}/{_s('p_xb'):.1f}/{_s('p_hr'):.1f}%</span></div>
    <div class="pcard-row"><span class="pk">K%</span><span class="pv">{_s('p_k'):.1f}% <span class="{k_cls}">({k_lg:+.1f})</span></span></div>
    <div class="pcard-row"><span class="pk">HR vs Lg</span><span class="pv {hr_cls}">{hr_lg:+.2f}%</span></div>
    <div class="pcard-row"><span class="pk">vs Grade</span><span class="pv">{int(_s('vs Grade'))}</span></div>
    {cond_str}{hist_row}
  </div>
</div>"""
        clip_lines.append(f"Leg {i+1}: {batter} ({row.get('Team','?')}) — {lbl} — Score {score:.1f}")

    st.markdown(f'<div class="parlay-grid">{leg_htmls}</div>', unsafe_allow_html=True)

    clip_text = "\n".join(clip_lines) + f"\nConfidence: {conf:.1f}/100"
    st.download_button("📋 Export this Parlay (txt)", clip_text,
                       file_name=f"parlay_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
                       mime="text/plain", key=f"parlay_export_{hash(str(combo_batters))}")

    _render_context_panel(list(combo_batters), pool)


def _render_context_panel(batters, pool):
    """Show game conditions context for the parlay players."""

    gc_cols = ['gc_hr4','gc_hits20','gc_k20','gc_walks8','gc_runs10','gc_qs']
    has_gc  = all(c in pool.columns for c in gc_cols)

    if not has_gc:
        st.markdown("""
<div class="notice notice-warn" style="margin-top:.75rem">
🔜 <b>Game Conditions Context</b> — Upload
<code>game_4plusHR.csv</code>, <code>game_20plushits.csv</code>,
<code>game_20plusK.csv</code>, <code>game_8pluswalks.csv</code>,
<code>game_10plusruns.csv</code>, <code>pitcher_quality_start.csv</code>
to your GitHub repo to see game context for parlay legs.
</div>""", unsafe_allow_html=True)
        return

    # Get unique games for the selected batters
    batter_rows = pool[pool['Batter'].isin(batters)][
        ['Batter','Team','Game'] + gc_cols
    ].drop_duplicates(subset=['Game'])

    if batter_rows.empty:
        return

    display_rows = []
    for _, row in batter_rows.iterrows():
        # Determine which conditions are favourable vs unfavourable
        hr4_flag    = "✅" if row['gc_hr4']    > CONFIG['gc_hr4_anchor']    else "⚠️"
        hits_flag   = "✅" if row['gc_hits20'] > CONFIG['gc_hits20_anchor'] else "⚠️"
        runs_flag   = "✅" if row['gc_runs10'] > CONFIG['gc_runs10_anchor'] else "⚠️"
        k_flag      = "✅" if row['gc_k20']    < CONFIG['gc_k20_anchor']    else "⚠️"
        walks_flag  = "✅" if row['gc_walks8'] < CONFIG['gc_walks8_anchor'] else "⚠️"

        display_rows.append({
            'Game':         row['Game'],
            '4+ HR %':      f"{hr4_flag} {row['gc_hr4']:.1f}%",
            '20+ Hits %':   f"{hits_flag} {row['gc_hits20']:.1f}%",
            '10+ Runs %':   f"{runs_flag} {row['gc_runs10']:.1f}%",
            '20+ Ks %':     f"{k_flag} {row['gc_k20']:.1f}%",
            '8+ Walks %':   f"{walks_flag} {row['gc_walks8']:.1f}%",
        })

    if display_rows:
        st.markdown("**🌦️ Game Environment for Parlay Legs**")
        st.dataframe(pd.DataFrame(display_rows), use_container_width=True, hide_index=True)
        st.markdown(
            '<div class="notice notice-info" style="font-size:.73rem;margin-top:.3rem">'
            '✅ = above median (favourable) · ⚠️ = below median (tighter conditions). '
            'For HR parlays look for ✅ on 4+ HR %. '
            'For Hit/XB parlays look for ✅ on 20+ Hits % and 10+ Runs %. '
            'For all parlays prefer ✅ on 20+ Ks (lower = better contact game).</div>',
            unsafe_allow_html=True
        )


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def info_page():
    st.title("📚 A1PICKS MLB Reference Manual")

    with st.expander("📖 What is this tool?", expanded=True):
        st.markdown("""
This app takes BallPark Pal's daily simulation data (3,000 runs per game before first pitch)
and filters + weights it to surface the best targets for four MLB player prop bet types:
**Base Hit · Single · Double/Triple · Home Run**.

BallPark Pal already accounts for weather, park factors, recent performance, pitcher tendencies,
and head-to-head matchup data. **We are not trying to out-predict BallPark Pal** — we are
extracting and ranking their output specifically for betting use cases.
        """)

    with st.expander("📊 Understanding the Data Columns", expanded=False):
        LG = CONFIG
        st.markdown(f"""
<div class="ref-section">
<h3>Probability Columns</h3>
<p>These come directly from BallPark Pal's 3,000-simulation output:</p>
<ul>
<li><b>1B Prob %</b> — Probability this batter gets a single in today's game</li>
<li><b>XB Prob %</b> — Probability of a double or triple</li>
<li><b>HR Prob %</b> — Probability of a home run</li>
<li><b>BB Prob %</b> — Probability of a walk (not a hit — this penalises scores)</li>
<li><b>K Prob %</b> — Probability of a strikeout (no hit — this penalises scores)</li>
<li><b>Hit%</b> — Combined: 1B% + XB% + HR% — total probability of getting any base hit</li>
</ul>
<p>Each stat has two versions: <b>with park factors</b> (includes dimensions, weather, wind)
and <b>without park factors</b> (pure player vs pitcher). The Park toggle blends both.</p>
</div>

<div class="ref-section">
<h3>Modifier Columns</h3>
<ul>
<li><b>vsPit</b> (vs Grade) — BallPark Pal's on-paper batter vs pitcher rating. −10 to +10.
  Based on pitch mix, location tendencies, and batter characteristics against that pitcher type.
  A +10 means this batter matches up very well against this pitcher's approach.
  <b>Note for HR bets:</b> this is weighted less for HR Score — our data showed 7/16 HR hitters on
  a sample day had negative vs Grades. Power is less dependent on matchup quality.</li>
<li><b>RC</b> (Runs Created) — BallPark Pal's overall matchup quality score. Used as a small
  supporting signal in all four scores.</li>
<li><b>PA / AVG vs P</b> — Historical plate appearances and batting average this batter has vs
  this specific pitcher. Positive bonus when PA ≥ 10 with hits. <b>Penalty</b> when PA ≥ 3 with zero hits — faced pitcher before and never got a hit. {LG['hist_min_pa']}.</li>
</ul>
</div>

<div class="ref-section">
<h3>Park Δ Column</h3>
<p>Shows exactly how many score points the park adjustment added or removed for each player.
A <b style="color:var(--pos)">+3.2</b> means park factors boosted this player's score by 3.2 points
compared to the pure base score. A <b style="color:var(--neg)">-2.1</b> means the park suppressed it.</p>
<p>Toggle <b>Park Factors</b> OFF in the sidebar to see the base scores without any environmental influence.</p>
</div>

<div class="ref-section">
<h3>Pitcher Grade (P.Grd)</h3>
<p>A+ through D based on today's game-level pitcher statistics — how likely is this pitcher to
allow 8+ hits, 2+ HRs, or 3+ walks in the game overall. Applied as a mild ±5% multiplier
to each player's score. Batter-level probabilities always dominate — the pitcher grade is context,
not a primary driver.</p>
</div>

<div class="ref-section">
<h3>K% ↓Lg / BB% ↓Lg / HR% ↑Lg</h3>
<p>How this player compares to the 4-year MLB league average:</p>
<ul>
<li><b>K% ↓Lg</b> — League avg: {LG['league_k_avg']}%. <span style="color:var(--pos)">Positive = better contact</span> (strikes out less than average).
  <span style="color:var(--neg)">Negative = more K risk</span> than average.</li>
<li><b>BB% ↓Lg</b> — League avg: {LG['league_bb_avg']}%. <span style="color:var(--pos)">Positive = more aggressive</span> (walks less — less likely to take a walk instead of a hit attempt).
  <span style="color:var(--neg)">Negative = more patient</span> (draws more walks).</li>
<li><b>HR% ↑Lg</b> — League avg: {LG['league_hr_avg']}%. <span style="color:var(--pos)">Positive = above average HR rate</span>.
  <span style="color:var(--neg)">Negative = below average</span>.</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with st.expander("🎯 The Four Scores — How They Work", expanded=False):
        LG = CONFIG
        st.markdown(f"""
<div class="ref-section">
<h3><span style="color:var(--hit)">🎯 Hit Score</span> — Any Base Hit</h3>
<p><b>Formula:</b> (1B×3.0 + XB×2.0 + HR×1.0 − K×2.5 − BB×1.0) × pitcher_hit_mult</p>
<p><b>Use for:</b> "Player to record a hit" props. The most common MLB player prop bet.</p>
<p><b>Logic:</b> Singles are weighted highest because they're the most common hit type.
XBs and HRs also count — any base hit satisfies this bet. K% carries the heaviest penalty
because a strikeout is the most certain way to not get a hit. BB% has a moderate penalty
because a walk consumes a plate appearance without a hit.</p>
<p><b>Best targets:</b> Players with high 1B%, moderate XB%, and well below-average K%.
Positive K% ↓Lg (green) is a strong signal here — player K% is below league average.</p>
</div>

<div class="ref-section">
<h3><span style="color:var(--single)">1️⃣ Single Score</span> — Single Specifically</h3>
<p><b>Formula:</b> (1B×5.0 − K×2.5 − BB×1.0 − XB×0.8 − HR×0.5) × pitcher_hit_mult</p>
<p><b>Use for:</b> "Player to hit a single" props.</p>
<p><b>Key difference from Hit Score:</b> High XB% and HR% are <b>penalised</b> here.
A batter who tends to make extra-base contact is less likely to have that ball drop in for
a clean single — it leaves the park or rolls to the wall instead. You want a true contact
hitter whose ball tends to stay in play at single-territory velocity.</p>
<p><b>Best targets:</b> High 1B%, low XB%, low HR%, low K%. Pure contact hitters. Positive K% ↓Lg and BB% ↓Lg both matter here — you want both below league average.</p>
</div>

<div class="ref-section">
<h3><span style="color:var(--xb)">🔥 XB Score</span> — Extra Base Hit (Double/Triple)</h3>
<p><b>Formula:</b> (XB×5.0 + HR×0.8 − K×1.5 − BB×1.0) × pitch_xb_mult</p>
<p><b>Use for:</b> "Player to hit a double or triple" props, or total bases overs.</p>
<p><b>Logic:</b> XB Prob dominates. HR% is a supporting signal — same hard-contact swing path
that produces extra bases also produces HRs. K% tolerance is moderate because
XB hitters tend to swing harder and naturally K more than pure contact hitters.
A 25% K rate doesn't eliminate a 7% XB rate.</p>
<p><b>Best targets:</b> High XB%, solid vs Grade, positive HR% ↑Lg as secondary confirmation.
K% is less disqualifying here than for Hit/Single scores.</p>
</div>

<div class="ref-section">
<h3><span style="color:var(--hr)">💣 HR Score</span> — Home Run</h3>
<p><b>Formula:</b> (HR×6.0 + XB×0.8 − K×0.8 − BB×1.0 + XBBoost×0.03) × pitcher_hr_mult</p>
<p><b>Use for:</b> "Player to hit a home run" props (anytime HR).</p>
<p><b>Why the light K% penalty:</b> Power hitters structurally have higher K rates — Stanton,
Judge, Ohtani all K at 25-35%+ but are legitimate HR threats daily. Penalising them the same
as contact hitters would bury them artificially. The K penalty here (×0.8) is less than a
third of the Hit Score penalty (×2.5).</p>
<p><b>Why vs Grade is weighted lightly:</b> Analysis of HR hitters showed 7 of 16 HR hitters
on a sample day had negative vs Grades. Power is less matchup-dependent than contact.
vs Grade still contributes but is not a gatekeeper.</p>
<p><b>XB Boost signal:</b> BallPark Pal's matchup-specific extra base adjustment is used as
a small hard-contact proxy — players like Mookie Betts who project for strong XB matchups
but below-average HR Prob still have legitimate power ceiling.</p>
<p><b>Best targets:</b> High HR%, high XB% as secondary confirmation, don't over-filter on K%.</p>
</div>
""", unsafe_allow_html=True)

    with st.expander("🗺️ How to Navigate the App", expanded=False):
        st.markdown("""
<div class="ref-section">
<h3>Step 1 — Choose Your Betting Target (Sidebar)</h3>
<p>The first sidebar dropdown is the most important decision. Select the score type that matches the bet you're making:</p>
<ul>
<li><b>Hit Score</b> if you're betting "player to record a hit"</li>
<li><b>Single Score</b> if you're betting specifically on a single</li>
<li><b>XB Score</b> for doubles/triples, or total bases overs where you need extra bases</li>
<li><b>HR Score</b> for anytime home run props</li>
</ul>
<p>The results table, sort order, and minimum probability filter will all automatically adjust to your chosen target.</p>
</div>

<div class="ref-section">
<h3>Step 2 — Set Your Filters</h3>
<p><b>Max K Prob %</b> — How much strikeout risk are you willing to accept?</p>
<ul>
<li>For Hit/Single bets: keep this tight (25–30% max). K kills contact bets.</li>
<li>For XB bets: 30–35% is fine. Power contact guys K more.</li>
<li>For HR bets: 35–40%+ is acceptable. Don't over-filter power hitters on K.</li>
</ul>
<p><b>Max BB Prob %</b> — Keep consistent across all bet types. A walk means the player didn't swing at a hit — bad for all props.</p>
<p><b>Min Hit/1B/XB/HR Prob %</b> — The minimum probability for your target stat. This is a hard floor.
Raising it gives you fewer but stronger candidates. Defaults are conservative starting points.</p>
<p><b>Min vs Grade</b> — Only use this as a positive filter for Hit/Single/XB bets.
For HR bets, leave it at -10 (no restriction) — negative grades don't disqualify HR threats.</p>
</div>

<div class="ref-section">
<h3>Step 3 — Read the Results Table</h3>
<p>The active score column (the one matching your target) is on the left after Player/Team/Pitcher.
All four scores are shown for reference — this lets you see if a player is also strong in other categories
(useful for parlay building).</p>
<p><b>Green cells</b> in score columns = top percentile for today's slate. The gradient is relative to today's pool.</p>
<p><b>Pitcher Grade</b> — Check this quickly. An A+ grade means this pitcher is projected to give up hits/HRs today.
A D grade means a dominant pitcher. This is game-level context layered on top of the batter's individual matchup probability.</p>
<p><b>Park Δ</b> — If positive and large (+5 or more), some of this player's score comes from park factors.
Toggle Park OFF to see if they're still strong on pure merit.</p>
</div>

<div class="ref-section">
<h3>Step 4 — Use the Best Per Target Section</h3>
<p>Below the results table, the <b>Best Per Target</b> expander shows the #1 player for each bet type
from the full slate — regardless of which filter you have active. This is your quick daily reference
for who stands out in each category before you apply any restrictions.</p>
</div>

<div class="ref-section">
<h3>Step 5 — The Parlay Builder</h3>
<p>Navigate to the <b>Parlay Builder</b> tab in the sidebar. Select:</p>
<ul>
<li><b>Cross-Game</b> for standard multi-game parlays (independent legs, different games)</li>
<li><b>SGP Stack</b> for same-game parlay where you want correlated players from the same team
  (if their offense explodes, all legs benefit)</li>
<li><b>SGP Split</b> for same-game parlays covering both teams (betting on a high-scoring game)</li>
</ul>
<p>The <b>Confidence score</b> uses harmonic mean of all leg scores. This penalises weak legs harder
than a simple average — a 3-leg parlay with scores of 90/90/30 scores lower than one with 70/70/70,
because that 30-score leg is a real liability.</p>
</div>

<div class="ref-section">
<h3>What to Look for in Parlay Legs</h3>
<p>Strong parlay legs share these characteristics:</p>
<ul>
<li>Score ≥ 65 for the active bet type</li>
<li>Pitcher grade B or better (A/A+ is ideal)</li>
<li>K% below your threshold for the bet type</li>
<li>Positive vs Grade for Hit/Single/XB legs (less important for HR)</li>
<li>Park Δ near neutral — you don't want a leg that relies entirely on park factors</li>
</ul>
</div>
""", unsafe_allow_html=True)

    with st.expander("⚖️ Strategy by Bet Type", expanded=False):
        LG = CONFIG
        st.markdown(f"""
<div class="ref-section">
<h3>Cash / Solo Props (Single Bet)</h3>
<ul>
<li><b>Hit props</b>: Target Hit Score ≥ 60, Hit% ≥ 28%, K% ≤ 25%, Pitcher Grade B+</li>
<li><b>Single props</b>: Single Score ≥ 55, 1B% ≥ 14%, XB% ≤ 6% (not a power guy), K% ≤ 22%</li>
<li><b>XB/Double props</b>: XB Score ≥ 60, XB% ≥ 5%, Pitcher Grade B+ (especially HR 2+ if available)</li>
<li><b>HR props</b>: HR Score ≥ 60, HR% ≥ 3.5% (above league avg {LG['league_hr_avg']}%), don't cap K% tightly</li>
</ul>
</div>

<div class="ref-section">
<h3>Multi-Game Parlays (2–4 Legs)</h3>
<ul>
<li>Each leg should score ≥ 60 in its target category</li>
<li>Mix bet types only if each player legitimately scores well in their assigned score type</li>
<li>Avoid putting a player with a D pitcher grade in a parlay — that 5% drag compounds across legs</li>
<li>Park Δ matters in parlays: if all legs have large positive Park Δ, the whole parlay relies on park conditions staying favorable</li>
</ul>
</div>

<div class="ref-section">
<h3>Same Game Parlays (SGP)</h3>
<ul>
<li><b>Stack (same team)</b>: If an offense projects well (team has high collective Hit% / XB%), stacking 3 players from that team creates positive correlation — if one hits, the others are more likely to as well</li>
<li><b>Split (both teams)</b>: If the game props CSV shows high 20+ Hits probability, both offenses may perform — use players from each team</li>
<li>For SGP stacks, the Pitcher Grade of the opposing pitcher matters a lot — A+ pitcher grade means it favors hits</li>
<li>Avoid SGP stacks in games where one pitcher has a D grade AND the team's collective hit prob is below average — the data is telling you it's a low-run game</li>
</ul>
</div>

<div class="ref-section">
<h3>Score Benchmarks</h3>
<ul>
<li>≥ 80 — Elite play for today's slate. Top tier of confidence.</li>
<li>65–79 — Strong play. Solid statistical foundation.</li>
<li>50–64 — Average. Fine as a supporting leg but not a primary anchor.</li>
<li>35–49 — Weak. Only use in large GPP-style parlays where you need differentiation.</li>
<li>≤ 34 — Avoid in any serious bet. The data isn't supporting this outcome today.</li>
</ul>
</div>

**League Baselines (4-year stable)** — K% {LG['league_k_avg']}% · BB% {LG['league_bb_avg']}% · HR% {LG['league_hr_avg']}% · AVG {LG['league_avg']}
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def main_page():
    render_header()

    with st.spinner("⚾ Loading today's matchups…"):
        raw_df     = load_matchups()
        pitcher_df = load_pitcher_data()
        game_cond  = load_game_conditions()
        qs_df      = load_pitcher_qs()

    if raw_df is None:
        st.error("❌ Could not load Matchups data. Check connection or try again.")
        return

    filters  = build_filters(raw_df)
    df       = compute_metrics(raw_df, use_park=filters['use_park'])
    df       = merge_pitcher_data(df, pitcher_df)
    df       = compute_scores(df)

    # Merge game conditions onto df (adds gc_* columns; neutral defaults if CSVs missing)
    df       = merge_game_conditions(df, game_cond, qs_df)

    # Compute GC score variants (*_gc columns) — always computed, only activated by toggle
    df       = compute_game_condition_scores(df)

    # When GC toggle is ON, use _gc score for sorting/filtering
    if filters.get('use_gc', False):
        gc_col = filters['score_col'] + '_gc'
        if gc_col in df.columns:
            filters['score_col'] = gc_col

    slate_df = get_slate_df(df, filters)

    render_stat_bar(df)
    render_pitcher_landscape(pitcher_df, df)
    render_park_notice(slate_df, filters)
    render_game_conditions_panel(slate_df, filters, game_cond, qs_df)
    render_score_summary_cards(slate_df, filters)

    filtered_df = apply_filters(df, filters)

    target_labels = {
        'Hit_Score':    '🎯 Any Base Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 Extra Base Hit',
        'HR_Score':     '💣 Home Run',
        'Hit_Score_gc':    '🎯 Any Base Hit ⛅',
        'Single_Score_gc': '1️⃣ Single ⛅',
        'XB_Score_gc':     '🔥 Extra Base Hit ⛅',
        'HR_Score_gc':     '💣 Home Run ⛅',
    }
    # Restore base score_col for display label lookup
    display_sc = filters['score_col_base']
    st.markdown(f"""
<div class="result-head">
  <span class="rh-label">{target_labels.get(filters['score_col'], target_labels.get(display_sc,'Hit'))} Candidates</span>
  <span class="rh-count">{len(filtered_df)} results</span>
</div>
""", unsafe_allow_html=True)

    render_results_table(filtered_df, filters)
    render_best_per_target(slate_df, filters)

    if not filtered_df.empty:
        render_visualizations(df, filtered_df, filters['score_col_base'])

    # ── Controls ─────────────────────────────────────────────────────────────
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    with c2:
        if not filtered_df.empty:
            st.download_button(
                "💾 Export CSV",
                filtered_df.to_csv(index=False),
                f"a1picks_mlb_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )


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

    page = st.sidebar.radio(
        "Navigate",
        ["⚾ Predictor", "⚡ Parlay Builder", "📚 Reference Manual"],
        index=0
    )

    if page == "⚾ Predictor":
        main_page()
    elif page == "⚡ Parlay Builder":
        raw_df = load_matchups()
        if raw_df is not None:
            pitcher_df = load_pitcher_data()
            game_cond  = load_game_conditions()
            qs_df      = load_pitcher_qs()
            df = compute_metrics(raw_df, use_park=True)
            df = merge_pitcher_data(df, pitcher_df)
            df = compute_scores(df)
            df = merge_game_conditions(df, game_cond, qs_df)
            df = compute_game_condition_scores(df)
            excl = st.session_state.get('excluded_players', [])
            if excl:
                df = df[~df['Batter'].isin(excl)]
            parlay_page(df)
        else:
            st.error("❌ Could not load data for Parlay Builder.")
    else:
        info_page()

    st.sidebar.markdown("---")
    st.sidebar.caption("V4.2 · Dark Precision · 4 Scores · Parlay Builder")


if __name__ == "__main__":
    main()
