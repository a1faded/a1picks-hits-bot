"""
A1PICKS MLB Hit Predictor — V4 Ground-Up Rebuild
=================================================
Data source  : BallPark Pal Matchups.csv (single file, replaces old two-CSV setup)
Scores       : Hit Score | Single Score | XB Score | HR Score
Park mode    : WITH park (blended avg) or WITHOUT park (toggle)
Pitcher CSVs : Dropped — pitcher context is baked into BallPark Pal Prob columns
Historical   : AVG/PA tiebreaker bonus when sample ≥ 10 PA
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
    'csv_url': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Matchups.csv',
    'cache_ttl': 900,           # 15 minutes
    'hist_min_pa': 10,          # minimum PA for historical tiebreaker
    'hist_bonus_max': 3.0,      # maximum bonus points from historical AVG
    'league_k_avg': 22.6,
    'league_bb_avg': 8.5,
}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── layout ── */
.block-container { padding-top: 1.5rem; }

/* ── score cards ── */
.card {
    padding: 0.9rem 1rem;
    border-radius: 10px;
    color: white;
    margin: 0.35rem 0;
    box-shadow: 0 3px 8px rgba(0,0,0,.18);
}
.card h4 { margin: 0 0 .3rem 0; font-size: .85rem; opacity: .9; }
.card h2 { margin: 0; font-size: 1.6rem; font-weight: 800; }
.card small { opacity: .75; font-size: .75rem; }

.card-blue   { background: linear-gradient(135deg,#2980b9,#6dd5fa); }
.card-green  { background: linear-gradient(135deg,#11998e,#38ef7d); }
.card-amber  { background: linear-gradient(135deg,#f7971e,#ffd200); color:#1a1a1a; }
.card-red    { background: linear-gradient(135deg,#c0392b,#e74c3c); }
.card-purple { background: linear-gradient(135deg,#667eea,#764ba2); }
.card-teal   { background: linear-gradient(135deg,#43b89c,#7bd9c8); }

/* ── staleness badge ── */
.badge {
    display:inline-block;
    padding:.3rem .75rem;
    border-radius:20px;
    font-size:.8rem;
    font-weight:700;
    color:white;
}
.badge-green  { background:#27ae60; }
.badge-yellow { background:#f39c12; color:#1a1a1a; }
.badge-red    { background:#e74c3c; }

/* ── park toggle notice ── */
.park-notice {
    background: #1a1a2e;
    border-left: 4px solid #f7971e;
    padding: .6rem 1rem;
    border-radius: 0 6px 6px 0;
    color: #ffd200;
    font-size: .85rem;
    margin: .5rem 0;
}

/* ── legend box ── */
.legend {
    background:#0d0d0d;
    border-radius:8px;
    padding:1rem 1.2rem;
    color:#e0e0e0;
    font-size:.83rem;
    margin:1rem 0;
    line-height:1.8;
}

/* ── matchup grade colours (used inline) ── */
.grade-ap { background:#1a9641; color:white; padding:2px 6px; border-radius:4px; font-weight:700; }
.grade-a  { background:#a6d96a; color:#111;  padding:2px 6px; border-radius:4px; font-weight:700; }
.grade-b  { background:#ffffbf; color:#111;  padding:2px 6px; border-radius:4px; }
.grade-c  { background:#fdae61; color:#111;  padding:2px 6px; border-radius:4px; }
.grade-d  { background:#d7191c; color:white; padding:2px 6px; border-radius:4px; font-weight:700; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def card(css_class: str, icon: str, title: str, value: str, subtitle: str = "") -> str:
    """Return an HTML metric card."""
    sub = f"<small>{subtitle}</small>" if subtitle else ""
    return f"""
    <div class="card {css_class}">
        <h4>{icon} {title}</h4>
        <h2>{value}</h2>
        {sub}
    </div>"""


def normalize_0_100(series: pd.Series) -> pd.Series:
    """Min-max normalise a series to 0–100, return 50 if flat."""
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(50.0, index=series.index)
    return ((series - mn) / (mx - mn) * 100).round(1)


def staleness_badge() -> str:
    if 'data_loaded_at' not in st.session_state:
        return ""
    mins = int((datetime.now() - st.session_state.data_loaded_at).total_seconds() / 60)
    if mins < 5:
        return f'<span class="badge badge-green">🟢 Data fresh ({mins}m ago)</span>'
    elif mins < 12:
        return f'<span class="badge badge-yellow">🟡 Data {mins}m old</span>'
    else:
        return f'<span class="badge badge-red">🔴 Data {mins}m old — consider refreshing</span>'


def color_grade(val: str) -> str:
    """CSS string for matchup grade cell colouring."""
    mapping = {
        'A+': 'background-color:#1a9641;color:white;font-weight:700',
        'A':  'background-color:#a6d96a;color:#111;font-weight:700',
        'B':  'background-color:#ffffbf;color:#111',
        'C':  'background-color:#fdae61;color:#111',
        'D':  'background-color:#d7191c;color:white;font-weight:700',
    }
    return mapping.get(str(val), '')


# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_data(url: str) -> pd.DataFrame | None:
    """Load and lightly validate the Matchups CSV."""
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        df = pd.read_csv(StringIO(resp.text))

        required = [
            'Game', 'Team', 'Batter', 'Pitcher',
            'HR Prob', 'XB Prob', '1B Prob', 'BB Prob', 'K Prob',
            'HR Prob (no park)', 'XB Prob (no park)', '1B Prob (no park)',
            'BB Prob (no park)', 'K Prob (no park)',
            'RC', 'vs Grade',
        ]
        missing = [c for c in required if c not in df.columns]
        if missing:
            st.error(f"❌ Missing columns in CSV: {missing}")
            return None

        # Strip whitespace from string columns
        for col in ['Team', 'Batter', 'Pitcher', 'Game']:
            if col in df.columns:
                df[col] = df[col].astype(str).str.strip()

        return df

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading data: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing data: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# METRIC COMPUTATION
# Blends park + no-park versions, computes final Prob columns,
# converts vs Grade → numeric modifier, historical tiebreaker.
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, use_park: bool) -> pd.DataFrame:
    """
    Derive the working probability columns the scoring engine needs.

    With use_park=True  → average of (with-park) and (no-park) columns.
    With use_park=False → use only the (no-park) columns.

    Resulting columns (always present after this function):
        p_hr, p_xb, p_1b, p_bb, p_k   — final probability values used in scoring
        p_hr_park, p_hr_base           — for the park-impact display
        p_xb_park, p_xb_base
        p_1b_park, p_1b_base
        vs_mod                         — vs Grade mapped to [-1, +1]
        hist_bonus                     — small tiebreaker from AVG when PA ≥ 10
        park_pct_hr / park_pct_xb / park_pct_1b  — % delta between park and base
    """
    df = df.copy()

    stats = ['HR', 'XB', '1B', 'BB', 'K']
    park_cols  = {s: f'{s} Prob'          for s in stats}
    base_cols  = {s: f'{s} Prob (no park)' for s in stats}

    for s in stats:
        pc = park_cols[s]
        bc = base_cols[s]
        df[f'p_{s.lower()}_park'] = pd.to_numeric(df[pc], errors='coerce').fillna(0)
        df[f'p_{s.lower()}_base'] = pd.to_numeric(df[bc], errors='coerce').fillna(0)

        if use_park:
            # Blend: straight average of park-adjusted and pure base
            df[f'p_{s.lower()}'] = (
                df[f'p_{s.lower()}_park'] + df[f'p_{s.lower()}_base']
            ) / 2
        else:
            df[f'p_{s.lower()}'] = df[f'p_{s.lower()}_base']

    # Park impact % for the park-delta display
    # How much does the park version differ from the base version (as % of base)
    for s in ['hr', 'xb', '1b']:
        base = df[f'p_{s}_base']
        park = df[f'p_{s}_park']
        # Positive = park boosts, negative = park suppresses
        df[f'park_delta_{s}'] = np.where(
            base > 0,
            ((park - base) / base * 100).round(1),
            0.0
        )

    # vs Grade → moderate continuous modifier [-1, +1] scaled from [-10, +10]
    df['vs_mod'] = pd.to_numeric(df['vs Grade'], errors='coerce').fillna(0).clip(-10, 10) / 10

    # RC as a quality signal (normalised to 0–1 range for use as a small weight)
    rc_col = 'RC' if use_park else 'RC (no park)'
    if rc_col not in df.columns:
        rc_col = 'RC'
    df['rc_norm'] = pd.to_numeric(df[rc_col], errors='coerce').fillna(0)

    # Historical tiebreaker: only activated when PA ≥ threshold
    df['PA']  = pd.to_numeric(df['PA'],  errors='coerce').fillna(0)
    df['AVG'] = pd.to_numeric(df['AVG'], errors='coerce').fillna(0)

    df['hist_bonus'] = np.where(
        df['PA'] >= CONFIG['hist_min_pa'],
        (df['AVG'] * CONFIG['hist_bonus_max']).round(2),
        0.0
    )

    # Ensure Starter is numeric
    df['Starter'] = pd.to_numeric(df['Starter'], errors='coerce').fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# SCORING ENGINE
# Four purpose-built scores, each normalised to 0–100.
# Park contribution and base score tracked separately for the park display.
# ─────────────────────────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Produces four score columns, each 0–100:

    Hit_Score
        Goal : identify who is most likely to get ANY base hit today
        Drivers : p_1b (heaviest) + p_xb + p_hr (all contribute to hit probability)
        Penalties: p_k (heavy — a K means no hit) + p_bb (moderate — a walk isn't a hit)
        vs/RC    : supporting signal

    Single_Score
        Goal : identify who is most likely to get a SINGLE specifically
        Drivers : p_1b (dominant)
        Penalties: p_k (heavy) + p_bb (moderate) + p_xb (light negative — big XB swing
                   means the ball leaves the park or goes for extra bases, not a clean single)
                   + p_hr (penalty — HR swing path ≠ single)
        Logic   : A pure singles hitter stays in the park. If a guy's XB% and HR% are high,
                  the ball probably isn't dropping in for a single.

    XB_Score
        Goal : identify who is most likely to hit a DOUBLE or TRIPLE
        Drivers : p_xb (dominant)
        Penalties: p_k (moderate — XB hitters swing harder, more K is acceptable)
                   + p_bb (same moderate as others — walks block plate appearances)
        Supporting: p_hr (same power swing path) + vs_mod + rc_norm

    HR_Score
        Goal : identify who is most likely to hit a HOME RUN
        Drivers : p_hr (dominant)
        Penalties: p_k (LIGHT — elite HR hitters naturally K more, don't punish them heavily)
                   + p_bb (moderate — same reason as all scores)
        Supporting: p_xb (same power swing), vs_mod, rc_norm

    All four scores also receive:
        + vs_mod bonus  (–1 to +1, scaled small so it doesn't dominate)
        + hist_bonus    (only when PA ≥ 10 vs this pitcher; max 3 pts before normalisation)
    """
    df = df.copy()

    # ── shared small modifiers ──────────────────────────────────────────────
    # vs Grade contribution: +/- up to 2 raw score points (won't dominate)
    vs_contrib = df['vs_mod'] * 2.0

    # RC contribution: small quality signal, normalised to -1..+1 range
    rc_min, rc_max = df['rc_norm'].min(), df['rc_norm'].max()
    if rc_max > rc_min:
        rc_contrib = ((df['rc_norm'] - rc_min) / (rc_max - rc_min) * 2) - 1  # -1..+1
    else:
        rc_contrib = pd.Series(0.0, index=df.index)

    # ── HIT SCORE ────────────────────────────────────────────────────────────
    hit_raw = (
          df['p_1b']  * 3.0    # Singles — most common hit type, primary driver
        + df['p_xb']  * 2.0    # XBs — also base hits
        + df['p_hr']  * 1.0    # HRs — also base hits
        - df['p_k']   * 2.5    # K = no hit, heaviest penalty
        - df['p_bb']  * 1.0    # BB = not a hit, moderate penalty
        + vs_contrib  * 1.0    # matchup quality
        + rc_contrib  * 0.5    # overall matchup context
        + df['hist_bonus']
    )
    df['Hit_Score'] = normalize_0_100(hit_raw)

    # ── SINGLE SCORE ─────────────────────────────────────────────────────────
    single_raw = (
          df['p_1b']  * 5.0    # Singles — the entire point, dominant weight
        - df['p_k']   * 2.5    # K = no hit, same heavy penalty as Hit Score
        - df['p_bb']  * 1.0    # BB = not a single, moderate penalty
        - df['p_xb']  * 0.8    # High XB% means the ball carries — bad for singles
        - df['p_hr']  * 0.5    # High HR% swing path conflicts with singles
        + vs_contrib  * 0.8
        + rc_contrib  * 0.4
        + df['hist_bonus']
    )
    df['Single_Score'] = normalize_0_100(single_raw)

    # ── XB SCORE ─────────────────────────────────────────────────────────────
    xb_raw = (
          df['p_xb']  * 5.0    # Extra base hits — dominant driver
        + df['p_hr']  * 0.8    # Same power swing path, supporting signal
        - df['p_k']   * 1.5    # Moderate penalty (XB hitters swing harder, more K ok)
        - df['p_bb']  * 1.0    # BB same moderate penalty as always
        + vs_contrib  * 1.2    # matchup matters for hard contact
        + rc_contrib  * 0.6
        + df['hist_bonus']
    )
    df['XB_Score'] = normalize_0_100(xb_raw)

    # ── HR SCORE ─────────────────────────────────────────────────────────────
    hr_raw = (
          df['p_hr']  * 6.0    # Home runs — everything
        + df['p_xb']  * 0.8    # Same power swing, supporting signal
        - df['p_k']   * 0.8    # LIGHT penalty — HR hitters K more, that's fine
        - df['p_bb']  * 1.0    # Same moderate penalty as all scores
        + vs_contrib  * 1.5    # pitcher matchup is extra critical for HRs
        + rc_contrib  * 0.5
        + df['hist_bonus']
    )
    df['HR_Score'] = normalize_0_100(hr_raw)

    # ── Park contribution tracking ────────────────────────────────────────────
    # For the park-delta display: what % of the blended score comes from park adjustment
    # We calculate the score using ONLY base (no park) and compare to blended
    hit_base_raw = (
          df['p_1b_base'] * 3.0
        + df['p_xb_base'] * 2.0
        + df['p_hr_base'] * 1.0
        - df['p_k_base']  * 2.5
        - df['p_bb_base'] * 1.0
        + vs_contrib
        + rc_contrib * 0.5
        + df['hist_bonus']
    )
    df['Hit_Score_base'] = normalize_0_100(hit_base_raw)

    single_base_raw = (
          df['p_1b_base'] * 5.0
        - df['p_k_base']  * 2.5
        - df['p_bb_base'] * 1.0
        - df['p_xb_base'] * 0.8
        - df['p_hr_base'] * 0.5
        + vs_contrib * 0.8
        + rc_contrib * 0.4
        + df['hist_bonus']
    )
    df['Single_Score_base'] = normalize_0_100(single_base_raw)

    xb_base_raw = (
          df['p_xb_base'] * 5.0
        + df['p_hr_base'] * 0.8
        - df['p_k_base']  * 1.5
        - df['p_bb_base'] * 1.0
        + vs_contrib * 1.2
        + rc_contrib * 0.6
        + df['hist_bonus']
    )
    df['XB_Score_base'] = normalize_0_100(xb_base_raw)

    hr_base_raw = (
          df['p_hr_base'] * 6.0
        + df['p_xb_base'] * 0.8
        - df['p_k_base']  * 0.8
        - df['p_bb_base'] * 1.0
        + vs_contrib * 1.5
        + rc_contrib * 0.5
        + df['hist_bonus']
    )
    df['HR_Score_base'] = normalize_0_100(hr_base_raw)

    return df


# ─────────────────────────────────────────────────────────────────────────────
# FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def build_filters(df: pd.DataFrame) -> dict:
    """Render sidebar filters and return a filter config dict."""

    st.sidebar.title("🏟️ A1PICKS Filters")
    st.sidebar.markdown("---")

    filters = {}

    # ── Score type / target ───────────────────────────────────────────────────
    st.sidebar.markdown("### 🎯 Betting Target")

    target_map = {
        "🎯 Hit Score  — Any Base Hit":     "hit",
        "1️⃣ Single Score — Single Specifically": "single",
        "🔥 XB Score  — Double / Triple":  "xb",
        "💣 HR Score  — Home Run":          "hr",
    }
    selected_label = st.sidebar.selectbox(
        "Choose Your Betting Target",
        list(target_map.keys()),
        help="Each target weights the data differently based on what outcome you're betting on."
    )
    filters['target'] = target_map[selected_label]

    score_col_map = {
        'hit':    'Hit_Score',
        'single': 'Single_Score',
        'xb':     'XB_Score',
        'hr':     'HR_Score',
    }
    filters['score_col'] = score_col_map[filters['target']]

    # ── Park adjustment toggle ────────────────────────────────────────────────
    st.sidebar.markdown("### 🏟️ Park Adjustment")
    filters['use_park'] = st.sidebar.toggle(
        "Include Park Factors",
        value=True,
        help=(
            "ON  → Blends park-adjusted + base probabilities. "
            "Includes ballpark dimensions, weather, etc.\n\n"
            "OFF → Uses base probabilities only (pure player vs pitcher)."
        )
    )

    # ── Starter filter ────────────────────────────────────────────────────────
    st.sidebar.markdown("### ⚾ Pitcher Type")
    filters['starters_only'] = st.sidebar.checkbox(
        "Starters only (exclude relievers)",
        value=False,
        help="Only include matchups against starting pitchers."
    )

    # ── K% / BB% hard caps ───────────────────────────────────────────────────
    st.sidebar.markdown("### 📊 Stat Filters")

    filters['max_k'] = st.sidebar.slider(
        "Max K Prob %",
        min_value=10.0, max_value=50.0,
        value=35.0, step=0.5,
        help="Filter out batters with K probability above this threshold."
    )

    filters['max_bb'] = st.sidebar.slider(
        "Max BB Prob %",
        min_value=2.0, max_value=20.0,
        value=15.0, step=0.5,
        help="Filter out batters with walk probability above this threshold."
    )

    # Target-specific minimum probability
    min_prob_labels = {
        'hit':    ("Min Hit Prob % (1B+XB+HR)", "total_hit_prob"),
        'single': ("Min 1B Prob %",             "p_1b"),
        'xb':     ("Min XB Prob %",             "p_xb"),
        'hr':     ("Min HR Prob %",             "p_hr"),
    }
    prob_label, prob_col = min_prob_labels[filters['target']]

    # Sensible defaults per target
    defaults = {'hit': 20.0, 'single': 10.0, 'xb': 4.0, 'hr': 2.0}
    max_vals  = {'hit': 50.0, 'single': 30.0, 'xb': 12.0, 'hr': 8.0}

    filters['min_prob']     = st.sidebar.slider(
        prob_label,
        min_value=0.0,
        max_value=max_vals[filters['target']],
        value=defaults[filters['target']],
        step=0.5,
    )
    filters['min_prob_col'] = prob_col

    # vs Grade minimum
    filters['min_vs'] = st.sidebar.slider(
        "Min vs Grade",
        min_value=-10, max_value=10,
        value=-10, step=1,
        help="Filter by minimum Batter vs Pitcher on-paper rating."
    )

    # ── Team filters ──────────────────────────────────────────────────────────
    st.sidebar.markdown("### 🏟️ Team Filters")
    all_teams = sorted(df['Team'].unique().tolist()) if df is not None else []

    filters['include_teams'] = st.sidebar.multiselect(
        "Include Only Teams",
        options=all_teams,
        help="Leave blank to include all teams."
    )
    filters['exclude_teams'] = st.sidebar.multiselect(
        "Exclude Teams",
        options=all_teams,
    )

    # ── Lineup exclusions ─────────────────────────────────────────────────────
    st.sidebar.markdown("### 🚫 Lineup Status")
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []

    all_players = sorted(df['Batter'].unique().tolist()) if df is not None else []
    selected_exclusions = st.sidebar.multiselect(
        "Players NOT Playing Today",
        options=all_players,
        default=st.session_state.excluded_players,
        help="Exclude confirmed scratches / injured players.",
        key="lineup_exclusions"
    )
    st.session_state.excluded_players = selected_exclusions
    filters['excluded_players'] = selected_exclusions

    if st.sidebar.button("🔄 Clear All Exclusions"):
        st.session_state.excluded_players = []
        st.rerun()

    # ── Sort & display count ──────────────────────────────────────────────────
    st.sidebar.markdown("### 🔢 Display")

    sort_options = {
        "Score (High→Low)":     (filters['score_col'], False),
        "Hit Prob % (High→Low)": ("total_hit_prob",    False),
        "1B Prob % (High→Low)":  ("p_1b",              False),
        "XB Prob % (High→Low)":  ("p_xb",              False),
        "HR Prob % (High→Low)":  ("p_hr",              False),
        "K Prob % (Low→High)":   ("p_k",               True),
        "BB Prob % (Low→High)":  ("p_bb",              True),
        "vs Grade (High→Low)":   ("vs Grade",          False),
    }
    filters['sort_label'] = st.sidebar.selectbox("Sort By", list(sort_options.keys()))
    filters['sort_col'], filters['sort_asc'] = sort_options[filters['sort_label']]

    filters['result_count'] = st.sidebar.selectbox(
        "Show Top N",
        options=[5, 10, 15, 20, 25, 30, "All"],
        index=2,
    )

    # ── Best per team ─────────────────────────────────────────────────────────
    filters['best_per_team'] = st.sidebar.checkbox(
        "🏟️ Best player per team only",
        value=False,
        help="Show only the top-scoring player from each team."
    )

    return filters


# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
# ─────────────────────────────────────────────────────────────────────────────

def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply all filters and return a sorted, limited result set."""

    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()

    # Starter filter
    if filters.get('starters_only'):
        out = out[out['Starter'] == 1]

    # Player exclusions
    excl = filters.get('excluded_players', [])
    if excl:
        n_before = len(out)
        out = out[~out['Batter'].isin(excl)]
        n_excl = n_before - len(out)
        if n_excl:
            st.info(f"🚫 Excluded {n_excl} player(s) from today's lineups")

    # Team filters
    inc_teams = filters.get('include_teams', [])
    if inc_teams:
        out = out[out['Team'].isin(inc_teams)]

    exc_teams = filters.get('exclude_teams', [])
    if exc_teams:
        before = len(out)
        out = out[~out['Team'].isin(exc_teams)]
        removed = before - len(out)
        if removed:
            st.info(f"🚫 Excluded {removed} players from teams: {', '.join(exc_teams)}")

    # Stat caps
    out = out[out['p_k']  <= filters['max_k']]
    out = out[out['p_bb'] <= filters['max_bb']]

    # Min prob (target-specific)
    min_col = filters.get('min_prob_col', 'total_hit_prob')
    if min_col in out.columns:
        out = out[out[min_col] >= filters['min_prob']]

    # vs Grade minimum
    if filters['min_vs'] > -10:
        out = out[pd.to_numeric(out['vs Grade'], errors='coerce').fillna(-10) >= filters['min_vs']]

    # Best per team
    score_col = filters['score_col']
    if filters.get('best_per_team') and not out.empty:
        out = out.loc[out.groupby('Team')[score_col].idxmax()].copy()
        st.info(f"🏟️ Showing best player from each of {len(out)} teams")

    # Sort
    sort_c = filters['sort_col']
    sort_a = filters['sort_asc']
    if sort_c in out.columns:
        out[sort_c] = pd.to_numeric(out[sort_c], errors='coerce')
        out = out.sort_values(sort_c, ascending=sort_a, na_position='last')

    # Limit
    n = filters['result_count']
    if n != "All":
        out = out.head(int(n))

    return out


# ─────────────────────────────────────────────────────────────────────────────
# DISPLAY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def render_header():
    c1, c2 = st.columns([1, 5])
    with c1:
        st.image(
            'https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true',
            width=180
        )
    with c2:
        st.title("⚾ A1PICKS MLB Hit Predictor")
        st.markdown("*Extract the best betting targets from BallPark Pal simulation data*")


def render_data_dashboard(df: pd.DataFrame):
    """Top-level data quality / overview cards."""
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(card("card-blue", "📈", "Matchups", str(len(df))), unsafe_allow_html=True)
    with c2:
        st.markdown(card("card-blue", "👤", "Batters", str(df['Batter'].nunique())), unsafe_allow_html=True)
    with c3:
        st.markdown(card("card-blue", "🏟️", "Teams", str(df['Team'].nunique())), unsafe_allow_html=True)
    with c4:
        avg_hp = df['total_hit_prob'].mean()
        st.markdown(card("card-green", "🎯", "Avg Hit Prob", f"{avg_hp:.1f}%"), unsafe_allow_html=True)
    with c5:
        sb = staleness_badge()
        age_html = sb if sb else '<span class="badge badge-green">🟢 Just loaded</span>'
        st.markdown(card("card-purple", "🕐", "Data Age", age_html, "Refreshes every 15 min"),
                    unsafe_allow_html=True)

    st.markdown("---")


def render_park_notice(filtered_df: pd.DataFrame, filters: dict):
    """Show park-adjustment context when park mode is active."""
    if not filters['use_park']:
        st.markdown(
            '<div class="park-notice">🏟️ <b>Park Adjustment OFF</b> — '
            'Scores use pure player vs pitcher probabilities only (no park factors, weather, etc.)</div>',
            unsafe_allow_html=True
        )
        return

    score_col      = filters['score_col']
    base_score_col = score_col + '_base'
    if base_score_col not in filtered_df.columns or filtered_df.empty:
        return

    avg_blended = filtered_df[score_col].mean()
    avg_base    = filtered_df[base_score_col].mean()
    delta       = avg_blended - avg_base
    pct_contrib = (delta / avg_base * 100) if avg_base != 0 else 0

    direction = "boosted" if delta >= 0 else "reduced"
    colour    = "#38ef7d" if delta >= 0 else "#e74c3c"

    st.markdown(
        f'<div class="park-notice">'
        f'🏟️ <b>Park Adjustment ON</b> — Park factors have '
        f'<span style="color:{colour};font-weight:700">{direction} scores by ~{abs(pct_contrib):.1f}%</span> '
        f'on average vs base (no-park) probabilities. '
        f'Toggle park off to see true base scores.'
        f'</div>',
        unsafe_allow_html=True
    )


def render_score_summary_cards(filtered_df: pd.DataFrame, filters: dict):
    """Four cards showing top player per score type."""
    if filtered_df.empty:
        return

    score_cols = ['Hit_Score', 'Single_Score', 'XB_Score', 'HR_Score']
    icons      = ['🎯', '1️⃣', '🔥', '💣']
    labels     = ['Hit Score', 'Single Score', 'XB Score', 'HR Score']
    css        = ['card-green', 'card-teal', 'card-amber', 'card-red']

    cols = st.columns(4)
    for i, (sc, icon, lbl, css_c) in enumerate(zip(score_cols, icons, labels, css)):
        if sc not in filtered_df.columns:
            continue
        row = filtered_df.loc[filtered_df[sc].idxmax()]
        base_col = sc + '_base'
        park_str = ""
        if filters['use_park'] and base_col in filtered_df.columns:
            base_val   = row[base_col]
            blend_val  = row[sc]
            park_delta = blend_val - base_val
            park_pct   = (park_delta / base_val * 100) if base_val != 0 else 0
            sign       = "+" if park_delta >= 0 else ""
            park_str   = f" (park {sign}{park_pct:.1f}%)"

        with cols[i]:
            st.markdown(
                card(css_c, icon, lbl,
                     f"{row['Batter']}",
                     f"Score: {row[sc]:.1f}{park_str} | {row['Team']}"),
                unsafe_allow_html=True
            )


def render_results_table(filtered_df: pd.DataFrame, filters: dict):
    """Render the main styled results dataframe."""

    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters. Try relaxing the thresholds.")
        return

    score_col      = filters['score_col']
    base_score_col = score_col + '_base'
    use_park       = filters['use_park']

    display_df = filtered_df.copy()

    # League context columns
    display_df['K% vs Lg']  = (CONFIG['league_k_avg']  - display_df['p_k']).round(1)
    display_df['BB% vs Lg'] = (CONFIG['league_bb_avg'] - display_df['p_bb']).round(1)

    # Park delta for active score
    if use_park and base_score_col in display_df.columns:
        display_df['Park Δ'] = (display_df[score_col] - display_df[base_score_col]).round(1)
    else:
        display_df['Park Δ'] = 0.0

    # Total hit prob display
    display_df['Hit Prob %'] = display_df['total_hit_prob'].round(1)

    # Historical sample flag
    display_df['Hist PA'] = display_df['PA'].astype(int)
    display_df['AVG vs P'] = display_df['AVG'].round(3)

    # Active score label
    score_label_map = {
        'Hit_Score':    '🎯 Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 XB',
        'HR_Score':     '💣 HR',
    }
    active_label = score_label_map.get(score_col, 'Score')

    # Rename vs Grade for clarity
    display_df['vs Grade'] = pd.to_numeric(display_df['vs Grade'], errors='coerce').round(0).astype(int)

    # Column selection for display
    cols_to_show = {
        'Batter':      'Batter',
        'Team':        'Team',
        'Pitcher':     'Pitcher',
        score_col:     active_label,
    }

    # Park base score column (if park mode on)
    if use_park and base_score_col in display_df.columns:
        cols_to_show[base_score_col] = f'{active_label} (no park)'
        cols_to_show['Park Δ']       = 'Park Δ'

    # All four scores always shown (for reference)
    all_score_cols = {
        'Hit_Score':    '🎯 Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 XB',
        'HR_Score':     '💣 HR',
    }
    for sc, sl in all_score_cols.items():
        if sc != score_col and sc in display_df.columns:
            cols_to_show[sc] = sl

    cols_to_show.update({
        'Hit Prob %': 'Hit Prob %',
        'p_1b':       '1B Prob %',
        'p_xb':       'XB Prob %',
        'p_hr':       'HR Prob %',
        'p_k':        'K Prob %',
        'p_bb':       'BB Prob %',
        'K% vs Lg':   'K% vs Lg',
        'BB% vs Lg':  'BB% vs Lg',
        'vs Grade':   'vs Grade',
        'Hist PA':    'Hist PA',
        'AVG vs P':   'AVG vs P',
    })

    # Build display frame
    existing_cols = [c for c in cols_to_show if c in display_df.columns]
    out_df = display_df[existing_cols].rename(columns=cols_to_show)

    # Format dict
    fmt = {}
    for new_name in out_df.columns:
        if 'Prob %' in new_name or new_name in ['1B Prob %', 'XB Prob %', 'HR Prob %', 'K Prob %', 'BB Prob %']:
            fmt[new_name] = "{:.1f}%"
        elif 'Score' in new_name or new_name in [active_label, f'{active_label} (no park)',
                                                   '🎯 Hit', '1️⃣ Single', '🔥 XB', '💣 HR']:
            fmt[new_name] = "{:.1f}"
        elif new_name == 'Park Δ':
            fmt[new_name] = "{:+.1f}"
        elif 'vs Lg' in new_name:
            fmt[new_name] = "{:+.1f}%"
        elif new_name == 'AVG vs P':
            fmt[new_name] = "{:.3f}"

    styled = out_df.style.format(fmt, na_rep="—")

    # Colour gradients on score columns
    score_display_names = [n for n in out_df.columns
                           if any(e in n for e in ['Hit', 'Single', 'XB', 'HR', '🎯', '1️⃣', '🔥', '💣'])
                           and 'no park' not in n and 'Park' not in n and 'Prob' not in n]
    for sdn in score_display_names:
        try:
            cmap = {
                '🎯 Hit':    'Greens',
                '1️⃣ Single': 'YlGn',
                '🔥 XB':     'YlOrBr',
                '💣 HR':     'YlOrRd',
            }
            styled = styled.background_gradient(
                subset=[sdn],
                cmap=cmap.get(sdn, 'Greens'),
                vmin=0, vmax=100
            )
        except Exception:
            pass

    # Park delta colouring
    if 'Park Δ' in out_df.columns:
        styled = styled.background_gradient(
            subset=['Park Δ'], cmap='RdYlGn', vmin=-10, vmax=10
        )

    # K% vs league (positive = better contact = green)
    if 'K% vs Lg' in out_df.columns:
        styled = styled.background_gradient(
            subset=['K% vs Lg'], cmap='RdYlGn', vmin=-8, vmax=12
        )

    # vs Grade colouring
    if 'vs Grade' in out_df.columns:
        styled = styled.background_gradient(
            subset=['vs Grade'], cmap='RdYlGn', vmin=-10, vmax=10
        )

    st.dataframe(styled, use_container_width=True)

    # ── Legend ────────────────────────────────────────────────────────────────
    park_note = (
        "Park Δ = how much park factors shifted this player's active score "
        "(positive = park helped, negative = park hurt). "
        "Toggle 'Include Park Factors' OFF in sidebar to see pure base scores."
        if use_park else
        "Park Adjustment is OFF — all scores based on pure player vs pitcher probabilities."
    )
    hist_note = f"Hist PA / AVG vs P: historical PA ≥ {CONFIG['hist_min_pa']} triggers a small score tiebreaker bonus."

    st.markdown(f"""
    <div class="legend">
        <b>📊 Score Guide</b><br>
        🎯 <b>Hit Score</b> — Best target for any base hit bet (1B+XB+HR). Heavy K% penalty.<br>
        1️⃣ <b>Single Score</b> — Best target for a single specifically.
            High XB%/HR% is penalised here — power swing ≠ single.<br>
        🔥 <b>XB Score</b> — Best target for a double or triple. adj_XB dominant.<br>
        💣 <b>HR Score</b> — Best target for a home run. Light K% penalty (power hitters K more).<br>
        <br>
        <b>vs Grade</b> — Batter vs this pitcher on-paper rating (−10 to +10).<br>
        <b>K% vs Lg</b> — Positive = better contact than league avg ({CONFIG['league_k_avg']}%).<br>
        <b>BB% vs Lg</b> — Positive = more aggressive than league avg ({CONFIG['league_bb_avg']}%).<br>
        <br>
        {park_note}<br>
        {hist_note}
    </div>
    """, unsafe_allow_html=True)


def render_profile_diversity(filtered_df: pd.DataFrame, use_park: bool):
    """Show the best player per betting target as a quick-scan summary."""

    if len(filtered_df) < 3:
        st.info("💡 Need at least 3 players for profile diversity analysis.")
        return

    with st.expander("🔍 Best Player per Betting Target", expanded=True):
        score_defs = [
            ('Hit_Score',    '🎯 Hit',    'card-green',  'Best for any base hit bet'),
            ('Single_Score', '1️⃣ Single', 'card-teal',   'Best for single specifically'),
            ('XB_Score',     '🔥 XB',     'card-amber',  'Best for double / triple'),
            ('HR_Score',     '💣 HR',     'card-red',    'Best for home run'),
        ]
        cols = st.columns(4)
        for i, (sc, lbl, css, desc) in enumerate(score_defs):
            if sc not in filtered_df.columns:
                continue
            row = filtered_df.loc[filtered_df[sc].idxmax()]

            base_col  = sc + '_base'
            park_note = ""
            if use_park and base_col in filtered_df.columns:
                delta    = row[sc] - row[base_col]
                pct      = (delta / row[base_col] * 100) if row[base_col] != 0 else 0
                sign     = "+" if delta >= 0 else ""
                park_note = f"Park: {sign}{pct:.1f}%"

            k_vs_lg  = CONFIG['league_k_avg']  - row['p_k']
            bb_vs_lg = CONFIG['league_bb_avg'] - row['p_bb']
            hist_str = (
                f"Hist: {int(row['PA'])} PA / {row['AVG']:.3f} avg"
                if row['PA'] >= CONFIG['hist_min_pa'] else "No history vs pitcher"
            )

            with cols[i]:
                st.markdown(f"**{lbl}** — {desc}")
                st.success(f"**{row['Batter']}** ({row['Team']})")
                st.markdown(f"""
| Metric | Value |
|--------|-------|
| Score | **{row[sc]:.1f}** {park_note} |
| Hit Prob % | {row['total_hit_prob']:.1f}% |
| 1B% / XB% / HR% | {row['p_1b']:.1f}% / {row['p_xb']:.1f}% / {row['p_hr']:.1f}% |
| K% | {row['p_k']:.1f}% (vs Lg: {k_vs_lg:+.1f}%) |
| BB% | {row['p_bb']:.1f}% (vs Lg: {bb_vs_lg:+.1f}%) |
| vs Grade | {int(row['vs Grade'])} |
| {hist_str} | |
""")


def render_visualizations(df: pd.DataFrame, filtered_df: pd.DataFrame, score_col: str):
    """Charts: score distribution + scatter."""

    st.subheader("📈 Analysis Charts")
    c1, c2 = st.columns(2)

    with c1:
        chart = alt.Chart(df).mark_bar(color='#2980b9', opacity=0.7).encode(
            alt.X(f'{score_col}:Q', bin=alt.Bin(maxbins=15), title='Score'),
            alt.Y('count()', title='Players'),
            tooltip=['count()']
        ).properties(title=f'{score_col} Distribution (All Players)', width=350, height=280)
        st.altair_chart(chart, use_container_width=True)

    with c2:
        if not filtered_df.empty:
            chart2 = alt.Chart(filtered_df).mark_circle(size=90, opacity=0.75).encode(
                alt.X('total_hit_prob:Q',  title='Hit Prob %'),
                alt.Y('p_k:Q',            title='K Prob %'),
                alt.Color(f'{score_col}:Q', scale=alt.Scale(scheme='viridis')),
                alt.Size('p_hr:Q',         title='HR Prob %'),
                tooltip=['Batter', 'Team', f'{score_col}', 'total_hit_prob', 'p_k', 'p_hr']
            ).properties(title='Hit Prob vs K Risk (filtered players)', width=350, height=280)
            st.altair_chart(chart2, use_container_width=True)

    # Four-score comparison (filtered, ≤30 players)
    if not filtered_df.empty and len(filtered_df) <= 30:
        score_melt = filtered_df[['Batter', 'Hit_Score', 'Single_Score', 'XB_Score', 'HR_Score']].melt(
            id_vars='Batter',
            var_name='Score Type',
            value_name='Score'
        )
        score_melt['Score Type'] = score_melt['Score Type'].map({
            'Hit_Score':    '🎯 Hit',
            'Single_Score': '1️⃣ Single',
            'XB_Score':     '🔥 XB',
            'HR_Score':     '💣 HR',
        })
        chart3 = alt.Chart(score_melt).mark_bar().encode(
            alt.X('Batter:N', sort='-y'),
            alt.Y('Score:Q'),
            alt.Color('Score Type:N', scale=alt.Scale(
                domain=['🎯 Hit', '1️⃣ Single', '🔥 XB', '💣 HR'],
                range=['#27ae60', '#43b89c', '#f39c12', '#e74c3c']
            )),
            alt.Column('Score Type:N', title=None),
            tooltip=['Batter', 'Score Type', alt.Tooltip('Score:Q', format='.1f')]
        ).properties(width=200, height=260, title='All Four Scores by Player')
        st.altair_chart(chart3, use_container_width=True)

    # Team summary
    if not filtered_df.empty:
        team_agg = filtered_df.groupby('Team').agg(
            Players=('Batter', 'count'),
            Avg_Hit_Prob=('total_hit_prob', 'mean'),
            Avg_Hit_Score=('Hit_Score', 'mean'),
            Avg_XB_Score=('XB_Score', 'mean'),
            Avg_HR_Score=('HR_Score', 'mean'),
        ).round(1).sort_values('Avg_Hit_Prob', ascending=False).reset_index()
        team_agg.columns = ['Team', 'Players', 'Avg Hit Prob%', '🎯 Hit Score', '🔥 XB Score', '💣 HR Score']
        st.subheader("🏟️ Team Summary")
        st.dataframe(team_agg, use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PAGE
# ─────────────────────────────────────────────────────────────────────────────

def main_page():
    render_header()

    # Track freshness
    if 'data_loaded_at' not in st.session_state:
        st.session_state.data_loaded_at = datetime.now()

    # Load raw data
    with st.spinner("⚾ Loading today's matchups from BallPark Pal..."):
        raw_df = load_data(CONFIG['csv_url'])

    if raw_df is None:
        st.error("❌ Could not load data. Check your connection or refresh.")
        return

    # Build filters FIRST (uses raw team/player lists before metric computation)
    filters = build_filters(raw_df)

    # Compute metrics (park blend depends on toggle)
    df = compute_metrics(raw_df, use_park=filters['use_park'])

    # Total hit prob is used in multiple places
    df['total_hit_prob'] = (df['p_1b'] + df['p_xb'] + df['p_hr']).clip(upper=100).round(1)

    # Compute scores
    df = compute_scores(df)

    # Dashboard
    render_data_dashboard(df)

    # Apply filters
    filtered_df = apply_filters(df, filters)

    # Park notice
    render_park_notice(filtered_df if not filtered_df.empty else df, filters)

    # Score summary cards
    if not filtered_df.empty:
        render_score_summary_cards(filtered_df, filters)
        st.markdown("---")

    # Result count header
    score_col = filters['score_col']
    target_labels = {
        'Hit_Score':    '🎯 Any Base Hit',
        'Single_Score': '1️⃣ Single',
        'XB_Score':     '🔥 Extra Base Hit',
        'HR_Score':     '💣 Home Run',
    }
    st.subheader(f"Top {len(filtered_df)} {target_labels.get(score_col, 'Hit')} Candidates")

    # Results table
    render_results_table(filtered_df, filters)

    # Profile diversity
    if not filtered_df.empty:
        render_profile_diversity(filtered_df, filters['use_park'])

    # Charts
    if not filtered_df.empty:
        render_visualizations(df, filtered_df, score_col)

    # ── Controls row ──────────────────────────────────────────────────────────
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.data_loaded_at = datetime.now()
            st.rerun()

    with col2:
        if not filtered_df.empty:
            csv_data = filtered_df.to_csv(index=False)
            st.download_button(
                "💾 Export Results (CSV)",
                csv_data,
                f"a1picks_mlb_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

    with col3:
        sb = staleness_badge()
        if sb:
            st.markdown(sb, unsafe_allow_html=True)

    # ── Quick Lineup Exclusions ───────────────────────────────────────────────
    if not filtered_df.empty:
        with st.expander("⚡ Quick Player Exclusions"):
            st.markdown("Quickly exclude players from the current results:")
            result_players   = filtered_df['Batter'].tolist()
            current_excl     = st.session_state.excluded_players
            available        = [p for p in result_players if p not in current_excl]

            if available:
                cl, cr = st.columns(2)
                for i, player in enumerate(available[:5]):
                    with cl:
                        if st.button(f"❌ {player}", key=f"qx_{i}"):
                            if player not in st.session_state.excluded_players:
                                st.session_state.excluded_players.append(player)
                            st.rerun()
                for i, player in enumerate(available[5:10]):
                    with cr:
                        if st.button(f"❌ {player}", key=f"qx2_{i}"):
                            if player not in st.session_state.excluded_players:
                                st.session_state.excluded_players.append(player)
                            st.rerun()
            else:
                st.success("✅ All current results are confirmed playing.")

            if current_excl:
                st.markdown(f"**Excluded:** {', '.join(current_excl)}")
                if st.button("🔄 Re-include All", key="main_clear"):
                    st.session_state.excluded_players = []
                    st.rerun()

    # ── Strategy guide ────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("""
### 🎯 V4 Betting Strategy Guide

#### Score → Bet Type Mapping
| Score | Use For | K% Tolerance | Key Stats |
|-------|---------|-------------|-----------|
| 🎯 Hit Score | Any base hit prop | Low — K kills hits | 1B Prob + XB Prob + HR Prob |
| 1️⃣ Single Score | Hit a single | Low — also penalises high XB/HR% | 1B Prob dominant |
| 🔥 XB Score | Double or Triple | Medium — power contact acceptable | XB Prob dominant |
| 💣 HR Score | Home run | High — power hitters K more, that's ok | HR Prob dominant |

#### Park Toggle
- **ON** (default): Blends park-adjusted + base probabilities. Best for real games — accounts for
  ballpark dimensions, wind, temperature.
- **OFF**: Pure player vs pitcher. Shows "true" skill matchup stripped of environment.
- The **Park Δ** column shows exactly how much park factors influenced each player's score.

#### Key Tips
- For **cash game / safe** bets: sort by Hit Score, target players with high vs Grade and low K%
- For **singles props**: use Single Score — the negative weight on XB/HR% is intentional
- For **doubles/triples**: XB Score with a positive vs Grade is the sweet spot
- For **HR props**: HR Score with forgiving K% filter — don't penalise power hitters for striking out

**V4.0 — Ground-Up Rebuild | Single CSV | 4 Purpose-Built Scores | Park Toggle | Smooth Scoring**
    """)


# ─────────────────────────────────────────────────────────────────────────────
# REFERENCE PAGE
# ─────────────────────────────────────────────────────────────────────────────

def info_page():
    st.title("📚 A1PICKS MLB Hit Predictor — Reference Manual")

    with st.expander("📖 System Overview", expanded=True):
        st.markdown(f"""
## Purpose
Extract the best betting targets from BallPark Pal's simulation data (3,000 runs per game).
BallPark Pal already handles weather, park factors, recent performance, and pitch-mix matchups.
This tool weights and filters that output to surface the best targets for four specific bet types.

## Data Source
Single `Matchups.csv` from BallPark Pal. Each row = one batter vs one pitcher matchup.

| Column Group | Meaning |
|---|---|
| `HR/XB/1B/BB/K Prob` | Probability % with park factors |
| `HR/XB/1B/BB/K Prob (no park)` | Same, without park/weather adjustment |
| `HR/XB/1B/BB/K Boost` | BallPark Pal's matchup adjustment factor |
| `RC` / `RC (no park)` | Runs Created — overall matchup quality indicator |
| `vs Grade` | Batter vs pitcher on-paper rating (−10 to +10) |
| `PA, AVG, H, HR (hist)...` | Historical PA this batter has vs this pitcher |

## Four Scores

### 🎯 Hit Score
Optimised for **any base hit** prop.
- p_1b × 3.0 + p_xb × 2.0 + p_hr × 1.0 − p_k × 2.5 − p_bb × 1.0
- K% carries the heaviest penalty — a strikeout guarantees no hit.

### 1️⃣ Single Score
Optimised for **single specifically**.
- p_1b × 5.0 − p_k × 2.5 − p_bb × 1.0 − p_xb × 0.8 − p_hr × 0.5
- High XB% and HR% are penalised: a ball hit with that authority doesn't stay a single.

### 🔥 XB Score
Optimised for **double or triple**.
- p_xb × 5.0 + p_hr × 0.8 − p_k × 1.5 − p_bb × 1.0
- Moderate K% tolerance — harder contact correlates with more swing-and-miss.

### 💣 HR Score
Optimised for **home run**.
- p_hr × 6.0 + p_xb × 0.8 − p_k × 0.8 − p_bb × 1.0
- Light K% penalty — power hitters naturally strike out more; that's acceptable here.

## Park Adjustment
Blended mode averages the with-park and no-park probability columns before scoring.
The **Park Δ** column shows the exact score impact of park factors for each player.

## Historical Tiebreaker
When a batter has ≥ {CONFIG['hist_min_pa']} plate appearances against this specific pitcher,
their batting average vs that pitcher adds up to {CONFIG['hist_bonus_max']} bonus points
to the raw score before normalisation. This is a small tiebreaker only — it won't override
the simulation probabilities.

## vs Grade
BallPark Pal's on-paper rating of this specific batter vs this specific pitcher (−10 to +10),
based on pitch mix, batter tendencies, and location tendencies. Applied as a small modifier
(±2 raw points) to all four scores.
        """)


# ─────────────────────────────────────────────────────────────────────────────
# NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

def main():
    st.sidebar.markdown("---")

    # Optional music
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

    page = st.sidebar.radio(
        "Navigate",
        ["⚾ Hit Predictor", "📚 Reference Manual"],
        index=0
    )

    if page == "⚾ Hit Predictor":
        main_page()
    else:
        info_page()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**V4.0** | Ground-Up Rebuild | 4 Scores | Park Toggle")


if __name__ == "__main__":
    main()
