import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime, timedelta
import time

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro",
    layout="wide",
    page_icon="⚾",
    menu_items={
        'Get Help': 'mailto:your@email.com',
        'Report a bug': "https://github.com/yourrepo/issues",
    }
)

# Constants and Configuration
CONFIG = {
    'csv_urls': {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_walks': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
        'pitcher_hrs': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
        'pitcher_hits': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv'
    },
    'base_hit_weights': {
        'adj_1B': 2.0,
        'adj_XB': 1.8,
        'adj_vs': 1.2,
        'adj_RC': 0.8,
        'adj_HR': 0.6,
        'adj_K': -2.0,
        'adj_BB': -0.8
    },
    'expected_columns': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
    'pitcher_columns': {
        'walks': ['Team', 'Name', 'Park', 'Prob'],
        'hrs': ['Team', 'Name', 'Park', 'Prob'],
        'hits': ['Team', 'Name', 'Park', 'Prob']
    },
    'cache_ttl': 900  # 15 minutes
}

# Custom CSS
st.markdown("""
<style>
    .reportview-container .main .block-container { padding-top: 2rem; }
    .sidebar .sidebar-content { padding-top: 2.5rem; }
    .stRadio [role=radiogroup] { align-items: center; gap: 0.5rem; }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem; border-radius: 10px; color: white;
        margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem; border-radius: 10px; color: white; margin: 0.5rem 0;
    }
    .score-card-hit {
        background: linear-gradient(135deg, #1a9641 0%, #52b788 100%);
        padding: 1rem; border-radius: 10px; color: white;
        margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-card-xb {
        background: linear-gradient(135deg, #e67e22 0%, #f39c12 100%);
        padding: 1rem; border-radius: 10px; color: white;
        margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .score-card-hr {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 1rem; border-radius: 10px; color: white;
        margin: 0.5rem 0; box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .staleness-fresh {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 0.5rem 1rem; border-radius: 8px; color: white;
        font-weight: bold; text-align: center; margin-bottom: 0.5rem;
    }
    .staleness-aging {
        background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%);
        padding: 0.5rem 1rem; border-radius: 8px; color: #333;
        font-weight: bold; text-align: center; margin-bottom: 0.5rem;
    }
    .staleness-stale {
        background: linear-gradient(135deg, #c0392b 0%, #e74c3c 100%);
        padding: 0.5rem 1rem; border-radius: 8px; color: white;
        font-weight: bold; text-align: center; margin-bottom: 0.5rem;
    }
    .color-legend {
        margin: 1rem 0; padding: 1rem; background: #000000;
        border-radius: 8px; color: white !important;
    }
    .hit-probability { font-size: 1.2em; font-weight: bold; color: #28a745; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# DATA LOADING & VALIDATION
# =============================================================================

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_csv_with_validation(url, description, expected_columns, key_columns=None):
    """Load and validate CSV data with comprehensive error handling."""
    try:
        with st.spinner(f'Loading {description}...'):
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                st.error(f"❌ {description}: No data found")
                return None

            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                st.error(f"❌ {description}: Missing columns {missing_cols}")
                return None

            key_cols = key_columns if key_columns is not None else ['Tm', 'Batter', 'Pitcher']
            existing_key_cols = [col for col in key_cols if col in df.columns]
            if existing_key_cols:
                null_counts = df[existing_key_cols].isnull().sum()
                if null_counts.any():
                    problematic_cols = null_counts[null_counts > 0].index.tolist()
                    st.error(f"❌ {description}: Null values in {problematic_cols}")
                    return None

            return df

    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading {description}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing {description}: {str(e)}")
        return None


def validate_merge_quality(prob_df, pct_df, merged_df):
    """Validate the quality of the merge operation."""
    original_count = len(prob_df)
    merged_count = len(merged_df)

    if merged_count < original_count:
        lost_records = original_count - merged_count
        st.warning(f"⚠️ Lost {lost_records} records during merge ({lost_records/original_count*100:.1f}%)")

    merge_keys = ['Tm', 'Batter', 'Pitcher']
    duplicates = merged_df.duplicated(subset=merge_keys).sum()
    if duplicates > 0:
        st.error(f"🔴 Found {duplicates} duplicate matchups after merge")

    return merged_df


# =============================================================================
# PITCHER MATCHUP GRADES — SMOOTH INTERPOLATION + MULTIPLICATIVE MODIFIER
# =============================================================================

def calculate_pitcher_matchup_grades(df, profile_type):
    """
    Calculate pitcher matchup multipliers and grades using smooth linear interpolation.

    CHANGES FROM V3:
    - Hard threshold step functions replaced with np.interp smooth curves.
      Example: previously 24.9% HR prob → 0 bonus, 25.0% → +15 (cliff).
      Now the bonus scales continuously — no more cliff effects.
    - Raw modifier mapped to a MULTIPLIER (0.85–1.18) for proportional impact.
      A+ matchup multiplies score by 1.18. D matchup multiplies by 0.85.
      This means elite players benefit MORE from great matchups (as they should).
    """
    # Initialize neutral defaults
    df['pitcher_matchup_modifier'] = 0.0
    df['pitcher_multiplier'] = 1.0
    df['pitcher_matchup_grade'] = 'B'

    required_cols = ['Walk_3Plus_Probability', 'HR_2Plus_Probability', 'Hit_8Plus_Probability']
    if not all(col in df.columns for col in required_cols):
        return df

    # Fill missing with neutral league averages
    df['Walk_3Plus_Probability'] = df['Walk_3Plus_Probability'].fillna(15.0)
    df['HR_2Plus_Probability']   = df['HR_2Plus_Probability'].fillna(12.0)
    df['Hit_8Plus_Probability']  = df['Hit_8Plus_Probability'].fillna(18.0)

    # Walk penalty — smooth, applies to ALL profiles
    # More walks = fewer PA ending in hits. Scales 0 to -8 across 0%–50% walk rate.
    walk_penalty = np.interp(
        df['Walk_3Plus_Probability'].values,
        [0,  20,  30,  40,  50],
        [0,  -2,  -4,  -6,  -8]
    )
    df['pitcher_matchup_modifier'] = walk_penalty

    if profile_type == 'power':
        # HR bonus — smooth, power profiles only
        # HR-prone pitchers are better targets for power hitters.
        # Scales from 0 (at 0% HR prob) to +15 (at 25%+ HR prob).
        hr_bonus = np.interp(
            df['HR_2Plus_Probability'].values,
            [0,  10,  15,  20,  25],
            [0,   4,   8,  12,  15]
        )
        df['pitcher_matchup_modifier'] += hr_bonus
    else:
        # Hit bonus — smooth, contact profiles only
        # Hit-friendly pitchers are better targets for contact hitters.
        # Scales from 0 (at 0% hit prob) to +12 (at 25%+ hit prob).
        hit_bonus = np.interp(
            df['Hit_8Plus_Probability'].values,
            [0,  10,  15,  20,  25],
            [0,   3,   6,   9,  12]
        )
        df['pitcher_matchup_modifier'] += hit_bonus

    # Map raw modifier → proportional multiplier
    # Modifier range -8 to +15 → Multiplier 0.85 to 1.18
    df['pitcher_multiplier'] = np.interp(
        df['pitcher_matchup_modifier'].values,
        [-8,   -5,   0,    5,    10,   15],
        [0.85, 0.91, 1.0,  1.08, 1.13, 1.18]
    )

    # Grade from multiplier
    conditions = [
        df['pitcher_multiplier'] >= 1.12,
        df['pitcher_multiplier'] >= 1.06,
        df['pitcher_multiplier'] >= 0.97,
        df['pitcher_multiplier'] >= 0.90
    ]
    df['pitcher_matchup_grade'] = np.select(
        conditions, ['A+', 'A', 'B', 'C'], default='D'
    )

    return df


# =============================================================================
# SCORING ENGINE — COMPOSITE + THREE PURPOSE-BUILT SCORES
# =============================================================================

def calculate_league_aware_scores(df, profile_type='contact'):
    """
    Four output scores:

      Score     — General composite (profile-aware, V3 structure preserved)
      Hit_Score — Optimized for base hits (singles focus, heavy K penalty)
      XB_Score  — Optimized for extra base hits (doubles/triples focus)
      HR_Score  — Optimized for home runs (HR focus, light K penalty)

    CHANGES FROM V3:
    1. Pitcher modifier is now MULTIPLICATIVE not additive.
       (base_score + bonuses) × pitcher_multiplier
       Proportional effect: elite players gain more from A+ matchups.
    2. Three dedicated scores surface the right player for each bet type
       without needing to reverse-engineer a single blended number.
    3. Composite Score structure (weights, bonuses, league anchoring) is
       preserved exactly from V3 — only the pitcher application changed.
    """

    df = calculate_pitcher_matchup_grades(df, profile_type)

    if 'pitcher_multiplier' not in df.columns:
        df['pitcher_multiplier'] = 1.0

    # -------------------------------------------------------------------------
    # COMPOSITE SCORE — Profile-aware (V3 structure preserved)
    # -------------------------------------------------------------------------
    if profile_type == 'power':
        weights = {
            'adj_XB': 3.0,
            'adj_HR': 2.5,
            'adj_vs': 1.5,
            'adj_RC': 1.0,
            'adj_1B': 0.5,
            'adj_K':  -1.0,
            'adj_BB': -0.3
        }
        df['power_bonus']           = np.where((df['adj_XB'] > 10) & (df['adj_HR'] > 4), 12, 0)
        df['clutch_power_bonus']    = np.where((df['adj_XB'] > 8)  & (df['adj_vs'] > 5), 8, 0)
        df['consistent_power_bonus']= np.where((df['adj_HR'] > 3)  & (df['adj_K'] < 25), 5, 0)
        df['power_prob']            = (df['adj_XB'] + df['adj_HR']).clip(upper=50)
        bonus_cols = ['power_bonus', 'clutch_power_bonus', 'consistent_power_bonus']
    else:
        weights = {
            'adj_1B': 2.0,
            'adj_XB': 1.8,
            'adj_vs': 1.2,
            'adj_RC': 0.8,
            'adj_HR': 0.6,
            'adj_K':  -2.0,
            'adj_BB': -0.8
        }
        df['contact_bonus']     = np.where((df['total_hit_prob'] > 40) & (df['adj_K'] < 18), 8, 0)
        df['consistency_bonus'] = np.where((df['adj_1B'] > 20) & (df['adj_XB'] > 8), 5, 0)
        df['matchup_bonus']     = np.where(df['adj_vs'] > 5, 3, 0)
        bonus_cols = ['contact_bonus', 'consistency_bonus', 'matchup_bonus']

    df['base_score'] = sum(df[col] * w for col, w in weights.items() if col in df.columns)
    bonus_sum = sum(df[col] for col in bonus_cols if col in df.columns)

    # Multiplicative pitcher application
    df['Score'] = (df['base_score'] + bonus_sum) * df['pitcher_multiplier']
    s_min, s_max = df['Score'].min(), df['Score'].max()
    df['Score'] = ((df['Score'] - s_min) / (s_max - s_min) * 100) if s_max != s_min else 50

    # -------------------------------------------------------------------------
    # HIT_SCORE — Optimized for base hits
    # Primary: adj_1B (singles — the purest base hit signal)
    # Secondary: adj_XB + adj_HR (also count as base hits)
    # Penalty: adj_K gets the heaviest weight of the three scores
    #          because strikeouts are the #1 enemy of base hits
    # -------------------------------------------------------------------------
    hit_weights = {
        'adj_1B': 3.5,
        'adj_XB': 2.0,
        'adj_HR': 1.0,
        'adj_vs': 1.5,
        'adj_RC': 0.8,
        'adj_K':  -2.5,
        'adj_BB': -1.0
    }
    df['hit_base'] = sum(df[col] * w for col, w in hit_weights.items() if col in df.columns)
    df['Hit_Score'] = df['hit_base'] * df['pitcher_multiplier']
    h_min, h_max = df['Hit_Score'].min(), df['Hit_Score'].max()
    df['Hit_Score'] = ((df['Hit_Score'] - h_min) / (h_max - h_min) * 100) if h_max != h_min else 50

    # -------------------------------------------------------------------------
    # XB_SCORE — Optimized for extra base hits (doubles / triples)
    # Primary: adj_XB rate directly from BallPark Pal simulations
    # Moderate K penalty — XB hitters can carry higher K rates than contact guys
    # -------------------------------------------------------------------------
    xb_weights = {
        'adj_XB': 5.0,
        'adj_vs': 1.5,
        'adj_RC': 1.0,
        'adj_1B': 0.5,
        'adj_HR': 0.3,
        'adj_K':  -1.5,
        'adj_BB': -0.5
    }
    df['xb_base'] = sum(df[col] * w for col, w in xb_weights.items() if col in df.columns)
    df['XB_Score'] = df['xb_base'] * df['pitcher_multiplier']
    xb_min, xb_max = df['XB_Score'].min(), df['XB_Score'].max()
    df['XB_Score'] = ((df['XB_Score'] - xb_min) / (xb_max - xb_min) * 100) if xb_max != xb_min else 50

    # -------------------------------------------------------------------------
    # HR_SCORE — Optimized for home runs
    # Primary: adj_HR anchored to 2024 MLB league averages (your existing logic)
    # Light K penalty — HR hitters naturally strike out more; penalizing them
    # heavily would wrongly discount legitimate power bats
    # -------------------------------------------------------------------------
    hr_weights = {
        'adj_HR': 5.5,
        'adj_XB': 1.5,
        'adj_vs': 1.5,
        'adj_RC': 0.8,
        'adj_1B': 0.2,
        'adj_K':  -0.8,
        'adj_BB': -0.3
    }
    df['hr_base'] = sum(df[col] * w for col, w in hr_weights.items() if col in df.columns)
    df['HR_Score'] = df['hr_base'] * df['pitcher_multiplier']
    hr_min, hr_max = df['HR_Score'].min(), df['HR_Score'].max()
    df['HR_Score'] = ((df['HR_Score'] - hr_min) / (hr_max - hr_min) * 100) if hr_max != hr_min else 50

    return df.round(1)


# =============================================================================
# DATA PIPELINE
# =============================================================================

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_and_process_data():
    """Enhanced data loading with pitcher matchup data integration."""

    prob_df = load_csv_with_validation(
        CONFIG['csv_urls']['probabilities'],
        "Base Probabilities",
        CONFIG['expected_columns'],
        key_columns=['Tm', 'Batter', 'Pitcher']
    )
    pct_df = load_csv_with_validation(
        CONFIG['csv_urls']['percent_change'],
        "Adjustment Factors",
        CONFIG['expected_columns'],
        key_columns=['Tm', 'Batter', 'Pitcher']
    )

    if prob_df is None or pct_df is None:
        st.error("❌ Failed to load required data files")
        return None

    try:
        merged_df = pd.merge(
            prob_df, pct_df,
            on=['Tm', 'Batter', 'Pitcher'],
            suffixes=('_prob', '_pct'),
            how='inner'
        )
        merged_df = validate_merge_quality(prob_df, pct_df, merged_df)
    except Exception as e:
        st.error(f"❌ Failed to merge datasets: {str(e)}")
        return None

    pitcher_data = load_pitcher_matchup_data()

    if pitcher_data is not None and not pitcher_data.empty:
        try:
            merged_df = pd.merge(
                merged_df, pitcher_data,
                left_on=['Tm', 'Pitcher'],
                right_on=['Opponent_Team', 'Pitcher_Name'],
                how='left'
            )
        except Exception as e:
            st.warning(f"⚠️ Pitcher data merge failed: {str(e)} - continuing without pitcher bonuses")
    else:
        merged_df['Walk_3Plus_Probability'] = 15.0
        merged_df['HR_2Plus_Probability']   = 12.0
        merged_df['Hit_8Plus_Probability']  = 18.0
        st.info("ℹ️ Using base analysis - pitcher matchup data not available")

    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']

    for metric in metrics:
        base_col = f'{metric}.1' if metric in ['K', 'BB'] else f'{metric}_prob'
        pct_col  = f'{metric}_pct'

        if base_col in merged_df.columns and pct_col in merged_df.columns:
            merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col] / 100)
            if metric in ['K', 'BB', '1B', 'XB', 'HR']:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            else:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
        else:
            st.warning(f"⚠️ Using fallback for {metric} - columns {base_col} or {pct_col} not found")
            merged_df[f'adj_{metric}'] = 20 if metric in ['K', 'BB'] else 0

    merged_df['total_hit_prob'] = (merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']).clip(upper=100)
    merged_df['power_combo']    = merged_df['adj_XB'] + merged_df['adj_HR']

    return merged_df


def load_pitcher_matchup_data():
    """Load and combine all pitcher matchup datasets."""
    try:
        walks_df = load_csv_with_validation(
            CONFIG['csv_urls']['pitcher_walks'], "Pitcher Walks Data",
            CONFIG['pitcher_columns']['walks'], key_columns=['Team', 'Name']
        )
        hrs_df = load_csv_with_validation(
            CONFIG['csv_urls']['pitcher_hrs'], "Pitcher HR Data",
            CONFIG['pitcher_columns']['hrs'], key_columns=['Team', 'Name']
        )
        hits_df = load_csv_with_validation(
            CONFIG['csv_urls']['pitcher_hits'], "Pitcher Hits Data",
            CONFIG['pitcher_columns']['hits'], key_columns=['Team', 'Name']
        )

        if walks_df is None or hrs_df is None or hits_df is None:
            st.info("ℹ️ Some pitcher data unavailable - using base analysis only")
            return None

        def clean_prob_column(df, new_col_name):
            df = df.copy()
            df['Prob_Clean'] = pd.to_numeric(
                df['Prob'].astype(str).str.replace('%', '').str.strip(),
                errors='coerce'
            )
            df = df.rename(columns={'Prob_Clean': new_col_name})
            df['Pitcher_LastName'] = df['Name'].str.split().str[-1]
            return df[['Team', 'Name', 'Pitcher_LastName', 'Park', new_col_name]]

        walks_clean = clean_prob_column(walks_df, 'Walk_3Plus_Probability')
        hrs_clean   = clean_prob_column(hrs_df,   'HR_2Plus_Probability')
        hits_clean  = clean_prob_column(hits_df,  'Hit_8Plus_Probability')

        pitcher_data = pd.merge(walks_clean, hrs_clean,
                                on=['Team', 'Pitcher_LastName', 'Park'], how='outer')
        pitcher_data = pd.merge(pitcher_data, hits_clean,
                                on=['Team', 'Pitcher_LastName', 'Park'], how='outer')

        pitcher_data['Walk_3Plus_Probability'] = pitcher_data['Walk_3Plus_Probability'].fillna(15)
        pitcher_data['HR_2Plus_Probability']   = pitcher_data['HR_2Plus_Probability'].fillna(12)
        pitcher_data['Hit_8Plus_Probability']  = pitcher_data['Hit_8Plus_Probability'].fillna(18)

        pitcher_data = pitcher_data.rename(columns={
            'Team':           'Pitcher_Team',
            'Pitcher_LastName':'Pitcher_Name',
            'Park':            'Opponent_Team'
        })

        return pitcher_data

    except Exception as e:
        st.warning(f"⚠️ Could not load pitcher data: {str(e)} - continuing with base analysis")
        return None


# =============================================================================
# FILTERS
# =============================================================================

def create_league_aware_filters(df=None):
    """Create baseball-intelligent filtering system based on league averages and player types."""
    st.sidebar.header("🎯 Baseball-Smart Filters")

    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []

    if st.session_state.get('clear_exclusions', False):
        st.session_state.excluded_players = []
        st.session_state.clear_exclusions = False

    if 'quick_exclude_players' in st.session_state:
        for player in st.session_state.quick_exclude_players:
            if player not in st.session_state.excluded_players:
                st.session_state.excluded_players.append(player)
        st.session_state.quick_exclude_players = []

    LEAGUE_K_AVG  = 22.6
    LEAGUE_BB_AVG = 8.5
    filters = {}

    st.sidebar.markdown("### **📊 2024 League Context**")
    st.sidebar.markdown(f"- **K% League Avg**: {LEAGUE_K_AVG}%\n- **BB% League Avg**: {LEAGUE_BB_AVG}%")

    if df is not None and not df.empty:
        st.sidebar.markdown(f"**📈 Today's Pool:** {len(df)} matchups")
        avg_k  = df['adj_K'].mean()  if 'adj_K'  in df.columns else 0
        avg_bb = df['adj_BB'].mean() if 'adj_BB' in df.columns else 0
        st.sidebar.markdown(f"**Today's Avg K%:** {avg_k:.1f}%")
        st.sidebar.markdown(f"**Today's Avg BB%:** {avg_bb:.1f}%")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### **🎯 Player Type Focus**")

    player_type_options = {
        "🏆 Contact-Aggressive Hitters": {
            'description': "Low K% + Low BB% (Elite for base hits)",
            'max_k': 19.0, 'max_bb': 7.0, 'min_hit_prob': 32, 'profile_type': 'contact'
        },
        "⭐ Elite Contact Specialists": {
            'description': "Ultra-low K% (Pure contact)",
            'max_k': 14.0, 'max_bb': 9.5, 'min_hit_prob': 28, 'profile_type': 'contact'
        },
        "⚡ Swing-Happy Hitters": {
            'description': "Ultra-low BB% (Aggressive approach)",
            'max_k': 24.0, 'max_bb': 5.0, 'min_hit_prob': 30, 'profile_type': 'contact'
        },
        "🔷 Above-Average Contact": {
            'description': "Better than league average K%",
            'max_k': 20.0, 'max_bb': 12.0, 'min_hit_prob': 25, 'profile_type': 'contact'
        },
        "💥 Contact Power Hitters": {
            'description': "Low K% + High XB% & HR% (Power with contact)",
            'max_k': 20.0, 'max_bb': 12.0, 'min_xb': 7.0, 'min_hr': 2.5,
            'min_vs': -5, 'profile_type': 'power'
        },
        "🚀 Pure Power Sluggers": {
            'description': "High XB% & HR% (Power over contact)",
            'max_k': 100, 'max_bb': 100, 'min_xb': 9.0, 'min_hr': 3.5,
            'min_vs': -10, 'profile_type': 'power'
        },
        "⚾ All Power Players": {
            'description': "All players ranked by power potential (Research mode)",
            'max_k': 100, 'max_bb': 100, 'min_xb': 0, 'min_hr': 0,
            'min_vs': -10, 'profile_type': 'power'
        },
        "🌐 All Players": {
            'description': "No restrictions",
            'max_k': 100, 'max_bb': 100, 'min_hit_prob': 20, 'profile_type': 'all'
        }
    }

    selected_type = st.sidebar.selectbox(
        "Choose Hitter Profile",
        options=list(player_type_options.keys()),
        index=0,
        help="Each profile targets different hitting approaches based on league averages"
    )

    type_config = player_type_options[selected_type]
    filters['max_k']        = type_config['max_k']
    filters['max_bb']       = type_config['max_bb']
    filters['profile_type'] = type_config['profile_type']

    if type_config['profile_type'] == 'power':
        filters['min_xb']       = type_config.get('min_xb', 0)
        filters['min_hr']       = type_config.get('min_hr', 0)
        filters['min_vs']       = type_config.get('min_vs', -10)
        filters['min_hit_prob'] = 0
    else:
        filters['min_hit_prob'] = type_config.get('min_hit_prob', 20)
        filters['min_xb'] = 0
        filters['min_hr'] = 0
        filters['min_vs'] = -10

    st.sidebar.markdown(f"**📋 {selected_type}**")
    st.sidebar.markdown(f"*{type_config['description']}*")
    st.sidebar.markdown(f"- Max K%: {filters['max_k']:.1f}%")
    st.sidebar.markdown(f"- Max BB%: {filters['max_bb']:.1f}%")

    if type_config['profile_type'] == 'power':
        st.sidebar.markdown(f"- Min XB%: {filters['min_xb']:.1f}%")
        st.sidebar.markdown(f"- Min HR%: {filters['min_hr']:.1f}%")
        st.sidebar.markdown(f"- Min vs Pitcher: {filters['min_vs']}")
    else:
        st.sidebar.markdown(f"- Min Hit Prob: {filters['min_hit_prob']}%")

    # ADVANCED OPTIONS
    with st.sidebar.expander("⚙️ Fine-Tune Filters"):
        max_k_value  = float(max(5.0,  min(35.0, filters['max_k'])))
        max_bb_value = float(max(2.0,  min(15.0, filters['max_bb'])))

        filters['custom_max_k'] = st.slider(
            "Custom Max K% Override", 5.0, 35.0, max_k_value, 0.5,
            help=f"League avg: {LEAGUE_K_AVG}% | Elite: ≤12.0%"
        )
        filters['custom_max_bb'] = st.slider(
            "Custom Max BB% Override", 2.0, 15.0, max_bb_value, 0.5,
            help=f"League avg: {LEAGUE_BB_AVG}% | Aggressive: ≤4.0%"
        )

        if filters['custom_max_k']  != filters['max_k']:  filters['max_k']  = filters['custom_max_k']
        if filters['custom_max_bb'] != filters['max_bb']: filters['max_bb'] = filters['custom_max_bb']

        filters['min_vs_pitcher'] = st.slider("vs Pitcher Rating", -10, 10, 0, 1,
            help="Matchup advantage/disadvantage vs this pitcher type")

        filters['best_per_team_only'] = st.checkbox(
            "🏟️ Show only best player per team", value=False,
            help="Filter to show only the highest-scoring player from each team"
        )

        team_options = sorted(df['Tm'].unique().tolist()) if df is not None and not df.empty else []

        filters['selected_teams'] = st.multiselect(
            "Filter by Teams (Include Only)", options=team_options,
            help="Leave empty to include all teams"
        )
        filters['excluded_teams'] = st.multiselect(
            "Exclude Teams", options=team_options,
            help="Select teams to completely exclude from analysis"
        )

        # Sort options — includes all three purpose-built scores
        st.markdown("**🔢 Sort Results**")
        sort_options = {
            "Score (High to Low)":          ("Score",          False),
            "Score (Low to High)":          ("Score",          True),
            "Hit Score (High to Low)":      ("Hit_Score",      False),
            "Hit Score (Low to High)":      ("Hit_Score",      True),
            "XB Score (High to Low)":       ("XB_Score",       False),
            "XB Score (Low to High)":       ("XB_Score",       True),
            "HR Score (High to Low)":       ("HR_Score",       False),
            "HR Score (Low to High)":       ("HR_Score",       True),
            "Hit Prob% (High to Low)":      ("total_hit_prob", False),
            "Hit Prob% (Low to High)":      ("total_hit_prob", True),
            "HR% (High to Low)":            ("adj_HR",         False),
            "HR% (Low to High)":            ("adj_HR",         True),
            "XB% (High to Low)":            ("adj_XB",         False),
            "XB% (Low to High)":            ("adj_XB",         True),
            "Contact% (High to Low)":       ("adj_1B",         False),
            "Contact% (Low to High)":       ("adj_1B",         True),
            "K% (Low to High)":             ("adj_K",          True),
            "K% (High to Low)":             ("adj_K",          False),
            "BB% (Low to High)":            ("adj_BB",         True),
            "BB% (High to Low)":            ("adj_BB",         False),
            "vs Pitcher (High to Low)":     ("adj_vs",         False),
            "vs Pitcher (Low to High)":     ("adj_vs",         True),
            "Power Combo (High to Low)":    ("power_combo",    False),
            "Power Combo (Low to High)":    ("power_combo",    True),
        }

        filters['primary_sort'] = st.selectbox(
            "Sort By", options=list(sort_options.keys()), index=0,
            help="Choose how to sort the results"
        )
        filters['sort_col'], filters['sort_asc'] = sort_options[filters['primary_sort']]

        filters['result_count'] = st.selectbox(
            "Number of Results", options=[5, 10, 15, 20, 25, 30, "All"], index=2,
            help="Choose how many results to display"
        )

    # LINEUP STATUS MANAGEMENT
    with st.sidebar.expander("🏟️ Lineup Status Management"):
        st.markdown("**Exclude players not in today's lineups:**")

        all_players = sorted(df['Batter'].unique().tolist()) if df is not None and not df.empty else []
        current_exclusions = st.session_state.excluded_players.copy()

        selected_exclusions = st.multiselect(
            "Players NOT Playing Today",
            options=all_players,
            default=current_exclusions,
            help="Select players who are confirmed out of lineups",
            key="lineup_exclusions"
        )

        st.session_state.excluded_players = selected_exclusions
        filters['excluded_players'] = selected_exclusions

        if selected_exclusions:
            st.info(f"🚫 Currently excluding {len(selected_exclusions)} players")

        if st.button("🔄 Clear All Exclusions", key="sidebar_clear"):
            st.session_state.excluded_players = []
            st.rerun()

        st.markdown("**Quick Exclude Options:**")
        filters['auto_exclude_injured'] = st.checkbox("🏥 Auto-exclude common injury-prone players")
        filters['show_lineup_warnings'] = st.checkbox("📊 Show lineup confidence warnings")

    # REAL-TIME SIDEBAR PREVIEW
    if df is not None and not df.empty:
        try:
            preview_df = df.copy()
            excluded_players = st.session_state.excluded_players
            if excluded_players:
                preview_df = preview_df[~preview_df['Batter'].isin(excluded_players)]

            excluded_teams = filters.get('excluded_teams', [])
            if excluded_teams:
                preview_df = preview_df[~preview_df['Tm'].isin(excluded_teams)]

            selected_teams = filters.get('selected_teams', [])
            if selected_teams:
                preview_df = preview_df[preview_df['Tm'].isin(selected_teams)]

            if filters.get('profile_type') == 'power':
                preview_query = (
                    f"adj_K <= {filters['max_k']:.1f} and "
                    f"adj_BB <= {filters['max_bb']:.1f} and "
                    f"adj_XB >= {filters['min_xb']:.1f} and "
                    f"adj_HR >= {filters['min_hr']:.1f} and "
                    f"adj_vs >= {filters['min_vs']}"
                )
            else:
                preview_query = (
                    f"adj_K <= {filters['max_k']:.1f} and "
                    f"adj_BB <= {filters['max_bb']:.1f} and "
                    f"total_hit_prob >= {filters['min_hit_prob']}"
                )

            preview_df = preview_df.query(preview_query)

            matching_count    = len(preview_df)
            excluded_count    = len(excluded_players)
            excluded_teams_count = len(excluded_teams)

            if matching_count == 0:
                st.sidebar.error("❌ No players match current profile")
                if excluded_count > 0:
                    st.sidebar.markdown(f"**💡 Note:** {excluded_count} players excluded")
                st.sidebar.markdown("**💡 Try:** Different player type or custom overrides")
            elif matching_count < 5:
                st.sidebar.warning(f"⚠️ Only {matching_count} players match")
                if excluded_count > 0:
                    st.sidebar.markdown(f"**📊 Pool:** {matching_count} playing + {excluded_count} excluded")
            else:
                st.sidebar.success(f"✅ {matching_count} players match profile")
                if excluded_count > 0:
                    st.sidebar.markdown(f"**📊 Lineup Status:** {matching_count} confirmed, {excluded_count} excluded")
                if excluded_teams_count > 0:
                    st.sidebar.markdown(f"**🚫 Excluded Teams:** {', '.join(excluded_teams)}")

                avg_k_filtered  = preview_df['adj_K'].mean()
                avg_bb_filtered = preview_df['adj_BB'].mean()
                k_vs_league     = LEAGUE_K_AVG  - avg_k_filtered
                bb_vs_league    = LEAGUE_BB_AVG - avg_bb_filtered

                result_count  = filters.get('result_count', 15)
                display_count = matching_count if result_count == "All" else min(matching_count, result_count)

                st.sidebar.markdown("**📊 vs League (Playing Players):**")
                st.sidebar.markdown(f"K%: {k_vs_league:+.1f}% {'better' if k_vs_league > 0 else 'worse'} than league")
                st.sidebar.markdown(f"BB%: {bb_vs_league:+.1f}% {'more aggressive' if bb_vs_league > 0 else 'less aggressive'} than league")

                if result_count == "All":
                    st.sidebar.markdown(f"**📋 Showing:** All {matching_count} players")
                else:
                    st.sidebar.markdown(f"**📋 Showing:** Top {display_count} of {matching_count}")

        except Exception as e:
            st.sidebar.warning(f"⚠️ Preview unavailable: {str(e)}")

    return filters


def apply_league_aware_filters(df, filters):
    """Apply baseball-intelligent filtering based on league averages and player types."""

    if df is None or df.empty:
        return df

    NUMERIC_SORT_COLS = [
        'adj_HR', 'adj_K', 'adj_XB', 'adj_1B', 'adj_BB', 'adj_vs',
        'Score', 'Hit_Score', 'XB_Score', 'HR_Score',
        'total_hit_prob', 'power_combo'
    ]

    excluded_players = st.session_state.get('excluded_players', [])
    if excluded_players:
        excluded_count = len(df[df['Batter'].isin(excluded_players)])
        df = df[~df['Batter'].isin(excluded_players)]
        if excluded_count > 0:
            st.info(f"🏟️ Excluded {excluded_count} players not in today's lineups")

    excluded_teams = filters.get('excluded_teams', [])
    if excluded_teams:
        excluded_team_players = len(df[df['Tm'].isin(excluded_teams)])
        df = df[~df['Tm'].isin(excluded_teams)]
        if excluded_team_players > 0:
            st.info(f"🚫 Excluded {excluded_team_players} players from teams: {', '.join(excluded_teams)}")

    query_parts = []

    if 'max_k' in filters and filters['max_k'] < 100:
        query_parts.append(f"adj_K <= {filters['max_k']:.1f}")
    if 'max_bb' in filters and filters['max_bb'] < 100:
        query_parts.append(f"adj_BB <= {filters['max_bb']:.1f}")

    if filters.get('profile_type') == 'power':
        if filters.get('min_xb', 0) > 0:
            query_parts.append(f"adj_XB >= {filters['min_xb']:.1f}")
        if filters.get('min_hr', 0) > 0:
            query_parts.append(f"adj_HR >= {filters['min_hr']:.1f}")
        if filters.get('min_vs', -10) > -10:
            query_parts.append(f"adj_vs >= {filters['min_vs']}")
    else:
        if filters.get('min_hit_prob', 0) > 0:
            query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']:.1f}")

    if filters.get('min_vs_pitcher', 0) != 0:
        query_parts.append(f"adj_vs >= {filters['min_vs_pitcher']}")

    if filters.get('selected_teams'):
        query_parts.append("Tm in " + str(filters['selected_teams']))
    if filters.get('excluded_teams'):
        query_parts.append("Tm not in " + str(filters['excluded_teams']))

    try:
        filtered_df = df.query(" and ".join(query_parts)) if query_parts else df.copy()

        if filters.get('best_per_team_only', False):
            filtered_df = filtered_df.loc[filtered_df.groupby('Tm')['Score'].idxmax()].copy()
            st.info(f"🏟️ Showing best player from each of {len(filtered_df['Tm'].unique())} teams")

        if 'power_combo' not in filtered_df.columns:
            filtered_df['power_combo'] = filtered_df['adj_XB'] + filtered_df['adj_HR']

        sort_col = filters.get('sort_col', 'Score')
        sort_asc = filters.get('sort_asc', False)

        try:
            if sort_col in NUMERIC_SORT_COLS and sort_col in filtered_df.columns:
                filtered_df[sort_col] = pd.to_numeric(filtered_df[sort_col], errors='coerce')
            filtered_df = filtered_df.sort_values(sort_col, ascending=sort_asc, na_position='last')
        except Exception as sort_error:
            st.warning(f"⚠️ Sorting failed, using default Score sort: {sort_error}")
            filtered_df = filtered_df.sort_values('Score', ascending=False)

        result_count = filters.get('result_count', 15)
        return filtered_df if result_count == "All" else filtered_df.head(result_count)

    except Exception as e:
        st.error(f"❌ Filter error: {str(e)}")
        if 'power_combo' not in df.columns:
            df['power_combo'] = df['adj_XB'] + df['adj_HR']
        try:
            sort_col = filters.get('sort_col', 'Score')
            sort_asc = filters.get('sort_asc', False)
            if sort_col in NUMERIC_SORT_COLS and sort_col in df.columns:
                df[sort_col] = pd.to_numeric(df[sort_col], errors='coerce')
            sorted_df = df.sort_values(sort_col, ascending=sort_asc, na_position='last')
        except Exception:
            sorted_df = df.sort_values('Score', ascending=False)
        result_count = filters.get('result_count', 15)
        return sorted_df if result_count == "All" else sorted_df.head(result_count)


# =============================================================================
# DISPLAY COMPONENTS
# =============================================================================

def create_league_aware_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', width=200)
    with col2:
        st.title("🎯 MLB League-Aware Hit Predictor Pro")
        st.markdown("*Find hitters with the best base hit probability using 2024 league context*")


def create_staleness_indicator():
    """
    Shows how fresh the cached data is. Green → Yellow → Red as data ages.
    Resets to green when the user clicks Refresh Data.
    """
    if 'data_load_time' not in st.session_state:
        st.session_state.data_load_time = datetime.now()

    elapsed   = int((datetime.now() - st.session_state.data_load_time).total_seconds())
    mins_old  = elapsed // 60
    secs_old  = elapsed % 60
    ttl_left  = max(0, CONFIG['cache_ttl'] - elapsed)
    ttl_mins  = ttl_left // 60

    if mins_old == 0:
        label     = f"🟢 Data is fresh — loaded {secs_old}s ago"
        css_class = "staleness-fresh"
    elif mins_old < 10:
        label     = f"🟡 Data is {mins_old}m {secs_old}s old — refreshes in ~{ttl_mins}m"
        css_class = "staleness-aging"
    else:
        label     = f"🔴 Data is {mins_old}m old — consider refreshing"
        css_class = "staleness-stale"

    st.markdown(f'<div class="{css_class}">{label}</div>', unsafe_allow_html=True)
    st.markdown("")


def create_data_quality_dashboard(df):
    if df is None or df.empty:
        st.error("No data available for quality analysis")
        return

    st.subheader("📊 Today's Data Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f'<div class="metric-card"><h3>📈 Total Matchups</h3><h2>{len(df)}</h2></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><h3>👤 Unique Batters</h3><h2>{df["Batter"].nunique()}</h2></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><h3>🏟️ Teams Playing</h3><h2>{df["Tm"].nunique()}</h2></div>',
                    unsafe_allow_html=True)
    with col4:
        avg_hit = df['total_hit_prob'].mean()
        st.markdown(f'<div class="success-card"><h3>🎯 Avg Hit Probability</h3><h2>{avg_hit:.1f}%</h2></div>',
                    unsafe_allow_html=True)


def display_league_aware_results(filtered_df, filters):
    """Display results with the three purpose-built scores and full league context."""

    LEAGUE_K_AVG  = 22.6
    LEAGUE_BB_AVG = 8.5

    if filtered_df.empty:
        st.warning("⚠️ No players match your current player type filters")
        st.markdown("""
        ### 💡 **Suggested Adjustments:**
        - Try **"Above-Average Contact"** for more options
        - Use **custom overrides** in advanced settings
        - Consider **"All Players"** to see the full pool
        """)
        return

    result_count  = filters.get('result_count', 15)
    best_per_team = filters.get('best_per_team_only', False)

    if best_per_team:
        st.subheader(f"🏟️ Best Player from Each Team ({len(filtered_df)} teams)"
                     if result_count == "All"
                     else f"🏟️ Top {len(filtered_df)} Teams - Best Player Each")
    else:
        st.subheader(f"🎯 All {len(filtered_df)} Base Hit Candidates"
                     if result_count == "All"
                     else f"🎯 Top {len(filtered_df)} Base Hit Candidates")

    # Key insights row
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        st.markdown(f'<div class="success-card"><h4>🥇 Best Hit Probability</h4>'
                    f'<h2>{best_hit_prob:.1f}%</h2><small>Target: 35%+</small></div>',
                    unsafe_allow_html=True)
    with col2:
        avg_k      = filtered_df['adj_K'].mean()
        k_vs_league = LEAGUE_K_AVG - avg_k
        color      = "success-card" if k_vs_league > 0 else "metric-card"
        st.markdown(f'<div class="{color}"><h4>⚾ K% vs League</h4>'
                    f'<h2>{k_vs_league:+.1f}%</h2><small>League: {LEAGUE_K_AVG}%</small></div>',
                    unsafe_allow_html=True)
    with col3:
        avg_bb       = filtered_df['adj_BB'].mean()
        bb_vs_league = LEAGUE_BB_AVG - avg_bb
        color        = "success-card" if bb_vs_league > 0 else "metric-card"
        st.markdown(f'<div class="{color}"><h4>🚶 BB% vs League</h4>'
                    f'<h2>{bb_vs_league:+.1f}%</h2><small>League: {LEAGUE_BB_AVG}%</small></div>',
                    unsafe_allow_html=True)
    with col4:
        if 'pitcher_matchup_grade' in filtered_df.columns and filtered_df['pitcher_matchup_grade'].notna().any():
            a_plus = (filtered_df['pitcher_matchup_grade'] == 'A+').sum()
            st.markdown(f'<div class="success-card"><h4>🎯 Elite Matchups</h4>'
                        f'<h2>{a_plus}/{len(filtered_df)}</h2><small>A+ Pitcher Spots</small></div>',
                        unsafe_allow_html=True)
        else:
            elite_k = (filtered_df['adj_K'] <= 12.0).sum()
            st.markdown(f'<div class="success-card"><h4>⭐ Elite Contact</h4>'
                        f'<h2>{elite_k}/{len(filtered_df)}</h2><small>K% ≤12.0%</small></div>',
                        unsafe_allow_html=True)

    # Three score leader cards
    st.markdown("---")
    st.markdown("#### 🎯 Purpose-Built Score Leaders")

    score_cols_exist = all(c in filtered_df.columns for c in ['Hit_Score', 'XB_Score', 'HR_Score'])

    if score_cols_exist:
        sc1, sc2, sc3 = st.columns(3)

        top_hit = filtered_df.loc[filtered_df['Hit_Score'].idxmax()]
        top_xb  = filtered_df.loc[filtered_df['XB_Score'].idxmax()]
        top_hr  = filtered_df.loc[filtered_df['HR_Score'].idxmax()]

        with sc1:
            st.markdown(f"""
            <div class="score-card-hit">
                <h4>🟢 Hit Score Leader</h4>
                <h3>{top_hit['Batter']}</h3>
                <p>Score: <strong>{top_hit['Hit_Score']:.1f}</strong> | Hit Prob: {top_hit['total_hit_prob']:.1f}%</p>
                <small>Best base hit target today</small>
            </div>""", unsafe_allow_html=True)

        with sc2:
            st.markdown(f"""
            <div class="score-card-xb">
                <h4>🟠 XB Score Leader</h4>
                <h3>{top_xb['Batter']}</h3>
                <p>Score: <strong>{top_xb['XB_Score']:.1f}</strong> | XB%: {top_xb['adj_XB']:.1f}%</p>
                <small>Best doubles/extra base target today</small>
            </div>""", unsafe_allow_html=True)

        with sc3:
            st.markdown(f"""
            <div class="score-card-hr">
                <h4>🔴 HR Score Leader</h4>
                <h3>{top_hr['Batter']}</h3>
                <p>Score: <strong>{top_hr['HR_Score']:.1f}</strong> | HR%: {top_hr['adj_HR']:.1f}%</p>
                <small>Best home run target today</small>
            </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Active profile + sort info
    filter_profile = "Custom"
    if filters.get('max_k', 100) <= 17 and filters.get('max_bb', 100) <= 6:
        filter_profile = "Contact-Aggressive Hitters"
    elif filters.get('max_k', 100) <= 12:
        filter_profile = "Elite Contact Specialists"
    elif filters.get('max_bb', 100) <= 4:
        filter_profile = "Swing-Happy Hitters"
    elif filters.get('max_k', 100) <= 17:
        filter_profile = "Above-Average Contact"
    elif filters.get('max_k', 100) >= 100:
        filter_profile = "All Players"

    col_p, col_s = st.columns(2)
    with col_p: st.markdown(f"**🎯 Active Profile:** {filter_profile}")
    with col_s: st.markdown(f"**🔢 Sorting:** {filters.get('primary_sort', 'Score (High to Low)')}")

    # Results table
    display_df = filtered_df.copy()
    display_df['K_vs_League']  = LEAGUE_K_AVG  - display_df['adj_K']
    display_df['BB_vs_League'] = LEAGUE_BB_AVG - display_df['adj_BB']

    if 'power_combo' not in display_df.columns:
        display_df['power_combo'] = display_df['adj_XB'] + display_df['adj_HR']

    excluded_players = st.session_state.get('excluded_players', [])
    display_df['Lineup_Status'] = display_df['Batter'].apply(
        lambda x: '🏟️' if x not in excluded_players else '❌'
    )

    display_columns = {'Lineup_Status': 'Status', 'Batter': 'Batter', 'Tm': 'Team', 'Pitcher': 'Pitcher'}

    if score_cols_exist:
        display_columns['Hit_Score'] = 'Hit Score'
        display_columns['XB_Score']  = 'XB Score'
        display_columns['HR_Score']  = 'HR Score'

    display_columns.update({
        'total_hit_prob': 'Hit Prob %',
        'adj_1B':         'Contact %',
        'adj_XB':         'XB %',
        'adj_HR':         'HR %',
        'power_combo':    'Power Combo %',
        'K_vs_League':    'K% vs League',
        'BB_vs_League':   'BB% vs League',
        'adj_vs':         'vs Pitcher',
        'Score':          'Score'
    })

    if 'pitcher_matchup_grade' in display_df.columns and display_df['pitcher_matchup_grade'].notna().any():
        display_columns['pitcher_matchup_grade'] = 'Matchup'

    available_cols = [c for c in display_columns if c in display_df.columns]
    rename_map     = {c: display_columns[c] for c in available_cols}
    styled_df      = display_df[available_cols].rename(columns=rename_map)

    fmt = {
        'Hit Prob %':    "{:.1f}%",
        'Contact %':     "{:.1f}%",
        'XB %':          "{:.1f}%",
        'HR %':          "{:.1f}%",
        'Power Combo %': "{:.1f}%",
        'K% vs League':  "{:+.1f}%",
        'BB% vs League': "{:+.1f}%",
        'vs Pitcher':    "{:.0f}",
        'Score':         "{:.1f}"
    }
    if score_cols_exist:
        fmt['Hit Score'] = "{:.1f}"
        fmt['XB Score']  = "{:.1f}"
        fmt['HR Score']  = "{:.1f}"

    styler = styled_df.style.format(fmt)
    styler = (styler
        .background_gradient(subset=['Score'],         cmap='RdYlGn',  vmin=0,   vmax=100)
        .background_gradient(subset=['Hit Prob %'],    cmap='Greens',  vmin=20,  vmax=50)
        .background_gradient(subset=['Power Combo %'], cmap='Oranges', vmin=0,   vmax=20)
        .background_gradient(subset=['K% vs League'],  cmap='RdYlGn',  vmin=-8,  vmax=12)
        .background_gradient(subset=['BB% vs League'], cmap='RdYlGn',  vmin=-5,  vmax=6)
    )

    if score_cols_exist:
        styler = (styler
            .background_gradient(subset=['Hit Score'], cmap='Greens',  vmin=0, vmax=100)
            .background_gradient(subset=['XB Score'],  cmap='Oranges', vmin=0, vmax=100)
            .background_gradient(subset=['HR Score'],  cmap='Reds',    vmin=0, vmax=100)
        )

    if 'Matchup' in styled_df.columns:
        def color_matchup_grade(val):
            colors = {
                'A+': 'background-color: #1a9641; color: white; font-weight: bold',
                'A':  'background-color: #a6d96a; color: black; font-weight: bold',
                'B':  'background-color: #ffffbf; color: black',
                'C':  'background-color: #fdae61; color: black',
                'D':  'background-color: #d7191c; color: white; font-weight: bold'
            }
            return colors.get(val, '')
        styler = styler.apply(
            lambda x: [color_matchup_grade(v) if x.name == 'Matchup' else '' for v in x], axis=0
        )

    st.dataframe(styler, use_container_width=True)

    # Legend
    matchup_guide = ""
    if 'Matchup' in styled_df.columns:
        matchup_guide = """<br>
        <strong>Matchup Grades (Smooth ×Multiplier):</strong>
        <span style="background-color:#1a9641;color:white;padding:2px 4px;border-radius:3px;">A+ ×1.18</span>
        <span style="background-color:#a6d96a;color:black;padding:2px 4px;border-radius:3px;">A ×1.08</span>
        <span style="background-color:#ffffbf;color:black;padding:2px 4px;border-radius:3px;">B ×1.00</span>
        <span style="background-color:#fdae61;color:black;padding:2px 4px;border-radius:3px;">C ×0.91</span>
        <span style="background-color:#d7191c;color:white;padding:2px 4px;border-radius:3px;">D ×0.85</span>"""

    st.markdown(f"""
    <div class="color-legend">
        <strong>📊 V4 Results Guide:</strong><br>
        <strong>Status:</strong> 🏟️ = Confirmed Playing | ❌ = Excluded<br>
        <strong>🟢 Hit Score:</strong> Optimized for base hits — highest adj_1B weight + heaviest K penalty<br>
        <strong>🟠 XB Score:</strong> Optimized for doubles/extra bases — driven by adj_XB from BallPark Pal sims<br>
        <strong>🔴 HR Score:</strong> Optimized for home runs — driven by adj_HR with light K penalty<br>
        <strong>Score:</strong> General composite (profile-aware) |
        <span style="color:#1a9641;">●</span> Elite 70+ |
        <span style="color:#fdae61;">●</span> Good 50-69 |
        <span style="color:#d7191c;">●</span> Risky &lt;50<br>
        <strong>K% vs League:</strong> <span style="color:#1a9641;">●</span> Positive = Better Contact |
        <span style="color:#d7191c;">●</span> Negative = More strikeouts<br>
        <strong>BB% vs League:</strong> <span style="color:#1a9641;">●</span> Positive = More Aggressive |
        <span style="color:#d7191c;">●</span> Negative = More patient{matchup_guide}
    </div>
    """, unsafe_allow_html=True)

    # Smart Profile Diversity Analysis
    if len(filtered_df) >= 3:
        st.markdown("### 🔍 **Smart Profile Diversity Analysis**")

        profile_criteria = {
            "🏆 Contact-Aggressive": {"max_k": 19.0, "max_bb": 7.0,  "icon": "🏆", "type": "contact"},
            "⭐ Elite Contact":       {"max_k": 14.0, "max_bb": 9.5,  "icon": "⭐", "type": "contact"},
            "⚡ Swing-Happy":         {"max_k": 24.0, "max_bb": 5.0,  "icon": "⚡", "type": "contact"},
            "🔷 Above-Average":       {"max_k": 20.0, "max_bb": 12.0, "icon": "🔷", "type": "contact"},
            "💥 Contact Power":       {"max_k": 20.0, "max_bb": 12.0, "min_xb": 7.0,  "min_hr": 2.5, "icon": "💥", "type": "power"},
            "🚀 Pure Power":          {"max_k": 100,  "max_bb": 100,  "min_xb": 9.0,  "min_hr": 3.5, "icon": "🚀", "type": "power"},
            "⚾ All Power":           {"max_k": 100,  "max_bb": 100,  "min_xb": 0,    "min_hr": 0,   "icon": "⚾", "type": "power"}
        }

        excl = st.session_state.get('excluded_players', [])

        def get_profile_players(criteria):
            mask = (
                (filtered_df['adj_K']  <= criteria['max_k']) &
                (filtered_df['adj_BB'] <= criteria['max_bb']) &
                (~filtered_df['Batter'].isin(excl))
            )
            if criteria["type"] == "power":
                mask &= (filtered_df['adj_XB'] >= criteria.get('min_xb', 0))
                mask &= (filtered_df['adj_HR'] >= criteria.get('min_hr', 0))
            return filtered_df[mask].copy()

        # Find overall best
        overall_best_player = None
        overall_best_score  = -1
        for profile_name, criteria in profile_criteria.items():
            pp = get_profile_players(criteria)
            if not pp.empty and pp.iloc[0]['Score'] > overall_best_score:
                overall_best_score  = pp.iloc[0]['Score']
                overall_best_player = pp.iloc[0]['Batter']

        player_usage = {'overall_best': overall_best_player, 'shown_contact': False, 'shown_power': False}
        profile_analysis = {}

        for profile_name, criteria in profile_criteria.items():
            pp = get_profile_players(criteria)
            if pp.empty:
                continue

            selected_player, selected_rank = None, 1

            for idx, candidate in pp.head(3).iterrows():
                is_best = candidate['Batter'] == player_usage['overall_best']
                if criteria["type"] == "power":
                    if not player_usage['shown_power'] or not is_best:
                        selected_player = candidate
                        selected_rank   = list(pp.head(3).index).index(idx) + 1
                        if is_best: player_usage['shown_power'] = True
                        break
                else:
                    if not (is_best and player_usage['shown_contact']):
                        selected_player = candidate
                        selected_rank   = list(pp.head(3).index).index(idx) + 1
                        if is_best: player_usage['shown_contact'] = True
                        break

            if selected_player is None:
                selected_player = pp.iloc[0]
                selected_rank   = 1

            overall_rank = filtered_df[filtered_df['Batter'] == selected_player['Batter']].index[0] + 1
            profile_analysis[profile_name] = {
                'player':           selected_player,
                'rank_overall':     overall_rank,
                'rank_in_profile':  selected_rank,
                'count_in_profile': len(pp)
            }

        if profile_analysis:
            st.markdown("**🎯 Top Player by Profile:**")

            shown_players      = set()
            profiles_to_display = {}
            for profile_name, analysis in profile_analysis.items():
                pname = analysis['player']['Batter']
                if pname not in shown_players or len(shown_players) < 3:
                    profiles_to_display[profile_name] = analysis
                    shown_players.add(pname)
                elif len(profiles_to_display) < 6:
                    profiles_to_display[profile_name] = analysis

            cols = st.columns(min(len(profiles_to_display), 4))

            for i, (profile_name, analysis) in enumerate(profiles_to_display.items()):
                player       = analysis['player']
                overall_rank = analysis['rank_overall']
                profile_rank = analysis['rank_in_profile']
                profile_count= analysis['count_in_profile']

                with cols[i % len(cols)]:
                    icon = profile_criteria[profile_name]['icon']
                    st.markdown(f"**{icon} {profile_name.split(' ', 1)[1]}**")

                    rank_label = f"#{overall_rank} overall" + (f", #{profile_rank} in profile" if profile_rank > 1 else "")
                    medal = "🥇" if overall_rank == 1 else "🥈" if overall_rank <= 3 else ""
                    label = f"{medal} **{player['Batter']}** ({rank_label})"

                    if overall_rank <= 3:
                        st.success(label)
                    else:
                        st.info(label)

                    if profile_rank > 1:
                        st.caption(f"💡 Showing #{profile_rank} for diversity")

                    if profile_criteria[profile_name]["type"] == "power":
                        xb_s  = f" | XB Score: {player['XB_Score']:.1f}"  if 'XB_Score'  in player.index else ""
                        hr_s  = f" | HR Score: {player['HR_Score']:.1f}"  if 'HR_Score'  in player.index else ""
                        st.markdown(
                            f"**XB%:** {player['adj_XB']:.1f}% | **HR%:** {player['adj_HR']:.1f}%  \n"
                            f"**Power Combo:** {(player['adj_XB']+player['adj_HR']):.1f}%{xb_s}{hr_s}  \n"
                            f"**vs Pitcher:** {player['adj_vs']:.0f} | **Score:** {player['Score']:.1f}"
                        )
                    else:
                        hit_s = f" | Hit Score: {player['Hit_Score']:.1f}" if 'Hit_Score' in player.index else ""
                        st.markdown(
                            f"**Hit Prob:** {player['total_hit_prob']:.1f}%{hit_s}  \n"
                            f"**K% vs League:** {LEAGUE_K_AVG - player['adj_K']:+.1f}%  \n"
                            f"**BB% vs League:** {LEAGUE_BB_AVG - player['adj_BB']:+.1f}%  \n"
                            f"**Score:** {player['Score']:.1f}"
                        )

                    st.caption(f"📊 {profile_count} players in profile")

            st.markdown("---")
            st.markdown("**📋 Profile Summary:**")

            best_overall  = max(profiles_to_display.values(), key=lambda x: x['player']['Score'])
            best_name     = best_overall['player']['Batter']
            best_profile  = next(k for k, v in profiles_to_display.items() if v['player']['Batter'] == best_name)
            all_unique    = {a['player']['Batter']: a['player'] for a in profiles_to_display.values()}

            insights = [f"🏆 **Overall Best**: {best_name} ({best_profile.split()[0]})"]

            elite_power = [n for n, p in all_unique.items() if p['adj_XB'] + p['adj_HR'] > 12]
            if elite_power:
                insights.append(f"💥 **Elite Power Available**: {', '.join(list(set(elite_power))[:3])}")

            high_hit = [n for n, p in all_unique.items() if p['total_hit_prob'] > 40]
            if high_hit:
                insights.append(f"🎯 **40%+ Hit Probability**: {', '.join(list(set(high_hit))[:3])}")

            combo = [n for n, p in all_unique.items() if p['adj_XB'] + p['adj_HR'] > 10 and p['adj_K'] <= 17]
            if combo:
                insights.append(f"⚡ **Power + Contact Combo**: {', '.join(list(set(combo))[:3])}")

            insights.append(f"📊 **Profile Diversity**: {len(profiles_to_display)}/7 profiles have viable options")

            for insight in insights:
                st.success(insight)

            st.markdown("**💡 Strategic Recommendations:**")
            power_avail   = any("Power" in p for p in profile_analysis)
            contact_avail = any(p in ["🏆 Contact-Aggressive", "⭐ Elite Contact"] for p in profile_analysis)

            if power_avail and contact_avail:
                st.info("⚖️ **Balanced Strategy**: Both power and contact available — diversify by contest type")
            elif "💥 Contact Power" in profile_analysis:
                st.info("🎯 **Premium Power**: Contact power available — ideal for tournament ceiling plays")
            elif "🚀 Pure Power" in profile_analysis:
                st.info("💥 **High-Risk/High-Reward**: Pure power available — GPP leverage play")
            elif "🏆 Contact-Aggressive" in profile_analysis:
                st.info("🛡️ **Safety**: Focus Contact-Aggressive for consistent base hits")
            elif "⚡ Swing-Happy" in profile_analysis:
                st.info("🔥 **Aggressive**: Swing-Happy available for leverage plays")

        else:
            st.warning("⚠️ No players available in any standard profiles after exclusions")

        excl_list = st.session_state.get('excluded_players', [])
        if excl_list:
            with st.expander("💡 Lineup Management Tips"):
                st.markdown(f"""
                **Players Currently Excluded**: {', '.join(excl_list)}
                - ✅ Check official lineups 2-3 hours before first pitch
                - ✅ Monitor injury reports and weather delays
                - ✅ Have backup players ready from same profile
                """)
        else:
            with st.expander("💡 Lineup Confirmation Reminder"):
                st.markdown("""
                **🏟️ Don't forget to verify lineups!**
                Check official team lineups 2-3 hours before games.
                Monitor for late scratches due to injury/rest/weather.
                """)
    else:
        st.info("💡 Need at least 3 players for Smart Profile Diversity Analysis")


def create_enhanced_visualizations(df, filtered_df):
    st.subheader("📈 Base Hit Analysis Charts")
    col1, col2 = st.columns(2)

    with col1:
        chart1 = alt.Chart(df).mark_bar(color='#1f77b4', opacity=0.7).encode(
            alt.X('Score:Q', bin=alt.Bin(maxbins=15), title='Base Hit Score'),
            alt.Y('count()', title='Number of Players'),
            tooltip=['count()']
        ).properties(title='Score Distribution (All Players)', width=350, height=300)
        st.altair_chart(chart1, use_container_width=True)

    with col2:
        chart2 = alt.Chart(filtered_df).mark_circle(size=100, opacity=0.7).encode(
            alt.X('total_hit_prob:Q', title='Total Hit Probability %'),
            alt.Y('adj_K:Q', title='Strikeout Risk %'),
            alt.Color('Score:Q', scale=alt.Scale(scheme='viridis')),
            alt.Size('adj_1B:Q', title='Single %'),
            tooltip=['Batter', 'total_hit_prob', 'adj_K', 'Score']
        ).properties(title='Hit Probability vs Strikeout Risk', width=350, height=300)
        st.altair_chart(chart2, use_container_width=True)

    if not filtered_df.empty:
        team_stats = filtered_df.groupby('Tm').agg({
            'total_hit_prob': 'mean', 'Score': 'mean', 'Batter': 'count'
        }).round(1).reset_index()
        team_stats.columns = ['Team', 'Avg Hit Prob %', 'Avg Score', 'Players']
        team_stats = team_stats.sort_values('Avg Hit Prob %', ascending=False)
        st.subheader("🏟️ Team Performance Summary")
        st.dataframe(team_stats, use_container_width=True)


# =============================================================================
# MAIN PAGE
# =============================================================================

def main_page():
    """Enhanced main page with staleness tracking and three-score system."""
    create_league_aware_header()

    # Initialize staleness clock (reset on Refresh)
    if 'data_load_time' not in st.session_state:
        st.session_state.data_load_time = datetime.now()

    create_staleness_indicator()

    with st.spinner('🔄 Loading and analyzing today\'s matchups...'):
        df = load_and_process_data()

    if df is None:
        st.error("❌ Unable to load data. Please check your internet connection and try again.")
        return

    create_data_quality_dashboard(df)

    filters     = create_league_aware_filters(df)
    profile_type = filters.get('profile_type', 'contact')
    df          = calculate_league_aware_scores(df, profile_type)
    filtered_df = apply_league_aware_filters(df, filters)

    display_league_aware_results(filtered_df, filters)
    create_enhanced_visualizations(df, filtered_df)

    # Export & controls
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        if st.button("📊 Export Results to CSV"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV", csv,
                f"mlb_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )

    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.session_state.data_load_time = datetime.now()  # Reset staleness clock
            st.rerun()

    with col3:
        if st.button("🏟️ Clear Exclusions"):
            st.session_state.excluded_players = []
            st.rerun()

    with col4:
        st.info(f"🕐 Session: {datetime.now().strftime('%H:%M:%S')}")

    # Quick lineup management
    if not filtered_df.empty:
        with st.expander("⚡ Quick Lineup Management"):
            st.markdown("**Quick exclude players from results:**")
            result_players     = filtered_df['Batter'].tolist()
            current_exclusions = st.session_state.excluded_players
            available_to_exclude = [p for p in result_players if p not in current_exclusions]

            if available_to_exclude:
                cl, cr = st.columns(2)
                with cl:
                    st.markdown("**Players in Current Results:**")
                    for i, player in enumerate(available_to_exclude[:5]):
                        if st.button(f"❌ Exclude {player}", key=f"exclude_{i}"):
                            if player not in st.session_state.excluded_players:
                                st.session_state.excluded_players.append(player)
                            st.rerun()
                with cr:
                    if len(available_to_exclude) > 5:
                        st.markdown("**More Players:**")
                        for i, player in enumerate(available_to_exclude[5:10]):
                            if st.button(f"❌ Exclude {player}", key=f"exclude_more_{i}"):
                                if player not in st.session_state.excluded_players:
                                    st.session_state.excluded_players.append(player)
                                st.rerun()
            else:
                st.info("🎯 All players in results are already confirmed playing")

            if current_exclusions:
                st.markdown("**Currently Excluded Players:**")
                st.info(f"🚫 {', '.join(current_exclusions)}")
                if st.button("🔄 Re-include All Excluded Players", key="main_clear"):
                    st.session_state.excluded_players = []
                    st.session_state.clear_exclusions = True
                    st.rerun()
            else:
                st.success("✅ All players currently included in analysis")

    # Strategy playbook
    st.markdown("---")
    st.markdown("""
    ### 🎯 **V4.0 DFS STRATEGIC PLAYBOOK**

    #### **🎯 Three-Score System — How to Use Each**
    - **🟢 Hit Score** → Sort by Hit Score for cash games and base hit props. Highest weight on adj_1B + heaviest K penalty. Your safest floor play each day.
    - **🟠 XB Score** → Sort by XB Score for doubles props and GPP upside. Primary driver is adj_XB direct from BallPark Pal simulations. Best for hitter-friendly slates.
    - **🔴 HR Score** → Sort by HR Score for HR props and GPP ceiling plays. Primary driver is adj_HR anchored to 2024 league averages. Light K penalty keeps legitimate power bats in view.

    #### **📊 Core Strategy Framework**
    - **Cash Games**: Sort by Hit Score | Target 35%+ hit probability + positive K% vs league
    - **Small GPP**: Mix Hit Score + XB Score leaders | Balance safety with ceiling
    - **Large GPP**: Lead with HR Score + XB Score | Prioritize differentiation

    #### **🏟️ Pitcher Matchup Grades (V4 Smooth Scale)**
    - Grades now use smooth linear interpolation — no cliff effects between thresholds
    - Applied multiplicatively: A+ multiplies score ×1.18 | D multiplies ×0.85
    - Better players benefit proportionally MORE from elite matchups

    #### **💰 Bankroll Management**
    - **20% Rule**: Never put more than 20% of bankroll in Pure Power or Swing-Happy profiles
    - **Diversification**: Build 60% Hit Score + 40% XB/HR Score portfolio
    - **Late Swap**: Always have a Hit Score backup for uncertain power plays

    **⚾ Three scores → Three bet types → Dominate the slate**

    **Version 4.0 — Three Purpose-Built Scores | Smooth Pitcher Grades (×Multiplier) | Data Staleness Indicator**
    """)


# =============================================================================
# INFO PAGE
# =============================================================================

def info_page():
    st.title("📚 MLB Hit Predictor V4 — Complete Reference Manual")

    with st.expander("📖 System Overview", expanded=True):
        st.markdown("""
        # 🎯 MLB Hit Predictor Pro V4.0

        ## Purpose
        Extracts the best targets from BallPark Pal's 3,000-simulation-per-game data.
        The tool does NOT try to predict better than BallPark Pal — it weights and filters
        the simulation output to surface the clearest hitter targets for each bet type.

        ## 📊 Core Metrics (unchanged from V3)
        - **Hit Probability**: Combined adj_1B% + adj_XB% + adj_HR%
        - **K% vs League**: 22.6% − Player K% (Positive = Better contact)
        - **BB% vs League**: 8.5% − Player BB% (Positive = More aggressive)
        - **Power Combo**: adj_XB% + adj_HR% combined rate

        ## 🎯 Three Purpose-Built Scores (NEW in V4)

        ### 🟢 Hit Score
        Optimized for base hit probability. Heaviest weight on adj_1B (singles)
        with the most aggressive K% penalty. Use for base hit props and cash games.

        ### 🟠 XB Score
        Optimized for extra base hits (doubles/triples). Primary driver is adj_XB
        from BallPark Pal simulations. Moderate K% penalty. Use for doubles props
        and GPP differentiation plays.

        ### 🔴 HR Score
        Optimized for home runs. Primary driver is adj_HR anchored to 2024 MLB
        league averages. Light K% penalty (HR hitters naturally K more).
        Use for HR props and GPP ceiling plays.

        ## 🏟️ Pitcher Matchup Grades (Improved in V4)

        Smooth linear interpolation replaces hard step thresholds. Grades now map
        to a proportional multiplier applied to all scores:

        | Grade | Multiplier | Meaning |
        |-------|-----------|---------|
        | A+    | ×1.18     | Elite matchup |
        | A     | ×1.08     | Great matchup |
        | B     | ×1.00     | Neutral |
        | C     | ×0.91     | Below average |
        | D     | ×0.85     | Avoid |

        Because the multiplier is proportional, a player with Score 80 gains more
        points from an A+ matchup than a player with Score 40 — as it should be.

        ## 🕐 Data Staleness Indicator (NEW in V4)
        Shows how old the current cached data is (green/yellow/red).
        BallPark Pal data refreshes every 15 minutes. Hit Refresh Data to force reload.

        ## 🏆 Strategic Framework

        ### Cash Games → Hit Score
        - Target: 35%+ hit probability
        - Sort by Hit Score
        - Profile: Contact-Aggressive + Elite Contact

        ### Small GPP → Hit + XB Score Mix
        - Lead with Hit Score safe plays
        - Add 1-2 XB Score leaders for upside

        ### Large GPP → HR + XB Score Targeting
        - 40% HR Score leaders (ceiling)
        - 30% XB Score leaders (doubles upside)
        - 30% Contrarian Hit Score plays

        ## 💰 Bankroll Management
        - **20% Rule**: Max 20% in high-variance profiles
        - **Portfolio**: 60% Hit Score + 40% XB/HR allocation
        - **Late Swap**: Always have a Hit Score backup ready
        """)


# =============================================================================
# MAIN
# =============================================================================

def main():
    st.sidebar.title("🏟️ Navigation")
    st.sidebar.markdown("---")

    if st.sidebar.checkbox("🎵 Background Music"):
        audio_url = "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        components.html(f"""
        <audio controls autoplay loop style="width: 100%;">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        """, height=60)

    app_mode = st.sidebar.radio(
        "Choose Section",
        ["🎯 League-Aware Predictor", "📚 Strategic Guide"],
        index=0
    )

    if app_mode == "🎯 League-Aware Predictor":
        main_page()
    else:
        info_page()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**V4.0** | Three Scores · Smooth Grades · Staleness Indicator")


if __name__ == "__main__":
    main()
