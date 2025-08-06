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
page_title=“A1PICKS MLB Hit Predictor Pro”,
layout=“wide”,
page_icon=“⚾”,
menu_items={
‘Get Help’: ‘mailto:your@email.com’,
‘Report a bug’: “https://github.com/yourrepo/issues”,
}
)

# Constants and Configuration

CONFIG = {
‘csv_urls’: {
‘probabilities’: ‘https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv’,
‘percent_change’: ‘https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv’
},
‘base_hit_weights’: {
‘adj_1B’: 2.0,      # Singles (primary base hit)
‘adj_XB’: 1.8,      # Extra bases (also base hits)
‘adj_vs’: 1.2,      # Matchup performance
‘adj_RC’: 0.8,      # Run creation
‘adj_HR’: 0.6,      # Home runs (also base hits)
‘adj_K’: -2.0,      # Heavy penalty for strikeouts (no hit)
‘adj_BB’: -0.8      # Moderate penalty for walks (not hits)
},
‘expected_columns’: [‘Tm’, ‘Batter’, ‘vs’, ‘Pitcher’, ‘RC’, ‘HR’, ‘XB’, ‘1B’, ‘BB’, ‘K’],
‘cache_ttl’: 900  # 15 minutes
}

# Custom CSS with enhanced styling

st.markdown(”””

<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        padding-top: 2.5rem;
    }
    .stRadio [role=radiogroup] {
        align-items: center;
        gap: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .color-legend {
        margin: 1rem 0;
        padding: 1rem;
        background: #000000;
        border-radius: 8px;
        color: white !important;
    }
    .hit-probability {
        font-size: 1.2em;
        font-weight: bold;
        color: #28a745;
    }
</style>

“””, unsafe_allow_html=True)

# Enhanced Data Loading with Error Handling and Validation

@st.cache_data(ttl=CONFIG[‘cache_ttl’])
def load_csv_with_validation(url, description, expected_columns):
“”“Load and validate CSV data with comprehensive error handling.”””
try:
with st.spinner(f’Loading {description}…’):
response = requests.get(url, timeout=15)
response.raise_for_status()

```
        df = pd.read_csv(StringIO(response.text))
        
        # Validate structure
        if df.empty:
            st.error(f"❌ {description}: No data found")
            return None
        
        # Check for required columns
        missing_cols = set(expected_columns) - set(df.columns)
        if missing_cols:
            st.error(f"❌ {description}: Missing columns {missing_cols}")
            return None
        
        # Validate key columns have no nulls
        key_cols = ['Tm', 'Batter', 'Pitcher']
        null_counts = df[key_cols].isnull().sum()
        if null_counts.any():
            problematic_cols = null_counts[null_counts > 0].index.tolist()
            st.error(f"❌ {description}: Null values in {problematic_cols}")
            return None
        
        st.success(f"✅ {description}: {len(df)} records loaded")
        return df
        
except requests.exceptions.RequestException as e:
    st.error(f"❌ Network error loading {description}: {str(e)}")
    return None
except Exception as e:
    st.error(f"❌ Error processing {description}: {str(e)}")
    return None
```

def validate_merge_quality(prob_df, pct_df, merged_df):
“”“Validate the quality of the merge operation.”””
original_count = len(prob_df)
merged_count = len(merged_df)

```
if merged_count < original_count:
    lost_records = original_count - merged_count
    st.warning(f"⚠️ Lost {lost_records} records during merge ({lost_records/original_count*100:.1f}%)")

# Check for duplicates
merge_keys = ['Tm', 'Batter', 'Pitcher']
duplicates = merged_df.duplicated(subset=merge_keys).sum()
if duplicates > 0:
    st.error(f"🔴 Found {duplicates} duplicate matchups after merge")

return merged_df
```

@st.cache_data(ttl=CONFIG[‘cache_ttl’])
def load_and_process_data():
“”“Enhanced data loading and processing with validation.”””

```
# Load both CSV files with enhanced validation
prob_df = load_csv_with_validation(
    CONFIG['csv_urls']['probabilities'], 
    "Base Probabilities", 
    CONFIG['expected_columns']
)

pct_df = load_csv_with_validation(
    CONFIG['csv_urls']['percent_change'], 
    "Adjustment Factors", 
    CONFIG['expected_columns']
)

if prob_df is None or pct_df is None:
    st.error("❌ Failed to load required data files")
    return None

# Enhanced merge with validation
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

# Calculate adjusted metrics with CORRECTED column mapping
metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']

for metric in metrics:
    # FIXED: Use correct columns based on actual CSV structure
    if metric in ['K', 'BB']:
        # For K and BB, use the .1 columns (actual probabilities)
        base_col = f'{metric}.1'  # Will be 'K.1', 'BB.1'
    else:
        # For other metrics, use normal _prob suffix
        base_col = f'{metric}_prob'
        
    pct_col = f'{metric}_pct'
    
    if base_col in merged_df.columns and pct_col in merged_df.columns:
        # Apply adjustment formula
        merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col]/100)
        
        # Smart clipping based on metric type
        if metric in ['K', 'BB']:
            # K and BB should be positive percentages
            merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
        elif metric in ['1B', 'XB', 'HR']:  # Probability metrics
            merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
        else:  # Other metrics (vs, RC)
            merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
            
    else:
        # Create a fallback column with reasonable defaults to prevent KeyError
        if metric in ['K', 'BB']:
            merged_df[f'adj_{metric}'] = 20  # Default reasonable K/BB rate
        else:
            merged_df[f'adj_{metric}'] = 0

# Calculate total base hit probability (key enhancement!)
merged_df['total_hit_prob'] = merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
merged_df['total_hit_prob'] = merged_df['total_hit_prob'].clip(upper=100)  # Cap at 100%

return merged_df
```

def calculate_league_aware_scores(df, profile_type=‘contact’):
“”“Enhanced scoring algorithm that adapts based on profile type.”””

```
if profile_type == 'power':
    # Power-focused scoring weights
    weights = {
        'adj_XB': 3.0,      # Primary focus: Extra bases
        'adj_HR': 2.5,      # Primary focus: Home runs  
        'adj_vs': 1.5,      # Enhanced matchup importance for power
        'adj_RC': 1.0,      # Run creation
        'adj_1B': 0.5,      # Reduced singles weight (not relevant for power)
        'adj_K': -1.0,      # Reduced K penalty (power hitters strike out more)
        'adj_BB': -0.3      # Minimal BB penalty (some power hitters walk more)
    }
    
    # Power-specific bonuses
    df['power_bonus'] = np.where(
        (df['adj_XB'] > 10) & (df['adj_HR'] > 4), 12, 0  # Elite power combo
    )
    
    df['clutch_power_bonus'] = np.where(
        (df['adj_XB'] > 8) & (df['adj_vs'] > 5), 8, 0  # Good power + good matchup
    )
    
    df['consistent_power_bonus'] = np.where(
        (df['adj_HR'] > 3) & (df['adj_K'] < 25), 5, 0  # HR power without excessive Ks
    )
    
    # Calculate power probability instead of hit probability
    df['power_prob'] = df['adj_XB'] + df['adj_HR']
    df['power_prob'] = df['power_prob'].clip(upper=50)  # Cap at reasonable power rate
    
    bonus_cols = ['power_bonus', 'clutch_power_bonus', 'consistent_power_bonus']
    
else:
    # Contact-focused scoring weights (original system)
    weights = {
        'adj_1B': 2.0,      # Singles (primary base hit)
        'adj_XB': 1.8,      # Extra bases (also base hits)
        'adj_vs': 1.2,      # Matchup performance
        'adj_RC': 0.8,      # Run creation
        'adj_HR': 0.6,      # Home runs (also base hits)
        'adj_K': -2.0,      # Heavy penalty for strikeouts (no hit)
        'adj_BB': -0.8      # Moderate penalty for walks (not hits)
    }
    
    # Contact-focused bonuses
    df['contact_bonus'] = np.where(
        (df['total_hit_prob'] > 40) & (df['adj_K'] < 18), 8, 0
    )
    
    df['consistency_bonus'] = np.where(
        (df['adj_1B'] > 20) & (df['adj_XB'] > 8), 5, 0
    )
    
    df['matchup_bonus'] = np.where(df['adj_vs'] > 5, 3, 0)
    
    bonus_cols = ['contact_bonus', 'consistency_bonus', 'matchup_bonus']

# Base weighted score calculation
df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)

# Calculate final score with appropriate bonuses
df['Score'] = df['base_score'] + sum(df[col] for col in bonus_cols if col in df.columns)

# Normalize to 0-100 scale
if df['Score'].max() != df['Score'].min():
    df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
else:
    df['Score'] = 50  # Default if all scores are identical

return df.round(1)
```

def create_league_aware_filters(df=None):
“”“Create baseball-intelligent filtering system based on league averages and player types.”””
st.sidebar.header(“🎯 Baseball-Smart Filters”)

```
# Initialize session state for exclusions if not exists
if 'excluded_players' not in st.session_state:
    st.session_state.excluded_players = []

# Handle clear exclusions command
if 'clear_exclusions' in st.session_state and st.session_state.clear_exclusions:
    st.session_state.excluded_players = []
    st.session_state.clear_exclusions = False

# Handle quick exclude additions
if 'quick_exclude_players' in st.session_state:
    for player in st.session_state.quick_exclude_players:
        if player not in st.session_state.excluded_players:
            st.session_state.excluded_players.append(player)
    st.session_state.quick_exclude_players = []  # Clear after processing

# League averages for 2024
LEAGUE_K_AVG = 22.6
LEAGUE_BB_AVG = 8.5

filters = {}

# Show league context
st.sidebar.markdown("### **📊 2024 League Context**")
st.sidebar.markdown(f"""
- **K% League Avg**: {LEAGUE_K_AVG}%
- **BB% League Avg**: {LEAGUE_BB_AVG}%
""")

if df is not None and not df.empty:
    st.sidebar.markdown(f"**📈 Today's Pool:** {len(df)} matchups")
    avg_k = df['adj_K'].mean() if 'adj_K' in df.columns else 0
    avg_bb = df['adj_BB'].mean() if 'adj_BB' in df.columns else 0
    st.sidebar.markdown(f"**Today's Avg K%:** {avg_k:.1f}%")
    st.sidebar.markdown(f"**Today's Avg BB%:** {avg_bb:.1f}%")

st.sidebar.markdown("---")

# PRIMARY FILTER: Player Type Selection
st.sidebar.markdown("### **🎯 Player Type Focus**")

player_type_options = {
    "🏆 Contact-Aggressive Hitters": {
        'description': "Low K% + Low BB% (Elite for base hits)",
        'max_k': 19.0,   # Updated: More inclusive
        'max_bb': 7.0,   # Updated: More inclusive
        'min_hit_prob': 32,  # Updated: Slightly lower
        'profile_type': 'contact'
    },
    "⭐ Elite Contact Specialists": {
        'description': "Ultra-low K% (Pure contact)",
        'max_k': 14.0,   # Updated: More inclusive
        'max_bb': 9.5,   # Updated: More inclusive
        'min_hit_prob': 28,  # Updated: Slightly lower
        'profile_type': 'contact'
    },
    "⚡ Swing-Happy Hitters": {
        'description': "Ultra-low BB% (Aggressive approach)",
        'max_k': 24.0,   # Updated: More inclusive
        'max_bb': 5.0,   # Updated: More inclusive
        'min_hit_prob': 30,  # Updated: Slightly lower
        'profile_type': 'contact'
    },
    "🔷 Above-Average Contact": {
        'description': "Better than league average K%",
        'max_k': 20.0,   # Updated: More inclusive
        'max_bb': 12.0,  # Updated: More inclusive
        'min_hit_prob': 25,  # Updated: More accessible
        'profile_type': 'contact'
    },
    "💥 Contact Power Hitters": {
        'description': "Low K% + High XB% & HR% (Power with contact)",
        'max_k': 20.0,   # Updated: More inclusive
        'max_bb': 12.0,  # Updated: More inclusive
        'min_xb': 7.0,   # Updated: More accessible
        'min_hr': 2.5,   # Updated: More accessible
        'min_vs': -5,    # Reasonable matchup
        'profile_type': 'power'
    },
    "🚀 Pure Power Sluggers": {
        'description': "High XB% & HR% (Power over contact)",
        'max_k': 100,    # Accept high strikeouts
        'max_bb': 100,   # Accept high walks
        'min_xb': 9.0,   # Updated: More accessible
        'min_hr': 3.5,   # Updated: More accessible
        'min_vs': -10,   # Any matchup if power is elite
        'profile_type': 'power'
    },
    "⚾ All Power Players": {
        'description': "All players ranked by power potential (Research mode)",
        'max_k': 100,    # No K% restrictions
        'max_bb': 100,   # No BB% restrictions
        'min_xb': 0,     # No XB% restrictions
        'min_hr': 0,     # No HR% restrictions
        'min_vs': -10,   # Any matchup
        'profile_type': 'power'
    },
    "🌐 All Players": {
        'description': "No restrictions",
        'max_k': 100,
        'max_bb': 100,
        'min_hit_prob': 20,
        'profile_type': 'all'
    }
}

selected_type = st.sidebar.selectbox(
    "Choose Hitter Profile",
    options=list(player_type_options.keys()),
    index=0,  # Default to Contact-Aggressive
    help="Each profile targets different hitting approaches based on league averages"
)

# Apply selected player type settings
type_config = player_type_options[selected_type]
filters['max_k'] = type_config['max_k']
filters['max_bb'] = type_config['max_bb']
filters['profile_type'] = type_config['profile_type']

# Profile-specific criteria
if type_config['profile_type'] == 'power':
    filters['min_xb'] = type_config.get('min_xb', 0)
    filters['min_hr'] = type_config.get('min_hr', 0)
    filters['min_vs'] = type_config.get('min_vs', -10)
    filters['min_hit_prob'] = 0  # Not used for power profiles
else:
    filters['min_hit_prob'] = type_config.get('min_hit_prob', 20)
    filters['min_xb'] = 0
    filters['min_hr'] = 0
    filters['min_vs'] = -10

# Show what this means
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

# ADVANCED OPTIONS (Collapsible)
with st.sidebar.expander("⚙️ Fine-Tune Filters"):
    
    # Ensure filters values are single numbers, not lists
    max_k_value = filters['max_k']
    max_bb_value = filters['max_bb']
    
    # Handle edge case where values might be lists
    if isinstance(max_k_value, (list, tuple)):
        max_k_value = max_k_value[0] if max_k_value else 17.0
    if isinstance(max_bb_value, (list, tuple)):
        max_bb_value = max_bb_value[0] if max_bb_value else 6.0
        
    # Ensure values are within slider bounds
    max_k_value = max(5.0, min(35.0, float(max_k_value)))
    max_bb_value = max(2.0, min(15.0, float(max_bb_value)))
    
    # Custom K% threshold
    filters['custom_max_k'] = st.slider(
        "Custom Max K% Override",
        min_value=5.0,
        max_value=35.0,
        value=max_k_value,
        step=0.5,
        help=f"League avg: {LEAGUE_K_AVG}% | Elite: ≤12.0%"
    )
    
    # Custom BB% threshold
    filters['custom_max_bb'] = st.slider(
        "Custom Max BB% Override",
        min_value=2.0,
        max_value=15.0,
        value=max_bb_value,
        step=0.5,
        help=f"League avg: {LEAGUE_BB_AVG}% | Aggressive: ≤4.0%"
    )
    
    # Use custom values if they differ from preset
    if filters['custom_max_k'] != filters['max_k']:
        filters['max_k'] = filters['custom_max_k']
        
    if filters['custom_max_bb'] != filters['max_bb']:
        filters['max_bb'] = filters['custom_max_bb']
    
    # vs Pitcher Rating 
    filters['min_vs_pitcher'] = st.slider(
        "vs Pitcher Rating",
        min_value=-10,
        max_value=10,
        value=0,
        step=1,
        help="Matchup advantage/disadvantage vs this pitcher type"
    )
    
    # Best Player Per Team Filter
    filters['best_per_team_only'] = st.checkbox(
        "🏟️ Show only best player per team",
        value=False,
        help="Filter to show only the highest-scoring player from each team (ideal for diversified betting)"
    )
    
    # Team Selection
    team_options = []
    if df is not None and not df.empty:
        team_options = sorted(df['Tm'].unique().tolist())
    
    filters['selected_teams'] = st.multiselect(
        "Filter by Teams",
        options=team_options,
        help="Leave empty to include all teams"
    )
    
    # Result count
    filters['result_count'] = st.selectbox(
        "Number of Results",
        options=[5, 10, 15, 20, 25, 30, "All"],
        index=2,
        help="Choose how many results to display, or 'All' to show everyone"
    )

# LINEUP STATUS MANAGEMENT (Unified System)
with st.sidebar.expander("🏟️ Lineup Status Management"):
    st.markdown("**Exclude players not in today's lineups:**")
    
    # Get list of all players for exclusion
    all_players = []
    if df is not None and not df.empty:
        all_players = sorted(df['Batter'].unique().tolist())
    
    # Use unified session state for exclusions
    current_exclusions = st.session_state.excluded_players.copy()
    
    # Multiselect that syncs with session state
    selected_exclusions = st.multiselect(
        "Players NOT Playing Today",
        options=all_players,
        default=current_exclusions,
        help="Select players who are confirmed out of lineups (injured, benched, etc.)",
        key="lineup_exclusions"
    )
    
    # Update session state when multiselect changes
    st.session_state.excluded_players = selected_exclusions
    filters['excluded_players'] = selected_exclusions
    
    # Show current exclusion count
    if selected_exclusions:
        st.info(f"🚫 Currently excluding {len(selected_exclusions)} players")
    
    # Quick clear button for sidebar
    if st.button("🔄 Clear All Exclusions", key="sidebar_clear"):
        st.session_state.excluded_players = []
        st.rerun()
    
    # Quick exclude options for common scenarios
    st.markdown("**Quick Exclude Options:**")
    
    if st.checkbox("🏥 Auto-exclude common injury-prone players"):
        filters['auto_exclude_injured'] = True
    else:
        filters['auto_exclude_injured'] = False
        
    if st.checkbox("📊 Show lineup confidence warnings"):
        filters['show_lineup_warnings'] = True
    else:
        filters['show_lineup_warnings'] = False

# REAL-TIME FEEDBACK with league context and lineup awareness
if df is not None and not df.empty:
    try:
        # Apply exclusions first using unified session state
        preview_df = df.copy()
        excluded_players = st.session_state.excluded_players
        if excluded_players:
            preview_df = preview_df[~preview_df['Batter'].isin(excluded_players)]
        
        # Build query based on profile type
        if filters.get('profile_type') == 'power':
            preview_query = f"adj_K <= {filters['max_k']:.1f} and adj_BB <= {filters['max_bb']:.1f} and adj_XB >= {filters['min_xb']:.1f} and adj_HR >= {filters['min_hr']:.1f} and adj_vs >= {filters['min_vs']}"
        else:
            preview_query = f"adj_K <= {filters['max_k']:.1f} and adj_BB <= {filters['max_bb']:.1f} and total_hit_prob >= {filters['min_hit_prob']}"
        
        preview_df = preview_df.query(preview_query)
        
        # Apply best per team filter in preview if enabled
        if filters.get('best_per_team_only', False):
            preview_df = preview_df.loc[preview_df.groupby('Tm')['Score'].idxmax()]
        
        matching_count = len(preview_df)
        excluded_count = len(excluded_players)
        
        # Context-aware feedback with lineup information
        if matching_count == 0:
            st.sidebar.error("❌ No players match current profile")
            if excluded_count > 0:
                st.sidebar.markdown(f"**💡 Note:** {excluded_count} players excluded due to lineup status")
            st.sidebar.markdown("**💡 Try:** Different player type or use custom overrides")
        elif matching_count < 5:
            st.sidebar.warning(f"⚠️ Only {matching_count} players match")
            if excluded_count > 0:
                st.sidebar.markdown(f"**📊 Pool:** {matching_count} playing + {excluded_count} excluded")
            if filters.get('best_per_team_only', False):
                st.sidebar.markdown(f"**🏟️ Teams:** {len(preview_df['Tm'].unique())} teams represented")
            st.sidebar.markdown("**💡 Consider:** Less restrictive profile or custom settings")
        else:
            st.sidebar.success(f"✅ {matching_count} players match profile")
            if excluded_count > 0:
                st.sidebar.markdown(f"**📊 Lineup Status:** {matching_count} confirmed playing, {excluded_count} excluded")
            
            if filters.get('best_per_team_only', False):
                unique_teams = len(preview_df['Tm'].unique())
                st.sidebar.markdown(f"**🏟️ Team Diversity:** Best player from {unique_teams} teams")
            
            if matching_count > 0:
                # Show corrected league vs player calculations
                avg_k_filtered = preview_df['adj_K'].mean()
                avg_bb_filtered = preview_df['adj_BB'].mean()
                
                # CORRECTED: League - Player = Positive when player is better
                k_vs_league = LEAGUE_K_AVG - avg_k_filtered  # Positive = better contact
                bb_vs_league = LEAGUE_BB_AVG - avg_bb_filtered  # Positive = more aggressive
                
                result_count = filters.get('result_count', 15)
                display_count = matching_count if result_count == "All" else min(matching_count, result_count)
                
                st.sidebar.markdown(f"**📊 vs League (Playing Players):**")
                st.sidebar.markdown(f"K%: {k_vs_league:+.1f}% {'better' if k_vs_league > 0 else 'worse'} than league")
                st.sidebar.markdown(f"BB%: {bb_vs_league:+.1f}% {'more aggressive' if bb_vs_league > 0 else 'less aggressive'} than league")
                
                if result_count == "All":
                    st.sidebar.markdown(f"**📋 Showing:** All {matching_count} players")
                else:
                    st.sidebar.markdown(f"**📋 Showing:** Top {display_count} of {matching_count}")
                
    except Exception as e:
        st.sidebar.warning(f"⚠️ Preview unavailable: {str(e)}")

return filters
```

def apply_league_aware_filters(df, filters):
“”“Apply baseball-intelligent filtering based on league averages and player types.”””

```
if df is None or df.empty:
    return df

# First, exclude players not in lineups using unified session state
excluded_players = st.session_state.get('excluded_players', [])
if excluded_players:
    excluded_count = len(df[df['Batter'].isin(excluded_players)])
    df = df[~df['Batter'].isin(excluded_players)]
    if excluded_count > 0:
        st.info(f"🏟️ Excluded {excluded_count} players not in today's lineups")

query_parts = []

# Primary filters based on player type
if 'max_k' in filters and filters['max_k'] < 100:
    query_parts.append(f"adj_K <= {filters['max_k']:.1f}")

if 'max_bb' in filters and filters['max_bb'] < 100:
    query_parts.append(f"adj_BB <= {filters['max_bb']:.1f}")

# Profile-specific filters
if filters.get('profile_type') == 'power':
    # Power profile filters
    if 'min_xb' in filters and filters['min_xb'] > 0:
        query_parts.append(f"adj_XB >= {filters['min_xb']:.1f}")
    
    if 'min_hr' in filters and filters['min_hr'] > 0:
        query_parts.append(f"adj_HR >= {filters['min_hr']:.1f}")
    
    if 'min_vs' in filters and filters['min_vs'] > -10:
        query_parts.append(f"adj_vs >= {filters['min_vs']}")
else:
    # Contact profile filters
    if 'min_hit_prob' in filters and filters['min_hit_prob'] > 0:
        query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']:.1f}")

# Advanced filters
if 'min_vs_pitcher' in filters and filters['min_vs_pitcher'] != 0:
    query_parts.append(f"adj_vs >= {filters['min_vs_pitcher']}")

# Team filter
if filters.get('selected_teams'):
    team_filter = "Tm in " + str(filters['selected_teams'])
    query_parts.append(team_filter)

# Apply filters with error handling
try:
    if query_parts:
        full_query = " and ".join(query_parts)
        filtered_df = df.query(full_query)
    else:
        filtered_df = df  # No filters applied
    
    # Apply "Best Player Per Team" filter if enabled
    if filters.get('best_per_team_only', False):
        # Group by team and keep only the highest scoring player from each team
        best_per_team = filtered_df.loc[filtered_df.groupby('Tm')['Score'].idxmax()]
        filtered_df = best_per_team.copy()
        
        # Show info about team filtering
        unique_teams = len(filtered_df['Tm'].unique())
        st.info(f"🏟️ Showing best player from each of {unique_teams} teams")
    
    # Sort by score and limit results
    result_count = filters.get('result_count', 15)
    
    if result_count == "All":
        # Show all results when "All" is selected
        result_df = filtered_df.sort_values('Score', ascending=False)
    else:
        # Limit to specified number
        result_df = filtered_df.sort_values('Score', ascending=False).head(result_count)
    
    return result_df
    
except Exception as e:
    st.error(f"❌ Filter error: {str(e)}")
    # Return top players by score if filtering fails
    result_count = filters.get('result_count', 15)
    
    if result_count == "All":
        return df.sort_values('Score', ascending=False)
    else:
        return df.sort_values('Score', ascending=False).head(result_count)
```

def create_league_aware_header():
“”“Create an enhanced header with league-aware focus.”””
col1, col2 = st.columns([1, 4])

```
with col1:
    st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
            width=200)

with col2:
    st.title("🎯 MLB League-Aware Hit Predictor Pro")
    st.markdown("*Find hitters with the best base hit probability using 2024 league context*")
```

def create_data_quality_dashboard(df):
“”“Display data quality metrics in an attractive dashboard.”””
if df is None or df.empty:
st.error(“No data available for quality analysis”)
return

```
st.subheader("📊 Today's Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown("""
    <div class="metric-card">
        <h3>📈 Total Matchups</h3>
        <h2>{}</h2>
    </div>
    """.format(len(df)), unsafe_allow_html=True)

with col2:
    unique_batters = df['Batter'].nunique()
    st.markdown("""
    <div class="metric-card">
        <h3>👤 Unique Batters</h3>
        <h2>{}</h2>
    </div>
    """.format(unique_batters), unsafe_allow_html=True)

with col3:
    unique_teams = df['Tm'].nunique()
    st.markdown("""
    <div class="metric-card">
        <h3>🏟️ Teams Playing</h3>
        <h2>{}</h2>
    </div>
    """.format(unique_teams), unsafe_allow_html=True)

with col4:
    avg_hit_prob = df['total_hit_prob'].mean()
    st.markdown("""
    <div class="success-card">
        <h3>🎯 Avg Hit Probability</h3>
        <h2>{:.1f}%</h2>
    </div>
    """.format(avg_hit_prob), unsafe_allow_html=True)
```

def display_league_aware_results(filtered_df, filters):
“”“Display results with league-average context and baseball intelligence.”””

```
# League averages for 2024
LEAGUE_K_AVG = 22.6
LEAGUE_BB_AVG = 8.5

if filtered_df.empty:
    st.warning("⚠️ No players match your current player type filters")
    
    # Smart suggestions based on league context
    st.markdown("""
    ### 💡 **Suggested Adjustments:**
    - Try **"Above-Average Contact"** for more options
    - Use **custom overrides** in advanced settings
    - Consider **"All Players"** to see the full pool
    """)
    return

# Display header with dynamic count and filter info
result_count = filters.get('result_count', 15)
best_per_team = filters.get('best_per_team_only', False)

if best_per_team:
    if result_count == "All":
        st.subheader(f"🏟️ Best Player from Each Team ({len(filtered_df)} teams)")
    else:
        st.subheader(f"🏟️ Top {len(filtered_df)} Teams - Best Player Each")
else:
    if result_count == "All":
        st.subheader(f"🎯 All {len(filtered_df)} Base Hit Candidates")
    else:
        st.subheader(f"🎯 Top {len(filtered_df)} Base Hit Candidates")

# Enhanced key insights with league context
col1, col2, col3, col4 = st.columns(4)

with col1:
    best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
    st.markdown(f"""
    <div class="success-card">
        <h4>🥇 Best Hit Probability</h4>
        <h2>{best_hit_prob:.1f}%</h2>
        <small>Target: 35%+</small>
    </div>
    """, unsafe_allow_html=True)

with col2:
    avg_k = filtered_df['adj_K'].mean()
    k_vs_league = LEAGUE_K_AVG - avg_k  # Positive = better contact
    color = "success-card" if k_vs_league > 0 else "metric-card"
    st.markdown(f"""
    <div class="{color}">
        <h4>⚾ K% vs League</h4>
        <h2>{k_vs_league:+.1f}%</h2>
        <small>League: {LEAGUE_K_AVG}%</small>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_bb = filtered_df['adj_BB'].mean()
    bb_vs_league = LEAGUE_BB_AVG - avg_bb  # Positive = more aggressive
    color = "success-card" if bb_vs_league > 0 else "metric-card"
    st.markdown(f"""
    <div class="{color}">
        <h4>🚶 BB% vs League</h4>
        <h2>{bb_vs_league:+.1f}%</h2>
        <small>League: {LEAGUE_BB_AVG}%</small>
    </div>
    """, unsafe_allow_html=True)

with col4:
    elite_contact_count = (filtered_df['adj_K'] <= 12.0).sum()  # Elite contact threshold
    st.markdown(f"""
    <div class="success-card">
        <h4>⭐ Elite Contact</h4>
        <h2>{elite_contact_count}/{len(filtered_df)}</h2>
        <small>K% ≤12.0%</small>
    </div>
    """, unsafe_allow_html=True)

# Show current filter settings
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

st.markdown(f"**🎯 Active Profile:** {filter_profile}")

# Enhanced results table with CORRECTED league context calculations
display_columns = {
    'Batter': 'Batter',
    'Tm': 'Team',
    'Pitcher': 'Pitcher',
    'total_hit_prob': 'Hit Prob %',
    'adj_1B': 'Contact %',
    'adj_XB': 'XB %',
    'adj_HR': 'HR %',
    'K_vs_League': 'K% vs League',
    'BB_vs_League': 'BB% vs League',
    'adj_vs': 'vs Pitcher',
    'Score': 'Score'
}

# Add CORRECTED league context columns
display_df = filtered_df.copy()
# CORRECTED: League - Player = Positive when player is better (lower K%, lower BB%)
display_df['K_vs_League'] = LEAGUE_K_AVG - display_df['adj_K']    # 22.6 - 15.0 = +7.6 (better contact)
display_df['BB_vs_League'] = LEAGUE_BB_AVG - display_df['adj_BB']  # 8.5 - 5.0 = +3.5 (more aggressive)

# Add lineup status indicators
excluded_players = st.session_state.get('excluded_players', [])
display_df['Lineup_Status'] = display_df['Batter'].apply(
    lambda x: '🏟️' if x not in excluded_players else '❌'
)

# Add lineup status to display columns
display_columns_with_status = {'Lineup_Status': 'Status', **display_columns}

styled_df = display_df[display_columns_with_status.keys()].rename(columns=display_columns_with_status)

# CORRECTED formatting with proper vs league calculations
styled_df = styled_df.style.format({
    'Hit Prob %': "{:.1f}%",
    'Contact %': "{:.1f}%", 
    'XB %': "{:.1f}%",
    'HR %': "{:.1f}%",
    'K% vs League': "{:+.1f}%",     # +7.6% = better contact (green)
    'BB% vs League': "{:+.1f}%",    # +3.5% = more aggressive (green)
    'vs Pitcher': "{:.0f}",
    'Score': "{:.1f}"
}).background_gradient(
    subset=['Score'],
    cmap='RdYlGn',
    vmin=0,
    vmax=100
).background_gradient(
    subset=['Hit Prob %'],
    cmap='Greens',
    vmin=20,
    vmax=50
).background_gradient(
    subset=['K% vs League'],
    cmap='RdYlGn',  # Positive = better contact = Green, Negative = worse contact = Red
    vmin=-8,        # Much worse than league (red)
    vmax=12         # Much better than league (green)
).background_gradient(
    subset=['BB% vs League'], 
    cmap='RdYlGn',  # Positive = more aggressive = Green, Negative = less aggressive = Red
    vmin=-5,        # Much less aggressive than league (red)
    vmax=6          # Much more aggressive than league (green)
)

st.dataframe(styled_df, use_container_width=True)

# CORRECTED interpretation guide
st.markdown("""
<div class="color-legend">
    <strong>📊 CORRECTED League vs Player Guide:</strong><br>
    <strong>Status:</strong> 🏟️ = Confirmed Playing | ❌ = Excluded from Lineups<br>
    <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
    <span style="color: #fdae61;">●</span> Good (50-69) | 
    <span style="color: #d7191c;">●</span> Risky (<50)<br>
    <strong>K% vs League:</strong> <span style="color: #1a9641;">●</span> Positive = Better Contact (strikes out less) | 
    <span style="color: #d7191c;">●</span> Negative = Worse Contact (strikes out more)<br>
    <strong>BB% vs League:</strong> <span style="color: #1a9641;">●</span> Positive = More Aggressive (walks less) | 
    <span style="color: #d7191c;">●</span> Negative = Less Aggressive (walks more)<br>
    <strong>✅ Rule: Positive vs League = Better Performance = Green!</strong>
</div>
""", unsafe_allow_html=True)

# Performance insights with FIXED league context - multi-profile analysis
if len(filtered_df) >= 3:
    st.markdown("### 🔍 **Advanced League Context Analysis**")
    
    # Define profile criteria for analysis (UPDATED with All Power Players added back)
    profile_criteria = {
        "🏆 Contact-Aggressive": {"max_k": 19.0, "max_bb": 7.0, "icon": "🏆", "type": "contact"},
        "⭐ Elite Contact": {"max_k": 14.0, "max_bb": 9.5, "icon": "⭐", "type": "contact"},
        "⚡ Swing-Happy": {"max_k": 24.0, "max_bb": 5.0, "icon": "⚡", "type": "contact"},
        "🔷 Above-Average": {"max_k": 20.0, "max_bb": 12.0, "icon": "🔷", "type": "contact"},
        "💥 Contact Power": {"max_k": 20.0, "max_bb": 12.0, "min_xb": 7.0, "min_hr": 2.5, "icon": "💥", "type": "power"},
        "🚀 Pure Power": {"max_k": 100, "max_bb": 100, "min_xb": 9.0, "min_hr": 3.5, "icon": "🚀", "type": "power"},
        "⚾ All Power": {"max_k": 100, "max_bb": 100, "min_xb": 0, "min_hr": 0, "icon": "⚾", "type": "power"}
    }
    
    excluded_players = st.session_state.get('excluded_players', [])
    
    # Find best player for each profile
    profile_analysis = {}
    
    for profile_name, criteria in profile_criteria.items():
        # Filter players that meet this profile's criteria
        if criteria["type"] == "power":
            # Power profile filtering
            profile_players = filtered_df[
                (filtered_df['adj_K'] <= criteria['max_k']) & 
                (filtered_df['adj_BB'] <= criteria['max_bb']) &
                (filtered_df['adj_XB'] >= criteria.get('min_xb', 0)) &
                (filtered_df['adj_HR'] >= criteria.get('min_hr', 0)) &
                (~filtered_df['Batter'].isin(excluded_players))  # Exclude non-playing players
            ].copy()
        else:
            # Contact profile filtering
            profile_players = filtered_df[
                (filtered_df['adj_K'] <= criteria['max_k']) & 
                (filtered_df['adj_BB'] <= criteria['max_bb']) &
                (~filtered_df['Batter'].isin(excluded_players))  # Exclude non-playing players
            ].copy()
        
        if not profile_players.empty:
            # Get the top player for this profile
            best_player = profile_players.iloc[0]
            profile_analysis[profile_name] = {
                'player': best_player,
                'rank_overall': filtered_df[filtered_df['Batter'] == best_player['Batter']].index[0] + 1,
                'count_in_profile': len(profile_players)
            }
    
    # Display analysis for each profile that has players
    if profile_analysis:
        st.markdown("**🎯 Top Player by Profile:**")
        
        # Create columns for profile analysis
        num_profiles = len(profile_analysis)
        if num_profiles == 1:
            cols = [st.columns(1)[0]]
        elif num_profiles == 2:
            cols = st.columns(2)
        elif num_profiles <= 4:
            cols = st.columns(min(num_profiles, 4))
        else:
            cols = st.columns(4)
        
        for i, (profile_name, analysis) in enumerate(profile_analysis.items()):
            player = analysis['player']
            overall_rank = analysis['rank_overall']
            profile_count = analysis['count_in_profile']
            
            with cols[i % len(cols)]:
                # Profile header with icon
                icon = profile_criteria[profile_name]['icon']
                st.markdown(f"**{icon} {profile_name.split(' ', 1)[1]}**")  # Remove icon from name since we show it
                
                # Player name with rank indication
                if overall_rank == 1:
                    st.success(f"🥇 **{player['Batter']}** (#{overall_rank})")
                elif overall_rank <= 3:
                    st.info(f"🥈 **{player['Batter']}** (#{overall_rank})")
                else:
                    st.info(f"**{player['Batter']}** (#{overall_rank})")
                
                # Key metrics with CORRECTED vs league calculations
                if criteria["type"] == "power":
                    # Power-focused metrics
                    st.markdown(f"""
                    **XB%:** {player['adj_XB']:.1f}% | **HR%:** {player['adj_HR']:.1f}%  
                    **Power Combo:** {(player['adj_XB'] + player['adj_HR']):.1f}%  
                    **vs Pitcher:** {player['adj_vs']:.0f} | **Score:** {player['Score']:.1f}
                    """)
                else:
                    # Contact-focused metrics with corrected calculations
                    k_vs_league = LEAGUE_K_AVG - player['adj_K']    # Positive = better contact
                    bb_vs_league = LEAGUE_BB_AVG - player['adj_BB']  # Positive = more aggressive
                    
                    st.markdown(f"""
                    **Hit Prob:** {player['total_hit_prob']:.1f}%  
                    **K% vs League:** {k_vs_league:+.1f}%  
                    **BB% vs League:** {bb_vs_league:+.1f}%  
                    **Score:** {player['Score']:.1f}
                    """)
                
                # Profile pool size
                st.caption(f"📊 {profile_count} players in profile")
        
        # Summary insights across profiles
        st.markdown("---")
        st.markdown("**📋 Profile Summary:**")
        
        insights = []
        
        # Find the highest scoring player across all profiles
        best_overall_player = max(profile_analysis.values(), key=lambda x: x['player']['Score'])
        best_player_name = best_overall_player['player']['Batter']
        best_profile = [k for k, v in profile_analysis.items() if v['player']['Batter'] == best_player_name][0]
        
        insights.append(f"🏆 **Overall Best**: {best_player_name} ({best_profile})")
        
        # Check for elite power across profiles
        elite_power_players = [analysis['player']['Batter'] for analysis in profile_analysis.values() 
                             if analysis['player']['adj_XB'] + analysis['player']['adj_HR'] > 12]
        if elite_power_players:
            insights.append(f"💥 **Elite Power Available**: {', '.join(elite_power_players)}")
        
        # Check for high hit probability players (contact profiles)
        high_hit_prob_players = [analysis['player']['Batter'] for analysis in profile_analysis.values() 
                               if analysis['player']['total_hit_prob'] > 40]
        if high_hit_prob_players:
            insights.append(f"🎯 **40%+ Hit Probability**: {', '.join(high_hit_prob_players)}")
        
        # Check for power+contact combo players
        power_contact_players = [analysis['player']['Batter'] for analysis in profile_analysis.values() 
                               if analysis['player']['adj_XB'] + analysis['player']['adj_HR'] > 10 and analysis['player']['adj_K'] <= 17]
        if power_contact_players:
            insights.append(f"⚡ **Power + Contact Combo**: {', '.join(power_contact_players)}")
        
        # Show profile diversity
        total_profiles_available = len(profile_analysis)
        insights.append(f"📊 **Profile Diversity**: {total_profiles_available}/7 profiles have viable options")
        
        for insight in insights:
            st.success(insight)
            
        # Strategic recommendations based on available profiles
        st.markdown("**💡 Strategic Recommendations:**")
        
        power_available = any("Power" in p for p in profile_analysis.keys())
        contact_available = any(p in ["🏆 Contact-Aggressive", "⭐ Elite Contact"] for p in profile_analysis.keys())
        
        if power_available and contact_available:
            st.info("⚖️ **Balanced Strategy**: Both power and contact options available - diversify based on contest type")
        elif "💥 Contact Power" in profile_analysis:
            st.info("🎯 **Premium Power Strategy**: Contact power player available - ideal for tournaments needing ceiling")
        elif "🚀 Pure Power" in profile_analysis:
            st.info("💥 **High-Risk/High-Reward**: Pure power available - perfect for GPP leverage plays")
        elif "⚾ All Power" in profile_analysis:
            st.info("🔬 **Power Research Available**: Complete power landscape analysis - find hidden gems")
        elif "🏆 Contact-Aggressive" in profile_analysis:
            st.info("🛡️ **Safety Strategy**: Focus on Contact-Aggressive for consistent base hits")
        elif "⚡ Swing-Happy" in profile_analysis:
            st.info("🔥 **Aggressive Strategy**: Swing-Happy options available for leverage plays")
        
    else:
        st.warning("⚠️ No players available in any standard profiles after exclusions")
        st.markdown("**💡 Suggestions:**")
        st.markdown("- Try reducing exclusions or expanding to 'All Players' profile")
        st.markdown("- Check if filters are too restrictive for today's slate")
    
    # Additional lineup management tips
    excluded_players = st.session_state.get('excluded_players', [])
    if excluded_players:
        with st.expander("💡 Lineup Management Tips"):
            st.markdown(f"""
            **Players Currently Excluded**: {', '.join(excluded_players)}
            
            **Best Practices:**
            - ✅ Check official lineups 2-3 hours before first pitch
            - ✅ Monitor injury reports and weather delays
            - ✅ Have backup players ready from same profile
            - ✅ Use late swap strategy for uncertain players
            
            **Quick Actions:**
            - Remove players from exclusion list if lineups are confirmed
            - Add more players to exclusion if lineup news breaks
            """)
    else:
        with st.expander("💡 Lineup Confirmation Reminder"):
            st.markdown("""
            **🏟️ Don't forget to verify lineups!**
            
            - Check official team lineups 2-3 hours before games
            - Monitor for late scratches due to injury/rest
            - Weather delays can cause lineup changes
            - Use the "Players NOT Playing Today" filter to exclude confirmed outs
            """)
else:
    st.info("💡 Need at least 3 players for League Context Analysis")
```

def create_enhanced_visualizations(df, filtered_df):
“”“Create enhanced visualizations focused on base hit analysis.”””

```
st.subheader("📈 Base Hit Analysis Charts")

col1, col2 = st.columns(2)

with col1:
    # Score distribution
    chart1 = alt.Chart(df).mark_bar(
        color='#1f77b4',
        opacity=0.7
    ).encode(
        alt.X('Score:Q', bin=alt.Bin(maxbins=15), title='Base Hit Score'),
        alt.Y('count()', title='Number of Players'),
        tooltip=['count()']
    ).properties(
        title='Score Distribution (All Players)',
        width=350,
        height=300
    )
    
    st.altair_chart(chart1, use_container_width=True)

with col2:
    # Hit probability vs K risk scatter
    chart2 = alt.Chart(filtered_df).mark_circle(
        size=100,
        opacity=0.7
    ).encode(
        alt.X('total_hit_prob:Q', title='Total Hit Probability %'),
        alt.Y('adj_K:Q', title='Strikeout Risk %'),
        alt.Color('Score:Q', scale=alt.Scale(scheme='viridis')),
        alt.Size('adj_1B:Q', title='Single %'),
        tooltip=['Batter', 'total_hit_prob', 'adj_K', 'Score']
    ).properties(
        title='Hit Probability vs Strikeout Risk',
        width=350,
        height=300
    )
    
    st.altair_chart(chart2, use_container_width=True)

# Team performance summary
if not filtered_df.empty:
    team_stats = filtered_df.groupby('Tm').agg({
        'total_hit_prob': 'mean',
        'Score': 'mean',
        'Batter': 'count'
    }).round(1).reset_index()
    
    team_stats.columns = ['Team', 'Avg Hit Prob %', 'Avg Score', 'Players']
    team_stats = team_stats.sort_values('Avg Hit Prob %', ascending=False)
    
    st.subheader("🏟️ Team Performance Summary")
    st.dataframe(team_stats, use_container_width=True)
```

def main_page():
“”“Enhanced main page with league-aware focus.”””
create_league_aware_header()

```
# Load and process data
with st.spinner('🔄 Loading and analyzing today\'s matchups...'):
    df = load_and_process_data()

if df is None:
    st.error("❌ Unable to load data. Please check your internet connection and try again.")
    return

# Show data quality dashboard
create_data_quality_dashboard(df)

# Create league-aware filters with baseball intelligence
filters = create_league_aware_filters(df)

# Calculate league-aware scores with profile-specific logic
profile_type = filters.get('profile_type', 'contact')
df = calculate_league_aware_scores(df, profile_type)

# Apply intelligent filters
filtered_df = apply_league_aware_filters(df, filters)

# Display league-aware results
display_league_aware_results(filtered_df, filters)

# Create visualizations
create_enhanced_visualizations(df, filtered_df)

# Export functionality and lineup management
st.markdown("---")
col1, col2, col3, col4 = st.columns(4)

with col1:
    if st.button("📊 Export Results to CSV"):
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "💾 Download CSV", 
            csv, 
            f"mlb_league_aware_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv"
        )

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

with col3:
    if st.button("🏟️ Clear Exclusions"):
        st.session_state.excluded_players = []
        st.rerun()

with col4:
    st.info(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")

# Quick Lineup Management Section
if not filtered_df.empty:
    with st.expander("⚡ Quick Lineup Management"):
        st.markdown("**Quick exclude players from results:**")
        
        # Show players in current results for quick exclusion
        result_players = filtered_df['Batter'].tolist()
        current_exclusions = st.session_state.excluded_players
        
        # Only show players not already excluded
        available_to_exclude = [p for p in result_players if p not in current_exclusions]
        
        if available_to_exclude:
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown("**Players in Current Results:**")
                for i, player in enumerate(available_to_exclude[:5]):  # Show first 5
                    if st.button(f"❌ Exclude {player}", key=f"exclude_{i}"):
                        # Add to unified session state
                        if player not in st.session_state.excluded_players:
                            st.session_state.excluded_players.append(player)
                        st.rerun()
            
            with col_right:
                if len(available_to_exclude) > 5:
                    st.markdown("**More Players:**")
                    for i, player in enumerate(available_to_exclude[5:10]):  # Show next 5
                        if st.button(f"❌ Exclude {player}", key=f"exclude_more_{i}"):
                            # Add to unified session state
                            if player not in st.session_state.excluded_players:
                                st.session_state.excluded_players.append(player)
                            st.rerun()
        else:
            st.info("🎯 All players in results are already confirmed playing")
        
        # Show currently excluded players
        if current_exclusions:
            st.markdown("**Currently Excluded Players:**")
            excluded_display = ", ".join(current_exclusions)
            st.info(f"🚫 {excluded_display}")
            
            # Unified clear button that works with both systems
            if st.button("🔄 Re-include All Excluded Players", key="main_clear"):
                st.session_state.excluded_players = []
                st.session_state.clear_exclusions = True
                st.rerun()
        else:
            st.success("✅ All players currently included in analysis")

# Bottom tips with CORRECTED explanations
st.markdown("---")
st.markdown("""
### 💡 **V2.4 COMPLETE Strategy Tips**
- **Positive K% vs League**: Player strikes out less than league average (BETTER CONTACT = GREEN!)
- **Positive BB% vs League**: Player walks less than league average (MORE AGGRESSIVE = GREEN!)
- **Scores 70+**: Elite opportunities with league-superior metrics
- **🔥 NEW: Complete Power System**: 3 power profiles including All Power research mode
- **💎 Hidden Gems Profiles**: Updated thresholds catch more viable players automatically
- **XB% + HR% = Power Combo**: Target 10%+ combined for solid power threats
- **⚾ All Power Players**: Complete power research - find overlooked gems
- **📊 Weather & Parks**: Already factored into data - focus on lineup verification and strategy
- **Always verify lineups before finalizing picks**

**✅ Complete System: 8 Profiles | Power + Contact | Pre-Integrated Environmental Data**
""")
```

def info_page():
“”“Comprehensive reference manual for the MLB Hit Predictor system.”””
st.title(“📚 MLB Hit Predictor - Complete Reference Manual”)

```
with st.expander("📖 Script Overview", expanded=True):
    st.markdown("""
    ## 🎯 **What is the MLB Hit Predictor?**
    
    The MLB Hit Predictor is an advanced baseball analytics system designed to identify players with the highest probability of recording base hits in daily fantasy sports and sports betting. Unlike traditional analysis tools that rely on basic stats or arbitrary cutoffs, this system employs **league-aware analysis** and **profile-specific scoring algorithms** to provide contextual, data-driven player recommendations.
    
    ## 🎛️ **Core Purpose & Problems Solved**
    
    ### **Traditional Analysis Problems:**
    - **Arbitrary Thresholds**: Most systems use random cutoffs (e.g., "K% under 20%") without context
    - **One-Size-Fits-All**: Same criteria applied regardless of contest type or strategy
    - **Lack of Context**: No comparison to league averages or peer performance
    - **Manual Process**: Time-intensive research across multiple data sources
    - **Static Analysis**: No adaptation based on player archetypes or hitting styles
    
    ### **Our Solution:**
    - **League Context Integration**: All metrics compared to 2024 MLB league averages (K% 22.6%, BB% 8.5%)
    - **Profile-Based Analysis**: 8 distinct player profiles for different strategies
    - **Dynamic Scoring**: Algorithms adapt based on contact vs power focus
    - **Automated Research**: Multi-profile analysis shows best options across all approaches
    - **Real-Time Optimization**: Live filtering and ranking based on current slate conditions
    
    ## 🚀 **Key Innovations**
    
    ### **1. League-Aware Metrics**
    Instead of showing raw percentages, the system calculates performance relative to league averages:
    - **K% vs League**: `22.6% - Player K% = Performance Above/Below League`
    - **BB% vs League**: `8.5% - Player BB% = Aggression Above/Below League`
    - **Positive values = Better performance**, making interpretation intuitive
    
    ### **2. Profile-Specific Scoring**
    The system uses different algorithms based on player archetype:
    - **Contact Profiles**: Emphasize singles (weight 2.0), penalize strikeouts heavily (-2.0)
    - **Power Profiles**: Emphasize XB% (weight 3.0) and HR% (weight 2.5), reduce K penalty (-1.0)
    - **Research Profiles**: No restrictions, complete visibility into player pool
    
    ### **3. Hidden Gems Detection**
    Updated thresholds based on real-world performance data catch players who:
    - Just miss traditional "elite" categories but perform well
    - Have strong underlying metrics overlooked by basic analysis
    - Offer value due to lower ownership or salary inefficiencies
    
    ## 📊 **Target Users**
    - **DFS Players**: Cash games, GPP tournaments, all contest types
    - **Sports Bettors**: Hit props, player performance bets
    - **Fantasy Baseball**: Season-long league optimization
    - **Baseball Analysts**: Research and player evaluation
    """)

with st.expander("⚙️ How the System Works - Technical Workflow"):
    st.markdown("""
    ## 🔄 **Step-by-Step System Workflow**
    
    ### **Phase 1: Data Acquisition & Validation**
    
    #### **Data Sources:**
    1. **Base Probabilities CSV**: Core hitting metrics from ballpark-adjusted models
    2. **Adjustment Factors CSV**: Performance modifiers based on recent form, matchups, conditions
    
    **Note: Both data sources already incorporate:**
    - Weather conditions (temperature, wind, humidity)
    - Ballpark factors (dimensions, altitude, environmental conditions)  
    - Pitcher-specific matchup adjustments
    - Recent form and trend analysis
    
    #### **Validation Process:**
    ```python
    # Data validation checks:
    - File accessibility and download success
    - Required columns present: ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K']
    - No null values in key identification fields
    - Reasonable data ranges (percentages 0-100, etc.)
    ```
    
    ### **Phase 2: Data Processing & Metric Calculation**
    
    #### **Merge Operation:**
    ```python
    # Combine datasets on player-pitcher matchup
    merged_df = pd.merge(probabilities, adjustments,
                       on=['Tm', 'Batter', 'Pitcher'],
                       how='inner')
    ```
    
    #### **Adjusted Metric Calculations:**
    For each core metric (1B, XB, HR, K, BB, vs, RC):
    ```python
    # Formula: Base Probability × (1 + Adjustment Factor)
    adj_metric = base_probability × (1 + percentage_change/100)
    
    # Example: 
    # Base K% = 15.0%, Adjustment = +10%
    # Adjusted K% = 15.0 × (1 + 10/100) = 16.5%
    ```
    
    #### **Key Derived Metrics:**
    ```python
    # Total Hit Probability (capped at 100%)
    total_hit_prob = adj_1B + adj_XB + adj_HR
    
    # League Context Calculations
    K_vs_League = 22.6 - adj_K    # Positive = better contact
    BB_vs_League = 8.5 - adj_BB   # Positive = more aggressive
    
    # Power Probability (for power profiles)
    power_prob = adj_XB + adj_HR  # Excludes singles
    ```
    
    ### **Phase 3: Profile-Based Scoring**
    
    #### **Contact Scoring Algorithm:**
    ```python
    contact_weights = {
        'adj_1B': 2.0,      # Primary base hit source
        'adj_XB': 1.8,      # Extra base hits
        'adj_vs': 1.2,      # Matchup performance
        'adj_RC': 0.8,      # Run creation
        'adj_HR': 0.6,      # Home runs
        'adj_K': -2.0,      # Heavy strikeout penalty
        'adj_BB': -0.8      # Walk penalty (not hits)
    }
    ```
    
    #### **Power Scoring Algorithm:**
    ```python
    power_weights = {
        'adj_XB': 3.0,      # Primary power focus
        'adj_HR': 2.5,      # Home run emphasis  
        'adj_vs': 1.5,      # Enhanced matchup weight
        'adj_RC': 1.0,      # Run creation
        'adj_1B': 0.5,      # Reduced singles focus
        'adj_K': -1.0,      # Reduced K penalty
        'adj_BB': -0.3      # Minimal walk penalty
    }
    ```
    
    #### **Bonus Calculations:**
    ```python
    # Contact Bonuses
    contact_bonus = +8 if (hit_prob > 40% AND K% < 18%)
    consistency_bonus = +5 if (1B% > 20% AND XB% > 8%)
    matchup_bonus = +3 if (vs_pitcher > 5)
    
    # Power Bonuses  
    elite_power_bonus = +12 if (XB% > 10% AND HR% > 4%)
    clutch_power_bonus = +8 if (XB% > 8% AND vs_pitcher > 5)
    consistent_power_bonus = +5 if (HR% > 3% AND K% < 25%)
    ```
    
    ### **Phase 4: Filtering & Ranking**
    
    #### **Profile Application:**
    Each profile applies specific filters:
    ```python
    # Example: Contact-Aggressive profile
    filtered_players = players.query(
        "adj_K <= 19.0 and adj_BB <= 7.0 and total_hit_prob >= 32"
    )
    ```
    
    #### **Final Score Calculation:**
    ```python
    # Weighted sum + bonuses
    base_score = sum(metric × weight for metric, weight in weights.items())
    final_score = base_score + sum(applicable_bonuses)
    
    # Normalize to 0-100 scale
    normalized_score = (score - min_score) / (max_score - min_score) × 100
    ```
    
    ### **Phase 5: Results & Analysis**
    
    #### **Multi-Profile Analysis:**
    System runs all 8 profiles simultaneously to identify:
    - Best player from each approach
    - Profile diversity available
    - Strategic recommendations based on available options
    
    #### **Real-Time Feedback:**
    Sidebar provides live preview showing:
    - Number of players matching current filters
    - Average performance vs league metrics
    - Exclusion impact from lineup management
    """)

with st.expander("🧮 Calculation Methods - Formulas & Logic"):
    st.markdown("""
    ## 📐 **Core Metric Calculations**
    
    ### **Base Metric Adjustments**
    
    #### **Adjustment Formula:**
    ```python
    # For all metrics (K, BB, 1B, XB, HR, vs, RC):
    adjusted_value = base_value × (1 + percentage_change / 100)
    
    # Real Examples:
    # Player A: Base K% = 20%, Adjustment = -15% (recent improvement)
    # Adjusted K% = 20 × (1 + (-15)/100) = 20 × 0.85 = 17.0%
    
    # Player B: Base HR% = 3.5%, Adjustment = +25% (hot streak)  
    # Adjusted HR% = 3.5 × (1 + 25/100) = 3.5 × 1.25 = 4.375%
    ```
    
    #### **Clipping & Validation:**
    ```python
    # Percentage metrics clipped to realistic ranges
    K_rate = clip(adjusted_K, min=0, max=100)
    BB_rate = clip(adjusted_BB, min=0, max=100)
    Hit_rates = clip(adjusted_hits, min=0, max=100)
    
    # Other metrics clipped to non-negative
    vs_pitcher = clip(adjusted_vs, min=0)
    run_creation = clip(adjusted_RC, min=0)
    ```
    
    ### **League Context Calculations**
    
    #### **Performance vs League Averages:**
    ```python
    # 2024 MLB League Averages
    LEAGUE_K_AVG = 22.6%    # Strikeout rate
    LEAGUE_BB_AVG = 8.5%    # Walk rate
    
    # Player Performance Calculation
    K_performance = LEAGUE_K_AVG - player_K_rate
    BB_performance = LEAGUE_BB_AVG - player_BB_rate
    
    # Interpretation:
    # K_performance = +5.0 → Player strikes out 5% less than league (GOOD)
    # K_performance = -3.0 → Player strikes out 3% more than league (BAD)
    # BB_performance = +2.0 → Player walks 2% less than league (MORE AGGRESSIVE)
    # BB_performance = -4.0 → Player walks 4% more than league (LESS AGGRESSIVE)
    ```
    
    ### **Hit Probability Calculations**
    
    #### **Total Hit Probability:**
    ```python
    # Sum of all ways to get a hit
    total_hit_prob = adj_1B + adj_XB + adj_HR
    
    # Cap at 100% (theoretical maximum)
    total_hit_prob = min(total_hit_prob, 100.0)
    
    # Example calculation:
    # Player: 1B=25%, XB=8%, HR=4%
    # Total = 25 + 8 + 4 = 37% hit probability
    ```
    
    #### **Power Probability (Power Profiles Only):**
    ```python
    # Excludes singles - focuses on extra bases
    power_prob = adj_XB + adj_HR
    
    # Realistic cap for power production
    power_prob = min(power_prob, 50.0)
    
    # Example:
    # Power Player: XB=12%, HR=5%
    # Power Prob = 12 + 5 = 17% extra base probability
    ```
    
    ### **Scoring Algorithm Details**
    
    #### **Contact Profile Scoring:**
    ```python
    def calculate_contact_score(player_metrics):
        # Base weighted calculation
        score = (
            player_metrics['adj_1B'] * 2.0 +      # Singles emphasis
            player_metrics['adj_XB'] * 1.8 +      # Extra bases
            player_metrics['adj_vs'] * 1.2 +      # Matchup
            player_metrics['adj_RC'] * 0.8 +      # Run creation
            player_metrics['adj_HR'] * 0.6 +      # Home runs
            player_metrics['adj_K'] * (-2.0) +    # K penalty
            player_metrics['adj_BB'] * (-0.8)     # BB penalty
        )
        
        # Conditional bonuses
        if player_metrics['total_hit_prob'] > 40 and player_metrics['adj_K'] < 18:
            score += 8  # Elite contact bonus
        
        if player_metrics['adj_1B'] > 20 and player_metrics['adj_XB'] > 8:
            score += 5  # Consistency bonus
        
        if player_metrics['adj_vs'] > 5:
            score += 3  # Matchup bonus
            
        return score
    ```
    
    #### **Power Profile Scoring:**
    ```python
    def calculate_power_score(player_metrics):
        # Power-focused weighting
        score = (
            player_metrics['adj_XB'] * 3.0 +      # Extra bases primary
            player_metrics['adj_HR'] * 2.5 +      # HR emphasis
            player_metrics['adj_vs'] * 1.5 +      # Enhanced matchup
            player_metrics['adj_RC'] * 1.0 +      # Run creation
            player_metrics['adj_1B'] * 0.5 +      # Reduced singles
            player_metrics['adj_K'] * (-1.0) +    # Reduced K penalty
            player_metrics['adj_BB'] * (-0.3)     # Minimal BB penalty
        )
        
        # Power-specific bonuses
        if player_metrics['adj_XB'] > 10 and player_metrics['adj_HR'] > 4:
            score += 12  # Elite power combo
        
        if player_metrics['adj_XB'] > 8 and player_metrics['adj_vs'] > 5:
            score += 8   # Clutch power
        
        if player_metrics['adj_HR'] > 3 and player_metrics['adj_K'] < 25:
            score += 5   # Consistent power
            
        return score
    ```
    
    #### **Score Normalization:**
    ```python
    # Convert raw scores to 0-100 scale
    def normalize_scores(scores):
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:  # All identical
            return [50.0] * len(scores)
        
        normalized = []
        for score in scores:
            norm_score = (score - min_score) / (max_score - min_score) * 100
            normalized.append(round(norm_score, 1))
        
        return normalized
    ```
    
    ### **Advanced Calculations**
    
    #### **Multi-Profile Ranking:**
    ```python
    # Each player gets multiple scores based on applicable profiles
    for profile in applicable_profiles:
        if profile_type == 'contact':
            score = calculate_contact_score(player)
        elif profile_type == 'power':
            score = calculate_power_score(player)
        
        player.scores[profile] = score
    
    # Rank within each profile
    for profile in profiles:
        ranked_players = sort_by_score(players_in_profile, profile)
    ```
    
    #### **Exclusion Impact Calculations:**
    ```python
    # Real-time filter preview
    def preview_filter_results(players, filters, exclusions):
        # Apply exclusions first
        active_players = players.exclude(exclusions)
        
        # Apply profile filters
        filtered_players = active_players.filter_by_profile(filters)
        
        # Calculate impact metrics
        exclusion_impact = len(players) - len(active_players)
        filter_impact = len(active_players) - len(filtered_players)
        
        return {
            'matching': len(filtered_players),
            'excluded': exclusion_impact,
            'filtered_out': filter_impact
        }
    ```
    """)

with st.expander("🎯 Profile System - Complete Breakdown"):
    st.markdown("""
    ## 🔍 **Profile Functions & Logic**
    
    ### **Contact Profile Category (4 Profiles)**
    
    #### **🏆 Contact-Aggressive Hitters**
    ```python
    # Filter Criteria
    max_K_rate = 19.0%          # Above-average contact
    max_BB_rate = 7.0%          # Aggressive approach
    min_hit_prob = 32.0%        # Solid hit probability
    
    # Key Stats Considered
    - K% (primary): Strikeout avoidance for contact
    - BB% (primary): Aggression level
    - 1B% (weighted 2.0): Singles production
    - Total Hit Prob: Combined hit likelihood
    
    # Player Archetype
    - Contact-first approach with aggressive swinging
    - Avoids walks, attacks hittable pitches
    - Consistent base hit production
    - Lower strikeout rates than league average
    
    # Ideal Use Cases
    - DFS cash games (safety-focused)
    - Pitcher-heavy slates (tough matchups)
    - Weather conditions favoring contact
    - Small slates needing reliable production
    
    # Example Players
    - Luis Arraez, José Altuve, Gleyber Torres (good years)
    ```
    
    #### **⭐ Elite Contact Specialists**
    ```python
    # Filter Criteria  
    max_K_rate = 14.0%          # Elite contact skills
    max_BB_rate = 9.5%          # Disciplined but not passive
    min_hit_prob = 28.0%        # High-floor players
    
    # Key Stats Considered
    - K% (elite threshold): Top 10% contact skills
    - Consistency metrics: Low volatility
    - Matchup resistance: Hits all pitcher types
    
    # Player Archetype
    - Elite bat-to-ball skills
    - Rarely strikes out (top 10% of MLB)
    - Can hit elite pitching
    - Premium plays with highest floors
    
    # Ideal Use Cases
    - High-stakes contests requiring confidence
    - Tournament finals (small fields)
    - Ace pitcher matchups (Cy Young candidates)
    - Maximum safety needed
    
    # Example Players
    - Steven Kwan, Luis Arraez, Juan Soto (contact years)
    ```
    
    #### **⚡ Swing-Happy Hitters**
    ```python
    # Filter Criteria
    max_K_rate = 24.0%          # Accept league-average contact
    max_BB_rate = 5.0%          # Ultra-aggressive approach
    min_hit_prob = 30.0%        # Decent hit rates
    
    # Key Stats Considered
    - BB% (primary): Extreme aggression
    - Swing patterns: First-pitch swinging tendencies
    - At-bat efficiency: Quick plate appearances
    
    # Player Archetype
    - Attack-first mentality
    - Rarely walks (bottom 10% of MLB)
    - More swings = more hit opportunities
    - Higher variance but unique approach
    
    # Ideal Use Cases
    - GPP tournaments (contrarian plays)
    - High-scoring environments
    - Fast-paced games (more at-bats)
    - Stacking strategies (team offense)
    
    # Example Players
    - Vladimir Guerrero Sr., Tim Anderson, Bo Bichette
    ```
    
    #### **🔷 Above-Average Contact**
    ```python
    # Filter Criteria
    max_K_rate = 20.0%          # Better than league average
    max_BB_rate = 12.0%         # Reasonable discipline
    min_hit_prob = 25.0%        # Accessible threshold
    
    # Key Stats Considered
    - Balanced approach to K% and BB%
    - Consistent production metrics
    - Versatility across matchups
    
    # Player Archetype
    - Well-rounded hitters
    - Better than 60% of MLB in contact
    - Reliable without being elite
    - Good sample size of options
    
    # Ideal Use Cases
    - Learning the system
    - Mixed contest types
    - Uncertain conditions
    - Building multiple lineups
    
    # Example Players
    - Most everyday MLB players with decent contact
    ```
    
    ### **Power Profile Category (3 Profiles)**
    
    #### **💥 Contact Power Hitters**
    ```python
    # Filter Criteria
    max_K_rate = 20.0%          # Reasonable contact
    max_BB_rate = 12.0%         # Balanced approach
    min_XB_rate = 7.0%          # Solid extra base production
    min_HR_rate = 2.5%          # Legitimate HR threat
    min_vs_pitcher = -5         # Decent matchups
    
    # Key Stats Considered
    - XB% (weighted 3.0): Extra base emphasis
    - HR% (weighted 2.5): Home run production
    - K% (reduced penalty): Contact still matters
    - Power + Contact balance
    
    # Player Archetype
    - Best of both worlds: contact + power
    - Can hit for average AND power
    - Lower strikeout risk than pure sluggers
    - Multiple ways to return value
    
    # Ideal Use Cases
    - Tournament play needing ceiling
    - Hitter-friendly parks (Coors, Yankees)
    - Good weather conditions
    - Balance of safety and upside
    
    # Example Players
    - Mookie Betts, Freddie Freeman, Vladimir Guerrero Jr.
    ```
    
    #### **🚀 Pure Power Sluggers**
    ```python
    # Filter Criteria
    max_K_rate = 100%           # Accept high strikeouts
    max_BB_rate = 100%          # Accept high walks
    min_XB_rate = 9.0%          # Elite extra base rate
    min_HR_rate = 3.5%          # Elite HR production
    min_vs_pitcher = -10        # Any matchup if power elite
    
    # Key Stats Considered
    - XB% + HR% combination (primary)
    - Raw power metrics over contact
    - Ceiling potential over floor
    - Matchup can be overcome by elite power
    
    # Player Archetype
    - Maximum power potential
    - Boom-or-bust profile
    - Can overcome poor matchups with raw power
    - Game-changing upside
    
    # Ideal Use Cases
    - GPP tournaments (large fields)
    - Power-friendly environments
    - Contrarian plays when others avoid power
    - Leverage/differentiation needs
    
    # Example Players
    - Aaron Judge, Pete Alonso, Kyle Schwarber, Mike Trout
    ```
    
    #### **⚾ All Power Players (Research Mode)**
    ```python
    # Filter Criteria
    max_K_rate = 100%           # No restrictions
    max_BB_rate = 100%          # No restrictions
    min_XB_rate = 0%            # No restrictions
    min_HR_rate = 0%            # No restrictions
    
    # Scoring Algorithm
    - Uses power scoring weights on entire pool
    - Reveals complete power hierarchy
    - Shows moderate power players ranked highly
    
    # Player Archetype
    - Complete spectrum from elite to moderate power
    - Hidden gems with decent power overlooked
    - Value plays not meeting strict thresholds
    - Research-focused comprehensive view
    
    # Ideal Use Cases
    - Initial power research
    - Finding contrarian power plays
    - Salary cap optimization
    - Large field differentiation
    
    # Output
    - All players ranked by power algorithm
    - May include players with 5-6% XB rates
    - Good for discovering overlooked options
    ```
    
    ### **Research Profile Category (1 Profile)**
    
    #### **🌐 All Players**
    ```python
    # Filter Criteria
    max_K_rate = 100%           # No restrictions
    max_BB_rate = 100%          # No restrictions
    min_hit_prob = 20%          # Minimal threshold
    
    # Scoring Algorithm
    - Uses contact scoring on entire pool
    - Shows complete player hierarchy
    - Useful for backup options
    
    # Use Cases
    - When profiles are too restrictive
    - Complete slate overview
    - Learning tool to understand metrics
    - Backup option discovery
    ```
    
    ### **Profile Selection Logic**
    
    #### **Contest Type Mapping:**
    ```python
    profile_recommendations = {
        'cash_games': {
            'primary': ['Contact-Aggressive', 'Elite Contact'],
            'secondary': ['Above-Average Contact'],
            'avoid': ['Pure Power', 'Swing-Happy']
        },
        'gpp_small_field': {
            'primary': ['Contact-Aggressive', 'Contact Power'],
            'secondary': ['Elite Contact', 'Pure Power'],
            'research': ['All Power']
        },
        'gpp_large_field': {
            'primary': ['Contact Power', 'Pure Power'],
            'secondary': ['Contact-Aggressive'],
            'research': ['All Power', 'Swing-Happy']
        }
    }
    ```
    
    #### **Environmental Adjustments:**
    ```python
    def adjust_profiles_for_conditions(base_profiles, conditions):
        if conditions['temperature'] < 60:
            reduce_power_weight(base_profiles, factor=0.7)
        
        if conditions['park_factor'] > 1.1:  # Hitter-friendly
            increase_power_weight(base_profiles, factor=1.3)
        
        if conditions['wind_speed'] > 15 and conditions['wind_direction'] == 'out':
            increase_power_weight(base_profiles, factor=1.2)
    ```
    """)

with st.expander("💡 Best Practices & Advanced Strategies"):
    st.markdown("""
    ## 🎯 **Optimal System Usage**
    
    ### **Profile Selection Strategy**
    
    #### **Contest Type Framework:**
    ```python
    # Decision Tree for Profile Selection
    
    if contest_type == 'cash_game':
        primary_profiles = ['Contact-Aggressive', 'Elite Contact']
        allocation = {'Contact-Aggressive': 70%, 'Elite Contact': 30%}
        avoid = ['Pure Power', 'Swing-Happy']  # Too volatile
    
    elif contest_type == 'gpp_small_field':  # <100 entries
        primary_profiles = ['Contact-Aggressive', 'Contact Power']
        allocation = {'Contact-Aggressive': 40%, 'Contact Power': 35%, 'Elite Contact': 25%}
    
    elif contest_type == 'gpp_large_field':  # 100+ entries
        primary_profiles = ['Contact Power', 'Pure Power', 'All Power']
        allocation = {'Contact Power': 35%, 'Pure Power': 30%, 'All Power': 20%, 'Other': 15%}
    ```
    
    #### **Data-Integrated Adjustments:**
    ```python
    # Note: Weather factors already incorporated in CSV data
    # No manual weather adjustments needed - focus on other factors
    
    def adjust_for_slate_conditions(base_allocation, conditions):
        pitcher_quality = conditions.average_pitcher_era
        park_factors = conditions.ballpark_effects  # Also in data
        slate_size = conditions.number_of_games
        
        if pitcher_quality < 3.50:  # Strong pitching slate
            increase_contact_profiles(base_allocation, increase=20%)
            reduce_power_profiles(base_allocation, reduction=15%)
        
        if slate_size < 6:  # Small slate
            increase_safety_profiles(base_allocation, increase=25%)
        
        if slate_size > 12:  # Large slate  
            increase_research_profiles(base_allocation, increase=20%)
    ```
    
    ### **Research Workflow Optimization**
    
    #### **Systematic Analysis Process:**
    ```
    1. START: All Power Players Profile
       - Get complete power landscape
       - Identify power tier breaks
       - Note elite power players (15%+ combined XB+HR)
    
    2. REFINE: Pure Power Sluggers
       - Focus on elite power stats
       - Check matchup quality for power players
       - Identify boom/bust candidates
    
    3. BALANCE: Contact Power Hitters  
       - Find power+safety combinations
       - Look for players with 10%+ power + <20% K rate
       - Tournament ceiling with decent floor
    
    4. SAFETY: Contact-Aggressive Hitters
       - Establish baseline reliable options
       - Focus on +3% or better vs league in both K% and BB%
       - Cash game anchors and GPP safety plays
    
    5. PREMIUM: Elite Contact Specialists
       - Identify truly elite contact (sub-14% K rate)
       - Premium plays for tough matchups
       - High-stakes contest options
    
    6. CONTRARIAN: Swing-Happy Hitters
       - Find ultra-aggressive approach players
       - Contrarian tournament plays
       - Stack-building components
    
    7. VALIDATE: All Players + Above-Average
       - Ensure no hidden gems missed
       - Check for value plays not meeting strict criteria
       - Backup options if main profiles limited
    ```
    
    #### **Advanced Filtering Techniques:**
    ```python
    # Custom Filter Combinations
    
    # Tournament Ceiling Hunt
    high_ceiling_filter = {
        'min_combined_power': 12,  # XB% + HR% ≥ 12%
        'max_K_rate': 25,          # Accept some swing-and-miss
        'min_vs_pitcher': 0,       # Decent matchup required
        'profile_diversity': True   # Multiple profile options
    }
    
    # Cash Game Safety Net
    safety_filter = {
        'max_K_rate': 18,          # Strong contact
        'min_hit_prob': 35,        # High hit probability
        'max_ownership': 25,       # Not overly popular
        'consistent_production': True  # Low volatility
    }
    
    # Value Hunting
    value_filter = {
        'salary_percentile': '<60', # Lower cost
        'recent_form': 'improving',  # Positive trend (in data)
        'matchup_grade': '>= B-',   # Decent spot
        'projected_ownership': '<15' # Contrarian
    }
    
    # Note: Weather and park factors already integrated in CSV data
    # Focus on strategic differentiation instead
    ```
    
    ### **Lineup Construction Integration**
    
    #### **Portfolio Approach:**
    ```python
    # Multi-Entry Tournament Strategy
    
    def build_lineup_portfolio(num_entries):
        base_allocations = {
            'anchor_contact': 0.4,      # 40% reliable contact
            'ceiling_power': 0.3,       # 30% power upside
            'contrarian': 0.2,          # 20% differentiation
            'value_hunting': 0.1        # 10% salary optimization
        }
        
        lineups = []
        for entry in range(num_entries):
            lineup = {}
            
            # Anchor positions (high-confidence plays)
            anchor_players = select_from_profile('Contact-Aggressive')
            lineup['anchors'] = anchor_players[:2]
            
            # Ceiling positions (tournament upside)
            ceiling_players = select_from_profile(['Contact Power', 'Pure Power'])
            lineup['ceiling'] = ceiling_players[:2]
            
            # Contrarian positions (differentiation)
            contrarian_players = select_from_profile(['Swing-Happy', 'All Power'])
            lineup['contrarian'] = contrarian_players[:1]
            
            lineups.append(lineup)
    ```
    
    ### **Advanced Metrics Interpretation**
    
    #### **League Context Deep Analysis:**
    ```python
    # Advanced Performance Metrics
    
    def analyze_league_context(player_stats):
        k_performance = 22.6 - player_stats.K_rate
        bb_performance = 8.5 - player_stats.BB_rate
        
        # Tier Classification
        if k_performance >= 8:
            contact_tier = "Elite"  # Top 5%
        elif k_performance >= 4:
            contact_tier = "Excellent"  # Top 20%
        elif k_performance >= 0:
            contact_tier = "Above Average"
        else:
            contact_tier = "Below Average"
        
        # Aggression Analysis
        if bb_performance >= 4:
            aggression = "Hyper-Aggressive"
        elif bb_performance >= 1:
            aggression = "Aggressive"
        elif bb_performance >= -1:
            aggression = "Balanced"
        else:
            aggression = "Patient"
        
        return {
            'contact_tier': contact_tier,
            'aggression_level': aggression,
            'k_advantage': k_performance,
            'bb_advantage': bb_performance
        }
    ```
    
    ### **Risk Management**
    
    #### **Exposure Guidelines:**
    ```python
    # Risk-Based Exposure Limits
    
    exposure_limits = {
        'Elite Contact': {'max': 40%, 'optimal': 25%},      # High floor
        'Contact-Aggressive': {'max': 50%, 'optimal': 35%}, # Balanced
        'Contact Power': {'max': 35%, 'optimal': 25%},      # Medium risk
        'Pure Power': {'max': 25%, 'optimal': 15%},         # High risk
        'Swing-Happy': {'max': 20%, 'optimal': 10%},        # Volatile
        'All Power Research': {'max': 30%, 'optimal': 20%}  # Discovery
    }
    ```
    
    #### **Bankroll Considerations:**
    ```python
    # Contest Selection Based on Profile Usage
    
    def recommend_contests(profile_usage):
        if profile_usage['safe_profiles'] > 70:  # Contact-heavy
            return ['cash_games', 'small_gpp', 'satellite_qualifiers']
        
        elif profile_usage['power_profiles'] > 50:  # Power-heavy
            return ['large_gpp', 'high_variance_tournaments']
        
        else:  # Balanced
            return ['medium_gpp', 'mixed_contests']
    ```
    """)

with st.expander("🔬 Results Interpretation & Advanced Insights"):
    st.markdown("""
    ## 📊 **Understanding Your Results**
    
    ### **Score Interpretation Framework**
    
    #### **Score Ranges & Meanings:**
    ```python
    score_tiers = {
        90-100: "Elite Opportunity",
        80-89:  "Excellent Play", 
        70-79:  "Very Good Option",
        60-69:  "Good Consideration",
        50-59:  "Average/Decent",
        40-49:  "Below Average",
        0-39:   "Avoid Unless Contrarian"
    }
    
    # Score Components Analysis
    def interpret_score(player_score, breakdown):
        base_performance = breakdown['weighted_stats']
        bonus_contribution = breakdown['bonuses']
        
        if bonus_contribution > 15:
            interpretation = "Bonus-driven (elite metrics combination)"
        elif base_performance > 70:
            interpretation = "Fundamentally strong (solid base stats)"
        else:
            interpretation = "Moderate opportunity (average performance)"
    ```
    
    #### **League Context Analysis:**
    ```python
    # Advanced League Comparison
    def analyze_vs_league_performance(player):
        k_advantage = 22.6 - player.K_rate
        bb_advantage = 8.5 - player.BB_rate
        
        # Combined Performance Score
        combined_advantage = k_advantage + bb_advantage
        
        if combined_advantage >= 10:
            return "Elite vs League (Top 5%)"
        elif combined_advantage >= 6:
            return "Excellent vs League (Top 15%)"
        elif combined_advantage >= 2:
            return "Above Average vs League"
        elif combined_advantage >= -2:
            return "Near League Average"
        else:
            return "Below League Average"
    ```
    
    ### **Color Coding Deep Dive**
    
    #### **Gradient Interpretation:**
    ```python
    # Green Intensity Meanings
    dark_green = "Significantly better than league average (top 10%)"
    medium_green = "Notably better than league average (top 25%)" 
    light_green = "Somewhat better than league average"
    
    # Red Intensity Meanings
    light_red = "Somewhat worse than league average"
    medium_red = "Notably worse than league average (bottom 25%)"
    dark_red = "Significantly worse than league average (bottom 10%)"
    
    # Score Color Mapping
    score_colors = {
        'Dark Green (85+)': "Must-play in most contests",
        'Green (70-84)': "Strong consideration",
        'Yellow-Green (55-69)': "Solid option",
        'Yellow (45-54)': "Average/filler play", 
        'Red (0-44)': "Avoid unless specific reason"
    }
    ```
    
    ### **Multi-Profile Analysis Interpretation**
    
    #### **Profile Diversity Signals:**
    ```python
    # Strategic Implications
    
    def interpret_profile_diversity(available_profiles):
        if len(available_profiles) >= 6:
            strategy = "Diverse slate - multiple approaches viable"
            recommendation = "Use balanced portfolio approach"
        
        elif 'Pure Power' in available_profiles and 'Elite Contact' in available_profiles:
            strategy = "Polar slate - choose your approach"
            recommendation = "Pick power OR contact, avoid middle"
        
        elif only_contact_profiles(available_profiles):
            strategy = "Contact-heavy slate - power limited"
            recommendation = "Focus on contact differentiation"
        
        elif only_power_profiles(available_profiles):
            strategy = "Power slate - contact limited"  
            recommendation = "Embrace power plays, accept volatility"
    ```
    
    ### **Common Interpretation Mistakes**
    
    #### **Pitfalls to Avoid:**
    ```python
    # Mistake 1: Raw Score Obsession
    wrong_approach = "Always pick highest score regardless of context"
    right_approach = "Consider score within contest type and ownership context"
    
    # Mistake 2: Ignoring League Context
    wrong_approach = "17% K rate looks average"
    right_approach = "17% K rate is +5.6% better than league = excellent contact"
    
    # Mistake 3: Profile Misuse
    wrong_approach = "Use Pure Power in cash games"
    right_approach = "Match profile risk to contest type"
    
    # Mistake 4: Single-Profile Analysis
    wrong_approach = "Only look at one profile"
    right_approach = "Compare across multiple profiles for complete picture"
    
    # Mistake 5: Ignoring Lineup Status
    wrong_approach = "Build lineup then check if players are playing"
    right_approach = "Exclude non-playing players before analysis"
    ```
    
    ### **Advanced Scenarios & Edge Cases**
    
    #### **When Profiles Disagree:**
    ```python
    # Resolution Framework
    
    if contact_score > power_score + 20:
        recommendation = "Clear contact play - avoid in power-heavy contests"
    
    elif power_score > contact_score + 20:
        recommendation = "Clear power play - avoid in contact-heavy contests"
    
    elif abs(contact_score - power_score) < 10:
        recommendation = "Versatile player - good for balanced contests"
    
    # Data-Driven Decision Logic (weather already factored in CSV)
    if pitcher_era < 3.00 and power_score > contact_score:
        recommendation = "Tough pitching favors contact despite power score"
    
    if slate_size > 12 and contact_score > power_score:
        recommendation = "Large slate allows contrarian power despite contact score"
    ```
    
    #### **Limited Profile Availability:**
    ```python
    # Adaptation Strategies
    
    def handle_limited_profiles(available_profiles):
        if len(available_profiles) < 3:
            strategy = "Restrictive slate - expand criteria slightly"
            action = "Use custom overrides to expand player pool"
        
        elif no_elite_profiles_available:
            strategy = "Average slate - no standout plays"
            action = "Focus on value and differentiation"
        
        elif only_research_profiles_available:
            strategy = "Deep dive needed - hidden value slate"
            action = "Use All Players and All Power for manual curation"
    ```
    
    ### **Performance Validation**
    
    #### **Tracking System Effectiveness:**
    ```python
    # Success Metrics to Track
    
    tracking_metrics = {
        'hit_rate_by_profile': {
            'Elite Contact': 'Should achieve 60%+ hit rate',
            'Contact-Aggressive': 'Should achieve 55%+ hit rate', 
            'Pure Power': 'Volatile - track extra base rate',
            'Contact Power': 'Balance of hits and power production'
        },
        
        'roi_by_contest_type': {
            'cash_games': 'Consistent small profits',
            'small_gpp': 'Steady cashes with occasional wins',
            'large_gpp': 'Lower cash rate but higher win rate'
        },
        
        'profile_correlation': {
            'weather_impact': 'Power profiles in cold vs warm',
            'park_factors': 'Power boost in hitter parks',
            'pitcher_quality': 'Contact profiles vs aces'
        }
    }
    ```
    
    ### **Troubleshooting Guide**
    
    #### **Common Issues & Solutions:**
    ```python
    # Problem: No players showing up
    solutions = [
        "Check if excluded players list is too extensive",
        "Try less restrictive profiles (Above-Average, All Players)",
        "Use custom overrides to expand criteria",
        "Verify data loaded correctly"
    ]
    
    # Problem: All scores seem low
    solutions = [
        "Scores are relative - focus on ranking not absolute values",
        "Low-scoring slate may indicate tough pitching",
        "Consider expanding to research profiles",
        "Check if filters too restrictive"
    ]
    
    # Problem: Profile recommendations seem off
    solutions = [
        "Verify contest type selection matches strategy",
        "Check slate-specific conditions (pitcher quality, game count)",
        "Consider ownership and contrarian opportunities", 
        "Use multi-profile analysis for broader perspective",
        "Remember: Weather/park factors already in data - focus on strategy"
    ]
    ```
    
    ## 🎯 **System Mastery Checklist**
    
    ### **Beginner Level:**
    - [ ] Understand league context calculations
    - [ ] Can interpret positive/negative vs league metrics
    - [ ] Know difference between contact and power profiles  
    - [ ] Can select appropriate profile for contest type
    
    ### **Intermediate Level:**
    - [ ] Use multi-profile analysis effectively
    - [ ] Understand score components and bonuses
    - [ ] Can adapt profiles based on weather/parks
    - [ ] Manage lineup exclusions systematically
    
    ### **Advanced Level:**
    - [ ] Build portfolio approaches across profiles
    - [ ] Identify hidden gems using research profiles
    - [ ] Integrate system with external factors
    - [ ] Track and validate system performance
    
    ### **Expert Level:**
    - [ ] Create custom filtering strategies
    - [ ] Combine profiles for unique approaches
    - [ ] Predict optimal profile allocation by slate
    - [ ] Mentor others on system usage
    """)

st.markdown("---")
st.markdown("""
## 📖 **Reference Manual Summary**

The MLB Hit Predictor represents a comprehensive evolution in baseball analytics, moving beyond traditional arbitrary thresholds to provide **contextual, league-aware analysis** tailored to specific contest types and player archetypes.

### **Key System Advantages:**
✅ **League Context Integration** - All metrics compared to 2024 MLB averages  
✅ **Profile-Specific Scoring** - Algorithms adapt to contact vs power focus  
✅ **Hidden Gems Detection** - Updated thresholds catch overlooked performers  
✅ **Multi-Profile Analysis** - See best options across all approaches  
✅ **Real-Time Optimization** - Live filtering with lineup management  
✅ **Advanced Research Tools** - Complete power and contact spectrum analysis  
✅ **Strategic Framework** - Contest-specific recommendations and best practices  
✅ **Comprehensive Documentation** - Complete understanding of all calculations and logic  

### **System Mastery Path:**
1. **Foundation** - Learn profile basics and league context
2. **Application** - Match profiles to contest types  
3. **Optimization** - Use multi-profile analysis and research tools
4. **Mastery** - Develop custom strategies and portfolio approaches

This system transforms baseball analysis from guesswork into **data-driven decision making**, providing the tools and knowledge needed to consistently identify optimal hitting opportunities across all contest types and market conditions.

---

*Complete Baseball Analytics Reference Manual | A1FADED V2.4 Comprehensive Edition*  
*Master every aspect of league-aware baseball analysis*
""")
```

def main():
“”“Enhanced main function with league-aware navigation.”””
st.sidebar.title(“🏟️ Navigation”)

```
# Optional music controls
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
    ["🎯 League-Aware Predictor", "📚 Baseball Guide"],
    index=0
)

if app_mode == "🎯 League-Aware Predictor":
    main_page()
else:
    info_page()

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**V2.4** | Complete + Hidden Gems")
```

if **name** == “**main**”:
main()