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
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'
    },
    'base_hit_weights': {
        'adj_1B': 2.0,      # Singles (primary base hit)
        'adj_XB': 1.8,      # Extra bases (also base hits)
        'adj_vs': 1.2,      # Matchup performance
        'adj_RC': 0.8,      # Run creation
        'adj_HR': 0.6,      # Home runs (also base hits)
        'adj_K': -2.0,      # Heavy penalty for strikeouts (no hit)
        'adj_BB': -0.8      # Moderate penalty for walks (not hits)
    },
    'expected_columns': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
    'cache_ttl': 900  # 15 minutes
}

# Custom CSS with enhanced styling
st.markdown("""
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
""", unsafe_allow_html=True)

# Enhanced Data Loading with Error Handling and Validation
@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_csv_with_validation(url, description, expected_columns):
    """Load and validate CSV data with comprehensive error handling."""
    try:
        with st.spinner(f'Loading {description}...'):
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
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

def validate_merge_quality(prob_df, pct_df, merged_df):
    """Validate the quality of the merge operation."""
    original_count = len(prob_df)
    merged_count = len(merged_df)
    
    if merged_count < original_count:
        lost_records = original_count - merged_count
        st.warning(f"⚠️ Lost {lost_records} records during merge ({lost_records/original_count*100:.1f}%)")
    
    # Check for duplicates
    merge_keys = ['Tm', 'Batter', 'Pitcher']
    duplicates = merged_df.duplicated(subset=merge_keys).sum()
    if duplicates > 0:
        st.error(f"🔴 Found {duplicates} duplicate matchups after merge")
    
    return merged_df

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_and_process_data():
    """Enhanced data loading and processing with validation."""
    
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

def calculate_league_aware_scores(df, profile_type='contact'):
    """Enhanced scoring algorithm that adapts based on profile type."""
    
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

def create_league_aware_filters(df=None):
    """Create baseball-intelligent filtering system based on league averages and player types."""
    st.sidebar.header("🎯 Baseball-Smart Filters")
    
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
            'max_k': 17.0,   # Above average contact
            'max_bb': 6.0,   # Above average aggressive
            'min_hit_prob': 35,
            'profile_type': 'contact'
        },
        "⭐ Elite Contact Specialists": {
            'description': "Ultra-low K% (Pure contact)",
            'max_k': 12.0,   # Elite contact
            'max_bb': 8.5,   # League average walks
            'min_hit_prob': 30,
            'profile_type': 'contact'
        },
        "⚡ Swing-Happy Hitters": {
            'description': "Ultra-low BB% (Aggressive approach)",
            'max_k': 22.6,   # League average strikeouts
            'max_bb': 4.0,   # Hyper-aggressive
            'min_hit_prob': 32,
            'profile_type': 'contact'
        },
        "🔷 Above-Average Contact": {
            'description': "Better than league average K%",
            'max_k': 17.0,   # Above average contact
            'max_bb': 10.0,  # Reasonable walks
            'min_hit_prob': 28,
            'profile_type': 'contact'
        },
        "💥 Contact Power Hitters": {
            'description': "Low K% + High XB% & HR% (Power with contact)",
            'max_k': 17.0,   # Good contact
            'max_bb': 10.0,  # Reasonable walks
            'min_xb': 8.0,   # Strong extra base rate
            'min_hr': 3.0,   # Solid HR rate
            'min_vs': -5,    # Reasonable matchup
            'profile_type': 'power'
        },
        "🚀 Pure Power Sluggers": {
            'description': "High XB% & HR% (Power over contact)",
            'max_k': 100,    # Accept high strikeouts
            'max_bb': 100,   # Accept high walks
            'min_xb': 10.0,  # Elite extra base rate
            'min_hr': 4.0,   # Elite HR rate
            'min_vs': -10,   # Any matchup if power is elite
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
                    # Show league context comparison for playing players only - FIXED CALCULATION
                    avg_k_filtered = preview_df['adj_K'].mean()
                    avg_bb_filtered = preview_df['adj_BB'].mean()
                    
                    # FIXED: Make positive values = better performance
                    k_improvement = LEAGUE_K_AVG - avg_k_filtered  # League - Player (positive = better)
                    bb_improvement = LEAGUE_BB_AVG - avg_bb_filtered  # League - Player (positive = more aggressive)
                    
                    result_count = filters.get('result_count', 15)
                    display_count = matching_count if result_count == "All" else min(matching_count, result_count)
                    
                    st.sidebar.markdown(f"**📊 vs League Avg (Playing Players):**")
                    st.sidebar.markdown(f"K%: {k_improvement:+.1f}% better than league" if k_improvement > 0 else f"K%: {k_improvement:+.1f}% worse than league")
                    st.sidebar.markdown(f"BB%: {bb_improvement:+.1f}% more aggressive than league" if bb_improvement > 0 else f"BB%: {bb_improvement:+.1f}% less aggressive than league")
                    
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

def create_league_aware_header():
    """Create an enhanced header with league-aware focus."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    
    with col2:
        st.title("🎯 MLB League-Aware Hit Predictor Pro")
        st.markdown("*Find hitters with the best base hit probability using 2024 league context*")

def create_data_quality_dashboard(df):
    """Display data quality metrics in an attractive dashboard."""
    if df is None or df.empty:
        st.error("No data available for quality analysis")
        return
    
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

def display_league_aware_results(filtered_df, filters):
    """Display results with league-average context and baseball intelligence."""
    
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
        # FIXED: Make positive = better (League - Player)
        k_improvement = LEAGUE_K_AVG - avg_k
        color = "success-card" if k_improvement > 2 else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>⚾ K% vs League</h4>
            <h2>{k_improvement:+.1f}%</h2>
            <small>League: {LEAGUE_K_AVG}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_bb = filtered_df['adj_BB'].mean()
        # FIXED: Make positive = more aggressive (League - Player)
        bb_improvement = LEAGUE_BB_AVG - avg_bb
        color = "success-card" if bb_improvement > 1 else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>🚶 BB% vs League</h4>
            <h2>{bb_improvement:+.1f}%</h2>
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
    
    # Enhanced results table with FIXED league context calculations
    display_columns = {
        'Batter': 'Batter',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'K_vs_League': 'K% Better',
        'BB_vs_League': 'BB% More Aggressive',
        'adj_vs': 'vs Pitcher',
        'Score': 'Score'
    }
    
    # Add FIXED league context columns to filtered_df
    display_df = filtered_df.copy()
    # FIXED: Make positive values = better performance
    display_df['K_vs_League'] = LEAGUE_K_AVG - display_df['adj_K']  # League - Player (positive = better contact)
    display_df['BB_vs_League'] = LEAGUE_BB_AVG - display_df['adj_BB']  # League - Player (positive = more aggressive)
    
    # Add lineup status indicators
    excluded_players = st.session_state.get('excluded_players', [])
    display_df['Lineup_Status'] = display_df['Batter'].apply(
        lambda x: '🏟️' if x not in excluded_players else '❌'
    )
    
    # Add lineup status to display columns
    display_columns_with_status = {'Lineup_Status': 'Status', **display_columns}
    
    styled_df = display_df[display_columns_with_status.keys()].rename(columns=display_columns_with_status)
    
    # Enhanced formatting with FIXED league context
    styled_df = styled_df.style.format({
        'Hit Prob %': "{:.1f}%",
        'Contact %': "{:.1f}%", 
        'XB %': "{:.1f}%",
        'HR %': "{:.1f}%",
        'K% Better': "{:+.1f}%",
        'BB% More Aggressive': "{:+.1f}%",
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
        subset=['K% Better'],
        cmap='RdYlGn',  # Red = worse contact, Green = better contact
        vmin=-8,   # Very bad contact (much worse than league)
        vmax=15    # Elite contact (much better than league)  
    ).background_gradient(
        subset=['BB% More Aggressive'],
        cmap='RdYlGn',  # Red = less aggressive, Green = more aggressive
        vmin=-4,   # Much less aggressive than league
        vmax=8     # Much more aggressive than league
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # FIXED interpretation guide with intuitive color meanings
    st.markdown("""
    <div class="color-legend">
        <strong>📊 FIXED League-Aware Color Guide:</strong><br>
        <strong>Status:</strong> 🏟️ = Confirmed Playing | ❌ = Excluded from Lineups<br>
        <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
        <span style="color: #fdae61;">●</span> Good (50-69) | 
        <span style="color: #d7191c;">●</span> Risky (<50)<br>
        <strong>K% Better:</strong> <span style="color: #1a9641;">●</span> Much Better Contact | 
        <span style="color: #d7191c;">●</span> Worse Contact<br>
        <strong>BB% More Aggressive:</strong> <span style="color: #1a9641;">●</span> More Aggressive | 
        <span style="color: #d7191c;">●</span> Less Aggressive<br>
        <strong>✅ NOW: Positive Numbers = Better Performance!</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Performance insights with FIXED league context - multi-profile analysis
    if len(filtered_df) >= 3:
        st.markdown("### 🔍 **Advanced League Context Analysis**")
        
        # Define profile criteria for analysis (UPDATED with more inclusive thresholds)
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
                    
                    # Key metrics with FIXED calculations
                    k_better = LEAGUE_K_AVG - player['adj_K']  # Positive = better
                    bb_more_aggressive = LEAGUE_BB_AVG - player['adj_BB']  # Positive = more aggressive
                    
                    if criteria["type"] == "power":
                        # Power-focused metrics
                        st.markdown(f"""
                        **XB%:** {player['adj_XB']:.1f}% | **HR%:** {player['adj_HR']:.1f}%  
                        **Power Combo:** {(player['adj_XB'] + player['adj_HR']):.1f}%  
                        **vs Pitcher:** {player['adj_vs']:.0f} | **Score:** {player['Score']:.1f}
                        """)
                    else:
                        # Contact-focused metrics
                        st.markdown(f"""
                        **Hit Prob:** {player['total_hit_prob']:.1f}%  
                        **K% Better:** {k_better:+.1f}%  
                        **BB% More Aggressive:** {bb_more_aggressive:+.1f}%  
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
            insights.append(f"📊 **Profile Diversity**: {total_profiles_available}/4 profiles have viable options")
            
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

def create_enhanced_visualizations(df, filtered_df):
    """Create enhanced visualizations focused on base hit analysis."""
    
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

def main_page():
    """Enhanced main page with league-aware focus."""
    create_league_aware_header()
    
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
    
    # Bottom tips with FIXED explanations
    st.markdown("---")
    st.markdown("""
    ### 💡 **V2.3 UPDATED Strategy Tips**
    - **Positive K% Better**: Player strikes out less than league average (GOOD!)
    - **Positive BB% More Aggressive**: Player walks less than league average (more swings = more hit chances)
    - **Scores 70+**: Elite opportunities with league-superior metrics
    - **🔥 NEW: Relaxed Thresholds**: Profiles now catch more "hidden gems" automatically
    - **💎 Less Manual Searching**: Reduced need to dig through All Players for overlooked options
    - **XB% + HR% = Power Combo**: Target 10%+ combined for solid power threats (lowered from 12%)
    - **⚾ All Power Players**: Use for research to find remaining hidden power gems
    - **Always verify lineups and weather before finalizing picks**
    
    **✅ Hidden Gem Revolution: Now Automatically Find Players Just Outside Old Thresholds!**
    """)

def info_page():
    """Comprehensive baseball strategy guide with detailed profile usage."""
    st.title("📚 Complete Baseball Strategy Guide")
    
    with st.expander("🏆 Understanding the League-Aware System", expanded=True):
        st.markdown("""
        ## 🚀 **Why League Context Matters**
        
        Traditional baseball analysis often uses arbitrary thresholds. Our system uses **real 2024 MLB data** to identify truly exceptional performers relative to their peers.
        
        ### **📊 2024 MLB Reality Check**
        | Metric | Elite (Top 10%) | Above Avg (Top 30%) | League Avg | Below Avg (Bottom 30%) |
        |--------|-----------------|---------------------|------------|------------------------|
        | **K%** | ≤12.0% | 12.0-17.0% | ~22.6% | ≥25.0% |
        | **BB%** | ≤4.0% | 4.0-6.0% | ~8.5% | ≥10.0% |
        | **BABIP** | ≥.320 | .300-.320 | ~.290 | ≤.280 |
        
        ### **✅ FIXED Display Logic**
        **Now the numbers make intuitive sense:**
        - **K% Better: +5.2%** = Player strikes out 5.2% less than league average (GOOD!)
        - **BB% More Aggressive: +3.1%** = Player walks 3.1% less than league average (more swings!)
        - **Positive values = Better performance** (no more confusing negative numbers)
        
        ### **✅ V2.3 Profile Updates: Real-World Performance Based**
        **Why we updated the criteria:**
        - **Hidden Gems Discovery**: Players just outside old thresholds were performing well in real games
        - **Reduced Manual Searching**: Less need to manually scan "All Players" for overlooked options
        - **Broader Safety Nets**: Slightly wider criteria catch more viable players without sacrificing quality
        - **Real-World Validation**: Thresholds adjusted based on actual player performance data
        
        **Key Changes:**
        - **Contact profiles**: 1-2% more flexible on K% and BB% thresholds
        - **Power profiles**: 0.5-1% more accessible power requirements
        - **Hit probability minimums**: Slightly lowered to be more inclusive
        - **Matchup requirements**: Slightly more lenient to catch good players with tough matchups
        
        ### **💡 Key Insight: Contact vs Aggression**
        - **Low K% = Better Contact** (fewer strikeouts = more balls in play)
        - **Low BB% = More Aggressive** (fewer walks = more swinging, quicker at-bats)
        - **Sweet Spot**: Low K% + Low BB% = Contact hitters who attack the zone
        - **Low K% = Better Contact** (fewer strikeouts = more balls in play)
        - **Low BB% = More Aggressive** (fewer walks = more swinging, quicker at-bats)
        - **Sweet Spot**: Low K% + Low BB% = Contact hitters who attack the zone
        """)
    
    with st.expander("🎯 Complete Hitter Profile Guide", expanded=False):
        st.markdown("""
        ## **When to Use Each Profile: Detailed Breakdown**
        
        ### **🏆 Contact-Aggressive Hitters** ⭐ RECOMMENDED DEFAULT
        **Profile**: K% ≤19%, BB% ≤7% | **Historical Examples**: Luis Arraez, José Altuve (prime), Tony Gwynn (all-time)
        
        #### **✅ UPDATED: More Inclusive Criteria**
        - **Expanded K% threshold** from 17% to 19% - captures more aggressive contact hitters
        - **Expanded BB% threshold** from 6% to 7% - includes players who are still aggressive but not ultra-selective
        - **Real-world benefit**: Catches players who perform well but were just outside old thresholds
        
        #### **✅ Use This Profile When:**
        - **Daily Fantasy Sports** - Maximum safety and consistency
        - **Cash Games** - You need reliable base hits
        - **Pitcher-Heavy Slates** - Tough pitching matchups across the board
        - **Bad Weather** - Wind/rain favors contact over power
        - **Small Slates** - Limited player pool, need reliable options
        - **New to the Tool** - Best balance of safety and opportunity
        
        #### **📈 Why It Works:**
        - Above-average contact skills (better than 70% of MLB)
        - Aggressive approach leads to more swings = more hit opportunities  
        - Avoids the "walk trap" (walks don't count as hits)
        - Typically 10-15 players per slate (good sample size)
        
        #### **⚠️ Avoid When:**
        - Offensive explosion expected (too conservative)
        - Tournament play where you need ceiling (limits upside)
        
        ---
        
        ### **⚾ All Power Players** 📊 POWER RESEARCH MODE
        **Profile**: No restrictions on K%, BB%, XB%, or HR% | **Use**: Complete power landscape analysis
        
        #### **✅ Use This Profile When:**
        - **Initial Power Research** - See the full power spectrum
        - **Looking for Hidden Power** - Find overlooked players with decent power metrics
        - **Contrarian GPP Strategy** - Players others might miss due to high K% or low traditional power
        - **Large Field Tournaments** - Need maximum differentiation in power plays
        - **Salary Cap Constraints** - Find cheaper power options that don't meet elite thresholds
        
        #### **📈 Why It's Valuable:**
        - Uses power scoring algorithm on entire player pool
        - Reveals players with moderate power who score well due to good matchups
        - Shows complete power hierarchy for the slate
        - Can find value plays overlooked by restrictive power filters
        
        #### **⚠️ Remember:**
        - Still uses power scoring (emphasizes XB%, HR%, reduces singles weight)
        - May include players with very low power numbers
        - Requires more manual filtering of results
        - Best used for research, not final selections
        
        ---
        
        ### **💥 Contact Power Hitters** ⚡ TOURNAMENT CEILING
        **Profile**: K% ≤20%, XB% ≥7%, HR% ≥2.5% | **Historical Examples**: Mookie Betts, Freddie Freeman, Vladimir Guerrero Jr.
        
        #### **✅ UPDATED: More Inclusive Power Criteria**
        - **Expanded K% threshold** from 17% to 20% - more contact power options
        - **Lowered XB% requirement** from 8% to 7% - catches emerging power players
        - **Lowered HR% requirement** from 3% to 2.5% - includes developing power hitters
        - **Real-world benefit**: Finds players with legitimate power potential just below old elite thresholds
        
        #### **✅ Use This Profile When:**
        - **Tournament Play** - Need ceiling without sacrificing too much floor
        - **Hitter-Friendly Parks** - Coors, Yankee Stadium, etc.
        - **Weak Pitching** - Taking advantage of poor opposing pitching
        - **Medium-Stakes Contests** - Balance of safety and upside
        - **Good Weather Conditions** - Wind helping, warm temperatures
        
        #### **📈 Why It Works:**
        - Combines contact skills with legitimate power
        - Lower strikeout risk than pure sluggers
        - Can provide 2+ total bases consistently
        - Multiple ways to return value (singles, doubles, HRs)
        
        #### **⚠️ Considerations:**
        - Higher salary cost than contact-only players
        - Weather-dependent (cold/wind affects power more)
        
        ---
        
        ### **🚀 Pure Power Sluggers** 💥 GPP LEVERAGE
        **Profile**: XB% ≥9%, HR% ≥3.5% (K% and BB% flexible) | **Historical Examples**: Aaron Judge, Mike Trout, Pete Alonso
        
        #### **✅ UPDATED: More Accessible Power Thresholds**
        - **Lowered XB% requirement** from 10% to 9% - catches more power players
        - **Lowered HR% requirement** from 4% to 3.5% - includes near-elite power hitters
        - **Real-world benefit**: Players just below old elite thresholds often still provide excellent power production
        
        #### **✅ Use This Profile When:**
        - **GPP Tournaments** - Maximum ceiling plays for large field events
        - **Offensive Environments** - Coors Field, Camden Yards, favorable weather
        - **Contrarian Strategy** - When others focus on contact in tough conditions  
        - **Late Slate Pivots** - Quick recognition of power spots
        - **Stacking Strategy** - Building around explosive offensive potential
        
        #### **📈 Why It's Valuable:**
        - Highest ceiling potential (3+ total base upside)
        - Lower ownership when conditions appear tough
        - Game-changing potential in tournaments
        - Can overcome poor matchups with raw power
        
        #### **⚠️ High Risk:**
        - High strikeout rates acceptable but risky
        - Weather/park dependent
        - Boom or bust nature unsuitable for cash games
        
        ---
        
        ### **⭐ Elite Contact Specialists** 🎯 PREMIUM PLAYS
        **Profile**: K% ≤12%, BB% ≤8.5% | **Historical Examples**: Luis Arraez, Steven Kwan, Juan Soto (contact years)
        
        #### **✅ Use This Profile When:**
        - **Tournament Finals** - You need the absolute best
        - **High-Stakes Contests** - Maximum confidence required
        - **Ace Pitcher Slates** - Only elite contact can succeed
        - **You Have Specific Intel** - Inside info on a particular matchup
        - **Playoff Baseball** - Pressure situations favor elite skills
        - **Small Field Tournaments** - Need every edge possible
        
        #### **📈 Why It's Elite:**
        - Top 10% contact skills in all of baseball
        - Proven ability to hit elite pitching
        - Rarely have 0-hit games
        - Multiple bonus scoring opportunities
        
        #### **⚠️ Limitations:**
        - Very limited player pool (3-8 players typically)
        - May miss out on power upside
        - Expensive in salary cap formats
        
        ---
        
        ### **⚡ Swing-Happy Hitters** 🔥 CONTRARIAN PLAYS  
        **Profile**: K% ≤22.6%, BB% ≤4% | **Historical Examples**: Bo Bichette, Tim Anderson, Vladimir Guerrero Sr.
        
        #### **✅ Use This Profile When:**
        - **GPP Tournaments** - Looking for contrarian leverage
        - **Offensive Slates** - Lots of runs expected
        - **Fast Pace Games** - More at-bats available
        - **Specific Matchups** - You know pitcher struggles with aggressive hitters
        - **Late Swap Strategy** - Quick decisions needed
        - **Stacking Strategy** - Building around team offense
        
        #### **📈 Why It's Valuable:**
        - Ultra-aggressive = more swings per at-bat
        - Lower ownership in tournaments (contrarian edge)
        - Quick at-bats = more plate appearances possible
        - Can catch fire in offensive environments
        
        #### **⚠️ High Risk:**
        - Strikeout risk near league average
        - Boom/bust potential higher
        - Less reliable for cash games
        
        ---
        
        ### **🔷 Above-Average Contact** 🛡️ BALANCED APPROACH
        **Profile**: K% ≤17%, BB% ≤10% | **Examples**: Most solid everyday players
        
        #### **✅ Use This Profile When:**
        - **Mixed Contests** - Balance of safety and upside
        - **Learning the Tool** - Good middle ground
        - **Uncertain Weather** - When conditions are unclear
        - **Pitcher Mix Slates** - Some good, some bad pitching
        - **Building Multiple Lineups** - Need variety in approach
        
        #### **📈 Benefits:**
        - Larger player pool (15-25+ options)
        - Good balance of contact and patience
        - Suitable for most game types
        - Less volatile than extreme profiles
        """)
    
    st.markdown("---")
    st.markdown("""
    **🔥 Complete Strategy System Features:**
    - ✅ **FIXED intuitive metrics** - Positive numbers = better performance
    - ✅ **NEW: Complete Power System** - 3 power profiles for every power need
    - 8 distinct player profiles including complete power spectrum
    - Profile-specific scoring (contact vs power algorithms)
    - Environmental and matchup-based selection guides
    - Real-time league context comparisons
    - Multi-profile analysis system with all power players
    - Lineup management with session persistence
    
    ### **🎯 Profile Selection Quick Guide:**
    **Cash Games:** 80% Contact-Aggressive, 15% Elite Contact, 5% Contact Power
    **GPP Tournaments:** 40% Contact-Aggressive, 20% Contact Power, 20% Pure Power, 20% Other
    **Hitter-Friendly Parks:** Increase Power profiles by 15-20%
    **Pitcher's Parks:** Stick to Elite Contact and Contact-Aggressive
    **Power Research:** Use "All Power Players" to see complete power landscape
    
    ### **⚾ Power Profile Usage:**
    - **Contact Power**: Tournament ceiling with safety net
    - **Pure Power**: GPP leverage and contrarian plays  
    - **All Power**: Research mode to find hidden power gems
    
    *Master the Art of Baseball Analytics | A1FADED V2.3 Hidden Gems Edition*
    """)

def main():
    """Enhanced main function with league-aware navigation."""
    st.sidebar.title("🏟️ Navigation")
    
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
    st.sidebar.markdown("**V2.3** | Hidden Gems Edition")

if __name__ == "__main__":
    main()
