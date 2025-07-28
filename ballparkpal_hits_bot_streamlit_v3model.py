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
            
            # Light data standardization (without aggressive type conversion that caused issues)
            # Only convert obvious numeric columns, let pandas handle the rest naturally
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
    """Enhanced data loading and processing with validation (RESTORED with fixed clipping)."""
    
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
            
            # FIXED: Smart clipping based on metric type
            if metric in ['K', 'BB']:
                # K and BB should be positive percentages
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            elif metric in ['1B', 'XB', 'HR']:  # Probability metrics
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            else:  # Other metrics (vs, RC)
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
                
            st.success(f"✅ Created adj_{metric} using {base_col} and {pct_col}")
        else:
            st.error(f"❌ Missing columns for {metric}: {base_col} or {pct_col}")
            # Create a fallback column with reasonable defaults to prevent KeyError
            if metric in ['K', 'BB']:
                merged_df[f'adj_{metric}'] = 20  # Default reasonable K/BB rate
            else:
                merged_df[f'adj_{metric}'] = 0
    
    # Calculate total base hit probability (key enhancement!)
    merged_df['total_hit_prob'] = merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
    merged_df['total_hit_prob'] = merged_df['total_hit_prob'].clip(upper=100)  # Cap at 100%
    
    return merged_df

def calculate_league_aware_scores(df):
    """Enhanced scoring algorithm that considers league averages and player types."""
    
    # League averages for 2024
    LEAGUE_K_AVG = 22.6
    LEAGUE_BB_AVG = 8.5
    
    # Base weighted score using league-aware weights
    weights = {
        'adj_1B': 2.0,      # Singles (primary base hit)
        'adj_XB': 1.8,      # Extra bases (also base hits)
        'adj_vs': 1.2,      # Matchup performance
        'adj_RC': 0.8,      # Run creation
        'adj_HR': 0.6,      # Home runs (also base hits)
        'adj_K': -2.5,      # Heavy penalty for strikeouts (no hit)
        'adj_BB': -0.6      # Light penalty for walks (not hits but not terrible)
    }
    
    df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
    
    # League-aware bonuses
    # Elite Contact Bonus (much better than league average K%)
    df['elite_contact_bonus'] = np.where(
        df['adj_K'] <= 12.0,  # Elite contact (≤12%)
        10, 0
    )
    
    # Aggressive Contact Bonus (low K% + low BB%)
    df['aggressive_contact_bonus'] = np.where(
        (df['adj_K'] <= 17.0) & (df['adj_BB'] <= 6.0),  # Above avg contact + aggressive
        8, 0
    )
    
    # High Hit Probability Bonus
    df['hit_prob_bonus'] = np.where(
        df['total_hit_prob'] > 40, 5, 0
    )
    
    # Strong Matchup Bonus
    df['matchup_bonus'] = np.where(df['adj_vs'] > 5, 3, 0)
    
    # League Comparison Bonus (better than league average in both K% and BB%)
    df['league_superior_bonus'] = np.where(
        (df['adj_K'] < LEAGUE_K_AVG) & (df['adj_BB'] < LEAGUE_BB_AVG),  # Better than league in both
        6, 0
    )
    
    # Calculate final score
    df['Score'] = (df['base_score'] + 
                   df['elite_contact_bonus'] + 
                   df['aggressive_contact_bonus'] + 
                   df['hit_prob_bonus'] + 
                   df['matchup_bonus'] + 
                   df['league_superior_bonus'])
    
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
            'min_hit_prob': 35
        },
        "⭐ Elite Contact Specialists": {
            'description': "Ultra-low K% (Pure contact)",
            'max_k': 12.0,   # Elite contact
            'max_bb': 8.5,   # League average walks
            'min_hit_prob': 30
        },
        "⚡ Swing-Happy Hitters": {
            'description': "Ultra-low BB% (Aggressive approach)",
            'max_k': 22.6,   # League average strikeouts
            'max_bb': 4.0,   # Hyper-aggressive
            'min_hit_prob': 32
        },
        "🔷 Above-Average Contact": {
            'description': "Better than league average K%",
            'max_k': 17.0,   # Above average contact
            'max_bb': 10.0,  # Reasonable walks
            'min_hit_prob': 28
        },
        "🌐 All Players": {
            'description': "No K% or BB% restrictions",
            'max_k': 100,
            'max_bb': 100,
            'min_hit_prob': 20
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
    filters['min_hit_prob'] = type_config['min_hit_prob']
    
    # Show what this means
    st.sidebar.markdown(f"**📋 {selected_type}**")
    st.sidebar.markdown(f"*{type_config['description']}*")
    st.sidebar.markdown(f"- Max K%: {filters['max_k']:.1f}%")
    st.sidebar.markdown(f"- Max BB%: {filters['max_bb']:.1f}%")
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
            
            preview_query = f"adj_K <= {filters['max_k']:.1f} and adj_BB <= {filters['max_bb']:.1f} and total_hit_prob >= {filters['min_hit_prob']}"
            
            preview_df = preview_df.query(preview_query)
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
                st.sidebar.markdown("**💡 Consider:** Less restrictive profile or custom settings")
            else:
                st.sidebar.success(f"✅ {matching_count} players match profile")
                if excluded_count > 0:
                    st.sidebar.markdown(f"**📊 Lineup Status:** {matching_count} confirmed playing, {excluded_count} excluded")
                
                if matching_count > 0:
                    # Show league context comparison for playing players only
                    avg_k_filtered = preview_df['adj_K'].mean()
                    avg_bb_filtered = preview_df['adj_BB'].mean()
                    
                    k_vs_league = avg_k_filtered - LEAGUE_K_AVG
                    bb_vs_league = avg_bb_filtered - LEAGUE_BB_AVG
                    
                    result_count = filters.get('result_count', 15)
                    display_count = matching_count if result_count == "All" else min(matching_count, result_count)
                    
                    st.sidebar.markdown(f"**📊 vs League Avg (Playing Players):**")
                    st.sidebar.markdown(f"K%: {k_vs_league:+.1f}% vs league")
                    st.sidebar.markdown(f"BB%: {bb_vs_league:+.1f}% vs league")
                    
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
    
    # Display header with dynamic count
    result_count = filters.get('result_count', 15)
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
        k_vs_league = avg_k - LEAGUE_K_AVG
        color = "success-card" if k_vs_league < -5 else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>⚾ Avg K% vs League</h4>
            <h2>{k_vs_league:+.1f}%</h2>
            <small>League: {LEAGUE_K_AVG}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        avg_bb = filtered_df['adj_BB'].mean()
        bb_vs_league = avg_bb - LEAGUE_BB_AVG
        color = "success-card" if bb_vs_league < -2 else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>🚶 Avg BB% vs League</h4>
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
    
    # Enhanced results table with league context and lineup awareness
    display_columns = {
        'Batter': 'Batter',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'adj_K': 'K% vs League',
        'adj_BB': 'BB% vs League',
        'adj_vs': 'vs Pitcher',
        'Score': 'Score'
    }
    
    # Add league context columns to filtered_df
    display_df = filtered_df.copy()
    display_df['K% vs League'] = display_df['adj_K'] - LEAGUE_K_AVG
    display_df['BB% vs League'] = display_df['adj_BB'] - LEAGUE_BB_AVG
    
    # Add lineup status indicators
    excluded_players = st.session_state.get('excluded_players', [])
    display_df['Lineup_Status'] = display_df['Batter'].apply(
        lambda x: '🏟️' if x not in excluded_players else '❌'
    )
    
    # Add lineup status to display columns
    display_columns_with_status = {'Lineup_Status': 'Status', **display_columns}
    
    styled_df = display_df[display_columns_with_status.keys()].rename(columns=display_columns_with_status)
    
    # Enhanced formatting with league context
    styled_df = styled_df.style.format({
        'Hit Prob %': "{:.1f}%",
        'Contact %': "{:.1f}%", 
        'XB %': "{:.1f}%",
        'HR %': "{:.1f}%",
        'K% vs League': "{:+.1f}%",
        'BB% vs League': "{:+.1f}%",
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
        cmap='RdYlGn',  # Green = below league (good), Red = above league (bad)
        vmin=-10,
        vmax=10
    ).background_gradient(
        subset=['BB% vs League'],
        cmap='RdYlGn',  # Green = below league (aggressive), Red = above league (passive)
        vmin=-5,
        vmax=5
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced interpretation guide with league context and lineup status
    st.markdown("""
    <div class="color-legend">
        <strong>📊 League-Aware Color Guide:</strong><br>
        <strong>Status:</strong> 🏟️ = Confirmed Playing | ❌ = Excluded from Lineups<br>
        <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
        <span style="color: #fdae61;">●</span> Good (50-69) | 
        <span style="color: #d7191c;">●</span> Risky (<50)<br>
        <strong>K% vs League:</strong> <span style="color: #1a9641;">●</span> Much Better | 
        <span style="color: #d7191c;">●</span> Much Worse<br>
        <strong>BB% vs League:</strong> <span style="color: #1a9641;">●</span> More Aggressive | 
        <span style="color: #d7191c;">●</span> More Passive
    </div>
    """, unsafe_allow_html=True)
    
    # Performance insights with league context - ENHANCED with multi-profile analysis
    if len(filtered_df) >= 3:
        st.markdown("### 🔍 **Advanced League Context Analysis**")
        
        # Define profile criteria for analysis
        profile_criteria = {
            "🏆 Contact-Aggressive": {"max_k": 17.0, "max_bb": 6.0, "icon": "🏆"},
            "⭐ Elite Contact": {"max_k": 12.0, "max_bb": 8.5, "icon": "⭐"},
            "⚡ Swing-Happy": {"max_k": 22.6, "max_bb": 4.0, "icon": "⚡"},
            "🔷 Above-Average": {"max_k": 17.0, "max_bb": 10.0, "icon": "🔷"}
        }
        
        excluded_players = st.session_state.get('excluded_players', [])
        
        # Find best player for each profile
        profile_analysis = {}
        
        for profile_name, criteria in profile_criteria.items():
            # Filter players that meet this profile's criteria
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
                    
                    # Key metrics
                    k_vs_league = player['adj_K'] - LEAGUE_K_AVG
                    bb_vs_league = player['adj_BB'] - LEAGUE_BB_AVG
                    
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
            
            # Check for elite contact across profiles
            elite_contact_players = [analysis['player']['Batter'] for analysis in profile_analysis.values() 
                                   if analysis['player']['adj_K'] <= 12.0]
            if elite_contact_players:
                insights.append(f"⭐ **Elite Contact Available**: {', '.join(elite_contact_players)}")
            
            # Check for high hit probability players
            high_hit_prob_players = [analysis['player']['Batter'] for analysis in profile_analysis.values() 
                                   if analysis['player']['total_hit_prob'] > 40]
            if high_hit_prob_players:
                insights.append(f"🎯 **40%+ Hit Probability**: {', '.join(high_hit_prob_players)}")
            
            # Show profile diversity
            total_profiles_available = len(profile_analysis)
            insights.append(f"📊 **Profile Diversity**: {total_profiles_available}/4 profiles have viable options")
            
            for insight in insights:
                st.success(insight)
                
            # Strategic recommendations based on available profiles
            st.markdown("**💡 Strategic Recommendations:**")
            
            if "🏆 Contact-Aggressive" in profile_analysis and "⚡ Swing-Happy" in profile_analysis:
                st.info("🎮 **Balanced Strategy**: Both conservative (Contact-Aggressive) and leverage (Swing-Happy) plays available")
            elif "⭐ Elite Contact" in profile_analysis:
                st.info("🎯 **Premium Strategy**: Elite contact player available - ideal for high-stakes situations")
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
    
    # Calculate league-aware scores (FIXED function name)
    df = calculate_league_aware_scores(df)
    
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
    
    # Bottom tips
    st.markdown("---")
    st.markdown("""
    ### 💡 **League-Aware Strategy Tips**
    - **Green K% vs League**: Much better contact than average (prioritize these)
    - **Scores 70+**: Elite opportunities with league-superior metrics
    - **Multiple Bonuses**: Players with 2+ bonuses are premium picks
    - **Always verify lineups and weather before finalizing picks**
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
        
        ### **💡 Key Insight: Contact vs Aggression**
        - **Low K% = Better Contact** (fewer strikeouts = more balls in play)
        - **Low BB% = More Aggressive** (fewer walks = more swinging, quicker at-bats)
        - **Sweet Spot**: Low K% + Low BB% = Contact hitters who attack the zone
        """)
    
    with st.expander("🎯 Complete Hitter Profile Guide", expanded=False):
        st.markdown("""
        ## **When to Use Each Profile: Detailed Breakdown**
        
        ### **🏆 Contact-Aggressive Hitters** ⭐ RECOMMENDED DEFAULT
        **Profile**: K% ≤17%, BB% ≤6% | **Historical Examples**: Luis Arraez, José Altuve (prime), Tony Gwynn (all-time)
        
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
        
        ---
        
        ### **🌐 All Players** 📊 RESEARCH MODE
        **Profile**: No restrictions | **Use**: Analysis and research
        
        #### **✅ Use This Profile When:**
        - **Initial Research** - See the full landscape
        - **Looking for Outliers** - Find unique opportunities
        - **Checking Your Logic** - Validate other profiles
        - **Large Field Tournaments** - Need maximum differentiation
        """)
    
    with st.expander("🎮 Game Theory & Strategy Selection", expanded=False):
        st.markdown("""
        ## **Choosing Profiles Based on Contest Type**
        
        ### **💰 Cash Games Strategy**
        **Goal**: Consistent base hits, minimize risk
        
        **Primary**: 🏆 Contact-Aggressive Hitters (80% of plays)
        **Secondary**: ⭐ Elite Contact Specialists (20% of plays)
        **Avoid**: ⚡ Swing-Happy Hitters (too volatile)
        
        **Why This Works:**
        - Cash games reward consistency over ceiling
        - Contact-Aggressive gives you 10-15 reliable options
        - Elite Contact for when you need absolute best
        
        ---
        
        ### **🏆 Tournament Strategy**
        **Goal**: High ceiling, willing to accept some risk
        
        **Core Approach (60%)**: 🏆 Contact-Aggressive Hitters
        **Leverage Plays (25%)**: ⚡ Swing-Happy Hitters  
        **Elite Spots (15%)**: ⭐ Elite Contact Specialists
        
        **Why This Mix:**
        - Contact-Aggressive as foundation (safe)
        - Swing-Happy for contrarian leverage (differentiation)
        - Elite Contact for absolute premium spots
        
        ---
        
        ### **⚖️ 50/50 & Double-Ups**
        **Goal**: Finish in top 50%, moderate safety
        
        **Primary**: 🏆 Contact-Aggressive Hitters (70%)
        **Secondary**: 🔷 Above-Average Contact (30%)
        
        **Strategy**: Cast wider net while maintaining quality floor
        
        ---
        
        ### **🎯 Head-to-Head**
        **Goal**: Beat one opponent, balanced approach
        
        **Flexible Mix**: All profiles depending on opponent tendencies
        - vs Conservative opponents: Use ⚡ Swing-Happy for leverage
        - vs Aggressive opponents: Use ⭐ Elite Contact for safety
        """)
    
    with st.expander("🌤️ Situational Profile Selection", expanded=False):
        st.markdown("""
        ## **Environmental & Matchup Factors**
        
        ### **⛈️ Weather Considerations**
        
        #### **Wind Blowing In/Cold Weather**
        - **Use**: 🏆 Contact-Aggressive or ⭐ Elite Contact
        - **Avoid**: Power-dependent profiles
        - **Why**: Contact becomes more valuable when power is suppressed
        
        #### **Wind Blowing Out/Hot Weather** 
        - **Use**: ⚡ Swing-Happy or 🔷 Above-Average Contact
        - **Why**: More aggressive swings can benefit from offensive conditions
        
        #### **Rain/Poor Conditions**
        - **Use**: ⭐ Elite Contact Specialists only
        - **Why**: Only the best contact skills succeed in tough conditions
        
        ---
        
        ### **🏟️ Ballpark Factors**
        
        #### **Pitcher-Friendly Parks** (Marlins Park, Tropicana, etc.)
        - **Use**: ⭐ Elite Contact Specialists
        - **Secondary**: 🏆 Contact-Aggressive  
        - **Avoid**: ⚡ Swing-Happy (strikeouts are killers)
        
        #### **Hitter-Friendly Parks** (Coors, Yankees, etc.)
        - **Use**: ⚡ Swing-Happy for leverage
        - **Why**: Aggressive approaches can capitalize on friendly environments
        
        #### **Neutral Parks**
        - **Use**: 🏆 Contact-Aggressive (default approach works)
        
        ---
        
        ### **🥎 Pitching Matchup Analysis**
        
        #### **Ace Pitcher Slates** (Cy Young candidates, sub-3.00 ERA)
        - **Use**: ⭐ Elite Contact Specialists ONLY
        - **Why**: Only elite contact skills can handle top-tier pitching
        - **Target**: Players with -8% or better K% vs League
        
        #### **Mixed Pitching Quality**
        - **Use**: 🏆 Contact-Aggressive (handles variety well)
        - **Why**: Balanced approach works against varied competition
        
        #### **Weak Pitching Slates** (ERA 4.50+, high walk rates)
        - **Use**: ⚡ Swing-Happy for maximum leverage
        - **Secondary**: 🔷 Above-Average Contact
        - **Why**: Aggressive approaches can feast on poor pitching
        
        #### **Rookie/Unknown Pitchers**
        - **Use**: ⚡ Swing-Happy + 🔷 Above-Average Contact
        - **Why**: Aggressive veterans often handle inexperienced pitching well
        """)
    
    with st.expander("📊 Advanced Metrics & Profile Optimization", expanded=False):
        st.markdown("""
        ## **Reading Between the Numbers**
        
        ### **🎯 Key Metrics by Profile**
        
        #### **🏆 Contact-Aggressive: What to Look For**
        - **K% vs League**: -3% to -8% (significantly better)
        - **BB% vs League**: -2% to -4% (moderately aggressive)
        - **Hit Probability**: 35-45% (solid chance)
        - **Ideal Score Range**: 60-80 points
        
        #### **⭐ Elite Contact: Premium Indicators**
        - **K% vs League**: -8% or better (elite tier)
        - **BB% vs League**: -1% to +1% (doesn't matter much)
        - **Hit Probability**: 40%+ (high confidence)
        - **Ideal Score Range**: 75-95 points
        - **Bonus Requirements**: Must have Elite Contact Bonus
        
        #### **⚡ Swing-Happy: Leverage Markers**
        - **K% vs League**: -2% to +2% (near league average acceptable)
        - **BB% vs League**: -4% or better (very aggressive)
        - **Hit Probability**: 30-40% (moderate chance but high volume)
        - **Ideal Score Range**: 45-70 points
        
        ---
        
        ### **🎁 Bonus Combinations to Target**
        
        #### **Premium Combinations (Prioritize These)**
        1. **Elite Contact + League Superior** = 16 bonus points
        2. **Aggressive Contact + Hit Probability** = 13 bonus points  
        3. **Elite Contact + Hit Probability** = 15 bonus points
        
        #### **Solid Combinations**
        - **League Superior + Hit Probability** = 11 bonus points
        - **Aggressive Contact + Matchup** = 11 bonus points
        
        #### **Red Flags (Avoid)**
        - **No bonuses** = Likely poor matchup
        - **Only Matchup bonus** = Weak underlying skills
        
        ---
        
        ### **🔍 Profile Validation Checklist**
        
        #### **Before Selecting Contact-Aggressive:**
        - [ ] 10+ players available in filter?
        - [ ] Average K% vs League better than -2%?
        - [ ] Multiple players with 2+ bonuses?
        
        #### **Before Selecting Elite Contact:**
        - [ ] 5+ players available in filter?
        - [ ] All players have Elite Contact bonus?
        - [ ] Hit probability 38%+ on top options?
        
        #### **Before Selecting Swing-Happy:**
        - [ ] Offensive environment confirmed?
        - [ ] Players have very low BB% (-3% vs league)?
        - [ ] Contrarian edge available (low ownership)?
        """)
    
    with st.expander("⚡ Real-Time Strategy Adjustments", expanded=False):
        st.markdown("""
        ## **Dynamic Profile Selection**
        
        ### **📈 Slate Development Strategy**
        
        #### **Early in Day (Morning)**
        1. Start with **🌐 All Players** - Survey the landscape
        2. Check weather, lineups, and pitching
        3. Narrow to appropriate profile based on conditions
        4. Build initial lineups with chosen profile
        
        #### **Mid-Day Adjustments**
        1. Monitor lineup changes and weather updates
        2. If conditions worsen → Move to **⭐ Elite Contact**
        3. If conditions improve → Consider **⚡ Swing-Happy**
        4. Always maintain **🏆 Contact-Aggressive** as backup
        
        #### **Late Swaps (30min before games)**
        1. **🏆 Contact-Aggressive** only (safest pivots)
        2. Quick substitutions within same profile
        3. Avoid profile switching this late
        
        ---
        
        ### **🎪 Contest-Specific Adaptations**
        
        #### **Large Field GPPs (1000+ entries)**
        - **Primary**: 🏆 Contact-Aggressive (60%)
        - **Leverage**: ⚡ Swing-Happy (30%)
        - **Premium**: ⭐ Elite Contact (10%)
        - **Goal**: Balance safety with differentiation
        
        #### **Small Field GPPs (<100 entries)**
        - **Primary**: ⭐ Elite Contact (70%)
        - **Secondary**: 🏆 Contact-Aggressive (30%)
        - **Goal**: Maximum quality, less differentiation needed
        
        #### **Beginner Contests**
        - **Primary**: 🏆 Contact-Aggressive (90%)
        - **Secondary**: 🔷 Above-Average Contact (10%)
        - **Goal**: Learn tool without high risk
        
        ---
        
        ### **🚨 Emergency Situations**
        
        #### **No Elite Options Available**
        - Fall back to **🔷 Above-Average Contact**
        - Widen search to **🌐 All Players**
        - Focus on matchup and ballpark advantages
        
        #### **Too Many Good Options**
        - Tighten to **⭐ Elite Contact Specialists**
        - Look for multiple bonus combinations
        - Prioritize proven performers in big spots
        
        #### **Slate Looking Chalky**
        - Shift to **⚡ Swing-Happy** for differentiation
        - Target players with good metrics but lower expected ownership
        - Accept higher risk for tournament leverage
        """)
    
    with st.expander("📚 Study Examples & Case Studies", expanded=False):
        st.markdown("""
        ## **Real-World Application Examples**
        
        ### **📖 Case Study 1: Pitcher's Duel Slate**
        **Scenario**: Two aces facing off, low run total (7.5 under)
        
        **Wrong Approach**: Using ⚡ Swing-Happy (high strikeout risk)
        **Correct Approach**: ⭐ Elite Contact Specialists
        
        **Key Metrics to Target**:
        - K% vs League: -8% or better
        - Hit Probability: 35%+ minimum
        - Must have Elite Contact bonus
        
        **Expected Results**: Lower ownership, higher hit rates
        
        ---
        
        ### **📖 Case Study 2: Coors Field Explosion**
        **Scenario**: High run total (11+ runs), wind blowing out
        
        **Wrong Approach**: Being too conservative with ⭐ Elite Contact
        **Correct Approach**: ⚡ Swing-Happy for leverage
        
        **Key Metrics to Target**:
        - BB% vs League: -4% or better (very aggressive)
        - Total Hit Probability: 30%+ (volume matters)
        - Target players others might avoid due to K%
        
        **Expected Results**: Higher variance but massive upside
        
        ---
        
        ### **📖 Case Study 3: Mixed Quality Slate**
        **Scenario**: Some good pitching, some bad, normal conditions
        
        **Optimal Approach**: 🏆 Contact-Aggressive Hitters
        **Why**: Handles variety well, good sample size
        
        **Portfolio Allocation**:
        - 70% Contact-Aggressive
        - 20% Elite Contact (premium spots)
        - 10% Swing-Happy (contrarian)
        
        ---
        
        ### **🎯 Success Patterns to Recognize**
        
        #### **High Success Indicators**
        1. Profile matches environmental conditions
        2. 2+ bonuses on most selected players
        3. Good sample size (8+ viable options)
        4. K% vs League consistently negative
        
        #### **Warning Signs**
        1. Forcing a profile despite conditions
        2. Very limited options (2-3 players)
        3. No players with multiple bonuses
        4. Having to reach for players with positive K% vs League
        """)
    
    with st.expander("🎓 Quick Reference & Cheat Sheets", expanded=False):
        st.markdown("""
        ## **Quick Decision Framework**
        
        ### **⚡ 30-Second Profile Selection**
        
        #### **Ask Yourself:**
        1. **What's the weather?** Bad → Elite Contact | Good → More options
        2. **What's the pitching?** Aces → Elite Contact | Weak → Swing-Happy
        3. **What's the contest?** Cash → Contact-Aggressive | GPP → Mix
        4. **What's your experience?** New → Contact-Aggressive | Advanced → Mix
        
        #### **Default Decision Tree:**
        ```
        Cash Game? → Contact-Aggressive (80%) + Elite Contact (20%)
        ↓
        Tournament? → Contact-Aggressive (60%) + Swing-Happy (25%) + Elite (15%)
        ↓
        Bad Weather/Aces? → Elite Contact Only
        ↓
        Great Conditions? → Add more Swing-Happy
        ```
        
        ---
        
        ### **📊 Profile Comparison at a Glance**
        
        | Profile | Player Pool | Safety | Upside | Best For |
        |---------|-------------|--------|--------|----------|
        | **🏆 Contact-Aggressive** | 10-15 | High | Medium | Default choice |
        | **⭐ Elite Contact** | 3-8 | Highest | Medium | Premium spots |
        | **⚡ Swing-Happy** | 8-20 | Medium | High | Leverage plays |
        | **🔷 Above-Average** | 15-25 | Medium | Medium | Learning/mixed |
        | **🌐 All Players** | 30+ | Low | Highest | Research only |
        
        ---
        
        ### **🚨 Emergency Cheat Sheet**
        
        #### **When Everything Looks Bad:**
        1. Switch to **🌐 All Players**
        2. Sort by Score (highest first)
        3. Look for hidden gems with good vs Pitcher ratings
        4. Focus on players with any bonuses
        
        #### **When You Can't Decide:**
        1. Default to **🏆 Contact-Aggressive**
        2. It works in 80% of situations
        3. Safe choice that rarely fails completely
        
        #### **When You're Behind in Tournament:**
        1. Switch to **⚡ Swing-Happy**
        2. Accept higher risk for differentiation
        3. Look for contrarian plays others avoid
        """)
    
    st.markdown("---")
    st.markdown("""
    **🔥 Complete Strategy System Features:**
    - 5 distinct player profiles for every situation
    - Environmental and matchup-based selection guides
    - Contest-specific strategy recommendations
    - Real-time adjustment frameworks
    - Detailed case studies and examples
    - Quick reference decision trees
    
    *Master the Art of Baseball Analytics | A1FADED V2.0 Complete Guide*
    """)

def main():
    """Enhanced main function with league-aware navigation."""
    st.sidebar.title("🏟️ Navigation")
    
    # Optional music controls (improved)
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
    st.sidebar.markdown("**V2.0** | League-Average Intelligence")

if __name__ == "__main__":
    main()
