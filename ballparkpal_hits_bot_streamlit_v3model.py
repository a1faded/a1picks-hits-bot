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
        
        # Custom K% threshold
        filters['custom_max_k'] = st.slider(
            "Custom Max K% Override",
            min_value=5.0,
            max_value=35.0,
            value=filters['max_k'],
            step=0.5,
            help=f"League avg: {LEAGUE_K_AVG}% | Elite: ≤12.0%"
        )
        
        # Custom BB% threshold
        filters['custom_max_bb'] = st.slider(
            "Custom Max BB% Override",
            min_value=2.0,
            max_value=15.0,
            value=filters['max_bb'],
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
            options=[5, 10, 15, 20, 25, 30],
            index=2
        )
    
    # REAL-TIME FEEDBACK with league context
    if df is not None and not df.empty:
        try:
            preview_query = f"adj_K <= {filters['max_k']:.1f} and adj_BB <= {filters['max_bb']:.1f} and total_hit_prob >= {filters['min_hit_prob']}"
            
            preview_df = df.query(preview_query)
            matching_count = len(preview_df)
            
            # Context-aware feedback
            if matching_count == 0:
                st.sidebar.error("❌ No players match current profile")
                st.sidebar.markdown("**💡 Try:** Different player type or use custom overrides")
            elif matching_count < 5:
                st.sidebar.warning(f"⚠️ Only {matching_count} players match")
                st.sidebar.markdown("**💡 Consider:** Less restrictive profile or custom settings")
            else:
                st.sidebar.success(f"✅ {matching_count} players match profile")
                
                if matching_count > 0:
                    # Show league context comparison
                    avg_k_filtered = preview_df['adj_K'].mean()
                    avg_bb_filtered = preview_df['adj_BB'].mean()
                    
                    k_vs_league = avg_k_filtered - LEAGUE_K_AVG
                    bb_vs_league = avg_bb_filtered - LEAGUE_BB_AVG
                    
                    st.sidebar.markdown(f"**📊 vs League Avg:**")
                    st.sidebar.markdown(f"K%: {k_vs_league:+.1f}% vs league")
                    st.sidebar.markdown(f"BB%: {bb_vs_league:+.1f}% vs league")
                    
        except Exception as e:
            st.sidebar.warning(f"⚠️ Preview unavailable: {str(e)}")
    
    return filters

def apply_league_aware_filters(df, filters):
    """Apply baseball-intelligent filtering based on league averages and player types."""
    
    if df is None or df.empty:
        return df
    
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
        result_df = filtered_df.sort_values('Score', ascending=False).head(result_count)
        
        return result_df
        
    except Exception as e:
        st.error(f"❌ Filter error: {str(e)}")
        # Return top players by score if filtering fails
        result_count = filters.get('result_count', 15)
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
    
    # Enhanced results table with league context
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
    
    styled_df = display_df[display_columns.keys()].rename(columns=display_columns)
    
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
    
    # Enhanced interpretation guide with league context
    st.markdown("""
    <div class="color-legend">
        <strong>📊 League-Aware Color Guide:</strong><br>
        <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
        <span style="color: #fdae61;">●</span> Good (50-69) | 
        <span style="color: #d7191c;">●</span> Risky (<50)<br>
        <strong>K% vs League:</strong> <span style="color: #1a9641;">●</span> Much Better | 
        <span style="color: #d7191c;">●</span> Much Worse<br>
        <strong>BB% vs League:</strong> <span style="color: #1a9641;">●</span> More Aggressive | 
        <span style="color: #d7191c;">●</span> More Passive
    </div>
    """, unsafe_allow_html=True)
    
    # Performance insights with league context
    if len(filtered_df) >= 3:
        st.markdown("### 🔍 **League Context Analysis**")
        
        # Top performer analysis
        top_player = filtered_df.iloc[0]
        k_improvement = LEAGUE_K_AVG - top_player['adj_K']
        bb_improvement = LEAGUE_BB_AVG - top_player['adj_BB']
        
        insights = []
        
        if k_improvement > 5:
            insights.append(f"**{top_player['Batter']}** has elite contact skills ({k_improvement:.1f}% better K% than league)")
        
        if bb_improvement > 2:
            insights.append(f"**{top_player['Batter']}** is aggressive at the plate ({bb_improvement:.1f}% fewer walks than league)")
        
        if top_player['total_hit_prob'] > 40:
            insights.append(f"**{top_player['Batter']}** has excellent hit probability ({top_player['total_hit_prob']:.1f}%)")
        
        for insight in insights[:2]:  # Show top 2 insights
            st.success(insight)

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
    
    # Export functionality
    st.markdown("---")
    col1, col2, col3 = st.columns(3)
    
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
        st.info(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")
    
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
    """Enhanced info page with league-aware filtering focus."""
    st.title("📚 League-Aware Hit Predictor Guide")
    
    with st.expander("🏆 League-Average Based System", expanded=True):
        st.markdown("""
        ## 🚀 **Baseball-Intelligent Filtering**
        
        ### **Why League Averages Matter**
        - **Context-Driven**: Filters based on real 2024 MLB performance
        - **Player Types**: Identifies specific hitting profiles 
        - **Meaningful Thresholds**: Elite/Above Average/League Average tiers
        
        ### **📊 2024 League Benchmarks**
        | Metric | Elite | Above Avg | League Avg | Below Avg |
        |--------|-------|-----------|------------|-----------|
        | **K%** | ≤12% | 12-17% | ~22.6% | ≥25% |
        | **BB%** | ≤4% | 4-6% | ~8.5% | ≥10% |
        
        ### **🎯 Player Type Profiles**
        
        #### **🏆 Contact-Aggressive Hitters** (Recommended)
        - **K% ≤17%** (Above average contact)
        - **BB% ≤6%** (Aggressive approach)
        - **Example**: Luis Arraez, Tim Anderson types
        - **Best for**: Base hit consistency
        
        #### **⭐ Elite Contact Specialists**
        - **K% ≤12%** (Elite contact skills)
        - **BB% ≤8.5%** (League average walks)
        - **Focus**: Pure contact ability
        
        #### **⚡ Swing-Happy Hitters**
        - **K% ≤22.6%** (League average strikeouts)
        - **BB% ≤4%** (Ultra-aggressive)
        - **Example**: Bo Bichette types
        - **Benefit**: Early swing, quick at-bats
        
        ### **🎁 Enhanced Scoring Bonuses**
        - **Elite Contact Bonus** (+10): K% ≤12% (much better than league)
        - **Aggressive Contact Bonus** (+8): K% ≤17% AND BB% ≤6%
        - **League Superior Bonus** (+6): Better than league in both K% and BB%
        - **High Hit Probability** (+5): Total hit chance >40%
        """)
    
    with st.expander("🔍 How to Use League-Aware Filters"):
        st.markdown("""
        ### **Quick Start Guide**
        
        #### **🔰 New User (Recommended)**
        1. Start with **"Contact-Aggressive Hitters"**
        2. Check the results - aim for 10-15 players
        3. Look for players with **green K% vs League** values
        4. Focus on **negative values** (better than league average)
        
        #### **🔧 Advanced Usage**
        - **Elite Contact**: For tournament/high-stakes plays
        - **Swing-Happy**: For aggressive game scripts
        - **Custom Overrides**: Fine-tune thresholds in advanced options
        
        ### **📊 Reading League Context**
        
        #### **K% vs League Column**
        - **Green/Negative**: Better contact than league average
        - **-5.0%**: 5% better strikeout rate than league (excellent)
        - **+3.0%**: 3% worse than league (concerning)
        
        #### **BB% vs League Column**  
        - **Green/Negative**: More aggressive than league average
        - **-3.0%**: 3% fewer walks than league (very aggressive)
        - **+2.0%**: 2% more walks than league (patient approach)
        
        ### **🎯 Interpretation Examples**
        
        | Player Profile | K% vs League | BB% vs League | Assessment |
        |----------------|--------------|---------------|------------|
        | **Elite Contact** | -8.0% | -2.0% | Perfect base hit candidate |
        | **Power Contact** | -3.0% | +1.0% | Good contact, patient approach |
        | **Aggressive** | +1.0% | -4.0% | Swing-first mentality |
        | **Risky** | +5.0% | +3.0% | Below league average in both |
        """)
    
    with st.expander("📈 Advanced Strategy Tips"):
        st.markdown("""
        ### **Game Situation Strategies**
        
        #### **Cash Games (Conservative)**
        - Use **"Contact-Aggressive Hitters"** or **"Elite Contact"**
        - Target players with **K% vs League < -3.0%**
        - Prioritize consistency over upside
        
        #### **Tournaments (Aggressive)**
        - Mix **"Elite Contact"** with some **"Swing-Happy"** 
        - Look for **high Hit Prob %** (40%+) regardless of profile
        - Accept some K% risk for upside
        
        #### **Pitcher-Heavy Slates**
        - Use **"Above-Average Contact"** for more options
        - Lower Hit Probability thresholds
        - Focus heavily on **K% vs League** values
        
        #### **Offense-Heavy Slates**
        - Can afford **"Elite Contact"** strictness
        - Raise Hit Probability thresholds
        - Look for bonus combinations
        
        ### **🔍 Reading the Bonuses**
        Your top players should have multiple bonuses:
        - **Elite Contact** + **League Superior** = Premium play
        - **Aggressive Contact** + **Hit Prob** = Solid choice
        - **Multiple bonuses** = Higher scores and better opportunities
        
        ### **⚠️ Red Flags to Avoid**
        - **K% vs League > +5.0%**: Much worse contact than league
        - **Both K% and BB% above league**: Poor plate discipline
        - **Hit Prob < 25%**: Very low chance of any base hit
        """)
    
    st.markdown("---")
    st.markdown("""
    **🔥 League-Aware Features in V2.0:**
    - Player type profiles based on 2024 MLB data
    - League context comparisons for all metrics  
    - Enhanced scoring with baseball-intelligent bonuses
    - Real-time league vs daily slate analysis
    - Contact vs power emphasis based on approach
    - Contextual suggestions and insights
    
    *Engineered with Real Baseball Analytics | A1FADED V2.0*
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
