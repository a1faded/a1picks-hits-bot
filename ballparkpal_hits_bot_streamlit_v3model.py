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

# Enhanced Data Loading with Error Handling and Validation (RESTORED)
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

# League-aware filtering system functions

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

def calculate_base_hit_scores(df):
    """Enhanced scoring algorithm focused specifically on base hit probability."""
    
    # Base weighted score using optimized weights for base hits
    weights = CONFIG['base_hit_weights']
    
    df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
    
    # Add contextual bonuses for base hit scenarios
    df['contact_bonus'] = np.where(
        (df['total_hit_prob'] > 40) & (df['adj_K'] < 18), 8, 0
    )
    
    df['consistency_bonus'] = np.where(
        (df['adj_1B'] > 20) & (df['adj_XB'] > 8), 5, 0
    )
    
    df['matchup_bonus'] = np.where(df['adj_vs'] > 70, 3, 0)
    
    # Calculate final score
    df['Score'] = df['base_score'] + df['contact_bonus'] + df['consistency_bonus'] + df['matchup_bonus']
    
    # Normalize to 0-100 scale
    if df['Score'].max() != df['Score'].min():
        df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    else:
        df['Score'] = 50  # Default if all scores are identical
    
    return df.round(1)

def create_enhanced_header():
    """Create an enhanced header with key metrics."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    
    with col2:
        st.title("🎯 MLB Base Hit Predictor Pro")
        st.markdown("*Find hitters with the highest probability of getting a base hit*")

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

def create_league_aware_header():
    """Create an enhanced header with league-aware focus."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    
    with col2:
        st.title("🎯 MLB League-Aware Hit Predictor Pro")
        st.markdown("*Find hitters with the best base hit probability using 2024 league context*")

# League-aware filtering system (percentile calculation removed as not needed)

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

def apply_smart_filters(df, filters):
    """Apply intelligent filtering logic optimized for base hits."""
    
    if df is None or df.empty:
        return df
    
    query_parts = []
    
    # Validate filter values and add to query
    if 'min_hit_prob' in filters and not np.isnan(filters['min_hit_prob']):
        query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']:.1f}")
    
    if 'max_k' in filters and not np.isnan(filters['max_k']):
        query_parts.append(f"adj_K <= {filters['max_k']:.1f}")
    
    # Advanced filters with validation
    if 'min_vs_pitcher' in filters and not np.isnan(filters['min_vs_pitcher']):
        query_parts.append(f"adj_vs >= {filters['min_vs_pitcher']}")
    
    if 'max_walk' in filters and not np.isnan(filters['max_walk']):
        query_parts.append(f"adj_BB <= {filters['max_walk']}")
    
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

def display_smart_results(filtered_df, filters):
    """Display results with intelligent insights and feedback."""
    
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters")
        
        # Smart suggestions
        st.markdown("""
        ### 💡 **Suggested Adjustments:**
        - Try **"Top 40% Hit Probability"** instead of higher tiers
        - Increase **Strikeout Risk Tolerance** to "Bottom 50%"  
        - Try **unchecking "Minimize Walks"** for more options
        """)
        return
    
    st.subheader(f"🎯 Top {len(filtered_df)} Base Hit Candidates")
    
    # Enhanced key insights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        league_avg = 32.5  # Typical MLB average
        color = "success-card" if best_hit_prob > league_avg else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>🥇 Best Hit Probability</h4>
            <h2>{best_hit_prob:.1f}%</h2>
            <small>League Avg: {league_avg}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = filtered_df['Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Average Score</h4>
            <h2>{avg_score:.1f}</h2>
            <small>0-100 Scale</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        elite_count = (filtered_df['Score'] >= 70).sum()
        st.markdown(f"""
        <div class="{'success-card' if elite_count > 0 else 'metric-card'}">
            <h4>⭐ Elite Plays</h4>
            <h2>{elite_count}/{len(filtered_df)}</h2>
            <small>Score ≥70</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        low_k_count = (filtered_df['adj_K'] <= 15).sum()
        st.markdown(f"""
        <div class="success-card">
            <h4>🛡️ Low K Risk</h4>
            <h2>{low_k_count}/{len(filtered_df)}</h2>
            <small>≤15% Strikeout</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Show filter summary (simplified)
    st.markdown(f"""
    **🎯 Applied Filters:** {filters['hit_prob_percentile']} • {filters['k_risk_percentile']}
    """)
    
    # Enhanced results table with better column selection
    display_columns = {
        'Batter': 'Batter',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Total Hit %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'adj_vs': 'vs Pitcher',
        'adj_K': 'K Risk %',
        'adj_BB': 'BB Risk %',
        'Score': 'Score'
    }
    
    styled_df = filtered_df[display_columns.keys()].rename(columns=display_columns)
    
    # Enhanced formatting with multiple gradients
    styled_df = styled_df.style.format({
        'Total Hit %': "{:.1f}%",
        'Contact %': "{:.1f}%", 
        'XB %': "{:.1f}%",
        'HR %': "{:.1f}%",
        'vs Pitcher': "{:.0f}",
        'K Risk %': "{:.1f}%", 
        'BB Risk %': "{:.1f}%",
        'Score': "{:.1f}"
    }).background_gradient(
        subset=['Score'],
        cmap='RdYlGn',
        vmin=0,
        vmax=100
    ).background_gradient(
        subset=['Total Hit %'],
        cmap='Greens',
        vmin=20,
        vmax=50
    ).background_gradient(
        subset=['K Risk %'],
        cmap='RdYlGn_r',  # Reversed so red=high risk
        vmin=10,
        vmax=30
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced interpretation guide
    st.markdown("""
    <div class="color-legend">
        <strong>📊 Smart Color Guide:</strong><br>
        <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
        <span style="color: #fdae61;">●</span> Good (50-69) | 
        <span style="color: #d7191c;">●</span> Risky (<50)<br>
        <strong>Total Hit %:</strong> <span style="color: #00441b;">●</span> Excellent chance | 
        <span style="color: #f7fcf5;">●</span> Lower chance<br>
        <strong>K Risk %:</strong> <span style="color: #1a9641;">●</span> Safe | 
        <span style="color: #d7191c;">●</span> Dangerous
    </div>
    """, unsafe_allow_html=True)
    
    # Performance insights
    if len(filtered_df) >= 5:
        st.markdown("### 🔍 **Quick Insights**")
        
        # Top performer analysis
        top_player = filtered_df.iloc[0]
        top_hit_prob = top_player['total_hit_prob']
        top_k_risk = top_player['adj_K']
        
        insight_text = f"**🏆 Top Pick:** {top_player['Batter']} ({top_player['Tm']}) has {top_hit_prob:.1f}% hit probability with only {top_k_risk:.1f}% strikeout risk"
        
        if top_hit_prob >= 45:
            insight_text += " - **Elite opportunity!**"
        elif top_hit_prob >= 35:
            insight_text += " - **Solid choice**"
        
        st.success(insight_text)

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
    
    # Calculate scores
    df = calculate_base_hit_scores(df)
    
    # Show data quality dashboard
    create_data_quality_dashboard(df)
    
    # Create league-aware filters with baseball intelligence
    filters = create_league_aware_filters(df)
    
    # Calculate league-aware scores
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
    """Enhanced info page with smart filtering focus."""
    st.title("📚 Smart Base Hit Predictor Guide")
    
    with st.expander("🧠 Intelligent Filtering System", expanded=True):
        st.markdown("""
        ## 🚀 **Smart Percentile-Based Filtering**
        
        ### **Why Percentile-Based?**
        - **Adapts Daily**: "Top 25%" adjusts to each day's player pool
        - **Always Relevant**: Works whether it's a pitcher's duel or slugfest
        - **Intuitive**: "Top 25% Hit Probability" is clearer than "≥35%"
        
        ### **🎯 Primary Filters (Always Visible)**
        
        #### **1. Hit Probability Tier** ⭐⭐⭐⭐⭐
        - **Top 10% (Elite)**: Only the absolute best opportunities
        - **Top 25% (Excellent)**: **← Recommended default**
        - **Top 40% (Good)**: Solid options with more choices
        - **Purpose**: Combined 1B + XB + HR probability (captures ALL base hit ability)
        
        #### **2. Strikeout Risk Tolerance** ⭐⭐⭐⭐⭐
        - **Bottom 10% (Safest)**: Ultra-low strikeout risk
        - **Bottom 25% (Safe)**: **← Recommended default**
        - **Bottom 50% (Moderate)**: Balanced approach
        - **Purpose**: Strikeouts prevent ALL hits
        
        ### **⚙️ Advanced Options (When You Need More Control)**
        
        | Filter | Impact | Default | Range | Why It Matters |
        |--------|--------|---------|-------|----------------|
        | **vs Pitcher** | Moderate | 0 | -10 to +10 | Matchup advantage/disadvantage |
        | **Minimize Walks** | Low | ✅ On | ≤10% or ≤20% | Walks aren't hits (simple toggle) |
        | **Team Filter** | Situational | All | - | Focus on specific games |
        
        ### **🎁 Smart Bonuses in Scoring**
        - **Contact Bonus** (+8): Hit Prob >40% AND K Risk <18%
        - **Consistency Bonus** (+5): Strong single + extra base rates
        - **Matchup Bonus** (+3): Excellent vs pitcher rating
        """)
    
    with st.expander("🎯 How to Use Smart Filters"):
        st.markdown("""
        ### **Quick Start Guide**
        
        #### **🔰 New User (Recommended)**
        1. Keep **"Top 25% Hit Probability"**
        2. Keep **"Bottom 25% Strikeout Risk"**
        3. Check results - aim for 10-15 players
        4. Adjust if needed using fine-tuning tips below
        
        #### **🔧 Fine-Tuning**
        - **Too few results?** → Lower hit probability tier OR increase K risk tolerance
        - **Too many results?** → Raise hit probability tier OR decrease K risk tolerance
        - **Want safer plays?** → Top 10% hit probability + Bottom 10% K risk
        - **Need more options?** → Top 40% hit probability + Bottom 50% K risk
        
        #### **📊 Real-Time Feedback**
        - **Green ✅**: Perfect number of matches
        - **Yellow ⚠️**: Few matches - consider expanding criteria  
        - **Red ❌**: No matches - suggestions provided automatically
        
        ### **🎯 Understanding Results**
        
        | Score | Color | Action | Confidence |
        |-------|-------|--------|------------|
        | **80-100** | 🟢 Green | Max bet/lineup | Elite |
        | **70-79** | 🟢 Green | High confidence | Excellent |
        | **60-69** | 🟡 Yellow | Good choice | Solid |
        | **50-59** | 🟡 Yellow | Proceed cautiously | Okay |
        | **<50** | 🔴 Red | Avoid | High risk |
        """)
    
    with st.expander("📈 Advanced Strategy Tips"):
        st.markdown("""
        ### **Daily Adaptation Strategies**
        
        ### **Daily Adaptation Strategies**
        
        #### **High-Offense Days** (lots of good hitters)
        - Use **"Top 10%"** hit probability
        - Can afford **"Bottom 10%"** K risk
        - Tighten advanced filters for selectivity
        
        #### **Pitcher-Heavy Days** (tough slate)
        - Use **"Top 40%"** hit probability  
        - Accept **"Bottom 50%"** K risk
        - Relax advanced filters for more options
        
        #### **Tournament Play**
        - Focus on **Elite (Score 70+)** players only
        - Accept higher variance for upside
        - Use **vs Pitcher +5 to +10** for maximum edge
        
        #### **Cash Games**
        - Target **15+ matching players**
        - Prioritize consistency over upside
        - Keep **vs Pitcher -5 to +5** for broader options
        - Keep K risk in "Bottom 25%"
        
        ### **🎯 vs Pitcher Rating Guide**
        - **+5 to +10**: Batter has significant advantage vs this pitcher type
        - **-2 to +4**: Neutral to slight advantage (good for cash games)
        - **-5 to -3**: Slight disadvantage but acceptable
        - **-10 to -6**: Avoid unless other metrics are elite
        
        ### **🔍 Reading the Data Quality Dashboard**
        - **47+ matchups**: Good daily slate
        - **Avg Hit Prob 33%+**: Above-average offensive day
        - **Top Teams**: Focus on favorable park factors
        """)
    
    st.markdown("---")
    st.markdown("""
    **🔥 Smart Features in V2.0:**
    - Percentile-based filtering that adapts daily
    - Real-time player count and suggestions  
    - Enhanced scoring with smart bonuses
    - League average comparisons
    - Automatic filter recommendations
    - Elite vs safe play identification
    
    *Engineered for Base Hit Success | A1FADED V2.0*
    """)

def main():
    """Enhanced main function with smart navigation."""
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
        ["🎯 Smart Hit Predictor", "📚 Smart Guide"],
        index=0
    )

    if app_mode == "🎯 Smart Hit Predictor":
        main_page()
    else:
        info_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**V2.0** | Smart Percentile Filtering")

if __name__ == "__main__":
    main()
