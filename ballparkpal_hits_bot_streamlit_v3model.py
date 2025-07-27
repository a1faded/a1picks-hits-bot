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
        'adj_1B': 2.0,      # Increased weight for singles (primary base hit)
        'adj_XB': 1.8,      # High weight for extra bases (also base hits)
        'adj_vs': 1.2,      # Matchup performance
        'adj_RC': 0.8,      # Run creation
        'adj_HR': 0.6,      # Home runs (also base hits)
        'adj_K': -2.0,      # Heavy penalty for strikeouts (no hit)
        'adj_BB': -0.8      # Moderate penalty for walks (not hits)
    },
    'expected_columns': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
    'strict_mode_limits': {'max_k': 15, 'max_bb': 10},
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
            
            # Standardize data types
            numeric_cols = ['HR', 'XB', '1B', 'BB', 'K', 'vs', 'RC']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
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
    
    # Load both CSV files
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
    
    # Merge datasets
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
    
    # Calculate adjusted metrics with enhanced base hit focus
    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
    
    for metric in metrics:
        base_col = f'{metric}_prob'
        pct_col = f'{metric}_pct'
        
        if base_col in merged_df.columns and pct_col in merged_df.columns:
            # Apply adjustment formula
            merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col]/100)
            
            # Ensure realistic bounds
            if metric in ['1B', 'XB', 'HR', 'K', 'BB']:  # Probability metrics
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            else:  # Other metrics
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
    
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

def create_enhanced_filters(df=None):
    """Create enhanced filtering options focused on base hits."""
    st.sidebar.header("🎯 Base Hit Filters")
    
    filters = {}
    
    # Core filtering mode
    filters['mode'] = st.sidebar.radio(
        "Filtering Strategy",
        ["🎯 Base Hit Focused", "🔒 Conservative", "⚡ Aggressive"],
        help="Choose your risk tolerance for base hit predictions"
    )
    
    if filters['mode'] == "🎯 Base Hit Focused":
        filters['min_hit_prob'] = st.sidebar.slider("Minimum Total Hit Probability", 25, 60, 35, help="Combined probability of 1B + XB + HR")
        filters['max_k'] = st.sidebar.slider("Maximum Strikeout Risk", 10, 30, 20)
        filters['max_bb'] = st.sidebar.slider("Maximum Walk Risk", 5, 25, 12)
        
    elif filters['mode'] == "🔒 Conservative":
        filters['min_hit_prob'] = st.sidebar.slider("Minimum Total Hit Probability", 30, 70, 40)
        filters['max_k'] = st.sidebar.slider("Maximum Strikeout Risk", 8, 20, 15)
        filters['max_bb'] = st.sidebar.slider("Maximum Walk Risk", 5, 15, 10)
        
    else:  # Aggressive
        filters['min_hit_prob'] = st.sidebar.slider("Minimum Total Hit Probability", 20, 50, 30)
        filters['max_k'] = st.sidebar.slider("Maximum Strikeout Risk", 15, 35, 25)
        filters['max_bb'] = st.sidebar.slider("Maximum Walk Risk", 8, 30, 15)
    
    # Additional filters
    filters['min_1b'] = st.sidebar.slider("Minimum Single Probability", 10, 35, 18)
    filters['min_vs'] = st.sidebar.slider("Minimum vs Pitcher", 0, 100, 50, help="Matchup rating vs opposing pitcher")
    
    # Team filter - now properly populated
    team_options = []
    if df is not None and not df.empty:
        team_options = sorted(df['Tm'].unique().tolist())
    
    filters['selected_teams'] = st.sidebar.multiselect(
        "Filter by Teams (optional)",
        options=team_options
    )
    
    # Number of results
    filters['num_players'] = st.sidebar.selectbox("Number of Top Results", [5, 10, 15, 20, 25], index=2)
    
    return filters

def apply_enhanced_filters(df, filters):
    """Apply enhanced filtering logic for base hit optimization."""
    
    query_parts = []
    
    # Core base hit filters
    query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']}")
    query_parts.append(f"adj_K <= {filters['max_k']}")
    query_parts.append(f"adj_BB <= {filters['max_bb']}")
    query_parts.append(f"adj_1B >= {filters['min_1b']}")
    query_parts.append(f"adj_vs >= {filters['min_vs']}")
    
    # Team filter
    if filters['selected_teams']:
        team_filter = "Tm in " + str(filters['selected_teams'])
        query_parts.append(team_filter)
    
    # Apply filters
    full_query = " and ".join(query_parts)
    
    try:
        filtered_df = df.query(full_query)
        return filtered_df.sort_values('Score', ascending=False).head(filters['num_players'])
    except Exception as e:
        st.error(f"Filter error: {str(e)}")
        return df.head(filters['num_players'])

def display_enhanced_results(filtered_df):
    """Display results with enhanced base hit focus."""
    
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters. Try adjusting the criteria.")
        return
    
    st.subheader(f"🎯 Top {len(filtered_df)} Base Hit Candidates")
    
    # Key insights
    col1, col2, col3 = st.columns(3)
    
    with col1:
        top_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="success-card">
            <h4>🥇 Best Hit Probability</h4>
            <h2>{top_hit_prob:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_score = filtered_df['Score'].mean()
        st.markdown(f"""
        <div class="metric-card">
            <h4>📊 Average Score</h4>
            <h2>{avg_score:.1f}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        low_k_count = (filtered_df['adj_K'] <= 15).sum()
        st.markdown(f"""
        <div class="success-card">
            <h4>🛡️ Low K Risk</h4>
            <h2>{low_k_count}/{len(filtered_df)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced results table
    display_columns = {
        'Batter': 'Batter',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': '1B %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'adj_vs': 'vs Pitcher',
        'adj_K': 'K Risk %',
        'adj_BB': 'BB Risk %',
        'Score': 'Score'
    }
    
    styled_df = filtered_df[display_columns.keys()].rename(columns=display_columns)
    
    # Format the dataframe
    styled_df = styled_df.style.format({
        'Hit Prob %': "{:.1f}%",
        '1B %': "{:.1f}%", 
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
        subset=['Hit Prob %'],
        cmap='Greens',
        vmin=20,
        vmax=60
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced color legend
    st.markdown("""
    <div class="color-legend">
        <strong>📊 Color Guide:</strong><br>
        <strong>Score:</strong> <span style="color: #d7191c;">●</span> Low (0-49) | 
        <span style="color: #fdae61;">●</span> Medium (50-69) | 
        <span style="color: #1a9641;">●</span> High (70-100)<br>
        <strong>Hit Prob:</strong> <span style="color: #f7fcf5;">●</span> Lower chance | 
        <span style="color: #00441b;">●</span> Higher chance
    </div>
    """, unsafe_allow_html=True)

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
    """Enhanced main page with improved base hit focus."""
    create_enhanced_header()
    
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
    
    # Create filters with team options populated
    filters = create_enhanced_filters(df)
    
    # Apply filters
    filtered_df = apply_enhanced_filters(df, filters)
    
    # Display results
    display_enhanced_results(filtered_df)
    
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
                f"mlb_base_hit_predictions_{datetime.now().strftime('%Y%m%d')}.csv",
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
    ### 💡 **Base Hit Strategy Tips**
    - **Scores 70+**: Elite base hit opportunities with low risk
    - **Hit Prob 40%+**: Strong chance of any type of hit
    - **K Risk <15%**: Minimal strikeout danger
    - **Always verify lineups and weather before finalizing picks**
    """)

def info_page():
    """Enhanced info page with base hit focus."""
    st.title("📚 Base Hit Predictor Guide")
    
    with st.expander("🎯 Enhanced Base Hit Algorithm", expanded=True):
        st.markdown("""
        ## 🚀 New & Improved Algorithm
        
        ### **Key Enhancements for Base Hit Prediction**
        
        #### 🎯 **Total Hit Probability**
        - **Formula**: `1B% + XB% + HR%` (capped at 100%)
        - **Purpose**: Comprehensive view of ANY base hit chance
        - **Usage**: Primary filter for identifying hit candidates
        
        #### ⚖️ **Optimized Scoring Weights**
        | Factor | Weight | Focus |
        |--------|--------|-------|
        | **Single %** | 2.0 | ⭐⭐⭐⭐⭐ Most common hit |
        | **Extra Base %** | 1.8 | ⭐⭐⭐⭐⭐ High-value hits |
        | **Strikeout Risk** | -2.0 | 🚨 Heavy penalty (no hit) |
        | **vs Pitcher** | 1.2 | ⭐⭐⭐ Matchup advantage |
        | **Walk Risk** | -0.8 | ⚠️ Not a hit |
        
        #### 🎁 **Smart Bonuses**
        - **Contact Bonus** (+8): Hit Prob >40% AND K Risk <18%
        - **Consistency Bonus** (+5): 1B >20% AND XB >8%
        - **Matchup Bonus** (+3): vs Pitcher >70
        
        #### 🛡️ **Risk Management**
        - Strikeout probability heavily weighted (prevents hits)
        - Walk probability moderately penalized (not a hit)
        - Balanced approach for sustainable hit prediction
        """)
    
    with st.expander("🔍 How to Use the Enhanced Filters"):
        st.markdown("""
        ### **Filter Strategies**
        
        #### 🎯 **Base Hit Focused** (Recommended)
        - Balanced approach for consistent base hits
        - Moderate risk tolerance
        - Good for daily fantasy and betting
        
        #### 🔒 **Conservative**
        - Lower risk, higher confidence
        - Better for smaller bankrolls
        - Focuses on sure things
        
        #### ⚡ **Aggressive**  
        - Higher upside potential
        - More risk tolerance
        - Good for tournaments
        
        ### **Key Filter Explanations**
        - **Total Hit Prob**: Combined chance of ANY base hit
        - **Minimum Single %**: Focus on most common hit type
        - **vs Pitcher**: How well batter performs against this pitcher type
        """)
    
    with st.expander("📊 Understanding the Results"):
        st.markdown("""
        ### **Result Interpretation**
        
        | Score Range | Recommendation | Action |
        |-------------|----------------|---------|
        | **80-100** | 🌟 Elite Play | Max confidence |
        | **70-79** | ⭐ Excellent | High confidence |
        | **60-69** | ✅ Good | Solid choice |
        | **50-59** | ⚠️ Okay | Proceed with caution |
        | **<50** | ❌ Avoid | High risk |
        
        ### **Color Coding**
        - **Green**: Favorable metrics (higher is better)
        - **Red to Yellow**: Risk indicators (lower is better)
        - **Size**: Represents relative importance
        """)
    
    st.markdown("---")
    st.markdown("""
    **🔥 New Features in V2.0:**
    - Enhanced base hit algorithm
    - Total hit probability calculation  
    - Smart filtering strategies
    - Improved data validation
    - Better error handling
    - Export functionality
    - Real-time data quality monitoring
    
    *Made with ❤️ by A1FADED | Focused on Base Hits*
    """)

def main():
    """Enhanced main function with better navigation."""
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
        ["🎯 Base Hit Predictor", "📚 Guide & Tips"],
        index=0
    )

    if app_mode == "🎯 Base Hit Predictor":
        main_page()
    else:
        info_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**V2.0** | Enhanced Base Hit Focus")

if __name__ == "__main__":
    main()
