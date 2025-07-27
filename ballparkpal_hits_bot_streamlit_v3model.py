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

# Simplified data loading functions (back to working approach)
# Removed: load_csv_with_validation and validate_merge_quality (were causing K% and BB% issues)

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_and_process_data():
    """Simplified data loading using the working approach from the old version."""
    
    try:
        with st.spinner('Loading and processing data...'):
            # Simple data loading (like the old working version)
            prob_df = pd.read_csv(StringIO(requests.get(CONFIG['csv_urls']['probabilities']).text))
            pct_df = pd.read_csv(StringIO(requests.get(CONFIG['csv_urls']['percent_change']).text))
            
            # Basic validation without aggressive type conversion
            if prob_df.empty or pct_df.empty:
                st.error("❌ One or both CSV files are empty")
                return None
            
            # Simple merge (like old version)
            merged_df = pd.merge(
                prob_df, pct_df,
                on=['Tm', 'Batter', 'Pitcher'],
                suffixes=('_prob', '_pct')
            )
            
            if merged_df.empty:
                st.error("❌ No matching records after merge")
                return None
            
            # Calculate adjusted metrics (using old working approach)
            metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
            
            for metric in metrics:
                base_col = f'{metric}_prob'
                pct_col = f'{metric}_pct'
                
                # Simple calculation like the old version
                merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col]/100)
                
                # Handle clipping based on metric type
                if metric in ['K', 'BB']:
                    # K and BB can be negative - only clip upper bound
                    merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(upper=100)
                else:
                    # Other metrics (1B, XB, HR, etc.) should be 0-100
                    merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            
            # Calculate total base hit probability (only positive contributors)
            merged_df['total_hit_prob'] = merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
            merged_df['total_hit_prob'] = merged_df['total_hit_prob'].clip(upper=100)
            
            st.success(f"✅ Successfully processed {len(merged_df)} matchups")
            return merged_df
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading data: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing data: {str(e)}")
        return None

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

def calculate_percentiles(df):
    """Calculate percentiles for smart filtering with NaN handling."""
    if df is None or df.empty:
        return {}
    
    percentiles = {}
    
    # Calculate percentiles for key metrics
    metrics = ['total_hit_prob', 'adj_K', 'adj_1B', 'adj_vs', 'adj_BB']
    
    for metric in metrics:
        if metric in df.columns:
            # Remove NaN values before calculating percentiles
            clean_data = df[metric].dropna()
            if len(clean_data) > 0:
                percentiles[metric] = {
                    'p10': np.percentile(clean_data, 10),
                    'p25': np.percentile(clean_data, 25),
                    'p50': np.percentile(clean_data, 50),
                    'p75': np.percentile(clean_data, 75),
                    'p90': np.percentile(clean_data, 90)
                }
            else:
                # Fallback values if no clean data
                percentiles[metric] = {
                    'p10': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p90': 0
                }
    
    return percentiles

def create_smart_filters(df=None):
    """Create intelligent filtering system optimized for base hits."""
    st.sidebar.header("🎯 Smart Base Hit Filters")
    
    filters = {}
    percentiles = calculate_percentiles(df) if df is not None else {}
    
    # Show data summary
    if df is not None and not df.empty:
        st.sidebar.markdown(f"**📊 Today's Pool:** {len(df)} matchups")
        avg_hit_prob = df['total_hit_prob'].mean() if 'total_hit_prob' in df.columns else 0
        st.sidebar.markdown(f"**🎯 Avg Hit Prob:** {avg_hit_prob:.1f}%")
    
    st.sidebar.markdown("---")
    
    # PRIMARY FILTERS
    st.sidebar.markdown("### **🎯 Primary Filters**")
    
    # Hit Probability Percentile Filter
    hit_prob_options = {
        "Top 10% (Elite)": 90,
        "Top 25% (Excellent)": 75,
        "Top 40% (Good)": 60,
        "Top 60% (Average+)": 40,
        "All Players": 0
    }
    
    filters['hit_prob_percentile'] = st.sidebar.selectbox(
        "Hit Probability Tier",
        options=list(hit_prob_options.keys()),
        index=1,  # Default to Top 25%
        help="Combined probability of getting ANY base hit (1B + XB + HR)"
    )
    
    # Convert to actual threshold with NaN handling
    percentile_val = hit_prob_options[filters['hit_prob_percentile']]
    if percentiles and 'total_hit_prob' in percentiles and percentile_val > 0:
        threshold = np.percentile(df['total_hit_prob'].dropna(), percentile_val)
        filters['min_hit_prob'] = threshold if not np.isnan(threshold) else 25  # Fallback
    else:
        # Fallback defaults
        fallback_values = {90: 45, 75: 38, 60: 32, 40: 28, 0: 0}
        filters['min_hit_prob'] = fallback_values[percentile_val]
    
    # Strikeout Risk Control
    k_risk_options = {
        "Bottom 10% (Safest)": 10,
        "Bottom 25% (Safe)": 25,
        "Bottom 50% (Moderate)": 50,
        "Bottom 75% (Flexible)": 75,
        "All Risk Levels": 100
    }
    
    filters['k_risk_percentile'] = st.sidebar.selectbox(
        "Strikeout Risk Tolerance",
        options=list(k_risk_options.keys()),
        index=1,  # Default to Bottom 25%
        help="Lower strikeout risk = higher chance of making contact"
    )
    
    # Convert to actual threshold with NaN handling
    k_percentile_val = k_risk_options[filters['k_risk_percentile']]
    if percentiles and 'adj_K' in percentiles and k_percentile_val < 100:
        threshold = np.percentile(df['adj_K'].dropna(), k_percentile_val)
        filters['max_k'] = threshold if not np.isnan(threshold) else 20  # Fallback
    else:
        # Fallback defaults
        k_fallback = {10: 12, 25: 16, 50: 20, 75: 25, 100: 100}
        filters['max_k'] = k_fallback[k_percentile_val]
    
    # REMOVED: Minimum Contact Rate slider (redundant with Hit Probability Tier)
    
    # ADVANCED FILTERS (Collapsible)
    with st.sidebar.expander("⚙️ Advanced Options"):
        
        # vs Pitcher Rating (moderate importance as requested)
        filters['min_vs_pitcher'] = st.slider(
            "vs Pitcher Rating",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            help="How well batter performs vs this pitcher type (+10=much better, -10=much worse, moderate importance)"
        )
        
        # Walk Risk Control (simplified)
        minimize_walks = st.checkbox(
            "Minimize Walk Risk",
            value=True,
            help="Checked = ≤10% walk risk (conservative) | Unchecked = ≤20% walk risk (more options)"
        )
        filters['max_walk'] = 10 if minimize_walks else 20
        
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
    
    # REAL-TIME FEEDBACK
    if df is not None and not df.empty:
        # Quick filter preview (simplified without redundant contact filter)
        try:
            preview_parts = []
            if 'min_hit_prob' in filters and not np.isnan(filters['min_hit_prob']):
                preview_parts.append(f"total_hit_prob >= {filters['min_hit_prob']:.1f}")
            if 'max_k' in filters and not np.isnan(filters['max_k']):
                preview_parts.append(f"adj_K <= {filters['max_k']:.1f}")
            
            if preview_parts:
                preview_query = " and ".join(preview_parts)
                preview_df = df.query(preview_query)
                matching_count = len(preview_df)
                
                if matching_count == 0:
                    st.sidebar.error("❌ No players match current filters")
                    st.sidebar.markdown("**💡 Try:** Lower hit probability tier or increase K risk tolerance")
                elif matching_count < 5:
                    st.sidebar.warning(f"⚠️ Only {matching_count} players match")
                    st.sidebar.markdown("**💡 Tip:** Lower hit probability tier for more options")
                else:
                    st.sidebar.success(f"✅ {matching_count} players match your criteria")
                    
                    if matching_count > 0:
                        top_hit_prob = preview_df['total_hit_prob'].max()
                        st.sidebar.markdown(f"**🎯 Best Hit Prob:** {top_hit_prob:.1f}%")
            else:
                # No valid filters
                st.sidebar.info("📊 All players included (no filters applied)")
                        
        except Exception as e:
            st.sidebar.warning(f"⚠️ Filter preview unavailable: {str(e)}")
    
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
    
    # Create smart filters with real-time feedback
    filters = create_smart_filters(df)
    
    # Apply intelligent filters
    filtered_df = apply_smart_filters(df, filters)
    
    # Display intelligent results
    display_smart_results(filtered_df, filters)
    
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
