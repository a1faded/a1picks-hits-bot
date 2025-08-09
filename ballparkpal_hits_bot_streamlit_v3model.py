import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== STREAMLIT CONFIGURATION ====================
st.set_page_config(
    page_title="MLB Hit Predictor Pro v5.0",
    layout="wide",
    page_icon="⚾",
    initial_sidebar_state="expanded"
)

# ==================== APPLE-INSPIRED CSS ====================
def inject_apple_css():
    """Inject Apple-inspired CSS styling"""
    st.markdown("""
    <style>
    /* Apple-inspired global styles */
    .stApp {
        background: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif;
        color: #1d1d1f;
    }
    
    .main .block-container {
        padding: 1rem;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* Apple header */
    .apple-header {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 2rem;
        margin: 1rem 0;
        border-radius: 16px;
        border: 0.5px solid rgba(0, 0, 0, 0.1);
        text-align: center;
    }
    
    .apple-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        color: #1d1d1f;
    }
    
    .apple-subtitle {
        font-size: 1rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        color: #86868b;
    }
    
    /* Apple cards */
    .apple-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 16px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.1);
        transition: all 0.3s ease;
    }
    
    .apple-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
    }
    
    /* Metric cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        padding: 1.5rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.1);
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
    }
    
    .metric-title {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #86868b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1d1d1f;
        margin: 0.25rem 0;
    }
    
    .metric-subtitle {
        font-size: 0.8rem;
        color: #86868b;
        margin-top: 0.25rem;
    }
    
    /* Success styling */
    .success-card {
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%);
        border-color: rgba(52, 199, 89, 0.2);
    }
    
    /* Apple buttons */
    .stButton button {
        background: #007AFF !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        transition: all 0.2s ease !important;
    }
    
    .stButton button:hover {
        background: #0056CC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.3) !important;
    }
    
    /* Sidebar styling */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        border-radius: 16px !important;
        padding: 1rem !important;
        margin: 1rem 0.5rem !important;
        border: 0.5px solid rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Form elements */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.8) !important;
        border: 0.5px solid rgba(0, 0, 0, 0.1) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
    }
    
    /* Table styling */
    .dataframe {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(20px) !important;
        -webkit-backdrop-filter: blur(20px) !important;
        margin: 1rem 0 !important;
    }
    
    .dataframe thead th {
        background: #1d1d1f !important;
        color: #f5f5f7 !important;
        font-weight: 600 !important;
        padding: 1rem !important;
        border: none !important;
        font-size: 0.8rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-align: center !important;
    }
    
    .dataframe tbody td {
        padding: 0.75rem !important;
        border-bottom: 0.5px solid rgba(0, 0, 0, 0.05) !important;
        font-weight: 500 !important;
        font-size: 0.8rem !important;
        text-align: center !important;
        color: #1d1d1f !important;
        background: rgba(255, 255, 255, 0.6) !important;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 122, 255, 0.05) !important;
    }
    
    /* Typography */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', system-ui, sans-serif !important;
        color: #1d1d1f !important;
        font-weight: 600 !important;
    }
    
    p, div, span {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', system-ui, sans-serif !important;
        color: #1d1d1f !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== CONFIGURATION ====================
class MLBConfig:
    CSV_URLS = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_walks': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
        'pitcher_hrs': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
        'pitcher_hits': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv'
    }
    
    LEAGUE_AVERAGES = {
        'K_PCT': 22.6,
        'BB_PCT': 8.5
    }
    
    PLAYER_PROFILES = {
        'contact_aggressive': {
            'name': '🏆 Contact-Aggressive Hitters',
            'description': 'Low K% + Low BB% (Elite for base hits)',
            'max_k': 19.0,
            'max_bb': 7.0,
            'min_hit_prob': 32,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'elite_contact': {
            'name': '⭐ Elite Contact Specialists',
            'description': 'Ultra-low K% (Pure contact)',
            'max_k': 14.0,
            'max_bb': 9.5,
            'min_hit_prob': 28,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'contact_power': {
            'name': '💥 Contact Power Hitters',
            'description': 'Low K% + High XB% & HR% (Power with contact)',
            'max_k': 20.0,
            'max_bb': 12.0,
            'min_xb': 7.0,
            'min_hr': 2.5,
            'min_vs': -5,
            'profile_type': 'power',
            'weights': {
                'adj_XB': 3.0, 'adj_HR': 2.5, 'adj_vs': 1.5,
                'adj_RC': 1.0, 'adj_1B': 0.5, 'adj_K': -1.0, 'adj_BB': -0.3
            }
        },
        'all_players': {
            'name': '🌐 All Players',
            'description': 'No restrictions',
            'max_k': 100,
            'max_bb': 100,
            'min_hit_prob': 20,
            'profile_type': 'all',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        }
    }

# ==================== SIMPLE COMPONENTS ====================
def create_header():
    """Create Apple-inspired header"""
    st.markdown("""
    <div class="apple-header">
        <h1 class="apple-title">⚾ MLB Hit Predictor</h1>
        <p class="apple-subtitle">Professional Analytics Platform</p>
    </div>
    """, unsafe_allow_html=True)

def create_metric_card(title: str, value: str, subtitle: str = "", card_type: str = "metric"):
    """Create metric card"""
    card_class = "metric-card"
    if card_type == "success":
        card_class += " success-card"
    
    return f"""
    <div class="{card_class}">
        <div class="metric-title">{title}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-subtitle">{subtitle}</div>
    </div>
    """

def create_dashboard_section(title: str):
    """Create dashboard section"""
    return f"""
    <div class="apple-card">
        <h2 style="text-align: center; margin-bottom: 1rem;">{title}</h2>
    </div>
    """

# ==================== DATA LOADING ====================
@st.cache_data(ttl=900)
def load_and_process_data():
    """Load and process MLB data"""
    try:
        # Load main data
        prob_response = requests.get(MLBConfig.CSV_URLS['probabilities'], timeout=10)
        pct_response = requests.get(MLBConfig.CSV_URLS['percent_change'], timeout=10)
        
        if prob_response.status_code != 200 or pct_response.status_code != 200:
            st.error("Failed to load data files")
            return None
        
        prob_df = pd.read_csv(StringIO(prob_response.text))
        pct_df = pd.read_csv(StringIO(pct_response.text))
        
        # Basic merge
        merged_df = pd.merge(
            prob_df, pct_df,
            on=['Tm', 'Batter', 'Pitcher'],
            suffixes=('_prob', '_pct'),
            how='inner'
        )
        
        # Calculate adjusted metrics
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        
        for metric in metrics:
            base_col = f'{metric}.1' if metric in ['K', 'BB'] else f'{metric}_prob'
            pct_col = f'{metric}_pct'
            
            if base_col in merged_df.columns and pct_col in merged_df.columns:
                merged_df[f'adj_{metric}'] = (
                    merged_df[base_col] * (1 + merged_df[pct_col]/100)
                ).clip(lower=0, upper=100 if metric in ['K', 'BB'] else None)
        
        # Calculate derived metrics
        merged_df['power_combo'] = merged_df['adj_XB'] + merged_df['adj_HR']
        merged_df['total_hit_prob'] = (
            merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
        ).clip(upper=100)
        
        # Simple scoring
        merged_df['Score'] = (
            merged_df['adj_1B'] * 2.0 + 
            merged_df['adj_XB'] * 1.8 + 
            merged_df['adj_HR'] * 1.5 + 
            merged_df['adj_vs'] * 1.2 - 
            merged_df['adj_K'] * 1.5
        )
        
        # Normalize scores
        if merged_df['Score'].max() != merged_df['Score'].min():
            merged_df['Score'] = (
                (merged_df['Score'] - merged_df['Score'].min()) / 
                (merged_df['Score'].max() - merged_df['Score'].min()) * 100
            )
        
        return merged_df.round(1)
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ==================== FILTERS ====================
def create_filters(df):
    """Create sidebar filters"""
    
    st.sidebar.markdown("""
    <div class="apple-card">
        <h3 style="margin: 0;">🎯 Filters</h3>
        <p style="color: #86868b; margin: 0.5rem 0 0 0;">Baseball Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    filters = {}
    
    # Profile selection
    profile_options = {v['name']: k for k, v in MLBConfig.PLAYER_PROFILES.items()}
    selected_profile_name = st.sidebar.selectbox(
        "Player Profile",
        options=list(profile_options.keys()),
        index=0
    )
    
    selected_profile_key = profile_options[selected_profile_name]
    profile_config = MLBConfig.PLAYER_PROFILES[selected_profile_key]
    
    filters['profile_key'] = selected_profile_key
    filters['profile_type'] = profile_config['profile_type']
    
    # Show profile info
    st.sidebar.markdown(f"""
    <div class="apple-card">
        <h4 style="margin: 0 0 0.5rem 0;">{selected_profile_name}</h4>
        <p style="color: #86868b; font-size: 0.8rem; margin: 0;">{profile_config['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Simple filters
    with st.sidebar.expander("⚙️ Advanced", expanded=False):
        filters['max_k'] = st.slider("Max K%", 5.0, 35.0, profile_config.get('max_k', 25.0), 0.5)
        filters['max_bb'] = st.slider("Max BB%", 2.0, 15.0, profile_config.get('max_bb', 12.0), 0.5)
        filters['result_count'] = st.selectbox("Results", [10, 15, 20, 25, 30], index=1)
    
    return filters

def apply_filters(df, filters):
    """Apply filters to dataframe"""
    if df is None or df.empty:
        return df
    
    # Apply K% filter
    df = df[df['adj_K'] <= filters['max_k']]
    
    # Apply BB% filter  
    df = df[df['adj_BB'] <= filters['max_bb']]
    
    # Sort by score
    df = df.sort_values('Score', ascending=False)
    
    # Limit results
    result_count = filters['result_count']
    df = df.head(result_count)
    
    return df

# ==================== DISPLAY FUNCTIONS ====================
def display_overview(df):
    """Display data overview"""
    if df is None or df.empty:
        st.error("No data available")
        return
    
    st.markdown(create_dashboard_section("📊 Today's Analytics"), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card(
            "Matchups", 
            str(len(df)),
            "Total available"
        ), unsafe_allow_html=True)
    
    with col2:
        unique_batters = df['Batter'].nunique()
        st.markdown(create_metric_card(
            "Players", 
            str(unique_batters),
            "Active today"
        ), unsafe_allow_html=True)
    
    with col3:
        unique_teams = df['Tm'].nunique()
        st.markdown(create_metric_card(
            "Teams", 
            str(unique_teams),
            "Organizations"
        ), unsafe_allow_html=True)
    
    with col4:
        avg_hit_prob = df['total_hit_prob'].mean()
        st.markdown(create_metric_card(
            "Avg Hit Prob", 
            f"{avg_hit_prob:.1f}%",
            "League average",
            card_type="success"
        ), unsafe_allow_html=True)

def display_results(filtered_df):
    """Display results table"""
    if filtered_df.empty:
        st.warning("No players match your filters")
        return
    
    st.markdown(create_dashboard_section(f"🎯 Top {len(filtered_df)} Players"), unsafe_allow_html=True)
    
    # Prepare display columns
    display_columns = {
        'Batter': 'Player',
        'Tm': 'Team', 
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'adj_K': 'K %',
        'adj_BB': 'BB %',
        'Score': 'Score'
    }
    
    display_df = filtered_df[display_columns.keys()].rename(columns=display_columns)
    
    # Format the dataframe
    styled_df = display_df.style.format({
        'Hit Prob %': "{:.1f}%",
        'Contact %': "{:.1f}%",
        'XB %': "{:.1f}%", 
        'HR %': "{:.1f}%",
        'K %': "{:.1f}%",
        'BB %': "{:.1f}%",
        'Score': "{:.1f}"
    }).background_gradient(
        subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100
    )
    
    st.dataframe(styled_df, use_container_width=True)

def display_charts(df, filtered_df):
    """Display charts"""
    st.markdown(create_dashboard_section("📈 Performance Analytics"), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Score distribution
        chart1 = alt.Chart(df).mark_bar(
            color='#007AFF',
            opacity=0.8
        ).encode(
            alt.X('Score:Q', bin=alt.Bin(maxbins=15), title='Performance Score'),
            alt.Y('count()', title='Number of Players'),
            tooltip=['count()']
        ).properties(
            title='Score Distribution',
            width=350,
            height=300
        )
        st.altair_chart(chart1, use_container_width=True)
    
    with col2:
        # Hit prob vs K%
        chart2 = alt.Chart(filtered_df).mark_circle(
            size=100,
            opacity=0.8
        ).encode(
            alt.X('total_hit_prob:Q', title='Hit Probability %'),
            alt.Y('adj_K:Q', title='Strikeout Rate %'),
            alt.Color('Score:Q', scale=alt.Scale(scheme='viridis')),
            tooltip=['Batter', 'total_hit_prob', 'adj_K', 'Score']
        ).properties(
            title='Hit Probability vs Contact Skills',
            width=350,
            height=300
        )
        st.altair_chart(chart2, use_container_width=True)

# ==================== MAIN APPLICATION ====================
def main():
    """Main application"""
    
    # Inject CSS
    inject_apple_css()
    
    # Create header
    create_header()
    
    # Sidebar
    st.sidebar.markdown("""
    <div class="apple-card">
        <h2 style="margin: 0; text-align: center;">🏟️ Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading MLB data..."):
        df = load_and_process_data()
    
    if df is None:
        st.error("Unable to load data. Please try again.")
        st.stop()
    
    # Display overview
    display_overview(df)
    
    # Create filters
    filters = create_filters(df)
    
    # Apply filters
    filtered_df = apply_filters(df, filters)
    
    # Display results
    if not filtered_df.empty:
        display_results(filtered_df)
        
        # Display charts
        display_charts(df, filtered_df)
        
        # Tools section
        st.markdown(create_dashboard_section("🛠️ Tools"), unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Export Data"):
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    csv,
                    f"mlb_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("🔄 Refresh Data"):
                st.cache_data.clear()
                st.rerun()
        
        with col3:
            st.info("⚡ System: Optimized")
        
        with col4:
            st.success("✅ Data: Loaded")
    
    # Footer
    st.sidebar.markdown("""
    <div class="apple-card">
        <div style="text-align: center;">
            <h4 style="margin: 0;">v5.0 Apple</h4>
            <p style="margin: 0.5rem 0 0 0; color: #86868b; font-size: 0.8rem;">Sleek & Modern</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Run the app
if __name__ == "__main__":
    main()
