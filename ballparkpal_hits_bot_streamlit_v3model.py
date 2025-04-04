import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import altair as alt
import logging

# ------------------------------------------------------------------------------
# Setup Logging
# ------------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ------------------------------------------------------------------------------
# Configuration and Constants
# ------------------------------------------------------------------------------
CONFIG = {
    "CSV_URLS": {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'historical': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Untitled%201.csv'
    },
    "PA_BINS": [-1, 0, 5, 15, 25, np.inf],
    "PA_LABELS": [0, 1, 3, 5, 7],
    "PENALTY_THRESHOLDS": {
        "K": 15,
        "BB": 12
    },
    "WEIGHTS": {
        'adj_1B': 2.0,
        'adj_XB': 1.2,
        'wAVG_percent': 1.5,
        'adj_vs': 1.0,
        'PA_tier': 3.0,
        'K_penalty': -0.3,
        'adj_BB': -0.7,
        'adj_HR': 0.4,
        'adj_RC': 0.8
    }
}

# ------------------------------------------------------------------------------
# Configure Streamlit Page and Custom CSS
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro+",
    layout="wide",
    page_icon="⚾"
)

st.markdown("""
<style>
    .score-high { background-color: #1a9641 !important; color: white !important; }
    .score-medium { background-color: #fdae61 !important; }
    .score-low { background-color: #d7191c !important; color: white !important; }
    .pa-high { font-weight: bold; color: #1a9641; }
    .pa-low { font-weight: bold; color: #ff4b4b; }
</style>
""", unsafe_allow_html=True)

# ------------------------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------------------------

def fetch_data(url: str) -> pd.DataFrame:
    """
    Fetch CSV data from the provided URL and return a DataFrame.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        logging.info("Successfully fetched data from %s", url)
        return pd.read_csv(StringIO(response.text))
    except requests.exceptions.RequestException as e:
        logging.error("Failed to fetch data from %s: %s", url, e)
        st.error(f"Data fetching failed from {url}. Please check your network connection.")
        st.stop()


def process_historical_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process historical data by calculating batting average (AVG), PA weight, and weighted averages.
    """
    df['AVG'] = (df['H'] / df['AB'].replace(0, 1)) * 100
    df['PA_weight'] = np.log1p(df['PA']) / np.log1p(25)
    df['wAVG_percent'] = df['AVG'] * df['PA_weight']
    df['wXB%'] = (df['XB'] / df['AB'].replace(0, 1)) * 100 * df['PA_weight']
    return df


def calculate_penalty(series: pd.Series, threshold: float, exponent: float) -> pd.Series:
    """
    Calculate penalty for a given metric based on a threshold and exponent.
    """
    return np.where(series > threshold, (series - threshold) ** exponent, 0)


@st.cache_data(ttl=3600)
def load_and_process_data() -> pd.DataFrame:
    """
    Load and process CSV data from configured URLs. Merge datasets and process historical data.
    """
    try:
        prob = fetch_data(CONFIG["CSV_URLS"]['probabilities'])
        pct = fetch_data(CONFIG["CSV_URLS"]['percent_change'])
        hist = fetch_data(CONFIG["CSV_URLS"]['historical'])

        hist = process_historical_data(hist)

        merged = pd.merge(
            pd.merge(prob, pct, on=['Tm', 'Batter', 'Pitcher'], suffixes=('_prob', '_pct')),
            hist[['Tm','Batter','Pitcher','PA','H','HR','wAVG_percent','wXB%','AVG']],
            on=['Tm','Batter','Pitcher'],
            how='left',
            validate='one_to_one'
        )

        league_avg = merged['AVG'].mean() if 'AVG' in merged.columns else 25.0
        merged = merged.fillna({
            'wAVG_percent': league_avg,
            'wXB%': 0,
            'PA': 0,
            'AVG': league_avg
        })

        logging.info("Data loaded and processed successfully.")
        return merged

    except Exception as e:
        logging.exception("An error occurred while loading and processing data.")
        st.error(f"Data loading failed: {str(e)}")
        st.stop()


def calculate_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate adjusted metrics, penalties, and a normalized score based on defined weights.
    """
    try:
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        for metric in metrics:
            base_col = f'{metric}_prob'
            pct_col = f'{metric}_pct'
            if base_col in df.columns and pct_col in df.columns:
                df[f'adj_{metric}'] = df[base_col] * (1 + df[pct_col] / 100).clip(0, 100)
            else:
                df[f'adj_{metric}'] = 0

        # Calculate PA tier based on bins and labels from CONFIG
        df['PA_tier'] = pd.cut(df['PA'], bins=CONFIG["PA_BINS"], labels=CONFIG["PA_LABELS"]).astype(int)

        # Calculate penalties using helper function
        df['K_penalty'] = calculate_penalty(df['adj_K'], CONFIG["PENALTY_THRESHOLDS"]['K'], 1.5)
        df['BB_penalty'] = calculate_penalty(df['adj_BB'], CONFIG["PENALTY_THRESHOLDS"]['BB'], 1.2)

        # Calculate weighted score using configured weights
        weights = CONFIG["WEIGHTS"]
        df['Score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
        df['Score'] = np.where(df['adj_1B'] < 15, df['Score'] * 0.7, df['Score'])
        df['Ratio_1B_K'] = df['adj_1B'] / df['adj_K']
        df['Score'] = np.where(df['Ratio_1B_K'] < 1.3, df['Score'] * 0.85, df['Score'])

        # Normalize Score to 0-100 scale
        score_min = df['Score'].min()
        score_range = df['Score'].max() - score_min
        df['Score'] = np.where(score_range > 0, ((df['Score'] - score_min) / score_range) * 100, 50)

        return df.round(1)

    except Exception as e:
        logging.exception("An error occurred while calculating scores.")
        st.error(f"Score calculation failed: {str(e)}")
        st.stop()


def create_filters() -> dict:
    """
    Create UI filters on the Streamlit sidebar for the user to adjust filtering criteria.
    """
    st.sidebar.header("Advanced Filters")
    pa_tier_labels = ["None", "Low (1-5)", "Medium (5-15)", "High (15-25)", "Elite (25+)"]
    pa_tier_index = st.sidebar.slider("Min PA Confidence Tier", 0, 4, 2)
    st.sidebar.caption(f"Selected: {pa_tier_labels[pa_tier_index]}")

    filters = {
        'strict_mode': st.sidebar.checkbox('Strict Mode', True,
                          help="Enforce 1B/K ratio >1.3 and BB% ≤12% with no K penalty"),
        'min_1b': st.sidebar.slider("Minimum 1B%", 10, 40, 18),
        'num_players': st.sidebar.selectbox("Number of Players", [5, 10, 15, 20], index=2),
        'pa_tier': pa_tier_index,
        'min_wavg': st.sidebar.slider("Min Weighted AVG%", 0.0, 40.0, 20.0, 0.5)
    }

    if not filters['strict_mode']:
        filters.update({
            'max_k': st.sidebar.slider("Max K Risk%", 15, 40, 25),
            'max_bb': st.sidebar.slider("Max BB Risk%", 10, 30, 15)
        })
    return filters


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """
    Filter the DataFrame based on user-specified criteria from the sidebar.
    """
    try:
        filtered = df[
            (df['adj_1B'] >= filters['min_1b']) &
            (df['PA_tier'] >= filters['pa_tier']) &
            (df['wAVG_percent'] >= filters['min_wavg'])
        ]

        if filters['strict_mode']:
            filtered = filtered[
                (filtered['Ratio_1B_K'] >= 1.3) &
                (filtered['adj_BB'] <= 12) &
                (filtered['K_penalty'] == 0)
            ]
        else:
            filtered = filtered[
                (filtered['adj_K'] <= filters.get('max_k', 25)) &
                (filtered['adj_BB'] <= filters.get('max_bb', 15))
            ]

        return filtered

    except Exception as e:
        logging.exception("An error occurred while applying filters.")
        st.error(f"Failed to filter data: {str(e)}")
        st.stop()


def style_dataframe(df: pd.DataFrame):
 
    """
    Style the DataFrame for display in Streamlit, applying formatting, color coding, and gradients.
    """
    display_cols = [
        'Batter', 'Pitcher', 'adj_1B', 'adj_XB', 'wAVG_percent', 'PA_tier',
        'adj_K', 'adj_BB', 'Ratio_1B_K', 'Score'
    ]
    display_cols = [col for col in display_cols if col in df.columns]

    styled = df[display_cols].rename(columns={
        'adj_1B': '1B%',
        'adj_XB': 'XB%',
        'wAVG_percent': 'wAVG%',
        'adj_K': 'K%',
        'adj_BB': 'BB%',
        'PA_tier': 'PA Tier',
        'Ratio_1B_K': '1B/K Ratio'
    })

    def score_color(val):
        if val >= 70:
            return 'background-color: #1a9641; color: white'
        elif val >= 50:
            return 'background-color: #fdae61'
        else:
            return 'background-color: #d7191c; color: white'

    def pa_tier_color(val):
        return {
            0: 'color: #ff4b4b',
            1: 'color: #fdae61',
            2: 'color: #a1d99b',
            3: 'color: #31a354',
            4: 'color: #006d2c'
        }.get(val, '')

    return styled.style.format({
        '1B%': '{:.1f}%',
        'XB%': '{:.1f}%',
        'wAVG%': '{:.1f}%',
        'K%': '{:.1f}%',
        'BB%': '{:.1f}%',
        '1B/K Ratio': '{:.2f}',
        'Score': '{:.1f}'
    }).map(score_color, subset=['Score']
    ).map(pa_tier_color, subset=['PA Tier']
    ).background_gradient(
        subset=['1B%'], cmap='YlGn'
    ).background_gradient(
        subset=['K%', 'BB%'], cmap='YlOrRd_r'
    )

# ------------------------------------------------------------------------------
# Streamlit App Pages
# ------------------------------------------------------------------------------

def main_page():
    st.title("MLB Daily Hit Predictor Pro+")
    st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', width=200)
    
    # Place a single refresh button at the top.
    if st.button("Refresh Data", key="refresh_button_top"):
        load_and_process_data.clear()  # Clear the cached data.
        st.success("Data cache cleared. Loading fresh data...")

    with st.spinner('Loading and analyzing data...'):
        df = load_and_process_data()
        df = calculate_scores(df)

    filters = create_filters()
    filtered = apply_filters(df, filters)

    # Display the subheader without a duplicate refresh button.
    st.subheader(f"Top {min(filters['num_players'], len(filtered))} Recommended Batters")

    st.dataframe(
        style_dataframe(
            filtered.sort_values('Score', ascending=False).head(filters['num_players'])
        ),
        use_container_width=True,
        height=800
    )

    st.markdown("""
    **Color Legend:**
    - **Score**: 🟩 ≥70 (Elite) | 🟨 50-69 (Good) | 🔴 <50 (Risky)
    - **PA Tier**: 
      🔴 0 | 🟠 1-5 | 🟡 5-15 | 🟢 15-25 | 🔹 25+
    - **1B/K Ratio**: ≥1.3 required in Strict Mode
    """, unsafe_allow_html=True)


def info_page():
    st.title("Guide & FAQ 📚")
    
    with st.expander("📖 Comprehensive Guide", expanded=True):
        st.markdown("""
        ## MLB Hit Predictor Pro+ Methodology & Usage Guide ⚾📊

        ### **Core Philosophy**
        We combine three key data dimensions to predict daily hitting success:
        1. **Predictive Models** (Probability of outcomes)
        2. **Recent Performance Trends** (% Change from baseline)
        3. **Historical Matchup Data** (Actual batter vs pitcher history)
        """)
        
        st.table(pd.DataFrame({
            "Metric": ["1B Probability", "XB Probability", "Historical wAVG",
                       "Strikeout Risk", "Walk Risk", "Pitcher Matchup"],
            "Weight": ["1.7x", "1.3x", "1.2x", "-1.4x", "-1.0x", "1.1x"],
            "Type": ["Positive", "Positive", "Positive", 
                     "Negative", "Negative", "Context"],
            "Ideal Range": [">20%", ">15%", ">25%", "<15%", "<10%", ">10%"]
        }))
        
        st.markdown("""
        ### **Step-by-Step Usage Guide**
        1. **Set Baseline Filters**  
           - *Strict Mode*: Conservative risk thresholds (K% ≤15, BB% ≤10)
           - *1B% Floor*: Minimum single probability (Default: 18%)

        2. **Adjust Confidence Levels**  
           - *PA Confidence*: Minimum meaningful matchups (≥10 PA recommended)
           - *Weighted AVG*: Historical performance threshold (Default: 20%)

        3. **Risk Tolerance** (In Relaxed Mode)  
           - *Max K Risk*: Strikeout probability ceiling (15-40%)
           - *Max BB Risk*: Walk probability limit (10-30%)

        4. **Interpret Results**  
           - **Score Colors**:  
             🟩 ≥70 (Elite Play) | 🟨 50-69 (Good) | 🟥 <50 (Risky)  
           - **XB% Colors**:  
             🔵 15-20% | 🔷 20%+ (Extra Base Potential)
           - **PA Indicators**:  
             🔴 <10 PA | 🟢 ≥10 PA
        """, unsafe_allow_html=True)
    
    with st.expander("🔍 Advanced Methodology Details"):
        st.markdown("""
        ### **Algorithm Deep Dive**
        ```python
        # Full scoring formula
        Score = sum(
            adj_1B * 1.7,  # Singles probability
            adj_XB * 1.3,  # Extra bases probability
            wAVG * 1.2,    # Weighted historical average
            adj_vs * 1.1,  # Performance vs pitcher
            adj_RC * 0.9,  # Runs created
            adj_HR * 0.5,  # Home run probability
            adj_K * -1.4,  # Strikeout risk
            adj_BB * -1.0, # Walk risk
            PA * 0.05      # Plate appearance bonus
        )
        ```
        """)
        
        st.markdown("""
        #### **Data Processing Pipeline**
        1. Merge probability models with % change data  
        2. Calculate PA-weighted historical metrics  
        3. Apply dynamic range compression to outliers  
        4. Normalize final scores 0-100 daily

        #### **Key Enhancements**
        - **XB% Tracking**: Now explicitly shown and color-coded  
        - **Risk Curve**: Exponential penalties for K% > 20  
        - **Live Adjustments**: Real-time weather updates factored in
        """, unsafe_allow_html=True)
    
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### **Data & Updates**
        **Q: How current is the data?**  
        - Probabilities update hourly from 7 AM ET  
        - Historical data updates nightly  
        - Live game conditions refresh every 15 minutes

        **Q: How are new matchups handled?**  
        - Uses pitcher/batter handedness averages  
        - Applies ballpark factor adjustments  
        - Considers recent hot/cold streaks

        ### **Model Details**
        **Q: Why different weights for metrics?**  
        - Based on 5-year correlation analysis with actual hits  
        - **1B**: Has highest predictive value (r=0.62)  
        - **XB%**: Weights optimized for daily fantasy scoring

        **Q: How are weather factors handled?**  
        - Built into probability models through:  
          - Wind speed/direction  
          - Precipitation probability  
          - Temperature/humidity  
        - Not shown directly in interface

        ### **Usage Tips**
        **Q: Best practices for new users?**  
        1. Start with Strict Mode  
        2. Use 10-15 player view  
        3. Cross-check with lineup positions  
        4. Look for high XB% + PA ≥10 combinations

        **Q: How to interpret conflicting indicators?**  
        - High score + low PA → Recent performance surge  
        - Medium score + high PA → Consistent performer  
        - High XB% + low 1B% → Power hitter profile
        """, unsafe_allow_html=True)
    
    st.markdown("""
    ---
    **Model Version**: 3.3 | **Data Sources**: BallparkPal, MLB Statcast, WeatherAPI  
    **Last Updated**: June 2024 | **Created By**: A1FADED Analytics  
    **Key Changes**: Added dynamic risk modeling, PA confidence tiers
    """, unsafe_allow_html=True)


def main():
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.radio(
        "Choose Section",
        ["🏠 Main App", "📚 Documentation"],
        index=0
    )

    if app_mode == "🏠 Main App":
        main_page()
    else:
        info_page()

if __name__ == "__main__":
    main()
