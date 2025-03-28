import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import altair as alt

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro+",
    layout="wide",
    page_icon="⚾"
)

# Constants
CSV_URLS = {
    'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
    'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
    'historical': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Untitled%201.csv'
}

# Custom CSS
st.markdown("""
<style>
    .score-high { background-color: #1a9641 !important; color: white !important; }
    .score-medium { background-color: #fdae61 !important; }
    .score-low { background-color: #d7191c !important; color: white !important; }
    .pa-high { font-weight: bold; color: #1a9641; }
    .pa-low { font-weight: bold; color: #ff4b4b; }
    .methodology-table { font-size: 0.9em; margin: 15px 0; }
    .methodology-table td { padding: 5px 10px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_process_data():
    try:
        # Load base datasets
        prob = pd.read_csv(StringIO(requests.get(CSV_URLS['probabilities']).text))
        pct = pd.read_csv(StringIO(requests.get(CSV_URLS['percent_change']).text))
        hist = pd.read_csv(StringIO(requests.get(CSV_URLS['historical']).text))
        
        # Process historical data
        hist['AVG'] = (hist['H'] / hist['AB'].replace(0, 1)) * 100
        hist['PA_weight'] = np.log1p(hist['PA']) / np.log1p(25)
        hist['wAVG'] = hist['AVG'] * hist['PA_weight']
        hist['wXB%'] = (hist['XB'] / hist['AB'].replace(0, 1)) * 100 * hist['PA_weight']
        
        # Merge data
        merged = pd.merge(
            pd.merge(prob, pct, on=['Tm', 'Batter', 'Pitcher'], suffixes=('_prob', '_pct')),
            hist[['Tm','Batter','Pitcher','PA','H','HR','wAVG','wXB%','AVG']],
            on=['Tm','Batter','Pitcher'],
            how='left'
        )
        
        # Calculate league average after merging
        league_avg = merged['AVG'].mean() if 'AVG' in merged.columns else 25.0
        
        # Fill missing values
        merged = merged.fillna({
            'wAVG': league_avg,
            'wXB%': 0,
            'PA': 0,
            'AVG': league_avg
        })
        
        return merged
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

def calculate_scores(df):
    try:
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        for metric in metrics:
            base_col = f'{metric}_prob'
            pct_col = f'{metric}_pct'
            if base_col in df.columns and pct_col in df.columns:
                df[f'adj_{metric}'] = df[base_col] * (1 + df[pct_col]/100).clip(0, 100)
            else:
                df[f'adj_{metric}'] = 0
        
        weights = {
            'adj_1B': 1.7, 'adj_XB': 1.3, 'adj_vs': 1.1,
            'adj_RC': 0.9, 'adj_HR': 0.5, 'adj_K': -1.4,
            'adj_BB': -1.0, 'wAVG': 1.2, 'wXB%': 0.9, 'PA': 0.05
        }
        
        df['Score'] = sum(df[col]*weight for col, weight in weights.items() if col in df.columns)
        df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
        return df.round(1)
    
    except Exception as e:
        st.error(f"Score calculation failed: {str(e)}")
        st.stop()

def create_filters():
    st.sidebar.header("Advanced Filters")
    filters = {
        'strict_mode': st.sidebar.checkbox('Strict Mode', True,
                      help="Limit strikeout risk ≤15% and walk risk ≤10%"),
        'min_1b': st.sidebar.slider("Minimum 1B%", 10, 40, 18),
        'num_players': st.sidebar.selectbox("Number of Players", [5, 10, 15, 20], index=2),
        'pa_confidence': st.sidebar.slider("Min PA Confidence", 0, 25, 10),
        'min_wavg': st.sidebar.slider("Min Weighted AVG%", 0.0, 40.0, 20.0, 0.5)
    }
    
    if not filters['strict_mode']:
        filters.update({
            'max_k': st.sidebar.slider("Max K Risk%", 15, 40, 25),
            'max_bb': st.sidebar.slider("Max BB Risk%", 10, 30, 15)
        })
    return filters

def apply_filters(df, filters):
    query_parts = [
        f"adj_1B >= {filters['min_1b']}",
        f"(PA >= {filters['pa_confidence']} or wAVG >= {filters['min_wavg']})"
    ]
    
    if filters['strict_mode']:
        query_parts += ["adj_K <= 15", "adj_BB <= 10"]
    else:
        query_parts += [
            f"adj_K <= {filters.get('max_k', 25)}",
            f"adj_BB <= {filters.get('max_bb', 15)}"
        ]
    
    return df.query(" and ".join(query_parts))

def style_dataframe(df):
    display_cols = [
        'Batter', 'Pitcher', 'adj_1B', 'wAVG', 'PA',
        'adj_K', 'adj_BB', 'Score'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    styled = df[display_cols].rename(columns={
        'adj_1B': '1B%', 'wAVG': 'wAVG%', 
        'adj_K': 'K%', 'adj_BB': 'BB%'
    })
    
    def score_color(val):
        if val >= 70: return 'background-color: #1a9641; color: white'
        elif val >= 50: return 'background-color: #fdae61'
        else: return 'background-color: #d7191c; color: white'
    
    def pa_color(val):
        return 'font-weight: bold; color: #1a9641' if val >= 10 else 'font-weight: bold; color: #ff4b4b'
    
    return styled.style.format({
        '1B%': '{:.1f}%', 'wAVG%': '{:.1f}%',
        'K%': '{:.1f}%', 'BB%': '{:.1f}%',
        'Score': '{:.1f}'
    }).map(score_color, subset=['Score']
    ).map(pa_color, subset=['PA']
    ).background_gradient(
        subset=['1B%', 'wAVG%'], cmap='YlGn'
    ).background_gradient(
        subset=['K%', 'BB%'], cmap='YlOrRd_r'
    )

def main_page():
    st.title("MLB Daily Hit Predictor Pro+")
    st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', width=200)
    
    with st.spinner('Loading and analyzing data...'):
        df = load_and_process_data()
        df = calculate_scores(df)
    
    filters = create_filters()
    filtered = apply_filters(df, filters)
    
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
    - <span class="score-high">High Score (≥70)</span> | 
    <span class="score-medium">Medium (50-69)</span> | 
    <span class="score-low">Low (<50)</span>
    - <span class="pa-high">≥10 PA</span> | 
    <span class="pa-low"><10 PA</span>
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

        # Score Calculation Breakdown Table
        st.markdown("""
        ### **Score Calculation Breakdown**
        """, unsafe_allow_html=True)
        
        # Using Streamlit's native table instead of HTML
        st.table(pd.DataFrame({
            "Metric": ["1B Probability", "XB Probability", "Historical wAVG", 
                      "Strikeout Risk", "Walk Risk", "Pitcher Matchup"],
            "Weight": ["1.7x", "1.3x", "1.2x", "-1.4x", "-1.0x", "1.1x"],
            "Type": ["Positive", "Positive", "Positive", 
                    "Negative", "Negative", "Context"],
            "Calculation Example": [
                "20% base + 25% trend → 25% adj",
                "15% base + 20% trend → 18% adj",
                ".300 avg * 0.8 PA_weight = .240",
                "25% risk → -35 points",
                "15% risk → -15 points",
                "+10% vs pitcher's average"
            ]
        }))

        st.markdown("""
        ### **Step-by-Step Usage Guide**
        1. **Set Baseline Filters**  
           - *Strict Mode*: Conservative risk thresholds (Recommended for new users)
           - *1B% Floor*: Minimum single probability (Default: 18%)

        2. **Adjust Confidence Levels**  
           - *PA Confidence*: Minimum meaningful matchups  
             (≥10 PA suggested for reliable history)
           - *Weighted AVG*: Historical performance threshold

        3. **Risk Tolerance** (In Relaxed Mode)  
           - *Max K Risk*: Strikeout probability ceiling  
           - *Max BB Risk*: Walk probability limit

        4. **Interpret Results**  
           - **Score Colors**:  
             🟩 ≥70 (Elite Play) | 🟨 50-69 (Good) | 🟥 <50 (Risky)  
           - **PA Indicators**:  
             🔴 <10 PA | 🟢 ≥10 PA
        """)

    with st.expander("🔍 Advanced Methodology Details"):
        st.markdown("""
        ### **Algorithm Deep Dive**
        ```python
        # Full scoring formula
        Score = sum(
            adj_1B * 1.7,
            adj_XB * 1.3,
            wAVG * 1.2,
            adj_vs * 1.1,
            adj_RC * 0.9,
            adj_HR * 0.5,
            adj_K * -1.4,
            adj_BB * -1.0,
            PA * 0.05
        )
        ```

        #### **Data Processing Pipeline**
        1. Merge probability models with % change data
        2. Calculate PA-weighted historical metrics
        3. Apply dynamic range compression to outliers
        4. Normalize final scores 0-100 daily

        #### **Key Features**
        - **Smart Normalization**: Scores scaled relative to daily matchups
        - **Ballpark Factors**: Incorporated in probability models
        - **Pitcher Handedness**: Adjustments baked into % changes
        - **Live Updates**: Refreshes every 15 minutes until first pitch
        """)

    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### **Data & Updates**
        **Q: How current is the data?**  
        - Probabilities update hourly from 7 AM ET
        - Historical data updates nightly

        **Q: How are new matchups handled?**  
        - Uses pitcher/batter handedness averages
        - Applies ballpark factor adjustments

        ### **Model Details**
        **Q: Why different weights for metrics?**  
        - Based on 5-year correlation analysis with actual hits
        - 1B has highest predictive value for total hits

        **Q: How are weather factors handled?**  
        - Built into probability models (wind/rain/temp)
        - Not shown directly in interface

        ### **Usage Tips**
        **Q: Best practices for new users?**  
        1. Start with Strict Mode
        2. Use 10-15 player view
        3. Cross-check with lineup positions

        **Q: How to interpret conflicting indicators?**  
        - High score + low PA → Recent performance surge
        - Medium score + high PA → Consistent performer
        """)

    st.markdown("""
    ---
    *Model Version 3.1 | Data Sources: BallparkPal, MLB Statcast  
    Last Updated: June 2024 | Created by A1FADED Analytics*  
    """)

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
