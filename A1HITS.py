import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor",
    layout="wide",
    page_icon="⚾",
    menu_items={
        'Get Help': 'mailto:your@email.com',
        'Report a bug': "https://github.com/yourrepo/issues",
    }
)

# Constants
CSV_URLS = {
    'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
    'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'
}

# Custom CSS
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
    .expanderHeader {
        font-size: 1.1em !important;
        font-weight: bold !important;
    }
    .color-legend {
        margin: 1rem 0;
        padding: 0.5rem;
        background: #f0f2f6;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# Data Loading and Processing
@st.cache_data(ttl=3600)
def load_and_process_data():
    prob = pd.read_csv(StringIO(requests.get(CSV_URLS['probabilities']).text))
    pct = pd.read_csv(StringIO(requests.get(CSV_URLS['percent_change']).text))
    
    merged = pd.merge(prob, pct, 
                     on=['Tm', 'Batter', 'Pitcher'],
                     suffixes=('_prob', '_pct'))
    
    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
    for metric in metrics:
        base_col = f'{metric}_prob'
        pct_col = f'{metric}_pct'
        merged[f'adj_{metric}'] = merged[base_col] * (1 + merged[pct_col]/100)
        merged[f'adj_{metric}'] = merged[f'adj_{metric}'].clip(lower=0, upper=100)
    
    return merged

def calculate_scores(df):
    weights = {
        'adj_1B': 1.7, 'adj_XB': 1.3, 'adj_vs': 1.1,
        'adj_RC': 0.9, 'adj_HR': 0.5, 'adj_K': -1.4,
        'adj_BB': -1.0   
    }
    
    df['Score'] = sum(df[col]*weight for col, weight in weights.items())
    df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    return df.round(1)

def visualize_results(df):
    chart = alt.Chart(df).mark_bar(
        color='#1f77b4',
        opacity=0.7
    ).encode(
        alt.X('Score:Q', bin=alt.Bin(maxbins=20), title='Prediction Score'),
        alt.Y('count()', title='Number of Players'),
        tooltip=['count()']
    ).properties(
        title='Score Distribution Across Players',
        width=800,
        height=400
    )
    st.altair_chart(chart)

def create_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    with col2:
        st.title("MLB Daily Hit Predictor Pro")

def create_filters():
    st.sidebar.header("Filter Options")
    filters = {
        'strict_mode': st.sidebar.checkbox('Strict Mode', True,
                      help="Limit strikeout risk ≤15% and walk risk ≤10%"),
        'min_1b': st.sidebar.slider("Minimum 1B%", 10, 40, 18),
        'num_players': st.sidebar.selectbox("Number of Players", [5, 10, 15], index=2)
    }
    
    if not filters['strict_mode']:
        filters['max_k'] = st.sidebar.slider("Maximum Strikeout Risk", 15, 40, 25)
        filters['max_bb'] = st.sidebar.slider("Maximum Walk Risk", 10, 30, 15)
    
    return filters

def main_page():
    create_header()
    
    with st.spinner('Crunching matchup data...'):
        df = load_and_process_data()
        df = calculate_scores(df)
    
    filters = create_filters()
    
    query_parts = []
    if filters['strict_mode']:
        query_parts.append("adj_K <= 15 and adj_BB <= 10")
    else:
        query_parts.append(f"adj_K <= {filters.get('max_k', 25)}")
        query_parts.append(f"adj_BB <= {filters.get('max_bb', 15)}")
    
    query_parts.append(f"adj_1B >= {filters['min_1b']}")
    full_query = " and ".join(query_parts)
    
    filtered = df.query(full_query).sort_values('Score', ascending=False).head(filters['num_players'])
    
    st.subheader(f"Top {len(filtered)} Recommended Batters")
    
    show_cols = {
        'Batter': 'Batter',
        'Pitcher': 'Pitcher',
        'adj_1B': '1B%',
        'adj_XB': 'XB%', 
        'adj_vs': 'vs Pitcher%',
        'adj_K': 'K Risk%',
        'adj_BB': 'BB Risk%',
        'Score': 'Score'
    }
    
    styled_df = filtered[show_cols.keys()].rename(columns=show_cols)
    styled_df = styled_df.style.format({
        'Score': "{:.1f}", 
        '1B%': "{:.1f}%", 
        'XB%': "{:.1f}%", 
        'vs Pitcher%': "{:.1f}%", 
        'K Risk%': "{:.1f}%", 
        'BB Risk%': "{:.1f}%"
    }).background_gradient(
        subset=['Score'],
        cmap='RdYlGn',  # Red-Yellow-Green gradient
        vmin=0,
        vmax=100
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Add color legend
    st.markdown("""
    <div class="color-legend">
        <strong>Score Color Guide:</strong><br>
        <span style="display: inline-block; width: 30px; height: 20px; background: #d7191c; margin: 5px 10px 5px 0;"></span>Low (0-49)  
        <span style="display: inline-block; width: 30px; height: 20px; background: #fdae61; margin: 5px 10px 5px 0;"></span>Medium (50-69)  
        <span style="display: inline-block; width: 30px; height: 20px; background: #1a9641; margin: 5px 10px 5px 0;"></span>High (70-100)  
    </div>
    """, unsafe_allow_html=True)
    
    visualize_results(df)
    
    st.markdown("---")
    st.markdown("""
    *Scores above 70 indicate elite matchups, 50-70 good matchups, below 50 marginal matchups.*  
    *Always verify lineups and weather conditions before finalizing picks.*
    """)

def info_page():
    st.title("Guide & FAQ 📚")
    
    with st.expander("📖 Full Guide", expanded=True):
        st.markdown("""
        ## MLB Hit Predictor Pro Guide 🔍⚾

        ### **Overview**
        This tool analyzes 300+ MLB batters daily to identify players with the highest probability 
        of getting a base hit while minimizing strikeout and walk risks.

        ### **How It Works**

        #### Data Sources
        - **Probability Model**: Base chances of outcomes (1B, HR, K, BB)
        - **% Change Model**: Performance vs player's average
        
        #### Scoring System
        We prioritize:
        - 🟢 **High** 1B & Extra Base (XB) probabilities
        - 🔴 **Low** Strikeout (K) & Walk (BB) risks
        - 🟡 Pitcher matchup performance (vs)

        | Factor | Weight     | Impact    |
        |--------|------------|-----------|
        | 1B%    | ★★★★★      | Positive  |
        | XB%    | ★★★★☆      | Positive  |
        | K%     | ★★★☆☆      | Negative  |
        | BB%    | ★★☆☆☆      | Negative  |

        ---

        ### **Using the Tool**
        #### Filters Panel (Left Sidebar)
        - *Strict Mode*: Limits max K% ≤15 and BB% ≤10
        - *Wider Mode*: Allows higher risks for more options
        - Adjust minimum 1B% threshold

        #### Main Results
        - **Score**: 0-100 rating (Higher = Better)
        - **Color Coding**:
          - 🟩 Green = Favorable metrics
          - 🟥 Red = Risk indicators
        - **Tooltips**: Hover over columns for definitions

        #### Visualizations
        - Score distribution shows how players compare
        - Historical trends available via date selector

        ---

        ### **Interpretation Guide**
        | Score Range | Recommendation       |
        |-------------|-----------------------|
        | 70-100      | ⭐⭐⭐⭐⭐ Elite play    |
        | 50-70       | ⭐⭐⭐⭐ Strong option  |
        | 30-50       | ⭐⭐ Situational use   |
        | <30         | ⚠️ High risk         |
        
        ### **Key Features**
        - **Smart Filters** - Customize risk thresholds
        - **Live Updates** - Data refreshes hourly
        - **Matchup Scoring** - 0-100 rating system
        """)
    
    with st.expander("❓ Frequently Asked Questions"):
        st.markdown("""
        ### FAQ
        
        ❓ *How often does data update?*  
        → Every 15 minutes from 9AM ET until first pitch
        
        ❓ *Why different scores for same player?*  
        → Adjusts for ballpark factors and pitcher handedness
        
        ❓ *Can I trust these picks blindly?*  
        → No! Use as one input in your research
        """)
    
    with st.expander("📊 Understanding Metrics"):
        st.markdown("""
        ### Metric Definitions
        
        | Term | Description |
        |------|-------------|
        | **1B%** | Probability of getting a single |
        | **XB%** | Chance of extra-base hit |
        | **K Risk%** | Strikeout probability |
        | **Score** | Overall matchup quality (0-100) |

        #### Adjusted Metrics
        ```python
        Adjusted 1B% = Base 1B% × (1 + % Change/100)
        ```
        *Example*: If a batter normally has 20% 1B chance (+25% today) → **25% actual**
        """)
    
    st.markdown("---")
    st.markdown("""
    *Made with ❤️ by A1FADED*  
    *Data Source: BallparkPal Analytics*  
    *Version 2.0 | Updated: March 2024*
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
