import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro+",
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
    'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
    'historical': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Untitled%201.csv'
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
        padding: 1rem;
        background: #000000;
        border-radius: 8px;
        color: white !important;
    }
    .color-legend strong {
        color: white !important;
    }
    .pa-warning { color: #ff4b4b; font-weight: bold; }
    .pa-strong { color: #1a9641; }
    .data-error { color: #ff4b4b; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

def validate_historical_data(df):
    """Validate the structure of historical data"""
    required_cols = ['Tm', 'Batter', 'Pitcher', 'PA', 'AB', 'H', 'XB']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        st.error(f"Historical data missing required columns: {', '.join(missing)}")
        st.stop()
    return True

# Data Loading and Processing
@st.cache_data(ttl=3600)
def load_and_process_data():
    try:
        # Load base datasets
        prob = pd.read_csv(StringIO(requests.get(CSV_URLS['probabilities']).text)
        pct = pd.read_csv(StringIO(requests.get(CSV_URLS['percent_change']).text))
        
        # Load and validate historical data
        hist = pd.read_csv(StringIO(requests.get(CSV_URLS['historical']).text))
        validate_historical_data(hist)
        
        # Calculate AVG if not present
        if 'AVG' not in hist.columns:
            hist['AVG'] = hist['H'] / hist['AB'].replace(0, 1)  # Prevent divide by zero
        
        # Convert to percentages
        hist['AVG'] = hist['AVG'] * 100
        
        # Calculate PA-weighted metrics
        hist['PA_weight'] = np.log1p(hist['PA']) / np.log1p(25)
        hist['wAVG'] = hist['AVG'] * hist['PA_weight']
        hist['wXB%'] = (hist['XB'] / hist['AB'].replace(0, 1)) * 100 * hist['PA_weight']
        
        # Merge all datasets
        merged = pd.merge(
            pd.merge(prob, pct, on=['Tm', 'Batter', 'Pitcher'], suffixes=('_prob', '_pct')),
            hist[['Tm','Batter','Pitcher','PA','H','HR','wAVG','wXB%','AVG']],
            on=['Tm','Batter','Pitcher'], 
            how='left'
        )
        
        # Impute missing values
        league_avg = merged['AVG'].mean() if 'AVG' in merged.columns else 25.0  # Default .250 avg
        merged['wAVG'] = merged['wAVG'].fillna(league_avg)
        merged['wXB%'] = merged['wXB%'].fillna(0)
        merged['PA'] = merged['PA'].fillna(0)
        merged['AVG'] = merged['AVG'].fillna(league_avg)
        
        return merged
    
    except Exception as e:
        st.error(f"Data loading failed: {str(e)}")
        st.stop()

def calculate_scores(df):
    try:
        # Calculate all adjusted metrics
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        for metric in metrics:
            base_col = f'{metric}_prob'
            pct_col = f'{metric}_pct'
            if base_col in df.columns and pct_col in df.columns:
                df[f'adj_{metric}'] = df[base_col] * (1 + df[pct_col]/100).clip(0, 100)
            else:
                df[f'adj_{metric}'] = 0  # Default if columns missing
        
        # Define weights
        weights = {
            'adj_1B': 1.7, 'adj_XB': 1.3, 'adj_vs': 1.1,
            'adj_RC': 0.9, 'adj_HR': 0.5, 'adj_K': -1.4,
            'adj_BB': -1.0,
            'wAVG': 1.2, 'wXB%': 0.9, 'PA': 0.05
        }
        
        # Calculate score only using existing columns
        df['Score'] = 0
        for col, weight in weights.items():
            if col in df.columns:
                df['Score'] += df[col] * weight
        
        # Normalize score
        if df['Score'].nunique() > 1:
            df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
        else:
            df['Score'] = 50  # Default score
        
        return df.round(1)
    
    except Exception as e:
        st.error(f"Score calculation failed: {str(e)}")
        st.stop()

def visualize_results(df):
    try:
        # Score distribution
        chart = alt.Chart(df).mark_bar().encode(
            alt.X('Score:Q', bin=alt.Bin(maxbins=20)),
            alt.Y('count()'),
            tooltip=['count()']
        ).properties(title='Score Distribution', width=400)
        
        # PA-weighted scatter
        scatter = alt.Chart(df).mark_circle(size=60).encode(
            x='wAVG:Q',
            y='Score:Q',
            color='PA:Q',
            tooltip=['Batter','Pitcher','PA','wAVG','Score']
        ).properties(title='PA-Weighted Performance', width=400)
        
        st.altair_chart(chart | scatter)
    except Exception as e:
        st.error(f"Visualization failed: {str(e)}")

def create_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    with col2:
        st.title("MLB Daily Hit Predictor Pro+")

def create_filters():
    st.sidebar.header("Advanced Filters")
    filters = {
        'strict_mode': st.sidebar.checkbox('Strict Mode', True,
                      help="Limit strikeout risk ≤15% and walk risk ≤10%"),
        'min_1b': st.sidebar.slider("Minimum 1B%", 10, 40, 18),
        'num_players': st.sidebar.selectbox("Max Players", [5, 10, 15, 20], index=2),
        'pa_confidence': st.sidebar.slider("PA Confidence", 0, 25, 10,
                                          help="Minimum plate appearances for reliable history"),
        'min_wavg': st.sidebar.slider("Min Weighted AVG", 0.0, 40.0, 20.0, step=0.5)
    }
    
    if not filters['strict_mode']:
        filters.update({
            'max_k': st.sidebar.slider("Max K Risk", 15, 40, 25),
            'max_bb': st.sidebar.slider("Max BB Risk", 10, 30, 15)
        })
    
    return filters

def main_page():
    create_header()
    
    try:
        with st.spinner('Analyzing matchups...'):
            df = load_and_process_data()
            if df.empty:
                st.error("No data loaded - check your data sources")
                return
                
            df = calculate_scores(df)
            
            filters = create_filters()
            
            # Build query
            query_parts = [
                f"adj_1B >= {filters['min_1b']}",
                f"(PA >= {filters['pa_confidence']} | wAVG >= {filters['min_wavg']})"
            ]
            
            if filters['strict_mode']:
                query_parts += ["adj_K <= 15", "adj_BB <= 10"]
            else:
                query_parts += [
                    f"adj_K <= {filters.get('max_k', 25)}",
                    f"adj_BB <= {filters.get('max_bb', 15)}"
                ]
            
            filtered = df.query(" and ".join(query_parts)).sort_values('Score', ascending=False)
            filtered = filtered.head(filters['num_players'])

            # Display results
            st.subheader(f"Top {len(filtered)} Recommended Batters")
            
            show_cols = {
                'Batter': 'Batter',
                'Pitcher': 'Pitcher',
                'adj_1B': '1B%',
                'wAVG': 'Weighted AVG%',
                'PA': 'PA',
                'Score': 'Score',
                'adj_K': 'K Risk%',
                'adj_BB': 'BB Risk%'
            }
            
            styled_df = filtered[show_cols.keys()].rename(columns=show_cols)
            styled_df = styled_df.style.format({
                'Score': "{:.1f}",
                '1B%': "{:.1f}%",
                'Weighted AVG%': "{:.1f}%",
                'K Risk%': "{:.1f}%",
                'BB Risk%': "{:.1f}%"
            }).background_gradient(
                subset=['Score'],
                cmap='RdYlGn',
                vmin=0,
                vmax=100
            ).applymap(lambda x: 'color: #1a9641' if x >= 70 else ('color: #fdae61' if x >=50 else 'color: #d7191c'),
                       subset=['Score'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Visualizations
            visualize_results(df)
            
            # Legend and notes
            st.markdown("""
            <div class="color-legend">
                <strong>Key:</strong>
                <span class="pa-strong">≥10 PA</span> | 
                <span class="pa-warning"><10 PA</span>
            </div>
            """, unsafe_allow_html=True)
            
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.error("Please check your data sources and try again")

def info_page():
    st.title("Guide & FAQ 📚")
    with st.expander("📖 Full Guide", expanded=True):
        st.markdown("""
        ## MLB Hit Predictor Pro+ Guide
        ### Data Sources
        - Uses three data inputs:
          1. Probability predictions
          2. Percentage change models
          3. Historical matchup data
        ### Scoring System
        - Combines predictive models with actual performance
        - Weighted by plate appearance volume
        """)
    with st.expander("❓ Troubleshooting"):
        st.markdown("""
        ### Common Issues
        - 'Data loading failed': Check your CSV URLs
        - Missing columns: Verify historical data contains required fields
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
