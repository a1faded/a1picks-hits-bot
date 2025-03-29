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
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=3600)
def load_and_process_data():
    try:
        # Load datasets with error handling
        def fetch_data(url):
            response = requests.get(url)
            response.raise_for_status()
            return pd.read_csv(StringIO(response.text))
        
        prob = fetch_data(CSV_URLS['probabilities'])
        pct = fetch_data(CSV_URLS['percent_change'])
        hist = fetch_data(CSV_URLS['historical'])
        
        # Process historical data
        hist['AVG'] = (hist['H'] / hist['AB'].replace(0, 1)) * 100
        hist['PA_weight'] = np.log1p(hist['PA']) / np.log1p(25)
        hist['wAVG'] = hist['AVG'] * hist['PA_weight']
        hist['wXB%'] = (hist['XB'] / hist['AB'].replace(0, 1)) * 100 * hist['PA_weight']
        
        # Merge data with validation
        merged = pd.merge(
            pd.merge(prob, pct, on=['Tm', 'Batter', 'Pitcher'], suffixes=('_prob', '_pct')),
            hist[['Tm','Batter','Pitcher','PA','H','HR','wAVG','wXB%','AVG']],
            on=['Tm','Batter','Pitcher'],
            how='left',
            validate='one_to_one'
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
        # Calculate adjusted metrics
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        for metric in metrics:
            base_col = f'{metric}_prob'
            pct_col = f'{metric}_pct'
            if base_col in df.columns and pct_col in df.columns:
                df[f'adj_{metric}'] = df[base_col] * (1 + df[pct_col]/100).clip(0, 100)
            else:
                df[f'adj_{metric}'] = 0

        # Create PA confidence tiers
        pa_bins = [-1, 0, 5, 15, 25, np.inf]
        pa_labels = [0, 1, 3, 5, 7]
        df['PA_tier'] = pd.cut(df['PA'], bins=pa_bins, labels=pa_labels).astype(int)

        # Non-linear penalties
        df['K_penalty'] = np.where(
            df['adj_K'] > 15,
            (df['adj_K'] - 15) ** 1.5,
            0
        )
        
        df['BB_penalty'] = np.where(
            df['adj_BB'] > 12,
            (df['adj_BB'] - 12) ** 1.2,
            0
        )

        # Updated weights
        weights = {
            'adj_1B': 2.0,    # Increased emphasis on singles
            'adj_XB': 1.2,     # Slightly reduced XB weight
            'wAVG': 1.5,       # More historical emphasis
            'adj_vs': 1.0,     # Matchup history
            'PA_tier': 3.0,    # Tiered PA bonus
            'K_penalty': -0.3, # Exponential K penalty
            'adj_BB': -0.7,    # Adjusted walk impact
            'adj_HR': 0.4,     # Reduced HR weight
            'adj_RC': 0.8      # Runs created
        }

        # Calculate base score
        df['Score'] = sum(df[col]*weight for col, weight in weights.items() if col in df.columns)
        
        # Apply safeguards
        df['Score'] = np.where(
            df['adj_1B'] < 15,
            df['Score'] * 0.7,  # 30% penalty for low 1B%
            df['Score']
        )
        
        # Apply ratio-based adjustments
        df['1B_K_ratio'] = df['adj_1B'] / df['adj_K']
        df['Score'] = np.where(
            df['1B_K_ratio'] < 1.3,
            df['Score'] * 0.85,
            df['Score']
        )

        # Normalize scores
        score_min = df['Score'].min()
        score_range = df['Score'].max() - score_min
        df['Score'] = np.where(
            score_range > 0,
            ((df['Score'] - score_min) / score_range) * 100,
            50  # Fallback for identical scores
        )
        
        return df.round(1)
    
    except Exception as e:
        st.error(f"Score calculation failed: {str(e)}")
        st.stop()

def create_filters():
    st.sidebar.header("Advanced Filters")
    filters = {
        'strict_mode': st.sidebar.checkbox('Strict Mode', True,
                      help="Enforce 1B/K ratio >1.5 and BB% ≤12%"),
        'min_1b': st.sidebar.slider("Minimum 1B%", 10, 40, 18),
        'num_players': st.sidebar.selectbox("Number of Players", [5, 10, 15, 20], index=2),
        'pa_tier': st.sidebar.slider("Min PA Confidence Tier", 0, 4, 2,
                   format=lambda x: ["None", "Low (1-5)", "Medium (5-15)", "High (15-25)", "Elite (25+)"][x]),
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
        f"PA_tier >= {filters['pa_tier']}",
        f"wAVG >= {filters['min_wavg']}"
    ]
    
    if filters['strict_mode']:
        query_parts += [
            "1B_K_ratio >= 1.3",
            "adj_BB <= 12",
            "K_penalty == 0"
        ]
    else:
        query_parts += [
            f"adj_K <= {filters.get('max_k', 25)}",
            f"adj_BB <= {filters.get('max_bb', 15)}"
        ]
    
    return df.query(" and ".join(query_parts))

def style_dataframe(df):
    display_cols = [
        'Batter', 'Pitcher', 'adj_1B', 'adj_XB', 'wAVG', 'PA_tier',
        'adj_K', 'adj_BB', '1B_K_ratio', 'Score'
    ]
    display_cols = [col for col in display_cols if col in df.columns]
    
    styled = df[display_cols].rename(columns={
        'adj_1B': '1B%', 
        'adj_XB': 'XB%',
        'wAVG': 'wAVG%', 
        'adj_K': 'K%', 
        'adj_BB': 'BB%',
        'PA_tier': 'PA Tier',
        '1B_K_ratio': '1B/K Ratio'
    })
    
    def score_color(val):
        if val >= 70: return 'background-color: #1a9641; color: white'
        elif val >= 50: return 'background-color: #fdae61'
        else: return 'background-color: #d7191c; color: white'
    
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
    - **Score**: 🟩 ≥70 (Elite) | 🟨 50-69 (Good) | 🟥 <50 (Risky)
    - **PA Tier**: 
      🔴 0 | 🟠 1-5 | 🟡 5-15 | 🟢 15-25 | 🔵 25+
    - **1B/K Ratio**: ≥1.3 required in Strict Mode
    """, unsafe_allow_html=True)

# ... (keep the info_page() and main() functions same as before)

if __name__ == "__main__":
    main()
