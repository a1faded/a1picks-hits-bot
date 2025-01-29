import streamlit as st
import pandas as pd
import requests
from io import StringIO
import matplotlib
matplotlib.use('Agg') 
import numpy as np

# Configure Streamlit page
st.set_page_config(page_title="A1PICKS MLB Hit Predictor", layout="wide", page_icon="⚾")

# Constants
CSV_URLS = {
    'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
    'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'
}

# Data Loading and Processing
@st.cache_data(ttl=3600)  # Refresh data every hour
def load_and_process_data():
    # Load datasets
    prob = pd.read_csv(StringIO(requests.get(CSV_URLS['probabilities']).text))
    pct = pd.read_csv(StringIO(requests.get(CSV_URLS['percent_change']).text))
    
    # Merge datasets
    merged = pd.merge(prob, pct, 
                     on=['Tm', 'Batter', 'Pitcher'],
                     suffixes=('_prob', '_pct'))
    
    # Calculate adjusted metrics
    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
    for metric in metrics:
        base_col = f'{metric}_prob'
        pct_col = f'{metric}_pct'
        merged[f'adj_{metric}'] = merged[base_col] * (1 + merged[pct_col]/100)
        merged[f'adj_{metric}'] = merged[f'adj_{metric}'].clip(lower=0, upper=100)
    
    return merged

# Score Calculation
def calculate_scores(df):
    # Priority-based weighting system
    weights = {
        'adj_1B': 1.7,    # Highest weight for singles probability
        'adj_XB': 1.3,    # Extra base boost
        'adj_vs': 1.1,    # Pitcher matchup performance
        'adj_RC': 0.9,    # Run creation context
        'adj_HR': 0.5,    # Home run bonus
        'adj_K': -1.4,    # Strikeout penalty
        'adj_BB': -1.0    # Walk penalty
    }
    
    # Calculate composite score
    df['Score'] = sum(df[col]*weight for col, weight in weights.items())
    
    # Normalize score to 0-100 scale
    df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    
    return df.round(1)

# UI Components
def create_header():
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    with col2:
        st.title("MLB Daily Hit Predictor Pro")
        st.markdown("""
        **Algorithm Features:**
        - Advanced BvP (Batter vs Pitcher) modeling
        - Park-factor adjusted projections
        - Real-time matchup quality scoring
        """)

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

def visualize_results(df):
    # Score distribution chart
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(df['Score'], bins=20, color='#1f77b4', alpha=0.7)
    ax.set_title('Score Distribution Across Players')
    ax.set_xlabel('Prediction Score')
    ax.set_ylabel('Number of Players')
    st.pyplot(fig)

# Main Application
def main():
    create_header()
    
    # Load and process data
    with st.spinner('Crunching matchup data...'):
        df = load_and_process_data()
        df = calculate_scores(df)
    
    # Create filters
    filters = create_filters()
    
    # Apply filters
    query_parts = []
    if filters['strict_mode']:
        query_parts.append("adj_K <= 15 and adj_BB <= 10")
    else:
        query_parts.append(f"adj_K <= {filters.get('max_k', 25)}")
        query_parts.append(f"adj_BB <= {filters.get('max_bb', 15)}")
    
    query_parts.append(f"adj_1B >= {filters['min_1b']}")
    full_query = " and ".join(query_parts)
    
    filtered = df.query(full_query).sort_values('Score', ascending=False).head(filters['num_players'])
    
    # Display results
    st.subheader(f"Top {len(filtered)} Recommended Batters")
    
    # Create styled dataframe
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
    styled_df = styled_df.style \
        .background_gradient(subset=['Score'], cmap='YlGn') \
        .bar(subset=['1B%', 'XB%'], color='#5fba7d') \
        .bar(subset=['K Risk%', 'BB Risk%'], color='#ff6961') \
        .format({'Score': "{:.1f}", '1B%': "{:.1f}%", 'XB%': "{:.1f}%", 
                'vs Pitcher%': "{:.1f}%", 'K Risk%': "{:.1f}%", 'BB Risk%': "{:.1f}%"})
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Visualizations
    visualize_results(df)
    
    # Disclaimer
    st.markdown("""
    <div style='border-left: 4px solid #ff6961; padding-left: 1rem; margin-top: 2rem;'>
    <p style='color: #666; font-size: 0.9rem;'>
    <strong>Note:</strong> These predictions combine historical performance and matchup analytics. 
    Always check lineups and weather conditions before finalizing selections. 
    Scores above 70 indicate elite matchups, 50-70 good matchups, below 50 marginal matchups.
    </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
