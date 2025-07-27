import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro",
    layout="wide",
    page_icon="⚾"
)

# Constants and Configuration
PER_BATTER_PA = 134
K_PERCENT_THRESHOLD = 0.134  # ~13.4% (18 Ks over 134 PA)
BB_PERCENT_THRESHOLD = 0.052  # ~5.2% (7 BBs over 134 PA)

CONFIG = {
    'csv_urls': {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'
    },
    'base_hit_weights': {
        'adj_1B': 2.0,
        'adj_XB': 1.8,
        'adj_vs': 1.2,
        'adj_RC': 0.8,
        'adj_HR': 0.6,
        'adj_K': -2.0,
        'adj_BB': -0.8
    },
    'expected_columns': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
    'cache_ttl': 900
}

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_csv_with_validation(url, description, expected_columns):
    try:
        with st.spinner(f'Loading {description}...'):
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            df = pd.read_csv(StringIO(response.text))

            if df.empty:
                st.error(f"{description}: No data found")
                return None

            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                st.error(f"{description}: Missing columns {missing_cols}")
                return None

            key_cols = ['Tm', 'Batter', 'Pitcher']
            null_counts = df[key_cols].isnull().sum()
            if null_counts.any():
                st.error(f"{description}: Null values in {null_counts[null_counts > 0].index.tolist()}")
                return None

            return df
    except Exception as e:
        st.error(f"{description}: {str(e)}")
        return None

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_and_process_data():
    prob_df = load_csv_with_validation(
        CONFIG['csv_urls']['probabilities'], "Base Probabilities", CONFIG['expected_columns']
    )
    pct_df = load_csv_with_validation(
        CONFIG['csv_urls']['percent_change'], "Adjustment Factors", CONFIG['expected_columns']
    )
    if prob_df is None or pct_df is None:
        return None

    try:
        merged_df = pd.merge(
            prob_df, pct_df,
            on=['Tm', 'Batter', 'Pitcher'],
            suffixes=('_prob', '_pct'),
            how='inner'
        )
    except Exception as e:
        st.error(f"Failed to merge datasets: {str(e)}")
        return None

    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
    for metric in metrics:
        base_col = f'{metric}.1' if metric in ['K', 'BB'] else f'{metric}_prob'
        pct_col = f'{metric}_pct'
        if base_col in merged_df.columns and pct_col in merged_df.columns:
            merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col] / 100)
            if metric in ['K', 'BB', '1B', 'XB', 'HR']:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            else:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
        else:
            merged_df[f'adj_{metric}'] = 20 if metric in ['K', 'BB'] else 0

    merged_df['total_hit_prob'] = (
        merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
    ).clip(upper=100)

    return merged_df

def calculate_base_hit_scores(df):
    weights = CONFIG['base_hit_weights']
    df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)

    df['contact_bonus'] = np.where(
        (df['total_hit_prob'] > 40) & (df['adj_K'] < 18), 8, 0
    )
    df['consistency_bonus'] = np.where(
        (df['adj_1B'] > 20) & (df['adj_XB'] > 8), 5, 0
    )
    df['matchup_bonus'] = np.where(df['adj_vs'] > 70, 3, 0)

    df['Score'] = df['base_score'] + df['contact_bonus'] + df['consistency_bonus'] + df['matchup_bonus']
    if df['Score'].max() != df['Score'].min():
        df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    else:
        df['Score'] = 50

    # Add contact-first swinger detection
    df['low_k_flag'] = df['adj_K'] <= (K_PERCENT_THRESHOLD * 100)
    df['low_bb_flag'] = df['adj_BB'] <= (BB_PERCENT_THRESHOLD * 100)
    df['contact_swinger'] = df['low_k_flag'] & df['low_bb_flag']

    return df.round(1)

def create_smart_filters(df):
    st.sidebar.header("🎯 Smart Base Hit Filters")
    filters = {}

    filters['min_hit_prob'] = st.sidebar.slider("Min Hit Probability", 0, 100, 30)
    filters['max_k'] = st.sidebar.slider("Max K%", 0, 100, 18)
    filters['max_walk'] = st.sidebar.slider("Max BB%", 0, 100, 10)
    filters['show_contact_swingers_only'] = st.sidebar.checkbox(
        "Only show Contact-First Swingers",
        value=False,
        help="Low strikeout + low walk hitters"
    )
    filters['result_count'] = st.sidebar.selectbox("Number of Results", [5, 10, 15, 20], index=2)
    return filters

def apply_smart_filters(df, filters):
    if df is None or df.empty:
        return df
    query_parts = []
    if 'min_hit_prob' in filters:
        query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']}")
    if 'max_k' in filters:
        query_parts.append(f"adj_K <= {filters['max_k']}")
    if 'max_walk' in filters:
        query_parts.append(f"adj_BB <= {filters['max_walk']}")
    if filters.get('show_contact_swingers_only'):
        query_parts.append("contact_swinger == True")

    if query_parts:
        return df.query(" and ".join(query_parts)).sort_values('Score', ascending=False).head(filters['result_count'])
    return df.sort_values('Score', ascending=False).head(filters['result_count'])

def display_results(df):
    st.subheader("🎯 Top Base Hit Candidates")
    if df.empty:
        st.warning("No matching players found.")
        return
    styled = df[['Batter', 'Tm', 'Pitcher', 'total_hit_prob', 'adj_K', 'adj_BB', 'Score', 'contact_swinger']].copy()
    styled['Contact Type'] = styled['contact_swinger'].apply(lambda x: "🔥 Contact-First" if x else "")
    styled = styled.rename(columns={
        'Batter': 'Player',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit %',
        'adj_K': 'K %',
        'adj_BB': 'BB %',
        'Score': 'Score'
    })
    st.dataframe(styled.drop(columns=['contact_swinger']), use_container_width=True)

def main():
    st.title("⚾ A1PICKS MLB Hit Predictor Pro")
    df = load_and_process_data()
    if df is not None:
        df = calculate_base_hit_scores(df)
        filters = create_smart_filters(df)
        filtered_df = apply_smart_filters(df, filters)
        display_results(filtered_df)
    else:
        st.error("Failed to load data.")

if __name__ == '__main__':
    main()
