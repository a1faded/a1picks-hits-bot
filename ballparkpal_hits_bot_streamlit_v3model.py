import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

st.set_page_config(layout="wide", page_title="A1PICKS MLB Predictor")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    # Using your requested URL setup to ensure compatibility
    csv_urls = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_data': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_data.csv'
    }

    # Loading the main dataframes
    # Note: 'percent_change' corresponds to the 'prob' variable in your logic
    # and 'probabilities' corresponds to the 'pct' variable.
    prob = pd.read_csv(csv_urls['percent_change'])
    pct = pd.read_csv(csv_urls['probabilities'])
    pitcher = pd.read_csv(csv_urls['pitcher_data'])

    # Merge main batter datasets
    df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], how="inner")

    # Safer pitcher merge (fixes name collision issues)
    # Creating a join key based on the last name to handle different formatting
    pitcher['Pitcher_Last'] = pitcher['Pitcher'].str.split().str[-1]
    df['Pitcher_Last'] = df['Pitcher'].str.split().str[-1]

    df = pd.merge(
        df,
        pitcher,
        left_on=["Tm", "Pitcher_Last"],
        right_on=["Opponent_Team", "Pitcher_Last"],
        how="left"
    )

    return df


# =========================
# ADJUSTED METRICS
# =========================
def apply_adjustments(df):
    for col in ['1B', 'XB', 'HR', 'K', 'BB']:
        pct_col = f"{col}_pct"
        if pct_col in df.columns:
            df[f'adj_{col}'] = df[col] * (1 + df[pct_col] / 100)
        else:
            df[f'adj_{col}'] = df[col]

    df['total_hit_prob'] = df['adj_1B'] + df['adj_XB'] + df['adj_HR']
    df['power_combo'] = df['adj_XB'] + df['adj_HR']

    return df


# =========================
# 🔥 NEW SCORING ENGINE V4
# =========================
def calculate_league_aware_scores(df, profile_type):
    df = df.copy()

    # Percentiles for ranking
    pos_cols = ['adj_1B', 'adj_XB', 'adj_HR', 'total_hit_prob']
    neg_cols = ['adj_K', 'adj_BB']

    for col in pos_cols:
        df[f'{col}_pct'] = df[col].rank(pct=True)

    for col in neg_cols:
        df[f'{col}_pct'] = 1 - df[col].rank(pct=True)

    def blend(raw, pct, alpha=0.7):
        return raw * alpha + pct * (1 - alpha)

    # League averages for boosting
    league_avg_hr = df['adj_HR'].mean()
    league_avg_xb = df['adj_XB'].mean()
    league_avg_1b = df['adj_1B'].mean()

    # Outcome scoring
    df['hit_value'] = (
        blend(df['adj_1B'], df['adj_1B_pct']) * 1.0 +
        blend(df['adj_XB'], df['adj_XB_pct']) * 1.6 +
        blend(df['adj_HR'], df['adj_HR_pct']) * 2.2
    )

    df['risk_penalty'] = (
        blend(df['adj_K'], df['adj_K_pct']) * 1.2 +
        blend(df['adj_BB'], df['adj_BB_pct']) * 0.4
    )

    df['base_score'] = df['hit_value'] - df['risk_penalty']

    # Smooth boosts
    df['power_boost'] = df['adj_HR'] / (league_avg_hr + 1e-9)
    df['extra_base_boost'] = df['adj_XB'] / (league_avg_xb + 1e-9)
    df['contact_boost'] = df['adj_1B'] / (league_avg_1b + 1e-9)

    # Profile weighting
    if profile_type == "contact":
        df['profile_score'] = (
            df['base_score'] * 0.7 +
            df['contact_boost'] * 0.2 +
            df['extra_base_boost'] * 0.1
        )
    elif profile_type == "power":
        df['profile_score'] = (
            df['base_score'] * 0.7 +
            df['power_boost'] * 0.25 +
            df['extra_base_boost'] * 0.05
        )
    else:
        df['profile_score'] = df['base_score']

    # 🔥 Smooth pitcher multiplier
    # Handles missing data with fillna(0)
    pitcher_factor = (
        df['Hit_Probability'].fillna(0) * 0.5 +
        df['HR_Probability'].fillna(0) * 0.3 +
        df['Walk_Probability'].fillna(0) * 0.2
    )

    df['pitcher_multiplier'] = 0.85 + (pitcher_factor * 0.6)
    df['final_score'] = df['profile_score'] * df['pitcher_multiplier']

    # Normalize to 0-100 scale
    min_score = df['final_score'].min()
    max_score = df['final_score'].max()
    df['Score'] = 100 * (df['final_score'] - min_score) / (max_score - min_score + 1e-9)

    # Confidence Metric
    df['confidence'] = (
        df['total_hit_prob'] * 0.6 +
        (1 - df['adj_K']) * 0.4
    )

    return df


# =========================
# MAIN APP
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor PRO V4")

try:
    df = load_data()
    df = apply_adjustments(df)

    profile = st.selectbox(
        "Select Betting Profile",
        ["all", "contact", "power"]
    )

    df = calculate_league_aware_scores(df, profile)

    # =========================
    # FILTERS
    # =========================
    min_score = st.slider("Minimum Score Filter", 0, 100, 50)
    filtered_df = df[df['Score'] >= min_score]

    # =========================
    # DISPLAY
    # =========================
    cols_to_show = [
        "Batter", "Pitcher", "Score", "confidence", 
        "total_hit_prob", "adj_K", "adj_HR", "adj_XB"
    ]

    st.dataframe(
        filtered_df.sort_values("Score", ascending=False)[cols_to_show],
        use_container_width=True
    )

    st.caption("Data refreshes every 15 minutes. Percentages are adjusted based on ballpark factors.")

except Exception as e:
    st.error(f"Error loading or processing data: {e}")
    st.info("Check if the GitHub CSV files are currently accessible and formatted correctly.")
