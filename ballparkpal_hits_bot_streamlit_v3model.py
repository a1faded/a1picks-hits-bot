import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

st.set_page_config(layout="wide", page_title="A1PICKS MLB Hit Predictor PRO V4")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    # These URLs are pulled directly from your working version
    csv_urls = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Ballpark%20Palmodel2.csv',
        'pitcher_data': 'https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/pitcher_data.csv'
    }

    try:
        # We use storage_options or standard read_csv as these links are public
        prob = pd.read_csv(csv_urls['percent_change'])
        pct = pd.read_csv(csv_urls['probabilities'])
        pitcher = pd.read_csv(csv_urls['pitcher_data'])

        # Merge core datasets
        df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], how="inner")

        # Safer pitcher merge logic from your V4 draft
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
    except Exception as e:
        st.error(f"Critical Error: Could not fetch data. {e}")
        return pd.DataFrame()

# =========================
# ADJUSTED METRICS
# =========================
def apply_adjustments(df):
    if df.empty: return df
    
    for col in ['1B', 'XB', 'HR', 'K', 'BB']:
        pct_col = f"{col}_pct"
        # Using the logic from your V4: adjusting raw counts by the % change
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
    if df.empty: return df
    df = df.copy()

    # Percentiles
    pos_cols = ['adj_1B', 'adj_XB', 'adj_HR', 'total_hit_prob']
    neg_cols = ['adj_K', 'adj_BB']

    for col in pos_cols:
        df[f'{col}_pct'] = df[col].rank(pct=True)
    for col in neg_cols:
        df[f'{col}_pct'] = 1 - df[col].rank(pct=True)

    def blend(raw, pct, alpha=0.7):
        return raw * alpha + pct * (1 - alpha)

    # League averages for the boosts
    l_hr, l_xb, l_1b = df['adj_HR'].mean(), df['adj_XB'].mean(), df['adj_1B'].mean()

    # Scoring Logic
    df['hit_value'] = (blend(df['adj_1B'], df['adj_1B_pct']) * 1.0 +
                       blend(df['adj_XB'], df['adj_XB_pct']) * 1.6 +
                       blend(df['adj_HR'], df['adj_HR_pct']) * 2.2)

    df['risk_penalty'] = (blend(df['adj_K'], df['adj_K_pct']) * 1.2 +
                          blend(df['adj_BB'], df['adj_BB_pct']) * 0.4)

    df['base_score'] = df['hit_value'] - df['risk_penalty']

    # Boosts
    df['power_boost'] = df['adj_HR'] / (l_hr + 1e-9)
    df['extra_base_boost'] = df['adj_XB'] / (l_xb + 1e-9)
    df['contact_boost'] = df['adj_1B'] / (l_1b + 1e-9)

    # Profile weighting
    if profile_type == "contact":
        df['profile_score'] = (df['base_score'] * 0.7 + df['contact_boost'] * 0.2 + df['extra_base_boost'] * 0.1)
    elif profile_type == "power":
        df['profile_score'] = (df['base_score'] * 0.7 + df['power_boost'] * 0.25 + df['extra_base_boost'] * 0.05)
    else:
        df['profile_score'] = df['base_score']

    # Pitcher factor
    p_factor = (df['Hit_Probability'].fillna(0) * 0.5 + 
                df['HR_Probability'].fillna(0) * 0.3 + 
                df['Walk_Probability'].fillna(0) * 0.2)

    df['final_score'] = df['profile_score'] * (0.85 + (p_factor * 0.6))

    # Normalize Score 0-100
    df['Score'] = 100 * (df['final_score'] - df['final_score'].min()) / (df['final_score'].max() - df['final_score'].min() + 1e-9)
    
    df['confidence'] = (df['total_hit_prob'] * 0.6 + (1 - df['adj_K']) * 0.4)
    return df

# =========================
# MAIN APP
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor PRO V4")

df = load_data()

if not df.empty:
    df = apply_adjustments(df)
    
    profile = st.selectbox("Select Profile", ["all", "contact", "power"])
    df = calculate_league_aware_scores(df, profile)

    min_score = st.slider("Minimum Score", 0, 100, 50)
    display_df = df[df['Score'] >= min_score].sort_values("Score", ascending=False)

    st.dataframe(
        display_df[[
            "Batter", "Pitcher", "Score", "confidence", 
            "total_hit_prob", "adj_K", "adj_HR", "adj_XB"
        ]],
        use_container_width=True
    )
    st.caption("Data successfully pulled using working GitHub branch heads.")
else:
    st.warning("Data could not be loaded. Please check your GitHub repository branch names.")
