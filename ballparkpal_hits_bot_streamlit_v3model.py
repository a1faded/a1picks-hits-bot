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
    # RESTORED: All 5 working links from your previous version
    csv_urls = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_walks': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
        'pitcher_hrs': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
        'pitcher_hits': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv'
    }

    try:
        # Load main batter datasets
        prob = pd.read_csv(csv_urls['probabilities'])
        pct = pd.read_csv(csv_urls['percent_change'])
        
        # Load the 3 pitcher files missing from your previous draft
        p_walks = pd.read_csv(csv_urls['pitcher_walks'])
        p_hrs = pd.read_csv(csv_urls['pitcher_hrs'])
        p_hits = pd.read_csv(csv_urls['pitcher_hits'])

        # Merge batter data
        df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], suffixes=('_prob', '_pct'))

        # Combine the 3 pitcher files into one 'Pitcher Intelligence' dataframe
        # We use 'Name' and 'Team' to join them as seen in your old script
        pitcher_intel = pd.merge(p_walks, p_hrs, on=['Team', 'Name', 'Park'], suffixes=('_w', '_hr'))
        pitcher_intel = pd.merge(pitcher_intel, p_hits, on=['Team', 'Name', 'Park'])
        
        # Format pitcher names for merging (extracting last name for better matching)
        pitcher_intel['Pitcher_Last'] = pitcher_intel['Name'].str.split().str[-1]
        df['Pitcher_Last'] = df['Pitcher'].str.split().str[-1]

        # Final merge: Batter data + Pitcher matchup data
        df = pd.merge(
            df,
            pitcher_intel,
            left_on=["Tm", "Pitcher_Last"],
            right_on=["Park", "Pitcher_Last"],
            how="left"
        )
        return df
    except Exception as e:
        st.error(f"⚠️ Data Sync Error: {e}")
        return pd.DataFrame()

# =========================
# SCORING ENGINE V4 (With Pitcher Data Integration)
# =========================
def calculate_v4_scores(df, profile):
    if df.empty: return df
    
    # 1. Apply Adjustments from 'percent_change' file
    metrics = ['1B', 'XB', 'HR', 'K', 'BB']
    for m in metrics:
        # Match columns from your CSV structure (e.g., 1B_prob and 1B_pct)
        df[f'adj_{m}'] = df[f'{m}_prob'] * (1 + df[f'{m}_pct'] / 100)

    # 2. Pitcher Matchup Modifier (Restored Logic)
    # Using 'Prob' columns from the pitcher files (renamed during merge)
    df['pitcher_mod'] = (
        df['Prob_x'].fillna(15).astype(float) * -0.2 + # Walks penalty
        df['Prob_y'].fillna(12).astype(float) * 0.5 +  # HR bonus
        df['Prob'].fillna(18).astype(float) * 0.3      # Hits bonus
    )

    # 3. Calculate Final Hit Probability
    df['total_hit_prob'] = df['adj_1B'] + df['adj_XB'] + df['adj_HR']
    
    # 4. Profile Weighting
    if profile == "power":
        df['base_score'] = (df['adj_HR'] * 2.5) + (df['adj_XB'] * 1.5) - (df['adj_K'] * 0.5)
    else:
        df['base_score'] = (df['adj_1B'] * 2.0) + (df['adj_XB'] * 1.2) - (df['adj_K'] * 1.5)

    # 5. Final Normalized Score
    df['Score'] = df['base_score'] + df['pitcher_mod']
    df['Score'] = 100 * (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min() + 1e-9)
    
    return df

# =========================
# UI / MAIN
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor V4 (Full Data Restored)")

df = load_data()

if not df.empty:
    profile = st.sidebar.selectbox("Select Strategy", ["contact", "power"])
    df = calculate_v4_scores(df, profile)
    
    min_score = st.sidebar.slider("Minimum Score", 0, 100, 50)
    
    filtered = df[df['Score'] >= min_score].sort_values("Score", ascending=False)
    
    st.dataframe(filtered[[
        "Batter", "Pitcher", "Score", "total_hit_prob", "adj_K", "adj_HR"
    ]], use_container_width=True)
else:
    st.warning("Could not connect to GitHub. Ensure the repository 'a1picks-hits-bot' is Public.")
