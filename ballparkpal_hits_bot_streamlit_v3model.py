import streamlit as st
import pandas as pd
import numpy as np
import requests
from io import StringIO

st.set_page_config(layout="wide")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    base_url = "https://raw.githubusercontent.com/a1faded/a1picks-hits-bot/main/"

    prob_url = base_url + "Ballpark Palmodel2.csv"
    pct_url = base_url + "Ballpark Pal.csv"
    pitcher_url = base_url + "pitcher_data.csv"

    prob = pd.read_csv(prob_url)
    pct = pd.read_csv(pct_url)
    pitcher = pd.read_csv(pitcher_url)

    df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], how="inner")

    # safer pitcher merge (fixes name collision issue)
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

    # Percentiles
    pos_cols = ['adj_1B', 'adj_XB', 'adj_HR', 'total_hit_prob']
    neg_cols = ['adj_K', 'adj_BB']

    for col in pos_cols:
        df[f'{col}_pct'] = df[col].rank(pct=True)

    for col in neg_cols:
        df[f'{col}_pct'] = 1 - df[col].rank(pct=True)

    def blend(raw, pct, alpha=0.7):
        return raw * alpha + pct * (1 - alpha)

    # League averages
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
    pitcher_factor = (
        df['Hit_Probability'].fillna(0) * 0.5 +
        df['HR_Probability'].fillna(0) * 0.3 +
        df['Walk_Probability'].fillna(0) * 0.2
    )

    df['pitcher_multiplier'] = 0.85 + (pitcher_factor * 0.6)

    df['final_score'] = df['profile_score'] * df['pitcher_multiplier']

    # Normalize
    min_score = df['final_score'].min()
    max_score = df['final_score'].max()

    df['Score'] = 100 * (df['final_score'] - min_score) / (max_score - min_score + 1e-9)

    # NEW: confidence
    df['confidence'] = (
        df['total_hit_prob'] * 0.6 +
        (1 - df['adj_K']) * 0.4
    )

    return df


# =========================
# MAIN APP
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor PRO V4")

df = load_data()
df = apply_adjustments(df)

profile = st.selectbox(
    "Select Profile",
    ["all", "contact", "power"]
)

df = calculate_league_aware_scores(df, profile)

# =========================
# FILTERS
# =========================
min_score = st.slider("Minimum Score", 0, 100, 50)
df = df[df['Score'] >= min_score]

# =========================
# DISPLAY
# =========================
st.dataframe(
    df.sort_values("Score", ascending=False)[[
        "Batter",
        "Pitcher",
        "Score",
        "confidence",
        "total_hit_prob",
        "adj_K",
        "adj_HR",
        "adj_XB"
    ]]
)

st.caption("Data refreshes every 15 minutes.")
