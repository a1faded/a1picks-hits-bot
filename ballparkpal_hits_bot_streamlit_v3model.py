import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    base_url = "https://raw.githubusercontent.com/a1faded/a1picks-hits-bot/main/"

    prob = pd.read_csv(base_url + "Ballpark%20Palmodel2.csv")
    pct = pd.read_csv(base_url + "Ballpark%20Pal.csv")
    pitcher = pd.read_csv(base_url + "pitcher_data.csv")

    df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], how="inner")

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
# ADJUSTMENTS
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
# 🔥 V4 SCORING ENGINE
# =========================
def calculate_league_aware_scores(df, profile_type):
    df = df.copy()

    pos_cols = ['adj_1B', 'adj_XB', 'adj_HR', 'total_hit_prob']
    neg_cols = ['adj_K', 'adj_BB']

    for col in pos_cols:
        df[f'{col}_pct'] = df[col].rank(pct=True)

    for col in neg_cols:
        df[f'{col}_pct'] = 1 - df[col].rank(pct=True)

    def blend(raw, pct, alpha=0.7):
        return raw * alpha + pct * (1 - alpha)

    league_avg_hr = df['adj_HR'].mean()
    league_avg_xb = df['adj_XB'].mean()
    league_avg_1b = df['adj_1B'].mean()

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

    df['power_boost'] = df['adj_HR'] / (league_avg_hr + 1e-9)
    df['extra_base_boost'] = df['adj_XB'] / (league_avg_xb + 1e-9)
    df['contact_boost'] = df['adj_1B'] / (league_avg_1b + 1e-9)

    if profile_type == "Contact":
        df['profile_score'] = (
            df['base_score'] * 0.7 +
            df['contact_boost'] * 0.2 +
            df['extra_base_boost'] * 0.1
        )
    elif profile_type == "Power":
        df['profile_score'] = (
            df['base_score'] * 0.7 +
            df['power_boost'] * 0.25 +
            df['extra_base_boost'] * 0.05
        )
    else:
        df['profile_score'] = df['base_score']

    pitcher_factor = (
        df['Hit_Probability'].fillna(0) * 0.5 +
        df['HR_Probability'].fillna(0) * 0.3 +
        df['Walk_Probability'].fillna(0) * 0.2
    )

    df['pitcher_multiplier'] = 0.85 + (pitcher_factor * 0.6)

    df['final_score'] = df['profile_score'] * df['pitcher_multiplier']

    min_score = df['final_score'].min()
    max_score = df['final_score'].max()

    df['Score'] = 100 * (df['final_score'] - min_score) / (max_score - min_score + 1e-9)

    df['confidence'] = (
        df['total_hit_prob'] * 0.6 +
        (1 - df['adj_K']) * 0.4
    )

    return df


# =========================
# APP UI
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor PRO V4")

df = load_data()
df = apply_adjustments(df)

# Sidebar controls
st.sidebar.header("Filters")

profile = st.sidebar.selectbox(
    "Profile Type",
    ["All", "Contact", "Power"]
)

min_score = st.sidebar.slider("Minimum Score", 0, 100, 50)
min_conf = st.sidebar.slider("Minimum Confidence", 0.0, 1.0, 0.3)

df = calculate_league_aware_scores(df, profile)

df = df[
    (df['Score'] >= min_score) &
    (df['confidence'] >= min_conf)
]

# =========================
# DISPLAY TABLE
# =========================
st.subheader("Top Hitters Today")

display_cols = [
    "Batter",
    "Pitcher",
    "Score",
    "confidence",
    "total_hit_prob",
    "adj_1B",
    "adj_XB",
    "adj_HR",
    "adj_K"
]

st.dataframe(
    df.sort_values("Score", ascending=False)[display_cols],
    use_container_width=True
)

# =========================
# QUICK STATS
# =========================
st.subheader("Slate Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Players", len(df))
col2.metric("Avg Score", round(df['Score'].mean(), 2))
col3.metric("Avg Hit Prob", round(df['total_hit_prob'].mean(), 3))

st.caption("Data auto-refreshes every 15 minutes")
