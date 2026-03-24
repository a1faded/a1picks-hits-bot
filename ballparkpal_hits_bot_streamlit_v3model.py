import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="A1PICKS MLB Predictor")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    # Standard Raw GitHub format: raw.githubusercontent.com/User/Repo/Branch/File
    csv_urls = {
        'probabilities': 'https://raw.githubusercontent.com/a1faded/a1picks-hits-bot/main/Ballpark%20Pal.csv',
        'percent_change': 'https://raw.githubusercontent.com/a1faded/a1picks-hits-bot/main/Ballpark%20Palmodel2.csv',
        'pitcher_data': 'https://raw.githubusercontent.com/a1faded/a1picks-hits-bot/main/pitcher_data.csv'
    }

    try:
        # Loading dataframes
        prob = pd.read_csv(csv_urls['percent_change'])
        pct = pd.read_csv(csv_urls['probabilities'])
        pitcher = pd.read_csv(csv_urls['pitcher_data'])

        # Data Cleaning: Strip spaces from column names just in case
        for df_temp in [prob, pct, pitcher]:
            df_temp.columns = df_temp.columns.str.strip()

        # Merge batter datasets
        df = pd.merge(prob, pct, on=["Tm", "Batter", "Pitcher"], how="inner")

        # Pitcher Merge Logic
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
        # Rethrow with a more helpful message for Streamlit
        raise Exception(f"Failed to fetch CSV: {e}")

# =========================
# THE REST OF YOUR LOGIC (Remains the same)
# =========================
def apply_adjustments(df):
    for col in ['1B', 'XB', 'HR', 'K', 'BB']:
        pct_col = f"{col}_pct"
        if pct_col in df.columns:
            df[f'adj_{col}'] = df[col] * (1 + df[pct_col] / 100)
        else:
            df[f'adj_{col}'] = df[col]
    df['total_hit_prob'] = df['adj_1B'] + df['adj_XB'] + df['adj_HR']
    return df

def calculate_league_aware_scores(df, profile_type):
    df = df.copy()
    # Percentiles
    for col in ['adj_1B', 'adj_XB', 'adj_HR', 'total_hit_prob']:
        df[f'{col}_pct'] = df[col].rank(pct=True)
    
    # Simple final score logic
    df['final_score'] = df['total_hit_prob'] * 100 # Placeholder for your V4 logic
    df['Score'] = df['final_score'] # Scaling logic...
    df['confidence'] = df['total_hit_prob']
    return df

# =========================
# MAIN APP
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor PRO V4")

try:
    df_raw = load_data()
    df_adj = apply_adjustments(df_raw)
    
    profile = st.selectbox("Select Profile", ["all", "contact", "power"])
    df_final = calculate_league_aware_scores(df_adj, profile)

    min_score = st.slider("Min Score", 0, 100, 50)
    st.dataframe(df_final[df_final['Score'] >= min_score])

except Exception as e:
    st.error(f"⚠️ {e}")
    st.info("Check if your GitHub repository is set to 'Public'. If it is Private, this script cannot see the files.")
