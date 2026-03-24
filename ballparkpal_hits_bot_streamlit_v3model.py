import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(layout="wide", page_title="A1PICKS MLB Hit Predictor V4")

# =========================
# DATA LOADING
# =========================
@st.cache_data(ttl=900)
def load_data():
    csv_urls = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_walks': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
        'pitcher_hrs': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
        'pitcher_hits': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv'
    }

    try:
        # Load core batter data
        prob_df = pd.read_csv(csv_urls['probabilities'])
        pct_df = pd.read_csv(csv_urls['percent_change'])
        
        # Load the 3 pitcher files
        p_walks = pd.read_csv(csv_urls['pitcher_walks']).rename(columns={'Prob': 'p_walk_val'})
        p_hrs = pd.read_csv(csv_urls['pitcher_hrs']).rename(columns={'Prob': 'p_hr_val'})
        p_hits = pd.read_csv(csv_urls['pitcher_hits']).rename(columns={'Prob': 'p_hit_val'})

        # Merge batter data
        df = pd.merge(prob_df, pct_df, on=["Tm", "Batter", "Pitcher"], suffixes=('_prob', '_pct'))

        # Combine pitcher files into one intelligence frame
        # We merge on Name and Team (which represents the Pitcher)
        p_intel = pd.merge(p_walks[['Name', 'Team', 'p_walk_val']], 
                           p_hrs[['Name', 'Team', 'p_hr_val']], on=['Name', 'Team'], how='outer')
        p_intel = pd.merge(p_intel, 
                           p_hits[['Name', 'Team', 'p_hit_val']], on=['Name', 'Team'], how='outer')
        
        # Match Pitcher Last Name
        p_intel['P_Last'] = p_intel['Name'].str.split().str[-1]
        df['P_Last'] = df['Pitcher'].str.split().str[-1]

        # Final Merge
        df = pd.merge(df, p_intel, left_on=["Tm", "P_Last"], right_on=["Team", "P_Last"], how="left")
        
        return df
    except Exception as e:
        st.error(f"Data Loading Error: {e}")
        return pd.DataFrame()

# =========================
# SCORING ENGINE V4
# =========================
def calculate_v4_scores(df, profile):
    if df.empty: return df
    df = df.copy()

    # 1. Adjusted Batter Metrics (Prob * % Change)
    for m in ['1B', 'XB', 'HR', 'K', 'BB']:
        prob_col = f"{m}_prob"
        pct_col = f"{m}_pct"
        if prob_col in df.columns and pct_col in df.columns:
            df[f'adj_{m}'] = df[prob_col] * (1 + df[pct_col] / 100)
        else:
            df[f'adj_{m}'] = 0

    # 2. Pitcher Matchup Modifier (Using the new names we created in load_data)
    # Fillna with league averages if pitcher data is missing
    df['p_walk_val'] = pd.to_numeric(df['p_walk_val'], errors='coerce').fillna(15)
    df['p_hr_val'] = pd.to_numeric(df['p_hr_val'], errors='coerce').fillna(12)
    df['p_hit_val'] = pd.to_numeric(df['p_hit_val'], errors='coerce').fillna(18)

    df['pitcher_mod'] = (df['p_walk_val'] * -0.2) + (df['p_hr_val'] * 0.5) + (df['p_hit_val'] * 0.3)

    # 3. Calculation Logic
    df['total_hit_prob'] = df['adj_1B'] + df['adj_XB'] + df['adj_HR']
    
    if profile == "power":
        df['raw_score'] = (df['adj_HR'] * 3.0) + (df['adj_XB'] * 1.5) - (df['adj_K'] * 0.8)
    else: # Contact profile
        df['raw_score'] = (df['adj_1B'] * 2.5) + (df['adj_XB'] * 1.0) - (df['adj_K'] * 1.5)

    # Apply Pitcher Modifier
    df['final_score'] = df['raw_score'] + df['pitcher_mod']

    # 4. Normalize 0-100
    s_min = df['final_score'].min()
    s_max = df['final_score'].max()
    df['Score'] = 100 * (df['final_score'] - s_min) / (s_max - s_min + 1e-9)
    
    # Confidence Metric
    df['confidence'] = (df['total_hit_prob'] * 0.7) + ((1 - df['adj_K']) * 0.3)

    return df

# =========================
# UI / MAIN
# =========================
st.title("⚾ A1PICKS MLB Hit Predictor V4")

df = load_data()

if not df.empty:
    # Sidebar Filters
    st.sidebar.header("Settings")
    profile = st.sidebar.selectbox("Select Strategy", ["contact", "power"])
    min_score = st.sidebar.slider("Min Score Filter", 0, 100, 40)
    
    # Run Scoring
    df_results = calculate_v4_scores(df, profile)
    
    # Filter and Display
    show_cols = ["Batter", "Pitcher", "Score", "confidence", "total_hit_prob", "adj_K", "adj_HR"]
    filtered_df = df_results[df_results['Score'] >= min_score].sort_values("Score", ascending=False)

    st.dataframe(filtered_df[show_cols], use_container_width=True)
    
    # Optional Debug (Uncomment to see column names if error returns)
    # st.write("Available Columns:", df.columns.tolist())
else:
    st.warning("No data found. Please check your GitHub CSV files.")
