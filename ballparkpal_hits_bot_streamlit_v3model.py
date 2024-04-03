import streamlit as st
import pandas as pd
import requests
from io import StringIO

# Set Streamlit app to always run in wide mode
st.set_page_config(layout="wide")

# Function to filter batters
def filter_batters(df, excluded_batters):
    return df[~df['Batter'].isin(excluded_batters)]

# URLs to the CSV files hosted on GitHub
url_probabilities = 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv'
url_percent_change = 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'

# Download CSV files from GitHub
response_probabilities = requests.get(url_probabilities)
response_percent_change = requests.get(url_percent_change)

# Read CSV Data
if response_probabilities.status_code == 200:
    df_probabilities = pd.read_csv(StringIO(response_probabilities.text))
else:
    st.error("Failed to download probabilities CSV file. Check the URL.")

if response_percent_change.status_code == 200:
    df_percent_change = pd.read_csv(StringIO(response_percent_change.text))
else:
    st.error("Failed to download percent change CSV file. Check the URL.")

# Merge DataFrames based on 'Batter' column
df_merged = pd.merge(df_probabilities, df_percent_change, on='Batter', suffixes=('_prob', '_change'))

# Define weights
weights = {
    '1B_prob': 0.5,
    'K_prob': -0.3,
    'BB_prob': -0.2,
    'XB_prob': 0.4,
    'vs_prob': 0.3,
    '1B_change': 0.5,
    'K_change': -0.3,
    'BB_change': -0.2,
    'XB_change': 0.4,
    'vs_change': 0.3,
    'HR_prob': 0.2,
    'HR_change': 0.2
}

# Function to update overall score
def update_overall_score(df):
    return (
        df['1B_prob'] * weights['1B_prob'] +
        df['K_prob'] * weights['K_prob'] +
        df['BB_prob'] * weights['BB_prob'] +
        df['XB_prob'] * weights['XB_prob'] +
        df['vs_prob'] * weights['vs_prob'] +
        df['1B_change'] * weights['1B_change'] +
        df['K_change'] * weights['K_change'] +
        df['BB_change'] * weights['BB_change'] +
        df['XB_change'] * weights['XB_change'] +
        df['vs_change'] * weights['vs_change'] +
        df['HR_prob'] * weights['HR_prob'] +
        df['HR_change'] * weights['HR_change']
    )

# Calculate overall quality score
df_merged['Overall Score'] = update_overall_score(df_merged)

# Streamlit UI
st.title('A1PICKS HITS BOT ALPHA')

# Display image at the top
image_url = 'https://example.com/your_image.jpg'  # Replace with your image URL
st.image(image_url, use_column_width=True)

st.write('The algorithm selectively extracts high-quality data from BallparkPals Batter versus Pitcher (BvP) Matchups, leveraging comprehensive BvP models to provide an overarching assessment. '
         'Additionally, it assigns a performance score aimed at forecasting the probability of a base hit.  '
         'Its imperative to note that while these results offer valuable insights, they should not be interpreted in isolation. '
         'Personal judgment, recent performance trends, and crucially, historical data against specific pitchers, as well as against left- and right-handed pitchers, must be considered for a comprehensive analysis. ')
         
# Exclude player input
excluded_player = st.text_input("Enter a batter's name to exclude them:")

if excluded_player:
    if excluded_player.upper() in df_merged['Batter'].str.upper().values:
        if st.button("Exclude Player"):
            df_merged = filter_batters(df_merged, [excluded_player.upper()])
            df_merged['Overall Score'] = update_overall_score(df_merged)
            st.success(f"{excluded_player} excluded successfully!")
    else:
        st.warning("Batter not found. Please try again.")

# Display top players
st.subheader("Top 15 Players based on Combined Data:")
top_15_players = df_merged[(df_merged['K_prob'] <= 16) & (df_merged['BB_prob'] <= 16)].sort_values(by='Overall Score', ascending=False).head(20)
st.write(top_15_players[['Batter', '1B_prob', 'K_prob', 'BB_prob', 'XB_prob', 'vs_prob', 'HR_prob',
                          '1B_change', 'K_change', 'BB_change', 'XB_change', 'vs_change', 'HR_change',
                          'RC_prob', 'RC_change', 'Overall Score']])

# Display additional information
st.write('Made With ♡ By FADED')
