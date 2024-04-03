import streamlit as st
import pandas as pd
import requests
from io import BytesIO

# Function to filter batters
def filter_batters(df, excluded_batters):
    return df[~df['Batter'].isin(excluded_batters)]

# URL to the Excel files hosted on the web host
url_probabilities = 'https://cdn.discordapp.com/attachments/779522546893586443/1224983971453210635/Ballpark_Pal.xlsx?ex=661f7a39&is=660d0539&hm=7e4298c10acba6f4e3720a9dd8d879815bd191be33a51ff375ac50f5460f12bf&'
url_percent_change = 'https://cdn.discordapp.com/attachments/779522546893586443/1224983971071397888/Ballpark_Palmodel2.xlsx?ex=661f7a39&is=660d0539&hm=9b15b3fad75e92c412330d605bf46847116d6b3093001945cbd619e842324e3f&'

# Download Excel files from the web host
response_probabilities = requests.get(url_probabilities)
response_percent_change = requests.get(url_percent_change)

# Read Excel Data
if response_probabilities.status_code == 200:
    with BytesIO(response_probabilities.content) as f:
        df_probabilities = pd.read_excel(f)
else:
    st.error("Failed to download probabilities Excel file. Check the URL.")

if response_percent_change.status_code == 200:
    with BytesIO(response_percent_change.content) as f:
        df_percent_change = pd.read_excel(f)
else:
    st.error("Failed to download percent change Excel file. Check the URL.")

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



# Export results to Excel
#if st.button("Export to Excel"):
    #desktop_path = os.path.join(os.path.expanduser("~"), "Desktop")
   # export_path = os.path.join(desktop_path, "mlb_top_players.xlsx")
    #top_15_players.to_excel(export_path, index=False)
    #st.success("Results exported successfully!")

st.write('Made With ♡ By FADED')
