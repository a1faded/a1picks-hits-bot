import streamlit as st
import pandas as pd
import requests
from io import StringIO
import os
import datetime

# Set Streamlit app to always run in wide mode
st.set_page_config(layout="wide")

# Function to filter batters
def filter_batters(df, excluded_batters):
    return df[~df['Batter'].isin(excluded_batters)]

# Function to get the last modified time of the Python script
def get_last_modified_time():
    script_path = os.path.realpath(__file__)
    return datetime.datetime.fromtimestamp(os.path.getmtime(script_path))

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

# Display image at the top with medium size
image_url = 'https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true'  # Replace with your image URL
st.image(image_url, width=250)  # Adjust width as needed

# Display last modified time in 24-hour time format
last_modified_time = get_last_modified_time()
formatted_last_modified_time = last_modified_time.strftime("%Y-%m-%d %H:%M:%S")
st.write(f"Last modified time: {formatted_last_modified_time}")

# Inject CSS styles
css = """
<style>
body {
  background-color: #333;
  display: flex;
  justify-content:center;
  align-items: center;
  height: 100vh;
}

.text-rainbow-animation {
  font-family:arial black;
  font-size:30px; /* Medium size */
  background-image: 
    linear-gradient(to right, red,orange,yellow,green,blue,indigo,violet, red); 
  -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;  
  animation: rainbow-animation 35s linear infinite;
}

@keyframes rainbow-animation {
    to {
        background-position: 4500vh;
    }
}
</style>
"""

st.markdown(css, unsafe_allow_html=True)

st.markdown("""<p style='color:green; font-weight:bold;'>The algorithm selectively extracts high-quality data from BallparkPals Batter versus Pitcher (BvP) Matchups, leveraging comprehensive BvP models to provide an overarching assessment. 
         Additionally, it assigns a performance score aimed at forecasting the probability of a base hit.  
         It's imperative to note that while these results offer valuable insights, they should not be interpreted in isolation. 
         Personal judgment, recent performance trends, and crucially, historical data against specific pitchers, as well as against left- and right-handed pitchers, must be considered for a comprehensive analysis.</p>""", unsafe_allow_html=True)

# Prompt for player input
#player_input = st.text_input("Enter a baseball player's name if they are not playing (BETA MAY NOT WORK AS INTENDED) :")

# Check if player input is provided
#if player_input:
    # Extract player name
    #player_name = player_input.strip().upper()
    
    # Check if the player is in the DataFrame
    #if player_name in df_merged['Batter'].str.upper().values:
        # Find another player within predefined parameters
       # replacement_player = df_merged[df_merged['Batter'].str.upper() != player_name].sample(n=1)
        
        # Exclude the original player and include the replacement player
       # df_merged = filter_batters(df_merged, [player_name])
       # df_merged = pd.concat([df_merged, replacement_player])
        #df_merged['Overall Score'] = update_overall_score(df_merged)
        
        # Sort DataFrame by 'Overall Score' in descending order
        #df_merged = df_merged.sort_values(by='Overall Score', ascending=False)
        
        # Display updated DataFrame
       # st.success(f"{player_name} substituted successfully with {replacement_player['Batter'].values[0]}!")
       # st.subheader("Updated Top Players based on Combined Data:")
       # st.write(df_merged[['Batter', '1B_prob', 'K_prob', 'BB_prob', 'XB_prob', 'vs_prob', 'HR_prob',
                         #   '1B_change', 'K_change', 'BB_change', #'XB_change', 'vs_change', 'HR_change',
                          #  'RC_prob', 'RC_change', 'Overall Score']].head(15))
   # else:
      #  st.warning("Player not found. Please try again.")

# Add buttons for selecting result type
option = st.radio("Choose result type:", ("STRICT RESULTS (DEFAULT)", "WIDER RESULTS (MORE OPTIONS BUT MAY INTRODUCE MORE BAD PICKS OR DNP PLAYERS)"))

st.markdown("""<p style='color:red; font-weight:bold;'>"Utilizing the 'WIDER FILTER' provides more options, but it may yield different results from the default setting. With more parameters, there's an increase in data, potentially altering the overall percentage and introducing new variables. Currently, the hit rate ranges from 50% to 61.5%."</p>""", unsafe_allow_html=True)

# Set values for K_prob and BB_prob based on the selected option
if option == "STRICT RESULTS (DEFAULT)":
    K_prob = 15.5
    BB_prob = 15.5
else:
    K_prob = 17.5
    BB_prob = 17.5

# Add buttons for selecting number of top players
num_players_option = st.radio("Choose number of top players:", ("TOP 5", "TOP 10", "TOP 15"))

# Set value for number of top players based on the selected option
if num_players_option == "TOP 5":
    num_players = 5
elif num_players_option == "TOP 10":
    num_players = 10
else:
    num_players = 15

# Filter DataFrame based on the selected options
top_players = df_merged[(df_merged['K_prob'] <= K_prob) & (df_merged['BB_prob'] <= BB_prob)].sort_values(by='Overall Score', ascending=False).head(num_players)

# Display the filtered DataFrame
st.subheader("Top Players based on Combined Data:")
st.write(top_players[['Batter', '1B_prob', 'K_prob', 'BB_prob', 'XB_prob', 'vs_prob', 'HR_prob',
                      '1B_change', 'K_change', 'BB_change', 'XB_change', 'vs_change', 'HR_change',
                      'RC_prob', 'RC_change', 'Overall Score']])

# Display additional information with rainbow animation
st.markdown("""<p class='text-rainbow-animation' style='font-weight:bold;'>Made With ♡ By FADED</p>""", unsafe_allow_html=True)
