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
def update_overall_scor
