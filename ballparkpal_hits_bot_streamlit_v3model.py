import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
import hashlib
import uuid

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro",
    layout="wide",
    page_icon="⚾",
    menu_items={
        'Get Help': 'mailto:your@email.com',
        'Report a bug': "https://github.com/yourrepo/issues",
    }
)

# Constants and Configuration
CONFIG = {
    'csv_urls': {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv'
    },
    'expected_columns': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
    'cache_ttl': 900,  # 15 minutes
}

# User Management Functions
def get_or_create_user_id():
    """Get or create a unique user identifier using browser session."""
    
    # Check if user already has an ID in session state
    if 'user_id' not in st.session_state:
        # Check if user wants to set a custom username
        if 'username_set' not in st.session_state:
            st.session_state.username_set = False
        
        if not st.session_state.username_set:
            # Show username setup
            st.sidebar.markdown("---")
            st.sidebar.markdown("### 👤 **User Setup**")
            
            username = st.sidebar.text_input(
                "Enter Your Username",
                placeholder="e.g., BaseballFan123",
                help="This keeps your picks private while contributing to community insights",
                key="user_setup_username_input"
            )
            
            if username:
                if st.sidebar.button("🔐 Set Username"):
                    # Create user ID from username
                    user_id = hashlib.md5(username.encode()).hexdigest()[:12]
                    st.session_state.user_id = user_id
                    st.session_state.username = username
                    st.session_state.username_set = True
                    st.sidebar.success(f"✅ Welcome, {username}!")
                    st.rerun()
            else:
                # Generate anonymous ID if no username provided
                if st.sidebar.button("🕵️ Continue Anonymously"):
                    anonymous_id = str(uuid.uuid4())[:12]
                    st.session_state.user_id = anonymous_id
                    st.session_state.username = f"User_{anonymous_id[:6]}"
                    st.session_state.username_set = True
                    st.rerun()
            
            return None  # Don't proceed until user is set up
    
    return st.session_state.user_id

def get_user_files(user_id):
    """Get file paths for individual user data."""
    return {
        'picks': f'user_{user_id}_picks.csv',
        'results': f'user_{user_id}_results.csv'
    }

def get_aggregate_files():
    """Get file paths for aggregate community data."""
    return {
        'community_results': 'community_verification_results.csv',
        'player_consensus': 'player_consensus_scores.csv'
    }

# Enhanced Player Tracking System
def initialize_user_tracking_system(user_id):
    """Initialize tracking system for specific user and community aggregates."""
    
    if not user_id:
        return
    
    user_files = get_user_files(user_id)
    aggregate_files = get_aggregate_files()
    
    # Initialize individual user files
    for file_type, filepath in user_files.items():
        if not os.path.exists(filepath):
            if file_type == 'picks':
                df = pd.DataFrame(columns=['Date', 'Player', 'Team', 'Opponent', 'Score', 'Hit_Prob', 'Status'])
            else:  # results
                df = pd.DataFrame(columns=['Date', 'Player', 'Team', 'Opponent', 'Predicted_Score', 'Got_Hit', 'Actual_Hits', 'Notes'])
            df.to_csv(filepath, index=False)
    
    # Initialize community aggregate files
    if not os.path.exists(aggregate_files['community_results']):
        community_df = pd.DataFrame(columns=[
            'Date', 'Player', 'Team', 'Opponent', 'User_ID', 'Username', 
            'Predicted_Score', 'Got_Hit', 'Actual_Hits', 'Verified_At'
        ])
        community_df.to_csv(aggregate_files['community_results'], index=False)
    
    if not os.path.exists(aggregate_files['player_consensus']):
        consensus_df = pd.DataFrame(columns=[
            'Player', 'Total_Verifications', 'Hit_Rate', 'Last_10_Games_Hit_Rate', 
            'Recent_Form', 'Form_Score_Adjustment', 'Last_Updated'
        ])
        consensus_df.to_csv(aggregate_files['player_consensus'], index=False)

def load_user_history(user_id):
    """Load historical performance data for specific user."""
    if not user_id:
        return pd.DataFrame()
    
    try:
        user_files = get_user_files(user_id)
        results_file = user_files['results']
        
        if os.path.exists(results_file):
            history_df = pd.read_csv(results_file)
            history_df['Date'] = pd.to_datetime(history_df['Date'])
            return history_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading user history: {e}")
        return pd.DataFrame()

def load_community_consensus():
    """Load community consensus data for all players."""
    try:
        aggregate_files = get_aggregate_files()
        consensus_file = aggregate_files['player_consensus']
        
        if os.path.exists(consensus_file):
            consensus_df = pd.read_csv(consensus_file)
            return consensus_df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading community consensus: {e}")
        return pd.DataFrame()

def calculate_community_form_adjustment(player_name, days_back=10):
    """Calculate form adjustment based on community verification data."""
    
    try:
        aggregate_files = get_aggregate_files()
        community_file = aggregate_files['community_results']
        
        if not os.path.exists(community_file):
            return 0
        
        community_df = pd.read_csv(community_file)
        community_df['Date'] = pd.to_datetime(community_df['Date'])
        
        # Get recent community verifications for this player
        cutoff_date = datetime.now() - timedelta(days=days_back)
        recent_verifications = community_df[
            (community_df['Player'] == player_name) & 
            (community_df['Date'] >= cutoff_date) &
            (community_df['Got_Hit'].notna())
        ]
        
        if len(recent_verifications) < 3:  # Need at least 3 community verifications
            return 0
        
        # Calculate community consensus hit rate
        community_hit_rate = recent_verifications['Got_Hit'].mean()
        league_avg_hit_rate = 0.25  # Approximate league average
        
        # Weight by number of verifications (more verifications = more confidence)
        verification_count = len(recent_verifications)
        confidence_multiplier = min(verification_count / 10, 1.0)  # Max confidence at 10+ verifications
        
        # Convert to adjustment factor (-15 to +15 points, scaled by confidence)
        base_adjustment = (community_hit_rate - league_avg_hit_rate) * 60
        final_adjustment = base_adjustment * confidence_multiplier
        
        return max(-15, min(15, final_adjustment))
        
    except Exception as e:
        return 0

def save_user_picks(user_id, selected_players, filtered_df):
    """Save selected players as today's picks for specific user."""
    if not user_id:
        return 0
    
    try:
        user_files = get_user_files(user_id)
        picks_file = user_files['picks']
        today = datetime.now().strftime('%Y-%m-%d')
        
        # Load existing user picks
        if os.path.exists(picks_file):
            existing_picks = pd.read_csv(picks_file)
            # Remove today's picks if they exist (allow overwriting)
            existing_picks = existing_picks[existing_picks['Date'] != today]
        else:
            existing_picks = pd.DataFrame()
        
        # Create new picks data
        new_picks = []
        for player_name in selected_players:
            player_data = filtered_df[filtered_df['Batter'] == player_name]
            if not player_data.empty:
                player_row = player_data.iloc[0]
                new_picks.append({
                    'Date': today,
                    'Player': player_name,
                    'Team': player_row['Tm'],
                    'Opponent': player_row['Pitcher'],
                    'Score': player_row['Score'],
                    'Hit_Prob': player_row['total_hit_prob'],
                    'Status': 'Pending'
                })
        
        if new_picks:
            new_picks_df = pd.DataFrame(new_picks)
            updated_picks = pd.concat([existing_picks, new_picks_df], ignore_index=True)
            updated_picks.to_csv(picks_file, index=False)
            return len(new_picks)
        return 0
    except Exception as e:
        st.error(f"Error saving picks: {e}")
        return 0

def load_user_pending_picks(user_id):
    """Load picks that need result verification for specific user."""
    if not user_id:
        return pd.DataFrame()
    
    try:
        user_files = get_user_files(user_id)
        picks_file = user_files['picks']
        
        if os.path.exists(picks_file):
            picks_df = pd.read_csv(picks_file)
            pending_picks = picks_df[picks_df['Status'] == 'Pending'].copy()
            pending_picks['Date'] = pd.to_datetime(pending_picks['Date'])
            return pending_picks.sort_values('Date', ascending=False)
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error loading user picks: {e}")
        return pd.DataFrame()

def update_user_pick_results(user_id, pick_index, got_hit, actual_hits=None, notes=""):
    """Update results for specific user pick AND contribute to community data."""
    if not user_id:
        return False
    
    try:
        user_files = get_user_files(user_id)
        aggregate_files = get_aggregate_files()
        
        picks_file = user_files['picks']
        results_file = user_files['results']
        community_file = aggregate_files['community_results']
        
        # Load user picks
        picks_df = pd.read_csv(picks_file)
        pick_row = picks_df.iloc[pick_index].copy()
        
        # Update user pick status
        picks_df.loc[pick_index, 'Status'] = 'Verified'
        picks_df.to_csv(picks_file, index=False)
        
        # Add to user results
        if os.path.exists(results_file):
            results_df = pd.read_csv(results_file)
        else:
            results_df = pd.DataFrame()
        
        new_result = {
            'Date': pick_row['Date'],
            'Player': pick_row['Player'],
            'Team': pick_row['Team'],
            'Opponent': pick_row['Opponent'],
            'Predicted_Score': pick_row['Score'],
            'Got_Hit': got_hit,
            'Actual_Hits': actual_hits if actual_hits is not None else (1 if got_hit else 0),
            'Notes': notes
        }
        
        new_results_df = pd.concat([results_df, pd.DataFrame([new_result])], ignore_index=True)
        new_results_df.to_csv(results_file, index=False)
        
        # CONTRIBUTE TO COMMUNITY DATA
        if os.path.exists(community_file):
            community_df = pd.read_csv(community_file)
        else:
            community_df = pd.DataFrame()
        
        community_result = {
            'Date': pick_row['Date'],
            'Player': pick_row['Player'],
            'Team': pick_row['Team'],
            'Opponent': pick_row['Opponent'],
            'User_ID': user_id,
            'Username': st.session_state.get('username', f'User_{user_id[:6]}'),
            'Predicted_Score': pick_row['Score'],
            'Got_Hit': got_hit,
            'Actual_Hits': actual_hits if actual_hits is not None else (1 if got_hit else 0),
            'Verified_At': datetime.now().isoformat()
        }
        
        new_community_df = pd.concat([community_df, pd.DataFrame([community_result])], ignore_index=True)
        new_community_df.to_csv(community_file, index=False)
        
        # Update player consensus scores
        update_player_consensus()
        
        return True
    except Exception as e:
        st.error(f"Error updating results: {e}")
        return False

def update_player_consensus():
    """Update aggregate player consensus scores based on all community verifications."""
    try:
        aggregate_files = get_aggregate_files()
        community_file = aggregate_files['community_results']
        consensus_file = aggregate_files['player_consensus']
        
        if not os.path.exists(community_file):
            return
        
        community_df = pd.read_csv(community_file)
        community_df['Date'] = pd.to_datetime(community_df['Date'])
        
        # Calculate consensus for each player
        consensus_data = []
        
        for player in community_df['Player'].unique():
            player_data = community_df[community_df['Player'] == player]
            
            # Overall stats
            total_verifications = len(player_data)
            hit_rate = player_data['Got_Hit'].mean()
            
            # Recent form (last 10 games/verifications)
            recent_data = player_data.nlargest(10, 'Date')
            recent_hit_rate = recent_data['Got_Hit'].mean() if len(recent_data) > 0 else hit_rate
            
            # Determine form
            if recent_hit_rate > hit_rate + 0.1:  # 10% better than overall
                recent_form = "🔥 Hot"
                form_adjustment = min(15, (recent_hit_rate - 0.25) * 60)
            elif recent_hit_rate < hit_rate - 0.1:  # 10% worse than overall
                recent_form = "❄️ Cold"
                form_adjustment = max(-15, (recent_hit_rate - 0.25) * 60)
            else:
                recent_form = "➡️ Normal"
                form_adjustment = 0
            
            consensus_data.append({
                'Player': player,
                'Total_Verifications': total_verifications,
                'Hit_Rate': round(hit_rate, 3),
                'Last_10_Games_Hit_Rate': round(recent_hit_rate, 3),
                'Recent_Form': recent_form,
                'Form_Score_Adjustment': round(form_adjustment, 1),
                'Last_Updated': datetime.now().isoformat()
            })
        
        # Save consensus data
        consensus_df = pd.DataFrame(consensus_data)
        consensus_df.to_csv(consensus_file, index=False)
        
    except Exception as e:
        st.error(f"Error updating player consensus: {e}")

def get_community_stats():
    """Get overall community tracking statistics."""
    try:
        aggregate_files = get_aggregate_files()
        community_file = aggregate_files['community_results']
        
        if not os.path.exists(community_file):
            return {
                'total_verifications': 0,
                'active_users': 0,
                'top_tracked_players': [],
                'community_hit_rate': 0
            }
        
        community_df = pd.read_csv(community_file)
        
        # Calculate stats
        total_verifications = len(community_df)
        active_users = community_df['User_ID'].nunique()
        community_hit_rate = community_df['Got_Hit'].mean() if len(community_df) > 0 else 0
        
        # Top tracked players
        player_counts = community_df['Player'].value_counts().head(5)
        top_tracked_players = [f"{player} ({count} verifications)" for player, count in player_counts.items()]
        
        return {
            'total_verifications': total_verifications,
            'active_users': active_users,
            'top_tracked_players': top_tracked_players,
            'community_hit_rate': community_hit_rate
        }
        
    except Exception as e:
        return {
            'total_verifications': 0,
            'active_users': 0,
            'top_tracked_players': [],
            'community_hit_rate': 0
        }

# Custom CSS with enhanced styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 2rem;
    }
    .sidebar .sidebar-content {
        padding-top: 2.5rem;
    }
    .stRadio [role=radiogroup] {
        align-items: center;
        gap: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .warning-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .community-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        border: 2px solid #ffd700;
    }
    .color-legend {
        margin: 1rem 0;
        padding: 1rem;
        background: #000000;
        border-radius: 8px;
        color: white !important;
    }
    .hit-probability {
        font-size: 1.2em;
        font-weight: bold;
        color: #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Enhanced Data Loading with Error Handling and Validation
@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_csv_with_validation(url, description, expected_columns):
    """Load and validate CSV data with comprehensive error handling."""
    try:
        with st.spinner(f'Loading {description}...'):
            response = requests.get(url, timeout=15)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text))
            
            # Validate structure
            if df.empty:
                st.error(f"❌ {description}: No data found")
                return None
            
            # Check for required columns
            missing_cols = set(expected_columns) - set(df.columns)
            if missing_cols:
                st.error(f"❌ {description}: Missing columns {missing_cols}")
                return None
            
            # Validate key columns have no nulls
            key_cols = ['Tm', 'Batter', 'Pitcher']
            null_counts = df[key_cols].isnull().sum()
            if null_counts.any():
                problematic_cols = null_counts[null_counts > 0].index.tolist()
                st.error(f"❌ {description}: Null values in {problematic_cols}")
                return None
            
            st.success(f"✅ {description}: {len(df)} records loaded")
            return df
            
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Network error loading {description}: {str(e)}")
        return None
    except Exception as e:
        st.error(f"❌ Error processing {description}: {str(e)}")
        return None

def validate_merge_quality(prob_df, pct_df, merged_df):
    """Validate the quality of the merge operation."""
    original_count = len(prob_df)
    merged_count = len(merged_df)
    
    if merged_count < original_count:
        lost_records = original_count - merged_count
        st.warning(f"⚠️ Lost {lost_records} records during merge ({lost_records/original_count*100:.1f}%)")
    
    # Check for duplicates
    merge_keys = ['Tm', 'Batter', 'Pitcher']
    duplicates = merged_df.duplicated(subset=merge_keys).sum()
    if duplicates > 0:
        st.error(f"🔴 Found {duplicates} duplicate matchups after merge")
    
    return merged_df

@st.cache_data(ttl=CONFIG['cache_ttl'])
def load_and_process_data():
    """Enhanced data loading and processing with validation."""
    
    # Load both CSV files with enhanced validation
    prob_df = load_csv_with_validation(
        CONFIG['csv_urls']['probabilities'], 
        "Base Probabilities", 
        CONFIG['expected_columns']
    )
    
    pct_df = load_csv_with_validation(
        CONFIG['csv_urls']['percent_change'], 
        "Adjustment Factors", 
        CONFIG['expected_columns']
    )
    
    if prob_df is None or pct_df is None:
        st.error("❌ Failed to load required data files")
        return None
    
    # Enhanced merge with validation
    try:
        merged_df = pd.merge(
            prob_df, pct_df,
            on=['Tm', 'Batter', 'Pitcher'],
            suffixes=('_prob', '_pct'),
            how='inner'
        )
        
        merged_df = validate_merge_quality(prob_df, pct_df, merged_df)
        
    except Exception as e:
        st.error(f"❌ Failed to merge datasets: {str(e)}")
        return None
    
    # Calculate adjusted metrics
    metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
    
    for metric in metrics:
        if metric in ['K', 'BB']:
            base_col = f'{metric}.1'
        else:
            base_col = f'{metric}_prob'
            
        pct_col = f'{metric}_pct'
        
        if base_col in merged_df.columns and pct_col in merged_df.columns:
            merged_df[f'adj_{metric}'] = merged_df[base_col] * (1 + merged_df[pct_col]/100)
            
            if metric in ['K', 'BB']:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            elif metric in ['1B', 'XB', 'HR']:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0, upper=100)
            else:
                merged_df[f'adj_{metric}'] = merged_df[f'adj_{metric}'].clip(lower=0)
                
            st.success(f"✅ Created adj_{metric} using {base_col} and {pct_col}")
        else:
            st.error(f"❌ Missing columns for {metric}: {base_col} or {pct_col}")
            if metric in ['K', 'BB']:
                merged_df[f'adj_{metric}'] = 20
            else:
                merged_df[f'adj_{metric}'] = 0
    
    # Calculate total base hit probability
    merged_df['total_hit_prob'] = merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
    merged_df['total_hit_prob'] = merged_df['total_hit_prob'].clip(upper=100)
    
    return merged_df

def calculate_league_aware_scores(df, user_id):
    """Enhanced scoring algorithm with community intelligence."""
    
    # League averages for 2024
    LEAGUE_K_AVG = 22.6
    LEAGUE_BB_AVG = 8.5
    
    # Base weighted score
    weights = {
        'adj_1B': 2.0,
        'adj_XB': 1.8,
        'adj_vs': 1.2,
        'adj_RC': 0.8,
        'adj_HR': 0.6,
        'adj_K': -2.5,
        'adj_BB': -0.6
    }
    
    df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
    
    # League-aware bonuses
    df['elite_contact_bonus'] = np.where(df['adj_K'] <= 12.0, 10, 0)
    df['aggressive_contact_bonus'] = np.where((df['adj_K'] <= 17.0) & (df['adj_BB'] <= 6.0), 8, 0)
    df['hit_prob_bonus'] = np.where(df['total_hit_prob'] > 40, 5, 0)
    df['matchup_bonus'] = np.where(df['adj_vs'] > 5, 3, 0)
    df['league_superior_bonus'] = np.where((df['adj_K'] < LEAGUE_K_AVG) & (df['adj_BB'] < LEAGUE_BB_AVG), 6, 0)
    
    # COMMUNITY FORM ADJUSTMENT (replaces individual tracking)
    df['community_form_adjustment'] = df['Batter'].apply(
        lambda player: calculate_community_form_adjustment(player)
    )
    
    # Add form indicators and verification counts
    consensus_df = load_community_consensus()
    
    def get_form_indicator(player_name):
        if consensus_df.empty:
            return "➡️"
        player_consensus = consensus_df[consensus_df['Player'] == player_name]
        if player_consensus.empty:
            return "➡️"
        return player_consensus.iloc[0]['Recent_Form'].split()[0]
    
    def get_verification_count(player_name):
        if consensus_df.empty:
            return 0
        player_consensus = consensus_df[consensus_df['Player'] == player_name]
        if player_consensus.empty:
            return 0
        return player_consensus.iloc[0]['Total_Verifications']
    
    df['form_indicator'] = df['Batter'].apply(get_form_indicator)
    df['community_verifications'] = df['Batter'].apply(get_verification_count)
    
    # Calculate final score
    df['Score'] = (df['base_score'] + 
                   df['elite_contact_bonus'] + 
                   df['aggressive_contact_bonus'] + 
                   df['hit_prob_bonus'] + 
                   df['matchup_bonus'] + 
                   df['league_superior_bonus'] +
                   df['community_form_adjustment'])
    
    # Normalize to 0-100 scale
    if df['Score'].max() != df['Score'].min():
        df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    else:
        df['Score'] = 50
    
    return df.round(1)

def create_league_aware_filters(df=None):
    """Create baseball-intelligent filtering system."""
    st.sidebar.header("🎯 Baseball-Smart Filters")
    
    # Initialize session state for exclusions
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []
    
    # Handle clear exclusions command
    if 'clear_exclusions' in st.session_state and st.session_state.clear_exclusions:
        st.session_state.excluded_players = []
        st.session_state.clear_exclusions = False
    
    # League averages for 2024
    LEAGUE_K_AVG = 22.6
    LEAGUE_BB_AVG = 8.5
    
    filters = {}
    
    # Show league context
    st.sidebar.markdown("### **📊 2024 League Context**")
    st.sidebar.markdown(f"""
    - **K% League Avg**: {LEAGUE_K_AVG}%
    - **BB% League Avg**: {LEAGUE_BB_AVG}%
    """)
    
    if df is not None and not df.empty:
        st.sidebar.markdown(f"**📈 Today's Pool:** {len(df)} matchups")
        avg_k = df['adj_K'].mean() if 'adj_K' in df.columns else 0
        avg_bb = df['adj_BB'].mean() if 'adj_BB' in df.columns else 0
        st.sidebar.markdown(f"**Today's Avg K%:** {avg_k:.1f}%")
        st.sidebar.markdown(f"**Today's Avg BB%:** {avg_bb:.1f}%")
    
    st.sidebar.markdown("---")
    
    # PRIMARY FILTER: Player Type Selection
    st.sidebar.markdown("### **🎯 Player Type Focus**")
    
    player_type_options = {
        "🏆 Contact-Aggressive Hitters": {
            'description': "Low K% + Low BB% (Elite for base hits)",
            'max_k': 17.0,
            'max_bb': 6.0,
            'min_hit_prob': 35
        },
        "⭐ Elite Contact Specialists": {
            'description': "Ultra-low K% (Pure contact)",
            'max_k': 12.0,
            'max_bb': 8.5,
            'min_hit_prob': 30
        },
        "⚡ Swing-Happy Hitters": {
            'description': "Ultra-low BB% (Aggressive approach)",
            'max_k': 22.6,
            'max_bb': 4.0,
            'min_hit_prob': 32
        },
        "🔷 Above-Average Contact": {
            'description': "Better than league average K%",
            'max_k': 17.0,
            'max_bb': 10.0,
            'min_hit_prob': 28
        },
        "🌐 All Players": {
            'description': "No K% or BB% restrictions",
            'max_k': 100,
            'max_bb': 100,
            'min_hit_prob': 20
        }
    }
    
    selected_type = st.sidebar.selectbox(
        "Choose Hitter Profile",
        options=list(player_type_options.keys()),
        index=0,
        help="Each profile targets different hitting approaches based on league averages"
    )
    
    # Apply selected player type settings
    type_config = player_type_options[selected_type]
    filters['max_k'] = type_config['max_k']
    filters['max_bb'] = type_config['max_bb']
    filters['min_hit_prob'] = type_config['min_hit_prob']
    
    # Show what this means
    st.sidebar.markdown(f"**📋 {selected_type}**")
    st.sidebar.markdown(f"*{type_config['description']}*")
    st.sidebar.markdown(f"- Max K%: {filters['max_k']:.1f}%")
    st.sidebar.markdown(f"- Max BB%: {filters['max_bb']:.1f}%")
    st.sidebar.markdown(f"- Min Hit Prob: {filters['min_hit_prob']}%")
    
    # ADVANCED OPTIONS
    with st.sidebar.expander("⚙️ Fine-Tune Filters"):
        
        # Ensure values are single numbers
        max_k_value = max(5.0, min(35.0, float(filters['max_k'])))
        max_bb_value = max(2.0, min(15.0, float(filters['max_bb'])))
        
        # Custom thresholds
        filters['custom_max_k'] = st.slider(
            "Custom Max K% Override",
            min_value=5.0,
            max_value=35.0,
            value=max_k_value,
            step=0.5,
            help=f"League avg: {LEAGUE_K_AVG}% | Elite: ≤12.0%"
        )
        
        filters['custom_max_bb'] = st.slider(
            "Custom Max BB% Override",
            min_value=2.0,
            max_value=15.0,
            value=max_bb_value,
            step=0.5,
            help=f"League avg: {LEAGUE_BB_AVG}% | Aggressive: ≤4.0%"
        )
        
        # Use custom values if different
        if filters['custom_max_k'] != filters['max_k']:
            filters['max_k'] = filters['custom_max_k']
        if filters['custom_max_bb'] != filters['max_bb']:
            filters['max_bb'] = filters['custom_max_bb']
        
        # vs Pitcher Rating
        filters['min_vs_pitcher'] = st.slider(
            "vs Pitcher Rating",
            min_value=-10,
            max_value=10,
            value=0,
            step=1,
            help="Matchup advantage/disadvantage vs this pitcher type"
        )
        
        # Team Selection
        team_options = []
        if df is not None and not df.empty:
            team_options = sorted(df['Tm'].unique().tolist())
        
        filters['selected_teams'] = st.multiselect(
            "Filter by Teams",
            options=team_options,
            help="Leave empty to include all teams"
        )
        
        # Result count
        filters['result_count'] = st.selectbox(
            "Number of Results",
            options=[5, 10, 15, 20, 25, 30, "All"],
            index=2,
            help="Choose how many results to display, or 'All' to show everyone"
        )
    
    # LINEUP STATUS MANAGEMENT
    with st.sidebar.expander("🏟️ Lineup Status Management"):
        st.markdown("**Exclude players not in today's lineups:**")
        
        # Get list of all players
        all_players = []
        if df is not None and not df.empty:
            all_players = sorted(df['Batter'].unique().tolist())
        
        # Use session state for exclusions
        current_exclusions = st.session_state.excluded_players.copy()
        
        selected_exclusions = st.multiselect(
            "Players NOT Playing Today",
            options=all_players,
            default=current_exclusions,
            help="Select players who are confirmed out of lineups",
            key="lineup_exclusions"
        )
        
        # Update session state
        st.session_state.excluded_players = selected_exclusions
        filters['excluded_players'] = selected_exclusions
        
        # Show current exclusion count
        if selected_exclusions:
            st.info(f"🚫 Currently excluding {len(selected_exclusions)} players")
        
        # Quick clear button
        if st.button("🔄 Clear All Exclusions", key="sidebar_clear"):
            st.session_state.excluded_players = []
            st.rerun()
    
    # REAL-TIME FEEDBACK
    if df is not None and not df.empty:
        try:
            # Apply exclusions
            preview_df = df.copy()
            excluded_players = st.session_state.excluded_players
            if excluded_players:
                preview_df = preview_df[~preview_df['Batter'].isin(excluded_players)]
            
            preview_query = f"adj_K <= {filters['max_k']:.1f} and adj_BB <= {filters['max_bb']:.1f} and total_hit_prob >= {filters['min_hit_prob']}"
            preview_df = preview_df.query(preview_query)
            matching_count = len(preview_df)
            excluded_count = len(excluded_players)
            
            # Show feedback
            if matching_count == 0:
                st.sidebar.error("❌ No players match current profile")
            elif matching_count < 5:
                st.sidebar.warning(f"⚠️ Only {matching_count} players match")
            else:
                st.sidebar.success(f"✅ {matching_count} players match profile")
                
                if excluded_count > 0:
                    st.sidebar.markdown(f"**📊 Lineup Status:** {matching_count} confirmed, {excluded_count} excluded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Preview unavailable")
    
    return filters

def apply_league_aware_filters(df, filters):
    """Apply filtering logic."""
    
    if df is None or df.empty:
        return df
    
    # Exclude players not in lineups
    excluded_players = st.session_state.get('excluded_players', [])
    if excluded_players:
        excluded_count = len(df[df['Batter'].isin(excluded_players)])
        df = df[~df['Batter'].isin(excluded_players)]
        if excluded_count > 0:
            st.info(f"🏟️ Excluded {excluded_count} players not in today's lineups")
    
    query_parts = []
    
    # Primary filters
    if 'max_k' in filters and filters['max_k'] < 100:
        query_parts.append(f"adj_K <= {filters['max_k']:.1f}")
    
    if 'max_bb' in filters and filters['max_bb'] < 100:
        query_parts.append(f"adj_BB <= {filters['max_bb']:.1f}")
    
    if 'min_hit_prob' in filters and filters['min_hit_prob'] > 0:
        query_parts.append(f"total_hit_prob >= {filters['min_hit_prob']:.1f}")
    
    # Advanced filters
    if 'min_vs_pitcher' in filters and filters['min_vs_pitcher'] != 0:
        query_parts.append(f"adj_vs >= {filters['min_vs_pitcher']}")
    
    # Team filter
    if filters.get('selected_teams'):
        team_filter = "Tm in " + str(filters['selected_teams'])
        query_parts.append(team_filter)
    
    # Apply filters
    try:
        if query_parts:
            full_query = " and ".join(query_parts)
            filtered_df = df.query(full_query)
        else:
            filtered_df = df
        
        # Sort and limit results
        result_count = filters.get('result_count', 15)
        
        if result_count == "All":
            result_df = filtered_df.sort_values('Score', ascending=False)
        else:
            result_df = filtered_df.sort_values('Score', ascending=False).head(result_count)
        
        return result_df
        
    except Exception as e:
        st.error(f"❌ Filter error: {str(e)}")
        result_count = filters.get('result_count', 15)
        
        if result_count == "All":
            return df.sort_values('Score', ascending=False)
        else:
            return df.sort_values('Score', ascending=False).head(result_count)

def create_league_aware_header():
    """Create enhanced header."""
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.image('https://github.com/a1faded/a1picks-hits-bot/blob/main/a1sports.png?raw=true', 
                width=200)
    
    with col2:
        st.title("🎯 MLB Community-Powered Hit Predictor Pro")
        st.markdown("*Powered by community intelligence and 2024 league context*")

def create_data_quality_dashboard(df, user_id):
    """Display data quality metrics and community stats."""
    if df is None or df.empty:
        st.error("No data available for quality analysis")
        return
    
    st.subheader("📊 Today's Data Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📈 Total Matchups</h3>
            <h2>{}</h2>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        unique_batters = df['Batter'].nunique()
        st.markdown("""
        <div class="metric-card">
            <h3>👤 Unique Batters</h3>
            <h2>{}</h2>
        </div>
        """.format(unique_batters), unsafe_allow_html=True)
    
    with col3:
        avg_hit_prob = df['total_hit_prob'].mean()
        st.markdown("""
        <div class="success-card">
            <h3>🎯 Avg Hit Probability</h3>
            <h2>{:.1f}%</h2>
        </div>
        """.format(avg_hit_prob), unsafe_allow_html=True)
    
    with col4:
        # Community stats
        community_stats = get_community_stats()
        st.markdown("""
        <div class="community-card">
            <h3>👥 Community Power</h3>
            <h2>{}</h2>
            <small>{} active users</small>
        </div>
        """.format(community_stats['total_verifications'], community_stats['active_users']), unsafe_allow_html=True)

def display_league_aware_results(filtered_df, filters):
    """Display results with community intelligence."""
    
    LEAGUE_K_AVG = 22.6
    LEAGUE_BB_AVG = 8.5
    
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters")
        return
    
    # Display header
    result_count = filters.get('result_count', 15)
    if result_count == "All":
        st.subheader(f"🎯 All {len(filtered_df)} Base Hit Candidates")
    else:
        st.subheader(f"🎯 Top {len(filtered_df)} Base Hit Candidates")
    
    # Enhanced key insights
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        st.markdown(f"""
        <div class="success-card">
            <h4>🥇 Best Hit Probability</h4>
            <h2>{best_hit_prob:.1f}%</h2>
            <small>Target: 35%+</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        avg_k = filtered_df['adj_K'].mean()
        k_vs_league = avg_k - LEAGUE_K_AVG
        color = "success-card" if k_vs_league < -5 else "metric-card"
        st.markdown(f"""
        <div class="{color}">
            <h4>⚾ Avg K% vs League</h4>
            <h2>{k_vs_league:+.1f}%</h2>
            <small>League: {LEAGUE_K_AVG}%</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Community-tracked players count
        community_tracked = (filtered_df['community_verifications'] > 0).sum()
        st.markdown(f"""
        <div class="community-card">
            <h4>👥 Community Tracked</h4>
            <h2>{community_tracked}/{len(filtered_df)}</h2>
            <small>Players with data</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        hot_players = (filtered_df['form_indicator'] == '🔥').sum()
        st.markdown(f"""
        <div class="success-card">
            <h4>🔥 Hot Streak Players</h4>
            <h2>{hot_players}</h2>
            <small>Community verified</small>
        </div>
        """, unsafe_allow_html=True)
    
    # Enhanced results table with community data
    display_columns = {
        'form_indicator': 'Form',
        'Batter': 'Batter', 
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'adj_K': 'K% vs League',
        'adj_BB': 'BB% vs League',
        'adj_vs': 'vs Pitcher',
        'community_verifications': 'Verifications',
        'Score': 'Score'
    }
    
    # Prepare display dataframe
    display_df = filtered_df.copy()
    display_df['K% vs League'] = display_df['adj_K'] - LEAGUE_K_AVG
    display_df['BB% vs League'] = display_df['adj_BB'] - LEAGUE_BB_AVG
    
    # Add lineup status
    excluded_players = st.session_state.get('excluded_players', [])
    display_df['Lineup_Status'] = display_df['Batter'].apply(
        lambda x: '🏟️' if x not in excluded_players else '❌'
    )
    
    display_columns_with_status = {'Lineup_Status': 'Status', **display_columns}
    styled_df = display_df[display_columns_with_status.keys()].rename(columns=display_columns_with_status)
    
    # Enhanced formatting
    styled_df = styled_df.style.format({
        'Hit Prob %': "{:.1f}%",
        'Contact %': "{:.1f}%", 
        'XB %': "{:.1f}%",
        'HR %': "{:.1f}%",
        'K% vs League': "{:+.1f}%",
        'BB% vs League': "{:+.1f}%",
        'vs Pitcher': "{:.0f}",
        'Verifications': "{:.0f}",
        'Score': "{:.1f}"
    }).background_gradient(
        subset=['Score'],
        cmap='RdYlGn',
        vmin=0,
        vmax=100
    ).background_gradient(
        subset=['Hit Prob %'],
        cmap='Greens',
        vmin=20,
        vmax=50
    ).background_gradient(
        subset=['Verifications'],
        cmap='Blues',
        vmin=0,
        vmax=20
    )
    
    st.dataframe(styled_df, use_container_width=True)
    
    # Enhanced interpretation guide
    st.markdown("""
    <div class="color-legend">
        <strong>📊 Enhanced Color Guide:</strong><br>
        <strong>Status:</strong> 🏟️ = Confirmed Playing | ❌ = Excluded from Lineups<br>
        <strong>Form:</strong> 🔥 = Community Hot Streak | ➡️ = Normal | ❄️ = Community Cold Streak<br>
        <strong>Verifications:</strong> Number of community users who have tracked this player<br>
        <strong>Score:</strong> <span style="color: #1a9641;">●</span> Elite (70+) | 
        <span style="color: #fdae61;">●</span> Good (50-69) | 
        <span style="color: #d7191c;">●</span> Risky (<50)
    </div>
    """, unsafe_allow_html=True)
    
    # League Context Analysis (same as before but with community data awareness)
    if len(filtered_df) >= 3:
        st.markdown("### 🔍 **Advanced League Context Analysis**")
        
        profile_criteria = {
            "🏆 Contact-Aggressive": {"max_k": 17.0, "max_bb": 6.0, "icon": "🏆"},
            "⭐ Elite Contact": {"max_k": 12.0, "max_bb": 8.5, "icon": "⭐"},
            "⚡ Swing-Happy": {"max_k": 22.6, "max_bb": 4.0, "icon": "⚡"},
            "🔷 Above-Average": {"max_k": 17.0, "max_bb": 10.0, "icon": "🔷"}
        }
        
        excluded_players = st.session_state.get('excluded_players', [])
        profile_analysis = {}
        
        for profile_name, criteria in profile_criteria.items():
            profile_players = filtered_df[
                (filtered_df['adj_K'] <= criteria['max_k']) & 
                (filtered_df['adj_BB'] <= criteria['max_bb']) &
                (~filtered_df['Batter'].isin(excluded_players))
            ].copy()
            
            if not profile_players.empty:
                best_player = profile_players.iloc[0]
                profile_analysis[profile_name] = {
                    'player': best_player,
                    'rank_overall': filtered_df[filtered_df['Batter'] == best_player['Batter']].index[0] + 1,
                    'count_in_profile': len(profile_players)
                }
        
        if profile_analysis:
            st.markdown("**🎯 Top Player by Profile:**")
            
            num_profiles = len(profile_analysis)
            if num_profiles == 1:
                cols = [st.columns(1)[0]]
            elif num_profiles == 2:
                cols = st.columns(2)
            elif num_profiles <= 4:
                cols = st.columns(min(num_profiles, 4))
            else:
                cols = st.columns(4)
            
            for i, (profile_name, analysis) in enumerate(profile_analysis.items()):
                player = analysis['player']
                overall_rank = analysis['rank_overall']
                
                with cols[i % len(cols)]:
                    # Profile header
                    icon = profile_criteria[profile_name]['icon']
                    st.markdown(f"**{icon} {profile_name.split(' ', 1)[1]}**")
                    
                    # Player name with rank
                    if overall_rank == 1:
                        st.success(f"🥇 **{player['Batter']}** (#{overall_rank})")
                    elif overall_rank <= 3:
                        st.info(f"🥈 **{player['Batter']}** (#{overall_rank})")
                    else:
                        st.info(f"**{player['Batter']}** (#{overall_rank})")
                    
                    # Key metrics with community data
                    k_vs_league = player['adj_K'] - LEAGUE_K_AVG
                    bb_vs_league = player['adj_BB'] - LEAGUE_BB_AVG
                    
                    st.markdown(f"""
                    **Hit Prob:** {player['total_hit_prob']:.1f}%  
                    **K% vs League:** {k_vs_league:+.1f}%  
                    **BB% vs League:** {bb_vs_league:+.1f}%  
                    **Score:** {player['Score']:.1f}  
                    **{player['form_indicator']} ({player['community_verifications']} verifications)**
                    """)

def main_page():
    """Enhanced main page with user-based tracking."""
    
    # Get user ID first - this will show setup if needed
    user_id = get_or_create_user_id()
    
    if not user_id:
        st.info("👆 Please set up your username in the sidebar to continue")
        return
    
    # Initialize user tracking system
    initialize_user_tracking_system(user_id)
    
    # Show welcome message for logged in user
    if 'username' in st.session_state:
        st.sidebar.success(f"👋 Welcome back, {st.session_state.username}!")
    
    create_league_aware_header()
    
    # Load and process data
    with st.spinner('🔄 Loading and analyzing today\'s matchups...'):
        df = load_and_process_data()
    
    if df is None:
        st.error("❌ Unable to load data. Please try again.")
        return
    
    # Show data quality dashboard with community stats
    create_data_quality_dashboard(df, user_id)
    
    # Create league-aware filters
    filters = create_league_aware_filters(df)
    
    # Calculate league-aware scores with community intelligence
    df = calculate_league_aware_scores(df, user_id)
    
    # Apply filters
    filtered_df = apply_league_aware_filters(df, filters)
    
    # Display results
    display_league_aware_results(filtered_df, filters)
    
    # Player Tracking System (User-Specific)
    st.markdown("---")
    st.header("🎯 Your Personal Player Tracking")
    
    # Show community stats
    community_stats = get_community_stats()
    if community_stats['total_verifications'] > 0:
        with st.expander("👥 Community Intelligence Stats", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Total Community Verifications", community_stats['total_verifications'])
            with col2:
                st.metric("Active Users Contributing", community_stats['active_users'])
            with col3:
                st.metric("Community Hit Rate", f"{community_stats['community_hit_rate']*100:.1f}%")
            
            if community_stats['top_tracked_players']:
                st.markdown("**🏆 Most Tracked Players:**")
                for player in community_stats['top_tracked_players'][:3]:
                    st.markdown(f"- {player}")
    
    # Create tabs for user tracking
    tab1, tab2, tab3 = st.tabs(["📝 Save Your Picks", "✅ Verify Your Results", "📊 Your Performance"])
    
    with tab1:
        st.subheader("📝 Save Your Player Selections")
        st.markdown("*Your picks are private, but your verifications help the community!*")
        
        if not filtered_df.empty:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                available_players = filtered_df['Batter'].tolist()
                selected_players = st.multiselect(
                    "Select Players to Track",
                    options=available_players,
                    help="Choose players you want to bet on and track results"
                )
                
                if selected_players:
                    # Show preview with community data
                    preview_df = filtered_df[filtered_df['Batter'].isin(selected_players)]
                    
                    preview_display = preview_df[['form_indicator', 'Batter', 'Tm', 'Pitcher', 'total_hit_prob', 'community_verifications', 'Score']].copy()
                    preview_display.columns = ['Form', 'Player', 'Team', 'vs Pitcher', 'Hit Prob %', 'Community Data', 'Score']
                    preview_display['Hit Prob %'] = preview_display['Hit Prob %'].apply(lambda x: f"{x:.1f}%")
                    preview_display['Score'] = preview_display['Score'].apply(lambda x: f"{x:.1f}")
                    
                    st.dataframe(preview_display, use_container_width=True, hide_index=True)
            
            with col2:
                st.markdown("**Quick Stats:**")
                if selected_players:
                    selected_data = filtered_df[filtered_df['Batter'].isin(selected_players)]
                    
                    st.metric("Players Selected", len(selected_players))
                    st.metric("Avg Hit Probability", f"{selected_data['total_hit_prob'].mean():.1f}%")
                    st.metric("Avg Community Verifications", f"{selected_data['community_verifications'].mean():.1f}")
                    
                    # Form distribution
                    hot_count = (selected_data['form_indicator'] == '🔥').sum()
                    cold_count = (selected_data['form_indicator'] == '❄️').sum()
                    
                    if hot_count > 0:
                        st.success(f"🔥 {hot_count} Community Hot Players")
                    if cold_count > 0:
                        st.warning(f"❄️ {cold_count} Community Cold Players")
            
            # Save picks button
            if selected_players:
                if st.button("💾 Save Your Picks", type="primary"):
                    saved_count = save_user_picks(user_id, selected_players, filtered_df)
                    if saved_count > 0:
                        st.success(f"✅ Successfully saved {saved_count} picks for today!")
                        st.info("🤝 Your verifications will help improve community insights!")
                    else:
                        st.error("❌ Failed to save picks. Please try again.")
        else:
            st.info("No players available. Adjust your filters to see more options.")
    
    with tab2:
        st.subheader("✅ Verify Your Results")
        st.markdown("*Help the community by verifying your picks!*")
        
        # Load user's pending picks
        pending_picks = load_user_pending_picks(user_id)
        
        if not pending_picks.empty:
            st.markdown(f"**📋 You have {len(pending_picks)} picks awaiting verification:**")
            
            # Group by date
            for date in pending_picks['Date'].dt.date.unique():
                date_picks = pending_picks[pending_picks['Date'].dt.date == date]
                
                with st.expander(f"📅 {date} ({len(date_picks)} picks)", expanded=date == datetime.now().date()):
                    
                    for idx, (_, pick) in enumerate(date_picks.iterrows()):
                        col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                        
                        with col1:
                            st.markdown(f"**{pick['Player']}** ({pick['Team']}) vs {pick['Opponent']}")
                            st.caption(f"Predicted: {pick['Score']:.1f} score, {pick['Hit_Prob']:.1f}% hit prob")
                        
                        with col2:
                            got_hit = st.selectbox(
                                "Got Hit?",
                                options=["Select...", "Yes", "No"],
                                key=f"hit_{pick.name}_{idx}"
                            )
                        
                        with col3:
                            actual_hits = st.number_input(
                                "# of Hits",
                                min_value=0,
                                max_value=5,
                                value=0,
                                key=f"hits_{pick.name}_{idx}",
                                help="Total hits in the game"
                            )
                        
                        with col4:
                            if got_hit != "Select...":
                                if st.button("💾 Save & Contribute", key=f"save_{pick.name}_{idx}"):
                                    hit_result = got_hit == "Yes"
                                    actual_pick_index = pick.name
                                    
                                    if update_user_pick_results(user_id, actual_pick_index, hit_result, actual_hits):
                                        st.success(f"✅ Updated result for {pick['Player']}")
                                        st.info("🤝 Your verification has been added to community data!")
                                        st.rerun()
                                    else:
                                        st.error("❌ Failed to save result")
        else:
            st.info("🎯 No pending picks to verify. Save some picks in the first tab!")
    
    with tab3:
        st.subheader("📊 Your Performance History")
        
        # Load user's historical data
        history_df = load_user_history(user_id)
        
        if not history_df.empty:
            # User performance summary
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_picks = len(history_df)
                st.metric("Your Total Picks", total_picks)
            
            with col2:
                hit_rate = history_df['Got_Hit'].mean() * 100 if 'Got_Hit' in history_df.columns else 0
                community_hit_rate = get_community_stats()['community_hit_rate'] * 100
                delta = hit_rate - community_hit_rate if community_hit_rate > 0 else None
                st.metric("Your Hit Rate", f"{hit_rate:.1f}%", delta=f"{delta:+.1f}%" if delta else None)
            
            with col3:
                recent_picks = history_df[history_df['Date'] >= (datetime.now() - timedelta(days=7))]
                recent_count = len(recent_picks)
                st.metric("Your Picks Last 7 Days", recent_count)
            
            with col4:
                if not recent_picks.empty and 'Got_Hit' in recent_picks.columns:
                    recent_hit_rate = recent_picks['Got_Hit'].mean() * 100
                    st.metric("Your Recent Hit Rate", f"{recent_hit_rate:.1f}%")
                else:
                    st.metric("Your Recent Hit Rate", "N/A")
            
            # User's top performing players
            st.markdown("**🏆 Your Best Tracked Players:**")
            
            if 'Got_Hit' in history_df.columns:
                player_stats = history_df.groupby('Player').agg({
                    'Got_Hit': ['count', 'sum', 'mean'],
                    'Predicted_Score': 'mean'
                }).round(3)
                
                player_stats.columns = ['Total_Picks', 'Total_Hits', 'Hit_Rate', 'Avg_Predicted_Score']
                player_stats = player_stats[player_stats['Total_Picks'] >= 2]
                player_stats = player_stats.sort_values('Hit_Rate', ascending=False)
                
                if not player_stats.empty:
                    display_stats = player_stats.reset_index()
                    display_stats['Hit_Rate'] = display_stats['Hit_Rate'].apply(lambda x: f"{x*100:.1f}%")
                    display_stats['Avg_Predicted_Score'] = display_stats['Avg_Predicted_Score'].apply(lambda x: f"{x:.1f}")
                    display_stats.columns = ['Player', 'Your Picks', 'Your Hits', 'Your Hit Rate', 'Avg Score']
                    
                    st.dataframe(display_stats, use_container_width=True, hide_index=True)
                else:
                    st.info("Track more players to see your performance statistics.")
            
            # Recent pick history
            if st.checkbox("📋 Show Your Recent Pick History"):
                recent_history = history_df.sort_values('Date', ascending=False).head(15)
                
                display_history = recent_history[['Date', 'Player', 'Team', 'Opponent', 'Predicted_Score', 'Got_Hit', 'Actual_Hits']].copy()
                display_history['Date'] = display_history['Date'].dt.strftime('%Y-%m-%d')
                display_history['Got_Hit'] = display_history['Got_Hit'].map({True: '✅', False: '❌', 1: '✅', 0: '❌'})
                display_history['Predicted_Score'] = display_history['Predicted_Score'].apply(lambda x: f"{x:.1f}")
                display_history.columns = ['Date', 'Player', 'Team', 'vs Pitcher', 'Predicted Score', 'Got Hit', 'Total Hits']
                
                st.dataframe(display_history, use_container_width=True, hide_index=True)
        else:
            st.info("📈 No personal data yet. Start tracking to build your performance history!")
            
            st.markdown("""
            **🚀 How It Works:**
            1. **Save Picks**: Select players and save them privately
            2. **Verify Results**: Update whether they got hits after games
            3. **Build History**: Track your success rate over time
            4. **Help Community**: Your verifications improve everyone's insights
            
            **🤝 Community Benefits:**
            - Your verifications help identify hot/cold players
            - More data = better predictions for everyone
            - Privacy maintained - only you see your picks
            """)
    
    # Export and management
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Your Results"):
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "💾 Download CSV", 
                csv, 
                f"mlb_predictions_{st.session_state.username}_{datetime.now().strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
    
    with col2:
        if st.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("🏟️ Clear Exclusions"):
            st.session_state.excluded_players = []
            st.rerun()
    
    with col4:
        st.info(f"🕐 Last updated: {datetime.now().strftime('%H:%M:%S')}")

def info_page():
    """Comprehensive baseball strategy guide."""
    st.title("📚 Complete Baseball Strategy Guide")
    
    with st.expander("🏆 Understanding the Community-Powered System", expanded=True):
        st.markdown("""
        ## 🚀 **Community Intelligence Revolution**
        
        Our system combines **real 2024 MLB data** with **crowdsourced verification** to identify truly exceptional performers.
        
        ### **👥 How Community Intelligence Works**
        - **Individual Privacy**: Your picks are private to you
        - **Collective Intelligence**: Your verifications help everyone
        - **Real Results**: Actual hit data improves predictions
        - **Hot/Cold Detection**: Community consensus identifies trends
        
        ### **📊 2024 MLB Reality + Community Data**
        | Metric | Elite (Top 10%) | Above Avg | League Avg | Community Boost |
        |--------|-----------------|-----------|------------|-----------------|
        | **K%** | ≤12.0% | 12.0-17.0% | ~22.6% | 🔥❄️ Indicators |
        | **BB%** | ≤4.0% | 4.0-6.0% | ~8.5% | Real verification |
        | **Form** | Community Hot | Normal | Community Cold | ±15 points |
        """)
    
    # Rest of the info page content would be the same as before...
    st.markdown("""
    ### 💡 **Key Innovation: Your Privacy + Community Power**
    - **Save picks privately** - Only you see your selections
    - **Verify results honestly** - Help build community database
    - **Benefit from crowd wisdom** - 🔥❄️ indicators from real data
    - **Improve over time** - More verifications = better predictions
    """)

def main():
    """Enhanced main function with user-based navigation."""
    st.sidebar.title("🏟️ Navigation")
    
    # Get user first
    user_id = get_or_create_user_id()
    
    # Show user info if logged in
    if user_id and 'username' in st.session_state:
        st.sidebar.markdown(f"**👤 Logged in as:** {st.session_state.username}")
        if st.sidebar.button("🔄 Switch User"):
            # Clear user session
            for key in ['user_id', 'username', 'username_set']:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()
        st.sidebar.markdown("---")
    
    # Optional music controls
    if st.sidebar.checkbox("🎵 Background Music"):
        audio_url = "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        components.html(f"""
        <audio controls autoplay loop style="width: 100%;">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        """, height=60)

    app_mode = st.sidebar.radio(
        "Choose Section",
        ["🎯 Community-Powered Predictor", "📚 Baseball Guide"],
        index=0
    )

    if app_mode == "🎯 Community-Powered Predictor":
        main_page()
    else:
        info_page()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**V3.0** | Community Intelligence")

if __name__ == "__main__":
    main()
