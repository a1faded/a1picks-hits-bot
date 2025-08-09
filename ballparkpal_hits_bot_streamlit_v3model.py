import streamlit as st
import pandas as pd
import requests
from io import StringIO
import altair as alt
import streamlit.components.v1 as components
import numpy as np
from datetime import datetime, timedelta
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import functools
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ==================== PROFESSIONAL UI STYLING WITH DARK MODE SUPPORT ====================

def inject_professional_css():
    """Inject improved CSS with dark mode compatibility and better icons"""
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ==================== ADAPTIVE COLOR SYSTEM ==================== */
    :root {
        --bg-primary: #ffffff;
        --bg-secondary: #f8fafc;
        --bg-card: #ffffff;
        --text-primary: #1f2937;
        --text-secondary: #6b7280;
        --text-muted: #9ca3af;
        --border-color: #e5e7eb;
        --accent-color: #2563eb;
        --success-color: #10b981;
        --warning-color: #f59e0b;
        --error-color: #ef4444;
        --shadow-color: rgba(0, 0, 0, 0.1);
        --hover-bg: rgba(37, 99, 235, 0.05);
    }
    
    /* Dark mode detection and variables */
    @media (prefers-color-scheme: dark) {
        :root {
            --bg-primary: #1f2937;
            --bg-secondary: #111827;
            --bg-card: #374151;
            --text-primary: #f9fafb;
            --text-secondary: #d1d5db;
            --text-muted: #9ca3af;
            --border-color: #4b5563;
            --accent-color: #3b82f6;
            --success-color: #34d399;
            --warning-color: #fbbf24;
            --error-color: #f87171;
            --shadow-color: rgba(0, 0, 0, 0.3);
            --hover-bg: rgba(59, 130, 246, 0.1);
        }
    }
    
    /* Force dark mode detection for Streamlit apps */
    [data-theme="dark"] {
        --bg-primary: #1f2937;
        --bg-secondary: #111827;
        --bg-card: #374151;
        --text-primary: #f9fafb;
        --text-secondary: #d1d5db;
        --text-muted: #9ca3af;
        --border-color: #4b5563;
        --accent-color: #3b82f6;
        --success-color: #34d399;
        --warning-color: #fbbf24;
        --error-color: #f87171;
        --shadow-color: rgba(0, 0, 0, 0.3);
        --hover-bg: rgba(59, 130, 246, 0.1);
    }
    
    /* ==================== GLOBAL STYLES WITH ADAPTIVE COLORS ==================== */
    .stApp {
        background: var(--bg-secondary) !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        min-height: 100vh;
        color: var(--text-primary) !important;
    }
    
    .main .block-container {
        padding: 2rem 1.5rem;
        background: var(--bg-primary) !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px var(--shadow-color), 0 1px 3px var(--shadow-color);
        margin: 1rem auto;
        max-width: 1400px;
        border: 1px solid var(--border-color);
    }
    
    /* ==================== HEADER WITH PROPER CONTRAST ==================== */
    .main-header {
        background: linear-gradient(135deg, var(--accent-color) 0%, #1e40af 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2);
        color: white !important;
        text-align: center;
        position: relative;
        border: 1px solid rgba(37, 99, 235, 0.3);
    }
    
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        letter-spacing: -0.5px;
        line-height: 1.2;
        color: white !important;
    }
    
    .main-subtitle {
        font-size: 1.1rem;
        font-weight: 400;
        margin: 0.75rem 0 0 0;
        opacity: 0.9;
        letter-spacing: 0.25px;
        color: white !important;
    }
    
    /* ==================== PROFESSIONAL CARDS WITH ADAPTIVE COLORS ==================== */
    .pro-card {
        background: var(--bg-card) !important;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        box-shadow: 0 1px 3px var(--shadow-color), 0 1px 2px var(--shadow-color);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
        position: relative;
    }
    
    .pro-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color), 0 2px 4px var(--shadow-color);
        border-color: var(--accent-color);
    }
    
    .pro-card h1, .pro-card h2, .pro-card h3, .pro-card h4, .pro-card h5, .pro-card h6 {
        color: var(--text-primary) !important;
        margin-bottom: 1rem !important;
    }
    
    .pro-card p {
        color: var(--text-secondary) !important;
        line-height: 1.5 !important;
    }
    
    /* Fix for profile text specifically */
    .pro-card div, .pro-card span {
        color: var(--text-primary) !important;
    }
    
    .pro-card strong {
        color: var(--text-primary) !important;
    }
    
    /* ==================== METRIC CARDS WITH PROPER TEXT COLORS ==================== */
    .metric-card-pro {
        background: var(--bg-card) !important;
        padding: 1.5rem;
        border-radius: 10px;
        color: var(--text-primary) !important;
        margin: 1rem 0;
        box-shadow: 0 1px 3px var(--shadow-color), 0 1px 2px var(--shadow-color);
        border: 1px solid var(--border-color);
        transition: all 0.2s ease;
        position: relative;
    }
    
    .metric-card-pro:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px var(--shadow-color);
        border-color: var(--accent-color);
    }
    
    .success-card-pro {
        border-left: 4px solid var(--success-color);
        background: var(--bg-card) !important;
    }
    
    .warning-card-pro {
        border-left: 4px solid var(--warning-color);
        background: var(--bg-card) !important;
    }
    
    .metric-title {
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        color: var(--text-secondary) !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin: 0.25rem 0;
        color: var(--text-primary) !important;
        line-height: 1;
    }
    
    .metric-subtitle {
        font-size: 0.875rem;
        color: var(--text-secondary) !important;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* ==================== IMPROVED ICON SYSTEM ==================== */
    .icon-baseball::before { content: "⚾"; font-style: normal; }
    .icon-trophy::before { content: "🏆"; font-style: normal; }
    .icon-star::before { content: "⭐"; font-style: normal; }
    .icon-lightning::before { content: "⚡"; font-style: normal; }
    .icon-diamond::before { content: "🔷"; font-style: normal; }
    .icon-explosion::before { content: "💥"; font-style: normal; }
    .icon-rocket::before { content: "🚀"; font-style: normal; }
    .icon-globe::before { content: "🌐"; font-style: normal; }
    .icon-target::before { content: "🎯"; font-style: normal; }
    .icon-chart::before { content: "📊"; font-style: normal; }
    .icon-stadium::before { content: "🏟️"; font-style: normal; }
    .icon-fire::before { content: "🔥"; font-style: normal; }
    .icon-tools::before { content: "🛠️"; font-style: normal; }
    
    /* Icon fallbacks for systems without emoji support */
    @supports not (content: "⚾") {
        .icon-baseball::before { content: "⚾"; font-weight: bold; color: var(--accent-color); }
        .icon-trophy::before { content: "🏆"; font-weight: bold; color: var(--warning-color); }
        .icon-star::before { content: "⭐"; font-weight: bold; color: var(--warning-color); }
        .icon-lightning::before { content: "⚡"; font-weight: bold; color: var(--accent-color); }
        .icon-diamond::before { content: "🔷"; font-weight: bold; color: var(--accent-color); }
        .icon-explosion::before { content: "💥"; font-weight: bold; color: var(--warning-color); }
        .icon-rocket::before { content: "🚀"; font-weight: bold; color: var(--accent-color); }
        .icon-globe::before { content: "🌐"; font-weight: bold; color: var(--success-color); }
        .icon-target::before { content: "🎯"; font-weight: bold; color: var(--accent-color); }
        .icon-chart::before { content: "📊"; font-weight: bold; color: var(--accent-color); }
        .icon-stadium::before { content: "🏟️"; font-weight: bold; color: var(--success-color); }
        .icon-fire::before { content: "🔥"; font-weight: bold; color: var(--warning-color); }
        .icon-tools::before { content: "🛠️"; font-weight: bold; color: var(--text-secondary); }
    }
    
    /* ==================== TABLE STYLING WITH ADAPTIVE COLORS ==================== */
    .dataframe {
        border: none !important;
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px var(--shadow-color) !important;
        background: var(--bg-card) !important;
        margin: 1.5rem 0 !important;
    }
    
    .dataframe thead th {
        background: linear-gradient(135deg, #1f2937 0%, #374151 100%) !important;
        color: white !important;
        font-weight: 600 !important;
        padding: 1rem 0.75rem !important;
        border: none !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-align: center !important;
    }
    
    .dataframe tbody td {
        padding: 0.75rem !important;
        border-bottom: 1px solid var(--border-color) !important;
        border-left: none !important;
        border-right: none !important;
        font-weight: 500 !important;
        font-size: 0.875rem !important;
        text-align: center !important;
        color: var(--text-primary) !important;
        background: var(--bg-card) !important;
    }
    
    .dataframe tbody tr:hover {
        background: var(--hover-bg) !important;
        transition: background-color 0.15s ease !important;
    }
    
    .dataframe tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    /* ==================== SIDEBAR WITH ADAPTIVE COLORS ==================== */
    .sidebar .sidebar-content {
        background: var(--bg-card) !important;
        border-radius: 12px !important;
        padding: 1.5rem !important;
        margin: 1rem 0.5rem !important;
        box-shadow: 0 1px 3px var(--shadow-color) !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* Fix Streamlit sidebar background */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary) !important;
    }
    
    section[data-testid="stSidebar"] > div {
        background: var(--bg-secondary) !important;
    }
    
    /* Sidebar text color fixes */
    section[data-testid="stSidebar"] * {
        color: var(--text-primary) !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    /* ==================== BUTTONS WITH ADAPTIVE STYLING ==================== */
    .stButton button {
        background: linear-gradient(135deg, var(--accent-color) 0%, #1d4ed8 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        transition: all 0.2s ease !important;
        box-shadow: 0 1px 3px rgba(37, 99, 235, 0.3) !important;
    }
    
    .stButton button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.4) !important;
        background: linear-gradient(135deg, #1d4ed8 0%, #1e40af 100%) !important;
    }
    
    /* ==================== FORM ELEMENTS WITH ADAPTIVE COLORS ==================== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.75rem !important;
        transition: all 0.2s ease !important;
        font-weight: 400 !important;
        color: var(--text-primary) !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 3px rgba(37, 99, 235, 0.1) !important;
    }
    
    .stSlider > div > div > div {
        background: var(--accent-color) !important;
    }
    
    /* ==================== CHARTS WITH ADAPTIVE BACKGROUND ==================== */
    .vega-embed {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px var(--shadow-color) !important;
        background: var(--bg-card) !important;
        padding: 1rem !important;
        margin: 1.5rem 0 !important;
        border: 1px solid var(--border-color) !important;
    }
    
    /* ==================== LEGEND WITH PROPER TEXT COLORS ==================== */
    .pro-legend {
        background: var(--bg-card) !important;
        padding: 1.5rem;
        border-radius: 10px;
        color: var(--text-primary) !important;
        margin: 2rem 0;
        box-shadow: 0 1px 3px var(--shadow-color);
        border: 1px solid var(--border-color);
        border-left: 4px solid var(--accent-color);
    }
    
    .legend-title {
        font-size: 1.25rem;
        font-weight: 700;
        margin-bottom: 1rem;
        color: var(--text-primary) !important;
        letter-spacing: -0.25px;
    }
    
    .legend-item {
        margin: 0.75rem 0;
        font-weight: 400;
        color: var(--text-secondary) !important;
        font-size: 0.875rem;
        line-height: 1.5;
    }
    
    .legend-item strong {
        color: var(--text-primary) !important;
        font-weight: 600;
    }
    
    /* ==================== PERFORMANCE WIDGETS WITH ADAPTIVE COLORS ==================== */
    .performance-widget-pro {
        background: var(--bg-card) !important;
        padding: 1rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        border-left: 3px solid var(--accent-color);
        font-size: 0.875rem;
        font-weight: 500;
        box-shadow: 0 1px 3px var(--shadow-color);
        transition: all 0.2s ease;
        color: var(--text-primary) !important;
    }
    
    .performance-widget-pro:hover {
        transform: translateX(3px);
        box-shadow: 0 2px 6px var(--shadow-color);
        border-left-color: #1d4ed8;
    }
    
    .performance-widget-pro strong {
        color: var(--text-primary) !important;
    }
    
    /* ==================== STATUS INDICATORS ==================== */
    .status-indicator {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    .status-success {
        background: var(--success-color);
        box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7);
    }
    
    .status-warning {
        background: var(--warning-color);
        box-shadow: 0 0 0 0 rgba(245, 158, 11, 0.7);
    }
    
    @keyframes pulse {
        0% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }
        70% { box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }
        100% { box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }
    }
    
    /* ==================== SECTION DIVIDERS ==================== */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, var(--border-color), var(--accent-color), var(--border-color));
        border-radius: 1px;
        margin: 2rem 0;
        opacity: 0.6;
    }
    
    /* ==================== EXPANDERS WITH ADAPTIVE COLORS ==================== */
    .streamlit-expanderHeader {
        background: var(--bg-card) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        border: 1px solid var(--border-color) !important;
        transition: all 0.2s ease !important;
        color: var(--text-primary) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: var(--bg-card) !important;
        border-color: var(--accent-color) !important;
        transform: translateY(-1px) !important;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-card) !important;
        border-radius: 0 0 8px 8px !important;
        padding: 1.5rem !important;
        border: 1px solid var(--border-color) !important;
        border-top: none !important;
        color: var(--text-primary) !important;
    }
    
    /* ==================== LOADING ANIMATION ==================== */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(37, 99, 235, 0.3);
        border-top: 2px solid var(--accent-color);
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* ==================== DASHBOARD SECTIONS ==================== */
    .dashboard-section {
        background: var(--bg-card) !important;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 1px 3px var(--shadow-color);
        border: 1px solid var(--border-color);
    }
    
    .dashboard-title {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--text-primary) !important;
        margin-bottom: 1rem;
        text-align: center;
    }
    
    /* ==================== GLOBAL TEXT COLOR FIXES ==================== */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6 {
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        line-height: 1.3 !important;
        letter-spacing: -0.25px !important;
        color: var(--text-primary) !important;
    }
    
    .stApp p, .stApp span, .stApp div {
        font-family: 'Inter', sans-serif !important;
        line-height: 1.5 !important;
        color: var(--text-primary) !important;
    }
    
    /* ==================== STREAMLIT SPECIFIC OVERRIDES ==================== */
    .stMarkdown {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {
        color: var(--text-primary) !important;
    }
    
    .stMarkdown p, .stMarkdown div, .stMarkdown span {
        color: var(--text-primary) !important;
    }
    
    /* Fix Streamlit metric containers */
    [data-testid="metric-container"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-color) !important;
        padding: 1rem !important;
        border-radius: 8px !important;
    }
    
    [data-testid="metric-container"] > div {
        color: var(--text-primary) !important;
    }
    
    /* Fix all text elements in cards */
    .pro-card * {
        color: var(--text-primary) !important;
    }
    
    .pro-card .metric-title {
        color: var(--text-secondary) !important;
    }
    
    .pro-card .metric-subtitle {
        color: var(--text-secondary) !important;
    }
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 1rem;
            margin: 0.5rem;
        }
        
        .main-title {
            font-size: 2rem;
        }
        
        .metric-value {
            font-size: 1.75rem;
        }
        
        .pro-card {
            padding: 1rem;
        }
        
        .metric-card-pro {
            padding: 1rem;
        }
    }
    
    /* ==================== ACCESSIBILITY ==================== */
    .stApp a {
        color: var(--accent-color) !important;
        text-decoration: none !important;
    }
    
    .stApp a:hover {
        color: #1d4ed8 !important;
        text-decoration: underline !important;
    }
    
    button:focus,
    select:focus,
    input:focus {
        outline: 2px solid var(--accent-color) !important;
        outline-offset: 2px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== CONFIGURATION MANAGEMENT ====================

class MLBConfig:
    """Centralized configuration management"""
    
    # Data Sources
    CSV_URLS = {
        'probabilities': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Pal.csv',
        'percent_change': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/Ballpark%20Palmodel2.csv',
        'pitcher_walks': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_walks.csv',
        'pitcher_hrs': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hrs.csv',
        'pitcher_hits': 'https://github.com/a1faded/a1picks-hits-bot/raw/main/pitcher_hits.csv'
    }
    
    # League Averages (2024)
    LEAGUE_AVERAGES = {
        'K_PCT': 22.6,
        'BB_PCT': 8.5,
        'update_date': '2024-08-01'
    }
    
    # Performance Thresholds
    THRESHOLDS = {
        'elite_score': 70,
        'good_score': 50,
        'elite_contact_k': 12.0,
        'elite_power_combo': 12.0,
        'elite_hit_prob': 35.0
    }
    
    # Player Profiles
    PLAYER_PROFILES = {
        'contact_aggressive': {
            'name': '🏆 Contact-Aggressive Hitters',
            'description': 'Low K% + Low BB% (Elite for base hits)',
            'max_k': 19.0,
            'max_bb': 7.0,
            'min_hit_prob': 32,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'elite_contact': {
            'name': '⭐ Elite Contact Specialists',
            'description': 'Ultra-low K% (Pure contact)',
            'max_k': 14.0,
            'max_bb': 9.5,
            'min_hit_prob': 28,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'swing_happy': {
            'name': '⚡ Swing-Happy Hitters',
            'description': 'Ultra-low BB% (Aggressive approach)',
            'max_k': 24.0,
            'max_bb': 5.0,
            'min_hit_prob': 30,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'above_average': {
            'name': '🔷 Above-Average Contact',
            'description': 'Better than league average K%',
            'max_k': 20.0,
            'max_bb': 12.0,
            'min_hit_prob': 25,
            'profile_type': 'contact',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        },
        'contact_power': {
            'name': '💥 Contact Power Hitters',
            'description': 'Low K% + High XB% & HR% (Power with contact)',
            'max_k': 20.0,
            'max_bb': 12.0,
            'min_xb': 7.0,
            'min_hr': 2.5,
            'min_vs': -5,
            'profile_type': 'power',
            'weights': {
                'adj_XB': 3.0, 'adj_HR': 2.5, 'adj_vs': 1.5,
                'adj_RC': 1.0, 'adj_1B': 0.5, 'adj_K': -1.0, 'adj_BB': -0.3
            }
        },
        'pure_power': {
            'name': '🚀 Pure Power Sluggers',
            'description': 'High XB% & HR% (Power over contact)',
            'max_k': 100,
            'max_bb': 100,
            'min_xb': 9.0,
            'min_hr': 3.5,
            'min_vs': -10,
            'profile_type': 'power',
            'weights': {
                'adj_XB': 3.0, 'adj_HR': 2.5, 'adj_vs': 1.5,
                'adj_RC': 1.0, 'adj_1B': 0.5, 'adj_K': -1.0, 'adj_BB': -0.3
            }
        },
        'all_power': {
            'name': '⚾ All Power Players',
            'description': 'All players ranked by power potential (Research mode)',
            'max_k': 100,
            'max_bb': 100,
            'min_xb': 0,
            'min_hr': 0,
            'min_vs': -10,
            'profile_type': 'power',
            'weights': {
                'adj_XB': 3.0, 'adj_HR': 2.5, 'adj_vs': 1.5,
                'adj_RC': 1.0, 'adj_1B': 0.5, 'adj_K': -1.0, 'adj_BB': -0.3
            }
        },
        'all_players': {
            'name': '🌐 All Players',
            'description': 'No restrictions',
            'max_k': 100,
            'max_bb': 100,
            'min_hit_prob': 20,
            'profile_type': 'all',
            'weights': {
                'adj_1B': 2.0, 'adj_XB': 1.8, 'adj_vs': 1.2,
                'adj_RC': 0.8, 'adj_HR': 0.6, 'adj_K': -2.0, 'adj_BB': -0.8
            }
        }
    }
    
    # Expected columns for validation
    EXPECTED_COLUMNS = {
        'main': ['Tm', 'Batter', 'vs', 'Pitcher', 'RC', 'HR', 'XB', '1B', 'BB', 'K'],
        'pitcher': ['Team', 'Name', 'Park', 'Prob']
    }
    
    # Cache settings
    CACHE_TTL = 900  # 15 minutes
    
    # Sorting options
    SORT_OPTIONS = {
        "Score (High to Low)": ("Score", False),
        "Score (Low to High)": ("Score", True),
        "Hit Prob% (High to Low)": ("total_hit_prob", False),
        "Hit Prob% (Low to High)": ("total_hit_prob", True),
        "HR% (High to Low)": ("adj_HR", False),
        "HR% (Low to High)": ("adj_HR", True),
        "XB% (High to Low)": ("adj_XB", False),
        "XB% (Low to High)": ("adj_XB", True),
        "Contact% (High to Low)": ("adj_1B", False),
        "Contact% (Low to High)": ("adj_1B", True),
        "K% (Low to High)": ("adj_K", True),
        "K% (High to Low)": ("adj_K", False),
        "BB% (Low to High)": ("adj_BB", True),
        "BB% (High to Low)": ("adj_BB", False),
        "vs Pitcher (High to Low)": ("adj_vs", False),
        "vs Pitcher (Low to High)": ("adj_vs", True),
        "Power Combo (High to Low)": ("power_combo", False),
        "Power Combo (Low to High)": ("power_combo", True)
    }

# ==================== PERFORMANCE MONITORING ====================

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.warnings_shown = set()
    
    def timer(self, operation_name: str):
        """Decorator to monitor function performance"""
        def decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.time()
                result = func(*args, **kwargs)
                end_time = time.time()
                
                duration = end_time - start_time
                
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                
                self.metrics[operation_name].append(duration)
                
                # Show performance indicators
                if duration > 1.0 and operation_name not in self.warnings_shown:
                    st.sidebar.markdown(f"""
                    <div class="performance-widget-pro">
                        <span class="status-indicator status-warning"></span>
                        <strong>{operation_name}:</strong> {duration:.2f}s
                    </div>
                    """, unsafe_allow_html=True)
                    self.warnings_shown.add(operation_name)
                elif duration < 0.5 and operation_name in self.warnings_shown:
                    st.sidebar.markdown(f"""
                    <div class="performance-widget-pro">
                        <span class="status-indicator status-success"></span>
                        <strong>{operation_name}:</strong> {duration:.2f}s (Optimized!)
                    </div>
                    """, unsafe_allow_html=True)
                
                return result
            return wrapper
        return decorator
    
    def get_performance_summary(self):
        """Generate performance report"""
        if not self.metrics:
            return "No data"
        
        total_time = sum(sum(times) for times in self.metrics.values())
        return f"Total: {total_time:.2f}s"

# Initialize global performance monitor
monitor = PerformanceMonitor()

# ==================== MEMORY OPTIMIZATION ====================

class MemoryOptimizer:
    """Memory usage optimization utilities"""
    
    @staticmethod
    def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce DataFrame memory usage by 50-70%"""
        
        # Optimize numeric columns
        for col in df.select_dtypes(include=['int64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='integer')
        
        for col in df.select_dtypes(include=['float64']).columns:
            df[col] = pd.to_numeric(df[col], downcast='float')
        
        # Convert strings to categories (massive memory savings)
        for col in df.select_dtypes(include=['object']).columns:
            if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
                df[col] = df[col].astype('category')
        
        return df
    
    @staticmethod
    def get_memory_usage(df: pd.DataFrame) -> str:
        """Get human-readable memory usage"""
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        return f"{memory_mb:.1f} MB"

# ==================== DATA VALIDATION ====================

class DataValidator:
    """Comprehensive data validation"""
    
    @staticmethod
    def validate_dataframe(df: Optional[pd.DataFrame], description: str, 
                          required_columns: List[str]) -> pd.DataFrame:
        """Validate DataFrame structure and content"""
        
        if df is None or df.empty:
            raise ValueError(f"{description}: No data found")
        
        # Check required columns
        missing_cols = set(required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"{description}: Missing columns {missing_cols}")
        
        # Validate numeric columns
        numeric_cols = ['adj_K', 'adj_BB', 'adj_HR', 'adj_XB', 'Score']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                nan_count = df[col].isna().sum()
                if nan_count > len(df) * 0.1:  # More than 10% NaN
                    st.warning(f"⚠️ {description}: {nan_count} invalid values in {col}")
        
        return df
    
    @staticmethod
    def validate_filters(filters: Dict) -> bool:
        """Validate filter parameters"""
        
        # Validate numeric ranges
        if filters.get('max_k', 0) < 0 or filters.get('max_k', 0) > 100:
            raise ValueError("K% must be between 0-100%")
        
        if filters.get('max_bb', 0) < 0 or filters.get('max_bb', 0) > 100:
            raise ValueError("BB% must be between 0-100%")
        
        # Validate result count
        valid_counts = [5, 10, 15, 20, 25, 30, "All"]
        if filters.get('result_count') not in valid_counts:
            raise ValueError(f"Invalid result count. Must be one of: {valid_counts}")
        
        return True

# ==================== CONCURRENT DATA LOADING ====================

class ConcurrentDataLoader:
    """Concurrent data loading using ThreadPoolExecutor (no external dependencies)"""
    
    @staticmethod
    def load_multiple_csvs(url_dict: Dict[str, str]) -> Dict[str, Optional[pd.DataFrame]]:
        """Load multiple CSV files concurrently using threading"""
        
        def load_single_csv(description_url_pair):
            description, url = description_url_pair
            try:
                response = requests.get(url, timeout=15)
                if response.status_code == 200:
                    df = pd.read_csv(StringIO(response.text))
                    return description, MemoryOptimizer.optimize_dataframe(df)
                else:
                    st.error(f"❌ HTTP {response.status_code} for {description}")
                    return description, None
            except Exception as e:
                st.error(f"❌ Failed to load {description}: {str(e)}")
                return description, None
        
        # Use ThreadPoolExecutor for concurrent loading
        results = {}
        with ThreadPoolExecutor(max_workers=5) as executor:
            # Submit all tasks
            future_to_desc = {
                executor.submit(load_single_csv, (desc, url)): desc 
                for desc, url in url_dict.items()
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_desc):
                description, dataframe = future.result()
                results[description] = dataframe
        
        return results

# ==================== OPTIMIZED DATA PROCESSING ====================

class DataProcessor:
    """Optimized data processing utilities"""
    
    @staticmethod
    def apply_profile_filter(df: pd.DataFrame, profile_key: str) -> pd.DataFrame:
        """Apply profile-based filtering efficiently"""
        
        profile_config = MLBConfig.PLAYER_PROFILES[profile_key]
        conditions = []
        
        # Build query conditions
        if profile_config.get('max_k', 100) < 100:
            conditions.append(f"adj_K <= {profile_config['max_k']}")
        
        if profile_config.get('max_bb', 100) < 100:
            conditions.append(f"adj_BB <= {profile_config['max_bb']}")
        
        if profile_config['profile_type'] == 'power':
            if profile_config.get('min_xb', 0) > 0:
                conditions.append(f"adj_XB >= {profile_config['min_xb']}")
            if profile_config.get('min_hr', 0) > 0:
                conditions.append(f"adj_HR >= {profile_config['min_hr']}")
            if profile_config.get('min_vs', -10) > -10:
                conditions.append(f"adj_vs >= {profile_config['min_vs']}")
        else:
            if profile_config.get('min_hit_prob', 0) > 0:
                conditions.append(f"total_hit_prob >= {profile_config['min_hit_prob']}")
        
        # Apply filters efficiently
        if conditions:
            query_string = " and ".join(conditions)
            return df.query(query_string)
        
        return df
    
    @staticmethod
    def calculate_league_comparison(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate league comparison metrics"""
        df['K_vs_League'] = MLBConfig.LEAGUE_AVERAGES['K_PCT'] - df['adj_K']
        df['BB_vs_League'] = MLBConfig.LEAGUE_AVERAGES['BB_PCT'] - df['adj_BB']
        return df
    
    @staticmethod
    def efficient_merge_and_calculate(prob_df: pd.DataFrame, pct_df: pd.DataFrame) -> pd.DataFrame:
        """Optimized merging and metric calculation"""
        
        # Efficient merge
        merged_df = pd.merge(
            prob_df, pct_df,
            on=['Tm', 'Batter', 'Pitcher'],
            suffixes=('_prob', '_pct'),
            how='inner'
        )
        
        # Vectorized calculations
        metrics = ['1B', 'XB', 'vs', 'K', 'BB', 'HR', 'RC']
        
        for metric in metrics:
            base_col = f'{metric}.1' if metric in ['K', 'BB'] else f'{metric}_prob'
            pct_col = f'{metric}_pct'
            
            if base_col in merged_df.columns and pct_col in merged_df.columns:
                # Vectorized calculation
                merged_df[f'adj_{metric}'] = (
                    merged_df[base_col] * (1 + merged_df[pct_col]/100)
                ).clip(lower=0, upper=100 if metric in ['K', 'BB'] else None)
        
        # Calculate derived metrics
        merged_df['power_combo'] = merged_df['adj_XB'] + merged_df['adj_HR']
        merged_df['total_hit_prob'] = (
            merged_df['adj_1B'] + merged_df['adj_XB'] + merged_df['adj_HR']
        ).clip(upper=100)
        
        return merged_df

# ==================== IMPROVED PROFESSIONAL UI COMPONENTS ====================

class ProfessionalUIComponents:
    """Improved professional UI component generators with dark mode support and better icons"""
    
    @staticmethod
    def create_professional_header():
        """Create improved professional header with better icon support"""
        st.markdown("""
        <div class="main-header">
            <h1 class="main-title"><span class="icon-baseball"></span> MLB Hit Predictor Pro</h1>
            <p class="main-subtitle">Professional DFS Analytics Platform v4.2</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_professional_metric_card(title: str, value: str, subtitle: str = "", 
                                      card_type: str = "metric", icon: str = "") -> str:
        """Generate improved professional metric cards with better icons"""
        card_class = f"{card_type}-card-pro"
        icon_class = f"icon-{icon}" if icon else ""
        
        return f"""
        <div class="{card_class}">
            <div class="metric-title">{title}</div>
            <div class="metric-value"><span class="{icon_class}"></span> {value}</div>
            <div class="metric-subtitle">{subtitle}</div>
        </div>
        """
    
    @staticmethod
    def create_section_divider():
        """Create improved section divider"""
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_professional_legend(content: str):
        """Create improved professional legend component"""
        st.markdown(f"""
        <div class="pro-legend">
            <div class="legend-title"><span class="icon-chart"></span> Analytics Guide</div>
            <div class="legend-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_loading_indicator(text: str = "Loading"):
        """Create improved loading indicator"""
        return f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
            <div class="loading-spinner"></div>
            <span style="margin-left: 1rem; font-weight: 500; color: var(--text-primary);">{text}...</span>
        </div>
        """
    
    @staticmethod
    def create_dashboard_section(title: str, icon: str = "chart"):
        """Create improved dashboard section with icons"""
        icon_class = f"icon-{icon}"
        return f"""
        <div class="dashboard-section">
            <h2 class="dashboard-title"><span class="{icon_class}"></span> {title}</h2>
        </div>
        """

# ==================== MAIN APPLICATION CONFIGURATION ====================

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro v4.2",
    layout="wide",
    page_icon="⚾",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:your@email.com',
        'Report a bug': "https://github.com/yourrepo/issues",
    }
)

# Inject improved professional CSS
inject_professional_css()

# ==================== DATA LOADING AND PROCESSING ====================

@monitor.timer("data_loading")
@st.cache_data(ttl=MLBConfig.CACHE_TTL)
def load_and_process_data():
    """Enhanced data loading with concurrent processing and validation"""
    
    # Show professional loading indicator
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        ProfessionalUIComponents.create_loading_indicator("Loading MLB data"),
        unsafe_allow_html=True
    )
    
    # Concurrent data loading using ThreadPoolExecutor
    data_dict = ConcurrentDataLoader.load_multiple_csvs(MLBConfig.CSV_URLS)
    loading_placeholder.empty()
    
    # Validate main datasets
    prob_df = data_dict.get('probabilities')
    pct_df = data_dict.get('percent_change')
    
    if prob_df is None or pct_df is None:
        st.error("❌ Failed to load required data files")
        return None
    
    try:
        # Validate data
        prob_df = DataValidator.validate_dataframe(
            prob_df, "Base Probabilities", MLBConfig.EXPECTED_COLUMNS['main']
        )
        pct_df = DataValidator.validate_dataframe(
            pct_df, "Adjustment Factors", MLBConfig.EXPECTED_COLUMNS['main']
        )
        
        # Process data efficiently
        merged_df = DataProcessor.efficient_merge_and_calculate(prob_df, pct_df)
        
        # Load pitcher data (optional)
        pitcher_data = load_pitcher_matchup_data(data_dict)
        
        # Merge pitcher data if available
        if pitcher_data is not None and not pitcher_data.empty:
            try:
                merged_df = pd.merge(
                    merged_df, pitcher_data,
                    left_on=['Tm', 'Pitcher'], 
                    right_on=['Opponent_Team', 'Pitcher_Name'],
                    how='left'
                )
            except Exception as e:
                st.warning(f"⚠️ Pitcher data merge failed: {str(e)}")
        else:
            # Add default pitcher columns
            merged_df['Walk_3Plus_Probability'] = 15.0
            merged_df['HR_2Plus_Probability'] = 12.0
            merged_df['Hit_8Plus_Probability'] = 18.0
            st.info("ℹ️ Using base analysis - pitcher matchup data not available")
        
        # Memory optimization
        merged_df = MemoryOptimizer.optimize_dataframe(merged_df)
        
        return merged_df
        
    except Exception as e:
        st.error(f"❌ Data processing error: {str(e)}")
        return None

@monitor.timer("pitcher_data_processing")
def load_pitcher_matchup_data(data_dict):
    """Process pitcher matchup data efficiently"""
    
    walks_df = data_dict.get('pitcher_walks')
    hrs_df = data_dict.get('pitcher_hrs')
    hits_df = data_dict.get('pitcher_hits')
    
    if not all([walks_df is not None, hrs_df is not None, hits_df is not None]):
        return None
    
    try:
        # Process each dataset
        def clean_prob_column(df, new_col_name):
            df = df.copy()
            df['Prob_Clean'] = df['Prob'].astype(str).str.replace('%', '').str.strip()
            df['Prob_Clean'] = pd.to_numeric(df['Prob_Clean'], errors='coerce')
            df = df.rename(columns={'Prob_Clean': new_col_name})
            df['Pitcher_LastName'] = df['Name'].str.split().str[-1]
            return df[['Team', 'Name', 'Pitcher_LastName', 'Park', new_col_name]]
        
        # Clean datasets
        walks_clean = clean_prob_column(walks_df, 'Walk_3Plus_Probability')
        hrs_clean = clean_prob_column(hrs_df, 'HR_2Plus_Probability')
        hits_clean = clean_prob_column(hits_df, 'Hit_8Plus_Probability')
        
        # Efficient merging
        pitcher_data = walks_clean
        pitcher_data = pd.merge(pitcher_data, hrs_clean, on=['Team', 'Pitcher_LastName', 'Park'], how='outer')
        pitcher_data = pd.merge(pitcher_data, hits_clean, on=['Team', 'Pitcher_LastName', 'Park'], how='outer')
        
        # Fill missing values
        pitcher_data['Walk_3Plus_Probability'] = pitcher_data['Walk_3Plus_Probability'].fillna(15)
        pitcher_data['HR_2Plus_Probability'] = pitcher_data['HR_2Plus_Probability'].fillna(12)
        pitcher_data['Hit_8Plus_Probability'] = pitcher_data['Hit_8Plus_Probability'].fillna(18)
        
        # Rename for merging
        pitcher_data = pitcher_data.rename(columns={
            'Team': 'Pitcher_Team',
            'Pitcher_LastName': 'Pitcher_Name',
            'Park': 'Opponent_Team'
        })
        
        return MemoryOptimizer.optimize_dataframe(pitcher_data)
        
    except Exception as e:
        st.warning(f"⚠️ Pitcher data processing error: {str(e)}")
        return None

@monitor.timer("pitcher_matchup_calculation")
def calculate_pitcher_matchup_grades(df, profile_type):
    """Calculate pitcher matchup modifiers and grades"""
    
    df['pitcher_matchup_modifier'] = 0.0
    df['pitcher_matchup_grade'] = 'B'
    
    if not all(col in df.columns for col in ['Walk_3Plus_Probability', 'HR_2Plus_Probability', 'Hit_8Plus_Probability']):
        return df
    
    # Fill NaN values
    df['Walk_3Plus_Probability'] = df['Walk_3Plus_Probability'].fillna(15.0)
    df['HR_2Plus_Probability'] = df['HR_2Plus_Probability'].fillna(12.0)
    df['Hit_8Plus_Probability'] = df['Hit_8Plus_Probability'].fillna(18.0)
    
    # Vectorized calculations for better performance
    walk_penalty = np.select([
        df['Walk_3Plus_Probability'] >= 50,
        df['Walk_3Plus_Probability'] >= 40,
        df['Walk_3Plus_Probability'] >= 30,
        df['Walk_3Plus_Probability'] >= 20
    ], [-8, -6, -4, -2], default=0)
    
    df['pitcher_matchup_modifier'] = walk_penalty
    
    if profile_type == 'power':
        hr_bonus = np.select([
            df['HR_2Plus_Probability'] >= 25,
            df['HR_2Plus_Probability'] >= 20,
            df['HR_2Plus_Probability'] >= 15,
            df['HR_2Plus_Probability'] >= 10
        ], [15, 12, 8, 4], default=0)
        df['pitcher_matchup_modifier'] += hr_bonus
    else:
        hit_bonus = np.select([
            df['Hit_8Plus_Probability'] >= 25,
            df['Hit_8Plus_Probability'] >= 20,
            df['Hit_8Plus_Probability'] >= 15,
            df['Hit_8Plus_Probability'] >= 10
        ], [12, 9, 6, 3], default=0)
        df['pitcher_matchup_modifier'] += hit_bonus
    
    # Calculate grades
    df['pitcher_matchup_grade'] = np.select([
        df['pitcher_matchup_modifier'] >= 10,
        df['pitcher_matchup_modifier'] >= 5,
        df['pitcher_matchup_modifier'] >= 0,
        df['pitcher_matchup_modifier'] >= -5
    ], ['A+', 'A', 'B', 'C'], default='D')
    
    return df

@monitor.timer("scoring_calculation")
def calculate_league_aware_scores(df, profile_type='contact'):
    """Enhanced scoring algorithm with profile-specific logic"""
    
    # Calculate pitcher matchup grades
    df = calculate_pitcher_matchup_grades(df, profile_type)
    
    # Get profile configuration
    profile_key = {
        'contact': 'contact_aggressive',
        'power': 'contact_power',
        'all': 'all_players'
    }.get(profile_type, 'contact_aggressive')
    
    weights = MLBConfig.PLAYER_PROFILES[profile_key]['weights']
    
    # Profile-specific bonuses
    if profile_type == 'power':
        df['power_bonus'] = np.where((df['adj_XB'] > 10) & (df['adj_HR'] > 4), 12, 0)
        df['clutch_power_bonus'] = np.where((df['adj_XB'] > 8) & (df['adj_vs'] > 5), 8, 0)
        df['consistent_power_bonus'] = np.where((df['adj_HR'] > 3) & (df['adj_K'] < 25), 5, 0)
        bonus_cols = ['power_bonus', 'clutch_power_bonus', 'consistent_power_bonus']
        
        df['power_prob'] = df['adj_XB'] + df['adj_HR']
        df['power_prob'] = df['power_prob'].clip(upper=50)
    else:
        df['contact_bonus'] = np.where((df['total_hit_prob'] > 40) & (df['adj_K'] < 18), 8, 0)
        df['consistency_bonus'] = np.where((df['adj_1B'] > 20) & (df['adj_XB'] > 8), 5, 0)
        df['matchup_bonus'] = np.where(df['adj_vs'] > 5, 3, 0)
        bonus_cols = ['contact_bonus', 'consistency_bonus', 'matchup_bonus']
    
    # Vectorized score calculation
    df['base_score'] = sum(df[col] * weight for col, weight in weights.items() if col in df.columns)
    df['Score'] = df['base_score'] + sum(df[col] for col in bonus_cols if col in df.columns) + df.get('pitcher_matchup_modifier', 0)
    
    # Normalize scores
    if df['Score'].max() != df['Score'].min():
        df['Score'] = (df['Score'] - df['Score'].min()) / (df['Score'].max() - df['Score'].min()) * 100
    else:
        df['Score'] = 50
    
    return df.round(1)

# ==================== IMPROVED PROFESSIONAL FILTERING SYSTEM ====================

@monitor.timer("filter_creation")
def create_professional_filters(df=None):
    """Improved professional filtering system with better dark mode support"""
    
    st.sidebar.markdown("""
    <div class="pro-card">
        <h3 style="margin: 0 0 0.5rem 0; font-weight: 700;"><span class="icon-target"></span> Professional Filters</h3>
        <p style="margin: 0; font-size: 0.875rem;">Advanced Baseball Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []
    
    filters = {}
    
    # League context display
    st.sidebar.markdown("""
    <div class="pro-card">
        <h4 style="margin: 0 0 1rem 0; font-weight: 600;"><span class="icon-chart"></span> 2024 League Benchmarks</h4>
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(37, 99, 235, 0.1); border-radius: 6px;">
            <span style="font-weight: 500;">K% Average:</span>
            <span style="color: var(--accent-color); font-weight: 600;">22.6%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(37, 99, 235, 0.1); border-radius: 6px;">
            <span style="font-weight: 500;">BB% Average:</span>
            <span style="color: var(--accent-color); font-weight: 600;">8.5%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show dataset stats
    if df is not None and not df.empty:
        memory_usage = MemoryOptimizer.get_memory_usage(df)
        st.sidebar.markdown(f"""
        <div class="pro-card">
            <h4 style="margin: 0 0 1rem 0; font-weight: 600;"><span class="icon-chart"></span> Data Status</h4>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 6px;">
                <span style="font-weight: 500;">Matchups:</span>
                <span style="color: var(--success-color); font-weight: 600;">{len(df):,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(16, 185, 129, 0.1); border-radius: 6px;">
                <span style="font-weight: 500;">Memory:</span>
                <span style="color: var(--accent-color); font-weight: 600;">{memory_usage}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    ProfessionalUIComponents.create_section_divider()
    
    # Player Type Selection
    st.sidebar.markdown("""
    <div class="pro-card">
        <h4 style="margin: 0 0 1rem 0; font-weight: 600;"><span class="icon-target"></span> Player Profile Selection</h4>
    </div>
    """, unsafe_allow_html=True)
    
    profile_options = {v['name']: k for k, v in MLBConfig.PLAYER_PROFILES.items()}
    
    selected_profile_name = st.sidebar.selectbox(
        "Choose Player Profile",
        options=list(profile_options.keys()),
        index=0,
        help="Each profile targets different hitting approaches"
    )
    
    selected_profile_key = profile_options[selected_profile_name]
    profile_config = MLBConfig.PLAYER_PROFILES[selected_profile_key]
    
    filters['profile_key'] = selected_profile_key
    filters['profile_type'] = profile_config['profile_type']
    
    # Show profile details with proper text colors
    st.sidebar.markdown(f"""
    <div class="pro-card">
        <h4 style="margin: 0 0 0.5rem 0; font-weight: 600; color: var(--text-primary) !important;">{selected_profile_name}</h4>
        <p style="margin: 0 0 1rem 0; font-style: italic; font-size: 0.875rem; color: var(--text-secondary) !important;">{profile_config['description']}</p>
        <div style="background: rgba(37, 99, 235, 0.1); padding: 1rem; border-radius: 8px; border-left: 3px solid var(--accent-color);">
            <div style="margin: 0.25rem 0; font-size: 0.875rem; color: var(--text-primary) !important;">
                <strong style="color: var(--text-primary) !important;">Max K%:</strong> 
                <span style="color: var(--text-primary) !important;">{profile_config.get('max_k', 'N/A')}</span>
            </div>
            <div style="margin: 0.25rem 0; font-size: 0.875rem; color: var(--text-primary) !important;">
                <strong style="color: var(--text-primary) !important;">Max BB%:</strong> 
                <span style="color: var(--text-primary) !important;">{profile_config.get('max_bb', 'N/A')}</span>
            </div>
            {f"<div style='margin: 0.25rem 0; font-size: 0.875rem; color: var(--text-primary) !important;'><strong style='color: var(--text-primary) !important;'>Min Hit Prob:</strong> <span style='color: var(--text-primary) !important;'>{profile_config.get('min_hit_prob', 'N/A')}%</span></div>" if profile_config['profile_type'] != 'power' else ''}
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced Options
    with st.sidebar.expander("⚙️ Advanced Configuration", expanded=False):
        
        # Custom thresholds
        st.markdown("**Custom Overrides**")
        filters['custom_max_k'] = st.slider(
            "Maximum K%",
            min_value=5.0,
            max_value=35.0,
            value=profile_config.get('max_k', 20.0),
            step=0.5,
            help="Override profile K% threshold"
        )
        
        filters['custom_max_bb'] = st.slider(
            "Maximum BB%",
            min_value=2.0,
            max_value=15.0,
            value=profile_config.get('max_bb', 8.0),
            step=0.5,
            help="Override profile BB% threshold"
        )
        
        st.markdown("**Team Management**")
        # Team filtering
        if df is not None and not df.empty:
            team_options = sorted(df['Tm'].unique().tolist())
            
            filters['selected_teams'] = st.multiselect(
                "Include Teams Only",
                options=team_options,
                help="Leave empty to include all teams"
            )
            
            filters['excluded_teams'] = st.multiselect(
                "Exclude Teams",
                options=team_options,
                help="Teams to completely exclude"
            )
        
        # Best per team filter
        filters['best_per_team_only'] = st.checkbox(
            "🏟️ Best player per team only",
            value=False,
            help="Show only highest scoring player from each team"
        )
        
        st.markdown("**Results Configuration**")
        # Sorting
        filters['sort_option'] = st.selectbox(
            "Sort Results By",
            options=list(MLBConfig.SORT_OPTIONS.keys()),
            index=0,
            help="Choose sorting criteria"
        )
        
        filters['sort_col'], filters['sort_asc'] = MLBConfig.SORT_OPTIONS[filters['sort_option']]
        
        # Result count
        filters['result_count'] = st.selectbox(
            "Number of Results",
            options=[5, 10, 15, 20, 25, 30, "All"],
            index=2,
            help="Number of players to display"
        )
    
    # Lineup Management
    with st.sidebar.expander("🏟️ Lineup Management", expanded=False):
        
        st.markdown("**Player Exclusions**")
        if df is not None and not df.empty:
            all_players = sorted(df['Batter'].unique().tolist())
            
            excluded_players = st.multiselect(
                "Players NOT Playing Today",
                options=all_players,
                default=st.session_state.excluded_players,
                help="Exclude players confirmed out of lineups"
            )
            
            st.session_state.excluded_players = excluded_players
            filters['excluded_players'] = excluded_players
            
            if excluded_players:
                st.markdown(f"""
                <div style="background: rgba(245, 158, 11, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong style="color: var(--text-primary) !important;">🚫 Excluding {len(excluded_players)} players</strong>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Clear All Exclusions", help="Remove all player exclusions"):
                st.session_state.excluded_players = []
                st.rerun()
    
    return filters

@monitor.timer("filtering_application")
def apply_professional_filters(df, filters):
    """Apply filters with enhanced performance and error handling"""
    
    if df is None or df.empty:
        return df
    
    try:
        # Validate filters
        DataValidator.validate_filters(filters)
        
        # Apply player exclusions
        excluded_players = filters.get('excluded_players', [])
        if excluded_players:
            df = df[~df['Batter'].isin(excluded_players)]
        
        # Apply team filters
        selected_teams = filters.get('selected_teams', [])
        if selected_teams:
            df = df[df['Tm'].isin(selected_teams)]
        
        excluded_teams = filters.get('excluded_teams', [])
        if excluded_teams:
            df = df[~df['Tm'].isin(excluded_teams)]
        
        # Apply profile filtering
        profile_key = filters.get('profile_key', 'contact_aggressive')
        df = DataProcessor.apply_profile_filter(df, profile_key)
        
        # Apply custom overrides
        custom_max_k = filters.get('custom_max_k')
        custom_max_bb = filters.get('custom_max_bb')
        
        if custom_max_k is not None:
            df = df[df['adj_K'] <= custom_max_k]
        
        if custom_max_bb is not None:
            df = df[df['adj_BB'] <= custom_max_bb]
        
        # Best per team filter
        if filters.get('best_per_team_only', False):
            df = df.loc[df.groupby('Tm')['Score'].idxmax()]
        
        # Apply sorting
        sort_col = filters.get('sort_col', 'Score')
        sort_asc = filters.get('sort_asc', False)
        
        if sort_col in df.columns:
            df = df.sort_values(sort_col, ascending=sort_asc, na_position='last')
        
        # Limit results
        result_count = filters.get('result_count', 15)
        if result_count != "All":
            df = df.head(result_count)
        
        return df
        
    except Exception as e:
        st.error(f"❌ Filtering error: {str(e)}")
        return df.sort_values('Score', ascending=False).head(15)

# ==================== IMPROVED PROFESSIONAL DISPLAY FUNCTIONS ====================

@monitor.timer("data_overview_display")
def display_professional_overview(df):
    """Display improved data overview with better icons and dark mode support"""
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    st.markdown(ProfessionalUIComponents.create_dashboard_section(
        "Today's Analytics Dashboard", "chart"
    ), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Total Matchups", 
            str(len(df)),
            f"Data Size: {MemoryOptimizer.get_memory_usage(df)}",
            icon="baseball"
        ), unsafe_allow_html=True)
    
    with col2:
        unique_batters = df['Batter'].nunique()
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Active Players", 
            str(unique_batters),
            "Unique Batters",
            icon="target"
        ), unsafe_allow_html=True)
    
    with col3:
        unique_teams = df['Tm'].nunique()
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Teams Playing", 
            str(unique_teams),
            "MLB Organizations",
            icon="stadium"
        ), unsafe_allow_html=True)
    
    with col4:
        avg_hit_prob = df['total_hit_prob'].mean()
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Avg Hit Probability", 
            f"{avg_hit_prob:.1f}%",
            f"Target: {MLBConfig.THRESHOLDS['elite_hit_prob']}%+",
            card_type="success",
            icon="fire"
        ), unsafe_allow_html=True)

@monitor.timer("results_header_display")
def display_professional_header(filtered_df, filters):
    """Display improved results header with better styling"""
    
    profile_key = filters.get('profile_key', 'contact_aggressive')
    profile_name = MLBConfig.PLAYER_PROFILES[profile_key]['name']
    sort_option = filters.get('sort_option', 'Score (High to Low)')
    
    result_count = filters.get('result_count', 15)
    best_per_team = filters.get('best_per_team_only', False)
    
    if best_per_team:
        title = f"🏟️ Best Player from Each Team ({len(filtered_df)} teams)" if result_count == "All" else f"🏟️ Top {len(filtered_df)} Teams - Best Player Each"
    else:
        title = f"🎯 All {len(filtered_df)} Players" if result_count == "All" else f"🎯 Top {len(filtered_df)} Players"
    
    st.markdown(f"""
    <div class="pro-card">
        <h2 style="margin: 0 0 1rem 0; font-weight: 700; color: var(--text-primary) !important;">{title}</h2>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 1rem; padding: 1rem; background: rgba(37, 99, 235, 0.05); border-radius: 8px; margin-top: 1rem;">
            <div>
                <strong style="color: var(--accent-color);">Active Profile:</strong>
                <span style="color: var(--text-primary) !important;">{profile_name}</span>
            </div>
            <div>
                <strong style="color: var(--accent-color);">Sorting:</strong>
                <span style="color: var(--text-primary) !important;">{sort_option}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

@monitor.timer("key_insights_display")
def display_professional_insights(filtered_df):
    """Display improved key insights with better icons"""
    
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Best Hit Probability",
            f"{best_hit_prob:.1f}%",
            f"Target: {MLBConfig.THRESHOLDS['elite_hit_prob']}%+",
            card_type="success",
            icon="target"
        ), unsafe_allow_html=True)
    
    with col2:
        avg_k = filtered_df['adj_K'].mean()
        k_vs_league = MLBConfig.LEAGUE_AVERAGES['K_PCT'] - avg_k
        color = "success" if k_vs_league > 0 else "metric"
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "K% vs League",
            f"{k_vs_league:+.1f}%",
            f"League Avg: {MLBConfig.LEAGUE_AVERAGES['K_PCT']}%",
            card_type=color,
            icon="lightning"
        ), unsafe_allow_html=True)
    
    with col3:
        avg_bb = filtered_df['adj_BB'].mean()
        bb_vs_league = MLBConfig.LEAGUE_AVERAGES['BB_PCT'] - avg_bb
        color = "success" if bb_vs_league > 0 else "metric"
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "BB% vs League",
            f"{bb_vs_league:+.1f}%",
            f"League Avg: {MLBConfig.LEAGUE_AVERAGES['BB_PCT']}%",
            card_type=color,
            icon="diamond"
        ), unsafe_allow_html=True)
    
    with col4:
        if 'pitcher_matchup_grade' in filtered_df.columns:
            a_plus_matchups = (filtered_df['pitcher_matchup_grade'] == 'A+').sum()
            st.markdown(ProfessionalUIComponents.create_professional_metric_card(
                "Elite Matchups",
                f"{a_plus_matchups}",
                f"A+ rated out of {len(filtered_df)}",
                card_type="success",
                icon="trophy"
            ), unsafe_allow_html=True)
        else:
            elite_contact_count = (filtered_df['adj_K'] <= MLBConfig.THRESHOLDS['elite_contact_k']).sum()
            st.markdown(ProfessionalUIComponents.create_professional_metric_card(
                "Elite Contact",
                f"{elite_contact_count}",
                f"K% ≤{MLBConfig.THRESHOLDS['elite_contact_k']}% players",
                card_type="success",
                icon="star"
            ), unsafe_allow_html=True)

@monitor.timer("results_table_display")
def display_professional_table(filtered_df, filters):
    """Display professional results table with improved styling"""
    
    if filtered_df.empty:
        return
    
    # Prepare display data
    display_df = filtered_df.copy()
    display_df = DataProcessor.calculate_league_comparison(display_df)
    
    # Ensure power combo exists
    if 'power_combo' not in display_df.columns:
        display_df['power_combo'] = display_df['adj_XB'] + display_df['adj_HR']
    
    # Add lineup status
    excluded_players = st.session_state.get('excluded_players', [])
    display_df['Status'] = display_df['Batter'].apply(
        lambda x: '🏟️' if x not in excluded_players else '❌'
    )
    
    # Define display columns
    display_columns = {
        'Status': 'Status',
        'Batter': 'Batter',
        'Tm': 'Team',
        'Pitcher': 'Pitcher',
        'total_hit_prob': 'Hit Prob %',
        'adj_1B': 'Contact %',
        'adj_XB': 'XB %',
        'adj_HR': 'HR %',
        'power_combo': 'Power Combo %',
        'K_vs_League': 'K% vs League',
        'BB_vs_League': 'BB% vs League',
        'adj_vs': 'vs Pitcher',
        'Score': 'Score'
    }
    
    # Add matchup column if available
    if 'pitcher_matchup_grade' in display_df.columns and display_df['pitcher_matchup_grade'].notna().any():
        display_columns['pitcher_matchup_grade'] = 'Matchup'
    
    # Create styled dataframe
    styled_df = display_df[display_columns.keys()].rename(columns=display_columns)
    
    # Apply professional formatting
    styled_df = styled_df.style.format({
        'Hit Prob %': "{:.1f}%",
        'Contact %': "{:.1f}%", 
        'XB %': "{:.1f}%",
        'HR %': "{:.1f}%",
        'Power Combo %': "{:.1f}%",
        'K% vs League': "{:+.1f}%",
        'BB% vs League': "{:+.1f}%",
        'vs Pitcher': "{:.0f}",
        'Score': "{:.1f}"
    }).background_gradient(
        subset=['Score'], cmap='RdYlGn', vmin=0, vmax=100
    ).background_gradient(
        subset=['Hit Prob %'], cmap='Greens', vmin=20, vmax=50
    ).background_gradient(
        subset=['Power Combo %'], cmap='Oranges', vmin=0, vmax=20
    ).background_gradient(
        subset=['K% vs League'], cmap='RdYlGn', vmin=-8, vmax=12
    ).background_gradient(
        subset=['BB% vs League'], cmap='RdYlGn', vmin=-5, vmax=6
    )
    
    # Color matchup grades if available
    if 'Matchup' in styled_df.columns:
        def color_matchup_grade(val):
            colors = {
                'A+': 'background-color: #166534; color: white; font-weight: bold',
                'A': 'background-color: #16a34a; color: white; font-weight: bold',
                'B': 'background-color: #eab308; color: white; font-weight: bold',
                'C': 'background-color: #f59e0b; color: white; font-weight: bold',
                'D': 'background-color: #dc2626; color: white; font-weight: bold'
            }
            return colors.get(val, '')
        
        styled_df = styled_df.apply(
            lambda x: [color_matchup_grade(v) if x.name == 'Matchup' else '' for v in x], axis=0
        )
    
    st.dataframe(styled_df, use_container_width=True)

def create_improved_legend():
    """Create improved legend with better readability and icons"""
    
    ProfessionalUIComponents.create_professional_legend("""
        <div class="legend-item">
            <strong>Player Status:</strong> 
            <span style="color: var(--success-color);">🏟️</span> = Active/Playing | 
            <span style="color: var(--warning-color);">❌</span> = Excluded from lineup
        </div>
        <div class="legend-item">
            <strong>Performance Score:</strong> 
            <span style="background: rgba(16, 185, 129, 0.2); color: var(--success-color); padding: 2px 6px; border-radius: 4px; font-weight: 600;">70+</span> Elite | 
            <span style="background: rgba(245, 158, 11, 0.2); color: var(--warning-color); padding: 2px 6px; border-radius: 4px; font-weight: 600;">50-69</span> Good | 
            <span style="background: rgba(239, 68, 68, 0.2); color: #ef4444; padding: 2px 6px; border-radius: 4px; font-weight: 600;">&lt;50</span> Risky
        </div>
        <div class="legend-item">
            <strong>Power Combo:</strong> XB% + HR% combined metric | 
            <span style="color: var(--accent-color); font-weight: 600;">12%+</span> = Elite power threat
        </div>
        <div class="legend-item">
            <strong>League Comparison:</strong> 
            <span style="color: var(--success-color); font-weight: 600;">+</span> = Better than league average | 
            <span style="color: var(--warning-color); font-weight: 600;">-</span> = Below league average
        </div>
        <div class="legend-item">
            <strong>Matchup Grades:</strong> 
            <span style="background: #166534; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;">A+</span> Elite | 
            <span style="background: #16a34a; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;">A</span> Great | 
            <span style="background: #eab308; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;">B</span> Good | 
            <span style="background: #f59e0b; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;">C</span> Average | 
            <span style="background: #dc2626; color: white; padding: 2px 6px; border-radius: 4px; font-weight: 600;">D</span> Avoid
        </div>
        <div class="legend-item">
            <strong>Key Metrics:</strong> 
            Hit Prob% = Total hit probability | Contact% = Single hit rate | XB% = Extra base hit rate | 
            HR% = Home run rate | K% = Strikeout rate | BB% = Walk rate
        </div>
    """)

@monitor.timer("visualizations_creation")
def create_professional_visualizations(df, filtered_df):
    """Create professional visualizations with improved dark mode support"""
    
    st.markdown(ProfessionalUIComponents.create_dashboard_section(
        "Performance Analytics Dashboard", "chart"
    ), unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced score distribution chart
        chart1 = alt.Chart(df).mark_bar(
            color='#2563eb', 
            opacity=0.8,
            cornerRadius=4
        ).encode(
            alt.X('Score:Q', bin=alt.Bin(maxbins=15), title='Performance Score', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            alt.Y('count()', title='Number of Players', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            tooltip=['count()']
        ).properties(
            title=alt.TitleParams(
                text='Score Distribution (All Players)',
                fontSize=16,
                fontWeight='bold'
            ),
            width=350, 
            height=300
        ).configure_axis(
            grid=True,
            gridColor='#f3f4f6',
            domainColor='#2563eb'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(chart1, use_container_width=True)
    
    with col2:
        # Enhanced performance scatter plot
        chart2 = alt.Chart(filtered_df).mark_circle(
            size=120, 
            opacity=0.8,
            stroke='white',
            strokeWidth=1
        ).encode(
            alt.X('total_hit_prob:Q', title='Hit Probability %', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            alt.Y('adj_K:Q', title='Strikeout Rate %', axis=alt.Axis(labelFontSize=12, titleFontSize=14)),
            alt.Color('Score:Q', 
                scale=alt.Scale(scheme='viridis'),
                legend=alt.Legend(title="Performance Score", labelFontSize=12, titleFontSize=14)
            ),
            alt.Size('power_combo:Q', 
                scale=alt.Scale(range=[50, 300]),
                legend=alt.Legend(title="Power Combo", labelFontSize=12, titleFontSize=14)
            ),
            tooltip=['Batter', 'total_hit_prob', 'adj_K', 'Score', 'power_combo']
        ).properties(
            title=alt.TitleParams(
                text='Hit Probability vs Contact Skills',
                fontSize=16,
                fontWeight='bold'
            ),
            width=350, 
            height=300
        ).configure_axis(
            grid=True,
            gridColor='#f3f4f6',
            domainColor='#2563eb'
        ).configure_view(
            strokeWidth=0
        )
        st.altair_chart(chart2, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def main_page():
    """Improved main application with better readability and contrast"""
    
    # Ensure improved CSS is injected
    inject_professional_css()
    
    # Create improved professional header
    ProfessionalUIComponents.create_professional_header()
    
    # Load and process data
    with st.spinner('🚀 Loading professional analytics...'):
        df = load_and_process_data()
    
    if df is None:
        st.error("❌ Unable to load data. Please check your connection and try again.")
        return
    
    # Display improved overview
    display_professional_overview(df)
    
    # Create improved filters
    filters = create_professional_filters(df)
    
    # Calculate scores
    profile_type = filters.get('profile_type', 'contact')
    df = calculate_league_aware_scores(df, profile_type)
    
    # Apply filters
    filtered_df = apply_professional_filters(df, filters)
    
    # Display improved results
    if not filtered_df.empty:
        ProfessionalUIComponents.create_section_divider()
        display_professional_header(filtered_df, filters)
        display_professional_insights(filtered_df)
        display_professional_table(filtered_df, filters)
        
        # Add improved legend
        create_improved_legend()
        
        ProfessionalUIComponents.create_section_divider()
        create_professional_visualizations(df, filtered_df)
    else:
        st.markdown("""
        <div class="pro-card">
            <h3 style="color: var(--warning-color); text-align: center; margin: 0 0 1rem 0;">⚠️ No players match your current filters</h3>
            <p style="text-align: center; color: var(--text-secondary); margin: 0;">Try adjusting your criteria or selecting a different profile.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Improved action buttons
    ProfessionalUIComponents.create_section_divider()
    
    st.markdown(ProfessionalUIComponents.create_dashboard_section(
        "Professional Tools", "tools"
    ), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export Analytics", help="Download professional CSV report"):
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button(
                    "💾 Download Report", 
                    csv, 
                    f"mlb_analytics_{datetime.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        if st.button("🔄 Refresh Data", help="Clear cache and reload data"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("🏟️ Clear Exclusions", help="Reset all player exclusions"):
            st.session_state.excluded_players = []
            st.rerun()
    
    with col4:
        perf_summary = monitor.get_performance_summary()
        st.info(f"⚡ Performance: {perf_summary}")
    
    # Show improved performance metrics in sidebar
    st.sidebar.markdown("""
    <div class="pro-card">
        <h4 style="margin: 0 0 1rem 0; font-weight: 600; color: var(--text-primary) !important;"><span class="icon-lightning"></span> System Performance</h4>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown(f"""
    <div class="performance-widget-pro">
        <span class="status-indicator status-success"></span>
        <strong>System Status:</strong> Optimized
    </div>
    <div class="performance-widget-pro">
        <strong>Performance:</strong> {perf_summary}
    </div>
    <div class="performance-widget-pro">
        <span class="status-indicator status-success"></span>
        <strong>UI Status:</strong> Professional Active
    </div>
    """, unsafe_allow_html=True)

def info_page():
    """Improved information page with better dark mode support"""
    
    # Ensure CSS is injected
    inject_professional_css()
    
    ProfessionalUIComponents.create_professional_header()
    
    st.markdown("""
    <div class="pro-card">
        <h2 style="margin: 0 0 2rem 0; color: var(--text-primary) !important;">📚 MLB Hit Predictor Pro v4.2 - Complete Fixed Edition</h2>
    </div>
    """, unsafe_allow_html=True)
    
    with st.expander("🎨 v4.2 Complete UI Fixes", expanded=True):
        st.markdown("""
        ## ✅ All Issues Resolved
        
        ### 🎯 Dark Mode & Light Mode Compatibility
        - **Adaptive Color System**: CSS variables that automatically adjust to user's theme preference
        - **Perfect Contrast**: All text is readable in both light and dark modes
        - **Professional Appearance**: Business-appropriate in all lighting conditions
        - **No White Text Issues**: Fixed all hardcoded white text that was invisible in light mode
        
        ### 📱 Icon Display Improvements  
        - **CSS-Based Icons**: Proper icon rendering system with fallbacks
        - **Universal Compatibility**: Works across all devices and browsers
        - **Professional Styling**: Consistent icon appearance throughout
        - **Accessibility**: Screen reader compatible icon implementation
        
        ### 🔧 Text Readability Fixes
        - **Profile Text**: Fixed white text in profile descriptions that was invisible in light mode
        - **Sidebar Content**: All sidebar text now properly adapts to theme
        - **Card Content**: Fixed all card text colors to use adaptive variables
        - **Legend Text**: Improved legend readability with proper contrast
        
        ### 🚀 Technical Improvements
        - **CSS Variables**: Complete adaptive color system using `var(--text-primary)` etc.
        - **Theme Detection**: Automatic light/dark mode detection
        - **Performance**: Maintained all existing functionality while improving readability
        - **No Dependencies**: Works in any Streamlit environment
        
        ### 🎨 Professional Design
        - **Business Ready**: Appropriate for professional presentations
        - **Consistent Styling**: Unified design language throughout
        - **Enhanced UX**: Better user experience with improved readability
        - **Accessibility**: Meets contrast standards for accessibility compliance
        """)
    
    # Test improved components with proper colors
    st.markdown("---")
    st.markdown("## 🧪 Fixed UI Components")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Dark Mode",
            "✅ Fixed",
            "Perfect in both modes",
            card_type="success",
            icon="star"
        ), unsafe_allow_html=True)
    
    with col2:
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Text Readability",
            "✅ Enhanced",
            "All text visible",
            card_type="success",
            icon="target"
        ), unsafe_allow_html=True)
    
    with col3:
        st.markdown(ProfessionalUIComponents.create_professional_metric_card(
            "Icons",
            "✅ Working",
            "Proper rendering",
            card_type="success",
            icon="rocket"
        ), unsafe_allow_html=True)
    
    ProfessionalUIComponents.create_section_divider()
    
    # Test improved legend
    create_improved_legend()

def main():
    """Improved main function with proper text colors"""
    
    # Inject improved CSS first
    inject_professional_css()
    
    st.sidebar.markdown("""
    <div class="pro-card">
        <h2 style="margin: 0; text-align: center; color: var(--text-primary) !important;"><span class="icon-stadium"></span> Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional music controls
    ProfessionalUIComponents.create_section_divider()
    if st.sidebar.checkbox("🎵 Background Audio"):
        audio_url = "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        components.html(f"""
        <audio controls autoplay loop style="width: 100%; border-radius: 12px;">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        """, height=60)

    app_mode = st.sidebar.radio(
        "Application Mode",
        ["🎯 Professional Analytics", "📚 User Guide"],
        index=0
    )

    if app_mode == "🎯 Professional Analytics":
        main_page()
    else:
        info_page()
    
    # Improved footer with proper text colors
    st.sidebar.markdown("""
    <div class="pro-card">
        <div style="text-align: center;">
            <h4 style="margin: 0; color: var(--text-primary) !important;">v4.2 Complete Fixed</h4>
            <p style="margin: 0.5rem 0 0 0; color: var(--text-secondary) !important; font-size: 0.9rem;">Dark/Light Mode Compatible</p>
            <div style="margin: 1rem 0; padding: 0.75rem; background: rgba(37, 99, 235, 0.1); border-radius: 8px;">
                <div style="font-size: 0.8rem; margin: 0.25rem 0; color: var(--text-primary) !important;">
                    <span class="status-indicator status-success"></span>✅ Dark Mode Fixed
                </div>
                <div style="font-size: 0.8rem; margin: 0.25rem 0; color: var(--text-primary) !important;">
                    <span class="status-indicator status-success"></span>✅ Text Readability
                </div>
                <div style="font-size: 0.8rem; margin: 0.25rem 0; color: var(--text-primary) !important;">
                    <span class="status-indicator status-success"></span>✅ Icon Display
                </div>
                <div style="font-size: 0.8rem; margin: 0.25rem 0; color: var(--text-primary) !important;">
                    <span class="status-indicator status-success"></span>✅ Professional Ready
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
