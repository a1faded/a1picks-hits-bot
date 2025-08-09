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

# ==================== APPLE-INSPIRED UI STYLING ====================

def inject_apple_inspired_css():
    """Inject sleek, Apple-inspired CSS styling"""
    st.markdown("""
    <style>
    /* Import SF Pro Display (fallback to system fonts) */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* ==================== APPLE-INSPIRED GLOBAL STYLES ==================== */
    .stApp {
        background: #f5f5f7;
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif;
        color: #1d1d1f;
        line-height: 1.4705882353;
        font-weight: 400;
        letter-spacing: -0.022em;
    }
    
    .main .block-container {
        padding: 0;
        background: transparent;
        max-width: 1200px;
        margin: 0 auto;
    }
    
    /* ==================== SLEEK HEADER ==================== */
    .apple-header {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 2rem 0;
        margin: 2rem 0;
        border-radius: 18px;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
        text-align: center;
        position: relative;
    }
    
    .apple-title {
        font-size: 3rem;
        font-weight: 700;
        margin: 0;
        color: #1d1d1f;
        letter-spacing: -0.015em;
        line-height: 1.0834933333;
    }
    
    .apple-subtitle {
        font-size: 1.125rem;
        font-weight: 400;
        margin: 0.5rem 0 0 0;
        color: #86868b;
        letter-spacing: -0.022em;
    }
    
    /* ==================== MINIMALIST CARDS ==================== */
    .apple-card {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border-radius: 18px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
    }
    
    .apple-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.08);
        border-color: rgba(0, 0, 0, 0.08);
    }
    
    .apple-card h2, .apple-card h3, .apple-card h4 {
        color: #1d1d1f !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        letter-spacing: -0.022em !important;
    }
    
    .apple-card p {
        color: #86868b !important;
        font-size: 0.875rem !important;
        line-height: 1.4285714286 !important;
        margin: 0 !important;
    }
    
    /* ==================== ELEGANT METRIC CARDS ==================== */
    .metric-card-apple {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 1.5rem;
        border-radius: 16px;
        margin: 0.75rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
        transition: all 0.3s cubic-bezier(0.25, 0.46, 0.45, 0.94);
        text-align: center;
    }
    
    .metric-card-apple:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.08);
    }
    
    .success-card-apple {
        background: linear-gradient(135deg, rgba(52, 199, 89, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%);
        border-color: rgba(52, 199, 89, 0.2);
    }
    
    .warning-card-apple {
        background: linear-gradient(135deg, rgba(255, 149, 0, 0.1) 0%, rgba(255, 255, 255, 0.8) 100%);
        border-color: rgba(255, 149, 0, 0.2);
    }
    
    .metric-title-apple {
        font-size: 0.75rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #86868b;
        margin-bottom: 0.5rem;
    }
    
    .metric-value-apple {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1d1d1f;
        line-height: 1;
        margin: 0.25rem 0;
        letter-spacing: -0.015em;
    }
    
    .metric-subtitle-apple {
        font-size: 0.8125rem;
        color: #86868b;
        font-weight: 400;
        margin-top: 0.25rem;
    }
    
    /* ==================== MODERN TABLE STYLING ==================== */
    .dataframe {
        border: none !important;
        border-radius: 16px !important;
        overflow: hidden !important;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        margin: 1.5rem 0 !important;
        border: 0.5px solid rgba(0, 0, 0, 0.04) !important;
    }
    
    .dataframe thead th {
        background: #1d1d1f !important;
        color: #f5f5f7 !important;
        font-weight: 600 !important;
        padding: 1rem 0.75rem !important;
        border: none !important;
        font-size: 0.8125rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        text-align: center !important;
    }
    
    .dataframe tbody td {
        padding: 0.875rem 0.75rem !important;
        border-bottom: 0.5px solid rgba(0, 0, 0, 0.04) !important;
        border-left: none !important;
        border-right: none !important;
        font-weight: 500 !important;
        font-size: 0.8125rem !important;
        text-align: center !important;
        color: #1d1d1f !important;
        background: rgba(255, 255, 255, 0.6) !important;
    }
    
    .dataframe tbody tr:hover {
        background: rgba(0, 122, 255, 0.04) !important;
        transition: background-color 0.2s ease !important;
    }
    
    .dataframe tbody tr:last-child td {
        border-bottom: none !important;
    }
    
    /* ==================== CLEAN SIDEBAR ==================== */
    .sidebar .sidebar-content {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border-radius: 18px !important;
        padding: 1.5rem !important;
        margin: 1rem 0.5rem !important;
        border: 0.5px solid rgba(0, 0, 0, 0.04) !important;
    }
    
    /* ==================== APPLE-STYLE BUTTONS ==================== */
    .stButton button {
        background: #007AFF !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 0.875rem !important;
        letter-spacing: -0.022em !important;
        transition: all 0.2s cubic-bezier(0.25, 0.46, 0.45, 0.94) !important;
        box-shadow: 0 1px 3px rgba(0, 122, 255, 0.3) !important;
    }
    
    .stButton button:hover {
        background: #0056CC !important;
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0, 122, 255, 0.4) !important;
    }
    
    .stButton button:active {
        transform: translateY(0) !important;
    }
    
    /* ==================== CLEAN FORM ELEMENTS ==================== */
    .stSelectbox > div > div,
    .stMultiSelect > div > div {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border: 0.5px solid rgba(0, 0, 0, 0.08) !important;
        border-radius: 12px !important;
        padding: 0.75rem !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
    }
    
    .stSelectbox > div > div:focus-within,
    .stMultiSelect > div > div:focus-within {
        border-color: #007AFF !important;
        box-shadow: 0 0 0 3px rgba(0, 122, 255, 0.1) !important;
    }
    
    .stSlider > div > div > div {
        background: #007AFF !important;
    }
    
    /* ==================== MODERN CHARTS ==================== */
    .vega-embed {
        border-radius: 16px !important;
        overflow: hidden !important;
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        padding: 1.5rem !important;
        margin: 1.5rem 0 !important;
        border: 0.5px solid rgba(0, 0, 0, 0.04) !important;
    }
    
    /* ==================== ELEGANT LEGEND ==================== */
    .apple-legend {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 1.5rem;
        border-radius: 16px;
        color: #1d1d1f;
        margin: 2rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
    }
    
    .legend-title-apple {
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 1rem;
        color: #1d1d1f;
        letter-spacing: -0.022em;
    }
    
    .legend-item-apple {
        margin: 0.75rem 0;
        font-weight: 400;
        color: #1d1d1f;
        font-size: 0.875rem;
        line-height: 1.4285714286;
    }
    
    .legend-item-apple strong {
        color: #1d1d1f;
        font-weight: 600;
    }
    
    /* ==================== SLEEK PERFORMANCE WIDGETS ==================== */
    .performance-widget-apple {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        padding: 1rem;
        border-radius: 12px;
        margin: 0.5rem 0;
        font-size: 0.8125rem;
        font-weight: 500;
        transition: all 0.2s ease;
        color: #1d1d1f;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
    }
    
    .performance-widget-apple:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    }
    
    /* ==================== MINIMAL STATUS INDICATORS ==================== */
    .status-indicator-apple {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-success-apple {
        background: #34C759;
    }
    
    .status-warning-apple {
        background: #FF9500;
    }
    
    /* ==================== SUBTLE SECTION DIVIDERS ==================== */
    .section-divider-apple {
        height: 1px;
        background: rgba(0, 0, 0, 0.08);
        margin: 3rem 0;
        border: none;
    }
    
    /* ==================== CLEAN EXPANDERS ==================== */
    .streamlit-expanderHeader {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border-radius: 12px !important;
        padding: 1rem !important;
        font-weight: 600 !important;
        border: 0.5px solid rgba(0, 0, 0, 0.04) !important;
        transition: all 0.2s ease !important;
        color: #1d1d1f !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: rgba(0, 122, 255, 0.04) !important;
        border-color: #007AFF !important;
    }
    
    .streamlit-expanderContent {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: saturate(180%) blur(20px) !important;
        -webkit-backdrop-filter: saturate(180%) blur(20px) !important;
        border-radius: 0 0 12px 12px !important;
        padding: 1.5rem !important;
        border: 0.5px solid rgba(0, 0, 0, 0.04) !important;
        border-top: none !important;
    }
    
    /* ==================== SMOOTH LOADING ANIMATION ==================== */
    .loading-spinner-apple {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 2px solid rgba(0, 122, 255, 0.2);
        border-top: 2px solid #007AFF;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    
    /* ==================== DASHBOARD SECTIONS ==================== */
    .dashboard-section-apple {
        background: rgba(255, 255, 255, 0.8);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border-radius: 18px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 0.5px solid rgba(0, 0, 0, 0.04);
    }
    
    .dashboard-title-apple {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1d1d1f;
        margin-bottom: 1.5rem;
        text-align: center;
        letter-spacing: -0.022em;
    }
    
    /* ==================== RESPONSIVE DESIGN ==================== */
    @media (max-width: 768px) {
        .main .block-container {
            padding: 0 1rem;
        }
        
        .apple-title {
            font-size: 2.5rem;
        }
        
        .metric-value-apple {
            font-size: 2rem;
        }
        
        .apple-card {
            padding: 1rem;
            margin: 0.75rem 0;
        }
        
        .metric-card-apple {
            padding: 1rem;
        }
    }
    
    /* ==================== TYPOGRAPHY REFINEMENTS ==================== */
    h1, h2, h3, h4, h5, h6 {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Inter', sans-serif !important;
        font-weight: 600 !important;
        line-height: 1.2 !important;
        letter-spacing: -0.022em !important;
        color: #1d1d1f !important;
    }
    
    p, span, div {
        font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Text', 'Inter', sans-serif !important;
        line-height: 1.4705882353 !important;
        color: #1d1d1f !important;
    }
    
    /* ==================== ACCESSIBILITY & CONTRAST ==================== */
    .stApp a {
        color: #007AFF !important;
        text-decoration: none !important;
    }
    
    .stApp a:hover {
        text-decoration: underline !important;
    }
    
    /* ==================== FOCUS STYLES ==================== */
    button:focus,
    select:focus,
    input:focus {
        outline: 2px solid #007AFF !important;
        outline-offset: 2px !important;
    }
    
    /* ==================== STREAMLIT OVERRIDES ==================== */
    .stMarkdown {
        color: #1d1d1f !important;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3 {
        color: #1d1d1f !important;
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
                    <div class="performance-widget-apple">
                        <span class="status-indicator-apple status-warning-apple"></span>
                        <strong>{operation_name}:</strong> {duration:.2f}s
                    </div>
                    """, unsafe_allow_html=True)
                    self.warnings_shown.add(operation_name)
                elif duration < 0.5 and operation_name in self.warnings_shown:
                    st.sidebar.markdown(f"""
                    <div class="performance-widget-apple">
                        <span class="status-indicator-apple status-success-apple"></span>
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

# ==================== APPLE-INSPIRED UI COMPONENTS ====================

class AppleUIComponents:
    """Apple-inspired UI component generators"""
    
    @staticmethod
    def create_header():
        """Create sleek Apple-inspired header"""
        st.markdown("""
        <div class="apple-header">
            <h1 class="apple-title">⚾ MLB Hit Predictor</h1>
            <p class="apple-subtitle">Professional Analytics Platform</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_metric_card(title: str, value: str, subtitle: str = "", 
                          card_type: str = "metric") -> str:
        """Generate Apple-inspired metric cards"""
        card_class = f"{card_type}-card-apple"
        return f"""
        <div class="{card_class}">
            <div class="metric-title-apple">{title}</div>
            <div class="metric-value-apple">{value}</div>
            <div class="metric-subtitle-apple">{subtitle}</div>
        </div>
        """
    
    @staticmethod
    def create_section_divider():
        """Create minimal section divider"""
        st.markdown('<div class="section-divider-apple"></div>', unsafe_allow_html=True)
    
    @staticmethod
    def create_legend(content: str):
        """Create Apple-inspired legend"""
        st.markdown(f"""
        <div class="apple-legend">
            <div class="legend-title-apple">📊 Analytics Guide</div>
            <div class="legend-content">{content}</div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_loading_indicator(text: str = "Loading"):
        """Create Apple-inspired loading indicator"""
        return f"""
        <div style="display: flex; align-items: center; justify-content: center; padding: 2rem;">
            <div class="loading-spinner-apple"></div>
            <span style="margin-left: 1rem; font-weight: 500; color: #1d1d1f;">{text}...</span>
        </div>
        """
    
    @staticmethod
    def create_dashboard_section(title: str, content: str = ""):
        """Create Apple-inspired dashboard section"""
        return f"""
        <div class="dashboard-section-apple">
            <h2 class="dashboard-title-apple">{title}</h2>
            {content}
        </div>
        """

# ==================== MAIN APPLICATION CONFIGURATION ====================

# Configure Streamlit page
st.set_page_config(
    page_title="A1PICKS MLB Hit Predictor Pro v5.0",
    layout="wide",
    page_icon="⚾",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'mailto:your@email.com',
        'Report a bug': "https://github.com/yourrepo/issues",
    }
)

# Inject Apple-inspired CSS
inject_apple_inspired_css()

# ==================== DATA LOADING AND PROCESSING ====================

@monitor.timer("data_loading")
@st.cache_data(ttl=MLBConfig.CACHE_TTL)
def load_and_process_data():
    """Enhanced data loading with concurrent processing and validation"""
    
    # Show Apple loading indicator
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        AppleUIComponents.create_loading_indicator("Loading MLB data"),
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

# ==================== APPLE-INSPIRED FILTERING SYSTEM ====================

@monitor.timer("filter_creation")
def create_apple_filters(df=None):
    """Create Apple-inspired filtering system"""
    
    st.sidebar.markdown("""
    <div class="apple-card">
        <h3 style="color: #1d1d1f; margin: 0 0 0.5rem 0; font-weight: 600;">🎯 Filters</h3>
        <p style="color: #86868b; margin: 0; font-size: 0.875rem;">Baseball Analytics</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize session state
    if 'excluded_players' not in st.session_state:
        st.session_state.excluded_players = []
    
    filters = {}
    
    # League context
    st.sidebar.markdown("""
    <div class="apple-card">
        <h4 style="color: #1d1d1f; margin: 0 0 1rem 0; font-weight: 600;">📊 2024 League</h4>
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(0, 122, 255, 0.04); border-radius: 8px;">
            <span style="font-weight: 500; color: #1d1d1f;">K% Average:</span>
            <span style="color: #007AFF; font-weight: 600;">22.6%</span>
        </div>
        <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(0, 122, 255, 0.04); border-radius: 8px;">
            <span style="font-weight: 500; color: #1d1d1f;">BB% Average:</span>
            <span style="color: #007AFF; font-weight: 600;">8.5%</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Show data status
    if df is not None and not df.empty:
        memory_usage = MemoryOptimizer.get_memory_usage(df)
        st.sidebar.markdown(f"""
        <div class="apple-card">
            <h4 style="color: #1d1d1f; margin: 0 0 1rem 0; font-weight: 600;">📈 Data Status</h4>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(52, 199, 89, 0.04); border-radius: 8px;">
                <span style="font-weight: 500; color: #1d1d1f;">Matchups:</span>
                <span style="color: #34C759; font-weight: 600;">{len(df):,}</span>
            </div>
            <div style="display: flex; justify-content: space-between; margin: 0.5rem 0; padding: 0.5rem; background: rgba(52, 199, 89, 0.04); border-radius: 8px;">
                <span style="font-weight: 500; color: #1d1d1f;">Memory:</span>
                <span style="color: #34C759; font-weight: 600;">{memory_usage}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    AppleUIComponents.create_section_divider()
    
    # Player profile selection
    st.sidebar.markdown("""
    <div class="apple-card">
        <h4 style="color: #1d1d1f; margin: 0 0 1rem 0; font-weight: 600;">🎯 Player Profile</h4>
    </div>
    """, unsafe_allow_html=True)
    
    profile_options = {v['name']: k for k, v in MLBConfig.PLAYER_PROFILES.items()}
    
    selected_profile_name = st.sidebar.selectbox(
        "Choose Profile",
        options=list(profile_options.keys()),
        index=0
    )
    
    selected_profile_key = profile_options[selected_profile_name]
    profile_config = MLBConfig.PLAYER_PROFILES[selected_profile_key]
    
    filters['profile_key'] = selected_profile_key
    filters['profile_type'] = profile_config['profile_type']
    
    # Show profile details
    st.sidebar.markdown(f"""
    <div class="apple-card">
        <h4 style="color: #1d1d1f; margin: 0 0 0.5rem 0; font-weight: 600;">{selected_profile_name}</h4>
        <p style="color: #86868b; margin: 0 0 1rem 0; font-style: italic; font-size: 0.875rem;">{profile_config['description']}</p>
        <div style="background: rgba(0, 122, 255, 0.04); padding: 1rem; border-radius: 12px;">
            <div style="margin: 0.25rem 0; color: #1d1d1f; font-size: 0.875rem;">
                <strong>Max K%:</strong> {profile_config.get('max_k', 'N/A')}
            </div>
            <div style="margin: 0.25rem 0; color: #1d1d1f; font-size: 0.875rem;">
                <strong>Max BB%:</strong> {profile_config.get('max_bb', 'N/A')}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Advanced options
    with st.sidebar.expander("⚙️ Advanced", expanded=False):
        filters['custom_max_k'] = st.slider(
            "Max K%", 5.0, 35.0, profile_config.get('max_k', 20.0), 0.5
        )
        filters['custom_max_bb'] = st.slider(
            "Max BB%", 2.0, 15.0, profile_config.get('max_bb', 8.0), 0.5
        )
        
        # Team filtering
        if df is not None and not df.empty:
            team_options = sorted(df['Tm'].unique().tolist())
            filters['selected_teams'] = st.multiselect(
                "Include Teams Only", options=team_options
            )
            filters['excluded_teams'] = st.multiselect(
                "Exclude Teams", options=team_options
            )
        
        filters['best_per_team_only'] = st.checkbox("🏟️ Best per team only", value=False)
        filters['sort_option'] = st.selectbox(
            "Sort By", options=list(MLBConfig.SORT_OPTIONS.keys()), index=0
        )
        filters['sort_col'], filters['sort_asc'] = MLBConfig.SORT_OPTIONS[filters['sort_option']]
        filters['result_count'] = st.selectbox(
            "Results", [10, 15, 20, 25, 30, "All"], index=1
        )
    
    # Lineup Management
    with st.sidebar.expander("🏟️ Lineup Management", expanded=False):
        if df is not None and not df.empty:
            all_players = sorted(df['Batter'].unique().tolist())
            excluded_players = st.multiselect(
                "Players NOT Playing Today",
                options=all_players,
                default=st.session_state.excluded_players
            )
            st.session_state.excluded_players = excluded_players
            filters['excluded_players'] = excluded_players
            
            if excluded_players:
                st.markdown(f"""
                <div style="background: rgba(255, 149, 0, 0.1); padding: 0.75rem; border-radius: 8px; margin: 0.5rem 0;">
                    <strong>🚫 Excluding {len(excluded_players)} players</strong>
                </div>
                """, unsafe_allow_html=True)
            
            if st.button("🔄 Clear Exclusions"):
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

# ==================== APPLE-INSPIRED DISPLAY FUNCTIONS ====================

def display_apple_overview(df):
    """Display Apple-inspired data overview"""
    
    if df is None or df.empty:
        st.error("No data available for analysis")
        return
    
    st.markdown(AppleUIComponents.create_dashboard_section(
        "📊 Today's Analytics"
    ), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(AppleUIComponents.create_metric_card(
            "Matchups", 
            str(len(df)),
            f"{MemoryOptimizer.get_memory_usage(df)}"
        ), unsafe_allow_html=True)
    
    with col2:
        unique_batters = df['Batter'].nunique()
        st.markdown(AppleUIComponents.create_metric_card(
            "Players", 
            str(unique_batters),
            "Active Today"
        ), unsafe_allow_html=True)
    
    with col3:
        unique_teams = df['Tm'].nunique()
        st.markdown(AppleUIComponents.create_metric_card(
            "Teams", 
            str(unique_teams),
            "Organizations"
        ), unsafe_allow_html=True)
    
    with col4:
        avg_hit_prob = df['total_hit_prob'].mean()
        st.markdown(AppleUIComponents.create_metric_card(
            "Avg Hit Prob", 
            f"{avg_hit_prob:.1f}%",
            f"Target: 35%+",
            card_type="success"
        ), unsafe_allow_html=True)

def display_apple_insights(filtered_df):
    """Display Apple-inspired key insights"""
    
    if filtered_df.empty:
        st.warning("⚠️ No players match your current filters")
        return
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        best_hit_prob = filtered_df['total_hit_prob'].iloc[0] if len(filtered_df) > 0 else 0
        st.markdown(AppleUIComponents.create_metric_card(
            "Best Hit Prob",
            f"{best_hit_prob:.1f}%",
            "Top Player",
            card_type="success"
        ), unsafe_allow_html=True)
    
    with col2:
        avg_k = filtered_df['adj_K'].mean()
        k_vs_league = 22.6 - avg_k
        color = "success" if k_vs_league > 0 else "metric"
        st.markdown(AppleUIComponents.create_metric_card(
            "K% vs League",
            f"{k_vs_league:+.1f}%",
            "vs 22.6% avg",
            card_type=color
        ), unsafe_allow_html=True)
    
    with col3:
        avg_bb = filtered_df['adj_BB'].mean()
        bb_vs_league = 8.5 - avg_bb
        color = "success" if bb_vs_league > 0 else "metric"
        st.markdown(AppleUIComponents.create_metric_card(
            "BB% vs League",
            f"{bb_vs_league:+.1f}%",
            "vs 8.5% avg",
            card_type=color
        ), unsafe_allow_html=True)
    
    with col4:
        elite_count = (filtered_df['Score'] >= 70).sum()
        st.markdown(AppleUIComponents.create_metric_card(
            "Elite Players",
            f"{elite_count}",
            f"Score 70+",
            card_type="success"
        ), unsafe_allow_html=True)

def display_professional_table(filtered_df, filters):
    """Display professional results table with Apple styling"""
    
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

def create_apple_legend():
    """Create Apple-inspired legend"""
    
    AppleUIComponents.create_legend("""
        <div class="legend-item-apple">
            <strong>Player Status:</strong> 
            🏟️ = Active | ❌ = Excluded
        </div>
        <div class="legend-item-apple">
            <strong>Performance Score:</strong> 
            <span style="background: rgba(52, 199, 89, 0.1); color: #1b5e20; padding: 2px 8px; border-radius: 6px; font-weight: 600;">70+</span> Elite | 
            <span style="background: rgba(255, 149, 0, 0.1); color: #e65100; padding: 2px 8px; border-radius: 6px; font-weight: 600;">50-69</span> Good | 
            <span style="background: rgba(255, 59, 48, 0.1); color: #c62828; padding: 2px 8px; border-radius: 6px; font-weight: 600;">&lt;50</span> Risky
        </div>
        <div class="legend-item-apple">
            <strong>Key Metrics:</strong> Hit Prob% = Total hit probability | XB% = Extra base rate | HR% = Home run rate
        </div>
    """)

def create_professional_visualizations(df, filtered_df):
    """Create professional visualizations with Apple styling"""
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Enhanced score distribution chart
        chart1 = alt.Chart(df).mark_bar(
            color='#007AFF', 
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
                fontWeight='bold',
                color='#1d1d1f'
            ),
            width=350, 
            height=300
        ).configure_axis(
            grid=True,
            gridColor='#f3f4f6',
            domainColor='#007AFF'
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
                fontWeight='bold',
                color='#1d1d1f'
            ),
            width=350, 
            height=300
        ).configure_axis(
            grid=True,
            gridColor='#f3f4f6',
            domainColor='#007AFF'
        )
        st.altair_chart(chart2, use_container_width=True)

# ==================== MAIN APPLICATION ====================

def main_apple_page():
    """Apple-inspired main page"""
    
    # Inject Apple CSS
    inject_apple_inspired_css()
    
    # Create header
    AppleUIComponents.create_header()
    
    # Load data
    with st.spinner('Loading analytics...'):
        df = load_and_process_data()
    
    if df is None:
        st.error("Unable to load data")
        return
    
    # Display overview
    display_apple_overview(df)
    
    # Create filters
    filters = create_apple_filters(df)
    
    # Calculate scores and apply filters
    profile_type = filters.get('profile_type', 'contact')
    df = calculate_league_aware_scores(df, profile_type)
    filtered_df = apply_professional_filters(df, filters)
    
    # Display results
    if not filtered_df.empty:
        AppleUIComponents.create_section_divider()
        
        # Header
        st.markdown(AppleUIComponents.create_dashboard_section(
            f"🎯 Top {len(filtered_df)} Players"
        ), unsafe_allow_html=True)
        
        # Insights
        display_apple_insights(filtered_df)
        
        # Table
        display_professional_table(filtered_df, filters)
        
        # Legend
        create_apple_legend()
        
        # Charts
        AppleUIComponents.create_section_divider()
        st.markdown(AppleUIComponents.create_dashboard_section(
            "📈 Performance Analytics"
        ), unsafe_allow_html=True)
        create_professional_visualizations(df, filtered_df)
    
    # Tools
    AppleUIComponents.create_section_divider()
    st.markdown(AppleUIComponents.create_dashboard_section(
        "🛠️ Tools"
    ), unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        if st.button("📊 Export"):
            if not filtered_df.empty:
                csv = filtered_df.to_csv(index=False)
                st.download_button("💾 Download", csv, f"mlb_analytics_{datetime.now().strftime('%Y%m%d')}.csv", mime="text/csv")
    
    with col2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()
    
    with col3:
        if st.button("🏟️ Clear"):
            st.session_state.excluded_players = []
            st.rerun()
    
    with col4:
        perf_summary = monitor.get_performance_summary()
        st.info(f"⚡ {perf_summary}")

def info_page():
    """Apple-inspired info page"""
    
    inject_apple_inspired_css()
    AppleUIComponents.create_header()
    
    st.markdown("""
    <div class="apple-card">
        <h2 style="color: #1d1d1f;">Apple-Inspired Design</h2>
        <p style="color: #86868b;">Clean, minimal, elegant analytics platform inspired by Apple's design principles.</p>
        
        <h3 style="color: #1d1d1f; margin-top: 2rem;">Key Features</h3>
        <ul style="color: #86868b;">
            <li>Glassmorphism effects with backdrop blur</li>
            <li>Apple's system colors and typography</li>
            <li>Smooth animations and transitions</li>
            <li>Clean, minimal interface design</li>
            <li>Professional data visualization</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

def main():
    """Apple-inspired main function"""
    
    inject_apple_inspired_css()
    
    st.sidebar.markdown("""
    <div class="apple-card">
        <h2 style="color: #1d1d1f; margin: 0; text-align: center; font-weight: 600;">🏟️ Navigation</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional music controls
    AppleUIComponents.create_section_divider()
    if st.sidebar.checkbox("🎵 Background Audio"):
        audio_url = "https://github.com/a1faded/a1picks-hits-bot/raw/refs/heads/main/Take%20Me%20Out%20to%20the%20Ballgame%20-%20Nancy%20Bea%20-%20Dodger%20Stadium%20Organ.mp3"
        components.html(f"""
        <audio controls autoplay loop style="width: 100%; border-radius: 12px;">
            <source src="{audio_url}" type="audio/mpeg">
        </audio>
        """, height=60)

    app_mode = st.sidebar.radio(
        "Mode",
        ["🎯 Analytics", "📚 Guide"],
        index=0
    )

    if app_mode == "🎯 Analytics":
        main_apple_page()
    else:
        info_page()
    
    # Show performance metrics in sidebar
    st.sidebar.markdown("""
    <div class="apple-card">
        <h4 style="color: #1d1d1f; margin: 0 0 1rem 0;">⚡ System Performance</h4>
    </div>
    """, unsafe_allow_html=True)
    
    perf_summary = monitor.get_performance_summary()
    st.sidebar.markdown(f"""
    <div class="performance-widget-apple">
        <span class="status-indicator-apple status-success-apple"></span>
        <strong>System:</strong> Optimized
    </div>
    <div class="performance-widget-apple">
        <strong>Performance:</strong> {perf_summary}
    </div>
    <div class="performance-widget-apple">
        <span class="status-indicator-apple status-success-apple"></span>
        <strong>UI:</strong> Apple Design Active
    </div>
    """, unsafe_allow_html=True)
    
    # Footer
    st.sidebar.markdown("""
    <div class="apple-card">
        <div style="text-align: center;">
            <h4 style="color: #1d1d1f; margin: 0; font-weight: 600;">v5.0 Apple</h4>
            <p style="margin: 0.5rem 0 0 0; color: #86868b; font-size: 0.875rem;">Sleek & Modern</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
