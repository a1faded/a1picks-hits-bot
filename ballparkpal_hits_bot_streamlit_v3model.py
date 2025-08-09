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
    
    /* ==================== GRADIENT ACCENTS ==================== */
    .gradient-accent {
        background: linear-gradient(135deg, #007AFF 0%, #5856D6 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* ==================== GLASS EFFECTS ==================== */
    .glass-effect {
        background: rgba(255, 255, 255, 0.25);
        backdrop-filter: saturate(180%) blur(20px);
        -webkit-backdrop-filter: saturate(180%) blur(20px);
        border: 0.5px solid rgba(255, 255, 255, 0.18);
    }
    
    /* ==================== SMOOTH SCROLLBARS ==================== */
    ::-webkit-scrollbar {
        width: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(0, 0, 0, 0.05);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0, 0, 0, 0.2);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0, 0, 0, 0.3);
    }
    </style>
    """, unsafe_allow_html=True)

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

# ==================== IMPROVED DISPLAY FUNCTIONS ====================

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
        filters['result_count'] = st.selectbox(
            "Results", [10, 15, 20, 25, 30, "All"], index=1
        )
    
    return filters

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
        st.info("⚡ Optimized")

def apple_main():
    """Apple-inspired main function"""
    
    inject_apple_inspired_css()
    
    st.sidebar.markdown("""
    <div class="apple-card">
        <h2 style="color: #1d1d1f; margin: 0; text-align: center; font-weight: 600;">🏟️ Navigation</h2>
    </div>
    """, unsafe_allow_html=True)

    app_mode = st.sidebar.radio(
        "Mode",
        ["🎯 Analytics", "📚 Guide"],
        index=0
    )

    if app_mode == "🎯 Analytics":
        main_apple_page()
    else:
        # Info page
        AppleUIComponents.create_header()
        st.markdown("""
        <div class="apple-card">
            <h2 style="color: #1d1d1f;">Apple-Inspired Design</h2>
            <p style="color: #86868b;">Clean, minimal, elegant analytics platform inspired by Apple's design principles.</p>
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
