import numpy as np
import pandas as pd
import streamlit as st
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

# ═══════════════════════════════════════════════════════════════════════════════
# 📱 SMARTPHONE PRICE PREDICTOR - Premium Dark Green Theme
# ═══════════════════════════════════════════════════════════════════════════════

# Page Configuration
st.set_page_config(
    page_title="📱 Smart Price Predictor",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎨 CUSTOM CSS - Dark Green Gradient Theme
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main App Background - Dark Green Gradient */
    .stApp {
        background: linear-gradient(135deg, #0a1f0a 0%, #0d2818 25%, #1a3a2a 50%, #0d2818 75%, #0a1f0a 100%);
        background-attachment: fixed;
    }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(10, 45, 25, 0.95) 0%, rgba(5, 30, 15, 0.98) 100%);
        border-right: 1px solid rgba(46, 204, 113, 0.2);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #e8f5e9;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #4caf50 !important;
        text-shadow: 0 0 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Main content text */
    .stMarkdown, p, span, label {
        color: #c8e6c9 !important;
    }
    
    /* Glassmorphism Cards */
    .glass-card {
        background: rgba(20, 60, 40, 0.4);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border-radius: 20px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        padding: 25px;
        margin: 15px 0;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.05);
        transition: all 0.3s ease;
    }
    
    .glass-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.4),
                    0 0 30px rgba(76, 175, 80, 0.15);
        border-color: rgba(76, 175, 80, 0.4);
    }
    
    /* Hero Section */
    .hero-container {
        background: linear-gradient(135deg, rgba(26, 72, 45, 0.6) 0%, rgba(15, 50, 30, 0.8) 100%);
        backdrop-filter: blur(15px);
        border-radius: 25px;
        padding: 40px;
        margin-bottom: 30px;
        border: 1px solid rgba(76, 175, 80, 0.3);
        text-align: center;
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(135deg, #4caf50 0%, #81c784 50%, #a5d6a7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 10px;
        animation: glow 3s ease-in-out infinite alternate;
    }
    
    @keyframes glow {
        from { filter: drop-shadow(0 0 10px rgba(76, 175, 80, 0.3)); }
        to { filter: drop-shadow(0 0 25px rgba(76, 175, 80, 0.6)); }
    }
    
    .hero-subtitle {
        font-size: 1.2rem;
        color: #a5d6a7 !important;
        font-weight: 300;
    }
    
    /* Section Headers */
    .section-header {
        display: flex;
        align-items: center;
        gap: 12px;
        margin: 25px 0 15px 0;
        padding-bottom: 10px;
        border-bottom: 2px solid rgba(76, 175, 80, 0.3);
    }
    
    .section-header span {
        font-size: 1.4rem;
        font-weight: 600;
        color: #81c784 !important;
    }
    
    /* Slider Styling - Neon Futuristic Design */
    .stSlider {
        padding: 10px 0;
    }
    
    /* Slider Track */
    .stSlider > div > div > div {
        background: linear-gradient(90deg, 
            rgba(0, 255, 200, 0.1) 0%, 
            rgba(0, 255, 150, 0.2) 50%, 
            rgba(0, 200, 100, 0.1) 100%) !important;
        height: 8px !important;
        border-radius: 10px !important;
        border: 1px solid rgba(0, 255, 180, 0.3) !important;
    }
    
    /* Slider Filled Track */
    .stSlider > div > div > div > div:first-child {
        background: linear-gradient(90deg, 
            #00ffcc 0%, 
            #00e6a0 30%, 
            #00cc88 60%, 
            #00b377 100%) !important;
        border-radius: 10px !important;
        box-shadow: 0 0 15px rgba(0, 255, 180, 0.5),
                    0 0 30px rgba(0, 255, 150, 0.3),
                    inset 0 1px 0 rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Slider Thumb */
    .stSlider > div > div > div > div:last-child > div {
        background: linear-gradient(135deg, #00ffcc 0%, #00e6a0 50%, #00cc88 100%) !important;
        border: 3px solid #ffffff !important;
        width: 22px !important;
        height: 22px !important;
        border-radius: 50% !important;
        box-shadow: 0 0 20px rgba(0, 255, 180, 0.8),
                    0 0 40px rgba(0, 255, 150, 0.5),
                    0 4px 15px rgba(0, 0, 0, 0.4) !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        animation: thumbPulse 2s ease-in-out infinite !important;
    }
    
    .stSlider > div > div > div > div:last-child > div:hover {
        transform: scale(1.2) !important;
        box-shadow: 0 0 30px rgba(0, 255, 180, 1),
                    0 0 60px rgba(0, 255, 150, 0.7),
                    0 6px 20px rgba(0, 0, 0, 0.5) !important;
    }
    
    @keyframes thumbPulse {
        0%, 100% { box-shadow: 0 0 20px rgba(0, 255, 180, 0.8), 0 0 40px rgba(0, 255, 150, 0.5); }
        50% { box-shadow: 0 0 30px rgba(0, 255, 180, 1), 0 0 60px rgba(0, 255, 150, 0.7); }
    }
    
    /* Slider Value Display */
    .stSlider > div > div > div > div:last-child {
        color: #00ffcc !important;
        font-weight: 600 !important;
        text-shadow: 0 0 10px rgba(0, 255, 180, 0.5) !important;
    }
    
    /* Slider Labels */
    .stSlider label {
        color: #a5d6a7 !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }
    
    /* Select Box Styling */
    .stSelectbox > div > div {
        background: rgba(20, 60, 40, 0.6) !important;
        border: 1px solid rgba(76, 175, 80, 0.3) !important;
        border-radius: 10px !important;
        color: #e8f5e9 !important;
    }
    
    /* Button Styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #2e7d32 0%, #388e3c 50%, #43a047 100%);
        color: white;
        font-size: 1.2rem;
        font-weight: 600;
        padding: 15px 30px;
        border: none;
        border-radius: 15px;
        cursor: pointer;
        transition: all 0.3s ease;
        box-shadow: 0 5px 20px rgba(46, 125, 50, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 30px rgba(46, 125, 50, 0.6),
                    0 0 40px rgba(76, 175, 80, 0.3);
    }
    
    /* Result Card */
    .result-card {
        background: linear-gradient(135deg, rgba(30, 80, 50, 0.7) 0%, rgba(20, 60, 35, 0.9) 100%);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 35px;
        margin: 25px 0;
        border: 2px solid rgba(76, 175, 80, 0.4);
        text-align: center;
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.4),
                    0 0 50px rgba(76, 175, 80, 0.15);
        animation: fadeIn 0.5s ease-out;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .result-title {
        font-size: 1.3rem;
        color: #a5d6a7 !important;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .result-price {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #66bb6a, #a5d6a7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 15px 0;
    }
    
    .price-range {
        font-size: 1.5rem;
        color: #81c784 !important;
        padding: 12px 25px;
        background: rgba(76, 175, 80, 0.15);
        border-radius: 50px;
        display: inline-block;
        border: 1px solid rgba(76, 175, 80, 0.3);
        margin-top: 10px;
    }
    
    /* Info boxes */
    .info-box {
        background: rgba(20, 60, 40, 0.5);
        border-left: 4px solid #4caf50;
        padding: 15px 20px;
        border-radius: 0 12px 12px 0;
        margin: 10px 0;
    }
    
    /* Metric styling */
    .metric-container {
        background: rgba(30, 70, 45, 0.5);
        border-radius: 15px;
        padding: 20px;
        text-align: center;
        border: 1px solid rgba(76, 175, 80, 0.2);
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4caf50 !important;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #a5d6a7 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Footer */
    .footer {
        background: linear-gradient(180deg, rgba(10, 35, 20, 0.8) 0%, rgba(5, 25, 15, 0.95) 100%);
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 30px;
        margin-top: 40px;
        border: 1px solid rgba(76, 175, 80, 0.2);
        text-align: center;
    }
    
    .footer a {
        color: #81c784;
        text-decoration: none;
        margin: 0 15px;
        font-weight: 500;
        transition: all 0.3s ease;
        padding: 10px 20px;
        border-radius: 25px;
        background: rgba(76, 175, 80, 0.1);
        border: 1px solid rgba(76, 175, 80, 0.2);
        display: inline-block;
    }
    
    .footer a:hover {
        color: #a5d6a7;
        background: rgba(76, 175, 80, 0.25);
        transform: translateY(-3px);
        box-shadow: 0 5px 20px rgba(76, 175, 80, 0.3);
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: rgba(30, 70, 45, 0.5) !important;
        border-radius: 12px !important;
        border: 1px solid rgba(76, 175, 80, 0.2) !important;
    }
    
    /* Progress/Bar Chart */
    .stProgress > div > div {
        background: linear-gradient(90deg, #1b5e20, #4caf50) !important;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Floating particles effect (optional visual enhancement) */
    .stApp::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            radial-gradient(circle at 20% 80%, rgba(76, 175, 80, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 80% 20%, rgba(129, 199, 132, 0.05) 0%, transparent 50%),
            radial-gradient(circle at 40% 40%, rgba(46, 125, 50, 0.03) 0%, transparent 30%);
        pointer-events: none;
        z-index: -1;
    }
    
    /* Toggle switch styling */
    .toggle-container {
        display: flex;
        justify-content: center;
        gap: 15px;
        margin: 10px 0;
    }
    
    .toggle-option {
        padding: 8px 20px;
        border-radius: 25px;
        background: rgba(76, 175, 80, 0.15);
        border: 1px solid rgba(76, 175, 80, 0.3);
        color: #a5d6a7;
        cursor: pointer;
        transition: all 0.3s ease;
    }
    
    .toggle-option.active {
        background: rgba(76, 175, 80, 0.4);
        border-color: rgba(76, 175, 80, 0.6);
    }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 LOAD DATA & MODEL
# ═══════════════════════════════════════════════════════════════════════════════
@st.cache_data
def load_data():
    return pd.read_csv("train.csv")

@st.cache_resource
def load_model():
    return keras.models.load_model("smartphone_price_ann.h5")

df = load_data()
model = load_model()

X = df.drop("price_range", axis=1)
feature_names = X.columns.tolist()

scaler = StandardScaler()
scaler.fit(X)

# Price Categories with Estimated Ranges
price_info = {
    0: {"label": "💚 Budget Friendly", "range": "₹5,000 - ₹12,000", "desc": "Great for basic use, calls, and light browsing", "color": "#4caf50"},
    1: {"label": "💛 Mid Range", "range": "₹12,000 - ₹25,000", "desc": "Perfect balance of features and value", "color": "#8bc34a"},
    2: {"label": "🧡 Premium", "range": "₹25,000 - ₹50,000", "desc": "High-end features and excellent performance", "color": "#cddc39"},
    3: {"label": "💎 Flagship", "range": "₹50,000+", "desc": "Top-of-the-line with cutting-edge technology", "color": "#00e676"}
}

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 HERO SECTION
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-container">
    <div class="hero-title">📱 Smart Price Predictor</div>
    <div class="hero-subtitle">
        Discover the perfect price range for any smartphone configuration!<br>
        <span style="color: #66bb6a; font-size: 1rem;">Powered by Artificial Neural Networks (ANN) 🧠</span>
    </div>
</div>
""", unsafe_allow_html=True)

# Quick Guide
with st.expander("📖 How to Use This App (Click to expand)", expanded=False):
    st.markdown("""
    <div class="glass-card">
        <h3 style="color: #4caf50; margin-bottom: 20px;">🎯 Simple 3-Step Guide</h3>
        
        <p><strong>Step 1: Configure Your Phone</strong> 👈<br>
        Use the sliders and options in the sidebar to set up your dream smartphone's specifications.</p>
        
        <p><strong>Step 2: Click Predict</strong> 🔮<br>
        Hit the "Predict Price Range" button to let our AI analyze your configuration.</p>
        
        <p><strong>Step 3: See Results</strong> 📊<br>
        Get an instant price category with an estimated price range in Indian Rupees!</p>
        
        <hr style="border-color: rgba(76, 175, 80, 0.2); margin: 20px 0;">
        
        <p style="color: #81c784;"><em>💡 Tip: Don't know what some specs mean? Hover over the info icons for explanations!</em></p>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📱 SIDEBAR - Phone Configuration
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 20px 0;">
        <h2 style="color: #4caf50; margin: 0;">⚙️ Phone Configuration</h2>
        <p style="color: #81c784; font-size: 0.9rem;">Customize your smartphone specs</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 🔋 POWER & PERFORMANCE
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 🔋 Power & Performance")
    
    battery_power = st.slider(
        "🔋 Battery Capacity (mAh)",
        min_value=int(df["battery_power"].min()),
        max_value=int(df["battery_power"].max()),
        value=1500,
        help="Higher mAh = Longer battery life. 3000+ is great for heavy users!"
    )
    
    ram = st.slider(
        "💾 RAM (MB)",
        min_value=int(df["ram"].min()),
        max_value=int(df["ram"].max()),
        value=3000,
        help="More RAM = Smoother multitasking. 4000MB+ recommended for gaming!"
    )
    
    n_cores = st.slider(
        "⚡ Processor Cores",
        min_value=int(df["n_cores"].min()),
        max_value=int(df["n_cores"].max()),
        value=4,
        help="More cores = Better performance. 8 cores is flagship level!"
    )
    
    clock_speed = st.slider(
        "🚀 Processor Speed (GHz)",
        min_value=float(df["clock_speed"].min()),
        max_value=float(df["clock_speed"].max()),
        value=2.0,
        step=0.1,
        help="Higher GHz = Faster processing. 2.0+ is good for gaming!"
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 💾 STORAGE & MEMORY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 💾 Storage")
    
    int_memory = st.slider(
        "📦 Internal Storage (GB)",
        min_value=int(df["int_memory"].min()),
        max_value=int(df["int_memory"].max()),
        value=64,
        help="More storage = More apps & photos. 128GB is ideal for most users!"
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📷 CAMERA
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📷 Camera")
    
    pc = st.slider(
        "📸 Primary Camera (MP)",
        min_value=int(df["pc"].min()),
        max_value=int(df["pc"].max()),
        value=13,
        help="Higher MP = More detailed photos. 48MP+ is flagship quality!"
    )
    
    fc = st.slider(
        "🤳 Front Camera (MP)",
        min_value=int(df["fc"].min()),
        max_value=int(df["fc"].max()),
        value=5,
        help="For selfies and video calls. 8MP+ is great for vloggers!"
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📺 DISPLAY
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📺 Display")
    
    px_height = st.slider(
        "📐 Pixel Height",
        min_value=int(df["px_height"].min()),
        max_value=int(df["px_height"].max()),
        value=800,
        help="Higher resolution = Sharper display. 1920+ is Full HD!"
    )
    
    px_width = st.slider(
        "📐 Pixel Width",
        min_value=int(df["px_width"].min()),
        max_value=int(df["px_width"].max()),
        value=1200,
        help="Higher resolution = Sharper display. 1080+ is Full HD!"
    )
    
    sc_h = st.slider(
        "📱 Screen Height (cm)",
        min_value=int(df["sc_h"].min()),
        max_value=int(df["sc_h"].max()),
        value=12,
        help="Physical screen height in centimeters"
    )
    
    sc_w = st.slider(
        "📱 Screen Width (cm)",
        min_value=int(df["sc_w"].min()),
        max_value=int(df["sc_w"].max()),
        value=7,
        help="Physical screen width in centimeters"
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📏 PHYSICAL SPECS
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📏 Physical Specs")
    
    mobile_wt = st.slider(
        "⚖️ Weight (grams)",
        min_value=int(df["mobile_wt"].min()),
        max_value=int(df["mobile_wt"].max()),
        value=150,
        help="Lighter = More comfortable. 150-180g is ideal!"
    )
    
    m_dep = st.slider(
        "📏 Thickness (cm)",
        min_value=float(df["m_dep"].min()),
        max_value=float(df["m_dep"].max()),
        value=0.5,
        step=0.1,
        help="Thinner = More sleek. 0.7-0.9cm is typical!"
    )
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📶 CONNECTIVITY & FEATURES
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📶 Connectivity & Features")
    
    col1, col2 = st.columns(2)
    with col1:
        blue = st.selectbox("🔵 Bluetooth", ["No", "Yes"], index=1)
        blue = 1 if blue == "Yes" else 0
        
        four_g = st.selectbox("📶 4G Support", ["No", "Yes"], index=1)
        four_g = 1 if four_g == "Yes" else 0
        
        touch_screen = st.selectbox("👆 Touch Screen", ["No", "Yes"], index=1)
        touch_screen = 1 if touch_screen == "Yes" else 0
    
    with col2:
        wifi = st.selectbox("📡 WiFi", ["No", "Yes"], index=1)
        wifi = 1 if wifi == "Yes" else 0
        
        three_g = st.selectbox("📶 3G Support", ["No", "Yes"], index=1)
        three_g = 1 if three_g == "Yes" else 0
        
        dual_sim = st.selectbox("📱 Dual SIM", ["No", "Yes"], index=1)
        dual_sim = 1 if dual_sim == "Yes" else 0
    
    st.markdown("---")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 📞 OTHER
    # ═══════════════════════════════════════════════════════════════════════════
    st.markdown("### 📞 Battery Usage")
    
    talk_time = st.slider(
        "📞 Talk Time (hours)",
        min_value=int(df["talk_time"].min()),
        max_value=int(df["talk_time"].max()),
        value=10,
        help="Hours of talk time on a single charge"
    )

# ═══════════════════════════════════════════════════════════════════════════════
# 🎯 MAIN CONTENT - Summary & Prediction
# ═══════════════════════════════════════════════════════════════════════════════

# Configuration Summary
st.markdown("""
<div class="section-header">
    <span>📋 Your Phone Configuration Summary</span>
</div>
""", unsafe_allow_html=True)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{battery_power}</div>
        <div class="metric-label">🔋 Battery (mAh)</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{ram}</div>
        <div class="metric-label">💾 RAM (MB)</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{int_memory}</div>
        <div class="metric-label">📦 Storage (GB)</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{pc}</div>
        <div class="metric-label">📸 Camera (MP)</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# Prepare sample for prediction
sample = {
    "battery_power": battery_power,
    "blue": blue,
    "clock_speed": clock_speed,
    "dual_sim": dual_sim,
    "fc": fc,
    "four_g": four_g,
    "int_memory": int_memory,
    "m_dep": m_dep,
    "mobile_wt": mobile_wt,
    "n_cores": n_cores,
    "pc": pc,
    "px_height": px_height,
    "px_width": px_width,
    "ram": ram,
    "sc_h": sc_h,
    "sc_w": sc_w,
    "talk_time": talk_time,
    "three_g": three_g,
    "touch_screen": touch_screen,
    "wifi": wifi
}

# Prediction Button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    predict_button = st.button("🔮 Predict Price Range", use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 PREDICTION RESULTS
# ═══════════════════════════════════════════════════════════════════════════════
if predict_button:
    with st.spinner("🤖 AI is analyzing your phone configuration..."):
        # Prepare input
        x = np.array([sample[col] for col in feature_names], dtype=float).reshape(1, -1)
        x_scaled = scaler.transform(x)
        
        # Get prediction
        proba = model.predict(x_scaled, verbose=0)
        pred_class = int(np.argmax(proba, axis=1)[0])
        confidence = float(proba[0][pred_class]) * 100
        
        # Get price info
        price = price_info[pred_class]
    
    # Display Results
    st.markdown(f"""
    <div class="result-card">
        <div class="result-title">🎯 Prediction Result</div>
        <div class="result-price">{price['label']}</div>
        <div class="price-range">💰 Expected Price: {price['range']}</div>
        <p style="color: #a5d6a7; margin-top: 20px; font-size: 1.1rem;">{price['desc']}</p>
        <p style="color: #66bb6a; margin-top: 15px;">🎯 Prediction Confidence: <strong>{confidence:.1f}%</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    # Probability Distribution
    st.markdown("""
    <div class="section-header">
        <span>📊 Detailed Analysis</span>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
        <p style="color: #81c784; margin-bottom: 15px;">
            <strong>What does this mean?</strong><br>
            The graph below shows how confident our AI is about each price category. 
            The highest bar indicates the most likely price range for your configuration.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create probability dataframe
    prob_df = pd.DataFrame({
        'Price Category': ['💚 Budget', '💛 Mid Range', '🧡 Premium', '💎 Flagship'],
        'Probability': proba[0] * 100
    })
    
    st.bar_chart(prob_df.set_index('Price Category'))
    
    # Detailed breakdown
    col1, col2, col3, col4 = st.columns(4)
    categories = [
        ("💚 Budget", "₹5K-12K", proba[0][0] * 100),
        ("💛 Mid Range", "₹12K-25K", proba[0][1] * 100),
        ("🧡 Premium", "₹25K-50K", proba[0][2] * 100),
        ("💎 Flagship", "₹50K+", proba[0][3] * 100)
    ]
    
    for col, (cat, range_str, prob) in zip([col1, col2, col3, col4], categories):
        with col:
            st.markdown(f"""
            <div class="metric-container" style="{'border: 2px solid #4caf50;' if prob == max(proba[0]) * 100 else ''}">
                <div style="font-size: 0.9rem; color: #81c784;">{cat}</div>
                <div class="metric-value">{prob:.1f}%</div>
                <div class="metric-label">{range_str}</div>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📋 FOOTER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <h3 style="color: #4caf50; margin-bottom: 20px;">🌟 Connect With Me</h3>
    <div style="margin-bottom: 20px;">
        <a href="https://www.linkedin.com/in/mayank-goyal-mg09/" target="_blank">💼 LinkedIn</a>
        <a href="https://github.com/mayank-goyal09" target="_blank">🐙 GitHub</a>
        <a href="https://mayank-goyal09.github.io/index.html" target="_blank">🌐 Portfolio</a>
    </div>
    <p style="color: #66bb6a; font-size: 0.9rem;">
        Built with ❤️ using Streamlit & TensorFlow<br>
        <span style="color: #4caf50;">© 2026 Mayank Goyal | Smart Price Predictor</span>
    </p>
</div>
""", unsafe_allow_html=True)
