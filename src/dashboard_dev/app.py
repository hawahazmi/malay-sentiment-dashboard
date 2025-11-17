# src/dashboard_dev/app.py
# to run: python -m streamlit run D:/UTP/UG/FYP/dev_root/src/dashboard_dev/app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))


import streamlit as st
from datetime import datetime
from src.dashboard_dev.utils import (
    load_data,
    refresh_data,
    filter_data
)


# ===== CUSTOM CSS FOR MODERN UI =====
def inject_custom_css():
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

    /* Global Styles */
    * {
        font-family: 'Poppins', sans-serif;
    }

    /* Main Container */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3c72 0%, #2a5298 100%);
        box-shadow: 4px 0 15px rgba(0,0,0,0.1);
    }

    [data-testid="stSidebar"] .css-1d391kg {
        padding-top: 2rem;
    }

    /* Sidebar Text */
    [data-testid="stSidebar"] * {
        color: #ffffff !important;
    }

    /* Radio Buttons - Page Navigation */
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 18px;
        font-weight: 600;
        color: #ffffff !important;
        margin-bottom: 1rem;
    }

    [data-testid="stSidebar"] .stRadio > div {
        background: rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 0.5rem;
        margin-bottom: 1.5rem;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        background: rgba(255,255,255,0.05);
        padding: 12px 20px;
        border-radius: 8px;
        margin: 5px 0;
        cursor: pointer;
        transition: all 0.3s ease;
        display: block;
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background: rgba(255,255,255,0.15);
        transform: translateX(5px);
    }

    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
        background: linear-gradient(90deg, #F63366 0%, #ff6b9d 100%);
        box-shadow: 0 4px 15px rgba(246,51,102,0.4);
        font-weight: 600;
    }

    /* Sidebar Buttons */
    [data-testid="stSidebar"] .stButton button {
        width: 100%;
        background: linear-gradient(90deg, #F63366 0%, #ff6b9d 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 20px;
        font-weight: 600;
        font-size: 14px;
        cursor: pointer;
        transition: all 0.3s ease;
        margin-bottom: 10px;
        box-shadow: 0 4px 15px rgba(246,51,102,0.3);
    }

    [data-testid="stSidebar"] .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(246,51,102,0.5);
    }

    [data-testid="stSidebar"] .stButton button:active {
        transform: translateY(0px);
        box-shadow: 0 2px 10px rgba(246,51,102,0.3);
    }

    /* Content Area Cards */
    .stApp > div > div > div {
        background: rgba(255,255,255,0.95);
        border-radius: 20px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 10px 40px rgba(0,0,0,0.1);
    }

    /* Metric Cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
        font-weight: 600;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Headers */
    h1 {
        color: #1e3c72;
        font-weight: 700;
        font-size: 2.5rem;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    h2 {
        color: #2a5298;
        font-weight: 600;
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }

    h3 {
        color: #667eea;
        font-weight: 600;
        font-size: 1.4rem;
    }

    /* Dividers */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #F63366, transparent);
        margin: 2rem 0;
    }

    /* Data Tables */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }

    .dataframe thead tr {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .dataframe tbody tr:nth-child(even) {
        background: #f8f9fa;
    }

    .dataframe tbody tr:hover {
        background: #e3f2fd;
        transition: all 0.3s ease;
    }

    /* Plotly Charts */
    .js-plotly-plot {
        border-radius: 15px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    }

    /* Success/Warning/Error Messages */
    .stSuccess {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        font-weight: 500;
    }

    .stWarning {
        background: linear-gradient(90deg, #fdcb6e 0%, #e17055 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        font-weight: 500;
    }

    .stError {
        background: linear-gradient(90deg, #d63031 0%, #e17055 100%);
        border-radius: 10px;
        padding: 1rem;
        color: white;
        font-weight: 500;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-radius: 10px;
        font-weight: 600;
        padding: 1rem;
    }

    .streamlit-expanderHeader:hover {
        box-shadow: 0 4px 15px rgba(102,126,234,0.3);
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background: transparent;
    }

    .stTabs [data-baseweb="tab"] {
        background: rgba(102,126,234,0.1);
        border-radius: 10px;
        padding: 10px 20px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Multiselect */
    .stMultiSelect [data-baseweb="select"] {
        border-radius: 10px;
        border: 2px solid #667eea;
    }

    /* Date Input */
    .stDateInput input {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
    }

    /* Text Input & Text Area */
    .stTextInput input, .stTextArea textarea {
        border-radius: 10px;
        border: 2px solid #667eea;
        padding: 10px;
        font-family: 'Poppins', sans-serif;
    }

    /* File Uploader */
    [data-testid="stFileUploader"] {
        border: 2px dashed #667eea;
        border-radius: 15px;
        padding: 2rem;
        background: rgba(102,126,234,0.05);
    }

    /* Download Button */
    .stDownloadButton button {
        background: linear-gradient(90deg, #00b894 0%, #00cec9 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stDownloadButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0,184,148,0.4);
    }

    /* Spinner */
    .stSpinner > div {
        border-top-color: #F63366 !important;
    }

    /* Custom Animations */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    .main > div {
        animation: fadeIn 0.5s ease-out;
    }

    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }

    ::-webkit-scrollbar-track {
        background: #f1f1f1;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #764ba2 0%, #667eea 100%);
    }
    </style>
    """, unsafe_allow_html=True)


# INITIAL PAGE CONFIGURATION
st.set_page_config(
    page_title="Sentily",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Inject custom CSS
inject_custom_css()


# ===== LOGO/HEADER =====
def render_logo():
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0 2rem 0;">
        <h1 style="
            font-size: 3rem; 
            font-weight: 800; 
            background: linear-gradient(90deg, #F63366 0%, #ff6b9d 50%, #667eea 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0;
            text-shadow: none;
        ">🎭 SENTILY</h1>
        <p style="
            color: rgba(255,255,255,0.8); 
            font-size: 1rem; 
            margin-top: 0.5rem;
            font-weight: 500;
            letter-spacing: 2px;
        ">MALAY SOCIAL MEDIA SENTIMENT INTELLIGENCE</p>
    </div>
    """, unsafe_allow_html=True)


# STYLIZED SIDEBAR DESIGN
with st.sidebar:
    render_logo()

    # PAGE NAVIGATION
    st.markdown(
        '<p style="font-size: 18px; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px;">📍 Navigation</p>',
        unsafe_allow_html=True)
    page = st.radio(
        "",
        [
            "Overview",
            "Audience Sentiment Insights",
            "Marketing Recommendations",
            "Advanced Sentiment Analyzer",
        ],
        label_visibility="collapsed",
    )

    st.markdown("---")

    # === GLOBAL FILTERS ===
    st.markdown(
        '<p style="font-size: 18px; font-weight: 600; margin-bottom: 1rem; text-transform: uppercase; letter-spacing: 1px;">🔍 Filters</p>',
        unsafe_allow_html=True)

    # LOAD DATA INTO SESSION (if not already loaded)
    if "comments" not in st.session_state or "videos" not in st.session_state:
        with st.spinner("Loading initial data..."):
            try:
                videos, comments = load_data()
                st.session_state["videos"] = videos
                st.session_state["comments"] = comments
            except Exception as e:
                st.error(f"Failed to load data: {e}")
                st.stop()

    comments = st.session_state["comments"]
    videos = st.session_state["videos"]

    # Video Filter
    selected_videos = st.multiselect(
        "Select Video(s):",
        options=videos["title"].dropna().unique(),
        default=None,
        help="Filter analysis by one or more uploaded videos"
    )

    # Date Filter
    date_filter_mode = st.radio(
        "Filter by Date Range:",
        options=["Last 7 Days", "Last 30 Days", "All-time", "Custom"],
        index=0,
        help="Choose which time range of comments to include"
    )

    if date_filter_mode == "Custom":
        selected_date = st.date_input(
            "Select start date:",
            value=None,
            help="Only include comments published after this date"
        )
    else:
        selected_date = None

    # Save to session
    st.session_state["selected_videos"] = selected_videos
    st.session_state["date_filter_mode"] = date_filter_mode
    st.session_state["selected_date"] = selected_date

    st.markdown("---")
    # REFRESH DATA BUTTON
    if st.button("Refresh Data", use_container_width=True):
        with st.spinner("Refreshing data..."):
            refresh_data()
            refresh_time = datetime.now().strftime('%d %b %Y, %I:%M %p')
            st.session_state["refresh_time"] = refresh_time
            st.success(f"Data refreshed at {refresh_time}")
            st.rerun()

    # Display last refresh time
    if "refresh_time" in st.session_state:
        st.markdown(f"""
        <div style="
            background: rgba(255,255,255,0.1); 
            padding: 10px; 
            border-radius: 8px; 
            text-align: center;
            margin-top: 1rem;
        ">
            <small>Last updated: {st.session_state['refresh_time']}</small>
        </div>
        """, unsafe_allow_html=True)

# APPLY GLOBAL FILTERS BEFORE PAGE RENDER
try:
    filtered_comments = filter_data(
        st.session_state["comments"],
        st.session_state["videos"],
        selected_videos=st.session_state["selected_videos"],
        selected_date=st.session_state["selected_date"],
        date_filter_mode=st.session_state["date_filter_mode"],
    )

    filtered_video_ids = filtered_comments["video_id"].unique()
    filtered_videos = st.session_state["videos"][
        st.session_state["videos"]["video_id"].isin(filtered_video_ids)
    ]

    st.session_state["filtered_comments"] = filtered_comments
    st.session_state["filtered_videos"] = filtered_videos

except Exception as e:
    st.error(f"Error applying filters: {e}")
    st.stop()

# -------------------------------------------------------------
# PAGE HANDLING LOGIC
# -------------------------------------------------------------

from src.dashboard_dev.modules import (
    p1_overview,
    p2_sentiment_insights,
    p3_presc_rec,
    p4_predict_sentiment,
)

if page == "Overview":
    p1_overview.show(st.session_state)

elif page == "Audience Sentiment Insights":
    p2_sentiment_insights.show(st.session_state)

elif page == "Marketing Recommendations":
    p3_presc_rec.show(st.session_state)

elif page == "Advanced Sentiment Analyzer":
    p4_predict_sentiment.show(st.session_state)