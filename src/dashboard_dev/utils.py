# src/dashboard_dev/utils.py

import pandas as pd
import streamlit as st
import os
import time
import subprocess
from src.config import LABELLED_COMMENTS, VIDEOS_CSV
from datetime import datetime, timedelta


# -------------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------------
def load_data():
    """Load and clean the comments and video datasets."""
    comments = pd.read_csv(LABELLED_COMMENTS)
    videos = pd.read_csv(VIDEOS_CSV)

    # Standardize column names
    comments.columns = comments.columns.str.strip().str.lower()
    videos.columns = videos.columns.str.strip().str.lower()

    # Ensure all date columns are proper datetime
    for df in [videos, comments]:
        if "published_at" in df.columns:
            df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce").dt.tz_localize(None)

    # Handle missing sentiment scores
    if "predicted_label" not in comments.columns:
        comments["predicted_label"] = "neutral"
    if "sentiment_score" not in comments.columns:
        if "predicted_confidence" in comments.columns and "predicted_label" in comments.columns:
            comments["sentiment_score"] = comments.apply(
                lambda row: (
                    row["predicted_confidence"]
                    if row["predicted_label"] == "positive"
                    else -row["predicted_confidence"]
                    if row["predicted_label"] == "negative"
                    else 0
                ),
                axis=1
            )
        else:
            comments["sentiment_score"] = 0

    # Compute engagement total if not present
    if "engagement_total" not in videos.columns:
        for col in ["likecount", "commentcount"]:
            if col not in videos.columns:
                videos[col] = 0
        videos["engagement_total"] = (
            videos["likecount"].fillna(0) + videos["commentcount"].fillna(0)
        )

    return videos, comments


# -------------------------------------------------------------
# DASHBOARD VISUALIZATION UTILS
# -------------------------------------------------------------
def compute_metrics(df_comments, df_videos):
    total_videos = len(df_videos)
    total_comments = len(df_comments)

    sentiment_counts = df_comments["predicted_label"].value_counts(normalize=True) * 100
    pos = sentiment_counts.get("positive", 0)
    neg = sentiment_counts.get("negative", 0)
    neu = sentiment_counts.get("neutral", 0)

    # Net sentiment score
    net_sentiment = ((pos - neg) / 100) * 100 if total_comments > 0 else 0

    # Engagement rate (likes + comments / views)
    if all(x in df_videos.columns for x in ["likecount", "commentcount", "viewcount"]):
        engagement_rate = (
            ((df_videos["likecount"] + df_videos["commentcount"]) / df_videos["viewcount"])
            .replace([float("inf"), float("nan")], 0)
            .mean() * 100
        )
    else:
        engagement_rate = 0

    return total_videos, total_comments, pos, neg, neu, net_sentiment, engagement_rate


# -------------------------------------------------------------
# POP-UP WINDOW
# -------------------------------------------------------------
def show_popup(message="Processing..."):
    """Display a centered popup overlay with animation and stay visible."""
    popup = st.empty()  # placeholder to keep popup visible

    popup.markdown(
        f"""
        <style>
        .popup-overlay {{
            position: fixed;
            top: 0; left: 0;
            width: 100%; height: 100%;
            background-color: rgba(0, 0, 0, 0.6);
            z-index: 9999;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        .popup-content {{
            background-color: #262730;
            padding: 40px 60px;
            border-radius: 15px;
            text-align: center;
            font-size: 15px;
            font-weight: 500;
            color: #333;
            box-shadow: 0 0 25px rgba(0,0,0,0.25);
            animation: fadeIn 0.3s ease-in-out;
        }}
        @keyframes fadeIn {{
            from {{ opacity: 0; transform: scale(0.95); }}
            to {{ opacity: 1; transform: scale(1); }}
        }}
        .loader {{
            border: 6px solid #f3f3f3;
            border-top: 6px solid #4CAF50;
            border-radius: 50%;
            width: 50px;
            height: 50px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }}
        @keyframes spin {{
            0% {{ transform: rotate(0deg); }}
            100% {{ transform: rotate(360deg); }}
        }}
        </style>

        <div class="popup-overlay">
            <div class="popup-content">
                <div class="loader"></div>
                <h3>🔄 {message}</h3>
                <p>Please wait while we process your request...</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Give Streamlit time to render before continuing
    time.sleep(0.5)
    return popup


# -------------------------------------------------------------
# FILTER DATA
# -------------------------------------------------------------
def filter_data(comments, videos, selected_videos=None, selected_date=None, date_filter_mode="All-time"):
    """Filter comments based on selected videos and/or date."""
    df = comments.copy()

    # Merge video info for display, not for date filtering
    if "video_title" not in df.columns and "video_id" in df.columns:
        df = df.merge(
            videos[["video_id", "title"]],
            on="video_id",
            how="left"
        )
        df.rename(columns={"title": "video_title"}, inplace=True)

    # Detect date column priority
    date_col = None
    for col in ["comment_date", "published_at"]:
        if col in df.columns:
            date_col = col
            break

    # Filter by date range
    if date_col:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

        # Only drop timezone if present
        if pd.api.types.is_datetime64tz_dtype(df[date_col]):
            df[date_col] = df[date_col].dt.tz_localize(None)

        today = datetime.now()

        if date_filter_mode == "Last 7 Days":
            start_date = today - timedelta(days=7)
            df = df[df[date_col] >= start_date]

        elif date_filter_mode == "Last 30 Days":
            start_date = today - timedelta(days=30)
            df = df[df[date_col] >= start_date]

        elif date_filter_mode == "Custom" and selected_date is not None:
            selected_date = pd.to_datetime(selected_date, errors="coerce")
            df = df[df[date_col] >= selected_date]

    # Filter by video titles
    if selected_videos and len(selected_videos) > 0:
        df = df[df["video_title"].isin(selected_videos)]

    return df


# -------------------------------------------------------------
# REFRESH HANDLER
# -------------------------------------------------------------
def refresh_data():
    """Reload data and re-run prescriptive analysis."""
    popup = show_popup("Fetching latest YouTube data and updating insights...")
    try:
        pipeline_path = os.path.abspath(os.path.join(os.getcwd(), "run_pipeline.py"))

        env = os.environ.copy()
        env["PYTHONUTF8"] = "1"  # forces UTF-8 mode for all output

        # -*- coding: utf-8 -*-
        result = subprocess.run(
            ["python", "run_pipeline.py"],
            capture_output=True,
            text=True,
            encoding="utf-8",  # ✅ explicitly decode UTF-8
            env=env  # ✅ ensure UTF-8 for subprocess
        )

        popup.empty()  # clear popup

    except Exception as e:
        popup.empty()
        st.error(f"Unexpected error while running pipeline:\n{e}")

    # Small pause before refreshing UI
    time.sleep(1)

    videos, comments = load_data()
    st.session_state["comments"] = comments
    st.session_state["videos"] = videos


# ---------------------------------------------------------------
# PRESCRIPTIVE RECOMMENDATION
# ---------------------------------------------------------------
def categorize_performance(avg_sentiment, avg_engagement):
    # Thresholds
    if avg_sentiment >= 5:
        sentiment_category = "High"
    elif -5 <= avg_sentiment < 5:
        sentiment_category = "Mid"
    elif avg_sentiment < -5:
        sentiment_category = "Low"
    else:
        sentiment_category = "None"

    if avg_engagement >= 70:
        engagement_category = "High"
    elif 30 <= avg_sentiment < 70:
        engagement_category = "Mid"
    elif avg_sentiment < 30:
        engagement_category = "Low"
    else:
        engagement_category = "None"

    # Combine to decide recommendation
    combination = (sentiment_category, engagement_category)
    actions = []

    if combination == ("Low", "Low"):
        directive = "Crisis & Damage Control: Total Failure"
        directive_desc = "Low engagement with negative audience sentiment."
        directive_color = "#b91c1c"
        icon = "🚨"
        actions = [
            "Contain and remove the content from public view to prevent further brand damage.",
            "Conduct an internal post-mortem to identify what went wrong.",
            "Review the video to identify elements that failed completely."
        ]

    elif combination == ("Low", "Mid"):
        directive = "Crisis & Damage Control: Contained Issue"
        directive_desc = "Moderate engagement with negative audience sentiment"
        directive_color = "#dc2626"
        icon = "⚠️"
        actions = [
            "Engage selectively by responding to valid negative comments professionally.",
            "Upload a positive distraction piece (e.g., blooper reel) to dilute the issue.",
        ]

    elif combination == ("Low", "High"):
        directive = "Crisis Mode: Immediate Apology Required"
        directive_desc = "High engagement with negative audience sentiment"
        directive_color = "#ef4444"
        icon = "🆘"
        actions = [
            "Post an immediate public statement addressing the issue.",
            "Pause all promotions and deploy dark posts to test audience sentiment before relaunching.",
            "Fast crisis response limits viral negativity."
        ]

    elif combination == ("High", "High"):
        directive = "Success Amplification: Home Run"
        directive_desc = "High engagement with positive audience sentiment"
        directive_color = "#16a34a"
        icon = "🏆"
        actions = [
            "Double down with aggressive paid promotions and user-generated content campaigns.",
            "Celebrate the audience’s support with a thank-you post or short video."
        ]

    elif combination == ("High", "Mid"):
        directive = "Success Amplification: Solid Performer"
        directive_desc = "Moderate engagement with positive audience sentiment"
        directive_color = "#22c55e"
        icon = "🎯"
        actions = [
            "Boost engagement by replying to every positive comment.",
            "Cross-promote the video across platforms."
        ]

    elif combination == ("High", "Low"):
        directive = "Success Optimization: Hidden Gem"
        directive_desc = "Low engagement with positive audience sentiment"
        directive_color = "#4ade80"
        icon = "💎"
        actions = [
            "Optimize title, thumbnail, and keywords to increase discoverability.",
            "Relaunch targeted ads to new audiences."
        ]

    elif combination == ("Mid", "High"):
        directive = "Optimization & Investigation: Polarization"
        directive_desc = "High engagement with moderate audience sentiment"
        directive_color = "#f59e0b"
        icon = "⚖️"
        actions = [
            "Engage your audience directly through pinned comments or polls asking about mixed reactions.",
            "Acknowledge both sides of the feedback to manage polarization and learn from it."
        ]

    elif combination == ("Mid", "Mid"):
        directive = "Optimization: Status Quo"
        directive_desc = "Moderate engagement with moderate audience sentiment"
        directive_color = "#eab308"
        icon = "🧭"
        actions = [
            "Experiment with small format tweaks in the next video.",
            "Implement a stronger Call-to-Action within the video asking viewers to interact or share the video.",
            "Collect performance data to drive iterative improvement."
        ]

    elif combination == ("Mid", "Low"):
        directive = "Optimization: Indifference"
        directive_desc = "Low engagement with moderate audience sentiment"
        directive_color = "#d97706"
        icon = "🪫"
        actions = [
            "Stop investing in this format and reallocate resources.",
            "Try new creative directions (e.g., shorts, vertical video teasers) to find a resonant concept."
        ]

    else:
        directive = "No Clear Category"
        directive_desc = "Insufficient data to determine marketing recommendation."
        directive_color = "#9ca3af"
        icon = "❓"
        actions = [
            "Wait for the video to collect sufficient data."
        ]

    return directive, directive_desc, directive_color, icon, actions

