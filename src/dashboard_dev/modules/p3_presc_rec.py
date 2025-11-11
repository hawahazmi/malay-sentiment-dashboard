# src/dashboard_dev/modules/p3_presc_rec.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ==========================================================
#  PRESCRIPTIVE ANALYTICS LOGIC
# ==========================================================
def calculate_sentiment_trends(comments_df):
    """Compute sentiment percentages and sentiment score per video."""
    if comments_df.empty:
        return pd.DataFrame(columns=[
            "video_id", "positive", "negative", "neutral",
            "positive_pct", "negative_pct", "neutral_pct", "sentiment_score"
        ])

    sentiment_summary = (
        comments_df.groupby(["video_id", "predicted_label"])
            .size()
            .unstack(fill_value=0)
            .reset_index()
    )

    for c in ["positive", "negative", "neutral"]:
        if c not in sentiment_summary.columns:
            sentiment_summary[c] = 0

    total_comments = sentiment_summary[["positive", "negative", "neutral"]].sum(axis=1).replace(0, np.nan)
    sentiment_summary["positive_pct"] = (sentiment_summary["positive"] / total_comments) * 100
    sentiment_summary["negative_pct"] = (sentiment_summary["negative"] / total_comments) * 100
    sentiment_summary["neutral_pct"] = (sentiment_summary["neutral"] / total_comments) * 100
    sentiment_summary["sentiment_score"] = (
            sentiment_summary["positive_pct"].fillna(0) - sentiment_summary["negative_pct"].fillna(0)
    )

    return sentiment_summary.fillna(0)


def calculate_engagement_trends(videos_df, min_views_threshold=1000):
    """Compute engagement performance metrics for each video."""
    for col in ["viewcount", "likecount", "commentcount"]:
        if col in videos_df.columns:
            videos_df[col] = pd.to_numeric(videos_df[col], errors="coerce").fillna(0)
        else:
            videos_df[col] = 0

    videos_df["engagement_score"] = (
                                            (videos_df["likecount"] + videos_df["commentcount"]) /
                                            videos_df["viewcount"].replace(0, np.nan)
                                    ) * 100
    videos_df["engagement_score"] = videos_df["engagement_score"].fillna(0)

    if "viewcount" in videos_df.columns:
        videos_df = videos_df[videos_df["viewcount"] >= min_views_threshold].copy()

    return videos_df


def generate_recommendations(row, avg_sentiment, avg_engagement):
    """Generate targeted recommendations and explanations for a single video."""
    recs = []
    alarm = False
    now = pd.Timestamp.now()

    if pd.isna(row.get("published_at")):
        return [{
            "rule": "Missing published_at — cannot determine recency.",
            "explanation": "This video has no valid publication date. Unable to evaluate time-based performance."
        }], False

    video_age_days = (now - row["published_at"]).days

    # Recency
    if video_age_days > 365:
        return [{
            "rule": "Video is over a year old — likely irrelevant for new marketing.",
            "explanation": "Older videos (>1 year) typically lose relevance and engagement value for ongoing campaigns."
        }], False

    # Sentiment
    if row["sentiment_score"] < avg_sentiment - 10:
        recs.append({
            "rule": "Negative sentiment rising — review viewer feedback & tone.",
            "explanation": "Audience sentiment is significantly below average. Indicates dissatisfaction or controversy."
        })
    elif row["sentiment_score"] < avg_sentiment:
        recs.append({
            "rule": "Slight drop in sentiment — adjust messaging style.",
            "explanation": "Slight decline in sentiment compared to the average; consider tone, pacing, or audience targeting."
        })
    elif row["sentiment_score"] > avg_sentiment + 10:
        recs.append({
            "rule": "High positive sentiment — replicate successful themes.",
            "explanation": "Audience response exceeds average positivity. Consider reusing similar topics or presentation style."
        })
    else:
        recs.append({
            "rule": "Stable audience sentiment — maintain consistency.",
            "explanation": "Sentiment remains stable near the average; consistent communication and tone are working well."
        })

    # Engagement
    if row["engagement_score"] < avg_engagement * 0.8:
        recs.append({
            "rule": "Low engagement — consider promoting or redesigning thumbnail.",
            "explanation": "Engagement is below 80% of the average; may need adjustments in visuals, titles, or promotion."
        })
    elif row["engagement_score"] > avg_engagement * 1.2:
        recs.append({
            "rule": "Excellent engagement — push similar campaigns or short clips.",
            "explanation": "Engagement exceeds 120% of the average. Consider boosting similar videos via ads or highlights."
        })
    else:
        recs.append({
            "rule": "Normal engagement — continue steady posting cadence.",
            "explanation": "Engagement is within normal range. Maintain your current publishing strategy."
        })

    # Recency-specific tips
    if video_age_days <= 30:
        if row["sentiment_score"] > avg_sentiment:
            recs.append({
                "rule": "New video trending well — boost via ads or collaborations.",
                "explanation": "Newly published video performing well. Boost its reach with paid promotion or influencer collabs."
            })
        elif row["sentiment_score"] < avg_sentiment:
            recs.append({
                "rule": "New video underperforming — tweak title, tags, or format.",
                "explanation": "Recently uploaded video is lagging. Optimize metadata or audience targeting."
            })

    alarm = row["sentiment_score"] < 0
    return recs, alarm


def generate_prescriptive_insights(comments_df, videos_df, min_views_threshold=1000):
    """Generate prescriptive insights using already-filtered dashboard data."""
    if comments_df.empty or videos_df.empty:
        return pd.DataFrame()

    videos_df["published_at"] = pd.to_datetime(videos_df["published_at"], errors="coerce").dt.tz_localize(None)
    sentiment_summary = calculate_sentiment_trends(comments_df)
    videos_df = calculate_engagement_trends(videos_df, min_views_threshold=min_views_threshold)

    merged_df = pd.merge(sentiment_summary, videos_df, on="video_id", how="inner")
    if merged_df.empty:
        return pd.DataFrame()

    now = pd.Timestamp.now()
    valid_pub_dates = merged_df["published_at"].dropna()

    if not valid_pub_dates.empty:
        days_since_pub = (now - valid_pub_dates).dt.days.clip(lower=0) + 1
        weights = 1 / days_since_pub
        avg_sentiment = np.average(merged_df.loc[valid_pub_dates.index, "sentiment_score"], weights=weights)
        avg_engagement = np.average(merged_df.loc[valid_pub_dates.index, "engagement_score"], weights=weights)
    else:
        avg_sentiment = merged_df["sentiment_score"].mean()
        avg_engagement = merged_df["engagement_score"].mean()

    recs, alarms = [], []
    for _, row in merged_df.iterrows():
        recommendations, alarm = generate_recommendations(row, avg_sentiment, avg_engagement)
        recs.append(recommendations)
        alarms.append(alarm)

    merged_df["recommendations"] = recs
    merged_df["alarm"] = alarms

    final_cols = [
        "title", "published_at", "viewcount", "likecount", "commentcount",
        "positive_pct", "negative_pct", "neutral_pct",
        "sentiment_score", "engagement_score", "recommendations", "alarm"
    ]
    merged_df = merged_df[final_cols].sort_values(by="sentiment_score", ascending=True).reset_index(drop=True)
    return merged_df


def show(session_state):
    # Modern Page Title
    st.markdown("""
    <h1 style="
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #F63366 0%, #ff6b9d 50%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    ">Marketing Recommendations</h1>
    """, unsafe_allow_html=True)

    comments_df = session_state.get("filtered_comments", pd.DataFrame())
    videos_df = session_state.get("filtered_videos", pd.DataFrame())

    if comments_df.empty or videos_df.empty:
        st.warning("Please ensure both comments and videos data are available in the dashboard.")
        return

    try:
        insights_df = generate_prescriptive_insights(comments_df, videos_df)
    except Exception as e:
        st.error(f"Error generating insights: {e}")
        return

    if insights_df.empty:
        st.warning("No prescriptive insights could be generated.")
        return

    # Calculate metrics
    avg_sentiment = np.nan_to_num(insights_df["sentiment_score"].mean(), nan=0.0)
    avg_engagement = np.nan_to_num(insights_df["engagement_score"].mean(), nan=0.0)
    total_videos = len(insights_df)
    total_alerts = insights_df["alarm"].sum()

    # ===== STRATEGIC DIRECTIVE BANNER =====
    st.markdown("### Strategic Marketing Directive")

    if avg_sentiment < -5:
        directive = "CRISIS MANAGEMENT MODE"
        directive_color = "linear-gradient(135deg, #d63031 0%, #e17055 100%)"
        directive_desc = "Immediate action required. Audience sentiment is critically negative. Review recent content and address concerns publicly."
        icon = "🚨"
    elif avg_sentiment < 5:
        directive = "STABILIZATION PHASE"
        directive_color = "linear-gradient(135deg, #fdcb6e 0%, #e17055 100%)"
        directive_desc = "Neutral sentiment detected. Focus on improving content quality and audience engagement to shift perception positively."
        icon = "⚠️"
    elif avg_sentiment < 15:
        directive = "GROWTH OPTIMIZATION"
        directive_color = "linear-gradient(135deg, #ffd700 0%, #f39c12 100%)"
        directive_desc = "Positive sentiment but room for improvement. Maintain quality while experimenting with new content formats."
        icon = "🟡"
    else:
        directive = "AMPLIFICATION STRATEGY"
        directive_color = "linear-gradient(135deg, #00b894 0%, #00cec9 100%)"
        directive_desc = "Excellent sentiment performance. Double down on successful themes and leverage this momentum for growth."
        icon = "✅"

    st.markdown(f"""
    <div style="
        background: {directive_color};
        padding: 2rem;
        border-radius: 20px;
        color: white;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin-bottom: 2rem;
    ">
        <div style="display: flex; align-items: center; justify-content: space-between;">
            <div style="flex: 1;">
                <div style="font-size: 3rem; margin-bottom: 0.5rem;">{icon}</div>
                <h2 style="color: white; margin: 0; font-size: 2rem;">{directive}</h2>
                <p style="margin: 10px 0 0 0; font-size: 1.1rem; opacity: 0.95;">{directive_desc}</p>
            </div>
            <div style="text-align: center; padding: 0 2rem;">
                <h1 style="color: white; font-size: 4rem; margin: 0;">{avg_sentiment:.1f}%</h1>
                <p style="margin: 5px 0 0 0; font-size: 0.9rem; opacity: 0.9;">Avg Sentiment</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ===== KPI CARDS =====
    st.markdown("### 📊 Performance Metrics")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(102,126,234,0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎬</div>
            <h2 style="color: white; font-size: 2.5rem; margin: 0;">{total_videos}</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Videos Analyzed</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #F63366 0%, #ff6b9d 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(246,51,102,0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">🚨</div>
            <h2 style="color: white; font-size: 2.5rem; margin: 0;">{int(total_alerts)}</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Critical Alerts</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(0,184,148,0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">📊</div>
            <h2 style="color: white; font-size: 2.5rem; margin: 0;">{avg_engagement:.1f}%</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Avg Engagement</p>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        positive_pct = (insights_df["sentiment_score"] > 0).sum() / len(insights_df) * 100
        st.markdown(f"""
        <div style="
            background: linear-gradient(135deg, #ffd700 0%, #f39c12 100%);
            padding: 1.5rem;
            border-radius: 15px;
            text-align: center;
            box-shadow: 0 8px 25px rgba(255,215,0,0.3);
        ">
            <div style="font-size: 2rem; margin-bottom: 0.5rem;">⭐</div>
            <h2 style="color: white; font-size: 2.5rem; margin: 0;">{positive_pct:.0f}%</h2>
            <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Positive Videos</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ===== VISUALIZATIONS =====

    # Scatter Plot
    fig_scatter = go.Figure()

    # Separate alarm and non-alarm videos
    alarm_videos = insights_df[insights_df["alarm"] == True]
    normal_videos = insights_df[insights_df["alarm"] == False]

    fig_scatter.add_trace(go.Scatter(
        x=normal_videos["engagement_score"],
        y=normal_videos["sentiment_score"],
        mode='markers',
        name='Normal',
        marker=dict(
            size=10,
            color='#00b894',
            line=dict(color='white', width=2)
        ),
        text=normal_videos["title"],
        hovertemplate='<b>%{text}</b><br>Engagement: %{x:.2f}%<br>Sentiment: %{y:.2f}<extra></extra>'
    ))

    fig_scatter.add_trace(go.Scatter(
        x=alarm_videos["engagement_score"],
        y=alarm_videos["sentiment_score"],
        mode='markers',
        name='⚠️ Alert',
        marker=dict(
            size=15,
            color='#e17055',
            symbol='diamond',
            line=dict(color='white', width=2)
        ),
        text=alarm_videos["title"],
        hovertemplate='<b>%{text}</b><br>Engagement: %{x:.2f}%<br>Sentiment: %{y:.2f}<br>🚨 ALERT<extra></extra>'
    ))

    fig_scatter.update_layout(
        title="Engagement vs Sentiment Matrix",
        xaxis_title="Engagement Score (%)",
        yaxis_title="Sentiment Score",
        font=dict(family="Poppins, sans-serif", size=12),
        height=500,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)'),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99
        )
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    # ===== Action Priority Index =====
    st.divider()
    st.markdown("### Action Priority Ranking")

    if 'alarm' in insights_df.columns and 'sentiment_score' in insights_df.columns:
        # Calculate priority score (lower sentiment + alarm = higher priority)
        insights_df['priority_score'] = 100 - insights_df['sentiment_score']
        insights_df.loc[insights_df['alarm'] == True, 'priority_score'] += 50

        top_priority = insights_df.nlargest(10, 'priority_score')[
            ['title', 'sentiment_score', 'engagement_score', 'alarm', 'priority_score']]

        top_priority['action'] = top_priority['alarm'].map({True: '🚨 Immediate', False: '⚠️ Soon'})

        fig_priority = px.bar(
            top_priority,
            x='priority_score',
            y='title',
            orientation='h',
            color='action',
            color_discrete_map={'🚨 Immediate': '#d63031', '⚠️ Soon': '#e17055'},
            text='priority_score'
        )

        fig_priority.update_traces(texttemplate='%{text:.0f}', textposition='outside')
        fig_priority.update_layout(
            title="Top 10 Videos Requiring Action (by Priority Score)",
            xaxis_title="Priority Score",
            yaxis_title="",
            height=500,
            font=dict(family="Poppins, sans-serif"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_priority, use_container_width=True)

    # ===== DETAILED RECOMMENDATIONS =====
    st.markdown("### Video-by-Video Action Plan")
    st.caption("Sorted by priority (most negative sentiment first)")

    for idx, row in insights_df.iterrows():
        sentiment_score = row["sentiment_score"]

        # Determine card color based on sentiment
        if sentiment_score < -10:
            card_gradient = "linear-gradient(135deg, #d63031 0%, #e17055 100%)"
            sentiment_emoji = "🔴"
            sentiment_label = "CRITICAL"
        elif sentiment_score < 0:
            card_gradient = "linear-gradient(135deg, #e17055 0%, #fdcb6e 100%)"
            sentiment_emoji = "🟠"
            sentiment_label = "NEEDS ATTENTION"
        elif sentiment_score < 10:
            card_gradient = "linear-gradient(135deg, #fdcb6e 0%, #ffd700 100%)"
            sentiment_emoji = "🟡"
            sentiment_label = "FAIR"
        else:
            card_gradient = "linear-gradient(135deg, #00b894 0%, #00cec9 100%)"
            sentiment_emoji = "🟢"
            sentiment_label = "EXCELLENT"

        with st.expander(f"{sentiment_emoji} {row['title']} — {sentiment_label} ({sentiment_score:.2f})"):
            st.markdown(f"""
            <div style="
                background: {card_gradient};
                padding: 1.5rem;
                border-radius: 10px;
                color: white;
                margin-bottom: 1rem;
            ">
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 1rem;">
                    <div>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Published</p>
                        <h4 style="margin: 5px 0 0 0; color: white;">{row.get("published_at", "Unknown")}</h4>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Views</p>
                        <h4 style="margin: 5px 0 0 0; color: white;">{int(row.get('viewcount', 0)):,}</h4>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Engagement</p>
                        <h4 style="margin: 5px 0 0 0; color: white;">{row['engagement_score']:.2f}%</h4>
                    </div>
                    <div>
                        <p style="margin: 0; font-size: 0.9rem; opacity: 0.9;">Sentiment</p>
                        <h4 style="margin: 5px 0 0 0; color: white;">{sentiment_score:.2f}</h4>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Actionable Recommendations:")
            for i, rec in enumerate(row["recommendations"], 1):
                st.markdown(f"""
                <div style="
                    background: rgba(102,126,234,0.1);
                    padding: 1rem;
                    border-left: 4px solid #667eea;
                    border-radius: 5px;
                    margin-bottom: 0.5rem;
                ">
                    <strong>{i}. {rec['rule']}</strong><br>
                    <span style="color: #666;">{rec['explanation']}</span>
                </div>
                """, unsafe_allow_html=True)

            if row["alarm"]:
                st.error(
                    "🚨 **CRITICAL ALERT**: This video requires immediate strategic review and potential corrective action.")