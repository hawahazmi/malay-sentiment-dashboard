# src/dashboard_dev/modules/p2_sentiment_insights.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import timedelta
import streamlit as st
import plotly.graph_objects as go
from wordcloud import WordCloud
import pandas as pd
import numpy as np


def show(session_state):
    # Page title
    st.markdown("""
            <h1 style="
                font-size: 2.8rem;
                font-weight: 700;
                color: black;
                display: inline-block;
                margin-bottom: 2rem;
                will-change: transform;
                backface-visibility: hidden;
                transform: translateZ(0);
            ">Audience Sentiment Insights</h1>
            """, unsafe_allow_html=True)

    # Load data
    comments = session_state.get("filtered_comments", pd.DataFrame())
    videos = session_state.get("filtered_videos", pd.DataFrame())

    if comments.empty or videos.empty:
        comments = session_state.get("comments", pd.DataFrame())
        videos = session_state.get("videos", pd.DataFrame())

    if comments.empty or videos.empty:
        st.warning("No data available. Please load data or check your filters.")
        return

    # Identify date columns
    CMT_DATE_COL = "published_at_x" if "published_at_x" in comments.columns else "published_at"
    VID_DATE_COL = "published_at_y" if "published_at_y" in videos.columns else "published_at"

    if CMT_DATE_COL in comments.columns:
        comments[CMT_DATE_COL] = pd.to_datetime(comments[CMT_DATE_COL], errors="coerce").dt.tz_localize(None)

    # ===== SENTIMENT TRENDS OVER TIME =====
    st.markdown("### Sentiment Trends Over Time")

    if CMT_DATE_COL in comments.columns:
        date_filter_mode = session_state.get("date_filter_mode", "All-time")
        selected_date = session_state.get("selected_date", None)

        today = pd.Timestamp.now()
        if date_filter_mode == "Last 7 Days":
            filtered_comments = comments[comments[CMT_DATE_COL] >= today - timedelta(days=7)]
        elif date_filter_mode == "Last 30 Days":
            filtered_comments = comments[comments[CMT_DATE_COL] >= today - timedelta(days=30)]
        elif date_filter_mode == "Custom" and selected_date is not None:
            filtered_comments = comments[comments[CMT_DATE_COL] >= pd.to_datetime(selected_date, errors="coerce")]
        else:
            filtered_comments = comments.copy()

        if not filtered_comments.empty:
            filtered_comments["date_only"] = filtered_comments[CMT_DATE_COL].dt.date

            daily_sentiment = (
                filtered_comments.groupby(["date_only", "predicted_label"])
                    .size()
                    .unstack(fill_value=0)
                    .reset_index()
            )

            if not daily_sentiment.empty:
                trend_df = daily_sentiment.copy()
                for col in ["positive", "neutral", "negative"]:
                    if col not in trend_df.columns:
                        trend_df[col] = 0

                trend_df["total"] = trend_df[["positive", "neutral", "negative"]].sum(axis=1)
                trend_df["Positive (%)"] = (trend_df["positive"] / trend_df["total"]) * 100
                trend_df["Negative (%)"] = (trend_df["negative"] / trend_df["total"]) * 100
                trend_df["Net Sentiment (%)"] = ((trend_df["positive"] - trend_df["negative"]) / trend_df[
                    "total"]) * 100

                # Modern multi-line chart
                fig_trends = go.Figure()

                fig_trends.add_trace(go.Scatter(
                    x=trend_df["date_only"],
                    y=trend_df["Positive (%)"],
                    name="Positive",
                    line=dict(color="#00b894", width=3),
                    fill='tozeroy',
                    fillcolor='rgba(0, 184, 148, 0.1)'
                ))

                fig_trends.add_trace(go.Scatter(
                    x=trend_df["date_only"],
                    y=trend_df["Negative (%)"],
                    name="Negative",
                    line=dict(color="#e17055", width=3),
                    fill='tozeroy',
                    fillcolor='rgba(225, 112, 85, 0.1)'
                ))

                fig_trends.add_trace(go.Scatter(
                    x=trend_df["date_only"],
                    y=trend_df["Net Sentiment (%)"],
                    name="Net Sentiment",
                    line=dict(color="#667eea", width=4, dash='dash'),
                    yaxis="y2"
                ))

                fig_trends.update_layout(
                    title=f"Sentiment Trend Analysis ({date_filter_mode})",
                    xaxis_title="Date",
                    yaxis_title="Percentage (%)",
                    yaxis2=dict(
                        title="Net Sentiment (%)",
                        overlaying="y",
                        side="right"
                    ),
                    hovermode="x unified",
                    font=dict(family="Poppins, sans-serif", size=12),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    height=500,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                )

                st.plotly_chart(fig_trends, use_container_width=True)
            else:
                st.info("Not enough sentiment data to plot trends.")
        else:
            st.info("No comments found within the selected date range.")
    else:
        st.info("Missing date column for comments.")

    st.divider()

    # ===== VIDEO RANKING =====
    st.markdown("### Top Performing Videos")

    sentiment_avg = (
        comments.groupby("video_id")["predicted_label"]
            .value_counts(normalize=True)
            .unstack(fill_value=0)
            .reset_index()
    )

    for col in ["positive", "negative"]:
        if col not in sentiment_avg.columns:
            sentiment_avg[col] = 0

    sentiment_avg["net_sentiment"] = sentiment_avg["positive"] - sentiment_avg["negative"]

    merged = pd.merge(videos, sentiment_avg, on="video_id", how="left")
    merged["engagement_score"] = (
            (merged["likecount"].astype(float) + merged["commentcount"].astype(float))
            / merged["viewcount"].replace(0, np.nan)
    )

    if not merged.empty:
        # Normalize metrics
        merged["views_norm"] = merged["viewcount"] / merged["viewcount"].max()
        merged["engagement_norm"] = merged["engagement_score"] / merged["engagement_score"].max()
        merged["sentiment_norm"] = (merged["net_sentiment"] - merged["net_sentiment"].min()) / (
                merged["net_sentiment"].max() - merged["net_sentiment"].min()
        )

        merged["performance_score"] = (
                0.4 * merged["views_norm"]
                + 0.3 * merged["engagement_norm"]
                + 0.3 * merged["sentiment_norm"]
        )

        top_perf = merged.sort_values("performance_score", ascending=False).head(10)

        # Horizontal bar chart
        fig_perf = go.Figure(go.Bar(
            x=top_perf["performance_score"],
            y=top_perf["title"],
            orientation='h',
            marker=dict(
                color=top_perf["performance_score"],
                colorscale='Viridis',
                line=dict(color='rgba(0,0,0,0.1)', width=1)
            ),
            text=top_perf["performance_score"].round(2),
            textposition='outside',
            hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<extra></extra>'
        ))

        fig_perf.update_layout(
            title="Top 10 Best Performing Videos",
            xaxis_title="Performance Score",
            yaxis_title="",
            font=dict(family="Poppins, sans-serif", size=12),
            height=500,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
            showlegend=False
        )

        st.plotly_chart(fig_perf, use_container_width=True)

        # Top/Bottom Videos Comparison
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
                padding: 1rem;
                border-radius: 15px;
                margin-bottom: 1rem;
            ">
                <h4 style="color: white; margin: 0; text-align: center;">Top 5 Most Positive</h4>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                merged.sort_values("net_sentiment", ascending=False)
                [["title", "viewcount", "likecount", "net_sentiment"]]
                    .rename(columns={"title": "Video",
                                     "viewcount": "Views",
                                     "likecount": "Likes",
                                     "net_sentiment": "Net Sentiment"})
                    .head(5)
                    .reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        with col2:
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
                padding: 1rem;
                border-radius: 15px;
                margin-bottom: 1rem;
            ">
                <h4 style="color: white; margin: 0; text-align: center;">Top 5 Most Negative</h4>
            </div>
            """, unsafe_allow_html=True)

            st.dataframe(
                merged.sort_values("net_sentiment", ascending=True)
                [["title", "viewcount", "likecount", "net_sentiment"]]
                    .rename(columns={"title": "Video",
                                     "viewcount": "Views",
                                     "likecount": "Likes",
                                     "net_sentiment": "Net Sentiment"})
                    .head(5)
                    .reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

    st.divider()

    # ===== WORD CLOUDS =====
    st.markdown("### Key Topics and Terms")

    col1, col2 = st.columns(2)

    pos_text = " ".join(comments[comments["predicted_label"] == "positive"]["text_clean"].astype(str))
    neg_text = " ".join(comments[comments["predicted_label"] == "negative"]["text_clean"].astype(str))

    with col1:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
            padding: 1rem;
            border-radius: 15px 15px 0 0;
        ">
            <h4 style="color: white; margin: 0; text-align: center;">Positive Keywords</h4>
        </div>
        """, unsafe_allow_html=True)

        if pos_text.strip():
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="Greens",
                contour_width=2,
                contour_color='#00b894'
            ).generate(pos_text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No positive comments found.")

    with col2:
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
            padding: 1rem;
            border-radius: 15px 15px 0 0;
        ">
            <h4 style="color: white; margin: 0; text-align: center;">Negative Keywords</h4>
        </div>
        """, unsafe_allow_html=True)

        if neg_text.strip():
            wc = WordCloud(
                width=800,
                height=400,
                background_color="white",
                colormap="Reds",
                contour_width=2,
                contour_color='#d63031'
            ).generate(neg_text)
            st.image(wc.to_array(), use_container_width=True)
        else:
            st.info("No negative comments found.")

    st.divider()

    # ===== ENGAGEMENT VS SENTIMENT =====
    st.markdown("### Engagement vs. Sentiment Analysis")

    fig_scatter = go.Figure()

    fig_scatter.add_trace(go.Scatter(
        x=merged["engagement_score"],
        y=merged["net_sentiment"],
        mode='markers',
        marker=dict(
            size=merged["viewcount"] / merged["viewcount"].max() * 50 + 10,
            color=merged["net_sentiment"],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Net<br>Sentiment"),
            line=dict(color='white', width=2)
        ),
        text=merged["title"],
        hovertemplate='<b>%{text}</b><br>Engagement: %{x:.4f}<br>Sentiment: %{y:.3f}<extra></extra>'
    ))

    fig_scatter.update_layout(
        title="Engagement vs. Net Sentiment per Video",
        xaxis_title="Engagement Score (Likes + Comments / Views)",
        yaxis_title="Net Sentiment (Positive - Negative)",
        font=dict(family="Poppins, sans-serif", size=12),
        height=600,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', zeroline=True, zerolinecolor='rgba(0,0,0,0.2)')
    )

    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()
    st.markdown("### Temporal Patterns")

    # ===== Day of Week Analysis =====
    col1, col2 = st.columns(2)

    with col1:
        if 'published_at' in comments.columns:
            comments['day_of_week'] = pd.to_datetime(comments['published_at']).dt.day_name()
            comments['hour'] = pd.to_datetime(comments['published_at']).dt.hour

            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

            dow_sentiment = comments.groupby('day_of_week')['predicted_label'].apply(
                lambda x: (x == 'positive').mean() * 100
            ).reindex(day_order)

            fig_dow = go.Figure(data=[
                go.Bar(
                    x=day_order,
                    y=dow_sentiment.values,
                    marker=dict(
                        color=dow_sentiment.values,
                        colorscale='RdYlGn',
                        cmin=0,
                        cmax=100,
                        showscale=True,
                        colorbar=dict(title="Positive %")
                    ),
                    text=dow_sentiment.values.round(1),
                    texttemplate='%{text}%',
                    textposition='outside'
                )
            ])

            fig_dow.update_layout(
                title="Sentiment by Day of Week",
                xaxis_title="Day",
                yaxis_title="Positive Sentiment (%)",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins, sans-serif", size=12)
            )

            st.plotly_chart(fig_dow, use_container_width=True)

    with col2:
        # Hour of Day Analysis
        if 'hour' in comments.columns:
            hour_sentiment = comments.groupby('hour')['predicted_label'].apply(
                lambda x: (x == 'positive').mean() * 100
            )

            fig_hour = go.Figure(data=[
                go.Scatter(
                    x=hour_sentiment.index,
                    y=hour_sentiment.values,
                    mode='lines+markers',
                    line=dict(color='#667eea', width=3),
                    fill='tozeroy',
                    fillcolor='rgba(102, 126, 234, 0.1)',
                    marker=dict(size=8)
                )
            ])

            fig_hour.update_layout(
                title="Sentiment by Hour of Day",
                xaxis_title="Hour (24h format)",
                yaxis_title="Positive Sentiment (%)",
                height=500,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins, sans-serif", size=12),
                xaxis=dict(dtick=2)
            )

            st.plotly_chart(fig_hour, use_container_width=True)

    # ===== Comment Activity Heatmap =====
    st.markdown("### Comment Activity Heatmap")

    if 'day_of_week' in comments.columns and 'hour' in comments.columns:
        heatmap_data = comments.groupby(['day_of_week', 'hour']).size().reset_index(name='count')
        heatmap_pivot = heatmap_data.pivot(index='hour', columns='day_of_week', values='count').fillna(0)
        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        heatmap_pivot = heatmap_pivot.reindex(columns=day_order, fill_value=0)

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_pivot.values,
            x=day_order,
            y=heatmap_pivot.index,
            colorscale='Viridis',
            hovertemplate='Day: %{x}<br>Hour: %{y}<br>Comments: %{z}<extra></extra>'
        ))

        fig_heatmap.update_layout(
            title="Comment Activity by Day and Hour",
            xaxis_title="Day of Week",
            yaxis_title="Hour of Day",
            height=500,
            font=dict(family="Poppins, sans-serif", size=12)
        )

        st.plotly_chart(fig_heatmap, use_container_width=True)

        st.caption("💡 Brighter colors indicate higher comment activity. Use this to optimize posting times!")