# src/dashboard_dev/modules/p1_overview.py
# Enhanced Modern UI for Overview Page

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from src.dashboard_dev.utils import compute_metrics


def render_metric_card(title, value, icon, delta=None, delta_color="normal"):
    """Render a modern metric card with gradient background"""
    delta_html = ""
    if delta is not None:
        color_map = {
            "normal": "#00b894",
            "inverse": "#d63031",
            "off": "#636e72"
        }
        delta_html = f'<p style="color: {color_map.get(delta_color, "#636e72")}; font-size: 1rem; margin: 5px 0 0 0; font-weight: 600;">{delta}</p>'

    st.markdown(f"""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
        text-align: center;
        transition: transform 0.3s ease;
    " onmouseover="this.style.transform='translateY(-5px)'" onmouseout="this.style.transform='translateY(0)'">
        <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{icon}</div>
        <h4 style="color: rgba(255,255,255,0.9); font-size: 0.9rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; margin: 0;">{title}</h4>
        <h2 style="color: white; font-size: 2.5rem; font-weight: 700; margin: 10px 0;">{value}</h2>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def show(session_state):
    # Page Title with gradient
    st.markdown("""
    <h1 style="
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    ">Channel Performance Overview</h1>
    """, unsafe_allow_html=True)

    # Retrieve filtered datasets
    comments = session_state.get("filtered_comments", pd.DataFrame())
    videos = session_state.get("filtered_videos", pd.DataFrame())

    if comments.empty or videos.empty:
        st.warning("No data available for the selected filters.")
        return

    # Compute metrics
    total_videos, total_comments, pos, neg, neu, net_sent, engagement = compute_metrics(comments, videos)

    # KPI Metrics Section with modern cards
    st.markdown("### Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_metric_card("Total Videos", f"{total_videos:,}", "🎬")

    with col2:
        render_metric_card("Total Comments", f"{total_comments:,}", "💬")

    with col3:
        render_metric_card("Engagement Rate", f"{engagement:.1f}%", "📊")

    with col4:
        color = "normal" if net_sent > 20 else "off" if net_sent > 0 else "inverse"
        render_metric_card("Net Sentiment", f"{net_sent:.1f}%", "🧮", delta_color=color)

    st.markdown("<br>", unsafe_allow_html=True)

    # Second row of metrics
    col5, col6, col7 = st.columns(3)

    with col5:
        render_metric_card("Positive", f"{pos:.1f}%", "😊", delta_color="off")

    with col6:
        render_metric_card("Neutral", f"{neu:.1f}%", "😐", delta_color="off")

    with col7:
        render_metric_card("Negative", f"{neg:.1f}%", "😞", delta_color="off")


    st.markdown("<br><br>", unsafe_allow_html=True)
    st.divider()

    # Sentiment Distribution with modern styling
    st.markdown("### Sentiment Distribution Analysis")

    col_chart, col_stats = st.columns([2, 1])

    with col_chart:
        sentiment_df = pd.DataFrame({
            "Sentiment": ["Positive", "Neutral", "Negative"],
            "Percentage": [pos, neu, neg],
            "Count": [
                (comments['predicted_label'] == 'positive').sum(),
                (comments['predicted_label'] == 'neutral').sum(),
                (comments['predicted_label'] == 'negative').sum()
            ]
        })

        fig_pie = px.pie(
            sentiment_df,
            names="Sentiment",
            values="Percentage",
            hole=0.5,
            color="Sentiment",
            color_discrete_map={
                "Positive": "#00b894",
                "Neutral": "#636efa",
                "Negative": "#e17055",
            },
            hover_data=["Count"]
        )

        fig_pie.update_traces(
            textinfo="percent+label",
            textposition="inside",
            textfont_size=14,
            marker=dict(line=dict(color='white', width=3))
        )

        fig_pie.update_layout(
            font=dict(family="Poppins, sans-serif", size=14),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            height=400,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with col_stats:


        st.markdown(f"""
            <div style="margin: 15px 0;">
                <p style="font-size: 0.9rem; margin: 5px 0; opacity: 0.9;">Most Common Sentiment:</p>
                <h2 style="color: black; margin: 5px 0;">{sentiment_df.loc[sentiment_df['Percentage'].idxmax(), 'Sentiment']}</h2>
            </div>
            <div style="margin: 15px 0;">
                <p style="font-size: 0.9rem; margin: 5px 0; opacity: 0.9;">Sentiment Diversity:</p>
                <h2 style="color: black; margin: 5px 0;">{"High" if abs(pos - neg) < 20 else "Low"}</h2>
            </div>
            <div style="margin: 15px 0;">
                <p style="font-size: 0.9rem; margin: 5px 0; opacity: 0.9;">Overall Mood:</p>
                <h2 style="color: black; margin: 5px 0;">{"Positive" if net_sent > 10 else "Neutral" if net_sent > -10 else "Concerning"}</h2>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Recent Comments Section with modern table
    st.markdown("### Recent Comments Preview")

    cols_to_show = [c for c in ["video_title", "text", "published_at", "predicted_label", "sentiment_score"]
                    if c in comments.columns]

    recent_comments = comments[cols_to_show].head(10).copy()

    # Add sentiment emoji
    if 'predicted_label' in recent_comments.columns:
        emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
        recent_comments['Sentiment'] = recent_comments['predicted_label'].map(emoji_map) + ' ' + recent_comments[
            'predicted_label'].str.capitalize()
        recent_comments = recent_comments.drop('predicted_label', axis=1)

    # Rename columns for better display
    column_rename = {
        'video_title': 'Video',
        'text': 'Comment',
        'published_at': 'Date',
        'sentiment_score': 'Score'
    }
    recent_comments = recent_comments.rename(columns=column_rename)

    st.dataframe(
        recent_comments,
        use_container_width=True,
        hide_index=True,
        height=400
    )

    st.divider()
    st.markdown("### Advanced Performance Indicators")

    # ===== ROW 1: Sentiment Velocity & Response Rate =====
    col1, col2, col3 = st.columns(3)

    with col1:
        # Sentiment Velocity (7-day trend)
        if 'published_at' in comments.columns:
            comments['date'] = pd.to_datetime(comments['published_at']).dt.date
            recent_7d = comments[comments['date'] >= (pd.Timestamp.now().date() - pd.Timedelta(days=7))]
            older_7d = comments[(comments['date'] >= (pd.Timestamp.now().date() - pd.Timedelta(days=14))) &
                                (comments['date'] < (pd.Timestamp.now().date() - pd.Timedelta(days=7)))]

            if len(recent_7d) > 0 and len(older_7d) > 0:
                recent_pos = (recent_7d['predicted_label'] == 'positive').mean() * 100
                older_pos = (older_7d['predicted_label'] == 'positive').mean() * 100
                velocity = recent_pos - older_pos

                velocity_color = "#00b894" if velocity > 0 else "#e17055"
                velocity_icon = "📈" if velocity > 0 else "📉"
                velocity_text = "Improving" if velocity > 0 else "Declining"

                st.markdown(f"""
                    <div style="
                        background: linear-gradient(135deg, {velocity_color} 0%, {'#00cec9' if velocity > 0 else '#d63031'} 100%);
                        padding: 1.5rem;
                        border-radius: 15px;
                        text-align: center;
                        box-shadow: 0 8px 25px rgba(102,126,234,0.3);
                    ">
                        <div style="font-size: 2rem; margin-bottom: 0.5rem;">{velocity_icon}</div>
                        <h4 style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">SENTIMENT VELOCITY</h4>
                        <h2 style="color: white; font-size: 2rem; margin: 10px 0;">{velocity:+.1f}%</h2>
                        <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">{velocity_text}</p>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.info("Need 14 days of data for velocity calculation")

    with col2:
        # Audience Response Rate (comments per view)
        if 'viewcount' in videos.columns and 'commentcount' in videos.columns:
            total_views = videos['viewcount'].sum()
            total_comments = len(comments)
            response_rate = (total_comments / total_views * 100) if total_views > 0 else 0

            # Industry benchmark: 0.5-2% is good
            benchmark_status = "Excellent" if response_rate > 2 else "Good" if response_rate > 0.5 else "Needs Improvement"
            benchmark_color = "#00b894" if response_rate > 2 else "#ffd700" if response_rate > 0.5 else "#e17055"

            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(102,126,234,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">💬</div>
                    <h4 style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">RESPONSE RATE</h4>
                    <h2 style="color: white; font-size: 2rem; margin: 10px 0;">{response_rate:.2f}%</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">{benchmark_status}</p>
                </div>
                """, unsafe_allow_html=True)

    with col3:
        # Engagement Quality Score (weighted metric)
        if 'sentiment_score' in comments.columns and 'viewcount' in videos.columns:
            # Calculate average sentiment score
            avg_sentiment = comments['sentiment_score'].mean()

            # Calculate engagement depth (comments per video)
            comments_per_video = len(comments) / len(videos) if len(videos) > 0 else 0

            # Normalize and combine (0-100 scale)
            sentiment_normalized = (avg_sentiment + 1) * 50  # Scale from -1,1 to 0,100
            engagement_normalized = min(comments_per_video / 50 * 100, 100)  # Cap at 100

            quality_score = (sentiment_normalized * 0.6 + engagement_normalized * 0.4)

            quality_rating = "Exceptional" if quality_score > 80 else "Strong" if quality_score > 60 else "Fair" if quality_score > 40 else "Weak"

            st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #F63366 0%, #ff6b9d 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(246,51,102,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">⭐</div>
                    <h4 style="color: rgba(255,255,255,0.9); font-size: 0.9rem; margin: 0;">QUALITY SCORE</h4>
                    <h2 style="color: white; font-size: 2rem; margin: 10px 0;">{quality_score:.0f}/100</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 0; font-size: 0.9rem;">{quality_rating}</p>
                </div>
                """, unsafe_allow_html=True)

    # ===== ROW 2: Detailed Breakdown =====
    st.markdown("<br>", unsafe_allow_html=True)

    col4, col5, col6, col7 = st.columns(4)

    with col4:
        # Average Comment Length
        if 'text' in comments.columns:
            avg_length = comments['text'].str.len().mean()
            st.metric("📝 Avg Comment Length", f"{avg_length:.0f} chars",
                      help="Longer comments often indicate higher engagement")

    with col5:
        # Comments per Day (Recent 30 days)
        if 'published_at' in comments.columns:
            recent_30d = comments[comments['date'] >= (pd.Timestamp.now().date() - pd.Timedelta(days=30))]
            comments_per_day = len(recent_30d) / 30
            st.metric("📅 Daily Comments (30d)", f"{comments_per_day:.1f}",
                      help="Average comments received per day in the last month")

    with col6:
        # Sentiment Consistency (Std Dev)
        if 'sentiment_score' in comments.columns:
            consistency = 100 - (comments['sentiment_score'].std() * 50)  # Lower std = higher consistency
            consistency = max(0, min(100, consistency))
            st.metric("🎯 Sentiment Consistency", f"{consistency:.0f}%",
                      help="How consistent is your audience sentiment")

    with col7:
        # Positive Momentum
        if 'predicted_label' in comments.columns:
            last_100 = comments.tail(100)
            positive_momentum = (last_100['predicted_label'] == 'positive').mean() * 100
            st.metric("🚀 Recent Positivity", f"{positive_momentum:.0f}%",
                      help="Positive sentiment in last 100 comments")

    # ===== Sentiment Timeline Sparkline =====
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Sentiment Timeline (Last 30 Days)")

    if 'published_at' in comments.columns:
        recent_30d = comments[comments['date'] >= (pd.Timestamp.now().date() - pd.Timedelta(days=30))].copy()

        if not recent_30d.empty:
            daily_sentiment = recent_30d.groupby('date').agg({
                'predicted_label': lambda x: (x == 'positive').mean() * 100
            }).reset_index()
            daily_sentiment.columns = ['date', 'positive_pct']

            fig_timeline = go.Figure()

            fig_timeline.add_trace(go.Scatter(
                x=daily_sentiment['date'],
                y=daily_sentiment['positive_pct'],
                mode='lines+markers',
                name='Positive %',
                line=dict(color='#00b894', width=3),
                fill='tozeroy',
                fillcolor='rgba(0, 184, 148, 0.1)',
                marker=dict(size=6, color='#00b894')
            ))

            fig_timeline.add_hline(y=50, line_dash="dash", line_color="gray",
                                   annotation_text="50% Baseline", annotation_position="right")

            fig_timeline.update_layout(
                height=250,
                margin=dict(l=0, r=0, t=0, b=0),
                xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', range=[0, 100]),
                hovermode='x unified',
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(family="Poppins, sans-serif", size=10)
            )

            st.plotly_chart(fig_timeline, use_container_width=True)

    # Footer with insights
    st.markdown("""
    <div style="
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        margin-top: 2rem;
        text-align: center;
    ">
        <h3 style="color: white; margin: 0;">💡 Quick Insight</h3>
        <p style="font-size: 1.1rem; margin: 10px 0 0 0; opacity: 0.95;">
            Your channel is receiving <strong>{:,}</strong> comments across <strong>{:,}</strong> videos.
            The overall sentiment is <strong>{}</strong> with a net sentiment score of <strong>{:.1f}%</strong>.
        </p>
    </div>
    """.format(
        total_comments,
        total_videos,
        "positive" if net_sent > 10 else "neutral" if net_sent > -10 else "concerning",
        net_sent
    ), unsafe_allow_html=True)