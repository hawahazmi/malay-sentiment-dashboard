# src/dashboard_dev/modules/p4_predict_sentiment.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import load_model
import plotly.express as px
import plotly.graph_objects as go
import re
from src.config import BILSTM_MODEL, TOKENIZER_PKL
from src.preprocessing.preprocess import (
    normalize_text,
    expand_slang,
    handle_negations,
    remove_stopwords,
    reduce_noise,
    handle_intensifier,
    reduce_repetitions,
    handle_kata_ganda
)
from src.preprocessing.translator import translate_text_to_malay


# ---------------------------
# Custom Attention Layer
# ---------------------------
class Attention(Layer):
    """Keras Attention Layer for BiLSTM."""

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(Attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        return K.sum(context, axis=1)


# ------------------------------------------------------
#  Load tokenizer and BiLSTM model
# ------------------------------------------------------
@st.cache_resource
def load_model_and_tokenizer():
    import pickle
    try:
        with open(TOKENIZER_PKL, "rb") as f:
            tokenizer = pickle.load(f)
        model = load_model(BILSTM_MODEL, custom_objects={'Attention': Attention})
        return model, tokenizer
    except Exception as e:
        st.error(f"Failed to load model/tokenizer: {e}")
        return None, None


# ------------------------------------------------------
#  Text preprocessing
# ------------------------------------------------------
def clean_text(text):
    raw_text = str(text)
    text = handle_kata_ganda(raw_text)
    text = normalize_text(text)
    text = expand_slang(text)
    text = reduce_repetitions(text)
    text = expand_slang(text)
    text = handle_negations(text)
    text = handle_intensifier(text)
    text = translate_text_to_malay(text)
    text = remove_stopwords(text)
    text = reduce_noise(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ------------------------------------------------------
#  Prediction function
# ------------------------------------------------------
def predict_sentiment(model, tokenizer, text_list):
    max_len = model.input_shape[1]
    cleaned = [clean_text(t) for t in text_list]
    seq = tokenizer.texts_to_sequences(cleaned)
    padded = pad_sequences(seq, maxlen=max_len, padding='post', truncating='post')
    preds = model.predict(padded)
    label_map = {0: "negative", 1: "neutral", 2: "positive"}
    predicted_labels = [label_map[np.argmax(p)] for p in preds]
    confidence = [np.max(p) for p in preds]
    return predicted_labels, confidence, cleaned


# ------------------------------------------------------
#  Streamlit Page
# ------------------------------------------------------
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
            ">Advanced Sentiment Analyzer</h1>
            """, unsafe_allow_html=True)

    # Load model
    model, tokenizer = load_model_and_tokenizer()
    if not model or not tokenizer:
        st.stop()

    # Model Info Card
    with st.expander("ℹ️ About the Model", expanded=False):
        st.markdown("""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 1.5rem;
            border-radius: 15px;
            color: white;
        ">
            <h4 style="color: white; margin-top: 0;">BiLSTM with Attention Mechanism</h4>
            <ul style="margin: 0; padding-left: 1.5rem;">
                <li>Bidirectional Long Short-Term Memory architecture</li>
                <li>Attention layer for context focus</li>
                <li>Pre-trained on Malay sentiment dataset</li>
                <li>Handles negations, intensifiers, and Manglish</li>
                <li>Real-time preprocessing pipeline</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Input Tabs with modern styling
    st.markdown("### Input Method")
    tab1, tab2 = st.tabs(["Manual Input", "CSV Upload"])

    comments_input = []

    with tab1:
        st.markdown("""
        <div style="
            background: rgba(102,126,234,0.05);
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px dashed #667eea;
            margin-bottom: 1rem;
        ">
            <p style="margin: 0; color: #667eea; font-weight: 500;">
                💡 Tip: Enter one comment per line. The model works best with natural Malay language text.
            </p>
        </div>
        """, unsafe_allow_html=True)

        text_input = st.text_area(
            "Enter comment(s):",
            placeholder="Examples:\nVideo ini sangat menarik dan bermanfaat!"
                        "\nKurang menarik, saya tidak suka."
                        "\nBiasa sahaja, tiada apa-apa yang istimewa.",
            height=200,
            help="Type or paste comments here, one per line"
        )
        if text_input.strip():
            comments_input = [c.strip() for c in text_input.split("\n") if c.strip()]
            st.success(f"{len(comments_input)} comment(s) ready for analysis")

    with tab2:
        st.markdown("""
        <div style="
            background: rgba(102,126,234,0.05);
            padding: 1.5rem;
            border-radius: 15px;
            border: 2px dashed #667eea;
            margin-bottom: 1rem;
        ">
            <p style="margin: 0; color: #667eea; font-weight: 500;">
                💡 Tip: Upload a CSV file with a column named 'comment' containing the text to analyze.
            </p>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Upload CSV file", type=["csv"], help="Must contain a 'comment' column")
        if uploaded:
            df = pd.read_csv(uploaded)
            if "comment" not in df.columns:
                st.error("CSV must contain a column named 'comment'.")
            else:
                comments_input = df["comment"].dropna().tolist()
                st.success(f"Loaded {len(comments_input)} comments from file")

                # Show preview
                with st.expander("Preview uploaded data"):
                    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("---")

    # Predict Button with modern styling
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_btn = st.button(
            "Analyze Sentiment",
            use_container_width=True,
            type="primary"
        )

    if predict_btn:
        if not comments_input:
            st.warning("Please enter or upload some comments first.")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.text("Preprocessing comments...")
            progress_bar.progress(25)

            with st.spinner("AI model analyzing sentiment..."):
                preds, confs, cleaned_texts = predict_sentiment(model, tokenizer, comments_input)
                progress_bar.progress(75)

                results = pd.DataFrame({
                    "Original Comment": comments_input,
                    "Cleaned Text": cleaned_texts,
                    "Predicted Sentiment": preds,
                    "Confidence": confs
                })

                progress_bar.progress(100)
                status_text.text("Analysis complete!")

            # Clear progress indicators
            import time
            time.sleep(0.5)
            progress_bar.empty()
            status_text.empty()

            st.success("Sentiment analysis completed successfully!")

            # Results Summary
            st.markdown("### Analysis Summary")

            col1, col2, col3, col4 = st.columns(4)

            pos_count = (results["Predicted Sentiment"] == "positive").sum()
            neu_count = (results["Predicted Sentiment"] == "neutral").sum()
            neg_count = (results["Predicted Sentiment"] == "negative").sum()
            avg_conf = results["Confidence"].mean() * 100

            with col1:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #00b894 0%, #00cec9 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,184,148,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">😊</div>
                    <h2 style="color: white; font-size: 2.5rem; margin: 0;">{pos_count}</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Positive</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #636efa 0%, #a29bfe 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(99,110,250,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">😐</div>
                    <h2 style="color: white; font-size: 2.5rem; margin: 0;">{neu_count}</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Neutral</p>
                </div>
                """, unsafe_allow_html=True)

            with col3:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #e17055 0%, #d63031 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(225,112,85,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">😞</div>
                    <h2 style="color: white; font-size: 2.5rem; margin: 0;">{neg_count}</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Negative</p>
                </div>
                """, unsafe_allow_html=True)

            with col4:
                st.markdown(f"""
                <div style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 1.5rem;
                    border-radius: 15px;
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(102,126,234,0.3);
                ">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🎯</div>
                    <h2 style="color: white; font-size: 2.5rem; margin: 0;">{avg_conf:.0f}%</h2>
                    <p style="color: rgba(255,255,255,0.8); margin: 5px 0 0 0;">Avg Confidence</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            # Visualization
            col_viz1, col_viz2 = st.columns(2)

            with col_viz1:
                # Pie Chart
                dist_data = pd.DataFrame({
                    "Sentiment": ["Positive", "Neutral", "Negative"],
                    "Count": [pos_count, neu_count, neg_count]
                })

                fig_pie = px.pie(
                    dist_data,
                    names="Sentiment",
                    values="Count",
                    hole=0.5,
                    color="Sentiment",
                    color_discrete_map={
                        "Positive": "#00b894",
                        "Neutral": "#636efa",
                        "Negative": "#e17055"
                    }
                )

                fig_pie.update_traces(
                    textinfo="percent+label+value",
                    textfont_size=13,
                    marker=dict(line=dict(color='white', width=3))
                )

                fig_pie.update_layout(
                    title="Sentiment Distribution",
                    font=dict(family="Poppins, sans-serif", size=12),
                    height=400,
                    showlegend=False
                )

                st.plotly_chart(fig_pie, use_container_width=True)

            with col_viz2:
                # Confidence Distribution
                fig_conf = go.Figure(data=[go.Histogram(
                    x=results["Confidence"] * 100,
                    nbinsx=20,
                    marker=dict(
                        color='#667eea',
                        line=dict(color='white', width=1)
                    )
                )])

                fig_conf.update_layout(
                    title="Confidence Score Distribution",
                    xaxis_title="Confidence (%)",
                    yaxis_title="Count",
                    font=dict(family="Poppins, sans-serif", size=12),
                    height=400,
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    xaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
                    yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)')
                )

                st.plotly_chart(fig_conf, use_container_width=True)

            # Results Table
            st.markdown("### Detailed Results")

            # Add emoji and format
            emoji_map = {'positive': '😊', 'neutral': '😐', 'negative': '😞'}
            results['Sentiment'] = results['Predicted Sentiment'].map(emoji_map) + ' ' + results[
                'Predicted Sentiment'].str.capitalize()
            results['Confidence'] = (results['Confidence'] * 100).round(2).astype(str) + '%'

            display_results = results[['Original Comment', 'Sentiment', 'Confidence']]

            st.dataframe(
                display_results,
                use_container_width=True,
                hide_index=True,
                height=200
            )

            # Download Options
            st.markdown("### Export Results")

            col_dl1, col_dl2 = st.columns(2)

            with col_dl1:
                csv = results.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download as CSV",
                    csv,
                    "sentily_predictions.csv",
                    "text/csv",
                    use_container_width=True,
                )

            with col_dl2:
                excel_buffer = pd.ExcelWriter("sentily_predictions.xlsx", engine='xlsxwriter')
                results.to_excel(excel_buffer, index=False, sheet_name='Predictions')
                excel_buffer.close()

                st.download_button(
                    "Download as Excel",
                    open("sentily_predictions.xlsx", "rb").read(),
                    "sentily_predictions.xlsx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True,
                )

            # Insights
            st.markdown("""
            <div style="
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                margin-top: 2rem;
            ">
                <h4 style="color: white; margin: 0 0 10px 0;">💡 Quick Insights</h4>
                <ul style="margin: 0; padding-left: 1.5rem;">
                    <li>Overall sentiment: <strong>{}</strong></li>
                    <li>Model confidence: <strong>{:.1f}%</strong> average</li>
                    <li>Most common sentiment: <strong>{}</strong></li>
                    <li>Sentiment diversity: <strong>{}</strong></li>
                </ul>
            </div>
            """.format(
                "Positive" if pos_count > max(neu_count,
                                              neg_count) else "Neutral" if neu_count > neg_count else "Negative",
                avg_conf,
                "Positive" if pos_count > max(neu_count,
                                              neg_count) else "Neutral" if neu_count > neg_count else "Negative",
                "High" if abs(pos_count - neg_count) < len(results) * 0.3 else "Low"
            ), unsafe_allow_html=True)

            st.divider()
            st.markdown("### Advanced Analysis")

            # ===== Confidence Distribution by Sentiment =====
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("#### Confidence by Sentiment")

                results['Confidence_num'] = results['Confidence'].str.rstrip('%').astype(float)

                fig_conf_sent = px.box(
                    results,
                    x='Predicted Sentiment',
                    y='Confidence_num',
                    color='Predicted Sentiment',
                    color_discrete_map={
                        'positive': '#00b894',
                        'neutral': '#636efa',
                        'negative': '#e17055'
                    }
                )

                fig_conf_sent.update_layout(
                    yaxis_title="Confidence (%)",
                    height=400,
                    showlegend=False,
                    font=dict(family="Poppins, sans-serif"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_conf_sent, use_container_width=True)

            with col2:
                st.markdown("#### Comment Length vs Confidence")

                results['length'] = results['Original Comment'].str.len()

                fig_length_conf = px.scatter(
                    results,
                    x='length',
                    y='Confidence_num',
                    color='Predicted Sentiment',
                    color_discrete_map={
                        'positive': '#00b894',
                        'neutral': '#636efa',
                        'negative': '#e17055'
                    },
                    trendline="lowess"
                )

                fig_length_conf.update_layout(
                    xaxis_title="Comment Length (characters)",
                    yaxis_title="Confidence (%)",
                    height=400,
                    font=dict(family="Poppins, sans-serif"),
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)'
                )

                st.plotly_chart(fig_length_conf, use_container_width=True)

            # ===== Text Complexity Analysis =====
            st.markdown("#### Text Characteristics")

            results['word_count'] = results['Original Comment'].str.split().str.len()
            results['unique_words'] = results['Original Comment'].apply(
                lambda x: len(set(str(x).lower().split()))
            )
            results['lexical_diversity'] = results['unique_words'] / results['word_count']

            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric("Avg Word Count", f"{results['word_count'].mean():.1f}")

            with col2:
                st.metric("Avg Unique Words", f"{results['unique_words'].mean():.1f}")

            with col3:
                st.metric("Lexical Diversity", f"{results['lexical_diversity'].mean():.2f}")