import os
import joblib
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from src.config import CLEANED_COMMENTS, TOKENIZER_PKL, BILSTM_MODEL, LABEL_ENCODER_PKL, LABELLED_COMMENTS


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


# =============================
# Utility Functions
# =============================
def load_cleaned_comments(path=CLEANED_COMMENTS):
    """Load cleaned comments from CSV and ensure datetime format is consistent."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Cleaned comments not found at {path} (run preprocessing first).")

    df = pd.read_csv(path)

    # Ensure required column exists
    if 'text_clean' not in df.columns:
        raise ValueError("Cleaned comments CSV must contain a 'text_clean' column.")

    # Handle published_at column if present
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], errors='coerce')
        df['published_at'] = df['published_at'].dt.tz_localize(None)  # remove timezone safely

    df['text_clean'] = df['text_clean'].astype(str)
    return df


# =============================
# Main Inference Function
# =============================
def predict_and_save(maxlen=150):
    """Run inference using trained BiLSTM model and save labelled CSV (overwrite mode)."""
    df = load_cleaned_comments()
    if df.empty:
        print("No comments to infer.")
        return

    # Load model components
    if not os.path.exists(TOKENIZER_PKL):
        raise FileNotFoundError(f"Tokenizer not found at {TOKENIZER_PKL}.")
    if not os.path.exists(BILSTM_MODEL):
        raise FileNotFoundError(f"Model file not found at {BILSTM_MODEL}.")
    if not os.path.exists(LABEL_ENCODER_PKL):
        raise FileNotFoundError(f"Label encoder not found at {LABEL_ENCODER_PKL}.")

    print("🔹 Loading BiLSTM model and tokenizer...")
    tokenizer = joblib.load(TOKENIZER_PKL)
    model = load_model(BILSTM_MODEL, custom_objects={'Attention': Attention})
    le = joblib.load(LABEL_ENCODER_PKL)

    # Tokenize & pad
    texts = df['text_clean'].tolist()
    seqs = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(seqs, maxlen=maxlen, padding='post', truncating='post')

    # Predict
    print("🔹 Running inference...")
    preds = model.predict(X, verbose=1)

    # Handle binary or multi-class predictions
    if preds.ndim == 1 or preds.shape[1] == 1:
        probs = preds.flatten()
        labels_num = (probs > 0.5).astype(int)
        confidence = probs
    else:
        labels_num = np.argmax(preds, axis=1)
        confidence = np.max(preds, axis=1)

    # Decode labels
    labels = le.inverse_transform(labels_num)

    # Attach predictions
    df['predicted_label'] = labels
    df['predicted_confidence'] = confidence

    # Ensure output directory exists
    os.makedirs(os.path.dirname(LABELLED_COMMENTS), exist_ok=True)

    # Replace the existing file (instead of appending)
    df.to_csv(LABELLED_COMMENTS, index=False, encoding="utf-8-sig")
    print(f"Saved labelled comments to {LABELLED_COMMENTS} ({len(df)} rows, file replaced).")
