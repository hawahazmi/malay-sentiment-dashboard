import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
from src.config import CLEANED_TRAINING, MODELS_DIR, TOKENIZER_PKL, BILSTM_MODEL, LABEL_ENCODER_PKL


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


# ---------------------------
# Data Loading
# ---------------------------
def load_training_data(path=CLEANED_TRAINING):
    """Loads and prepares training data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training CSV not found at {path}. Please run preprocessing.")
    df = pd.read_csv(path)
    if 'text_clean' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Training CSV must contain 'text_clean' and 'sentiment' columns.")
    df.dropna(subset=['text_clean', 'sentiment'], inplace=True)
    return df


# ---------------------------
# Tokenizer Preparation
# ---------------------------
def prepare_tokenizer(texts, num_words=20000):
    """Fits and saves the Keras Tokenizer."""
    tokenizer = Tokenizer(num_words=num_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(texts)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(tokenizer, TOKENIZER_PKL)
    print(f"[tokenizer] saved to {TOKENIZER_PKL}")
    return tokenizer


# ---------------------------
# Compute Class Weights
# ---------------------------
def compute_class_weights(y_labels):
    """Computes class weights for handling class imbalance."""
    classes = np.unique(y_labels)
    weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_labels)
    class_weight_dict = dict(zip(classes, weights))

    print("\n===== Class Weight Analysis =====")
    for cls, weight in class_weight_dict.items():
        count = np.sum(y_labels == cls)
        print(f"Class {cls}: weight={weight:.4f} | samples={count} ({count/len(y_labels)*100:.2f}%)")
    print("="*35)
    return class_weight_dict


# ---------------------------
# Model Architecture
# ---------------------------
def build_bilstm(vocab_size, embed_dim=300, maxlen=150, num_classes=3):
    """Defines the BiLSTM-Attention model architecture (no pretrained embeddings)."""
    inputs = Input(shape=(maxlen,))
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embed_dim,
        input_length=maxlen,
        trainable=True  # learn embeddings from scratch
    )(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = Attention()(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=1e-4),
        metrics=['accuracy']
    )
    return model


# ---------------------------
# Train Function
# ---------------------------
def train_model(
        epochs=20,
        batch_size=64,
        num_words=20000,
        maxlen=150,
        embed_dim=300,
        use_class_weights=True
):
    """Main function to load data, build, train, and evaluate the model."""
    df = load_training_data()
    texts = df['text_clean'].tolist()
    labels = df['sentiment'].tolist()

    # Encode Labels
    le = LabelEncoder()
    y_num = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y_cat = np.eye(num_classes)[y_num]

    # Print class distribution
    print("\n===== Sentiment Class Distribution =====")
    for cls_name, cls_idx in zip(le.classes_, range(num_classes)):
        count = np.sum(y_num == cls_idx)
        print(f"{cls_name}: {count} ({count/len(y_num)*100:.2f}%)")
    print("="*40)

    # Compute class weights
    class_weight_dict = compute_class_weights(y_num) if use_class_weights else None

    # Tokenize and Pad
    tokenizer = prepare_tokenizer(texts, num_words=num_words)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.25, random_state=42, stratify=y_num
    )

    # Build and Compile Model
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    model = build_bilstm(vocab_size, embed_dim, maxlen, num_classes)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(BILSTM_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6, verbose=1)

    # Train Model
    print("\n===== Starting Model Training =====")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop, reduce_lr],
        class_weight=class_weight_dict,
        verbose=2
    )

    # Save Artifacts
    joblib.dump(le, LABEL_ENCODER_PKL)
    print(f"[label encoder] saved to {LABEL_ENCODER_PKL}")
    print(f"Training completed. Best model saved to {BILSTM_MODEL}")

    # Evaluate Model
    print("\n===== Evaluating Model Performance =====")
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)

    report = classification_report(y_val_true, y_val_pred, target_names=le.classes_, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose().round(4)
    metrics_path = os.path.join(MODELS_DIR, "bilstm_metrics.csv")
    metrics_df.to_csv(metrics_path)
    print(metrics_df)

    cm = confusion_matrix(y_val_true, y_val_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_df.to_csv(os.path.join(MODELS_DIR, "bilstm_confusion_matrix.csv"))
    print("\nConfusion Matrix:\n", cm_df)

    print("\nModel training and evaluation completed successfully.")


# ---------------------------
# Main Run
# ---------------------------
if __name__ == "__main__":
    train_model()
