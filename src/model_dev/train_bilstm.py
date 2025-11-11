import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight  # ✅ NEW IMPORT
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM, Dense,
    Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import AdamW
from src.config import CLEANED_TRAINING, MODELS_DIR, TOKENIZER_PKL, BILSTM_MODEL, LABEL_ENCODER_PKL
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer
import malaya


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
    """Loads and prepares training data, including negation handling."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Training CSV not found at {path}. Please run preprocessing.")
    df = pd.read_csv(path)
    if 'text_clean' not in df.columns or 'sentiment' not in df.columns:
        raise ValueError("Training CSV must contain 'text_clean' and 'sentiment' columns.")
    df.dropna(subset=['text_clean', 'sentiment'], inplace=True)
    return df


# ---------------------------
# Tokenizer & Embeddings
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
# Load Pretrained Embeddings (Fixed)
# ---------------------------
def load_pretrained_embeddings(tokenizer, embed_path, embed_dim=300, num_words=20000):
    """
    Loads pretrained embeddings (Malaya or local), ensures correct shape alignment.
    Returns embedding_matrix of shape (vocab_size, embed_dim)
    """
    from gensim.models import KeyedVectors

    def fallback_matrix(vocab_size, dim):
        print("[embedding][fallback] Creating random embedding matrix (training will still run).")
        rng = np.random.default_rng(42)
        return rng.normal(0, 0.05, size=(vocab_size, dim)).astype(np.float32)

    os.makedirs(os.path.dirname(embed_path), exist_ok=True)

    # Try to load if available, otherwise fetch from Malaya
    if not os.path.exists(embed_path) or os.path.getsize(embed_path) == 0:
        print(f"[embedding] pretrained file not found or empty at {embed_path}. Attempting to fetch from Malaya...")
        try:
            model_obj = malaya.wordvector.load(model='socialmedia')
            if hasattr(model_obj, 'wv'):
                kv = model_obj.wv
            else:
                kv = model_obj
            kv.save_word2vec_format(embed_path, binary=True)
            print(f"[embedding] Saved converted Malaya embeddings to {embed_path}")
        except Exception as e:
            print(f"[embedding] Failed to download Malaya embeddings: {e}")
            vocab_size = min(num_words, len(tokenizer.word_index) + 1)
            return fallback_matrix(vocab_size, embed_dim)

    # Load Word2Vec binary
    try:
        print(f"[embedding] Loading KeyedVectors from {embed_path} ...")
        wv = KeyedVectors.load_word2vec_format(embed_path, binary=True)
    except Exception as e:
        print(f"[embedding] Error loading saved KeyedVectors: {e}")
        vocab_size = min(num_words, len(tokenizer.word_index) + 1)
        return fallback_matrix(vocab_size, embed_dim)

    vocab_size = min(num_words, len(tokenizer.word_index) + 1)
    embedding_matrix = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    found = 0
    for word, idx in tokenizer.word_index.items():
        if idx >= vocab_size:
            continue
        if word in wv:
            embedding_matrix[idx] = wv[word]
            found += 1

    print(f"[embedding] Filled embedding matrix: {found} found in pretrained | shape={embedding_matrix.shape}")
    return embedding_matrix


# ---------------------------
# Compute Class Weights
# ---------------------------
def compute_class_weights(y_labels):
    """
    Computes class weights to handle imbalanced sentiment classes.

    Args:
        y_labels: array of numeric labels (0, 1, 2, etc.)

    Returns:
        dict: {0: weight_0, 1: weight_1, 2: weight_2, ...}
    """
    classes = np.unique(y_labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=y_labels
    )
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
def build_bilstm(vocab_size, embedding_matrix, maxlen=150, num_classes=3):
    """Defines the BiLSTM-Attention model architecture."""
    inputs = Input(shape=(maxlen,))
    x = Embedding(
        input_dim=vocab_size,
        output_dim=embedding_matrix.shape[1],
        weights=[embedding_matrix],
        input_length=maxlen,
        trainable=True  # <--- fine-tune embeddings
    )(inputs)

    x = Bidirectional(LSTM(128, return_sequences=True, dropout=0.3))(x)
    x = Bidirectional(LSTM(64, return_sequences=True, dropout=0.3))(x)
    x = Attention()(x)  # <--- Custom attention layer
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.4)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs, outputs)
    model.compile(
        loss='categorical_crossentropy',
        optimizer=AdamW(learning_rate=1e-4, weight_decay=1e-5),
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
        embed_path=r"D:\UTP\UG\FYP\dev_root\data\external\word2vec_bahasa.bin",
        embed_dim=300,
        use_class_weights=True  # ✅ NEW PARAMETER
):
    """Main function to load data, build, train, and evaluate the model."""
    df = load_training_data()
    texts = df['text_clean'].tolist()
    labels = df['sentiment'].tolist()

    # 1. Prepare Labels
    le = LabelEncoder()
    y_num = le.fit_transform(labels)
    num_classes = len(le.classes_)
    y_cat = np.eye(num_classes)[y_num]

    # Analyze class distribution
    print("\n===== Sentiment Class Distribution =====")
    for cls_name, cls_idx in zip(le.classes_, range(num_classes)):
        count = np.sum(y_num == cls_idx)
        print(f"{cls_name}: {count} ({count/len(y_num)*100:.2f}%)")
    print("="*40)

    # Compute class weights
    class_weight_dict = None
    if use_class_weights:
        class_weight_dict = compute_class_weights(y_num)

    # Tokenize and Pad
    tokenizer = prepare_tokenizer(texts, num_words=num_words)
    sequences = tokenizer.texts_to_sequences(texts)
    X = pad_sequences(sequences, maxlen=maxlen, padding='post', truncating='post')

    # Split Data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_cat, test_size=0.15, random_state=42, stratify=y_num
    )

    # Load Embeddings
    embedding_matrix = load_pretrained_embeddings(tokenizer, embed_path, embed_dim, num_words)
    vocab_size = min(num_words, len(tokenizer.word_index) + 1)

    # Build and Compile Model
    model = build_bilstm(vocab_size, embedding_matrix, maxlen=maxlen, num_classes=num_classes)
    model.summary()

    # Callbacks
    checkpoint = ModelCheckpoint(BILSTM_MODEL, monitor='val_accuracy', save_best_only=True, verbose=1)
    earlystop = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-6, verbose=1)

    # Train Model with Class Weights
    print("\n===== Starting Model Training =====")
    if use_class_weights:
        print("Training WITH class weights (addressing class imbalance)")
    else:
        print("Training WITHOUT class weights")

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[checkpoint, earlystop, reduce_lr],
        class_weight=class_weight_dict,
        verbose=2
    )

    # Save artifacts
    joblib.dump(le, LABEL_ENCODER_PKL)
    print(f"\n[label encoder] saved to {LABEL_ENCODER_PKL}")

    print(f"Training completed. Best model saved to {BILSTM_MODEL}")

    # Evaluation
    print("\n===== Evaluating Model Performance =====")
    y_val_true = np.argmax(y_val, axis=1)
    y_val_pred = np.argmax(model.predict(X_val), axis=1)

    # Classification Report
    report = classification_report(y_val_true, y_val_pred, target_names=le.classes_, output_dict=True)
    metrics_df = pd.DataFrame(report).transpose().round(4)
    metrics_path = os.path.join(MODELS_DIR, "bilstm_metrics.csv")
    metrics_df.to_csv(metrics_path, index=True)
    print("\n===== Validation Performance Metrics =====")
    print(metrics_df)
    print(f"[metrics] saved to {metrics_path}")

    # Confusion Matrix
    cm = confusion_matrix(y_val_true, y_val_pred)
    cm_df = pd.DataFrame(cm, index=le.classes_, columns=le.classes_)
    cm_path = os.path.join(MODELS_DIR, "bilstm_confusion_matrix.csv")
    cm_df.to_csv(cm_path)
    print(f"[confusion matrix] saved to {cm_path}")

    # Calculate per-class metrics for detailed analysis
    print("\n===== Per-Class Performance Summary =====")
    for i, cls_name in enumerate(le.classes_):
        true_count = np.sum(y_val_true == i)
        pred_count = np.sum(y_val_pred == i)
        correct = np.sum((y_val_true == i) & (y_val_pred == i))

        precision = metrics_df.loc[cls_name, 'precision'] if cls_name in metrics_df.index else 0
        recall = metrics_df.loc[cls_name, 'recall'] if cls_name in metrics_df.index else 0
        f1 = metrics_df.loc[cls_name, 'f1-score'] if cls_name in metrics_df.index else 0

        print(f"\n{cls_name.upper()}:")
        print(f"  Samples in validation: {true_count}")
        print(f"  Correctly predicted: {correct} ({correct/true_count*100:.1f}%)")
        print(f"  Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f}")