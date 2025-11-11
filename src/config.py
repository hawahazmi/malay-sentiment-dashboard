# src/config.py
from dotenv import load_dotenv

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Load .env automatically
load_dotenv()

# YouTube API: set environment variable YT_API_KEY before running
YT_API_KEY = os.getenv("YOUTUBE_API_KEY")  # REQUIRED

# Channel ID for PrimeWorks Studios (example)
CHANNEL_ID = os.getenv("CHANNEL_ID")

# Data directories (relative to repo root)
ROOT = os.path.dirname(os.path.dirname(__file__))
DATA_RAW = os.path.join(ROOT, "data", "raw")
DATA_PROC = os.path.join(ROOT, "data", "processed")
DATA_EXT = os.path.join(ROOT, "data", "external")
MODELS_DIR = os.path.join(ROOT, "data", "model")
LOGS_DIR = os.path.join(ROOT, "logs")

# Filenames
TRAIN_RAW = os.path.join(DATA_RAW, "training_labelled_sentiment.csv")
VIDEOS_CSV = os.path.join(DATA_RAW, "youtube_videos.csv")
COMMENTS_CSV = os.path.join(DATA_RAW, "youtube_comments.csv")
SLANG_JSON = os.path.join(DATA_RAW, "slang_malay_dict.json")
EMOJI_JSON = os.path.join(DATA_RAW, "emoji_dict.json")
STOPWORDS_JSON = os.path.join(DATA_RAW, "stopwords.json")

CLEANED_COMMENTS = os.path.join(DATA_PROC, "cleaned_youtube_comments.csv")
CLEANED_TRAINING = os.path.join(DATA_PROC, "cleaned_training_data.csv")
LABELLED_COMMENTS = os.path.join(DATA_PROC, "labelled_youtube_comments.csv")

# Model files
TOKENIZER_PKL = os.path.join(MODELS_DIR, "tokenizer.pkl")
BILSTM_MODEL = os.path.join(MODELS_DIR, "bilstm_model.h5")
LABEL_ENCODER_PKL = os.path.join(MODELS_DIR, "label_encoder.pkl")

# Create directories if not exist
os.makedirs(DATA_RAW, exist_ok=True)
os.makedirs(DATA_PROC, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
