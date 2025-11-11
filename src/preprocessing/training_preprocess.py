import os
import pandas as pd
from src.config import CLEANED_TRAINING, DATA_RAW
from src.preprocessing.translator import translate_text_to_malay
from src.preprocessing.preprocess import (
    normalize_text,
    expand_slang,
    handle_negations,
    remove_stopwords,
    reduce_noise,
    handle_intensifier,
    reduce_repetitions
)


def preprocess_training_file(inpath=os.path.join(DATA_RAW, "training_labelled_sentiment.csv"), outpath=CLEANED_TRAINING):
    if not os.path.exists(inpath):
        print(f"Training dataset not found: {inpath}")
        return None

    df = pd.read_csv(inpath)
    if "comment" not in df.columns:
        print("Training data missing 'comment' column.")
        return None

    print(f"Cleaning labelled training data ({len(df)} rows)...")

    df["text_clean"] = df["comment"].astype(str)
    df["text_clean"] = df["text_clean"].apply(reduce_noise)
    df["text_clean"] = df["text_clean"].apply(normalize_text)
    df["text_clean"] = df["text_clean"].apply(translate_text_to_malay)
    df["text_clean"] = df["text_clean"].apply(expand_slang)
    df["text_clean"] = df["text_clean"].apply(lambda x: handle_negations(x, tag_negation=True))
    df["text_clean"] = df["text_clean"].apply(handle_intensifier)
    df["text_clean"] = df["text_clean"].apply(reduce_repetitions)
    df["text_clean"] = df["text_clean"].apply(remove_stopwords)

    df = df.dropna(subset=["text_clean"])
    df = df[df["text_clean"].str.strip() != ""]

    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned training dataset → {outpath} ({len(df)} rows).")
    return df


# =============================
# Run directly
# =============================
if __name__ == "__main__":
    preprocess_training_file()
