import os
import pandas as pd
from src.config import COMMENTS_CSV, CLEANED_COMMENTS
from src.preprocessing.translator import translate_text_to_malay
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


def preprocess_comments_file(inpath=COMMENTS_CSV, outpath=CLEANED_COMMENTS):
    if not os.path.exists(inpath):
        print(f"No comment file found at {inpath}")
        return None

    df = pd.read_csv(inpath)
    if df.empty:
        print("No comments to preprocess.")
        return None

    print(f"Preprocessing {len(df)} comments...")

    df["text_clean"] = df["text"].astype(str)

    df["text_clean"] = df["text_clean"].apply(handle_kata_ganda)
    df["text_clean"] = df["text_clean"].apply(normalize_text)
    df["text_clean"] = df["text_clean"].apply(expand_slang)
    df["text_clean"] = df["text_clean"].apply(reduce_repetitions)
    df["text_clean"] = df["text_clean"].apply(expand_slang)
    df["text_clean"] = df["text_clean"].apply(lambda x: handle_negations(x, tag_negation=True))
    df["text_clean"] = df["text_clean"].apply(handle_intensifier)
    df["text_clean"] = df["text_clean"].apply(translate_text_to_malay)
    df["text_clean"] = df["text_clean"].apply(remove_stopwords)
    df["text_clean"] = df["text_clean"].apply(reduce_noise)

    df = df.dropna(subset=["text_clean"])
    df = df[df["text_clean"].str.strip() != ""]

    # Save results
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    df.to_csv(outpath, index=False, encoding="utf-8-sig")

    print(f"Saved cleaned scraped comments → {outpath} ({len(df)} rows).")
    return df


# =============================
# Run directly
# =============================
if __name__ == "__main__":
    os.makedirs(os.path.dirname(CLEANED_COMMENTS), exist_ok=True)
    preprocess_comments_file()
