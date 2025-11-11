import os
import re
import json
import emoji
from src.config import SLANG_JSON, EMOJI_JSON, STOPWORDS_JSON


# =============================
# Load dictionaries
# =============================
def load_json_dict(path: str):
    '''Generic loader for slang, emoji, or stopword JSON files.'''
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        print(f"File not found: {path}")
        return {}


# Slang and emoji mappings
SLANG = {k.lower(): v.lower() for k, v in load_json_dict(SLANG_JSON).items()}
EMOJI = {k.lower(): v.lower() for k, v in load_json_dict(EMOJI_JSON).items()}


# Stopwords (Malay + English)
def load_stopwords(path=STOPWORDS_JSON):
    data = load_json_dict(path)
    malay = set([w.lower() for w in data.get("malay", [])])
    english = set([w.lower() for w in data.get("english", [])])
    return malay.union(english)


STOPWORDS = load_stopwords()

# =============================
# Regex patterns
# =============================
URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#\w+")
NON_ALPHANUM = re.compile(r"[^a-zA-Z\u00C0-\u017F\s]")
MULTISPACE_RE = re.compile(r"\s+")
REPEATED_CHAR_RE = re.compile(r"(.)\1{2,}")  # "heyyy" → "hey"

# Negation patterns
NEGATION_WORDS = ["tak", "tidak", "bukan", "jgn", "jangan"]

# Malay intensifiers (kata penguat)
KATA_PENGUAT = [
    "sangat", "teruk", "gila", "betul", "betul-betul",
    "amat", "habis", "benar", "teramat", "terlebih"
]


# =============================
# Core Normalization Functions
# =============================
def emoji_to_malay(text: str) -> str:
    """Replace emojis with Malay words using EMOJI_MALAY_MAP or emoji.demojize()."""
    for emo, malay_word in EMOJI.items():
        text = text.replace(emo, f" {malay_word} ")
    text = emoji.demojize(text, language="my")
    text = text.replace(":", " ")
    return text


def normalize_text(t: str) -> str:
    """Normalize and clean social media text."""
    if not isinstance(t, str) or not t.strip():
        return ""

    txt = t.strip().lower()
    txt = emoji_to_malay(txt)
    txt = URL_RE.sub(" ", txt)
    txt = MENTION_RE.sub(" ", txt)
    txt = HASHTAG_RE.sub(" ", txt)
    txt = re.sub(r"\d+", " ", txt)
    txt = NON_ALPHANUM.sub(" ", txt)
    txt = REPEATED_CHAR_RE.sub(r"\1", txt)
    txt = MULTISPACE_RE.sub(" ", txt).strip()
    return txt


def expand_slang(text: str):
    """Expand slang words using slang dictionary."""
    if not text:
        return text
    words = text.split()
    return " ".join([SLANG.get(w, w) for w in words])


def reduce_repetitions(text: str) -> str:
    if not text:
        return text
    # Replaces 'aaat' with 'at', 'bbbb' with 'b'
    return REPEATED_CHAR_RE.sub(r"\1", text)


def handle_negations(text: str, tag_negation: bool = True) -> str:
    if not text:
        return text

    words = text.split()
    new_words = []
    skip_next = False

    for i, word in enumerate(words):
        if skip_next:
            skip_next = False
            continue

        # Handle "x" prefix words: e.g., xbest → tidak best
        if word.startswith("x") and len(word) > 1 and word[1:].isalpha():
            negated_word = word[1:]
            if tag_negation:
                new_words.append(f"NOT_{negated_word}")
            else:
                new_words.append("tidak")
                new_words.append(negated_word)
            continue

        # Handle explicit negations
        if word in NEGATION_WORDS and i + 1 < len(words):
            negated_word = words[i + 1]
            if tag_negation:
                new_words.append(f"NOT_{negated_word}")
            else:
                new_words.append(word)
                new_words.append(negated_word)
            skip_next = True
        else:
            new_words.append(word)

    return " ".join(new_words)


def handle_intensifier(text: str) -> str:
    if not text:
        return text

    temp_text = text

    for kp in KATA_PENGUAT:
        pattern = rf'(\b\w+)\s+{kp}\b'
        temp_text = re.sub(pattern, rf'\1_{kp}', temp_text)

    return temp_text


def remove_stopwords(text: str):
    """Remove Malay + English stopwords."""
    if not text:
        return text
    words = [w for w in text.split() if w not in STOPWORDS]
    return " ".join(words)


def reduce_noise(text: str) -> str:
    txt = re.sub(r"\b[a-zA-Z]\b", " ", text)
    txt = re.sub(r"\s{2,}", " ", txt)
    return txt.strip()

