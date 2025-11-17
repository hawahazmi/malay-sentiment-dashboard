import os
import re
import json
import emoji
from src.config import SLANG_JSON, EMOJI_JSON, STOPWORDS_JSON

from src.preprocessing.translator import translate_text_to_malay


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
# Only collapse repeated characters at the END of a word
REPEATED_CHAR_END_RE = re.compile(r'(\w*?)(\w)\2+$')


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


def handle_kata_ganda(text: str) -> str:
    # Pattern: word + digit "2"
    pattern = r"\b([a-zA-Z]+)2\b"

    def expand(match):
        word = match.group(1)
        return f"{word}-{word}"

    return re.sub(pattern, expand, text)


def remove_trailing_repeats(word: str) -> str:
    """Remove only trailing repeated characters at end of a word."""
    return REPEATED_CHAR_END_RE.sub(lambda m: m.group(1) + m.group(2), word)


def reduce_repetitions(text):
    new_words = []
    for w in text.split():
        new_words.append(remove_trailing_repeats(w))
    return " ".join(new_words)


def normalize_text(t: str) -> str:
    if not isinstance(t, str) or not t.strip():
        return ""

    # lowercasing
    txt = t.strip().lower()
    # separate x from words when it starts with x (xnak, xdpt)
    txt = re.sub(r"\bx([a-zA-Z])", r"x \1", txt)
    # replace emoji
    txt = emoji_to_malay(txt)
    # remove URL
    txt = URL_RE.sub(" ", txt)
    # remove mention
    txt = MENTION_RE.sub(" ", txt)
    # remove hashtag
    txt = HASHTAG_RE.sub(" ", txt)
    # remove numbers
    txt = re.sub(r"\d+", " ", txt)
    # remove non-characters
    txt = NON_ALPHANUM.sub(" ", txt)
    # remove extra space
    txt = MULTISPACE_RE.sub(" ", txt).strip()
    return txt


def expand_slang(text: str):
    """Expand slang words using slang dictionary."""
    if not text:
        return text
    words = text.split()
    return " ".join([SLANG.get(w, w) for w in words])


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


keep_single = {"x", "u", "n", "k"}


def reduce_noise(text: str) -> str:
    return " ".join(
        w if not (len(w) == 1 and w.lower() not in keep_single) else ""
        for w in text.split()
    )


if __name__ == "__main__":
    text = input("Enter malay sentence: ")
    raw_text = str(text)

    ganda = handle_kata_ganda(raw_text)
    normalize = normalize_text(ganda)
    slang1 = expand_slang(normalize)
    remove_repeat = reduce_repetitions(slang1)
    slang2 = expand_slang(remove_repeat)
    negation = handle_negations(slang2)
    intensifier = handle_intensifier(negation)
    translate = translate_text_to_malay(intensifier)
    stopword = remove_stopwords(translate)
    reduce_noise = reduce_noise(stopword)

    print("Raw:\t\t\t\t\t\t\t\t\t\t", raw_text)
    print("After handling informal kata ganda:\t\t\t", ganda)
    print("After normalization:\t\t\t\t\t\t", normalize)
    print("After first slang handling:\t\t\t\t\t", slang1)
    print("After removing repetitive characters at end:", remove_repeat)
    print("After second slang handling:\t\t\t\t", slang2)
    print("After negation handling:\t\t\t\t\t", negation)
    print("After intensifier handling:\t\t\t\t\t", intensifier)
    print("After english to malay translation:\t\t\t", translate)
    print("After stopwords removal:\t\t\t\t\t", stopword)
    print("After noise reduction:\t\t\t\t\t\t", reduce_noise)






