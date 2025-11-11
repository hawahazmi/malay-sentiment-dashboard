import re

# Small bilingual dictionary
EN_TO_MY = {
    # Basic greetings and reactions
    "hello": "hai",
    "hi": "hai",
    "hey": "hoi",
    "thanks": "terima kasih",
    "thank": "terima kasih",
    "welcome": "selamat datang",
    "bye": "selamat tinggal",
    "goodbye": "selamat tinggal",
    "sorry": "maaf",
    "please": "tolong",
    "ok": "okey",
    "okay": "okey",
    "sure": "baiklah",

    # Positive sentiments
    "love": "sayang",
    "like": "suka",
    "good": "baik",
    "great": "hebat",
    "awesome": "hebat gila",
    "cool": "best",
    "wow": "wow",
    "omg": "terkejut",
    "beautiful": "cantik",
    "handsome": "kacak",
    "amazing": "menakjubkan",
    "nice": "bagus",
    "best": "terbaik",
    "perfect": "sempurna",
    "enjoy": "nikmati",
    "fun": "seronok",
    "funny": "kelakar",
    "laugh": "ketawa",
    "cute": "comel",
    "adorable": "comel gila",
    "respect": "hormat",
    "support": "sokong",
    "proud": "bangga",
    "good job": "kerja bagus",
    "well done": "tahniah",

    # Negative sentiments
    "bad": "buruk",
    "boring": "membosankan",
    "bored": "bosan",
    "ugly": "hodoh",
    "hate": "benci",
    "dislike": "tak suka",
    "weird": "pelik",
    "cringe": "malu sendiri",
    "fake": "palsu",
    "annoying": "menyakitkan hati",
    "disappointed": "kecewa",
    "sad": "sedih",
    "angry": "marah",
    "frustrated": "geram",
    "trash": "sampah",
    "lame": "lemah",
    "lazy": "malas",

    # Movie & Film context
    "movie": "filem",
    "film": "filem",
    "show": "rancangan",
    "episode": "episod",
    "scene": "babak",
    "character": "watak",
    "actor": "pelakon lelaki",
    "actress": "pelakon perempuan",
    "director": "pengarah",
    "producer": "penerbit",
    "writer": "penulis skrip",
    "plot": "jalan cerita",
    "storyline": "jalan cerita",
    "trailer": "treler",
    "cinematography": "sinematografi",
    "camera": "kamera",
    "editing": "penyuntingan",
    "script": "skrip",
    "dialogue": "dialog",
    "soundtrack": "runut bunyi",
    "music": "muzik",
    "background": "latar belakang",
    "visual": "visual",
    "effects": "kesan khas",
    "vfx": "kesan visual",
    "bloopers": "adegan lucu",
    "scene cut": "potongan babak",

    # YouTube / Social Media
    "youtube": "youtube",
    "video": "video",
    "channel": "saluran",
    "subscribe": "langgan",
    "sub": "langgan",
    "comment": "komen",
    "share": "kongsi",
    "upload": "muat naik",
    "download": "muat turun",
    "watch": "tonton",
    "view": "tontonan",
    "views": "tontonan",
    "followers": "pengikut",
    "fans": "peminat",
    "supporter": "penyokong",
    "trending": "tular",
    "viral": "tular",
    "stream": "tonton langsung",
    "live": "siaran langsung",
    "reaction": "reaksi",
    "review": "ulasan",
    "content": "kandungan",
    "creator": "pencipta kandungan",
    "influencer": "pempengaruh",
    "collab": "kolaborasi",
    "behind the scenes": "di sebalik tabir",

    # Emotions and expressions
    "lol": "ketawa",
    "lmao": "gelak kuat",
    "rofl": "gelak guling",
    "haha": "ketawa",
    "hehe": "senyum",
    "wtf": "apa ni",
    "eh": "eh",
    "meh": "tak bagus",
    "haih": "mengeluh",
    "sigh": "mengeluh",
    "idk": "tak tahu",
    "ikr": "betul tu",
    "btw": "sebenarnya",
    "imo": "pada pendapat saya",
    "idc": "tak kisah",
    "smh": "geleng kepala",
    "tbh": "sejujurnya",
    "bruh": "adoi",
    "bro": "bro",
    "sis": "sis",
    "dude": "kawan",
    "fam": "kawan-kawan",

    # Time / event words
    "today": "hari ini",
    "yesterday": "semalam",
    "tomorrow": "esok",
    "now": "sekarang",
    "soon": "nanti",
    "later": "kemudian",
    "already": "dah",
    "again": "lagi",
    "still": "masih",
    "finally": "akhirnya",
    "next": "seterusnya",
    "before": "sebelum",
    "after": "lepas",
}


WORD_RE = re.compile(r"\b[a-zA-Z']+\b")


def translate_text_to_malay(text):
    if not isinstance(text, str) or not text:
        return text

    def repl(match):
        w = match.group(0)
        wl = w.lower()
        if wl in EN_TO_MY:
            return EN_TO_MY[wl]
        return w

    # Replace only ASCII words (don't touch Malay words that may contain non-ascii)
    result = re.sub(WORD_RE, repl, text)
    return result
