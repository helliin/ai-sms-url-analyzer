import pandas as pd
import numpy as np
import re


# ============================================================
# DATASET B
# ============================================================

dataset_path = "data/sms/exais_clean.csv"

df = pd.read_csv(dataset_path)

print("=" * 70)
print("DATASET B - HATA ÖZELLİK ANALİZİ")
print("=" * 70)

print(f"\nToplam mesaj: {len(df)}")


# ============================================================
# ERROR FILES
# ============================================================

false_negatives = pd.read_csv(
    "data/sms/exais_false_negatives.csv"
)

false_positives = pd.read_csv(
    "data/sms/exais_false_positives.csv"
)

print(f"False Negative: {len(false_negatives)}")
print(f"False Positive: {len(false_positives)}")


# ============================================================
# FEATURE FUNCTION
# ============================================================

def extract_features(text):

    text = str(text)

    characters = len(text)
    words = len(text.split())

    uppercase_letters = sum(
        1 for c in text
        if c.isalpha() and c.isupper()
    )

    letters = sum(
        1 for c in text
        if c.isalpha()
    )

    digits = sum(
        1 for c in text
        if c.isdigit()
    )

    uppercase_ratio = (
        uppercase_letters / letters
        if letters > 0 else 0
    )

    digit_ratio = (
        digits / characters
        if characters > 0 else 0
    )

    url_count = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text
        )
    )

    return {
        "length": characters,
        "word_count": words,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "url_count": url_count,
    }


# ============================================================
# ANALYZE ERROR GROUP
# ============================================================

def analyze_group(name, data):

    features = pd.DataFrame(
        [extract_features(text) for text in data["text"]]
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"\nMesaj sayısı: {len(data)}")

    print(
        f"Ortalama mesaj uzunluğu: "
        f"{features['length'].mean():.2f}"
    )

    print(
        f"Ortalama kelime sayısı: "
        f"{features['word_count'].mean():.2f}"
    )

    print(
        f"Ortalama uppercase oranı: "
        f"{features['uppercase_ratio'].mean():.4f}"
    )

    print(
        f"Ortalama digit oranı: "
        f"{features['digit_ratio'].mean():.4f}"
    )

    print(
        f"URL içeren mesaj oranı: "
        f"{(features['url_count'] > 0).mean():.4f}"
    )

    print(
        f"Çok kısa mesajlar (<= 10 karakter): "
        f"{(features['length'] <= 10).sum()}"
    )

    print(
        f"Kısa mesajlar (<= 30 karakter): "
        f"{(features['length'] <= 30).sum()}"
    )


# ============================================================
# FALSE NEGATIVE
# ============================================================

analyze_group(
    "FALSE NEGATIVE - KAÇIRILAN SPAMLAR",
    false_negatives
)


# ============================================================
# FALSE POSITIVE
# ============================================================

analyze_group(
    "FALSE POSITIVE - YANLIŞ SPAM TAHMİNLERİ",
    false_positives
)


# ============================================================
# VERY SHORT FALSE NEGATIVES
# ============================================================

fn_features = pd.DataFrame(
    [extract_features(text) for text in false_negatives["text"]]
)

short_fn = false_negatives[
    fn_features["length"] <= 30
]

print("\n" + "=" * 70)
print("ÇOK KISA FALSE NEGATIVE MESAJLAR")
print("=" * 70)

print(
    f"30 karakter veya daha kısa kaçırılan spam: "
    f"{len(short_fn)}"
)

print(
    f"Oran: "
    f"{len(short_fn) / len(false_negatives):.2%}"
)


# ============================================================
# NUMERIC / DIGIT HEAVY FALSE NEGATIVES
# ============================================================

digit_heavy_fn = false_negatives[
    fn_features["digit_ratio"] > 0.20
]

print("\n" + "=" * 70)
print("RAKAM AĞIRLIKLI FALSE NEGATIVE MESAJLAR")
print("=" * 70)

print(
    f"Digit ratio > 20% olan kaçırılan spam: "
    f"{len(digit_heavy_fn)}"
)

print(
    f"Oran: "
    f"{len(digit_heavy_fn) / len(false_negatives):.2%}"
)


# ============================================================
# URL FALSE NEGATIVES
# ============================================================

url_fn = false_negatives[
    fn_features["url_count"] > 0
]

print("\n" + "=" * 70)
print("URL İÇEREN FALSE NEGATIVE MESAJLAR")
print("=" * 70)

print(
    f"URL içeren kaçırılan spam: "
    f"{len(url_fn)}"
)

print(
    f"Oran: "
    f"{len(url_fn) / len(false_negatives):.2%}"
)


# ============================================================
# KEYWORD ANALYSIS
# ============================================================

keywords = [
    "bank",
    "account",
    "acct",
    "debit",
    "credit",
    "airtel",
    "glo",
    "subscribe",
    "subscription",
    "offer",
    "free",
    "call",
    "dial",
    "customer",
    "alert",
    "activate",
    "renew",
    "plan",
    "data",
    "bonus"
]


print("\n" + "=" * 70)
print("FALSE NEGATIVE KEYWORD ANALİZİ")
print("=" * 70)

fn_text = false_negatives["text"].str.lower()

for keyword in keywords:

    count = fn_text.str.contains(
        keyword,
        regex=False
    ).sum()

    if count > 0:

        print(
            f"{keyword:<15} "
            f"{count:>4} "
            f"({count / len(false_negatives):.2%})"
        )


# ============================================================
# FALSE POSITIVE KEYWORD ANALYSIS
# ============================================================

print("\n" + "=" * 70)
print("FALSE POSITIVE KEYWORD ANALİZİ")
print("=" * 70)

fp_text = false_positives["text"].str.lower()

for keyword in keywords:

    count = fp_text.str.contains(
        keyword,
        regex=False
    ).sum()

    if count > 0:

        print(
            f"{keyword:<15} "
            f"{count:>4} "
            f"({count / len(false_positives):.2%})"
        )


print("\n" + "=" * 70)
print("ANALİZ TAMAMLANDI")
print("=" * 70)