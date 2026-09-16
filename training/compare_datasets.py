import pandas as pd
import re


# ============================================================
# LOAD DATASETS
# ============================================================

dataset_a_path = "data/sms/SMSSpamCollection"
dataset_b_path = "data/sms/exais_clean.csv"


df_a = pd.read_csv(
    dataset_a_path,
    sep="\t",
    header=None,
    names=["label", "text"]
)

df_a = df_a.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

df_a["label"] = df_a["label"].map({
    "ham": 0,
    "spam": 1
})


df_b = pd.read_csv(dataset_b_path)

# Dataset B zaten 0 = HAM, 1 = SPAM


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_features(text):

    text = str(text)

    length = len(text)

    words = len(text.split())

    letters = sum(
        1 for c in text
        if c.isalpha()
    )

    uppercase = sum(
        1 for c in text
        if c.isalpha() and c.isupper()
    )

    digits = sum(
        1 for c in text
        if c.isdigit()
    )

    uppercase_ratio = (
        uppercase / letters
        if letters > 0 else 0
    )

    digit_ratio = (
        digits / length
        if length > 0 else 0
    )

    urls = len(
        re.findall(
            r"https?://\S+|www\.\S+",
            text
        )
    )

    return {
        "length": length,
        "words": words,
        "uppercase_ratio": uppercase_ratio,
        "digit_ratio": digit_ratio,
        "url_count": urls
    }


# ============================================================
# DATASET ANALYSIS
# ============================================================

def analyze_dataset(name, df):

    features = pd.DataFrame(
        [
            extract_features(text)
            for text in df["text"]
        ]
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"\nToplam mesaj: {len(df)}")

    spam_count = (df["label"] == 1).sum()
    ham_count = (df["label"] == 0).sum()

    print(f"HAM : {ham_count}")
    print(f"SPAM: {spam_count}")

    print(
        f"SPAM oranı: "
        f"{spam_count / len(df):.2%}"
    )

    print("\nMesaj özellikleri:")

    print(
        f"Ortalama uzunluk: "
        f"{features['length'].mean():.2f}"
    )

    print(
        f"Ortalama kelime sayısı: "
        f"{features['words'].mean():.2f}"
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
        f"{(features['url_count'] > 0).mean():.2%}"
    )

    print(
        f"10 karakterden kısa/eşit: "
        f"{(features['length'] <= 10).sum()}"
    )

    print(
        f"30 karakterden kısa/eşit: "
        f"{(features['length'] <= 30).sum()}"
    )

    print(
        f"100 karakterden uzun: "
        f"{(features['length'] > 100).sum()}"
    )


# ============================================================
# RUN
# ============================================================

analyze_dataset(
    "DATASET A - SMSSpamCollection",
    df_a
)

analyze_dataset(
    "DATASET B - ExAIS",
    df_b
)


# ============================================================
# COMPARISON
# ============================================================

print("\n" + "=" * 70)
print("DATASET COMPARISON")
print("=" * 70)

spam_a = (df_a["label"] == 1).mean()
spam_b = (df_b["label"] == 1).mean()

print(
    f"\nSpam oranı farkı: "
    f"{abs(spam_a - spam_b):.2%}"
)

print(
    f"Dataset A spam oranı: "
    f"{spam_a:.2%}"
)

print(
    f"Dataset B spam oranı: "
    f"{spam_b:.2%}"
)