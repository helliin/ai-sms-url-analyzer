import pandas as pd
import re

INPUT_FILE = "training/error_analysis_results.csv"
OUTPUT_FILE = "training/error_analysis_categorized.csv"

df = pd.read_csv(INPUT_FILE)

# --------------------------------------------------
# TEMEL ÖZELLİKLER
# --------------------------------------------------

df["text"] = df["text"].fillna("").astype(str)
df["text_lower"] = df["text"].str.lower()

df["text_length"] = df["text"].str.len()
df["word_count"] = df["text"].str.split().str.len()

# --------------------------------------------------
# MESAJ KATEGORİLERİ
# --------------------------------------------------

def contains_any(text, keywords):
    return any(keyword in text for keyword in keywords)


df["has_url"] = df["text_lower"].apply(
    lambda x: bool(re.search(r"https?://|www\.", x))
)

df["has_phone_or_shortcode"] = df["text_lower"].apply(
    lambda x: bool(re.search(r"\b\d{4,}\b|\*\d+#", x))
)

df["has_money"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "£", "$", "€", "money", "cash", "price",
            "cost", "payment", "pay", "paid", "credit"
        ]
    )
)

df["has_promotion"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "offer", "sale", "discount", "free",
            "win", "winner", "prize", "claim",
            "promotion", "promo", "bonus"
        ]
    )
)

df["has_urgency"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "urgent", "immediately", "now", "hurry",
            "asap", "important", "expires",
            "limited", "act now"
        ]
    )
)

df["has_financial"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "bank", "account", "loan", "credit",
            "card", "balance", "transaction",
            "cash", "payment"
        ]
    )
)

df["has_subscription"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "subscribe", "subscription", "unsubscribe",
            "opted out", "campaign", "daily",
            "per day", "/day"
        ]
    )
)

df["has_reward"] = df["text_lower"].apply(
    lambda x: contains_any(
        x,
        [
            "congratulations", "congrats", "winner",
            "won", "prize", "reward", "gift"
        ]
    )
)

df["has_question"] = df["text"].apply(lambda x: "?" in x)
df["has_exclamation"] = df["text"].apply(lambda x: "!" in x)

# --------------------------------------------------
# MESAJ UZUNLUĞU
# --------------------------------------------------

def length_category(length):
    if length <= 30:
        return "Very Short"
    elif length <= 80:
        return "Short"
    elif length <= 160:
        return "Medium"
    else:
        return "Long"


df["length_category"] = df["text_length"].apply(length_category)

# --------------------------------------------------
# MODEL SONUÇLARI
# --------------------------------------------------

def error_type(row):

    svm_wrong = row["svm_prediction"] != row["true_label"]
    bert_wrong = row["distilbert_prediction"] != row["true_label"]

    if not svm_wrong and not bert_wrong:
        return "Both Correct"

    if svm_wrong and not bert_wrong:
        return "SVM Wrong Only"

    if not svm_wrong and bert_wrong:
        return "DistilBERT Wrong Only"

    return "Both Wrong"


df["error_type"] = df.apply(error_type, axis=1)

# --------------------------------------------------
# KATEGORİ ANALİZİ
# --------------------------------------------------

categories = [
    "has_url",
    "has_phone_or_shortcode",
    "has_money",
    "has_promotion",
    "has_urgency",
    "has_financial",
    "has_subscription",
    "has_reward",
    "has_question",
    "has_exclamation",
]

print("=" * 70)
print("ERROR ANALYSIS - CATEGORY ANALYSIS")
print("=" * 70)

print(f"\nToplam test mesajı: {len(df)}")

print("\nMODEL HATA TİPLERİ")
print("-" * 70)

print(df["error_type"].value_counts())

print("\nKATEGORİ BAZLI ANALİZ")
print("-" * 70)

results = []

for category in categories:

    subset = df[df[category] == True]

    if len(subset) == 0:
        continue

    svm_errors = (
        subset["svm_prediction"] != subset["true_label"]
    ).sum()

    bert_errors = (
        subset["distilbert_prediction"] != subset["true_label"]
    ).sum()

    results.append({
        "category": category,
        "message_count": len(subset),
        "svm_errors": svm_errors,
        "distilbert_errors": bert_errors,
        "svm_error_rate": round(svm_errors / len(subset), 4),
        "distilbert_error_rate": round(bert_errors / len(subset), 4),
    })

    print(f"\n{category}")
    print(f"  Mesaj sayısı       : {len(subset)}")
    print(f"  SVM hata           : {svm_errors}")
    print(f"  DistilBERT hata    : {bert_errors}")
    print(f"  SVM hata oranı     : {svm_errors / len(subset):.2%}")
    print(f"  DistilBERT hata oranı: {bert_errors / len(subset):.2%}")


# --------------------------------------------------
# UZUNLUK ANALİZİ
# --------------------------------------------------

print("\n" + "=" * 70)
print("MESAJ UZUNLUĞUNA GÖRE HATA ANALİZİ")
print("=" * 70)

length_results = []

for category in [
    "Very Short",
    "Short",
    "Medium",
    "Long"
]:

    subset = df[df["length_category"] == category]

    if len(subset) == 0:
        continue

    svm_errors = (
        subset["svm_prediction"] != subset["true_label"]
    ).sum()

    bert_errors = (
        subset["distilbert_prediction"] != subset["true_label"]
    ).sum()

    length_results.append({
        "length_category": category,
        "message_count": len(subset),
        "svm_errors": svm_errors,
        "distilbert_errors": bert_errors,
        "svm_error_rate": round(svm_errors / len(subset), 4),
        "distilbert_error_rate": round(bert_errors / len(subset), 4),
    })

    print(f"\n{category}")
    print(f"  Mesaj sayısı       : {len(subset)}")
    print(f"  SVM hata oranı     : {svm_errors / len(subset):.2%}")
    print(f"  DistilBERT hata oranı: {bert_errors / len(subset):.2%}")


# --------------------------------------------------
# SONUÇLARI KAYDET
# --------------------------------------------------

summary_df = pd.DataFrame(results)
length_df = pd.DataFrame(length_results)

summary_df.to_csv(
    "training/error_analysis_category_summary.csv",
    index=False
)

length_df.to_csv(
    "training/error_analysis_length_summary.csv",
    index=False
)

# Kategorize edilmiş tüm veriyi kaydet
df.drop(columns=["text_lower"]).to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 70)
print("DOSYALAR OLUŞTURULDU")
print("=" * 70)

print(f"\n{OUTPUT_FILE}")
print("training/error_analysis_category_summary.csv")
print("training/error_analysis_length_summary.csv")