import sys
from pathlib import Path
import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Proje ana klasörünü Python path'ine ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from backend.ai.rule_analyzer_en import analyze_rules_en
from backend.ai.url_analyzer import analyze_url


# ============================================================
# 1. DATASET
# ============================================================

df = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


print("İlk dataset boyutu:")
print(df.shape)


# Duplicate temizle
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)


print("\nDuplicate temizlendikten sonra:")
print(df.shape)


# Özellik ve hedef
X = df["message"]
y = df["label"]


# Aynı train/test ayrımını kullan
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 2. ML ANALYZER
# ============================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# Tahmin
ml_predictions = model.predict(X_test_tfidf)


# SVM decision score
ml_decision_scores = model.decision_function(X_test_tfidf)


# ============================================================
# 3. RULE ANALYZER
# ============================================================

rule_results = X_test.apply(analyze_rules_en)


rule_scores = rule_results.apply(
    lambda result: result["risk_score"]
)


# ============================================================
# 4. URL ANALYZER
# ============================================================

URL_PATTERN = r"https?://\S+|www\.\S+"


def extract_urls(message):
    """
    SMS içindeki URL'leri bulur.
    """

    return re.findall(
        URL_PATTERN,
        message
    )


def analyze_message_urls(message):
    """
    Mesajdaki URL'leri analiz eder.

    Birden fazla URL varsa
    en yüksek risk skorunu kullanır.
    """

    urls = extract_urls(message)

    if not urls:
        return {
            "url_count": 0,
            "url_score": 0
        }

    url_results = []

    for url in urls:

        url = url.rstrip(
            ".,!?;:)]}"
        )

        result = analyze_url(url)

        url_results.append(result)


    highest_risk = max(
        url_results,
        key=lambda result: result["risk_score"]
    )


    return {
        "url_count": len(urls),
        "url_score": highest_risk["risk_score"]
    }


url_results = X_test.apply(
    analyze_message_urls
)


url_scores = url_results.apply(
    lambda result: result["url_score"]
)


url_counts = url_results.apply(
    lambda result: result["url_count"]
)


# ============================================================
# 5. SONUÇLARI TEK TABLODA BİRLEŞTİR
# ============================================================

results = pd.DataFrame({

    "message": X_test,

    "true_label": y_test,

    "ml_prediction": ml_predictions,

    "ml_decision_score": ml_decision_scores,

    "rule_score": rule_scores,

    "url_score": url_scores,

    "url_count": url_counts

})


# ============================================================
# 6. SONUÇLARI GÖSTER
# ============================================================

print("\nAnaliz sonuçlarının ilk 20 satırı:")

print(
    results.head(20).to_string(
        index=False
    )
)


print("\nML tahmin dağılımı:")

print(
    results["ml_prediction"].value_counts()
)


print("\nRule risk dağılımı:")

print(
    results["rule_score"]
    .value_counts()
    .sort_index()
)


print("\nURL risk dağılımı:")

print(
    results["url_score"]
    .value_counts()
    .sort_index()
)


print("\nURL içeren mesaj sayısı:")

print(
    (results["url_count"] > 0).sum()
)


print("\nToplam test mesajı:")

print(
    len(results)
)