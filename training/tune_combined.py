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


X = df["message"]
y = df["label"]


# ============================================================
# 2. TEST SETİNİ AYIR
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 3. TRAIN / VALIDATION AYIR
# ============================================================

X_train, X_validation, y_train, y_validation = train_test_split(
    X_temp,
    y_temp,
    test_size=0.20,
    random_state=42,
    stratify=y_temp
)


print("\nTrain boyutu:")
print(X_train.shape)

print("\nValidation boyutu:")
print(X_validation.shape)

print("\nTest boyutu:")
print(X_test.shape)


print("\nTrain etiket dağılımı:")
print(y_train.value_counts())


print("\nValidation etiket dağılımı:")
print(y_validation.value_counts())


print("\nTest etiket dağılımı:")
print(y_test.value_counts())


# ============================================================
# 4. ML ANALYZER
# ============================================================

vectorizer = TfidfVectorizer()

X_train_tfidf = vectorizer.fit_transform(X_train)

X_validation_tfidf = vectorizer.transform(
    X_validation
)


model = LinearSVC()

model.fit(
    X_train_tfidf,
    y_train
)


# Validation için SVM decision score
ml_decision_scores = model.decision_function(
    X_validation_tfidf
)


# ============================================================
# 5. RULE ANALYZER
# ============================================================

rule_results = X_validation.apply(
    analyze_rules_en
)


rule_scores = rule_results.apply(
    lambda result: result["risk_score"]
)


# ============================================================
# 6. URL ANALYZER
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

        # URL sonunda bulunan noktalama işaretlerini temizle
        url = url.rstrip(
            ".,!?;:)]}"
        )

        result = analyze_url(url)

        url_results.append(result)


    # En yüksek riskli URL'yi bul
    highest_risk = max(
        url_results,
        key=lambda result: result["risk_score"]
    )


    return {
        "url_count": len(urls),
        "url_score": highest_risk["risk_score"]
    }


url_results = X_validation.apply(
    analyze_message_urls
)


url_scores = url_results.apply(
    lambda result: result["url_score"]
)


url_counts = url_results.apply(
    lambda result: result["url_count"]
)


# ============================================================
# 7. SONUÇLARI GÖSTER
# ============================================================

print("\nValidation ML skorlarından ilk 10:")

print(
    ml_decision_scores[:10]
)


print("\nValidation Rule skor dağılımı:")

print(
    rule_scores
    .value_counts()
    .sort_index()
)


print("\nValidation URL skor dağılımı:")

print(
    url_scores
    .value_counts()
    .sort_index()
)


print("\nValidation URL içeren mesaj sayısı:")

print(
    (url_counts > 0).sum()
)
# ============================================================
# 8. SKORLARI 0-100 ARASINA NORMALIZE ET
# ============================================================

def normalize_ml_score(score):
    """
    SVM decision score değerini 0-100 arasına çevirir.
    """

    # Basit sigmoid dönüşümü
    probability = 1 / (1 + __import__("math").exp(-score))

    return probability * 100


ml_scores_normalized = pd.Series(
    ml_decision_scores,
    index=X_validation.index
).apply(
    normalize_ml_score
)


# ============================================================
# 9. VALIDATION SONUÇ TABLOSU
# ============================================================

validation_results = pd.DataFrame({

    "message": X_validation,

    "true_label": y_validation,

    "ml_score": ml_scores_normalized,

    "rule_score": rule_scores,

    "url_score": url_scores

})


print("\nNormalize edilmiş ML skorlarından ilk 10:")

print(
    validation_results[
        ["ml_score", "rule_score", "url_score"]
    ].head(10)
)


# ============================================================
# 10. AĞIRLIK KOMBİNASYONLARINI DENE
# ============================================================

from sklearn.metrics import f1_score


weight_combinations = [

    (0.9, 0.05, 0.05),
    (0.8, 0.1, 0.1),
    (0.8, 0.15, 0.05),
    (0.8, 0.05, 0.15),

    (0.7, 0.2, 0.1),
    (0.7, 0.1, 0.2),
    (0.7, 0.15, 0.15),

    (0.6, 0.3, 0.1),
    (0.6, 0.2, 0.2),
    (0.6, 0.1, 0.3),

    (0.5, 0.3, 0.2),
    (0.5, 0.2, 0.3),
    (0.5, 0.4, 0.1),

]


tuning_results = []


for ml_weight, rule_weight, url_weight in weight_combinations:

    combined_score = (
        validation_results["ml_score"] * ml_weight
        + validation_results["rule_score"] * rule_weight
        + validation_results["url_score"] * url_weight
    )


    # İlk threshold
    predictions = (
        combined_score >= 50
    ).map({
        True: "spam",
        False: "ham"
    })


    f1 = f1_score(
        validation_results["true_label"],
        predictions,
        pos_label="spam"
    )


    tuning_results.append({

        "ml_weight": ml_weight,

        "rule_weight": rule_weight,

        "url_weight": url_weight,

        "f1": f1

    })


# ============================================================
# 11. SONUÇLARI SIRALA
# ============================================================

tuning_df = pd.DataFrame(
    tuning_results
)


tuning_df = tuning_df.sort_values(
    by="f1",
    ascending=False
)


print("\nAğırlık tuning sonuçları:")

print(
    tuning_df.to_string(
        index=False
    )
)
# ============================================================
# 12. EN İYİ AĞIRLIĞI AL
# ============================================================

best_row = tuning_df.iloc[0]

best_ml_weight = best_row["ml_weight"]
best_rule_weight = best_row["rule_weight"]
best_url_weight = best_row["url_weight"]


print("\nEn iyi ağırlıklar:")

print(
    "ML:",
    best_ml_weight
)

print(
    "Rule:",
    best_rule_weight
)

print(
    "URL:",
    best_url_weight
)


# ============================================================
# 13. EN İYİ AĞIRLIKLA COMBINED SCORE
# ============================================================

validation_results["combined_score"] = (

    validation_results["ml_score"] * best_ml_weight

    + validation_results["rule_score"] * best_rule_weight

    + validation_results["url_score"] * best_url_weight

)


# ============================================================
# 14. THRESHOLD TUNING
# ============================================================

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


threshold_results = []


thresholds = [
    30,
    35,
    40,
    45,
    50,
    55,
    60,
    65,
    70
]


for threshold in thresholds:

    predictions = (
        validation_results["combined_score"] >= threshold
    ).map({
        True: "spam",
        False: "ham"
    })


    precision = precision_score(
        validation_results["true_label"],
        predictions,
        pos_label="spam"
    )


    recall = recall_score(
        validation_results["true_label"],
        predictions,
        pos_label="spam"
    )


    f1 = f1_score(
        validation_results["true_label"],
        predictions,
        pos_label="spam"
    )


    matrix = confusion_matrix(
        validation_results["true_label"],
        predictions,
        labels=["ham", "spam"]
    )


    false_positive = matrix[0][1]
    false_negative = matrix[1][0]


    threshold_results.append({

        "threshold": threshold,

        "precision": precision,

        "recall": recall,

        "f1": f1,

        "false_positive": false_positive,

        "false_negative": false_negative

    })


# ============================================================
# 15. THRESHOLD SONUÇLARI
# ============================================================

threshold_df = pd.DataFrame(
    threshold_results
)


print("\nThreshold tuning sonuçları:")

print(
    threshold_df.to_string(
        index=False
    )
)


# ============================================================
# 16. EN İYİ THRESHOLD
# ============================================================

best_threshold_row = threshold_df.loc[
    threshold_df["f1"].idxmax()
]


print("\nEn iyi threshold:")

print(
    best_threshold_row.to_string()
)