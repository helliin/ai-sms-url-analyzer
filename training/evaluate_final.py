
import sys
from pathlib import Path
import re

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)


# ============================================================
# 1. PROJE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


from backend.ai.rule_analyzer_en import analyze_rules_en
from backend.ai.url_analyzer import analyze_url


# ============================================================
# 2. DATASET
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
# 3. AYNI TRAIN / VALIDATION / TEST AYRIMI
# ============================================================

X_temp, X_test, y_temp, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


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


# ============================================================
# 4. ML MODELİNİ TRAIN ET
# ============================================================

vectorizer = TfidfVectorizer()


X_train_tfidf = vectorizer.fit_transform(
    X_train
)


X_test_tfidf = vectorizer.transform(
    X_test
)


model = LinearSVC()


model.fit(
    X_train_tfidf,
    y_train
)


# ============================================================
# 5. ML TEST SKORU
# ============================================================

ml_decision_scores = model.decision_function(
    X_test_tfidf
)


def normalize_ml_score(score):
    """
    SVM decision score değerini 0-100 arasına çevirir.
    """

    import math

    probability = 1 / (
        1 + math.exp(-score)
    )

    return probability * 100


ml_scores_normalized = pd.Series(
    ml_decision_scores,
    index=X_test.index
).apply(
    normalize_ml_score
)


# ============================================================
# 6. RULE ANALYZER
# ============================================================

rule_results = X_test.apply(
    analyze_rules_en
)


rule_scores = rule_results.apply(
    lambda result: result["risk_score"]
)


# ============================================================
# 7. URL ANALYZER
# ============================================================

URL_PATTERN = r"https?://\S+|www\.\S+"


def extract_urls(message):

    return re.findall(
        URL_PATTERN,
        message
    )


def analyze_message_urls(message):

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


        result = analyze_url(
            url
        )


        url_results.append(
            result
        )


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
# 8. FINAL PARAMETRELER
# ============================================================

ML_WEIGHT = 0.90
RULE_WEIGHT = 0.05
URL_WEIGHT = 0.05

THRESHOLD = 40


print("\nFinal parametreler:")

print(
    "ML weight:",
    ML_WEIGHT
)

print(
    "Rule weight:",
    RULE_WEIGHT
)

print(
    "URL weight:",
    URL_WEIGHT
)

print(
    "Threshold:",
    THRESHOLD
)


# ============================================================
# 9. COMBINED SCORE
# ============================================================

combined_scores = (

    ml_scores_normalized * ML_WEIGHT

    + rule_scores * RULE_WEIGHT

    + url_scores * URL_WEIGHT
)


# ============================================================
# 10. FINAL TAHMİNLER
# ============================================================

final_predictions = (
    combined_scores >= THRESHOLD
).map({
    True: "spam",
    False: "ham"
})


# ============================================================
# 11. FINAL SONUÇ TABLOSU
# ============================================================

results = pd.DataFrame({

    "message": X_test,

    "true_label": y_test,

    "ml_score": ml_scores_normalized,

    "rule_score": rule_scores,

    "url_score": url_scores,

    "combined_score": combined_scores,

    "prediction": final_predictions,

    "url_count": url_counts

})


# ============================================================
# 12. PERFORMANS
# ============================================================

accuracy = accuracy_score(
    y_test,
    final_predictions
)


precision = precision_score(
    y_test,
    final_predictions,
    pos_label="spam"
)


recall = recall_score(
    y_test,
    final_predictions,
    pos_label="spam"
)


f1 = f1_score(
    y_test,
    final_predictions,
    pos_label="spam"
)


print("\n========================================")
print("FINAL COMBINED MODEL PERFORMANSI")
print("========================================")


print(
    "\nAccuracy:",
    round(accuracy, 4)
)


print(
    "Spam Precision:",
    round(precision, 4)
)


print(
    "Spam Recall:",
    round(recall, 4)
)


print(
    "Spam F1:",
    round(f1, 4)
)


# ============================================================
# 13. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        final_predictions
    )
)


# ============================================================
# 14. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_test,
    final_predictions,
    labels=["ham", "spam"]
)


print("\nConfusion Matrix:")

print(matrix)


print("\nConfusion Matrix açıklaması:")

print(
    "True Ham → Ham:",
    matrix[0][0]
)

print(
    "True Ham → Spam (False Positive):",
    matrix[0][1]
)

print(
    "True Spam → Ham (False Negative):",
    matrix[1][0]
)

print(
    "True Spam → Spam:",
    matrix[1][1]
)


# ============================================================
# 15. TAHMİN DAĞILIMI
# ============================================================

print("\nFinal tahmin dağılımı:")

print(
    final_predictions.value_counts()
)


# ============================================================
# 16. URL BİLGİSİ
# ============================================================

print("\nTest mesajlarında URL sayısı:")

print(
    (url_counts > 0).sum()
)


# ============================================================
# 17. İLK 20 SONUÇ
# ============================================================

print("\nİlk 20 final sonuç:")

print(
    results[
        [
            "true_label",
            "ml_score",
            "rule_score",
            "url_score",
            "combined_score",
            "prediction"
        ]
    ].head(20)
)

