import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PROJE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


from backend.ai.llm_analyzer import analyze_with_llm


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
# 3. TRAIN / TEST AYRIMI
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTrain boyutu:")
print(X_train.shape)

print("\nTest boyutu:")
print(X_test.shape)


# ============================================================
# 4. LLM TESTİ
# ============================================================

print("\nLLM testine başlanıyor...")


llm_predictions = []
llm_scores = []
llm_confidences = []


for i, message in enumerate(X_test):

    print(
        f"LLM analiz ediyor: {i + 1}/{len(X_test)}"
    )

    result = analyze_with_llm(message)

    prediction = result["prediction"]

    # Beklenmeyen cevapları ham olarak kabul etme
    if prediction not in ["spam", "ham"]:
        prediction = "unknown"

    llm_predictions.append(prediction)

    llm_scores.append(
        result["risk_score"]
    )

    llm_confidences.append(
        result["confidence"]
    )


# ============================================================
# 5. SONUÇLARI DATAFRAME'E AL
# ============================================================

results = pd.DataFrame({

    "message": X_test.values,

    "true_label": y_test.values,

    "llm_prediction": llm_predictions,

    "llm_score": llm_scores,

    "llm_confidence": llm_confidences

})


# ============================================================
# 6. UNKNOWN SONUÇLARI KONTROL ET
# ============================================================

unknown_count = (
    results["llm_prediction"] == "unknown"
).sum()


print("\nUnknown tahmin sayısı:")
print(unknown_count)


# ============================================================
# 7. SADECE GEÇERLİ TAHMİNLERİ DEĞERLENDİR
# ============================================================

valid_results = results[
    results["llm_prediction"].isin(
        ["ham", "spam"]
    )
]


y_true = valid_results["true_label"]
y_pred = valid_results["llm_prediction"]


# ============================================================
# 8. PERFORMANS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)


precision = precision_score(
    y_true,
    y_pred,
    pos_label="spam"
)


recall = recall_score(
    y_true,
    y_pred,
    pos_label="spam"
)


f1 = f1_score(
    y_true,
    y_pred,
    pos_label="spam"
)


print("\n========================================")
print("LLM MODEL PERFORMANSI")
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
# 9. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred
    )
)


# ============================================================
# 10. CONFUSION MATRIX
# ============================================================

matrix = confusion_matrix(
    y_true,
    y_pred,
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
# 11. TAHMİN DAĞILIMI
# ============================================================

print("\nLLM tahmin dağılımı:")

print(
    results["llm_prediction"].value_counts()
)


# ============================================================
# 12. ORTALAMA LLM SKORU
# ============================================================

print("\nOrtalama LLM risk skoru:")

print(
    round(
        results["llm_score"].mean(),
        2
    )
)


# ============================================================
# 13. İLK 20 SONUÇ
# ============================================================

print("\nİlk 20 LLM sonucu:")

print(
    results[
        [
            "true_label",
            "llm_prediction",
            "llm_score",
            "llm_confidence"
        ]
    ].head(20)
)