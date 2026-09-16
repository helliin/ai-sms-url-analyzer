import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


# ============================================================
# 1. DATASET A - SMSSpamCollection
# ============================================================

dataset_a_path = "data/sms/SMSSpamCollection"

df_a = pd.read_csv(
    dataset_a_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Duplicate temizliği
df_a = df_a.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

# Label dönüşümü
df_a["label"] = df_a["label"].map({
    "ham": 0,
    "spam": 1
})

print("=" * 70)
print("DATASET A - SMSSpamCollection")
print("=" * 70)

print(f"Toplam mesaj: {len(df_a)}")
print("\nLabel dağılımı:")
print(df_a["label"].value_counts())


# ============================================================
# 2. DATASET A -> TRAIN / TEST
# ============================================================

X_a = df_a["message"]
y_a = df_a["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X_a,
    y_a,
    test_size=0.20,
    random_state=42,
    stratify=y_a
)


# ============================================================
# 3. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

# ÖNEMLİ:
# TF-IDF sadece TRAIN üzerinde fit ediliyor.
X_train_tfidf = vectorizer.fit_transform(X_train)

# Dataset A test
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 4. LINEAR SVM
# ============================================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# 5. DATASET A -> DATASET A TEST
# ============================================================

y_pred_a = model.predict(X_test_tfidf)

accuracy_a = accuracy_score(y_test, y_pred_a)
precision_a = precision_score(y_test, y_pred_a)
recall_a = recall_score(y_test, y_pred_a)
f1_a = f1_score(y_test, y_pred_a)

print("\n")
print("=" * 70)
print("TEST 1: DATASET A -> DATASET A")
print("=" * 70)

print(f"Accuracy : {accuracy_a:.4f}")
print(f"Precision: {precision_a:.4f}")
print(f"Recall   : {recall_a:.4f}")
print(f"F1       : {f1_a:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred_a,
        target_names=["ham", "spam"]
    )
)


# ============================================================
# 6. DATASET B - ExAIS
# ============================================================

dataset_b_path = "data/sms/exais_clean.csv"

df_b = pd.read_csv(dataset_b_path)

print("=" * 70)
print("DATASET B - ExAIS")
print("=" * 70)

print(f"Toplam mesaj: {len(df_b)}")

print("\nLabel dağılımı:")
print(df_b["label"].value_counts())


# ============================================================
# 7. DATASET B HAZIRLAMA
# ============================================================

X_b = df_b["text"]
y_b = df_b["label"]


# ============================================================
# 8. DATASET B'Yİ TF-IDF'E DÖNÜŞTÜR
# ============================================================

# DİKKAT:
#
# Burada fit_transform KULLANMIYORUZ.
#
# Çünkü TF-IDF Dataset A'nın eğitim verisinden öğrenildi.
#
# Dataset B sadece transform ediliyor.

X_b_tfidf = vectorizer.transform(X_b)


# ============================================================
# 9. DATASET A -> DATASET B TEST
# ============================================================

y_pred_b = model.predict(X_b_tfidf)

accuracy_b = accuracy_score(y_b, y_pred_b)
precision_b = precision_score(y_b, y_pred_b)
recall_b = recall_score(y_b, y_pred_b)
f1_b = f1_score(y_b, y_pred_b)

print("\n")
print("=" * 70)
print("TEST 2: DATASET A -> DATASET B")
print("=" * 70)

print(f"Accuracy : {accuracy_b:.4f}")
print(f"Precision: {precision_b:.4f}")
print(f"Recall   : {recall_b:.4f}")
print(f"F1       : {f1_b:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_b,
        y_pred_b,
        target_names=["ham", "spam"]
    )
)


# ============================================================
# 10. KARŞILAŞTIRMA
# ============================================================

print("\n")
print("=" * 70)
print("GENERALIZATION KARŞILAŞTIRMASI")
print("=" * 70)

print(
    f"{'Test':<25}"
    f"{'Accuracy':<12}"
    f"{'Precision':<12}"
    f"{'Recall':<12}"
    f"{'F1':<12}"
)

print("-" * 70)

print(
    f"{'Dataset A -> A':<25}"
    f"{accuracy_a:<12.4f}"
    f"{precision_a:<12.4f}"
    f"{recall_a:<12.4f}"
    f"{f1_a:<12.4f}"
)

print(
    f"{'Dataset A -> B':<25}"
    f"{accuracy_b:<12.4f}"
    f"{precision_b:<12.4f}"
    f"{recall_b:<12.4f}"
    f"{f1_b:<12.4f}"
)


# ============================================================
# 11. PERFORMANS DEĞİŞİMİ
# ============================================================

print("\n")
print("=" * 70)
print("PERFORMANS DEĞİŞİMİ")
print("=" * 70)

print(f"Accuracy değişimi : {accuracy_b - accuracy_a:+.4f}")
print(f"Precision değişimi: {precision_b - precision_a:+.4f}")
print(f"Recall değişimi   : {recall_b - recall_a:+.4f}")
print(f"F1 değişimi       : {f1_b - f1_a:+.4f}")