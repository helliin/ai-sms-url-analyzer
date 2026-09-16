import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)
from scipy.sparse import hstack, csr_matrix


# ============================================================
# STYLE FEATURE FUNCTION
# ============================================================

def extract_style_features(texts):
    features = []

    for text in texts:
        text = str(text)

        length = len(text)

        if length > 0:
            uppercase_ratio = sum(
                1 for c in text if c.isupper()
            ) / length

            digit_ratio = sum(
                1 for c in text if c.isdigit()
            ) / length
        else:
            uppercase_ratio = 0
            digit_ratio = 0

        features.append([
            uppercase_ratio,
            digit_ratio
        ])

    return np.array(features)


# ============================================================
# DATASET A
# ============================================================

print("=" * 70)
print("DATASET A")
print("=" * 70)

df_a = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

df_a = df_a.drop_duplicates()

df_a["label"] = df_a["label"].map({
    "ham": 0,
    "spam": 1
})

print(f"Toplam mesaj: {len(df_a)}")
print(f"HAM: {(df_a['label'] == 0).sum()}")
print(f"SPAM: {(df_a['label'] == 1).sum()}")


# ============================================================
# DATASET B
# ============================================================

print()
print("=" * 70)
print("DATASET B")
print("=" * 70)

df_b = pd.read_csv(
    "data/sms/exais_clean.csv"
)

print(f"Toplam mesaj: {len(df_b)}")
print(f"HAM: {(df_b['label'] == 0).sum()}")
print(f"SPAM: {(df_b['label'] == 1).sum()}")


# ============================================================
# DATASET B SPLIT
# ============================================================

b_train, b_test = train_test_split(
    df_b,
    test_size=0.20,
    random_state=42,
    stratify=df_b["label"]
)

print()
print("=" * 70)
print("DATASET B SPLIT")
print("=" * 70)

print(f"B Train: {len(b_train)}")
print(f"B Test : {len(b_test)}")


# ============================================================
# COMBINE TRAINING DATA
# ============================================================

combined_train = pd.concat(
    [df_a, b_train],
    ignore_index=True
)

print()
print("=" * 70)
print("COMBINED TRAINING DATA")
print("=" * 70)

print(f"Toplam training mesajı: {len(combined_train)}")
print(f"HAM: {(combined_train['label'] == 0).sum()}")
print(f"SPAM: {(combined_train['label'] == 1).sum()}")


# ============================================================
# TF-IDF
# ============================================================

print()
print("=" * 70)
print("TF-IDF + STYLE FEATURES")
print("=" * 70)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2)
)

X_tfidf_train = vectorizer.fit_transform(
    combined_train["text"]
)

X_tfidf_test = vectorizer.transform(
    b_test["text"]
)


# ============================================================
# STYLE FEATURES
# ============================================================

X_style_train = extract_style_features(
    combined_train["text"]
)

X_style_test = extract_style_features(
    b_test["text"]
)


# ============================================================
# SCALE STYLE FEATURES
# ============================================================

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_style_train = scaler.fit_transform(
    X_style_train
)

X_style_test = scaler.transform(
    X_style_test
)


# Sparse matrix'e çevir
X_style_train = csr_matrix(X_style_train)
X_style_test = csr_matrix(X_style_test)


# ============================================================
# COMBINE TF-IDF + STYLE
# ============================================================

X_train = hstack([
    X_tfidf_train,
    X_style_train
])

X_test = hstack([
    X_tfidf_test,
    X_style_test
])

print(f"TF-IDF feature sayısı: {X_tfidf_train.shape[1]}")
print("Style feature sayısı: 2")
print(f"Toplam feature sayısı: {X_train.shape[1]}")


# ============================================================
# TRAIN MODEL
# ============================================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(
    X_train,
    combined_train["label"]
)


# ============================================================
# TEST
# ============================================================

y_pred = model.predict(X_test)
y_test = b_test["label"]


# ============================================================
# RESULTS
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)


print()
print("=" * 70)
print("COMBINED DATASET + STYLE FEATURES RESULT")
print("=" * 70)

print()
print("Dataset A + Dataset B Train")
print("          ↓")
print(" TF-IDF + Style Features")
print("          ↓")
print("   LinearSVC")
print("          ↓")
print(" Unseen Dataset B Test")
print()

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["HAM", "SPAM"],
        zero_division=0
    )
)