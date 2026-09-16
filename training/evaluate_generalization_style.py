import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack


# ============================================================
# FEATURE EXTRACTION
# ============================================================

def extract_style_features(texts):
    features = []

    for text in texts:
        text = str(text)

        if len(text) == 0:
            uppercase_ratio = 0
            digit_ratio = 0
        else:
            letters = [c for c in text if c.isalpha()]
            digits = [c for c in text if c.isdigit()]

            uppercase_letters = sum(c.isupper() for c in letters)

            if len(letters) > 0:
                uppercase_ratio = uppercase_letters / len(letters)
            else:
                uppercase_ratio = 0

            digit_ratio = len(digits) / len(text)

        features.append([
            uppercase_ratio,
            digit_ratio
        ])

    return np.array(features)


# ============================================================
# DATASET A
# ============================================================

dataset_a_path = "data/sms/SMSSpamCollection"

df_a = pd.read_csv(
    dataset_a_path,
    sep="\t",
    header=None,
    names=["label", "text"]
)

df_a = df_a.drop_duplicates(subset=["text"]).reset_index(drop=True)

df_a["label"] = df_a["label"].map({
    "ham": 0,
    "spam": 1
})

X_a = df_a["text"]
y_a = df_a["label"]


# Train / Test split

X_train, X_test, y_train, y_test = train_test_split(
    X_a,
    y_a,
    test_size=0.20,
    random_state=42,
    stratify=y_a
)


# ============================================================
# TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# STYLE FEATURES
# ============================================================

X_train_style = extract_style_features(X_train)
X_test_style = extract_style_features(X_test)


scaler = StandardScaler()

X_train_style = scaler.fit_transform(X_train_style)
X_test_style = scaler.transform(X_test_style)


# TF-IDF + Style

X_train_combined = hstack([
    X_train_tfidf,
    X_train_style
])

X_test_combined = hstack([
    X_test_tfidf,
    X_test_style
])


# ============================================================
# MODEL
# ============================================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)

model.fit(X_train_combined, y_train)


# ============================================================
# DATASET A → DATASET A
# ============================================================

y_pred_a = model.predict(X_test_combined)

accuracy_a = accuracy_score(y_test, y_pred_a)
precision_a = precision_score(y_test, y_pred_a)
recall_a = recall_score(y_test, y_pred_a)
f1_a = f1_score(y_test, y_pred_a)


# ============================================================
# DATASET B
# ============================================================

dataset_b_path = "data/sms/exais_clean.csv"

df_b = pd.read_csv(dataset_b_path)

X_b = df_b["text"]
y_b = df_b["label"]


# IMPORTANT:
# Dataset B üzerinde fit etmiyoruz.
# Sadece Dataset A'da öğrenilen dönüşümleri kullanıyoruz.

X_b_tfidf = vectorizer.transform(X_b)

X_b_style = extract_style_features(X_b)
X_b_style = scaler.transform(X_b_style)


# TF-IDF + Style

X_b_combined = hstack([
    X_b_tfidf,
    X_b_style
])


# Prediction

y_pred_b = model.predict(X_b_combined)


# ============================================================
# DATASET A → DATASET B
# ============================================================

accuracy_b = accuracy_score(y_b, y_pred_b)
precision_b = precision_score(y_b, y_pred_b)
recall_b = recall_score(y_b, y_pred_b)
f1_b = f1_score(y_b, y_pred_b)


# ============================================================
# RESULTS
# ============================================================

print("=" * 70)
print("GENERALIZATION TEST - TF-IDF + STYLE FEATURES")
print("=" * 70)

print("\nDataset A → Dataset A")
print("-" * 70)

print(f"Accuracy : {accuracy_a:.4f}")
print(f"Precision: {precision_a:.4f}")
print(f"Recall   : {recall_a:.4f}")
print(f"F1 Score : {f1_a:.4f}")


print("\nDataset A → Dataset B")
print("-" * 70)

print(f"Accuracy : {accuracy_b:.4f}")
print(f"Precision: {precision_b:.4f}")
print(f"Recall   : {recall_b:.4f}")
print(f"F1 Score : {f1_b:.4f}")


# ============================================================
# PERFORMANCE DROP
# ============================================================

print("\n" + "=" * 70)
print("PERFORMANCE DROP")
print("=" * 70)

print(f"Accuracy drop : {accuracy_a - accuracy_b:.4f}")
print(f"Precision drop: {precision_a - precision_b:.4f}")
print(f"Recall drop   : {recall_a - recall_b:.4f}")
print(f"F1 drop       : {f1_a - f1_b:.4f}")


# ============================================================
# COMPARISON WITH BASELINE
# ============================================================

print("\n" + "=" * 70)
print("BASELINE vs STYLE FEATURES - DATASET B")
print("=" * 70)

print("\nBaseline Dataset A → B:")
print("Accuracy : 0.6833")
print("Precision: 0.6647")
print("Recall   : 0.3405")
print("F1 Score : 0.4503")

print("\nStyle Features Dataset A → B:")
print(f"Accuracy : {accuracy_b:.4f}")
print(f"Precision: {precision_b:.4f}")
print(f"Recall   : {recall_b:.4f}")
print(f"F1 Score : {f1_b:.4f}")