import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


# ============================================================
# 1. DATASET A - TRAIN
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


X_train, X_test, y_train, y_test = train_test_split(
    X_a,
    y_a,
    test_size=0.20,
    random_state=42,
    stratify=y_a
)


# ============================================================
# 2. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)

model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

model.fit(X_train_tfidf, y_train)


# ============================================================
# 3. DATASET B
# ============================================================

dataset_b_path = "data/sms/exais_clean.csv"

df_b = pd.read_csv(dataset_b_path)

X_b = df_b["text"]
y_b = df_b["label"]


# Dataset B'de TF-IDF'i yeniden fit etmiyoruz.
X_b_tfidf = vectorizer.transform(X_b)


# ============================================================
# 4. PREDICTION
# ============================================================

y_pred_b = model.predict(X_b_tfidf)


# ============================================================
# 5. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(y_b, y_pred_b)

tn, fp, fn, tp = cm.ravel()

print("=" * 70)
print("DATASET B - GENERALIZATION ERROR ANALYSIS")
print("=" * 70)

print("\nConfusion Matrix:")
print(cm)

print(f"\nTrue Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")


# ============================================================
# 6. ERROR COUNTS
# ============================================================

print("\n")
print("=" * 70)
print("HATA ORANLARI")
print("=" * 70)

print(f"Toplam mesaj: {len(df_b)}")
print(f"Yanlış tahmin: {fp + fn}")
print(f"Doğru tahmin : {tn + tp}")

print(f"\nFalse Positive: {fp}")
print(f"False Negative: {fn}")


# ============================================================
# 7. FALSE NEGATIVE ÖRNEKLERİ
# ============================================================

false_negatives = df_b[
    (y_b == 1) & (y_pred_b == 0)
].copy()

print("\n")
print("=" * 70)
print("FALSE NEGATIVE ÖRNEKLERİ")
print("=" * 70)

print(
    f"Toplam kaçırılan spam: {len(false_negatives)}"
)

for i, row in false_negatives.head(20).iterrows():
    print("\n---")
    print(f"Index: {i}")
    print(f"Gerçek label: SPAM")
    print(f"Model tahmini: HAM")
    print(f"Mesaj: {row['text']}")


# ============================================================
# 8. FALSE POSITIVE ÖRNEKLERİ
# ============================================================

false_positives = df_b[
    (y_b == 0) & (y_pred_b == 1)
].copy()

print("\n")
print("=" * 70)
print("FALSE POSITIVE ÖRNEKLERİ")
print("=" * 70)

print(
    f"Toplam yanlış spam tahmini: {len(false_positives)}"
)

for i, row in false_positives.head(20).iterrows():
    print("\n---")
    print(f"Index: {i}")
    print(f"Gerçek label: HAM")
    print(f"Model tahmini: SPAM")
    print(f"Mesaj: {row['text']}")


# ============================================================
# 9. HATA ÖRNEKLERİNİ DOSYAYA KAYDET
# ============================================================

false_negatives[["label", "text"]].to_csv(
    "data/sms/exais_false_negatives.csv",
    index=False
)

false_positives[["label", "text"]].to_csv(
    "data/sms/exais_false_positives.csv",
    index=False
)

print("\n")
print("=" * 70)
print("HATA DOSYALARI")
print("=" * 70)

print("False negative dosyası:")
print("data/sms/exais_false_negatives.csv")

print("\nFalse positive dosyası:")
print("data/sms/exais_false_positives.csv")