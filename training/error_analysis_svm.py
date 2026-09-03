import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)
# 1. Dataseti oku
file_path = "data/sms/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)
# 2. Tekrarlanan mesajları temizle
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

# 3. Etiketleri sayısal hale getir
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})
# 4. Özellik ve hedef değişken
X = df["message"]
y = df["label"]


# 5. Train / Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)
# 6. TF-IDF
vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# 7. Linear SVM modeli
model = LinearSVC(
    class_weight="balanced",
    random_state=42
)


# 8. Modeli eğit
model.fit(X_train_tfidf, y_train)


# 9. Test verisi üzerinde tahmin
y_pred = model.predict(X_test_tfidf)

# 10. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

tn, fp, fn, tp = cm.ravel()

print("\n=== Confusion Matrix ===")
print(cm)

print("\n=== Hata Analizi ===")
print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")

# 11. Test sonuçlarını tablo haline getir
results = pd.DataFrame({
    "message": X_test.values,
    "actual": y_test.values,
    "predicted": y_pred
})

# Etiketleri tekrar okunabilir hale getir
results["actual"] = results["actual"].map({
    0: "ham",
    1: "spam"
})

results["predicted"] = results["predicted"].map({
    0: "ham",
    1: "spam"
})
# 12. False Positive
false_positives = results[
    (results["actual"] == "ham") &
    (results["predicted"] == "spam")
]

print("\n=== FALSE POSITIVE ===")
print(f"Toplam False Positive: {len(false_positives)}")

for _, row in false_positives.iterrows():
    print(f"\nMesaj: {row['message']}")
    # 13. False Negative
false_negatives = results[
    (results["actual"] == "spam") &
    (results["predicted"] == "ham")
]

print("\n=== FALSE NEGATIVE ===")
print(f"Toplam False Negative: {len(false_negatives)}")

for _, row in false_negatives.iterrows():
    print(f"\nMesaj: {row['message']}")
    # 14. Classification Report
print("\n=== Classification Report ===")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)
# 15. Hatalı tahminleri ayrı bir dosyaya kaydet

error_results = results[
    results["actual"] != results["predicted"]
].copy()

error_results["error_type"] = error_results.apply(
    lambda row: (
        "False Positive"
        if row["actual"] == "ham" and row["predicted"] == "spam"
        else "False Negative"
    ),
    axis=1
)

error_results = error_results[
    ["error_type", "actual", "predicted", "message"]
]

error_results.to_csv(
    "training/svm_error_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nHatalı tahminler kaydedildi:")
print("training/svm_error_analysis.csv")
# 16. Error Analysis özeti

total_samples = len(y_test)

false_positive_rate = fp / (fp + tn)
false_negative_rate = fn / (fn + tp)

print("\n=== ERROR ANALYSIS SUMMARY ===")
print(f"Test samples: {total_samples}")
print(f"False Positive: {fp}")
print(f"False Negative: {fn}")
print(f"False Positive Rate: {false_positive_rate:.4f}")
print(f"False Negative Rate: {false_negative_rate:.4f}")