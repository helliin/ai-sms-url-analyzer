import re
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
)


# ============================================================
# 1. Dataseti oku
# ============================================================

file_path = "data/sms/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)


# ============================================================
# 2. Tekrarlanan mesajları temizle
# ============================================================

df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)


# ============================================================
# 3. Etiketleri sayısal hale getir
# ============================================================

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 4. Feature Engineering fonksiyonu
# ============================================================

def extract_features(text):
    """
    SMS metninden sayısal özellikler çıkarır.
    """

    text = str(text)

    # Kelimeler
    words = text.split()

    # URL'ler
    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text.lower()
    )

    # Büyük harfler
    uppercase_chars = sum(
        1 for char in text if char.isupper()
    )

    # Rakamlar
    digit_chars = sum(
        1 for char in text if char.isdigit()
    )

    # Sıfıra bölme problemini önlemek için
    text_length = max(len(text), 1)

    features = {
        "text_length": len(text),

        "word_count": len(words),

        "uppercase_ratio": (
            uppercase_chars / text_length
        ),

        "digit_ratio": (
            digit_chars / text_length
        ),

        "exclamation_count": text.count("!"),

        "question_count": text.count("?"),

        "url_count": len(urls),

        "has_url": int(len(urls) > 0),
    }

    return features


# ============================================================
# 5. Train / Test ayrımı
# ============================================================

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# ============================================================
# 6. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 7. Ek özellikleri çıkar
# ============================================================

train_features = pd.DataFrame(
    [extract_features(text) for text in X_train]
)

test_features = pd.DataFrame(
    [extract_features(text) for text in X_test]
)


print("\nEk özelliklerin ilk 5 satırı:")
print(train_features.head())


# ============================================================
# 8. Sayısal özellikleri sparse matrix'e dönüştür
# ============================================================

X_train_features = csr_matrix(
    train_features.values
)

X_test_features = csr_matrix(
    test_features.values
)
# Ek özellikleri ölçeklendir
scaler = StandardScaler()

X_train_features_scaled = scaler.fit_transform(
    train_features.values
)

X_test_features_scaled = scaler.transform(
    test_features.values
)

X_train_features_scaled = csr_matrix(
    X_train_features_scaled
)

X_test_features_scaled = csr_matrix(
    X_test_features_scaled
)


# ============================================================
# 9. TF-IDF + Ek özellikleri birleştir
# ============================================================

X_train_combined = hstack([
    X_train_tfidf,
    X_train_features_scaled
])

X_test_combined = hstack([
    X_test_tfidf,
    X_test_features_scaled
])


print("\nTF-IDF feature sayısı:")
print(X_train_tfidf.shape[1])

print("\nEk feature sayısı:")
print(X_train_features.shape[1])

print("\nToplam feature sayısı:")
print(X_train_combined.shape[1])


# ============================================================
# 10. Linear SVM
# ============================================================

model = LinearSVC(
    class_weight="balanced",
    random_state=42,
    max_iter=10000
)


# ============================================================
# 11. Modeli eğit
# ============================================================

model.fit(
    X_train_combined,
    y_train
)


# ============================================================
# 12. Tahmin
# ============================================================

y_pred = model.predict(
    X_test_combined
)


# ============================================================
# 13. Sonuçlar
# ============================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n===================================")
print("FEATURE ENGINEERING SONUÇLARI")
print("===================================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)