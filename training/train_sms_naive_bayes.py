import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report


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


# 7. Naive Bayes modelini oluştur
model = MultinomialNB()


# 8. Modeli eğit
model.fit(X_train_tfidf, y_train)


# 9. Test verisi üzerinde tahmin
y_pred = model.predict(X_test_tfidf)


# 10. Sonuçları göster
accuracy = accuracy_score(y_test, y_pred)

print("Model: Multinomial Naive Bayes")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["ham", "spam"]
    )
)