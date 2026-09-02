import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix


# Dataseti oku
df = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("İlk dataset boyutu:")
print(df.shape)


# Duplicate mesajları temizle
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

print("\nDuplicate temizlendikten sonra:")
print(df.shape)


# Özellik ve hedef değişken
X = df["message"]
y = df["label"]


# Train / Test ayır
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# TF-IDF Vectorizer oluştur
vectorizer = TfidfVectorizer()


# Sadece train verisi üzerinde öğren
X_train_tfidf = vectorizer.fit_transform(X_train)


# Test verisini aynı vocabulary ile dönüştür
X_test_tfidf = vectorizer.transform(X_test)


print("\nTF-IDF train boyutu:")
print(X_train_tfidf.shape)

print("\nTF-IDF test boyutu:")
print(X_test_tfidf.shape)


# Linear SVM modeli
model = LinearSVC()


# Modeli train verisiyle eğit
model.fit(X_train_tfidf, y_train)


# Test verisi üzerinde tahmin yap
y_pred = model.predict(X_test_tfidf)


print("\nİlk 20 tahmin:")
print(y_pred[:20])


# Model performansı
print("\nModel Performansı:")
print(classification_report(y_test, y_pred))


# Confusion Matrix
cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["ham", "spam"]
)

print("\nConfusion Matrix:")
print(cm)


print("\nTrain boyutu:")
print(X_train.shape)

print("\nTest boyutu:")
print(X_test.shape)


print("\nTrain etiket dağılımı:")
print(y_train.value_counts())


print("\nTest etiket dağılımı:")
print(y_test.value_counts())