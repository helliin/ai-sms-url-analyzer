import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# 1. Dataseti oku
file_path = "data/sms/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)


# 2. Duplicate temizleme
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)


# 3. Etiketleri sayısal hale getir
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# 4. X ve y
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
# 7. Modeller

models = {
    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        random_state=42
    ),

    "Linear SVM": LinearSVC(
        class_weight="balanced",
        random_state=42
    )
}
# 8. Modelleri eğit ve değerlendir

results = []

for model_name, model in models.items():

    print(f"\nModel çalıştırılıyor: {model_name}")

    # Modeli eğit
    model.fit(X_train_tfidf, y_train)

    # Test tahmini
    y_pred = model.predict(X_test_tfidf)

    # Metrikleri hesapla
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

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })
    # 9. Sonuçları tablo olarak göster

results_df = pd.DataFrame(results)

print("\n=== MODEL COMPARISON ===")

print(
    results_df.to_string(
        index=False,
        formatters={
            "Accuracy": "{:.4f}".format,
            "Precision": "{:.4f}".format,
            "Recall": "{:.4f}".format,
            "F1": "{:.4f}".format
        }
    )
)