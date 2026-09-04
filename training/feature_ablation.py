import re
import pandas as pd
from scipy.sparse import csr_matrix, hstack

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


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

df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# ============================================================
# 2. Feature çıkarma
# ============================================================

def extract_features(text):

    text = str(text)
    words = text.split()

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text.lower()
    )

    uppercase_chars = sum(
        1 for char in text if char.isupper()
    )

    digit_chars = sum(
        1 for char in text if char.isdigit()
    )

    text_length = max(len(text), 1)

    return {
        "text_length": len(text),
        "word_count": len(words),
        "uppercase_ratio": uppercase_chars / text_length,
        "digit_ratio": digit_chars / text_length,
        "exclamation_count": text.count("!"),
        "question_count": text.count("?"),
        "url_count": len(urls),
        "has_url": int(len(urls) > 0)
    }


# ============================================================
# 3. Train / Test
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
# 4. TF-IDF
# ============================================================

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# ============================================================
# 5. Ek özellikleri çıkar
# ============================================================

train_features = pd.DataFrame(
    [extract_features(text) for text in X_train]
)

test_features = pd.DataFrame(
    [extract_features(text) for text in X_test]
)


# ============================================================
# 6. Feature grupları
# ============================================================

feature_groups = {

    "Text": [
        "text_length",
        "word_count"
    ],

    "Style": [
        "uppercase_ratio",
        "digit_ratio"
    ],

    "Punctuation": [
        "exclamation_count",
        "question_count"
    ],

    "URL": [
        "url_count",
        "has_url"
    ]
}


# ============================================================
# 7. Her feature grubunu ayrı ayrı test et
# ============================================================

results = []


for group_name, selected_features in feature_groups.items():

    print("\n===================================")
    print(f"TEST EDİLEN GRUP: {group_name}")
    print("===================================")

    train_selected = train_features[
        selected_features
    ]

    test_selected = test_features[
        selected_features
    ]

    # Ölçeklendirme
    scaler = StandardScaler()

    train_scaled = scaler.fit_transform(
        train_selected
    )

    test_scaled = scaler.transform(
        test_selected
    )

    train_scaled = csr_matrix(train_scaled)
    test_scaled = csr_matrix(test_scaled)

    # TF-IDF + seçilen feature grubu
    X_train_combined = hstack([
        X_train_tfidf,
        train_scaled
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        test_scaled
    ])

    # Model
    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    model.fit(
        X_train_combined,
        y_train
    )

    y_pred = model.predict(
        X_test_combined
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred
    )

    recall = recall_score(
        y_test,
        y_pred
    )

    f1 = f1_score(
        y_test,
        y_pred
    )

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1       : {f1:.4f}")

    results.append({
        "Feature Group": group_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })


# ============================================================
# 8. Sonuçları tablo halinde göster
# ============================================================

results_df = pd.DataFrame(results)

print("\n\n===================================")
print("FEATURE ABLATION SONUÇLARI")
print("===================================")

print(
    results_df.to_string(index=False)
)