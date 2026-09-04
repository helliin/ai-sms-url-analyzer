import re

import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


DATA_PATH = "data/sms/SMSSpamCollection"


def extract_features(texts):
    features = []

    url_pattern = r"https?://\S+|www\.\S+"

    for text in texts:
        text_length = len(text)

        words = text.split()
        word_count = len(words)

        uppercase_count = sum(1 for c in text if c.isupper())
        digit_count = sum(1 for c in text if c.isdigit())

        uppercase_ratio = (
            uppercase_count / text_length
            if text_length > 0
            else 0
        )

        digit_ratio = (
            digit_count / text_length
            if text_length > 0
            else 0
        )

        exclamation_count = text.count("!")
        question_count = text.count("?")

        urls = re.findall(url_pattern, text)
        url_count = len(urls)
        has_url = 1 if url_count > 0 else 0

        features.append({
            "text_length": text_length,
            "word_count": word_count,
            "uppercase_ratio": uppercase_ratio,
            "digit_ratio": digit_ratio,
            "exclamation_count": exclamation_count,
            "question_count": question_count,
            "url_count": url_count,
            "has_url": has_url,
        })

    return pd.DataFrame(features)


def evaluate_model(name, X_train, X_test, y_train, y_test):
    model = LinearSVC(
        class_weight="balanced",
        random_state=42,
        max_iter=10000
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, zero_division=0)
    recall = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    return {
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
    }


# --------------------------------------------------
# 1. DATASET
# --------------------------------------------------

df = pd.read_csv(
    DATA_PATH,
    sep="\t",
    header=None,
    names=["label", "text"]
)

df = df.drop_duplicates()

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

X = df["text"]
y = df["label"]


# --------------------------------------------------
# 2. TRAIN / TEST SPLIT
# --------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# --------------------------------------------------
# 3. TF-IDF
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=2
)

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# --------------------------------------------------
# 4. EXTRA FEATURES
# --------------------------------------------------

train_features = extract_features(X_train)
test_features = extract_features(X_test)


feature_groups = {
    "Text": [
        "text_length",
        "word_count",
    ],

    "Style": [
        "uppercase_ratio",
        "digit_ratio",
    ],

    "Punctuation": [
        "exclamation_count",
        "question_count",
    ],

    "URL": [
        "url_count",
        "has_url",
    ],
}


# --------------------------------------------------
# 5. MODEL COMBINATIONS
# --------------------------------------------------

combinations = {
    "Baseline": [],

    "Style": [
        "Style"
    ],

    "Style + Punctuation": [
        "Style",
        "Punctuation"
    ],

    "Style + URL": [
        "Style",
        "URL"
    ],

    "Style + Text": [
        "Style",
        "Text"
    ],

    "All": [
        "Style",
        "Punctuation",
        "URL",
        "Text"
    ],
}


results = []


# --------------------------------------------------
# 6. BASELINE
# --------------------------------------------------

print("\nBaseline model çalıştırılıyor...")

baseline_result = evaluate_model(
    "Baseline",
    X_train_tfidf,
    X_test_tfidf,
    y_train,
    y_test
)

results.append(baseline_result)


# --------------------------------------------------
# 7. COMBINATIONS
# --------------------------------------------------

for combination_name, groups in combinations.items():

    if combination_name == "Baseline":
        continue

    selected_features = []

    for group in groups:
        selected_features.extend(feature_groups[group])

    print(
        f"\n{combination_name} çalıştırılıyor..."
    )

    scaler = StandardScaler()

    train_extra = scaler.fit_transform(
        train_features[selected_features]
    )

    test_extra = scaler.transform(
        test_features[selected_features]
    )

    train_extra = csr_matrix(train_extra)
    test_extra = csr_matrix(test_extra)

    X_train_combined = hstack([
        X_train_tfidf,
        train_extra
    ])

    X_test_combined = hstack([
        X_test_tfidf,
        test_extra
    ])

    result = evaluate_model(
        combination_name,
        X_train_combined,
        X_test_combined,
        y_train,
        y_test
    )

    results.append(result)


# --------------------------------------------------
# 8. RESULTS
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 70)
print("FEATURE COMBINATION RESULTS")
print("=" * 70)

print(
    results_df.to_string(
        index=False,
        float_format=lambda x: f"{x:.4f}"
    )
)


# --------------------------------------------------
# 9. BEST MODEL
# --------------------------------------------------

best_model = results_df.loc[
    results_df["F1"].idxmax()
]

print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)

print(
    f"Model     : {best_model['Model']}"
)

print(
    f"Accuracy  : {best_model['Accuracy']:.4f}"
)

print(
    f"Precision : {best_model['Precision']:.4f}"
)

print(
    f"Recall    : {best_model['Recall']:.4f}"
)

print(
    f"F1        : {best_model['F1']:.4f}"
)