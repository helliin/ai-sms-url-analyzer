import pandas as pd
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# SETTINGS
# ============================================================

DISTILBERT_PATH = "models/distilbert_combined"

MAX_LENGTH = 128
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("ERROR ANALYSIS - SVM vs DISTILBERT")
print("=" * 70)

print()
print(f"Device: {DEVICE}")


# ============================================================
# DATASET A
# ============================================================

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


# ============================================================
# DATASET B
# ============================================================

df_b = pd.read_csv(
    "data/sms/exais_clean.csv"
)


# ============================================================
# DATASET B SPLIT
# ============================================================

b_train, b_test = train_test_split(
    df_b,
    test_size=0.20,
    random_state=42,
    stratify=df_b["label"]
)


# ============================================================
# COMBINED TRAINING DATA
# ============================================================

combined_train = pd.concat(
    [df_a, b_train],
    ignore_index=True
)


# ============================================================
# TEXT DATA
# ============================================================

X_train = combined_train["text"].values
y_train = combined_train["label"].values

X_test = b_test["text"].values
y_test = b_test["label"].values


# ============================================================
# SVM
# ============================================================

print()
print("=" * 70)
print("TRAINING SVM")
print("=" * 70)

vectorizer = TfidfVectorizer(
    lowercase=True,
    ngram_range=(1, 2),
    min_df=1,
    sublinear_tf=True
)

X_train_tfidf = vectorizer.fit_transform(
    X_train
)

X_test_tfidf = vectorizer.transform(
    X_test
)

svm_model = LinearSVC(
    class_weight="balanced",
    random_state=42
)

svm_model.fit(
    X_train_tfidf,
    y_train
)

svm_predictions = svm_model.predict(
    X_test_tfidf
)


# ============================================================
# DISTILBERT
# ============================================================

print()
print("=" * 70)
print("LOADING DISTILBERT")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    DISTILBERT_PATH
)

bert_model = AutoModelForSequenceClassification.from_pretrained(
    DISTILBERT_PATH
)

bert_model.to(DEVICE)
bert_model.eval()


# ============================================================
# DISTILBERT PREDICTION
# ============================================================

print()
print("=" * 70)
print("DISTILBERT EVALUATION")
print("=" * 70)

distilbert_predictions = []


with torch.no_grad():

    for start in range(
        0,
        len(X_test),
        BATCH_SIZE
    ):

        batch_texts = X_test[
            start:start + BATCH_SIZE
        ].tolist()

        encoding = tokenizer(
            batch_texts,
            truncation=True,
            padding=True,
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        input_ids = encoding[
            "input_ids"
        ].to(DEVICE)

        attention_mask = encoding[
            "attention_mask"
        ].to(DEVICE)

        outputs = bert_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        distilbert_predictions.extend(
            predictions.cpu().numpy()
        )


distilbert_predictions = np.array(
    distilbert_predictions
)


# ============================================================
# ERROR MASKS
# ============================================================

svm_fp_mask = (
    (y_test == 0) &
    (svm_predictions == 1)
)

svm_fn_mask = (
    (y_test == 1) &
    (svm_predictions == 0)
)

bert_fp_mask = (
    (y_test == 0) &
    (distilbert_predictions == 1)
)

bert_fn_mask = (
    (y_test == 1) &
    (distilbert_predictions == 0)
)


# ============================================================
# ERROR COUNTS
# ============================================================

print()
print("=" * 70)
print("ERROR COUNTS")
print("=" * 70)

print()

print(
    f"SVM False Positive : "
    f"{svm_fp_mask.sum()}"
)

print(
    f"SVM False Negative : "
    f"{svm_fn_mask.sum()}"
)

print()

print(
    f"DistilBERT False Positive : "
    f"{bert_fp_mask.sum()}"
)

print(
    f"DistilBERT False Negative : "
    f"{bert_fn_mask.sum()}"
)


# ============================================================
# SVM FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("SVM FALSE POSITIVES")
print("=" * 70)

svm_fp_indices = np.where(
    svm_fp_mask
)[0]

for idx in svm_fp_indices:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print("Gerçek: HAM")
    print("SVM: SPAM")
    print(
        f"DistilBERT: "
        f"{'SPAM' if distilbert_predictions[idx] == 1 else 'HAM'}"
    )


# ============================================================
# SVM FALSE NEGATIVES
# ============================================================

print()
print("=" * 70)
print("SVM FALSE NEGATIVES")
print("=" * 70)

svm_fn_indices = np.where(
    svm_fn_mask
)[0]

for idx in svm_fn_indices:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print("Gerçek: SPAM")
    print("SVM: HAM")
    print(
        f"DistilBERT: "
        f"{'SPAM' if distilbert_predictions[idx] == 1 else 'HAM'}"
    )


# ============================================================
# DISTILBERT FALSE POSITIVES
# ============================================================

print()
print("=" * 70)
print("DISTILBERT FALSE POSITIVES")
print("=" * 70)

bert_fp_indices = np.where(
    bert_fp_mask
)[0]

for idx in bert_fp_indices:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print("Gerçek: HAM")
    print(
        f"SVM: "
        f"{'SPAM' if svm_predictions[idx] == 1 else 'HAM'}"
    )
    print("DistilBERT: SPAM")


# ============================================================
# DISTILBERT FALSE NEGATIVES
# ============================================================

print()
print("=" * 70)
print("DISTILBERT FALSE NEGATIVES")
print("=" * 70)

bert_fn_indices = np.where(
    bert_fn_mask
)[0]

for idx in bert_fn_indices:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print("Gerçek: SPAM")
    print(
        f"SVM: "
        f"{'SPAM' if svm_predictions[idx] == 1 else 'HAM'}"
    )
    print("DistilBERT: HAM")


# ============================================================
# BOTH MODELS WRONG
# ============================================================

both_wrong_mask = (
    (svm_predictions != y_test) &
    (distilbert_predictions != y_test)
)

both_wrong_indices = np.where(
    both_wrong_mask
)[0]


print()
print("=" * 70)
print("BOTH MODELS WRONG")
print("=" * 70)

print()
print(
    f"İki modelin birlikte hata yaptığı mesaj sayısı: "
    f"{len(both_wrong_indices)}"
)


for idx in both_wrong_indices:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print(
        f"Gerçek: "
        f"{'SPAM' if y_test[idx] == 1 else 'HAM'}"
    )
    print(
        f"SVM: "
        f"{'SPAM' if svm_predictions[idx] == 1 else 'HAM'}"
    )
    print(
        f"DistilBERT: "
        f"{'SPAM' if distilbert_predictions[idx] == 1 else 'HAM'}"
    )


# ============================================================
# MODEL DISAGREEMENT
# ============================================================

disagreement_mask = (
    svm_predictions != distilbert_predictions
)

disagreement_indices = np.where(
    disagreement_mask
)[0]


print()
print("=" * 70)
print("MODEL DISAGREEMENT")
print("=" * 70)

print()

print(
    f"Farklı tahmin sayısı: "
    f"{len(disagreement_indices)}"
)


for idx in disagreement_indices[:30]:

    print()
    print(f"Mesaj: {X_test[idx]}")

    print(
        f"Gerçek: "
        f"{'SPAM' if y_test[idx] == 1 else 'HAM'}"
    )

    print(
        f"SVM: "
        f"{'SPAM' if svm_predictions[idx] == 1 else 'HAM'}"
    )

    print(
        f"DistilBERT: "
        f"{'SPAM' if distilbert_predictions[idx] == 1 else 'HAM'}"
    )


# ============================================================
# CONFUSION MATRICES
# ============================================================

svm_cm = confusion_matrix(
    y_test,
    svm_predictions
)

bert_cm = confusion_matrix(
    y_test,
    distilbert_predictions
)


print()
print("=" * 70)
print("CONFUSION MATRICES")
print("=" * 70)

print()
print("SVM:")
print(svm_cm)

print()
print("DistilBERT:")
print(bert_cm)


# ============================================================
# ERROR DATAFRAME
# ============================================================

error_df = pd.DataFrame({
    "text": X_test,
    "true_label": y_test,
    "svm_prediction": svm_predictions,
    "distilbert_prediction": distilbert_predictions
})


error_df["svm_error"] = (
    error_df["true_label"] !=
    error_df["svm_prediction"]
)

error_df["distilbert_error"] = (
    error_df["true_label"] !=
    error_df["distilbert_prediction"]
)

error_df["both_wrong"] = (
    error_df["svm_error"] &
    error_df["distilbert_error"]
)

error_df["models_disagree"] = (
    error_df["svm_prediction"] !=
    error_df["distilbert_prediction"]
)


# ============================================================
# SAVE RESULTS
# ============================================================

output_path = (
    "training/error_analysis_results.csv"
)

error_df.to_csv(
    output_path,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)

print()
print(
    f"Error Analysis dosyası: "
    f"{output_path}"
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("ERROR ANALYSIS COMPLETED")
print("=" * 70)

print()

print(
    f"Toplam test mesajı: {len(y_test)}"
)

print(
    f"SVM hata sayısı: "
    f"{svm_fp_mask.sum() + svm_fn_mask.sum()}"
)

print(
    f"DistilBERT hata sayısı: "
    f"{bert_fp_mask.sum() + bert_fn_mask.sum()}"
)

print(
    f"İki modelin birlikte hata yaptığı: "
    f"{len(both_wrong_indices)}"
)

print(
    f"Modellerin farklı tahmin yaptığı: "
    f"{len(disagreement_indices)}"
)