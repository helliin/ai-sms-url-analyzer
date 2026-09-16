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
    classification_report,
    confusion_matrix
)

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# SETTINGS
# ============================================================

MODEL_NAME = "distilbert-base-uncased"
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
print("SVM vs DISTILBERT COMPARISON")
print("=" * 70)

print()
print(f"Device: {DEVICE}")


# ============================================================
# DATASET A
# ============================================================

print()
print("=" * 70)
print("DATASET A")
print("=" * 70)

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

print(f"Toplam: {len(df_a)}")
print(f"HAM: {(df_a['label'] == 0).sum()}")
print(f"SPAM: {(df_a['label'] == 1).sum()}")


# ============================================================
# DATASET B
# ============================================================

print()
print("=" * 70)
print("DATASET B")
print("=" * 70)

df_b = pd.read_csv(
    "data/sms/exais_clean.csv"
)

print(f"Toplam: {len(df_b)}")
print(f"HAM: {(df_b['label'] == 0).sum()}")
print(f"SPAM: {(df_b['label'] == 1).sum()}")


# ============================================================
# DATASET B SPLIT
# ============================================================

df_b["_normalized_text"] = (
    df_b["text"]
    .astype(str)
    .str.strip()
    .str.lower()
)

df_b = df_b.drop_duplicates(
    subset="_normalized_text"
).drop(
    columns="_normalized_text"
)

b_train, b_test = train_test_split(
    df_b,
    test_size=0.20,
    random_state=42,
    stratify=df_b["label"]
)

print()
print("=" * 70)
print("DATASET B SPLIT")
print("=" * 70)

print(f"B Train: {len(b_train)}")
print(f"B Test : {len(b_test)}")


# ============================================================
# COMBINED TRAINING DATA
# ============================================================

combined_train = pd.concat(
    [df_a, b_train],
    ignore_index=True
)

print()
print("=" * 70)
print("COMBINED TRAINING DATA")
print("=" * 70)

print(f"Toplam: {len(combined_train)}")
print(f"HAM: {(combined_train['label'] == 0).sum()}")
print(f"SPAM: {(combined_train['label'] == 1).sum()}")


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

X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

print(f"TF-IDF feature sayısı: {X_train_tfidf.shape[1]}")

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
# SVM METRICS
# ============================================================

svm_accuracy = accuracy_score(
    y_test,
    svm_predictions
)

svm_precision = precision_score(
    y_test,
    svm_predictions,
    zero_division=0
)

svm_recall = recall_score(
    y_test,
    svm_predictions,
    zero_division=0
)

svm_f1 = f1_score(
    y_test,
    svm_predictions,
    zero_division=0
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

    for start in range(0, len(X_test), BATCH_SIZE):

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

        input_ids = encoding["input_ids"].to(DEVICE)
        attention_mask = encoding["attention_mask"].to(DEVICE)

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
# DISTILBERT METRICS
# ============================================================

bert_accuracy = accuracy_score(
    y_test,
    distilbert_predictions
)

bert_precision = precision_score(
    y_test,
    distilbert_predictions,
    zero_division=0
)

bert_recall = recall_score(
    y_test,
    distilbert_predictions,
    zero_division=0
)

bert_f1 = f1_score(
    y_test,
    distilbert_predictions,
    zero_division=0
)


# ============================================================
# COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Metric':<15}"
    f"{'SVM':>12}"
    f"{'DistilBERT':>15}"
    f"{'Difference':>15}"
)

print("-" * 57)

print(
    f"{'Accuracy':<15}"
    f"{svm_accuracy:>12.4f}"
    f"{bert_accuracy:>15.4f}"
    f"{bert_accuracy - svm_accuracy:>15.4f}"
)

print(
    f"{'Precision':<15}"
    f"{svm_precision:>12.4f}"
    f"{bert_precision:>15.4f}"
    f"{bert_precision - svm_precision:>15.4f}"
)

print(
    f"{'Recall':<15}"
    f"{svm_recall:>12.4f}"
    f"{bert_recall:>15.4f}"
    f"{bert_recall - svm_recall:>15.4f}"
)

print(
    f"{'F1':<15}"
    f"{svm_f1:>12.4f}"
    f"{bert_f1:>15.4f}"
    f"{bert_f1 - svm_f1:>15.4f}"
)


# ============================================================
# SVM CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("SVM CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        svm_predictions,
        target_names=["HAM", "SPAM"],
        zero_division=0
    )
)


# ============================================================
# DISTILBERT CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("DISTILBERT CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        y_test,
        distilbert_predictions,
        target_names=["HAM", "SPAM"],
        zero_division=0
    )
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
print("SVM CONFUSION MATRIX")
print("=" * 70)

print(svm_cm)

svm_tn, svm_fp, svm_fn, svm_tp = svm_cm.ravel()

print()
print(f"TN: {svm_tn}")
print(f"FP: {svm_fp}")
print(f"FN: {svm_fn}")
print(f"TP: {svm_tp}")


print()
print("=" * 70)
print("DISTILBERT CONFUSION MATRIX")
print("=" * 70)

print(bert_cm)

bert_tn, bert_fp, bert_fn, bert_tp = bert_cm.ravel()

print()
print(f"TN: {bert_tn}")
print(f"FP: {bert_fp}")
print(f"FN: {bert_fn}")
print(f"TP: {bert_tp}")


# ============================================================
# ERROR COMPARISON
# ============================================================

print()
print("=" * 70)
print("ERROR COMPARISON")
print("=" * 70)

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


print()
print(f"SVM False Positive : {svm_fp_mask.sum()}")
print(f"SVM False Negative : {svm_fn_mask.sum()}")

print()
print(f"DistilBERT False Positive : {bert_fp_mask.sum()}")
print(f"DistilBERT False Negative : {bert_fn_mask.sum()}")


# ============================================================
# MODEL AGREEMENT
# ============================================================

same_prediction = (
    svm_predictions == distilbert_predictions
)

different_prediction = (
    svm_predictions != distilbert_predictions
)

print()
print("=" * 70)
print("MODEL AGREEMENT")
print("=" * 70)

print()
print(
    f"Aynı tahmin: {same_prediction.sum()}"
)

print(
    f"Farklı tahmin: {different_prediction.sum()}"
)


# ============================================================
# DIFFERENT PREDICTIONS
# ============================================================

print()
print("=" * 70)
print("MODELS DISAGREE - EXAMPLES")
print("=" * 70)

different_indices = np.where(
    different_prediction
)[0]


for idx in different_indices[:20]:

    print()
    print(f"Mesaj: {X_test[idx]}")
    print(f"Gerçek: {'SPAM' if y_test[idx] == 1 else 'HAM'}")
    print(
        f"SVM: "
        f"{'SPAM' if svm_predictions[idx] == 1 else 'HAM'}"
    )
    print(
        f"DistilBERT: "
        f"{'SPAM' if distilbert_predictions[idx] == 1 else 'HAM'}"
    )


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("COMPARISON COMPLETED")
print("=" * 70)