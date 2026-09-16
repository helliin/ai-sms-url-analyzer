import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

from sklearn.model_selection import train_test_split
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

MAX_LENGTH = 128
BATCH_SIZE = 8
EPOCHS = 2
LEARNING_RATE = 2e-5

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 70)
print("DEVICE")
print("=" * 70)

print(DEVICE)


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

print(f"Toplam mesaj: {len(df_a)}")
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

print(f"Toplam mesaj: {len(df_b)}")
print(f"HAM: {(df_b['label'] == 0).sum()}")
print(f"SPAM: {(df_b['label'] == 1).sum()}")


# ============================================================
# DATASET B CLEAN + SPLIT
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
# DATASET CLASS
# ============================================================

class SMSDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.tolist()
        self.labels = labels.tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=MAX_LENGTH,
            return_tensors="pt"
        )

        item = {
            key: value.squeeze(0)
            for key, value in encoding.items()
        }

        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.long
        )

        return item


# ============================================================
# TOKENIZER
# ============================================================

print()
print("=" * 70)
print("TOKENIZER")
print("=" * 70)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)


# ============================================================
# CREATE DATASETS
# ============================================================

train_dataset = SMSDataset(
    combined_train["text"],
    combined_train["label"],
    tokenizer
)

test_dataset = SMSDataset(
    b_test["text"],
    b_test["label"],
    tokenizer
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

print()
print("=" * 70)
print("MODEL")
print("=" * 70)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2
)

model.to(DEVICE)


# ============================================================
# CLASS WEIGHTS
# ============================================================

ham_count = (combined_train["label"] == 0).sum()
spam_count = (combined_train["label"] == 1).sum()

total = ham_count + spam_count

weight_ham = total / (2 * ham_count)
weight_spam = total / (2 * spam_count)

class_weights = torch.tensor(
    [weight_ham, weight_spam],
    dtype=torch.float
).to(DEVICE)

print()
print("=" * 70)
print("CLASS WEIGHTS")
print("=" * 70)

print(f"HAM weight : {weight_ham:.4f}")
print(f"SPAM weight: {weight_spam:.4f}")


# ============================================================
# LOSS
# ============================================================

criterion = torch.nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

model.train()

for epoch in range(EPOCHS):

    total_loss = 0

    print()
    print(f"Epoch {epoch + 1}/{EPOCHS}")

    for step, batch in enumerate(train_loader):

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        loss = criterion(
            outputs.logits,
            labels
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

        if (step + 1) % 100 == 0:
            print(
                f"Step {step + 1}/{len(train_loader)} "
                f"- Loss: {loss.item():.4f}"
            )

    average_loss = total_loss / len(train_loader)

    print(
        f"Epoch {epoch + 1} Average Loss: "
        f"{average_loss:.4f}"
    )


# ============================================================
# EVALUATION
# ============================================================

print()
print("=" * 70)
print("EVALUATION")
print("=" * 70)

model.eval()

all_predictions = []
all_labels = []

with torch.no_grad():

    for batch in test_loader:

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        predictions = torch.argmax(
            outputs.logits,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)


# ============================================================
# RESULTS
# ============================================================

print()
print("=" * 70)
print("COMBINED DATASET + DISTILBERT RESULT")
print("=" * 70)

print()
print("Dataset A + Dataset B Train")
print("          ↓")
print("      DistilBERT")
print("          ↓")
print(" Unseen Dataset B Test")
print()

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=["HAM", "SPAM"],
        zero_division=0
    )
)
# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

tn, fp, fn, tp = cm.ravel()

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()
print(cm)

print()
print(f"True Negative  (TN): {tn}")
print(f"False Positive (FP): {fp}")
print(f"False Negative (FN): {fn}")
print(f"True Positive  (TP): {tp}")
# ============================================================
# ERROR ANALYSIS
# ============================================================

b_test_results = b_test.copy()

b_test_results["prediction"] = all_predictions

false_positives = b_test_results[
    (b_test_results["label"] == 0) &
    (b_test_results["prediction"] == 1)
]

false_negatives = b_test_results[
    (b_test_results["label"] == 1) &
    (b_test_results["prediction"] == 0)
]

print()
print("=" * 70)
print("ERROR ANALYSIS")
print("=" * 70)

print()
print(f"False Positives: {len(false_positives)}")
print(f"False Negatives: {len(false_negatives)}")


print()
print("=" * 70)
print("FALSE POSITIVE EXAMPLES")
print("=" * 70)

for i, row in false_positives.head(10).iterrows():
    print()
    print(f"Gerçek: HAM")
    print(f"Tahmin: SPAM")
    print(f"Mesaj: {row['text']}")


print()
print("=" * 70)
print("FALSE NEGATIVE EXAMPLES")
print("=" * 70)

for i, row in false_negatives.head(10).iterrows():
    print()
    print(f"Gerçek: SPAM")
    print(f"Tahmin: HAM")
    print(f"Mesaj: {row['text']}")
    # ============================================================
# SAVE DISTILBERT MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING DISTILBERT MODEL")
print("=" * 70)

MODEL_OUTPUT_DIR = "models/distilbert_combined"

model.save_pretrained(MODEL_OUTPUT_DIR)
tokenizer.save_pretrained(MODEL_OUTPUT_DIR)

print()
print(f"Model kaydedildi: {MODEL_OUTPUT_DIR}")