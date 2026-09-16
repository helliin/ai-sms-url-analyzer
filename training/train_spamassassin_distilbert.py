import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW

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

TRAIN_PATH = "data/spamassassin_train.csv"
VAL_PATH = "data/spamassassin_val.csv"
TEST_PATH = "data/spamassassin_test.csv"

MODEL_OUTPUT_DIR = "models/distilbert_spamassassin"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DEVICE
# ============================================================

print("=" * 70)
print("SPAMASSASSIN + DISTILBERT")
print("=" * 70)

print()
print(f"Device: {DEVICE}")


# ============================================================
# LOAD DATA
# ============================================================

print()
print("=" * 70)
print("DATASET")
print("=" * 70)

train_df = pd.read_csv(TRAIN_PATH)
val_df = pd.read_csv(VAL_PATH)
test_df = pd.read_csv(TEST_PATH)

print()
print(f"Train: {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test: {len(test_df)}")

print()
print("Train label dağılımı:")
print(train_df["label"].value_counts())

print()
print("Validation label dağılımı:")
print(val_df["label"].value_counts())

print()
print("Test label dağılımı:")
print(test_df["label"].value_counts())


# ============================================================
# DATASET CLASS
# ============================================================

class EmailDataset(Dataset):

    def __init__(self, texts, labels, tokenizer):
        self.texts = texts.astype(str).tolist()
        self.labels = labels.astype(int).tolist()
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
# LABEL ENCODING
# ============================================================

label_map = {
    "ham": 0,
    "spam": 1
}

train_df["label"] = train_df["label"].map(label_map)
val_df["label"] = val_df["label"].map(label_map)
test_df["label"] = test_df["label"].map(label_map)


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

train_dataset = EmailDataset(
    train_df["text"],
    train_df["label"],
    tokenizer
)

val_dataset = EmailDataset(
    val_df["text"],
    val_df["label"],
    tokenizer
)

test_dataset = EmailDataset(
    test_df["text"],
    test_df["label"],
    tokenizer
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
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

ham_count = (train_df["label"] == 0).sum()
spam_count = (train_df["label"] == 1).sum()

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
# VALIDATION FUNCTION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch in loader:

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

    return accuracy, precision, recall, f1


# ============================================================
# TRAINING
# ============================================================

print()
print("=" * 70)
print("TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    model.train()

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

    val_accuracy, val_precision, val_recall, val_f1 = evaluate(
        model,
        val_loader
    )

    print()
    print(f"Epoch {epoch + 1} Average Loss: {average_loss:.4f}")
    print(f"Validation Accuracy : {val_accuracy:.4f}")
    print(f"Validation Precision: {val_precision:.4f}")
    print(f"Validation Recall   : {val_recall:.4f}")
    print(f"Validation F1       : {val_f1:.4f}")


# ============================================================
# FINAL TEST
# ============================================================

print()
print("=" * 70)
print("FINAL TEST")
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
print("SPAMASSASSIN DISTILBERT RESULT")
print("=" * 70)

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
# SAVE MODEL
# ============================================================

print()
print("=" * 70)
print("SAVING MODEL")
print("=" * 70)

model.save_pretrained(
    MODEL_OUTPUT_DIR
)

tokenizer.save_pretrained(
    MODEL_OUTPUT_DIR
)

print()
print(f"Model kaydedildi: {MODEL_OUTPUT_DIR}")

print()
print("=" * 70)
print("TRAINING TAMAMLANDI")
print("=" * 70)