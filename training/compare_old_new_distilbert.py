import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# SETTINGS
# ============================================================

TEST_PATH = "data/spamassassin_test.csv"

OLD_MODEL_PATH = "models/distilbert_combined"
NEW_MODEL_PATH = "models/distilbert_spamassassin"

MAX_LENGTH = 128
BATCH_SIZE = 8

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# START
# ============================================================

print("=" * 70)
print("OLD DISTILBERT vs NEW DISTILBERT")
print("=" * 70)

print()
print(f"Device: {DEVICE}")


# ============================================================
# LOAD TEST DATA
# ============================================================

print()
print("=" * 70)
print("TEST DATASET")
print("=" * 70)

test_df = pd.read_csv(TEST_PATH)

label_map = {
    "ham": 0,
    "spam": 1
}

test_df["label"] = test_df["label"].map(label_map)

print()
print(f"Test samples: {len(test_df)}")

print()
print("Label distribution:")
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
# EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_path, model_name):

    print()
    print("=" * 70)
    print(model_name)
    print("=" * 70)

    print()
    print("Model yükleniyor...")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        model_path
    )

    model.to(DEVICE)
    model.eval()

    dataset = EmailDataset(
        test_df["text"],
        test_df["label"],
        tokenizer
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    all_predictions = []
    all_labels = []

    print()
    print("Test başlıyor...")

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

    cm = confusion_matrix(
        all_labels,
        all_predictions
    )

    print()
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1 Score : {f1:.4f}")

    print()
    print("Confusion Matrix:")
    print(cm)

    print()
    print("Classification Report:")
    print(
        classification_report(
            all_labels,
            all_predictions,
            target_names=["HAM", "SPAM"],
            zero_division=0
        )
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ============================================================
# OLD MODEL
# ============================================================

old_results = evaluate_model(
    OLD_MODEL_PATH,
    "OLD DISTILBERT - SMS + ExAIS"
)


# ============================================================
# NEW MODEL
# ============================================================

new_results = evaluate_model(
    NEW_MODEL_PATH,
    "NEW DISTILBERT - SpamAssassin"
)


# ============================================================
# FINAL COMPARISON
# ============================================================

print()
print("=" * 70)
print("FINAL COMPARISON")
print("=" * 70)

print()

print(
    f"{'Metric':<15}"
    f"{'Old Model':<15}"
    f"{'New Model':<15}"
)

print("-" * 45)

print(
    f"{'Accuracy':<15}"
    f"{old_results['accuracy']:<15.4f}"
    f"{new_results['accuracy']:<15.4f}"
)

print(
    f"{'Precision':<15}"
    f"{old_results['precision']:<15.4f}"
    f"{new_results['precision']:<15.4f}"
)

print(
    f"{'Recall':<15}"
    f"{old_results['recall']:<15.4f}"
    f"{new_results['recall']:<15.4f}"
)

print(
    f"{'F1 Score':<15}"
    f"{old_results['f1']:<15.4f}"
    f"{new_results['f1']:<15.4f}"
)

print()
print("=" * 70)
print("COMPARISON TAMAMLANDI")
print("=" * 70)