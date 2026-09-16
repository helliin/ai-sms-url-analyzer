import sys
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from sklearn.model_selection import train_test_split

from backend.ai.llm_analyzer import classify_with_local_llm


# ============================================================
# AYARLAR
# ============================================================

DATASET_PATH = BASE_DIR / "data" / "sms" / "exais_clean.csv"
RESULTS_PATH = BASE_DIR / "training" / "local_llm_results.csv"

TEST_LIMIT = None


# ============================================================
# BAŞLANGIÇ
# ============================================================

print("=" * 70)
print("LOCAL LLM EVALUATION - QWEN 3 4B")
print("=" * 70)

print("\nDataset yükleniyor...")

df = pd.read_csv(DATASET_PATH)

print(f"Toplam dataset: {len(df)}")


# ============================================================
# DUPLICATE TEMİZLEME
# ============================================================

print("\nDuplicate mesajlar temizleniyor...")

before = len(df)

df = df.drop_duplicates(
    subset=["text"]
).reset_index(drop=True)

after = len(df)

print(f"Önce: {before}")
print(f"Sonra: {after}")
print(f"Silinen duplicate: {before - after}")


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

train_df, test_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["label"]
)

print(f"\nTrain seti: {len(train_df)}")
print(f"Test seti: {len(test_df)}")


# ============================================================
# TEST LIMIT
# ============================================================

if TEST_LIMIT is not None:
    test_df = test_df.head(TEST_LIMIT)

print(
    f"Evaluation için kullanılacak mesaj: {len(test_df)}"
)


# ============================================================
# LABEL DÖNÜŞÜMÜ
# ============================================================

test_df = test_df.copy()

test_df["true_label"] = test_df["label"].map({
    0: "ham",
    1: "spam"
})


# ============================================================
# EVALUATION
# ============================================================

results = []

start_time = time.time()


for index, row in test_df.iterrows():

    message = str(row["text"])
    true_label = row["true_label"]

    print("\n" + "-" * 70)

    print(
        f"Mesaj {len(results) + 1}/{len(test_df)}"
    )

    print(
        f"Gerçek etiket: {true_label}"
    )

    print(
        f"SMS: {message[:120]}"
    )

    message_start = time.time()

    try:

        prediction = classify_with_local_llm(
            message
        )

    except Exception as e:

        prediction = "unknown"

        print(
            f"Qwen hatası: {e}"
        )

    elapsed = time.time() - message_start

    print(
        f"Qwen tahmini: {prediction}"
    )

    print(
        f"Doğru mu: {prediction == true_label}"
    )

    print(
        f"Süre: {elapsed:.2f} saniye"
    )

    results.append({
        "message": message,
        "true_label": true_label,
        "prediction": prediction,
        "elapsed_seconds": elapsed
    })


# ============================================================
# SONUÇLARI KAYDET
# ============================================================

total_time = time.time() - start_time

results_df = pd.DataFrame(
    results
)

results_df.to_csv(
    RESULTS_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSonuçlar kaydedildi:")

print(
    RESULTS_PATH
)


# ============================================================
# METRİKLER
# ============================================================

valid_results = results_df[
    results_df["prediction"].isin(
        ["ham", "spam"]
    )
].copy()

unknown_count = (
    len(results_df)
    - len(valid_results)
)


print("\n" + "=" * 70)
print("EVALUATION SONUÇLARI")
print("=" * 70)

print(
    f"Toplam mesaj: {len(results_df)}"
)

print(
    f"Geçerli tahmin: {len(valid_results)}"
)

print(
    f"Unknown/başarısız: {unknown_count}"
)


if len(valid_results) > 0:

    y_true = valid_results["true_label"]

    y_pred = valid_results["prediction"]


    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        pos_label="spam",
        zero_division=0
    )


    print("\nMetrics:")

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=["ham", "spam"]
    )

    print("\nConfusion Matrix:")

    print(cm)


    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    print("\nClassification Report:")

    print(
        classification_report(
            y_true,
            y_pred,
            labels=["ham", "spam"],
            zero_division=0
        )
    )


# ============================================================
# PERFORMANS
# ============================================================

if len(results_df) > 0:

    average_time = (
        results_df["elapsed_seconds"].mean()
    )

    print("\n" + "=" * 70)
    print("PERFORMANCE")
    print("=" * 70)

    print(
        f"Toplam süre: {total_time:.2f} saniye"
    )

    print(
        f"Mesaj başına ortalama: "
        f"{average_time:.2f} saniye"
    )


print("\nEvaluation tamamlandı.")