import sys
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


# ============================================================
# 1. PROJE PATH
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

sys.path.append(
    str(BASE_DIR)
)


from backend.ai.llm_analyzer import analyze_with_llm


# ============================================================
# 2. SETTINGS
# ============================================================

DATASET_A_PATH = (
    "data/sms/SMSSpamCollection"
)

DATASET_B_PATH = (
    "data/sms/exais_clean.csv"
)

RANDOM_STATE = 42

TEST_SIZE = 0.20

# İlk test için 10 yapacağız.
# Daha sonra None yapıp tamamını çalıştıracağız.
TEST_LIMIT = 10

OUTPUT_PATH = (
    "training/llm_evaluation_results.csv"
)


# ============================================================
# 3. HEADER
# ============================================================

print("=" * 70)
print("GEMINI LLM EVALUATION")
print("=" * 70)

print()
print("Evaluation protokolü:")
print("Dataset A + Dataset B Train → eğitim verisi")
print("Dataset B Test → LLM değerlendirme seti")

print()
print(f"Random State: {RANDOM_STATE}")
print(f"Test Size: {TEST_SIZE}")
print(f"Test Limit: {TEST_LIMIT}")


# ============================================================
# 4. DATASET A
# ============================================================

print()
print("=" * 70)
print("DATASET A")
print("=" * 70)


df_a = pd.read_csv(
    DATASET_A_PATH,
    sep="\t",
    header=None,
    names=["label", "text"]
)


df_a = df_a.drop_duplicates()


df_a["label"] = df_a["label"].map({
    "ham": 0,
    "spam": 1
})


print(
    f"Toplam: {len(df_a)}"
)

print(
    f"HAM: {(df_a['label'] == 0).sum()}"
)

print(
    f"SPAM: {(df_a['label'] == 1).sum()}"
)


# ============================================================
# 5. DATASET B
# ============================================================

print()
print("=" * 70)
print("DATASET B")
print("=" * 70)


df_b = pd.read_csv(
    DATASET_B_PATH
)


print(
    f"Orijinal toplam: {len(df_b)}"
)


print(
    f"HAM: {(df_b['label'] == 0).sum()}"
)


print(
    f"SPAM: {(df_b['label'] == 1).sum()}"
)


# ============================================================
# 6. DATASET B DUPLICATE TEMİZLİĞİ
# ============================================================

df_b["_normalized_text"] = (
    df_b["text"]
    .astype(str)
    .str.strip()
    .str.lower()
)


before_cleaning = len(df_b)


df_b = df_b.drop_duplicates(
    subset="_normalized_text"
).drop(
    columns="_normalized_text"
).reset_index(
    drop=True
)


print()
print(
    f"Duplicate temizlendi: "
    f"{before_cleaning - len(df_b)}"
)

print(
    f"Temiz Dataset B: "
    f"{len(df_b)}"
)


# ============================================================
# 7. DATASET B TRAIN / TEST
# ============================================================

b_train, b_test = train_test_split(
    df_b,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=df_b["label"]
)


print()
print("=" * 70)
print("DATASET B SPLIT")
print("=" * 70)


print(
    f"B Train: {len(b_train)}"
)

print(
    f"B Test: {len(b_test)}"
)


print()
print(
    f"B Test HAM: "
    f"{(b_test['label'] == 0).sum()}"
)

print(
    f"B Test SPAM: "
    f"{(b_test['label'] == 1).sum()}"
)


# ============================================================
# 8. TEST LIMIT
# ============================================================

if TEST_LIMIT is not None:

    b_test = b_test.head(
        TEST_LIMIT
    ).copy()


print()
print(
    f"LLM tarafından değerlendirilecek "
    f"mesaj sayısı: {len(b_test)}"
)


# ============================================================
# 9. LLM TESTİ
# ============================================================

print()
print("=" * 70)
print("LLM TESTİ")
print("=" * 70)


llm_predictions = []

llm_scores = []

llm_confidences = []

llm_reasons = []


for i, message in enumerate(
    b_test["text"]
):

    print()
    print(
        f"LLM analiz ediyor: "
        f"{i + 1}/{len(b_test)}"
    )

    print(
        f"Mesaj: {str(message)[:100]}"
    )


    try:

        result = analyze_with_llm(
            str(message)
        )


        prediction = (
            result
            .get(
                "prediction",
                "unknown"
            )
            .lower()
            .strip()
        )


        if prediction not in [
            "spam",
            "ham"
        ]:

            prediction = "unknown"


        llm_predictions.append(
            prediction
        )


        llm_scores.append(
            result.get(
                "risk_score",
                0
            )
        )


        llm_confidences.append(
            result.get(
                "confidence",
                0
            )
        )


        llm_reasons.append(
            result.get(
                "reason",
                ""
            )
        )


        print(
            f"Gemini tahmini: "
            f"{prediction.upper()}"
        )

        print(
            f"Risk skoru: "
            f"{result.get('risk_score', 0)}"
        )

        print(
            f"Güven: "
            f"{result.get('confidence', 0)}"
        )


    except Exception as e:

        print(
            f"LLM HATASI: {e}"
        )


        llm_predictions.append(
            "unknown"
        )

        llm_scores.append(
            0
        )

        llm_confidences.append(
            0
        )

        llm_reasons.append(
            str(e)
        )


# ============================================================
# 10. SONUÇLAR DATAFRAME
# ============================================================

results = pd.DataFrame({

    "message": b_test[
        "text"
    ].values,

    "true_label": b_test[
    "label"
].map({
    0: "ham",
    1: "spam"
}).values,

    "llm_prediction": (
        llm_predictions
    ),

    "llm_score": (
        llm_scores
    ),

    "llm_confidence": (
        llm_confidences
    ),

    "llm_reason": (
        llm_reasons
    )

})


# ============================================================
# 11. UNKNOWN KONTROLÜ
# ============================================================

unknown_count = (
    results[
        "llm_prediction"
    ] == "unknown"
).sum()


print()
print("=" * 70)
print("UNKNOWN SONUÇLARI")
print("=" * 70)


print()
print(
    f"Unknown tahmin sayısı: "
    f"{unknown_count}"
)


# ============================================================
# 12. GEÇERLİ TAHMİNLER
# ============================================================

valid_results = results[
    results[
        "llm_prediction"
    ].isin(
        ["ham", "spam"]
    )
].copy()


print()
print(
    f"Geçerli tahmin sayısı: "
    f"{len(valid_results)}"
)


# ============================================================
# 13. PERFORMANS
# ============================================================

if len(valid_results) > 0:

    y_true = valid_results[
    "true_label"
]

    y_pred = valid_results[
        "llm_prediction"
    ]


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


    # ========================================================
    # 14. FINAL METRICS
    # ========================================================

    print()
    print("=" * 70)
    print("GEMINI FINAL METRICS")
    print("=" * 70)


    print()

    print(
        f"Accuracy       : "
        f"{accuracy:.4f}"
    )

    print(
        f"Spam Precision : "
        f"{precision:.4f}"
    )

    print(
        f"Spam Recall    : "
        f"{recall:.4f}"
    )

    print(
        f"Spam F1        : "
        f"{f1:.4f}"
    )


    # ========================================================
    # 15. CLASSIFICATION REPORT
    # ========================================================

    print()
    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)


    print()

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "HAM",
                "SPAM"
            ],
            zero_division=0
        )
    )


    # ========================================================
    # 16. CONFUSION MATRIX
    # ========================================================

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            "ham",
            "spam"
        ]
    )


    print()
    print("=" * 70)
    print("CONFUSION MATRIX")
    print("=" * 70)


    print()

    print(matrix)


    print()

    print(
        "True Ham → Ham:",
        matrix[0][0]
    )

    print(
        "True Ham → Spam "
        "(False Positive):",
        matrix[0][1]
    )

    print(
        "True Spam → Ham "
        "(False Negative):",
        matrix[1][0]
    )

    print(
        "True Spam → Spam:",
        matrix[1][1]
    )


# ============================================================
# 17. TAHMİN DAĞILIMI
# ============================================================

print()
print("=" * 70)
print("LLM TAHMİN DAĞILIMI")
print("=" * 70)


print()

print(
    results[
        "llm_prediction"
    ].value_counts()
)


# ============================================================
# 18. ORTALAMA RİSK SKORU
# ============================================================

print()
print("=" * 70)
print("LLM SKORLARI")
print("=" * 70)


print()

print(
    "Ortalama LLM risk skoru:",
    round(
        results[
            "llm_score"
        ].mean(),
        2
    )
)


print(
    "Ortalama LLM confidence:",
    round(
        results[
            "llm_confidence"
        ].mean(),
        4
    )
)


# ============================================================
# 19. HATALI TAHMİNLER
# ============================================================

if len(valid_results) > 0:

    errors = valid_results[
        valid_results[
            "true_label"
        ] != valid_results[
            "llm_prediction"
        ]
    ]


    print()
    print("=" * 70)
    print("LLM HATALI TAHMİNLER")
    print("=" * 70)


    print()

    print(
        f"Toplam hata: "
        f"{len(errors)}"
    )


    for _, row in errors.head(
        20
    ).iterrows():

        print()
        print("-" * 70)

        print(
            f"Mesaj: "
            f"{row['message']}"
        )

        print(
            f"Gerçek: "
            f"{row['true_label'].upper()}"
        )

        print(
            f"Gemini: "
            f"{row['llm_prediction'].upper()}"
        )

        print(
            f"Risk: "
            f"{row['llm_score']}"
        )

        print(
            f"Confidence: "
            f"{row['llm_confidence']}"
        )

        print(
            f"Neden: "
            f"{row['llm_reason']}"
        )


# ============================================================
# 20. CSV KAYDI
# ============================================================

results.to_csv(
    OUTPUT_PATH,
    index=False,
    encoding="utf-8-sig"
)


print()
print("=" * 70)
print("RESULTS SAVED")
print("=" * 70)


print()

print(
    f"Sonuçlar kaydedildi:"
)

print(
    OUTPUT_PATH
)


# ============================================================
# FINAL
# ============================================================

print()
print("=" * 70)
print("GEMINI EVALUATION COMPLETED")
print("=" * 70)