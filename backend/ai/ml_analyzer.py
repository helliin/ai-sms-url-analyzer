
import torch

from pathlib import Path

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# PROJE ANA KLASÖRÜ
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent


# ============================================================
# DISTILBERT MODEL YOLU
# ============================================================

MODEL_PATH = BASE_DIR / "models" / "distilbert_spamassassin"


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODEL VE TOKENIZER'I YÜKLE
# ============================================================

print("=" * 70)
print("DISTILBERT ML MODEL")
print("=" * 70)

print(f"Device: {DEVICE}")
print(f"Model: {MODEL_PATH}")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


print("DistilBERT başarıyla yüklendi.")


# ============================================================
# SMS ANALİZİ
# ============================================================

def analyze_ml(message: str) -> dict:
    """
    Eğitilmiş DistilBERT modeli ile SMS'i analiz eder.

    Dönen bilgiler:
        prediction       -> ham / spam
        ham_probability  -> HAM olasılığı
        spam_probability -> SPAM olasılığı
    """

    # --------------------------------------------------------
    # SMS'i DistilBERT formatına dönüştür
    # --------------------------------------------------------

    encoding = tokenizer(
        message,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )


    # --------------------------------------------------------
    # Tensor'ları GPU / CPU'ya gönder
    # --------------------------------------------------------

    input_ids = encoding["input_ids"].to(DEVICE)

    attention_mask = encoding["attention_mask"].to(DEVICE)


    # --------------------------------------------------------
    # MODEL TAHMİNİ
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )


        # Model çıktısını olasılığa dönüştür
        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )


        # En yüksek olasılığa sahip sınıf
        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()


    # --------------------------------------------------------
    # LABEL
    # --------------------------------------------------------

    if prediction == 1:
        label = "spam"
    else:
        label = "ham"


    # --------------------------------------------------------
    # OLASILIKLAR
    # --------------------------------------------------------

    ham_probability = probabilities[0][0].item()

    spam_probability = probabilities[0][1].item()


    # --------------------------------------------------------
    # SONUÇ
    # --------------------------------------------------------

    return {
        "prediction": label,
        "ham_probability": round(ham_probability, 4),
        "spam_probability": round(spam_probability, 4)
    }

