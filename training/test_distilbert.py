import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
)


# ============================================================
# AYARLAR
# ============================================================

MODEL_PATH = "models/distilbert_combined"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MODELİ YÜKLE
# ============================================================

print("=" * 70)
print("DISTILBERT SMS TEST")
print("=" * 70)

print()
print(f"Device: {DEVICE}")

print()
print("DistilBERT yükleniyor...")


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH
)

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_PATH
)

model.to(DEVICE)
model.eval()


print("Model başarıyla yüklendi.")


# ============================================================
# TEST FONKSİYONU
# ============================================================

def analyze_message(message):

    encoding = tokenizer(
        message,
        truncation=True,
        padding=True,
        max_length=128,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        probabilities = torch.softmax(
            outputs.logits,
            dim=1
        )

        prediction = torch.argmax(
            outputs.logits,
            dim=1
        ).item()

    if prediction == 1:
        label = "SPAM"
    else:
        label = "HAM"

    ham_probability = probabilities[0][0].item()
    spam_probability = probabilities[0][1].item()

    return label, ham_probability, spam_probability


# ============================================================
# TEST MESAJLARI
# ============================================================

test_messages = [

    "Congratulations! You have won a free prize! Call now!",

    "Hey, are we still meeting for lunch today?",

    "URGENT! You have won $1000. Click this link to claim your prize.",

    "Can you send me the homework when you get home?",

    "Your account has been selected for a special reward. Verify now!",

]


# ============================================================
# TESTLERİ ÇALIŞTIR
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)


for i, message in enumerate(
    test_messages,
    start=1
):

    label, ham_probability, spam_probability = (
        analyze_message(message)
    )

    print()
    print(f"Test {i}")
    print("-" * 70)

    print(f"Mesaj: {message}")

    print(f"Tahmin: {label}")

    print(
        f"HAM olasılığı : {ham_probability:.4f}"
    )

    print(
        f"SPAM olasılığı: {spam_probability:.4f}"
    )


# ============================================================
# INTERACTIVE TEST
# ============================================================

print()
print("=" * 70)
print("KENDİ SMS'İNİ TEST ET")
print("=" * 70)

message = input(
    "\nTest etmek istediğin SMS'i yaz: "
)


if message.strip():

    label, ham_probability, spam_probability = (
        analyze_message(message)
    )

    print()
    print("-" * 70)

    print(
        f"Tahmin: {label}"
    )

    print(
        f"HAM olasılığı : {ham_probability:.4f}"
    )

    print(
        f"SPAM olasılığı: {spam_probability:.4f}"
    )

else:

    print()
    print("Mesaj girilmedi.")


print()
print("=" * 70)
print("TEST COMPLETED")
print("=" * 70)