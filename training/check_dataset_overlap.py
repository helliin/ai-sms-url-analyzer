import pandas as pd
import re


# ============================================================
# 1. DATASETLERİ YÜKLE
# ============================================================

print("=" * 70)
print("DATASET OVERLAP / DATA LEAKAGE CHECK")
print("=" * 70)


# Dataset A
df_a = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "text"]
)

df_a = df_a.drop_duplicates()


# Dataset B
df_b = pd.read_csv(
    "data/sms/exais_clean.csv"
)


print()
print(f"Dataset A toplam mesaj : {len(df_a)}")
print(f"Dataset B toplam mesaj : {len(df_b)}")


# ============================================================
# 2. MESAJLARI TEMİZLE
# ============================================================

def normalize_text(text):
    """
    Mesajları karşılaştırmadan önce
    küçük harfe çevirir ve gereksiz boşlukları temizler.
    """

    text = str(text).lower()

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


df_a["normalized_text"] = df_a["text"].apply(
    normalize_text
)

df_b["normalized_text"] = df_b["text"].apply(
    normalize_text
)


# ============================================================
# 3. DATASET A VE B'DE ORTAK MESAJLARI BUL
# ============================================================

a_messages = set(
    df_a["normalized_text"]
)

b_messages = set(
    df_b["normalized_text"]
)


common_messages = (
    a_messages.intersection(b_messages)
)


print()
print("=" * 70)
print("DATASET A <-> DATASET B OVERLAP")
print("=" * 70)

print()
print(
    f"A ve B arasında ortak mesaj sayısı: "
    f"{len(common_messages)}"
)


# ============================================================
# 4. ORTAK MESAJ ORANLARI
# ============================================================

if len(df_a) > 0:

    overlap_rate_a = (
        len(common_messages) /
        len(df_a)
    ) * 100

else:

    overlap_rate_a = 0


if len(df_b) > 0:

    overlap_rate_b = (
        len(common_messages) /
        len(df_b)
    ) * 100

else:

    overlap_rate_b = 0


print(
    f"Dataset A'ya göre overlap oranı: "
    f"{overlap_rate_a:.2f}%"
)

print(
    f"Dataset B'ye göre overlap oranı: "
    f"{overlap_rate_b:.2f}%"
)


# ============================================================
# 5. DATASET B'Yİ AYNI ŞEKİLDE TRAIN / TEST AYIR
# ============================================================

from sklearn.model_selection import train_test_split


b_train, b_test = train_test_split(
    df_b,
    test_size=0.20,
    random_state=42,
    stratify=df_b["label"]
)


print()
print("=" * 70)
print("DATASET B TRAIN / TEST")
print("=" * 70)

print()
print(f"B Train: {len(b_train)}")
print(f"B Test : {len(b_test)}")


# ============================================================
# 6. TEST SETİNDE DATASET A'DAN GELEN MESAJLARI BUL
# ============================================================

a_message_set = set(
    df_a["normalized_text"]
)

b_test_messages = (
    b_test["normalized_text"]
)


test_overlap = b_test_messages.isin(
    a_message_set
)


test_overlap_count = test_overlap.sum()


print()
print("=" * 70)
print("TEST SET OVERLAP")
print("=" * 70)

print()
print(
    f"B Test içinde Dataset A'da bulunan mesaj: "
    f"{test_overlap_count}"
)


# ============================================================
# 7. TEST OVERLAP ORANI
# ============================================================

if len(b_test) > 0:

    test_overlap_rate = (
        test_overlap_count /
        len(b_test)
    ) * 100

else:

    test_overlap_rate = 0


print(
    f"B Test overlap oranı: "
    f"{test_overlap_rate:.2f}%"
)


# ============================================================
# 8. ÖRNEK ORTAK MESAJLARI GÖSTER
# ============================================================

print()
print("=" * 70)
print("ORTAK MESAJ ÖRNEKLERİ")
print("=" * 70)


common_list = list(common_messages)


if len(common_list) == 0:

    print()
    print("Ortak mesaj bulunamadı.")

else:

    for i, message in enumerate(
        common_list[:10],
        start=1
    ):

        print()
        print(f"{i}. {message}")


# ============================================================
# 9. TEST OVERLAP ÖRNEKLERİ
# ============================================================

print()
print("=" * 70)
print("TEST SETİNDEKİ OVERLAP ÖRNEKLERİ")
print("=" * 70)


overlap_test_rows = b_test[
    test_overlap
]


if len(overlap_test_rows) == 0:

    print()
    print("Test setinde Dataset A ile ortak mesaj yok.")

else:

    for i, (_, row) in enumerate(
        overlap_test_rows.head(10).iterrows(),
        start=1
    ):

        print()
        print(f"{i}. {row['text']}")
        print(
            f"Label: "
            f"{'SPAM' if row['label'] == 1 else 'HAM'}"
        )


# ============================================================
# 10. FINAL DEĞERLENDİRME
# ============================================================

print()
print("=" * 70)
print("FINAL RESULT")
print("=" * 70)

print()

if test_overlap_count == 0:

    print(
        "✓ B Test setinde Dataset A ile ortak mesaj bulunamadı."
    )

    print(
        "✓ Bu açıdan belirgin bir data leakage görünmüyor."
    )

else:

    print(
        "⚠ B Test setinde Dataset A ile ortak mesajlar bulundu."
    )

    print(
        "⚠ Model karşılaştırmasında data leakage riski var."
    )


print()
print("=" * 70)
print("CHECK COMPLETED")
print("=" * 70)