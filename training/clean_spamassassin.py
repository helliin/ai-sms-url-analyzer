import pandas as pd

INPUT_PATH = "data/spamassassin_raw.csv"
OUTPUT_PATH = "data/spamassassin_clean.csv"

print("=" * 60)
print("SPAMASSASSIN DATASET TEMİZLEME")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"\nİlk kayıt sayısı: {len(df)}")

# Boş metinleri kaldır
before = len(df)

df["text"] = df["text"].fillna("").astype(str).str.strip()

df = df[df["text"] != ""].copy()

removed_empty = before - len(df)

# Etiketleri standartlaştır
df["label"] = df["label"].str.lower().str.strip()

# Sadece ham ve spam etiketlerini tut
df = df[df["label"].isin(["ham", "spam"])].copy()

# Aynı e-posta metninin tekrarlarını kontrol et
duplicate_count = df.duplicated(subset=["text"]).sum()

print(f"\nBoş metin nedeniyle çıkarılan: {removed_empty}")
print(f"Tekrarlanan text sayısı: {duplicate_count}")

# Duplicate kayıtları kaldır
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)

print(f"\nTemizleme sonrası kayıt sayısı: {len(df)}")

print("\nEtiket dağılımı:")
print(df["label"].value_counts())

# CSV olarak kaydet
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nTemiz dataset kaydedildi:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("TEMİZLEME TAMAMLANDI")
print("=" * 60)