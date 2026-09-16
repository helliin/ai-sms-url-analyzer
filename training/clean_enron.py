import pandas as pd

INPUT_PATH = "data/enron_spam_data.csv"
OUTPUT_PATH = "data/enron_spam_clean.csv"


print("=" * 60)
print("ENRON DATASET TEMİZLEME")
print("=" * 60)

# Dataseti oku
df = pd.read_csv(INPUT_PATH)

print(f"\nİlk kayıt sayısı: {len(df)}")

# Gereksiz indeks sütununu kaldır
if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])

# Subject ve Message sütunlarını metne çevir
df["Subject"] = df["Subject"].fillna("").astype(str)
df["Message"] = df["Message"].fillna("").astype(str)

# Subject + Message birleştir
df["text"] = (
    df["Subject"].str.strip()
    + " "
    + df["Message"].str.strip()
).str.strip()

# Mesajı tamamen boş olan kayıtları çıkar
before = len(df)

df = df[df["text"].str.len() > 0].copy()

removed_empty = before - len(df)

# Label değerlerini standartlaştır
df["label"] = df["Spam/Ham"].str.lower().str.strip()

# Sadece spam ve ham kayıtlarını tut
df = df[df["label"].isin(["spam", "ham"])].copy()

# Gereksiz eski sütunları kaldır
df = df[["text", "label", "Date"]]

# Duplicate kontrolü ve temizliği
before_duplicates = len(df)

df = df.drop_duplicates(subset=["text", "label"]).reset_index(drop=True)

removed_duplicates = before_duplicates - len(df)

# Temizlenmiş dataset'i kaydet
df.to_csv(OUTPUT_PATH, index=False)

print(f"\nBoş mesaj nedeniyle çıkarılan kayıt: {removed_empty}")
print(f"Duplicate nedeniyle çıkarılan kayıt: {removed_duplicates}")

print(f"\nTemizleme sonrası kayıt sayısı: {len(df)}")

print("\nLabel dağılımı:")
print(df["label"].value_counts())

print("\nSon sütunlar:")
print(df.columns.tolist())

print("\nTemizlenmiş dataset kaydedildi:")
print(OUTPUT_PATH)

print("\n" + "=" * 60)
print("TEMİZLEME TAMAMLANDI")
print("=" * 60)