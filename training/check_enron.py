import pandas as pd

# Enron datasetini oku
file_path = "data/enron_spam_data.csv"

df = pd.read_csv(file_path)

print("=" * 60)
print("ENRON DATASET KONTROLÜ")
print("=" * 60)

print("\n1. Dataset boyutu:")
print(df.shape)

print("\n2. Sütunlar:")
print(df.columns.tolist())

print("\n3. İlk 5 satır:")
print(df.head())

print("\n4. Eksik değerler:")
print(df.isnull().sum())

print("\n5. Etiket dağılımı:")
print(df["Spam/Ham"].value_counts())

print("\n6. Tekrarlanan satır sayısı:")
print(df.duplicated().sum())

print("\n7. Boş Subject sayısı:")
print(df["Subject"].isna().sum())

print("\n8. Boş Message sayısı:")
print(df["Message"].isna().sum())

print("\n" + "=" * 60)
print("KONTROL TAMAMLANDI")
print("=" * 60)