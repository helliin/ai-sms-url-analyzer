import pandas as pd
from pathlib import Path


DATASET_DIR = Path("data/sms/ExAIS_SMS Spam Dataset")


all_labels = set()

csv_files = sorted(DATASET_DIR.glob("*.csv"))

print(f"Bulunan CSV dosyası sayısı: {len(csv_files)}")

for file in csv_files:
    print(f"\nKontrol ediliyor: {file.name}")

    df = pd.read_csv(
        file,
        header=None
    )

    print(f"Satır sayısı: {len(df)}")
    print(f"Sütun sayısı: {len(df.columns)}")

    # 7. sütun = label
    labels = (
        df.iloc[:, 6]
        .dropna()
        .astype(str)
        .str.strip()
        .str.upper()
    )

    all_labels.update(labels.unique())


print("\n" + "=" * 50)
print("DATASET B LABEL'LARI")
print("=" * 50)

for label in sorted(all_labels):
    print(label)