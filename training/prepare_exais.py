from pathlib import Path

import pandas as pd


# Projenin ana klasörü
BASE_DIR = Path(__file__).resolve().parent.parent

# ExAIS dataset klasörü
DATASET_DIR = BASE_DIR / "data" / "sms" / "ExAIS_SMS Spam Dataset"

# Hazırlanmış Dataset B'nin kaydedileceği yer
OUTPUT_FILE = BASE_DIR / "data" / "sms" / "exais_clean.csv"


def load_exais_dataset():
    csv_files = sorted(DATASET_DIR.glob("*.csv"))

    print(f"Bulunan CSV dosyası: {len(csv_files)}")

    all_data = []

    for file in csv_files:
        try:
            df = pd.read_csv(file)

            # ExAIS dosyalarında:
            # 7. kolon -> label
            # 8. kolon -> SMS metni
            if df.shape[1] < 8:
                print(f"Atlandı: {file.name} (yetersiz kolon)")
                continue

            temp = df.iloc[:, [6, 7]].copy()
            temp.columns = ["label", "text"]

            all_data.append(temp)

            print(
                f"{file.name}: "
                f"{len(temp)} satır"
            )

        except Exception as e:
            print(f"Hata - {file.name}: {e}")

    if not all_data:
        raise ValueError("Hiçbir CSV dosyası okunamadı.")

    combined = pd.concat(all_data, ignore_index=True)

    return combined


def clean_dataset(df):
    # Label ve text eksik olan satırları sil
    df = df.dropna(subset=["label", "text"])

    # Metinleri string'e çevir
    df["text"] = df["text"].astype(str).str.strip()

    # Boş mesajları sil
    df = df[df["text"] != ""]

    # Label'ları normalize et
    df["label"] = (
        df["label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    # Sadece HAM ve SPAM etiketlerini tut
    df = df[df["label"].isin(["HAM", "SPAM"])]

    # Sayısal label
    df["label"] = df["label"].map({
        "HAM": 0,
        "SPAM": 1
    })

    # Duplicate mesajları temizle
    df = df.drop_duplicates(subset=["text"])

    # Index'i yeniden oluştur
    df = df.reset_index(drop=True)

    return df


def main():
    print("=" * 50)
    print("ExAIS Dataset Hazırlama")
    print("=" * 50)

    df = load_exais_dataset()

    print("\nBirleştirme sonrası:")
    print(f"Toplam satır: {len(df)}")

    df = clean_dataset(df)

    print("\nTemizlik sonrası:")
    print(f"Toplam SMS: {len(df)}")

    print("\nLabel dağılımı:")
    print(df["label"].value_counts().sort_index())

    print("\nLabel yüzdeleri:")
    print(
        df["label"]
        .value_counts(normalize=True)
        .sort_index()
        .mul(100)
        .round(2)
    )

    # Sonucu kaydet
    df.to_csv(OUTPUT_FILE, index=False)

    print("\nDataset B kaydedildi:")
    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()