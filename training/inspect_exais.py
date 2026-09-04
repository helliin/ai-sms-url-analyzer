import pandas as pd
from pathlib import Path


DATASET_DIR = Path("data/sms/ExAIS_SMS Spam Dataset")

csv_files = sorted(DATASET_DIR.glob("*.csv"))


for file in csv_files:

    print("\n" + "=" * 70)
    print(file.name)
    print("=" * 70)

    df = pd.read_csv(
        file,
        header=None
    )

    print("Sütun sayısı:", len(df.columns))

    print("\nİlk 3 satır:")
    print(df.head(3).to_string())
    