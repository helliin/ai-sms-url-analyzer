import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/spamassassin_clean.csv"

TRAIN_PATH = "data/spamassassin_train.csv"
VAL_PATH = "data/spamassassin_val.csv"
TEST_PATH = "data/spamassassin_test.csv"

RANDOM_STATE = 42


print("=" * 60)
print("SPAMASSASSIN TRAIN / VALIDATION / TEST SPLIT")
print("=" * 60)

df = pd.read_csv(INPUT_PATH)

print(f"\nToplam kayıt: {len(df)}")
print("\nGenel label dağılımı:")
print(df["label"].value_counts())


# Önce %15 test ayır
train_val, test = train_test_split(
    df,
    test_size=0.15,
    random_state=RANDOM_STATE,
    stratify=df["label"]
)


# Kalan %85'in %17.65'ini validation olarak ayır
# Böylece toplam datasetin yaklaşık %15'i validation olur.
train, val = train_test_split(
    train_val,
    test_size=0.1765,
    random_state=RANDOM_STATE,
    stratify=train_val["label"]
)


train = train.reset_index(drop=True)
val = val.reset_index(drop=True)
test = test.reset_index(drop=True)


train.to_csv(TRAIN_PATH, index=False)
val.to_csv(VAL_PATH, index=False)
test.to_csv(TEST_PATH, index=False)


print("\n" + "=" * 60)
print("SPLIT SONUÇLARI")
print("=" * 60)

print(f"\nTRAIN: {len(train)}")
print(train["label"].value_counts())

print(f"\nVALIDATION: {len(val)}")
print(val["label"].value_counts())

print(f"\nTEST: {len(test)}")
print(test["label"].value_counts())

print("\nDosyalar:")
print(TRAIN_PATH)
print(VAL_PATH)
print(TEST_PATH)

print("\n" + "=" * 60)
print("SPLIT TAMAMLANDI")
print("=" * 60)