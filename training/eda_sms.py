import pandas as pd

file_path = "data/sms/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

print("İlk 5 veri:")
print(df.head())

print("\nDataset boyutu:")
print(df.shape)

print("\nSütunlar:")
print(df.columns)

print("\nEtiket dağılımı:")
print(df["label"].value_counts())
# URL içeren mesajlar
url_pattern = r"(http[s]?://|www\.|\.com|\.net|\.org)"

df["has_url"] = df["message"].str.contains(
    url_pattern,
    case=False,
    regex=True
)

print("\nURL içeren mesaj sayısı:")
print(df["has_url"].sum())

print("\nSpam mesajlarda URL:")
print(df[df["label"] == "spam"]["has_url"].value_counts())

print("\nNormal mesajlarda URL:")
print(df[df["label"] == "ham"]["has_url"].value_counts())

# Duplicate mesajlar
print("\nDuplicate mesaj sayısı:")
print(df["message"].duplicated().sum())
# Duplicate mesajları kontrol et
duplicate_count = df["message"].duplicated().sum()

print("\nDuplicate mesaj sayısı:")
print(duplicate_count)

# Duplicate mesajları kaldır
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

print("\nDuplicate temizlendikten sonra dataset boyutu:")
print(df.shape)

print("\nYeni etiket dağılımı:")
print(df["label"].value_counts())
from sklearn.model_selection import train_test_split

# Mesajlar ve etiketler
X = df["message"]
y = df["label"]

# Train / Test ayırma
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain veri boyutu:")
print(X_train.shape)

print("\nTest veri boyutu:")
print(X_test.shape)

print("\nTrain etiket dağılımı:")
print(y_train.value_counts())

print("\nTest etiket dağılımı:")
print(y_test.value_counts())