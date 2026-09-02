import pandas as pd
from sklearn.model_selection import train_test_split


# Dataseti oku
file_path = "data/sms/SMSSpamCollection"

df = pd.read_csv(
    file_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)


# Tekrarlanan mesajları temizle
df = df.drop_duplicates(subset=["message"]).reset_index(drop=True)


# Etiketleri sayısal hale getir
df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})


# Özellik ve hedef değişken
X = df["message"]
y = df["label"]


# Train / Test ayrımı
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("Toplam veri:", len(df))
print("Eğitim verisi:", len(X_train))
print("Test verisi:", len(X_test))

print("\nEğitim seti dağılımı:")
print(y_train.value_counts())

print("\nTest seti dağılımı:")
print(y_test.value_counts())