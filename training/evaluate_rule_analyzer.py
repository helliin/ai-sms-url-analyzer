import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# Proje ana klasörünü Python path'ine ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from backend.ai.rule_analyzer_en import analyze_rules_en

# Dataseti oku
df = pd.read_csv(
    "data/sms/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)


# Duplicate mesajları temizle
df = df.drop_duplicates(
    subset=["message"]
).reset_index(drop=True)


# Aynı train/test ayrımını kullan
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# Rule Analyzer'ı test mesajlarına uygula
rule_results = X_test.apply(analyze_rules_en)


# Risk skorlarını ayrı bir sütun olarak oluştur
rule_scores = rule_results.apply(
    lambda result: result["risk_score"]
)


# Sonuçları DataFrame'e koy
results = pd.DataFrame({
    "message": X_test,
    "true_label": y_test,
    "rule_score": rule_scores
})


print("Test veri sayısı:")
print(len(results))


print("\nRisk skoru dağılımı:")
print(results["rule_score"].value_counts().sort_index())


print("\nGerçek HAM mesajların risk skorları:")
print(
    results[results["true_label"] == "ham"]["rule_score"]
    .value_counts()
    .sort_index()
)


print("\nGerçek SPAM mesajların risk skorları:")
print(
    results[results["true_label"] == "spam"]["rule_score"]
    .value_counts()
    .sort_index()
)