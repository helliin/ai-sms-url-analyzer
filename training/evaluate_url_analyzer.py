import sys
from pathlib import Path
import re

import pandas as pd
from sklearn.model_selection import train_test_split


# Proje ana klasörünü Python path'ine ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

from backend.ai.url_analyzer import analyze_url


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


# Train / Test ayır
X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


# URL bulmak için kullanılacak pattern
URL_PATTERN = r"https?://\S+|www\.\S+"


def extract_urls(message):
    """
    SMS içindeki URL'leri bulur.
    """
    return re.findall(URL_PATTERN, message)


def analyze_message_urls(message):
    """
    Mesajdaki URL'leri analiz eder.
    Birden fazla URL varsa en yüksek risk skorunu döndürür.
    """

    urls = extract_urls(message)

    if not urls:
        return {
            "url_count": 0,
            "max_risk_score": 0,
            "risk_factors": []
        }

    url_results = []

    for url in urls:
        # Noktalama işaretlerini temizle
        url = url.rstrip(".,!?;:)]}")

        result = analyze_url(url)
        url_results.append(result)

    # En yüksek riskli URL'yi bul
    highest_risk_result = max(
        url_results,
        key=lambda result: result["risk_score"]
    )

    return {
        "url_count": len(urls),
        "max_risk_score": highest_risk_result["risk_score"],
        "risk_factors": highest_risk_result["risk_factors"]
    }


# Test mesajlarına URL Analyzer uygula
url_results = X_test.apply(analyze_message_urls)


# Sonuçları DataFrame'e koy
results = pd.DataFrame({
    "message": X_test,
    "true_label": y_test,
    "url_count": url_results.apply(
        lambda result: result["url_count"]
    ),
    "url_score": url_results.apply(
        lambda result: result["max_risk_score"]
    )
})


print("Test veri sayısı:")
print(len(results))


print("\nURL içeren test mesajı sayısı:")
print((results["url_count"] > 0).sum())


print("\nURL içermeyen test mesajı sayısı:")
print((results["url_count"] == 0).sum())


print("\nRisk skoru dağılımı:")
print(
    results["url_score"]
    .value_counts()
    .sort_index()
)


print("\nGerçek HAM mesajların URL risk skorları:")
print(
    results[
        results["true_label"] == "ham"
    ]["url_score"]
    .value_counts()
    .sort_index()
)


print("\nGerçek SPAM mesajların URL risk skorları:")
print(
    results[
        results["true_label"] == "spam"
    ]["url_score"]
    .value_counts()
    .sort_index()
)