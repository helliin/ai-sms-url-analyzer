import joblib
from pathlib import Path


# Proje ana klasörünü bul
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Kaydedilmiş model ve TF-IDF dosyalarının yolları
MODEL_PATH = BASE_DIR / "models" / "sms_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


# Model ve vectorizer'ı yükle
model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)


def analyze_ml(message: str) -> dict:
    """
    Eğitilmiş Linear SVM modeli ile SMS'i analiz eder.
    """

    # SMS'i TF-IDF formatına dönüştür
    message_tfidf = vectorizer.transform([message])

    # Model tahmini
    prediction = model.predict(message_tfidf)[0]

    # Sonucu okunabilir hale getir
    if prediction == 1:
        label = "spam"
    else:
        label = "ham"

    return {
        "prediction": label
    }