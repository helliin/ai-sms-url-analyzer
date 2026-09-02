import os
import json

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY bulunamadı. "
        ".env dosyanızı kontrol edin."
    )


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# 3. LLM ANALYZER
# ============================================================

def analyze_with_llm(message: str) -> dict:
    """
    SMS'i Gemini LLM ile analiz eder.
    """

    prompt = f"""
Sen bir SMS güvenlik analiz asistanısın.

Görevin, verilen SMS'in spam, phishing veya smishing
riski taşıyıp taşımadığını değerlendirmektir.

SMS:
{message}

Aşağıdaki kriterleri değerlendir:

- Kullanıcı üzerinde aciliyet veya baskı oluşturuyor mu?
- Kullanıcıdan para veya ödeme istiyor mu?
- Şifre, kart bilgisi, kimlik bilgisi veya doğrulama kodu
  gibi hassas bilgiler istiyor mu?
- Banka, kargo, devlet kurumu veya başka bir kuruluşu
  taklit ediyor mu?
- Ödül, hediye veya kazanç vaadiyle kullanıcıyı
  kandırmaya çalışıyor mu?
- Şüpheli bir bağlantı veya yönlendirme içeriyor mu?
- Genel olarak sosyal mühendislik veya dolandırıcılık
  belirtisi taşıyor mu?

Mesajın tamamını ve bağlamını değerlendir.

Sadece aşağıdaki JSON formatında cevap ver:

{{
    "prediction": "spam" veya "ham",
    "risk_score": 0 ile 100 arasında sayı,
    "confidence": 0 ile 1 arasında sayı,
    "reason": "kısa açıklama",
    "risk_factors": ["faktör1", "faktör2"]
}}

Başka hiçbir açıklama, markdown veya kod bloğu ekleme.
"""


    # ========================================================
    # 4. GEMINI INTERACTIONS API
    # ========================================================

    interaction = client.interactions.create(
        model="gemini-3.6-flash",
        input=prompt
    )


    # ========================================================
    # 5. RESPONSE
    # ========================================================

    result = interaction.output_text.strip()


    # Gemini JSON'u markdown içinde döndürürse temizle
    if result.startswith("```"):
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()


    # ========================================================
    # 6. JSON PARSE
    # ========================================================

    try:

        parsed_result = json.loads(result)

        prediction = parsed_result.get(
            "prediction",
            "unknown"
        )

        risk_score = float(
            parsed_result.get(
                "risk_score",
                0
            )
        )

        confidence = float(
            parsed_result.get(
                "confidence",
                0
            )
        )

        reason = parsed_result.get(
            "reason",
            ""
        )

        risk_factors = parsed_result.get(
            "risk_factors",
            []
        )


        # Risk score 0-100 arasında olsun

        risk_score = max(
            0,
            min(
                100,
                risk_score
            )
        )


        # Confidence 0-1 arasında olsun

        confidence = max(
            0,
            min(
                1,
                confidence
            )
        )


        return {
            "prediction": prediction,
            "risk_score": risk_score,
            "confidence": confidence,
            "reason": reason,
            "risk_factors": risk_factors
        }


    except (
        json.JSONDecodeError,
        TypeError,
        ValueError,
        AttributeError
    ):

        return {
            "prediction": "unknown",
            "risk_score": 0,
            "confidence": 0,
            "reason": "LLM çıktısı beklenen JSON formatında değil.",
            "risk_factors": []
        }