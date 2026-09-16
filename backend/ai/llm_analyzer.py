import os
import json
import re
import urllib.request
import urllib.error

from dotenv import load_dotenv
from google import genai


# ============================================================
# 1. ENVIRONMENT
# ============================================================

load_dotenv()

LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "local"
).lower()

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "qwen3:4b"
)

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate"
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


# ============================================================
# 2. GEMINI CLIENT
# ============================================================

client = None

if GEMINI_API_KEY:
    client = genai.Client(
        api_key=GEMINI_API_KEY
    )


# ============================================================
# 3. COMMON PROMPT
# ============================================================

def create_prompt(message: str) -> str:
    return f"""
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
    "prediction": "spam",
    "risk_score": 0,
    "confidence": 0,
    "reason": "kısa açıklama",
    "risk_factors": ["faktör1", "faktör2"]
}}

Kurallar:

- prediction sadece "spam" veya "ham" olabilir.
- risk_score 0 ile 100 arasında olmalıdır.
- confidence 0 ile 1 arasında olmalıdır.
- reason kısa olmalıdır.
- risk_factors bir liste olmalıdır.
- JSON dışında hiçbir açıklama yazma.
- Markdown veya kod bloğu kullanma.
"""


# ============================================================
# 4. JSON PARSER
# ============================================================

def parse_llm_result(result: str) -> dict:
    result = result.strip()

    if result.startswith("```"):
        result = result.replace("```json", "")
        result = result.replace("```", "")
        result = result.strip()

    if not result.startswith("{"):
        start = result.find("{")
        end = result.rfind("}")

        if start != -1 and end != -1:
            result = result[start:end + 1]

    try:
        parsed_result = json.loads(result)

        prediction = str(
            parsed_result.get(
                "prediction",
                "unknown"
            )
        ).lower()

        if prediction not in ["spam", "ham"]:
            prediction = "unknown"

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

        reason = str(
            parsed_result.get(
                "reason",
                ""
            )
        )

        risk_factors = parsed_result.get(
            "risk_factors",
            []
        )

        if not isinstance(risk_factors, list):
            risk_factors = [str(risk_factors)]

        risk_score = max(
            0,
            min(
                100,
                risk_score
            )
        )

        confidence = max(
            0,
            min(
                1,
                confidence
            )
        )

        return {
            "prediction": prediction,
            "risk_score": round(risk_score, 2),
            "confidence": round(confidence, 4),
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


# ============================================================
# 5. LOCAL QWEN / OLLAMA
# ============================================================

def classify_with_local_llm(message: str) -> str:
    """
    Evaluation için Qwen'den spam/ham tahmini alır.

    Few-shot örnekleri kullanarak Qwen'in özellikle
    HAM mesajları SPAM olarak sınıflandırma eğilimini azaltır.
    """

    prompt = f"""
You are an SMS spam classification system.

Classify the SMS as exactly one of:

SPAM
HAM

Important rules:

1. Personal conversations between people are HAM.
2. Normal questions, greetings, directions, appointments,
   work conversations and requests for contact information are HAM.
3. Promotional, fraudulent, phishing, scam or unsolicited
   commercial messages are SPAM.
4. A message containing words such as "free", "credit",
   "experience", "call", "number" or "balance" is NOT
   automatically SPAM. Consider the whole message.
5. Very short or unclear personal messages should normally
   be classified as HAM unless there is clear evidence of spam.
6. Do not assume that every advertisement-like message is spam.
7. Return only valid JSON.

Examples:

SMS: "Hey, are you coming home tonight?"
Answer:
{{"label":"HAM"}}

SMS: "Morning sir. Can u kindly send me Dr. Ezobo's number?"
Answer:
{{"label":"HAM"}}

SMS: "The bus comes to Monroeville. Call me when you arrive."
Answer:
{{"label":"HAM"}}

SMS: "Tnks plant t.v"
Answer:
{{"label":"HAM"}}

SMS: "Congratulations! You have won a FREE prize. Click here now!"
Answer:
{{"label":"SPAM"}}

SMS: "You have won $5000. Claim your reward immediately."
Answer:
{{"label":"SPAM"}}

SMS: "Your account has been selected for a special cash reward. Reply now."
Answer:
{{"label":"SPAM"}}

Now classify this SMS:

SMS: "{message}"

Return ONLY this JSON format:

{{"label":"SPAM"}}

or

{{"label":"HAM"}}
"""

    payload = {
        "model": OLLAMA_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a binary SMS spam classifier. "
                    "Return only JSON with a label field. "
                    "The label must be SPAM or HAM."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "think": False,
        "format": {
            "type": "object",
            "properties": {
                "label": {
                    "type": "string",
                    "enum": ["SPAM", "HAM"]
                }
            },
            "required": ["label"]
        },
        "options": {
            "temperature": 0,
            "num_predict": 64
        }
    }

    data = json.dumps(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        "http://localhost:11434/api/chat",
        data=data,
        headers={
            "Content-Type": "application/json"
        },
        method="POST"
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=120
        ) as response:

            response_data = json.loads(
                response.read().decode("utf-8")
            )

        message_data = response_data.get(
            "message",
            {}
        )

        raw_response = message_data.get(
            "content",
            ""
        )

        print(
            "QWEN RAW RESPONSE:",
            repr(raw_response)
        )

        # JSON parse
        try:

            parsed = json.loads(
                raw_response
            )

            label = str(
                parsed.get(
                    "label",
                    ""
                )
            ).strip().lower()

            if label in ["spam", "ham"]:
                return label

        except (
            json.JSONDecodeError,
            TypeError,
            AttributeError
        ):
            pass

        # Tam eşleşen satırları kontrol et
        lines = [
            line.strip().lower()
            for line in raw_response.splitlines()
            if line.strip()
        ]

        for line in lines:

            if line == "spam":
                return "spam"

            if line == "ham":
                return "ham"

        # JSON içindeki label alanını ara
        match = re.search(
            r'"label"\s*:\s*"(spam|ham)"',
            raw_response.lower()
        )

        if match:
            return match.group(1)

        return "unknown"

    except urllib.error.URLError as e:

        print(
            f"Qwen/Ollama bağlantı hatası: {e}"
        )

        return "unknown"

    except Exception as e:

        print(
            f"Qwen classification hatası: {e}"
        )

        return "unknown"
    # ============================================================
# 6. LLM ANA ANALİZ FONKSİYONU
# ============================================================

def analyze_with_llm(message: str) -> dict:
    """
    SMS/mail içeriğini seçili LLM ile analiz eder.

    LLM_PROVIDER:
    - local  -> Ollama / Qwen
    - gemini -> Gemini API
    """

    # --------------------------------------------------------
    # LOCAL QWEN
    # --------------------------------------------------------

    if LLM_PROVIDER == "local":

        prediction = classify_with_local_llm(message)

        if prediction == "spam":
            return {
                "prediction": "spam",
                "risk_score": 80,
                "confidence": 0.80,
                "reason": "Qwen mesajı spam olarak değerlendirdi.",
                "risk_factors": []
            }

        if prediction == "ham":
            return {
                "prediction": "ham",
                "risk_score": 10,
                "confidence": 0.80,
                "reason": "Qwen mesajı normal olarak değerlendirdi.",
                "risk_factors": []
            }

        return {
            "prediction": "unknown",
            "risk_score": 0,
            "confidence": 0,
            "reason": "Qwen analiz sonucu alınamadı.",
            "risk_factors": []
        }

    # --------------------------------------------------------
    # GEMINI
    # --------------------------------------------------------

    if LLM_PROVIDER == "gemini":

        if client is None:
            return {
                "prediction": "unknown",
                "risk_score": 0,
                "confidence": 0,
                "reason": "Gemini API anahtarı bulunamadı.",
                "risk_factors": []
            }

        try:

            prompt = create_prompt(message)

            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )

            raw_result = response.text

            return parse_llm_result(raw_result)

        except Exception as e:

            print(
                f"Gemini analiz hatası: {e}"
            )

            return {
                "prediction": "unknown",
                "risk_score": 0,
                "confidence": 0,
                "reason": "Gemini analizi sırasında hata oluştu.",
                "risk_factors": []
            }

    # --------------------------------------------------------
    # BİLİNMEYEN PROVIDER
    # --------------------------------------------------------

    return {
        "prediction": "unknown",
        "risk_score": 0,
        "confidence": 0,
        "reason": f"Bilinmeyen LLM provider: {LLM_PROVIDER}",
        "risk_factors": []
    }