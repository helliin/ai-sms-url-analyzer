import sys
from pathlib import Path


# Proje ana klasörünü Python path'ine ekle
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))


from backend.ai.mock_llm import analyze_with_mock_llm


# ============================================================
# TEST MESAJLARI
# ============================================================

test_messages = [

    "Hello, how are you today?",

    "Your account will be suspended. Verify your account immediately.",

    "Congratulations! You won a free prize. Claim your prize now.",

    "Please send your card number and security code.",

    "Can you meet me at 5 pm?",

    "Your account is blocked. Legal action will be taken."
]


# ============================================================
# MOCK LLM TESTİ
# ============================================================

for message in test_messages:

    result = analyze_with_mock_llm(message)

    print("\n===================================")

    print("SMS:")
    print(message)

    print("\nLLM Decision:")
    print(result["decision"])

    print("LLM Risk Score:")
    print(result["risk_score"])

    print("LLM Reasons:")
    print(result["reasons"])