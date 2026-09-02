import re


RISK_PATTERNS = {
    "urgency": [
        "acil",
        "hemen",
        "son şans",
        "derhal",
        "24 saat",
        "bugün"
    ],

    "financial_request": [
        "kart bilgileri",
        "kart numarası",
        "banka bilgileri",
        "para gönder",
        "ödeme yap",
        "havale yap"
    ],

    "personal_info_request": [
        "şifrenizi",
        "şifreni",
        "tc kimlik",
        "kimlik numarası",
        "kişisel bilgilerinizi",
        "doğrulama kodu"
    ],

    "impersonation": [
        "banka",
        "vergi dairesi",
        "e-devlet",
        "ptt",
        "kargo",
        "polis"
    ],

    "reward_scam": [
        "ödül kazandınız",
        "ödülünüz",
        "tebrikler kazandınız",
        "hediye kazandınız",
        "çekiliş kazandınız",
        "bedava"
    ],

    "threat_or_penalty": [
        "hesabınız kapatılacak",
        "hesabın kapatılacak",
        "yasal işlem",
        "ceza",
        "borcunuz",
        "borcunuz bulunmaktadır"
    ]
}


def analyze_rules(message: str) -> dict:
    """
    SMS'i kural tabanlı olarak analiz eder.
    """

    message_lower = message.lower()

    detected_categories = []
    matched_patterns = []

    for category, patterns in RISK_PATTERNS.items():
        for pattern in patterns:
            if pattern in message_lower:
                if category not in detected_categories:
                    detected_categories.append(category)

                matched_patterns.append(pattern)

    risk_score = min(len(detected_categories) * 15, 100)

    return {
        "risk_score": risk_score,
        "detected_categories": detected_categories,
        "matched_patterns": matched_patterns
    }
