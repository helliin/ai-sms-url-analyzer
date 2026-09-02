RISK_PATTERNS = {
    "urgency": [
        "urgent",
        "immediately",
        "right away",
        "act now",
        "last chance",
        "within 24 hours",
        "today"
    ],

    "financial_request": [
        "card details",
        "card number",
        "bank details",
        "send money",
        "make a payment",
        "transfer money"
    ],

    "personal_info_request": [
        "your password",
        "password",
        "social security number",
        "personal information",
        "verification code",
        "security code"
    ],

    "impersonation": [
        "bank",
        "government",
        "paypal",
        "amazon",
        "police",
        "delivery",
        "courier"
    ],

    "reward_scam": [
        "you won",
        "winner",
        "congratulations",
        "free prize",
        "claim your prize",
        "gift",
        "reward"
    ],

    "threat_or_penalty": [
        "account will be closed",
        "account suspended",
        "legal action",
        "penalty",
        "fine",
        "debt"
    ]
}


def analyze_rules_en(message: str) -> dict:
    """
    İngilizce SMS'i kural tabanlı olarak analiz eder.
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