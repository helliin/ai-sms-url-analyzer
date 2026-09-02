def analyze_with_mock_llm(message: str) -> dict:
    """
    Gerçek LLM yerine kullanılan test amaçlı mock analiz.
    """

    message_lower = message.lower()

    risk_score = 0
    reasons = []

    # Hassas bilgi isteme
    sensitive_patterns = [
        "password",
        "pin",
        "verification code",
        "security code",
        "card number",
        "bank details",
        "account details",
        "verify your account"
    ]

    for pattern in sensitive_patterns:
        if pattern in message_lower:
            risk_score += 30
            reasons.append(
                f"sensitive information request: {pattern}"
            )

    # Aciliyet
    urgency_patterns = [
        "urgent",
        "immediately",
        "act now",
        "expires today",
        "within 24 hours",
        "as soon as possible"
    ]

    for pattern in urgency_patterns:
        if pattern in message_lower:
            risk_score += 20
            reasons.append(
                f"urgency: {pattern}"
            )

    # Ödül / dolandırıcılık
    reward_patterns = [
        "you won",
        "winner",
        "free prize",
        "congratulations",
        "claim your prize"
    ]

    for pattern in reward_patterns:
        if pattern in message_lower:
            risk_score += 25
            reasons.append(
                f"reward/scam indicator: {pattern}"
            )

    # Tehdit / hesap kapatma
    threat_patterns = [
        "account will be suspended",
        "account will be closed",
        "legal action",
        "your account is blocked",
        "penalty"
    ]

    for pattern in threat_patterns:
        if pattern in message_lower:
            risk_score += 25
            reasons.append(
                f"threat/penalty: {pattern}"
            )

    # 100'ü geçmesin
    risk_score = min(risk_score, 100)

    if risk_score >= 50:
        decision = "spam"
    else:
        decision = "ham"

    return {
        "decision": decision,
        "risk_score": risk_score,
        "reasons": reasons
    }