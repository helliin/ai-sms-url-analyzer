# ============================================================
# RULE ANALYZER
# ============================================================

# ============================================================
# TÜRKÇE RİSK KURALLARI
# ============================================================

TR_RISK_PATTERNS = {
    "urgency": [
        "acil işlem yapın",
        "hemen işlem yapın",
        "derhal işlem yapın",
        "son şans",
        "24 saat içinde",
        "hemen doğrulayın",
        "acilen doğrulayın"
    ],

    "financial_request": [
        "kart bilgilerinizi girin",
        "kart numaranızı girin",
        "banka bilgilerinizi girin",
        "para gönderin",
        "ödeme yapın",
        "havale yapın"
    ],

    "personal_info_request": [
        "şifrenizi girin",
        "şifrenizi paylaşın",
        "tc kimlik numaranızı girin",
        "kimlik numaranızı girin",
        "kişisel bilgilerinizi girin",
        "doğrulama kodunuzu girin",
        "doğrulama kodunu paylaşın"
    ],

    "impersonation": [
        "banka hesabınız",
        "banka hesabınızla",
        "banka tarafından",
        "vergi dairesi tarafından",
        "e-devlet hesabınız",
        "ptt tarafından",
        "polis tarafından"
    ],

    "reward_scam": [
        "ödül kazandınız",
        "ödülünüzü almak için",
        "tebrikler kazandınız",
        "hediyenizi almak için",
        "çekiliş kazandınız",
        "bedava ödül"
    ],

    "threat_or_penalty": [
        "hesabınız kapatılacak",
        "hesabınız askıya alınacak",
        "hesabınız bloke edilecek",
        "yasal işlem başlatılacak",
        "ceza uygulanacak",
        "borcunuz bulunmaktadır",
        "borcunuzu ödeyin"
    ]
}


# ============================================================
# İNGİLİZCE RİSK KURALLARI
# ============================================================

EN_RISK_PATTERNS = {
    "urgency": [
        "act immediately",
        "act now",
        "right away",
        "last chance",
        "within 24 hours",
        "verify immediately",
        "urgent action required",
        "immediate action required"
    ],

    "financial_request": [
        "enter your card details",
        "enter your card number",
        "provide your bank details",
        "send money",
        "make a payment",
        "transfer money",
        "confirm your payment"
    ],

    "personal_info_request": [
        "enter your password",
        "provide your password",
        "enter your social security number",
        "provide your personal information",
        "enter your verification code",
        "share your security code"
    ],

    "impersonation": [
        "your bank account",
        "your bank account has",
        "your account at the bank",
        "government agency",
        "paypal account",
        "amazon account",
        "police department"
    ],

    "reward_scam": [
        "you won a prize",
        "you have won a prize",
        "you won the prize",
        "claim your prize",
        "claim your reward",
        "you have won a reward",
        "free prize"
    ],

    "threat_or_penalty": [
        "your account will be closed",
        "your account will be suspended",
        "your account has been suspended",
        "legal action will be taken",
        "you will be fined",
        "pay your debt",
        "outstanding debt"
    ]
}


# ============================================================
# ORTAK KURAL ANALİZİ
# ============================================================

def _analyze_with_patterns(message: str, patterns: dict) -> dict:
    """
    Verilen pattern listesine göre mesajı analiz eder.

    Tek başına masum olabilecek kelimeler yerine,
    daha güçlü ve bağlamsal ifadeler kullanılır.
    """

    message_lower = message.lower()

    detected_categories = []
    matched_patterns = []

    for category, category_patterns in patterns.items():

        for pattern in category_patterns:

            if pattern in message_lower:

                if category not in detected_categories:
                    detected_categories.append(category)

                matched_patterns.append(pattern)

    risk_score = min(
        len(detected_categories) * 15,
        100
    )

    return {
        "risk_score": risk_score,
        "detected_categories": detected_categories,
        "matched_patterns": matched_patterns
    }


# ============================================================
# TÜRKÇE ANALİZ
# ============================================================

def analyze_rules(message: str) -> dict:
    """
    Türkçe mesajı kural tabanlı olarak analiz eder.
    """

    return _analyze_with_patterns(
        message,
        TR_RISK_PATTERNS
    )


# ============================================================
# İNGİLİZCE ANALİZ
# ============================================================

def analyze_rules_en(message: str) -> dict:
    """
    İngilizce mesajı kural tabanlı olarak analiz eder.
    """

    return _analyze_with_patterns(
        message,
        EN_RISK_PATTERNS
    )
NORMAL_EMAIL_PATTERNS = [
    "unsubscribe",
    "newsletter",
    "google classroom",
    "google llc",
    "üniversitesi",
    "mühendislik fakültesi",
    "duyuru",
    "eğitim-öğretim",
    "coursera",
]

def analyze_context(message: str, sender: str = "", subject: str = "") -> dict:
    """
    Normal e-posta / newsletter / kurumsal bildirim sinyallerini tespit eder.
    Bunlar tek başına maili güvenli ilan etmez; sadece bağlam bilgisi sağlar.
    """

    combined_text = f"{sender} {subject} {message}".lower()

    detected_patterns = []

    for pattern in NORMAL_EMAIL_PATTERNS:
        if pattern in combined_text:
            detected_patterns.append(pattern)

    return {
        "normal_email_signals": detected_patterns,
        "has_normal_email_context": len(detected_patterns) > 0
    }