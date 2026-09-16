from urllib.parse import urlparse
import re


# ============================================================
# URL RISK ANALYZER
# ============================================================


# ============================================================
# GÜVENİLİR DOMAINLER
# ============================================================

TRUSTED_DOMAINS = {
    "google.com",
    "googleusercontent.com",
    "gmail.com",
    "classroom.google.com",
    "microsoft.com",
    "microsoftonline.com",
    "office.com",
    "outlook.com",
    "github.com",
    "linkedin.com",
    "youtube.com",
    "facebook.com",
    "instagram.com",
}


# ============================================================
# GÜÇLÜ PHISHING ANAHTAR KELİMELERİ
# ============================================================

STRONG_SUSPICIOUS_KEYWORDS = [
    "login",
    "signin",
    "verify",
    "verification",
    "password",
]


# ============================================================
# ZAYIF ANAHTAR KELİMELER
# ============================================================

# Bunlar tek başına risk olarak değerlendirilmez.
# Çünkü normal ve güvenilir sitelerde de sık görülürler.

WEAK_SUSPICIOUS_KEYWORDS = [
    "account",
    "secure",
    "bank",
    "update",
]


# ============================================================
# ŞÜPHELİ UZANTILAR
# ============================================================

SUSPICIOUS_EXTENSIONS = [
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq",
]


# ============================================================
# DOMAIN GÜVENİLİR Mİ?
# ============================================================

def is_trusted_domain(hostname: str) -> bool:
    """
    Domain'in güvenilir bir domain olup olmadığını kontrol eder.

    Örneğin:

    accounts.google.com
    -> google.com altında olduğu için güvenilir.

    google.com
    -> güvenilir.

    fake-google.com
    -> güvenilir DEĞİL.
    """

    hostname = hostname.lower().strip(".")

    for trusted_domain in TRUSTED_DOMAINS:

        if (
            hostname == trusted_domain
            or hostname.endswith("." + trusted_domain)
        ):
            return True

    return False


# ============================================================
# URL ANALİZİ
# ============================================================

def analyze_url(url: str) -> dict:

    parsed_url = urlparse(url)

    domain = parsed_url.netloc.lower()
    scheme = parsed_url.scheme.lower()
    path = parsed_url.path.lower()

    is_https = scheme == "https"

    hostname = parsed_url.hostname or ""
    hostname = hostname.lower()

    # --------------------------------------------------------
    # IP adresi kontrolü
    # --------------------------------------------------------

    is_ip_address = bool(
        re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}",
            hostname
        )
    )

    url_length = len(url)

    risk_factors = []
    risk_score = 0

    # --------------------------------------------------------
    # Güvenilir domain kontrolü
    # --------------------------------------------------------

    trusted_domain = is_trusted_domain(hostname)

    # --------------------------------------------------------
    # HTTPS
    # --------------------------------------------------------

    if not is_https:

        risk_factors.append("https_not_used")
        risk_score += 5

    # --------------------------------------------------------
    # IP ADDRESS
    # --------------------------------------------------------

    if is_ip_address:

        risk_factors.append("ip_address_used")
        risk_score += 30

    # --------------------------------------------------------
    # VERY LONG URL
    # --------------------------------------------------------

    if url_length > 200:

        risk_factors.append("very_long_url")
        risk_score += 5

    # --------------------------------------------------------
    # @ SYMBOL
    # --------------------------------------------------------

    # URL authentication kullanılıyorsa risk ver.
    # Query parametresindeki email adreslerine risk verme.

    if (
        parsed_url.username is not None
        or parsed_url.password is not None
    ):

        risk_factors.append("at_symbol_used")
        risk_score += 15

    # --------------------------------------------------------
    # STRONG PHISHING KEYWORDS
    # --------------------------------------------------------

    found_strong_keywords = []

    for keyword in STRONG_SUSPICIOUS_KEYWORDS:

        if keyword in path or keyword in hostname:

            found_strong_keywords.append(keyword)

    # Güvenilir domainlerde tek başına keyword bulunması
    # risk olarak değerlendirilmez.

    if not trusted_domain:

        for keyword in found_strong_keywords:

            risk_factors.append(
                f"suspicious_keyword:{keyword}"
            )

            risk_score += 5

    # --------------------------------------------------------
    # WEAK KEYWORDS
    # --------------------------------------------------------

    # account, secure, bank, update gibi kelimeler
    # tek başına risk değildir.
    #
    # Bunları yalnızca güvenilir olmayan domainlerde
    # daha güçlü phishing işaretleriyle birlikte dikkate alıyoruz.

    found_weak_keywords = []

    for keyword in WEAK_SUSPICIOUS_KEYWORDS:

        if keyword in path or keyword in hostname:

            found_weak_keywords.append(keyword)

    if (
        not trusted_domain
        and len(found_weak_keywords) > 0
        and len(found_strong_keywords) > 0
    ):

        risk_factors.append(
            "weak_keyword_with_phishing_keyword"
        )

        risk_score += 5

    # --------------------------------------------------------
    # MANY SUBDOMAINS
    # --------------------------------------------------------

    if not is_ip_address:

        subdomain_count = hostname.count(".")

        if subdomain_count >= 4:

            risk_factors.append("many_subdomains")
            risk_score += 5

    # --------------------------------------------------------
    # SUSPICIOUS TLD
    # --------------------------------------------------------

    for extension in SUSPICIOUS_EXTENSIONS:

        if hostname.endswith(extension):

            risk_factors.append(
                f"suspicious_extension:{extension}"
            )

            risk_score += 25

            break

    # --------------------------------------------------------
    # MULTIPLE PHISHING KEYWORDS
    # --------------------------------------------------------

    if len(found_strong_keywords) >= 2:

        # Güvenilir domainlerde bile iki güçlü keyword
        # tek başına phishing anlamına gelmez.
        #
        # Güvenilir domainlerde bu kuralı uygulamıyoruz.

        if not trusted_domain:

            risk_factors.append(
                "multiple_phishing_keywords"
            )

            risk_score += 20

    # --------------------------------------------------------
    # HTTP + IP COMBINATION
    # --------------------------------------------------------

    if not is_https and is_ip_address:

        risk_factors.append(
            "http_ip_combination"
        )

        risk_score += 15

    # --------------------------------------------------------
    # HTTP + PHISHING KEYWORD
    # --------------------------------------------------------

    if (
        not is_https
        and len(found_strong_keywords) >= 1
        and not trusted_domain
    ):

        risk_factors.append(
            "http_phishing_keyword_combination"
        )

        risk_score += 10

    # --------------------------------------------------------
    # TRUSTED DOMAIN
    # --------------------------------------------------------

    if trusted_domain:

        # Güvenilir domain olması riskleri sıfırlamaz.
        # Ancak normal domain sinyallerinden gelen gereksiz
        # keyword risklerini engeller.

        risk_score = max(
            risk_score - (
                len(found_strong_keywords) * 5
            ),
            0
        )

    # --------------------------------------------------------
    # MAX RISK
    # --------------------------------------------------------

    risk_score = min(
        risk_score,
        100
    )

    # --------------------------------------------------------
    # RESULT
    # --------------------------------------------------------

    return {
        "url": url,
        "domain": domain,
        "is_https": is_https,
        "is_ip_address": is_ip_address,
        "url_length": url_length,
        "risk_score": risk_score,
        "risk_factors": risk_factors
    }