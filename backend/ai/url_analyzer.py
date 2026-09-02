from urllib.parse import urlparse
import re


SUSPICIOUS_KEYWORDS = [
    "login",
    "verify",
    "verification",
    "secure",
    "account",
    "password",
    "signin",
    "bank",
    "update"
]


SUSPICIOUS_EXTENSIONS = [
    ".tk",
    ".ml",
    ".ga",
    ".cf",
    ".gq"
]


def analyze_url(url: str) -> dict:
    """
    URL'nin teknik ve phishing açısından şüpheli özelliklerini analiz eder.
    """

    parsed_url = urlparse(url)

    domain = parsed_url.netloc
    scheme = parsed_url.scheme
    path = parsed_url.path.lower()

    is_https = scheme == "https"

    is_ip_address = bool(
        re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", domain)
    )

    url_length = len(url)

    risk_factors = []

    if not is_https:
        risk_factors.append("https_not_used")

    if is_ip_address:
        risk_factors.append("ip_address_used")

    if url_length > 100:
        risk_factors.append("very_long_url")

    if "@" in url:
        risk_factors.append("at_symbol_used")

    for keyword in SUSPICIOUS_KEYWORDS:
        if keyword in path or keyword in domain.lower():
            risk_factors.append(f"suspicious_keyword:{keyword}")

    if not is_ip_address:
        subdomain_count = domain.count(".")

        if subdomain_count >= 3:
            risk_factors.append("many_subdomains")

    for extension in SUSPICIOUS_EXTENSIONS:
        if domain.lower().endswith(extension):
            risk_factors.append(f"suspicious_extension:{extension}")

    risk_score = min(len(risk_factors) * 15, 100)

    return {
        "url": url,
        "domain": domain,
        "is_https": is_https,
        "is_ip_address": is_ip_address,
        "url_length": url_length,
        "risk_score": risk_score,
        "risk_factors": risk_factors
    }