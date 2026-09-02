
from backend.ai.sms_preprocessor import preprocess_sms
from backend.ai.rule_analyzer import analyze_rules
from backend.ai.url_analyzer import analyze_url
from backend.ai.ml_analyzer import analyze_ml
from backend.ai.llm_analyzer import analyze_with_llm

def analyze_sms(message: str, external_url: str = "") -> dict:
    """
    SMS'i ve varsa ayrıca girilen URL'yi analiz eder.
    """

    preprocessed = preprocess_sms(message)

    # ML modeli ile analiz
    ml_result = analyze_ml(
        preprocessed["cleaned_message"]
    )

    # Kural analizi
    rule_result = analyze_rules(
        preprocessed["text_without_urls"]
    )

    # SMS'in içinden bulunan URL'ler
    urls_to_analyze = list(preprocessed["urls"])

    # Web sitesindeki ayrı URL alanından gelen URL
    if external_url.strip():
        urls_to_analyze.append(external_url.strip())

    # URL analizi
    url_results = []

    for url in urls_to_analyze:
        url_result = analyze_url(url)
        url_results.append(url_result)

    rule_score = rule_result["risk_score"]

    if url_results:
        url_score = max(
            result["risk_score"] for result in url_results
        )
    else:
        url_score = 0

    # ML tahminini risk skoruna dönüştür
    if ml_result["prediction"] == "spam":
        ml_score = 100
    else:
        ml_score = 0

    # Genel risk skoru
    weighted_score = (
        (rule_score * 0.30)
        + (url_score * 0.30)
        + (ml_score * 0.40)
    )

    # Tek bir analiz bileşeni ciddi risk tespit ettiğinde
    # genel skorun gereğinden fazla düşük kalmasını önle.
    overall_risk_score = round(weighted_score)

    if url_score >= 60:
        overall_risk_score = max(
            overall_risk_score,
            url_score
        )

    if rule_score >= 70:
        overall_risk_score = max(
            overall_risk_score,
            rule_score
        )

    if ml_score >= 100:
        overall_risk_score = max(
            overall_risk_score,
            70
        )

    # ============================================================
    # LLM KARAR KONTROLÜ
    # ============================================================

    llm_result = None

    # ML ve URL sonuçları çelişiyorsa LLM'den
    # ek değerlendirme iste.
    ml_is_spam = ml_result["prediction"] == "spam"
    url_is_suspicious = url_score >= 60
    rule_is_suspicious = rule_score >= 70

    llm_needed = (
        (ml_is_spam and not url_is_suspicious)
        or
        (not ml_is_spam and url_is_suspicious)
        or
        (rule_is_suspicious and not ml_is_spam)
    )

    if llm_needed:
        llm_result = analyze_with_llm(
            preprocessed["cleaned_message"]
        )

        print(">>> LLM DEVREYE GİRDİ <<<")
    return {
        "message": preprocessed["cleaned_message"],
        "urls": url_results,
        "rule_analysis": rule_result,
        "ml_analysis": ml_result,
        "overall_risk_score": overall_risk_score
    }

