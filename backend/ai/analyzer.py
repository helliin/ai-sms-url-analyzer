from backend.ai.sms_preprocessor import preprocess_sms
from backend.ai.rule_analyzer import (
    analyze_rules,
    analyze_rules_en,
    analyze_context
)
from backend.ai.url_analyzer import analyze_url
from backend.ai.ml_analyzer import analyze_ml
from backend.ai.llm_analyzer import analyze_with_llm


def analyze_sms(
    message: str,
    external_url: str = "",
    sender: str = "",
    subject: str = ""
) -> dict:
    """
    SMS ve varsa harici URL'yi analiz eder.

    Analiz bileşenleri:
    - DistilBERT: SMS spam sınıflandırması
    - Rule Analyzer: metinsel risk kuralları
    - URL Analyzer: URL güvenlik analizi
    - LLM: belirsiz/çelişkili durumlarda ikinci görüş

    Bu bileşenlerin sonuçları birleştirilerek
    0-100 arası genel risk skoru oluşturulur.
    """

    # ============================================================
    # SMS ÖN İŞLEME
    # ============================================================

    preprocessed = preprocess_sms(message)

    cleaned_message = preprocessed["cleaned_message"]
    text_without_urls = preprocessed["text_without_urls"]

    # ============================================================
    # DISTILBERT ANALİZİ
    # ============================================================

    ml_result = analyze_ml(cleaned_message)

    ml_score = ml_result["spam_probability"] * 100
    ml_is_spam = ml_result["prediction"] == "spam"

    # ============================================================
    # RULE ANALİZİ
    # ============================================================

    tr_rule_result = analyze_rules(text_without_urls)
    en_rule_result = analyze_rules_en(text_without_urls)

    context_result = analyze_context(
        message=message,
        sender=sender,
        subject=subject
    )

    detected_categories = list(
        dict.fromkeys(
            tr_rule_result["detected_categories"]
            + en_rule_result["detected_categories"]
        )
    )

    matched_patterns = list(
        dict.fromkeys(
            tr_rule_result["matched_patterns"]
            + en_rule_result["matched_patterns"]
        )
    )

    rule_score = min(
        len(detected_categories) * 15,
        100
    )

    rule_result = {
        "risk_score": rule_score,
        "detected_categories": detected_categories,
        "matched_patterns": matched_patterns
    }

    # ============================================================
    # URL ANALİZİ
    # ============================================================

    urls_to_analyze = list(
        preprocessed["urls"]
    )

    if external_url.strip():
        urls_to_analyze.append(
            external_url.strip()
        )

    url_results = []

    for url in urls_to_analyze:
        url_result = analyze_url(url)
        url_results.append(url_result)

    if url_results:
        url_score = max(
            result["risk_score"]
            for result in url_results
        )
    else:
        url_score = 0

    url_is_suspicious = url_score >= 40

    # ============================================================
    # GENEL RİSK SKORU
    # ============================================================

    weighted_score = (
        (ml_score * 0.50)
        + (rule_score * 0.20)
        + (url_score * 0.30)
    )

    overall_risk_score = round(
        weighted_score
    )

    # ============================================================
    # NORMAL E-POSTA BAĞLAMI
    # ============================================================

    has_normal_context = context_result["has_normal_email_context"]

    if (
        has_normal_context
        and rule_score == 0
        and url_score == 0
        and ml_score >= 75
    ):
        overall_risk_score = round(
            overall_risk_score * 0.60
        )

    # ============================================================
    # GÜÇLÜ DISTILBERT SİNYALİ
    # ============================================================

    if ml_score >= 90:
        if has_normal_context and rule_score == 0 and url_score == 0:
            overall_risk_score = max(
                overall_risk_score,
                50
            )
        else:
            overall_risk_score = max(
                overall_risk_score,
                80
            )

    elif ml_score >= 75:
        if has_normal_context and rule_score == 0 and url_score == 0:
            overall_risk_score = max(
                overall_risk_score,
                45
            )
        else:
            overall_risk_score = max(
                overall_risk_score,
                70
            )

    # ============================================================
    # GÜÇLÜ URL SİNYALİ
    # ============================================================

    if url_score >= 60:
        overall_risk_score = max(
            overall_risk_score,
            url_score
        )

    # ============================================================
    # GÜÇLÜ RULE SİNYALİ
    # ============================================================

    if rule_score >= 70:
        overall_risk_score = max(
            overall_risk_score,
            rule_score
        )

    # ============================================================
    # LLM İKİNCİ GÖRÜŞ KARARI
    # ============================================================

    ml_uncertain = 35 <= ml_score <= 75

    ml_url_conflict = (
        (ml_is_spam and not url_is_suspicious)
        or
        (not ml_is_spam and url_is_suspicious)
    )

    rule_ml_conflict = (
        rule_score >= 30
        and not ml_is_spam
    )

    medium_risk = 30 <= overall_risk_score <= 70

    high_risk = overall_risk_score >= 70

    llm_needed = (
        ml_uncertain
        or ml_url_conflict
        or rule_ml_conflict
        or medium_risk
        or high_risk
    )

    # ============================================================
    # LLM ANALİZİ
    # ============================================================

    llm_result = None

    if llm_needed:
        try:
            llm_result = analyze_with_llm(
                cleaned_message
            )

        except Exception as e:
            llm_result = {
                "prediction": "unavailable",
                "risk_score": 0,
                "confidence": 0,
                "reason": "LLM analizi kullanılamadı.",
                "risk_factors": [],
                "error": str(e)
            }

    # ============================================================
    # LLM SKORUNU GENEL RİSK SKORUNA DAHİL ET
    # ============================================================

    if (
        llm_result is not None
        and str(
            llm_result.get("prediction", "")
        ).upper() in ["HAM", "SPAM"]
    ):
        print("DEBUG LLM:", llm_result)
        llm_score = float(
            llm_result.get("risk_score", 0)
        )

        llm_confidence = float(
            llm_result.get("confidence", 0)
        )

        # LLM yalnızca yeterli güvene sahipse
        # final skoru etkiler.
        if llm_confidence >= 0.60:

            overall_risk_score = round(
                (overall_risk_score * 0.85)
                + (llm_score * 0.15)
            )

            overall_risk_score = max(
                0,
                min(
                    overall_risk_score,
                    100
                )
            )

    # ============================================================
    # SONUÇ
    # ============================================================

    return {
        "message": cleaned_message,
        "urls": url_results,
        "rule_analysis": rule_result,
        "context_analysis": context_result,
        "ml_analysis": ml_result,
        "llm_analysis": llm_result,
        "overall_risk_score": overall_risk_score
    }