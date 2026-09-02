import re


def extract_urls(message: str) -> list[str]:
    """
    SMS içerisindeki URL'leri bulur ve liste olarak döndürür.
    """

    url_pattern = r"https?://[^\s]+"

    urls = re.findall(url_pattern, message)

    return urls

def clean_text(message: str) -> str:
    """
    SMS metnindeki gereksiz boşlukları temizler.
    """

    cleaned_message = re.sub(r"\s+", " ", message)

    return cleaned_message.strip()

def preprocess_sms(message: str) -> dict:
    """
    SMS'i analiz için hazırlar.
    """

    cleaned_message = clean_text(message)
    urls = extract_urls(cleaned_message)

    text_without_urls = cleaned_message

    for url in urls:
        text_without_urls = text_without_urls.replace(url, "")

    text_without_urls = clean_text(text_without_urls)

    return {
        "original_message": message,
        "cleaned_message": cleaned_message,
        "text_without_urls": text_without_urls,
        "urls": urls
    }