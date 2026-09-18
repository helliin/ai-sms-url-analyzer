import os
import base64
from bs4 import BeautifulSoup

from backend.ai.analyzer import analyze_sms

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build


SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]


def get_gmail_service(gmail_address=None):
    """Gmail OAuth bağlantısını oluşturur ve doğru hesabı kontrol eder."""

    creds = None

    # Daha önce kaydedilmiş OAuth bilgilerini kullan
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file(
            "token.json",
            SCOPES
        )

    if not creds or not creds.valid:

        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())

        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json",
                SCOPES
            )

            if gmail_address:
                creds = flow.run_local_server(
                    port=0,
                    login_hint=gmail_address,
                    prompt="select_account"
                )
            else:
                creds = flow.run_local_server(
                    port=0
                )

        # OAuth bilgilerini kaydet
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    service = build(
        "gmail",
        "v1",
        credentials=creds
    )

    # Kullanıcının gerçekten hangi Gmail hesabına
    # bağlandığını kontrol et
    if gmail_address:

        profile = (
            service.users()
            .getProfile(
                userId="me"
            )
            .execute()
        )

        connected_email = profile.get(
            "emailAddress",
            ""
        ).lower().strip()

        requested_email = (
            gmail_address
            .lower()
            .strip()
        )

        if connected_email != requested_email:

            raise ValueError(
                f"Girilen Gmail adresi ({gmail_address}) "
                f"ile bağlanan hesap ({connected_email}) aynı değil."
            )

    return service


def decode_body(data):
    """Gmail'den gelen Base64URL verisini metne çevirir."""

    decoded = base64.urlsafe_b64decode(
        data + "==="
    )

    return decoded.decode(
        "utf-8",
        errors="replace"
    )


def get_email_body(payload):
    """Mail gövdesinden okunabilir metin çıkarır."""

    parts = payload.get("parts", [])

    # 1. Önce text/plain ara
    for part in parts:

        mime_type = part.get("mimeType", "")

        if mime_type == "text/plain":

            data = part.get("body", {}).get("data")

            if data:
                return decode_body(data)

    # 2. text/plain yoksa text/html ara
    for part in parts:

        mime_type = part.get("mimeType", "")

        if mime_type == "text/html":

            data = part.get("body", {}).get("data")

            if data:

                html = decode_body(data)

                soup = BeautifulSoup(
                    html,
                    "html.parser"
                )

                # Gereksiz HTML bölümlerini kaldır
                for tag in soup([
                    "style",
                    "script",
                    "head",
                    "title",
                    "meta",
                    "link"
                ]):
                    tag.decompose()

                # Linklerin sadece görünen yazısını bırak.
                # Uzun URL'leri kullanıcıya göstermiyoruz.

                return soup.get_text(
                    "\n",
                    strip=True
                )

    # 3. İç içe MIME parçalarını kontrol et
    for part in parts:

        if part.get("parts"):

            result = get_email_body(part)

            if result:
                return result

    # 4. Direkt body.data kontrol et
    body = payload.get("body", {})

    if body.get("data"):

        data = decode_body(
            body["data"]
        )

        # Eğer HTML ise temizle
        if "<html" in data.lower() or "<!doctype" in data.lower():

            soup = BeautifulSoup(
                data,
                "html.parser"
            )

            for tag in soup([
                "style",
                "script",
                "head",
                "title",
                "meta",
                "link"
            ]):
                tag.decompose()

            # Linklerin sadece görünen yazısını bırak.
            # Uzun URL'leri kullanıcıya göstermiyoruz.

            return soup.get_text(
                "\n",
                strip=True
            )

        return data

    return ""
def get_email(
    message_id,
    gmail_address=None
):
    """Belirli bir Gmail mailini getirir."""

    service = get_gmail_service(
        gmail_address
    )

    email = (
        service.users()
        .messages()
        .get(
            userId="me",
            id=message_id,
            format="full"
        )
        .execute()
    )

    headers = {
        header["name"]: header["value"]
        for header in email["payload"].get(
            "headers",
            []
        )
    }

    body = get_email_body(
        email["payload"]
    )

    return {
        "id": message_id,
        "from": headers.get(
            "From",
            ""
        ),
        "subject": headers.get(
            "Subject",
            ""
        ),
        "date": headers.get(
            "Date",
            ""
        ),
        "body": body
    }


def get_recent_emails(
    max_results=10,
    gmail_address=None
):
    """Kullanıcının son Gmail maillerini getirir."""

    service = get_gmail_service(
        gmail_address
    )

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=max_results
        )
        .execute()
    )

    messages = response.get(
        "messages",
        []
    )

    emails = []

    for message in messages:

        email = get_email(
            message["id"],
            gmail_address
        )

        emails.append({
            "id": email["id"],
            "from": email["from"],
            "subject": email["subject"],
            "date": email["date"],
            "body": email["body"]
        })

    return emails


def list_recent_emails(
    max_results=5
):
    """Son mailleri terminalde listeler."""

    service = get_gmail_service()

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=max_results
        )
        .execute()
    )

    messages = response.get(
        "messages",
        []
    )

    print("\n==============================")
    print("SON MAİLLER")
    print("==============================")

    if not messages:

        print("Mail bulunamadı.")
        return

    for message in messages:

        email = get_email(
            message["id"]
        )

        print("\n------------------------------")
        print(
            "ID:",
            email["id"]
        )

        print(
            "Gönderen:",
            email["from"]
        )

        print(
            "Konu:",
            email["subject"]
        )

        print(
            "Tarih:",
            email["date"]
        )

        print("\nMAİL İÇERİĞİ:")

        print(
            email["body"][:1000]
        )


if __name__ == "__main__":

    service = get_gmail_service()

    response = (
        service.users()
        .messages()
        .list(
            userId="me",
            maxResults=1
        )
        .execute()
    )

    messages = response.get(
        "messages",
        []
    )

    if not messages:

        print("Mail bulunamadı.")

    else:

        message_id = messages[0]["id"]

        print("\nGMAIL MESSAGE ID:")
        print(message_id)

        email = get_email(
            message_id
        )

        print("\n==============================")
        print("GMAİL ANALİZ TESTİ")
        print("==============================")

        print("\nGönderen:")
        print(email["from"])

        print("\nKonu:")
        print(email["subject"])

        print("\nMail içeriği alındı.")
        print("AI analizi başlıyor...")

        result = analyze_sms(
             email["body"],
            sender=email["from"],
            subject=email["subject"]
)

        print("\n==============================")
        print("ANALİZ SONUCU")
        print("==============================")

        print(
            "Risk skoru:",
            result["overall_risk_score"]
        )

        print(
            "ML sonucu:",
            result["ml_analysis"]
        )

        print(
            "Kural sonucu:",
            result["rule_analysis"]
        )

        print(
            "URL sonucu:",
            result["urls"]
        )