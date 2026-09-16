from fastapi import APIRouter, Query

from backend.app.schemas.analysis import AnalysisRequest
from backend.ai.analyzer import analyze_sms
from backend.services.gmail_service import (
    get_recent_emails,
    get_email
)


router = APIRouter()


@router.post("/analyze")
def analyze(request: AnalysisRequest):
    result = analyze_sms(
        request.message,
        request.url
    )

    return result


@router.get("/gmail/emails")
def gmail_emails(
    email: str = Query(...)
):
    emails = get_recent_emails(
        10,
        email
    )

    return {
        "emails": emails
    }


@router.post("/gmail/analyze/{message_id}")
def analyze_gmail(message_id: str):

    email = get_email(message_id)

    result = analyze_sms(
        email["body"]
    )

    return {
        "email": {
            "id": email["id"],
            "from": email["from"],
            "subject": email["subject"],
            "date": email["date"],
            "body": email["body"]
        },
        "analysis": result
    }