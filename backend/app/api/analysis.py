
from fastapi import APIRouter

from app.schemas.analysis import AnalysisRequest
from backend.ai.analyzer import analyze_sms


router = APIRouter()


@router.post("/analyze")
def analyze(request: AnalysisRequest):

    result = analyze_sms(
        request.message,
        request.url
    )

    return result

