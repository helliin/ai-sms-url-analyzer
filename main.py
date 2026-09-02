from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from backend.ai.analyzer import analyze_sms


app = FastAPI(title="AI SMS URL Analyzer")


app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "AI SMS URL Analyzer API is running!"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok"
    }


class SMSRequest(BaseModel):
    message: str
    url: str = ""


@app.post("/analyze")
def analyze_sms_endpoint(request: SMSRequest):

    return analyze_sms(
        request.message,
        request.url
    )
