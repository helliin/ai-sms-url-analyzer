
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.analysis import router as analysis_router


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


app.include_router(analysis_router)


@app.get("/")
def root():
    return {
        "message": "AI SMS URL Analyzer API çalışıyor!"
    }


@app.get("/api/health")
def health_check():
    return {
        "status": "ok"
    }

