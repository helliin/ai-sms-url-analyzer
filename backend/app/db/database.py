import psycopg

DATABASE_URL = (
    "postgresql://postgres:435460@localhost:5432/ai_sms_url_analyzer"
)

def get_connection():
    return psycopg.connect(DATABASE_URL)