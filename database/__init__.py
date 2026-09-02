from app.database import engine, Base
from database.models import User, SMSMessage


Base.metadata.create_all(bind=engine)

print("Database tabloları başarıyla oluşturuldu.")