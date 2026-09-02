from app.database import engine, Base
from app.models.analysis import Analysis

Base.metadata.create_all(bind=engine)

print("Veritabanı tabloları oluşturuldu.")