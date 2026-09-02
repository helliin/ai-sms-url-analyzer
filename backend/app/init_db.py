from app.database import engine, Base
from app.models.analysis import Analysis

# Modelleri buraya import edeceğiz
# Böylece SQLAlchemy tabloları tanıyacak.

Base.metadata.create_all(bind=engine)

print("Veritabanı tabloları oluşturuldu.")
