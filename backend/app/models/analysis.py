from sqlalchemy import Column, Integer, String, Text, Float, DateTime
from sqlalchemy.sql import func

from app.database import Base


class Analysis(Base):
    __tablename__ = "analyses"

    id = Column(Integer, primary_key=True, index=True)

    input_text = Column(Text, nullable=False)

    input_type = Column(String(20), nullable=False)

    risk_level = Column(String(20), nullable=False)

    risk_score = Column(Float, nullable=False)

    result = Column(Text, nullable=True)

    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )