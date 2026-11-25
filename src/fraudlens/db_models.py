
from datetime import datetime

from sqlalchemy import (
    Integer,
    Float,
    String,
    DateTime,
    CheckConstraint,
    func,
)

from sqlalchemy.orm import Mapped, mapped_column

from fraudlens.database import Base


class PredictionRecord(Base):

    __tablename__ = "predictions"

    __table_args__ = (
        CheckConstraint(
            "fraud_probability >= 0 "
            "AND fraud_probability <= 1"
        ),
        CheckConstraint(
            "prediction IN (0, 1)"
        ),
        CheckConstraint(
            "latency_ms >= 0"
        ),
    )

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
    )

    request_id: Mapped[str] = mapped_column(
        String(36),
        unique=True,
        nullable=False,
    )

    fraud_probability: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    threshold: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    prediction: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    label: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    model_version: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
    )

    latency_ms: Mapped[float] = mapped_column(
        Float,
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        index=True,
    )
