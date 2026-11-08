from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text, create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from .config import get_bootstrap_settings


class Base(DeclarativeBase):
    pass


class MLBacktestJob(Base):
    __tablename__ = "ml_backtest_jobs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    status: Mapped[str] = mapped_column(String(24), index=True, default="QUEUED")
    stage: Mapped[str] = mapped_column(String(80), default="Queued")
    progress: Mapped[int] = mapped_column(Integer, default=0)
    symbol: Mapped[str] = mapped_column(String(32), index=True)
    decision_timeframe: Mapped[str] = mapped_column(String(16))
    trend_timeframe: Mapped[str] = mapped_column(String(16))
    period_years: Mapped[int] = mapped_column(Integer)
    initial_capital: Mapped[float] = mapped_column(Float)
    request_json: Mapped[str] = mapped_column(Text)
    replay_parent_job_id: Mapped[str | None] = mapped_column(
        String(36), nullable=True, index=True
    )
    settings_snapshot_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    result_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    cancel_requested: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class MLBacktestAudit(Base):
    """Compressed complete out-of-sample audit, stored separately from UI data."""

    __tablename__ = "ml_backtest_audits"

    job_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("ml_backtest_jobs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    archive_zip: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    recorded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )

 
class MLResearchStrategy(Base):
    """Persisted LightGBM strategy and its optimization campaign."""
    __tablename__ = "ml_research_strategies"

    id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    source_job_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("ml_backtest_jobs.id"), nullable=False, index=True
    )
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(24), nullable=False, default="DRAFT")
    mode: Mapped[str | None] = mapped_column(String(24), nullable=True)
    progress: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trials_target: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    trials_json: Mapped[str] = mapped_column(Text, nullable=False, default="[]")
    champion_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    summary_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    execution_job_id: Mapped[str | None] = mapped_column(
        String(36), ForeignKey("ml_backtest_jobs.id"), nullable=True
    )
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )



class MLRuntimeParameter(Base):
    __tablename__ = "ml_runtime_parameters"

    key: Mapped[str] = mapped_column(String(160), primary_key=True)
    category: Mapped[str] = mapped_column(String(80), index=True)
    label: Mapped[str] = mapped_column(String(160))
    description: Mapped[str] = mapped_column(Text)
    value_type: Mapped[str] = mapped_column(String(32))
    value_json: Mapped[str] = mapped_column(Text)
    constraints_json: Mapped[str] = mapped_column(Text, default="{}")
    public: Mapped[bool] = mapped_column(Boolean, default=True)
    requires_restart: Mapped[bool] = mapped_column(Boolean, default=False)
    display_order: Mapped[int] = mapped_column(Integer, default=0)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc)
    )

    def decoded_value(self):  # type: ignore[no-untyped-def]
        return json.loads(self.value_json)


bootstrap = get_bootstrap_settings()
_connect_args = {"check_same_thread": False} if bootstrap.database_url.startswith("sqlite") else {}
engine = create_engine(bootstrap.database_url, connect_args=_connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


if bootstrap.database_url.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def _sqlite_pragmas(dbapi_connection, _connection_record) -> None:  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


def _add_missing_job_columns() -> None:
    inspector = inspect(engine)
    if "ml_backtest_jobs" not in inspector.get_table_names():
        return
    columns = {column["name"] for column in inspector.get_columns("ml_backtest_jobs")}
    if "settings_snapshot_json" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE ml_backtest_jobs ADD COLUMN settings_snapshot_json TEXT"))
    if "replay_parent_job_id" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE ml_backtest_jobs ADD COLUMN replay_parent_job_id VARCHAR(36)"))
    with engine.begin() as connection:
        connection.execute(text(
            "CREATE INDEX IF NOT EXISTS ix_ml_backtest_jobs_replay_parent_job_id "
            "ON ml_backtest_jobs (replay_parent_job_id)"
        ))


def initialize_database() -> None:
    Base.metadata.create_all(engine)
    _add_missing_job_columns()

    from .settings_service import synchronize_parameter_catalog

    synchronize_parameter_catalog()
    now = datetime.now(timezone.utc)
    with SessionLocal.begin() as session:
        interrupted = session.query(MLBacktestJob).filter(
            MLBacktestJob.status.in_(["QUEUED", "RUNNING"])
        )
        for job in interrupted:
            job.status = "FAILED"
            job.stage = "Interrupted by service restart"
            job.error_message = "The ML API restarted before this backtest completed."
            job.completed_at = now

        interrupted_tuning = session.query(MLResearchStrategy).filter(
            MLResearchStrategy.status.in_(["RUNNING", "CANCEL_REQUESTED"])
        )
        for strategy in interrupted_tuning:
            strategy.status = "FAILED"
            strategy.error_message = "A ML API reiniciou antes de concluir a otimização."
            strategy.updated_at = now


def session_scope() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
