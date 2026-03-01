"""Database session management."""

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from vaanidub.config import AppConfig
from vaanidub.db.models import Base

_engine = None
_SessionLocal = None


def init_db(config: AppConfig) -> None:
    """Initialize database engine and create tables."""
    global _engine, _SessionLocal

    connect_args = {}
    kwargs = {}

    if config.database.url.startswith("sqlite"):
        connect_args["check_same_thread"] = False
        # For in-memory SQLite, share a single connection across threads
        if config.database.url in ("sqlite:///", "sqlite://"):
            kwargs["poolclass"] = StaticPool

    _engine = create_engine(
        config.database.url,
        pool_pre_ping=True,
        echo=config.debug,
        connect_args=connect_args,
        **kwargs,
    )
    _SessionLocal = sessionmaker(bind=_engine)

    # Create all tables
    Base.metadata.create_all(bind=_engine)


def get_session() -> Session:
    """Get a new database session."""
    if _SessionLocal is None:
        raise RuntimeError("Database not initialized. Call init_db() first.")
    return _SessionLocal()
