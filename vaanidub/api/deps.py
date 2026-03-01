"""FastAPI dependency injection."""

from fastapi import Request
from sqlalchemy.orm import Session

from vaanidub.config import AppConfig
from vaanidub.db.session import get_session


def get_config(request: Request) -> AppConfig:
    """Get app configuration from request state."""
    return request.app.state.config


def get_db() -> Session:
    """Get a database session."""
    session = get_session()
    try:
        yield session
    finally:
        session.close()
