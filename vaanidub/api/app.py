"""FastAPI application factory."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from vaanidub import __version__
from vaanidub.config import AppConfig
from vaanidub.db.session import init_db
from vaanidub.logging_config import setup_logging


def create_app(config: AppConfig | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if config is None:
        config = AppConfig()
        config.resolve_secrets()
        config.ensure_directories()

    setup_logging(config.log_level, getattr(config, "log_format", "console"))

    app = FastAPI(
        title="VaaniDub API",
        description="AI-powered regional dubbing tool for Indian languages",
        version=__version__,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store config in app state
    app.state.config = config

    # Initialize database
    init_db(config)

    # Register routes
    from vaanidub.api.routes import health, jobs, languages
    app.include_router(jobs.router, prefix="/api/v1")
    app.include_router(languages.router, prefix="/api/v1")
    app.include_router(health.router, prefix="/api/v1")

    return app
