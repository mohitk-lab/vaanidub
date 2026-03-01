"""Health check API routes."""

from fastapi import APIRouter, Depends

from vaanidub import __version__
from vaanidub.api.deps import get_config
from vaanidub.api.schemas import HealthResponse
from vaanidub.config import AppConfig
from vaanidub.models.gpu_manager import GPUManager

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health_check(config: AppConfig = Depends(get_config)):
    """System health check."""
    gpu_mgr = GPUManager(device=config.gpu.device)

    # Check Redis
    redis_ok = False
    try:
        import redis
        r = redis.from_url(config.redis.url, socket_timeout=2)
        r.ping()
        redis_ok = True
    except Exception:
        pass

    # Check DB
    db_ok = False
    try:
        from sqlalchemy import text
        from vaanidub.db.session import get_session
        session = get_session()
        session.execute(text("SELECT 1"))
        session.close()
        db_ok = True
    except Exception:
        pass

    # GPU info
    gpu_available = False
    gpu_device = None
    gpu_free = None
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_device = torch.cuda.get_device_name(0)
            gpu_free = gpu_mgr.get_free_vram_mb()
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if (redis_ok and db_ok) else "degraded",
        version=__version__,
        gpu_available=gpu_available,
        gpu_device=gpu_device,
        gpu_free_vram_mb=gpu_free,
        redis_connected=redis_ok,
        db_connected=db_ok,
    )
