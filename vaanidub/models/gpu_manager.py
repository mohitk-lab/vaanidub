"""GPU memory tracking and model offloading."""

import structlog

logger = structlog.get_logger()


class GPUManager:
    """Tracks GPU memory usage and manages model placement."""

    def __init__(self, device: str = "cuda:0", max_usage_percent: int = 90):
        self.device = device
        self.max_usage_percent = max_usage_percent

    def get_free_vram_mb(self) -> int:
        """Get available GPU VRAM in MB."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            free_mem, _ = torch.cuda.mem_get_info(0)
            return free_mem // (1024 * 1024)
        except Exception:
            return 0

    def get_used_vram_mb(self) -> int:
        """Get used GPU VRAM in MB."""
        try:
            import torch
            if not torch.cuda.is_available():
                return 0
            total = torch.cuda.get_device_properties(0).total_mem
            free, _ = torch.cuda.mem_get_info(0)
            return (total - free) // (1024 * 1024)
        except Exception:
            return 0

    def can_fit_model(self, required_vram_mb: int) -> bool:
        """Check if a model can fit in available VRAM."""
        free = self.get_free_vram_mb()
        return free >= required_vram_mb

    def clear_cache(self) -> None:
        """Clear GPU memory cache."""
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                import gc
                gc.collect()
                logger.info("gpu_cache_cleared", free_mb=self.get_free_vram_mb())
        except Exception as e:
            logger.warning("gpu_cache_clear_failed", error=str(e))
