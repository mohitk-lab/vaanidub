"""Tests for GPU memory manager."""

from unittest.mock import MagicMock, patch

from vaanidub.models.gpu_manager import GPUManager


class TestGPUManagerInit:
    def test_default_device(self):
        manager = GPUManager()
        assert manager.device == "cuda:0"

    def test_custom_device(self):
        manager = GPUManager(device="cuda:1")
        assert manager.device == "cuda:1"

    def test_default_max_usage(self):
        manager = GPUManager()
        assert manager.max_usage_percent == 90

    def test_custom_max_usage(self):
        manager = GPUManager(max_usage_percent=80)
        assert manager.max_usage_percent == 80


class TestGetFreeVram:
    def test_get_free_vram_with_gpu(self):
        """Mock CUDA available with 4GB free."""
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # 4GB free, 8GB total
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024 ** 3, 8 * 1024 ** 3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = manager.get_free_vram_mb()

        assert result == 4096

    def test_get_free_vram_no_gpu(self):
        """When CUDA not available, return 0."""
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = manager.get_free_vram_mb()

        assert result == 0

    def test_get_free_vram_no_torch(self):
        """When torch is not installed, return 0."""
        manager = GPUManager()
        # Simulate ImportError by making the import fail
        with patch.dict("sys.modules", {"torch": None}):
            result = manager.get_free_vram_mb()
        assert result == 0


class TestGetUsedVram:
    def test_get_used_vram_with_gpu(self):
        """Mock CUDA with 4GB used (8GB total - 4GB free)."""
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.get_device_properties.return_value = MagicMock(
            total_mem=8 * 1024 ** 3
        )
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024 ** 3, 8 * 1024 ** 3)

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = manager.get_used_vram_mb()

        assert result == 4096

    def test_get_used_vram_no_gpu(self):
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            result = manager.get_used_vram_mb()

        assert result == 0


class TestCanFitModel:
    def test_can_fit_model_yes(self):
        manager = GPUManager()
        with patch.object(manager, "get_free_vram_mb", return_value=4096):
            assert manager.can_fit_model(2000) is True

    def test_can_fit_model_no(self):
        manager = GPUManager()
        with patch.object(manager, "get_free_vram_mb", return_value=1000):
            assert manager.can_fit_model(2000) is False

    def test_can_fit_model_exact(self):
        manager = GPUManager()
        with patch.object(manager, "get_free_vram_mb", return_value=2000):
            assert manager.can_fit_model(2000) is True


class TestClearCache:
    def test_clear_cache_calls_torch(self):
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.mem_get_info.return_value = (4 * 1024 ** 3, 8 * 1024 ** 3)

        mock_gc = MagicMock()

        with patch.dict("sys.modules", {"torch": mock_torch, "gc": mock_gc}):
            manager.clear_cache()

        mock_torch.cuda.empty_cache.assert_called_once()

    def test_clear_cache_no_gpu(self):
        """Should not raise when CUDA is unavailable."""
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False

        with patch.dict("sys.modules", {"torch": mock_torch}):
            manager.clear_cache()  # Should not raise

    def test_clear_cache_exception_handled(self):
        """Should handle exceptions gracefully."""
        manager = GPUManager()
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.empty_cache.side_effect = RuntimeError("CUDA error")

        with patch.dict("sys.modules", {"torch": mock_torch}):
            manager.clear_cache()  # Should not raise
