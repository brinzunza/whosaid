import pytest

from whosaid import disk


def test_ensure_free_raises_when_low_and_passes_when_ok(monkeypatch):
    class U:  # noqa
        def __init__(self, free): self.free = free
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda p: U(1.0 * 2**30))
    with pytest.raises(disk.LowDisk, match="stopping before building x"):
        disk.ensure_free("building x", min_gb=2.0)
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda p: U(5.0 * 2**30))
    assert disk.ensure_free("x", min_gb=2.0) == pytest.approx(5.0)


def test_threshold_from_env(monkeypatch):
    class U: free = 3 * 2**30
    monkeypatch.setattr(disk.shutil, "disk_usage", lambda p: U())
    monkeypatch.setenv("WHOSAID_MIN_FREE_GB", "4")
    with pytest.raises(disk.LowDisk):
        disk.ensure_free("x")
