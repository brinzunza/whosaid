import numpy as np
import torch

from whosaid import sync


def _toy(n=60):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (n, sync.SIDE, sync.SIDE), dtype=np.uint8), rng.normal(size=(n * 4, 13)).astype(np.float32)


def test_both_architectures_score_every_frame():
    crops, mfcc = _toy()
    for arch in ("lag", "concat"):
        m = sync.new_model(arch).eval()
        s = sync.score_track(m, mfcc, crops, 0, chunk=40)
        assert s.shape == (60,) and np.isfinite(s).all() and (s != -5.0).all()


def test_checkpoint_roundtrip(tmp_path):
    crops, mfcc = _toy()
    m = sync.new_model("lag").eval()
    torch.save(dict(arch="lag", state=m.state_dict()), tmp_path / "s.pt")
    m2 = sync.load_sync(tmp_path / "s.pt")
    assert np.allclose(sync.score_track(m, mfcc, crops, 0), sync.score_track(m2, mfcc, crops, 0), atol=1e-5)


def test_info_nce_prefers_aligned_embeddings():
    torch.manual_seed(0)
    v = torch.nn.functional.normalize(torch.randn(2, 8, 30), dim=1)
    y = torch.ones(2, 30)
    aligned = sync.info_nce(v, v.clone(), y)
    shuffled = sync.info_nce(v, v[:, :, torch.randperm(30)], y)
    assert aligned < shuffled
    assert float(sync.info_nce(v, v, torch.zeros(2, 30))) == 0.0


def test_downscale_halves_resolution():
    x = np.full((3, 112, 112), 200, np.uint8)
    assert sync.downscale(x).shape == (3, 56, 56) and (sync.downscale(x) == 200).all()
