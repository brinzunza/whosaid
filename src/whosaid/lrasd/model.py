import torch
import torch.nn as nn

from .Classifier import Detector, Fusion
from .Encoder import audio_encoder, visual_encoder


class ASDModel(nn.Module):
    """LR-ASD network plus the audio-visual classification head (state-dict keys match upstream)."""

    def __init__(self):
        super().__init__()
        self.model = _Backbone()
        self.lossAV = _Head()

    @torch.no_grad()
    def score(self, audio_mfcc: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
        """audio_mfcc (1, 4T, 13) float, faces (1, T, 112, 112) float 0-255 -> (T,) speaking logit."""
        a = self.model.forward_audio_frontend(audio_mfcc)
        v = self.model.forward_visual_frontend(faces)
        out = self.model.forward_audio_visual_backend(a, v)
        return self.lossAV.FC(out)[:, 1]


class _Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.FC = nn.Linear(128, 2)


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visualEncoder = visual_encoder()
        self.audioEncoder = audio_encoder()
        self.fusion = Fusion(256)
        self.detector = Detector(256)

    def forward_visual_frontend(self, x):
        B, T, W, H = x.shape
        x = x.view(B, 1, T, W, H)
        x = (x / 255 - 0.4161) / 0.1688
        return self.visualEncoder(x)

    def forward_audio_frontend(self, x):
        x = x.unsqueeze(1).transpose(2, 3)
        return self.audioEncoder(x)

    def forward_audio_visual_backend(self, x1, x2):
        x = self.fusion(x1, x2)
        x = self.detector(x)
        return torch.reshape(x, (-1, 128))


def load_model(path) -> ASDModel:
    m = ASDModel()
    state = torch.load(path, map_location="cpu")
    m.load_state_dict({k: v for k, v in state.items() if not k.startswith("lossV.")})
    return m.eval()
