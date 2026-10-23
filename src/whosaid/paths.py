import os
from pathlib import Path

MODELS_DIR = Path(os.environ.get("WHOSAID_MODELS", Path(__file__).resolve().parents[2] / "models"))
FACE_MODEL = MODELS_DIR / "yunet.onnx"
ASD_MODEL = MODELS_DIR / "lrasd_talkset.model"
