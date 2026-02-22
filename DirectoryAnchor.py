from pathlib import Path
from typing import Final

PROJECT_ROOT_DIR: Final[Path] = Path(__file__).parent
DATASET_DIR: Final[Path] = PROJECT_ROOT_DIR / "dataset"
SVM_MODEL_DIR: Final[Path] = PROJECT_ROOT_DIR / "hogsvm" / "model"
YOLO_MODEL_DIR: Final[Path] = PROJECT_ROOT_DIR / "yolo" / "model"
