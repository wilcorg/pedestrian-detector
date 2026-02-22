"""
Roman Szlachtun, 272330

Program, wykrywający ludzi na obrazach za pomocą modelu SVM i cech HOG w Python.
"""

import time
from pathlib import Path

import cv2, numpy as np

from DirectoryAnchor import (
    SVM_MODEL_DIR,
    DATASET_DIR,
)
from hogsvm.SvmConfig import SvmConfig

cv2.setUseOptimized(True)

def load_frame(imgPath: Path):
    img = cv2.imread(str(imgPath), cv2.IMREAD_COLOR)
    if img is None: raise SystemExit("Cannot read frame")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def get_svm_detector_vector(svm: cv2.ml.SVM) -> np.ndarray:
    sv = svm.getSupportVectors()

    if sv.shape[0] != 1:
        sv = svm.getUncompressedSupportVectors()
    else:
        sv = sv[0]

    rho, alpha, svidx = svm.getDecisionFunction(0)
    svTotal = svm.getSupportVectors().shape[0]

    assert svTotal == 1
    assert alpha.shape[0] == 1 and alpha.shape[1] == 1
    assert svidx.shape[0] == 1 and svidx.shape[1] == 1

    return np.append(sv, -rho).astype(np.float32)

def get_svm_detector_vector(npz_path: Path) -> np.ndarray:
    d = np.load(str(npz_path))
    w = d["w"].astype(np.float32).ravel()
    rho = float(d["rho"])
    detector_vec = np.hstack([w, -rho]).astype(np.float32)
    return detector_vec


def main():
    imgPath = DATASET_DIR / "frames" / "4484.png"
    gray = load_frame(imgPath)
    cfg = SvmConfig()
    hog = cv2.HOGDescriptor(cfg.windowSize, cfg.blockSize, cfg.blockStride, cfg.pixelPerCell, cfg.orientationNumber)
    hog.setSVMDetector(get_svm_detector_vector(SVM_MODEL_DIR / 'unified' / 'stage4' / "8-16-8-9-l2hys.npz"))

    start = time.time()
    rects, weights = hog.detectMultiScale(
        gray,
        hitThreshold=0.3,
        winStride=(8, 8),
        padding=(0, 0),
        scale=1.05,
        groupThreshold=1
    )
    end = time.time()
    print(f"Czas detekcji: {end-start:.3f}")

    print(f"Detekcje: {len(rects)}")
    for (x, y, w, h), s in zip(rects, weights):
        print(f"pozycja: [{x},{y},{w},{h}], wynik: {s:.3f}")

if __name__ == "__main__":
    main()
