"""
Roman Szlachtun, 272330

Zapisuje wszystkie detekcje, zrobione modelem SVM.

Cel - zebrać detekcje, które detektor uznał za prawdziwe, ale naprawdę są fałyszywymi.

Wykorzystanie powyższych przykładów jako negatywne znacznie poprawia jakość detektora.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final, List

import cv2
import numpy as np

from DirectoryAnchor import (
    SVM_MODEL_DIR,
    DATASET_DIR,
)
from hogsvm.SvmConfig import SvmConfig


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float


@dataclass
class Recognition:
    box: Box
    score: float

def prepareHog(svmPath: Path) -> cv2.HOGDescriptor:
    cfg = SvmConfig()
    hog = cv2.HOGDescriptor(cfg.windowSize, cfg.blockSize, cfg.blockStride, cfg.pixelPerCell, cfg.orientationNumber)
    svm = cv2.ml.SVM.load(str(svmPath))

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

    det = np.append(sv, -rho).astype(np.float32)

    hog.setSVMDetector(det)
    return hog

def nms(rects, weights, iou: float) -> List[Recognition]:
    keep = []
    if len(rects) == 0:
        return keep

    smin = min(weights)
    shift = -smin + 1e-6 if smin < 0 else 0.0
    weights = [w + shift for w in weights]
    idx = cv2.dnn.NMSBoxes(
        rects, weights, score_threshold=0.0, nms_threshold=iou
    )
    idx = np.asarray(idx).tolist()
    for i in idx:
        keep.append(Recognition(Box(*rects[i]), weights[i]))

    return keep


def miner(modelDir: Path, samplesSaveDir: Path):
    hog : Final[cv2.HOGDescriptor] = prepareHog(modelDir / "svm-model.cv2")

    for imgNum in range(3, 3600, 3):
        imgPath = DATASET_DIR / 'frames' / f"{imgNum}.png"
        img = cv2.imread(str(imgPath))

        rects, weights = hog.detectMultiScale(
            img,
            hitThreshold=0.2,
            winStride=(8, 8),
            padding=(0, 0),
            scale=1.2,
            groupThreshold=2
        )

        recognitions = nms(rects, weights, 0.4)
        max_score = 0
        if len(recognitions) > 0:
            max_score = int(recognitions[0].score * 100)

        for recognition in recognitions[:min(40, len(recognitions))]:
            cropped = img[int(recognition.box.y):int(recognition.box.y+recognition.box.h), int(recognition.box.x):int(recognition.box.x+recognition.box.w)]
            cropped = cv2.resize(cropped, (64, 128), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(samplesSaveDir / f'{imgNum}_{recognition.box.x}_{recognition.box.y}_{recognition.box.w}_{recognition.box.h}_{int(recognition.score * 100)}_{int(recognition.score * 100)}_{max_score}.png'), cropped)

        print(imgNum, len(recognitions))


def main():
    modelDir: Path = SVM_MODEL_DIR / "unified" / "stage4"
    samplesSaveDir: Path = Path("unified") / "stage4" / "0-unfiltered"

    os.makedirs(samplesSaveDir, exist_ok=True)
    miner(modelDir, samplesSaveDir)


if __name__ == "__main__":
    main()
