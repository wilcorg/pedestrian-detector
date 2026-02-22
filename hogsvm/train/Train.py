"""
Roman Szlachtun, 272330

Trenowanie modelu SVM (epsilon-SVR) na podstawie wydobytych cech HOG.
"""

import os
import shelve
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
import cv2
from hogsvm.SvmConfig import SvmConfig

from DirectoryAnchor import (
    SVM_MODEL_DIR,
    DATASET_DIR,
)

def trainModel(stage: str, positiveCount: int, negativeCount: int):
    features: shelve.Shelf[NDArray[np.float32]]
    positiveFeatures: NDArray[np.float32]
    negativeFeatures: NDArray[np.float32]

    os.makedirs(os.path.join(SVM_MODEL_DIR, stage), exist_ok=True)
    positiveDir: Path = DATASET_DIR / 'svm' / 'minimal' / stage / 'positive'
    negativeDir: Path = DATASET_DIR / 'svm' / 'minimal' / stage / 'negative'

    positiveImgs = loadImages(positiveDir, positiveCount)
    negativeImgs = loadImages(negativeDir, negativeCount)

    positiveFeatures = extractFeaturesFromImgList(positiveImgs)
    negativeFeatures = extractFeaturesFromImgList(negativeImgs)

    data = np.vstack((positiveFeatures, negativeFeatures)).astype(np.float32)
    labels = np.hstack((np.ones(positiveFeatures.shape[0], dtype=np.int32), np.zeros(negativeFeatures.shape[0], dtype=np.int32)))

    svm: cv2.ml.SVM = cv2.ml.SVM.create()
    svm.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 100000, 1e-6))
    svm.setGamma(0)
    svm.setKernel(cv2.ml.SVM_LINEAR)
    svm.setP(0.1)
    svm.setC(0.01)
    svm.setType(cv2.ml.SVM_EPS_SVR)

    svm.train(data, cv2.ml.ROW_SAMPLE, labels)
    saveModel(svm, stage)

def saveModel(svm: cv2.ml.SVM, stage: str):
    svm.save(str(SVM_MODEL_DIR / 'unified' / stage  / 'svm-model.cv2'))
    os.makedirs(os.path.join(SVM_MODEL_DIR, stage), exist_ok=True)
    sv = svm.getSupportVectors()
    rho, alpha, svidx = svm.getDecisionFunction(0)
    w = sv[0]

    np.savez_compressed(SVM_MODEL_DIR / 'unified' / stage / 'svm-model.npz', w=np.float32(w), rho=np.float32(rho))

def loadImages(src_dir: Path, count: int) -> NDArray[np.uint8]:
    images = []

    for num, img_path in enumerate(src_dir.rglob("*.png")):
        if num > count:
            break

        img = cv2.imread(str(img_path), cv2.IMREAD_COLOR_RGB)
        images.append(img)

    print(f'Pobrano {len(images)} obrazów.')
    return np.asarray(images, dtype=np.uint8)

def extractFeaturesFromImgList(imgs: NDArray[np.uint8]) -> NDArray[np.float32]:
    features = []
    config: SvmConfig = SvmConfig()
    hog: cv2.HOGDescriptor = cv2.HOGDescriptor(
        config.windowSize,
        config.blockSize,
        config.blockStride,
        config.pixelPerCell,
        config.orientationNumber,
    )

    if len(imgs) == 0:
        raise ValueError("Nie obrano obrazów do obliczenia cech.")

    for image in imgs:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        vec = np.asarray(hog.compute(image, winStride=config.blockStride)).astype(np.float32)
        features.append(vec.reshape(-1).astype(np.float32))

    return np.vstack(features).astype(np.float32)

if __name__ == '__main__':
    stage = 'stage4'
    trainModel(stage, 3000, 21815)
