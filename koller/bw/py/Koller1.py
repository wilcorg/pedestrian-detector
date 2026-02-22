"""
Roman Szlachtun, 272330

Implementacja jednokanałowego algorytmu odejmowania tła w Python.

Przyjmuje kolejne klatki nagrania, konwertuje ich do odcieni szarości
i zwraca ramki detekcji ruchu.
"""

import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Optional

import cv2
import numpy as np
from numpy import ndarray

from DirectoryAnchor import DATASET_DIR, PROJECT_ROOT_DIR

cv2.setUseOptimized(True)


@dataclass(frozen=True)
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int


class KollerBw:
    deltaThreshold_: int
    alpha1_: float
    alpha2_: float
    morphKernel_: cv2.typing.MatLike
    minAreaFraction_: float

    def __init__(
        self,
        deltaThreshold: int,
        alpha1: float,
        alpha2: float,
        minAreaFraction: float,
    ) -> None:
        self.diff_: Optional[np.ndarray] = None
        self.background_: Optional[np.ndarray] = None
        self.deltaThreshold_: int = deltaThreshold
        self.alpha1_: float = alpha1
        self.alpha2_: float = alpha2
        self.morphKernel_: cv2.Mat = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.minAreaFraction_: float = minAreaFraction

    def predict(self, grayFrame: np.ndarray) -> List[BoundingBox]:
        self.diff_ = self.kollerDiff(grayFrame)
        contours, _ = cv2.findContours(self.diff_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return self.filterContours(contours)

    def kollerDiff(self, grayFrame: np.ndarray) -> np.ndarray:
        """
        Implementuje formułę selekcyjnej aktualizacji tła:
        B_{t+1} = B_t + (alpha1*(1 - M_t) + alpha2*M_t) * D_t
        """
        assert grayFrame.ndim == 2, "Są dozwolone tylko obrazy o jednym kanale kolorowym."
        assert self.background_ is not None, "Model tła nie jest zainicjalizowany."

        Dt = grayFrame - self.background_
        # Binarna maska Mt odzwierciedla przekroczenie progu w Dt
        Mt = (np.abs(Dt) >= self.deltaThreshold_).astype(np.float32)

        alpha_map = self.alpha1_ * (1.0 - Mt) + self.alpha2_ * Mt
        self.background_ = self.background_ + alpha_map * Dt
        np.clip(self.background_, 0.0, 255.0, out=self.background_)

        diff = (Mt * 255.0).astype(np.uint8)
        diff = cv2.morphologyEx(diff, cv2.MORPH_OPEN, self.morphKernel_, iterations=1)
        diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE, self.morphKernel_, iterations=2)

        return diff

    def backgroundModelInit(self, frameList: List[Path], downscaleFactor: float) -> None:
        images: List[np.ndarray] = []

        for framePath in frameList:
            image = readGrayscale(framePath, downscaleFactor)
            images.append(image.astype(np.float32))

        self.background_ = np.median(images, axis=0).astype(np.float32)

    def filterContours(self, contours: Sequence[ndarray]) -> List[BoundingBox]:
        """
        Przekształca kontury obszarów ruchu w prostokątne ramki
        """
        frameHeight, frameWidth = self.background_.shape
        minAreaPx = max(1, int(self.minAreaFraction_ * frameHeight * frameWidth))
        boxes: List[BoundingBox] = []

        for contour in contours:
            area = cv2.contourArea(contour)
            if area < minAreaPx:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            x1 = int(max(0, x))
            y1 = int(max(0, y))
            x2 = int(min(frameWidth, x + w))
            y2 = int(min(frameHeight, y + h))

            if x2 > x1 and y2 > y1:
                boxes.append(BoundingBox(x1, y1, x2, y2))

        return boxes

    def drawDetectionBoxes(self, imgPath: Path, boxes: List[BoundingBox], downscaleFactor: float) -> np.ndarray:
        """
        Pozycje ramek detekcji są mnożone o downscaleFactor i
        nakładane na wejściową klatkę
        """
        img = cv2.imread(str(imgPath), cv2.IMREAD_COLOR_BGR)
        for box in boxes:
            cv2.rectangle(
                img,
                (int(box.x1 * downscaleFactor), int(box.y1 * downscaleFactor)),
                (int(box.x2 * downscaleFactor), int(box.y2 * downscaleFactor)),
                (0, 0, 255),
                thickness=2,
            )

        return img


def readGrayscale(imgPath: Path, downscaleFactor: float) -> np.ndarray:
    img = cv2.imread(str(imgPath), cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(
        img,
        (int(img.shape[1] / downscaleFactor), int(img.shape[0] / downscaleFactor)),
        interpolation=cv2.INTER_AREA,
    )
    return img


def main():
    startImg = 4050
    endImg = 4500
    downscaleFactor = 1.0
    imagesCountForBackground = 20
    backgroundInitImgStep = 25

    saveResult = True
    saveMask = True

    detector = KollerBw(
        deltaThreshold=15,
        alpha1=2e-3,
        alpha2=2e-4,
        minAreaFraction=15e-4,
    )

    assert downscaleFactor >= 1.0, "Współczynnik przeskalowywania w dół nie może być mniejszy od 1."
    assert imagesCountForBackground > 0
    assert 0 < startImg <= endImg <= 4500

    backgroundStartImg = (startImg - 1) - (imagesCountForBackground - 1) * backgroundInitImgStep
    backgroundEndImg = startImg - 1

    startTime = time.time()
    initImgs = []
    for i in range(backgroundStartImg, backgroundEndImg + 1, backgroundInitImgStep):
        imgPath = DATASET_DIR / "frames" / f"{i}.png"
        initImgs.append(imgPath)

    detector.backgroundModelInit(initImgs, downscaleFactor)
    print(f"Czas inicjalizacji modelu tła: {time.time() - startTime:.3f}s")

    total = 0.0
    for i in range(startImg, endImg + 1):
        imgPath = DATASET_DIR / "frames" / f"{i}.png"

        startTime = time.time()
        boxes = detector.predict(readGrayscale(imgPath, downscaleFactor=downscaleFactor))
        print(f"Czas detekcji obiektów na obrazie {imgPath.name}: {time.time() - startTime:.3f}s")
        total += time.time() - startTime

        if saveResult:
            result = detector.drawDetectionBoxes(imgPath, boxes, downscaleFactor=downscaleFactor)
            cv2.imwrite(str(PROJECT_ROOT_DIR / "benchmark" / "results" / "koller1" / f"{i}-image.jpg"), result)

        if saveMask:
            cv2.imwrite(str(PROJECT_ROOT_DIR / "benchmark" / "results" / "koller1" / f"{i}-mask.jpg"), detector.diff_)

    if total != 0.0:
        print(f"Średni czas detekcji na klatkę: {total / (endImg - startImg + 1)}")
        print(f"Średni FPS: {1.0 / (total / (endImg - startImg + 1))}")


if __name__ == "__main__":
    main()
