"""
Roman Szlachtun, 272330

Program do wykrywania podobnych detekcji na podstawie bliskości cech HOG.

Ucina ~60% detekcji, które mają bardzo podobny obraz.

Pozwala hamować wzrost liczby przykładów negatywnych
i zwiększać ilość iteracji z eliminacją false positive.
"""

import os
import shutil
from pathlib import Path

import cv2
import numpy as np

from hogsvm.SvmConfig import SvmConfig

cfg = SvmConfig()
HOG = cv2.HOGDescriptor(cfg.windowSize, cfg.blockSize, cfg.blockStride, cfg.pixelPerCell, cfg.orientationNumber)


def readGrayImg(imgPath: Path):
    img = cv2.imread(str(imgPath))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def deduplicate(srcDir: Path, targetDir: Path):
    os.makedirs(targetDir, exist_ok=True)
    # sortowanie wycinków według numeru klatki nagrania
    srcImgPaths = sorted(srcDir.glob("*.png"), key=lambda p: int(p.stem.split("_")[0]))

    srcImgPaths = [img.name for img in srcImgPaths]

    keepImgPaths = []

    for index, imgPath in enumerate(srcImgPaths):
        img = readGrayImg(srcDir / imgPath)
        ok = True

        for keepImgPath in keepImgPaths[-200::-1]:
            keepImg = readGrayImg(srcDir / keepImgPath)
            if nearDuplicate(img, keepImg):
                ok = False
                break

        if ok:
            keepImgPaths.append(imgPath)
            shutil.copy(srcDir / imgPath, targetDir / imgPath)

        if index % 100 == 0:
            print(index, len(keepImgPaths))



def nearDuplicate(aImg, bImg) -> bool:
    if hogCosine(aImg, bImg) >= 0.95:
        return True
    else:
        return False

def hogCosine(a, b):
    fa = HOG.compute(a).reshape(-1).astype(np.float32)
    fb = HOG.compute(b).reshape(-1).astype(np.float32)
    denom = np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-12
    return float(np.dot(fa, fb) / denom)


def main():
    stage = "stage4"
    deduplicate(Path('unified') / Path(stage) / "1-filtered", Path('unified') / Path(stage) / "2-deduplicated")


if __name__ == "__main__":
    main()
