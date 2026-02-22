"""
Roman Szlachtun, 272330

Zapisuje wszystkie predykcje false positive,
które nie zawierają człowieka w wystarczającej ilości.
"""

import csv
import os
import shutil
from pathlib import Path

from DirectoryAnchor import DATASET_DIR

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Box:
    x: float
    y: float
    w: float
    h: float

def iou(a: Box, b: Box) -> float:
    ax1, ay1, ax2, ay2 = a.x, a.y, a.x + a.w, a.y + a.h
    bx1, by1, bx2, by2 = b.x, b.y, b.x + b.w, b.y + b.h
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    denom = a.w * a.h + b.w * b.h - inter + 1e-12
    return inter / denom


def parse_top(gt_path: str) -> Dict[int, List[Box]]:
    gt: Dict[int, List[Box]] = defaultdict(list)
    with open(gt_path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue

            frame = int(float(row[1]))
            x = float(row[8])
            y = float(row[9])
            w = float(row[10]) - x
            h = float(row[11]) - y

            if w > 0 and h > 0:
                gt[frame].append(Box(x, y, w, h))
    return gt

def sampleFilter(samplesInputDir: Path, samplesOutputDir: Path):
    gt: Dict[int, List[Box]] = parse_top(str(DATASET_DIR / "groundtruth.top"))
    inputDir: Path = samplesInputDir
    outputDir: Path = samplesOutputDir
    os.makedirs(outputDir, exist_ok=True)

    for imgPath in sorted(inputDir.glob("*.png"), key=lambda p: int(p.stem.split("_")[0])):
        metadata = imgPath.stem.split("_")
        imgNum = int(metadata[0])
        box = Box(float(metadata[1]), float(metadata[2]), float(metadata[3]), float(metadata[4]))
        toSave = True

        for person in gt[imgNum]:
            iouResult = iou(box, person)

            if iouResult > 0.10:
                toSave = False

            if not toSave:
                break

        if toSave:
            shutil.copy(imgPath, outputDir / imgPath.name)


def main():
    rootDir: Path = Path(".") / 'unified' / "stage4"
    samplesInputDir: Path = rootDir / "0-unfiltered"
    samplesOutputDir: Path = rootDir / "1-filtered"


    sampleFilter(samplesInputDir, samplesOutputDir)

if __name__ == '__main__':
    main()