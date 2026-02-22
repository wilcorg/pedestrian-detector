"""
Roman Szlachtun, 272330

Ogranicza liczbę przykładów negatywnych w sposób losowy.

Jest stosowany przy zbyt dużych ilościach false positive.
"""

import os
import shutil
from pathlib import Path
import random

def downsample(srcDir: Path, targetDir: Path, target: int = 4000):
    os.makedirs(targetDir, exist_ok=True)

    srcImgs = [file.name for file in srcDir.glob('*.png')]
    random.shuffle(srcImgs)
    srcImgs = srcImgs[:min(len(srcImgs), target)]

    for img in srcImgs:
        shutil.copy(srcDir / img, targetDir / img)

def main():
    stage = 'stage4'
    downsample(Path(f'{stage}/2-deduplicated'), Path(f'{stage}/3-final'))

if __name__ == '__main__':
    main()
