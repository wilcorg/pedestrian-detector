"""
Roman Szlachtun, 272330

Singleton konfiguracji HOG-SVM pomiędzy wszystkimi plikami.
"""

from typing import Final


class SvmConfig:
    def __init__(self):
        self.windowSize: Final[tuple[int, int]] = (64, 128)
        self.pixelPerCell: Final[tuple[int, int]] = (8, 8)
        self.cellPerBlock: Final[tuple[int, int]] = (2, 2)
        self.blockSize: Final[tuple[int, int]] = (16, 16)
        self.blockStride: Final[tuple[int, int]] = (8, 8)
        self.orientationNumber: Final[int] = 9
