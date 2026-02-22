"""
Roman Szlachtun, 272330

Program do demonstracji gradientów obrazu.

Gradient oraz kąty gradientów są potrzebne do obliczenia cech HOG.
"""

import cv2
from pathlib import Path
from PIL import Image, ImageDraw

import numpy as np

from DirectoryAnchor import DATASET_DIR


def gradients(img_path: Path):
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    grad_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    mag, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

    np.mod(angle, 180.0, out=angle)
    cv2.normalize(angle, angle, 0, 255, cv2.NORM_MINMAX)

    cv2.imwrite('gradients.png', mag)

def draw(img_path: Path):
    img = cv2.imread(str(img_path))
    image = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)

    for i in range(8, 64, 8):
        draw.line([(i, 0), (i, 128)], fill=(0, 128, 0), width=1)

    for j in range(8, 128, 8):
        draw.line([(0, j), (64, j)], fill=(0, 128, 0), width=1)

    image.show()


if __name__ == '__main__':
    gradients(DATASET_DIR / 'svm/stage1/positive/OTC_p0005_f000020.png')
    # draw(DATASET_DIR / 'svm/stage1/positive/OTC_p0005_f000020.png')

