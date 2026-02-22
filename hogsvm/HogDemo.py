"""
Roman Szlachtun, 272330

Program do demonstracji cech HOG z obrazu.

Dla większej czytelności cechy HOG zostały pogrubione.
"""

import cv2

from skimage.feature import hog
from skimage import exposure

from DirectoryAnchor import DATASET_DIR

def main():
    img = cv2.imread(str(DATASET_DIR / 'frames'/ '2.png'), cv2.IMREAD_GRAYSCALE)

    fd, hog_image = hog(img, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 25))

    cv2.normalize(hog_image_rescaled, hog_image_rescaled, 0, 255, cv2.NORM_MINMAX)
    cv2.imwrite('hog_visualization.png', hog_image_rescaled)


if __name__ == '__main__':
    main()
