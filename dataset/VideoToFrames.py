"""
Roman Szlachtun, 272330

Program do konwertacji nagrania wideo w zbiór klatek.
"""

import os
import cv2

from PIL import Image

if __name__ == "__main__":
    cap = cv2.VideoCapture('TownCentreXVID.mp4')
    success, image = cap.read()
    count = 0

    while count <= 4500:
        success, image = cap.read()
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        pil_image.save(os.path.join(f"frames", f"{count}.png"))
        count += 1

        if not success:
            break

    print("Gotowe.")
