"""
Roman Szlachtun, 272330

Program do detekcji pieszych na nagraniu za pomocą modelu YOLO.
"""

import time

from ultralytics import YOLO
from pathlib import Path

from DirectoryAnchor import DATASET_DIR, PROJECT_ROOT_DIR
from PIL import Image, ImageDraw


def main():
    model = YOLO(PROJECT_ROOT_DIR / 'yolo' / 'model' / '1088x1920' / 'best.pt', task='detect')

    imgPath = Path(DATASET_DIR / 'frames' / f'{4470}.png')
    start = time.time()
    results = model.predict(imgPath, device='cpu', rect=True, nms=True)
    total_time = time.time() - start

    img = Image.open(imgPath)
    draw = ImageDraw.Draw(img)

    print(f'Czas detekcji: {total_time:.3f}s')

    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0]
            draw.rectangle([x1, y1, x2, y2], outline="red", width=4)

    # img.show()


if __name__ == '__main__':
    main()
