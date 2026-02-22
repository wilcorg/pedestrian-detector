"""
Roman Szlachtun, 272330

Program do konwersji obrazów detekcji w nagranie wideo.
"""

import cv2
from pathlib import Path

FRAMES_DIR = Path("YOLO")
START, END = 4050, 4500
FPS = 25
OUT = "out.mp4"


def open_writer(out_path: str, fps: float, size_wh: tuple[int, int]):
    candidates = ["avc1", "mp4v"]
    for tag in candidates:
        fourcc = cv2.VideoWriter_fourcc(*tag)
        w = cv2.VideoWriter(out_path, fourcc, fps, size_wh, True)
        if w.isOpened():
            return w
        w.release()
    raise RuntimeError("Nie udało się otworzyć VideoWriter dla .mp4 .")


if __name__ == "__main__":
    h, w = 1080, 1920
    writer = open_writer(OUT, FPS, (w, h))

    for i in range(START, END + 1):
        p = FRAMES_DIR / f"{i}.png"
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        writer.write(img)

    writer.release()
    print(f"Gotowe: {OUT}")
