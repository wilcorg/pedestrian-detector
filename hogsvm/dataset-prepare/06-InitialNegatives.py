"""
Roman Szlachtun, 272330

Program do generacji przykładów negatywnych, nie zawierających oznaczeń ludzi.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np


"""
Oblicza stopień przecięcia się dwóch ramek
"""
def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0: return 0.0
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    denom = area_a + area_b - inter
    return inter / denom if denom > 0 else 0.0

"""
Ogranicza położenie ramki do granic okna
"""
def clamp_window(x, y, w, h, W, H):
    x = int(x); y = int(y)
    if w > W or h > H:
        return None
    x = min(max(0, x), W - w)
    y = min(max(0, y), H - h)
    return x, y

"""
Odczytuje oznaczenia ze zbioru danych
"""
def load_top_annotations(top_path):
    per_frame = defaultdict(list)
    with open(top_path, "r", newline="", encoding="utf-8") as f:
        rdr = csv.reader(f)
        for row in rdr:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                pn, fr, hv, bv = int(float(row[0])), int(float(row[1])), int(float(row[2])), int(float(row[3]))
                if bv != 1:
                    continue
                bl = float(row[8]); bt = float(row[9]); br = float(row[10]); bb = float(row[11])
                x1, y1, x2, y2 = int(round(bl)), int(round(bt)), int(round(br)), int(round(bb))
                if x2 > x1 and y2 > y1:
                    per_frame[fr].append((x1, y1, x2, y2))
            except Exception:
                continue
    return per_frame

"""
Odczytuje określoną klatkę z nagrania
"""
def get_frame_from_video(cap, idx):
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    return frame if ok else None


"""
Tworzy wycinki negatywne, zawierające częsciowo ludzi. 
Takie wycinki zwiększają jakość detekcji SVM w nieoczywistych sytuacjach.
"""
def sample_near_negatives(W, H, boxes, w=64, h=128, tries=40, iou_min=0.02, iou_max=0.20):
    out = []
    if not boxes:
        return out
    for _ in range(tries):
        bx1, by1, bx2, by2 = boxes[random.randrange(len(boxes))]
        bw = bx2 - bx1; bh = by2 - by1
        # Odchylenie od oznaczenia człowieka
        cx = (bx1 + bx2) * 0.5 + random.uniform(-0.6*bw, 0.6*bw)
        cy = (by1 + by2) * 0.5 + random.uniform(-0.6*bh, 0.6*bh)
        x = int(round(cx - w/2)); y = int(round(cy - h/2))
        clamped = clamp_window(x, y, w, h, W, H)
        if clamped is None:
            continue
        x, y = clamped
        cand = (x, y, x + w, y + h)

        # Obliczenie najwyższego uzyskanego IoU
        max_iou = 0.0
        for b in boxes:
            ii = iou_xyxy(cand, b)
            if ii > max_iou:
                max_iou = ii
                if max_iou > iou_max:
                    break

        # Wycinek jest zapisywany, jeśli jest zawarty w ograniczeniach
        if iou_min <= max_iou <= iou_max:
            out.append((x, y, w, h))
    return out

"""
Tworzy wycinki negatywne, nie zawierający ludzi. 
Takie wycinki zwiększają jakość detekcji SVM w sytuacjach ogólnych (najczęstszych).
"""
def sample_easy_negatives(W, H, boxes, w=64, h=128, tries=40, iou_max=0.01):
    out = []
    for _ in range(tries):
        x = random.randint(0, max(0, W - w))
        y = random.randint(0, max(0, H - h))
        cand = (x, y, x + w, y + h)
        ok = True
        if boxes:
            for b in boxes:
                if iou_xyxy(cand, b) > iou_max:
                    ok = False
                    break
        if ok:
            out.append((x, y, w, h))
    return out


def main():
    top_path = Path('dataset') / 'groundtruth.top'
    video_path = Path('dataset') / 'TownCentreXVID.avi'
    crops_path = Path('crops')
    crops_target = 5000  # liczba wycinków do generacji
    near_ratio = 0.7 # stosunek wycinków, zawierających częściowo ludzi
    crops_per_frame_tries = 30 # liczba prób generacji wycinków dla każdej klatki

    iou_easy_max = 0.01  # maksymalne przecięcie z człowiekiem dla przykładów negatywnych łatwych
    iou_neg_max = 0.2  # maksymalne przecięcie z człowiekiem dla przykładów negatywnych trudnych
    iou_near_min = 0.02 # minimalne przecięcie z człowiekiem dla przykładów negatywnych trudnych
    rng = random.Random(0)
    np.random.seed(0)

    per_frame = load_top_annotations(top_path)
    if not per_frame:
        print("Nie udało się odczytać plik z oznaczeniami zbioru danych")
        return

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print("Nie udało się otworzyć nagranie:", video_path)
        return

    first_frame = get_frame_from_video(cap, 0)
    if first_frame is None:
        print("Nie udało się pobrać pierwszą klatkę nagrania.")
        return

    H, W = first_frame.shape[:2]

    out_dir = Path(crops_path); out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0

    frame_indices = sorted(per_frame.keys())
    total_frames = len(frame_indices)

    for fi, frame_idx in enumerate(frame_indices, 1):
        if saved >= crops_target:
            break

        frame = get_frame_from_video(cap, frame_idx)

        if frame is None:
            continue

        boxes = per_frame[frame_idx]

        tries = crops_per_frame_tries
        near_tries = int(round(tries * near_ratio))
        easy_tries = tries - near_tries

        near_cands = sample_near_negatives(
            W, H, boxes, w=64, h=128,
            tries=near_tries,
            iou_min=iou_near_min,
            iou_max=iou_neg_max,
        )
        easy_cands = sample_easy_negatives(
            W, H, boxes, w=64, h=128,
            tries=easy_tries,
            iou_max=iou_easy_max,
        )

        cands = near_cands + easy_cands
        rng.shuffle(cands)

        for (x, y, w, h) in cands:
            crop = frame[y:y+h, x:x+w]
            if crop.size == 0 or crop.shape[0] != 128 or crop.shape[1] != 64:
                continue
            out_name = f"neg_f{frame_idx:05d}_x{x}_y{y}.png"
            out_path = out_dir / out_name
            ok, buf = cv2.imencode(".png", crop)
            if ok:
                out_path.write_bytes(buf)
                saved += 1
                if saved >= crops_target:
                    break

        if fi % 50 == 0:
            print(f"[{fi}/{total_frames}] zapisano wycinków: {saved}")

    print(f"Gotowe. Zapisano {saved} negatywnych wycinków do: {out_dir}.")

if __name__ == "__main__":
    main()
