"""
Roman Szlachtun, 272330

Wycina oznaczenia ludzi do osobnych obrazów.

Wycinki ludzi służą podstawą do zbioru przykładów pozytywnych.
"""

import csv
import os
from collections import defaultdict

import cv2


"""
Odczyt oznaczeń ze zbioru danych
"""
def parse_top(top_path):
    frames = defaultdict(list)
    with open(top_path, 'r', newline='') as f:
        reader = csv.reader(f, skipinitialspace=True)
        for row in reader:
            if not row or row[0].startswith('#'):
                continue
            try:
                person_id = int(row[0])
                frame_no = int(row[1])
                head_valid = int(row[2])
                body_valid = int(row[3])
                vals = list(map(float, row[4:12]))
                direction = float(row[12])
                if len(vals) != 8:
                    continue
                headL, headT, headR, headB, bodyL, bodyT, bodyR, bodyB = vals
            except ValueError:
                continue
            frames[frame_no].append({
                "person_id": person_id,
                "head_valid": head_valid,
                "body_valid": body_valid,
                "head": (headL, headT, headR, headB),
                "body": (bodyL, bodyT, bodyR, bodyB),
                "direction": direction,
            })
    return frames

"""
Przycięcie oznaczenia do granic okna
"""
def clamp_box(x1, y1, x2, y2, W, H):
    x1 = max(0.0, min(x1, W - 1.0))
    y1 = max(0.0, min(y1, H - 1.0))
    x2 = max(0.0, min(x2, W - 1.0))
    y2 = max(0.0, min(y2, H - 1.0))
    if x2 < x1: x1, x2 = x2, x1
    if y2 < y1: y1, y2 = y2, y1
    fail = (x1 <= 1.0) or (y1 <= 1.0) or (x2 >= W - 2.0) or (y2 >= H - 2.0)
    return x1, y1, x2, y2, fail

"""
Oblicza wspólne pole dwóch oznaczeń
"""
def iarea(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    return iw * ih

"""
Rozszerza wymiary oznaczeń do proporcji szerokości : wysokości = 1 : 2
"""
def expand_no_leak(bbox, other_bboxes, W, H, m=0.0):
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1

    ex1 = int(x1 - m * w)
    ey1 = int(y1 - m * h)
    ex2 = int(x2 + m * w)
    ey2 = int(y2 + m * h)

    ex1, ey1, ex2, ey2, fail = clamp_box(ex1, ey1, ex2, ey2, W, H)
    if fail:
        return None, None
    ex = abs(ex2 - ex1)
    ey = abs(ey2 - ey1)

    if 2 * ex < ey:
        cx = (ex1 + ex2) // 2
        ex = ey // 2
        ex1, ex2 = cx - ex // 2, cx + ex // 2
        ex1, ey1, ex2, ey2, fail = clamp_box(ex1, ey1, ex2, ey2, W, H)
    elif 2 * ex > ey:
        cy = (ey1 + ey2) // 2
        ey = ex * 2
        ey1, ey2 = cy - ey // 2, cy + ey // 2
        ex1, ey1, ex2, ey2, fail = clamp_box(ex1, ey1, ex2, ey2, W, H)

    if fail:
        return None, None

    expanded = (ex1, ey1, ex2, ey2)
    for ob in other_bboxes:
        if iarea(expanded, ob) > 5_000.0:
            return None, None

    return expanded, m

"""
Przeksalowanie wycinków oznaczenia do 64 x 128 pikseli
"""
def letterbox_to_64x128(roi, blur_sigma=0.0):
    h, w = roi.shape[:2]
    # przeskalowanie do 64x128
    s = min(64.0 / w, 128.0 / h)
    new_w = max(1, int(round(w * s)))
    new_h = max(1, int(round(h * s)))

    resized = cv2.resize(roi, (64, 128), interpolation=cv2.INTER_AREA)
    if blur_sigma and blur_sigma > 0.0:
        k = int(max(3, 2 * int(blur_sigma * 3) + 1))
        resized = cv2.GaussianBlur(resized, (k, k), blur_sigma)

    pad_w = 64 - new_w
    pad_h = 128 - new_h
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top

    return resized, s, (left, right, top, bottom)

def main(video, top, out, margin_frac=0.0, max_upscale=2.0, blur_sigma=0.0):
    os.makedirs(out, exist_ok=True)
    # Tworzenie "bazy danych" przechowującej pozycje oznaczeń - zostanie wykorzystana w 03-MoveImage.py
    manifest_path = os.path.join(out, 'manifest.csv')
    man = open(manifest_path, 'w', newline='')
    man_writer = csv.writer(man)
    man_writer.writerow([
        'filename','frame','person_id',
        'bbox_x1','bbox_y1','bbox_x2','bbox_y2',
        'margin_used','scale','pads_left_right_top_bottom'
    ])

    frames = parse_top(top)
    wanted_frames = sorted(frames.keys())

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise SystemExit(f"Nie udało się otworzyć nagranie wideo: {video}")

    saved = skipped_small = skipped_overlap = 0

    for fno in wanted_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, fno)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        H, W = frame.shape[:2]

        bodies = [rec for rec in frames[fno] if rec['body_valid'] == 1]
        body_boxes = [tuple(rec['body']) for rec in bodies]

        for idx, rec in enumerate(bodies):
            person_id = rec['person_id']
            bbox = tuple(rec['body'])

            others = body_boxes[:idx] + body_boxes[idx+1:]
            expanded, margin_used = expand_no_leak(
                bbox, others, W, H,
                m=margin_frac
            )
            if expanded is None:
                skipped_overlap += 1
                continue

            x1, y1, x2, y2 = expanded
            bw, bh = (x2 - x1), (y2 - y1)
            if bh <= 0 or bw <= 0:
                continue

            # sprawdzenie, czy wycinek nie ma zbyt małych wymiarów
            height_upscale = 128.0 / bh
            if height_upscale > max_upscale:
                skipped_small += 1
                continue

            roi = frame[y1:y2, x1:x2].copy()

            crop, scale, pads = letterbox_to_64x128(roi, blur_sigma=blur_sigma)

            # Save
            fname = f"OTC_p{person_id:04d}_f{fno:06d}.png"
            print(f"Zapisano wycinek {fname} with x1: {x1}, y1: {y1}, x2: {x2}, y2: {y2}")

            cv2.imwrite(os.path.join(out, fname), crop)

            man_writer.writerow([fname, fno, person_id, x1, y1, x2, y2, margin_used, scale, pads])
            saved += 1

    man.close()
    cap.release()

    print(f"Gotowe. Zapisano: {saved}, Pominięto - zbyt małe: {skipped_small}, Pominięto - przecinanie się oznaczeń: {skipped_overlap}")
    print(f"Manifest: {manifest_path}")

if __name__ == '__main__':
    os.makedirs('crop')

    main('../../0-init/town_center/TownCentreXVID.mp4', '../../3-directed/town_center/original.csv', 'crop-original-v4/')
