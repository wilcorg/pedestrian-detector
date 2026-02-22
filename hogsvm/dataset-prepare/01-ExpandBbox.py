"""
Roman Szlachtun, 272330

Poszerza wymiary ramek oznaczeń w zbiorze danych.

Uzyskane oznaczenia _nie_ są używane w testach.

Służy do zmniejszenia ilości pracy przy ręcznym (koniecznym) centrowaniu obrazów ludzi.
"""

"""
Ogranicza wymiary ramki oznaczenia do granic okna
"""
def clamp_bbox(left, top, right, bottom, W, H):
    left = max(0.0, min(W - 1.0, left))
    right = max(0.0, min(W - 1.0, right))
    top = max(0.0, min(H - 1.0, top))
    bottom = max(0.0, min(H - 1.0, bottom))
    if left > right: left, right = right, left
    if top > bottom: top, bottom = bottom, top
    return left, top, right, bottom


input_file = open('../../0-init/town_center/TownCentre-groundtruth.top', 'r', encoding='utf8')
output_training = open('training/groundtruth.csv', 'w', encoding='utf8')

for line in input_file:
    parts = line.strip().split(',')
    person, frame, head_valid, body_valid = map(int, parts[0:4])
    head_2_x, head_2_y, head_4_x, head_4_y = map(float, parts[4:8])
    body_2_x, body_2_y, body_4_x, body_4_y = map(float, parts[8:12])

    if frame < 2:
        continue

    body_width = body_4_x - body_2_x
    body_height = body_4_y - body_2_y

    body_extended_width = int(body_width * 1.2)
    body_extended_height = int(body_height * 1.2)

    body_center_x = body_2_x + body_width // 2
    body_center_y = body_2_y + body_height // 2

    body_2_x = body_center_x - body_extended_width / 2.0
    body_2_y = body_center_y - body_extended_height / 2.0

    body_4_x = body_center_x + body_extended_width / 2.0
    body_4_y = body_center_y + body_extended_height / 2.0

    body_2_x, body_2_y, body_4_x, body_4_y = clamp_bbox(body_2_x, body_2_y, body_4_x, body_4_y, 1920.0, 1080.0)

    output_training.write(
        f"{person},{frame},{head_valid},{body_valid},{head_2_x:.3f},{head_2_y:.3f},{head_4_x:.3f},{head_4_y:.3f},{body_2_x:.3f},{body_2_y:.3f},{body_4_x:.3f},{body_4_y:.3f}\n")

input_file.close()
output_training.close()
