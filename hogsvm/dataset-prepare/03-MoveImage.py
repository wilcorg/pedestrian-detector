"""
Roman Szlachtun, 272330

Przemieszcza grupowo wycinki z różnych klatek w górę/dół/lewo/prawo oraz przybliża/oddala.

Pozwala centrować kilkadziesiąt wycinków jednocześnie.

Centrowanie jest konieczne dla wytrenowania dobrego detektora SVM.

Utrzymuje "bazę danych" pozycji wszystkich wycinków,
aktualizuje pozycję wybranych wycinków i wycina ich ponownie z klatki.
"""

import csv
import os
import pathlib
import sys
import tempfile
from pathlib import Path
from typing import List

import cv2
from PIL import Image, ImageDraw


def offset(manifest_src: Path, move_imgs: List[Path], dx: int, dy: int, scale: float, showLines: bool) -> None:
    filenames: List[str] = list(map(lambda p: p.name, move_imgs))

    fd, temp_path = tempfile.mkstemp(suffix='.tmp')
    os.close(fd)

    with open(manifest_src, 'r') as in_file, open(temp_path, 'w', newline='') as out_file:
        reader = csv.DictReader(in_file)
        fieldnames = reader.fieldnames

        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            if row['filename'] in filenames:
                x1 = int(row['bbox_x1'])
                y1 = int(row['bbox_y1'])
                x2 = int(row['bbox_x2'])
                y2 = int(row['bbox_y2'])

                x1 += dx
                x2 += dx
                y1 += dy
                y2 += dy

                w: float = (x2 - x1) / scale
                h: float = (y2 - y1) / scale
                cx: float = (x1 + x2) / 2.0
                cy: float = (y1 + y2) / 2.0

                nx1: float = cx - w / 2.0
                nx2: float = cx + w / 2.0
                ny1: float = cy - h / 2.0
                ny2: float = cy + h / 2.0

                row['bbox_x1'] = int(nx1)
                row['bbox_y1'] = int(ny1)
                row['bbox_x2'] = int(nx2)
                row['bbox_y2'] = int(ny2)

                if nx1 < 0.0 or ny1 < 0.0 or nx2 > 1920.0 or ny2 > 1080.0:
                    print(f"Uwaga: ramka wychodzi poza granice okna: {row['filename']} {nx1},{ny1},{nx2},{ny2}")
                else:
                    image = cv2.imread('frames/' + row['frame'] + '.png')
                    cropped = image[int(ny1):int(ny2), int(nx1):int(nx2)]
                    resized = cv2.resize(cropped, (64, 128), interpolation=cv2.INTER_AREA)

                    os.remove('to_move/' + row['filename'])
                    cv2.imwrite('to_move/' + row['filename'], resized)

                    if showLines:
                        """
                        Rysuje poziome linie, w których powinna zmieścić się wysokość osoby
                        oraz pionową linię, po środku której powinna znajdować się osoba.
                        
                        Pozwala to bardzo szybko i dokładnie centrować wycinki osób, 
                        co jest konieczne dla trenowania dobrego detektora SVM.
                        
                        Na koniec centrowania showLines=false ponownie wycina dokładnie
                        wycentrowane osoby, nie rysując linii na nich.
                        """
                        img = Image.open('to_move/' + row['filename'])
                        draw = ImageDraw.Draw(img)

                        draw.line((0, 16, 64, 16), fill="red", width=1)
                        draw.line((0, 112, 64, 112), fill="red", width=1)
                        draw.line((32, 0, 32, 128), fill="green", width=1)

                        img.save('to_move/' + row['filename'])

            writer.writerow(row)
    os.replace(temp_path, manifest_src)

def main():
    values = sys.argv
    dx_: int = int(values[1])
    dy_: int = int(values[2])
    scale_: float = float(values[3])
    showLines: bool = False

    manifest_src_: Path = pathlib.Path('manifest.csv')
    move_imgs_: List[Path] = list(pathlib.Path('to_move').rglob('*.png'))

    offset(manifest_src_, move_imgs_, dx_, dy_, scale_, showLines)

if __name__ == "__main__":
    main()
