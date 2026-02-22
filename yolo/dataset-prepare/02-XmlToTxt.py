"""
Roman Szlachtun, 272330

Konwertacja plików .xml do formatu oznaczeń, używanego do trenowania modelu YOLO.
"""

import os
from pathlib import Path

import xml.etree.ElementTree as ET


def xmlToTxt(xmlDir: Path):
    xmls = list(xmlDir.glob('*.xml'))

    for xmlFile in xmls:
        tree = ET.parse(xmlFile)
        root = tree.getroot()

        width = int(root.find('size').find('width').text)
        height = int(root.find('size').find('height').text)

        txtFilePath = Path('txts') / (xmlFile.stem + '.txt')
        with open(txtFilePath, 'w') as txtFile:
            for obj in root.findall('object'):
                bndbox = obj.find('bndbox')

                xmin = int(bndbox.find('xmin').text)
                ymin = int(bndbox.find('ymin').text)
                xmax = int(bndbox.find('xmax').text)
                ymax = int(bndbox.find('ymax').text)

                assert(0 <= xmin < xmax <= width)
                assert(0 <= ymin < ymax <= height)

                xmin = float(max(0, min(xmin, width)))
                ymin = float(max(0, min(ymin, height)))
                xmax = float(max(0, min(xmax, width)))
                ymax = float(max(0, min(ymax, height)))

                centerX = (xmin + xmax) / 2.0 / width
                centerY = (ymin + ymax) / 2.0 / height
                boxWidth = (xmax - xmin) / width
                boxHeight = (ymax - ymin) / height

                centerX = max(0.0, min(centerX, 1.0))
                centerY = max(0.0, min(centerY, 1.0))
                boxWidth = max(0.0, min(boxWidth, 1.0))
                boxHeight = max(0.0, min(boxHeight, 1.0))

                txtFile.write(f"{0} {centerX:.6f} {centerY:.6f} {boxWidth:.6f} {boxHeight:.6f}\n")

def main():
    os.makedirs('txts', exist_ok=True)
    xmlToTxt(Path('xmls'))

if __name__ == '__main__':
    main()