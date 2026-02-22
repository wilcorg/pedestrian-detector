"""
Roman Szlachtun, 272330

Nakłada oznaczenia zbioru danych na klatkę nagrania.
"""

from PIL import Image, ImageDraw

import xml.etree.ElementTree as ET

def draw(imgName: int):
    image = Image.open(f'../frames/{imgName}.png')
    draw = ImageDraw.Draw(image)

    xml = ET.parse(f'xmls/{imgName}.xml')
    root = xml.getroot()

    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')

        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=4)

    image.show()

def main():
    # draw(1344)
    draw(4217)

if __name__ == '__main__':
    main()