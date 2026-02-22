"""
Roman Szlachtun, 272330

Program do demonstracji operacji średniej, stosowanej w
konwolucyjnych sieciach neuronowych.

Program iteracyjne zmniejsza rozdzielczość obrazu razy 2,
a następnie przywraca początkową rozdzielczość, duplikując piksele
"""

import cv2
import skimage.data
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import colormaps

"""
Wykonanie operacji średniej z jądrem 2x2 pikseli i krokiem 2 pikseli.
Zmniejsza rozdzielczość obrazu razy 2.

Wspiera obrazy o 1 kanale barwnym.
"""
def downsample_by2_avgpool(image_np: np.ndarray) -> np.ndarray:
    x = image_np

    H, W = x.shape
    # Przypadek dla wymiarów nieparzystych - rozszerzenie obrazu do wymiarów parzystych
    pad_h = H & 1
    pad_w = W & 1
    if pad_h or pad_w:
        x = np.pad(x, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

    y = x.reshape(x.shape[0] // 2, 2, x.shape[1] // 2, 2).mean(axis=1)
    return y[..., 0]

"""
Przeskalowywanie obrazu w górę.
Zwiększenie rozdzielczości obrazu razy 2.
"""
def upsample_to_src_pixelrep(image_np: np.ndarray, src: np.ndarray) -> np.ndarray:
    return cv2.resize(image_np, (src.shape[1], src.shape[0]), interpolation=cv2.INTER_NEAREST)

"""
Normalizuje jasność pikseli w obrazie do [0, 1].
Jest to konieczne dla nakładnia kolorowych masek na obraz. 
"""
def normalize01(x: np.ndarray) -> np.ndarray:
    m, M = float(np.min(x)), float(np.max(x))
    if M <= m + 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - m) / (M - m)).astype(np.float32)

"""
Nakłada kolorową maskę na obraz (albo zostawia odcienie szarości).
Normalizuje jasność pikseli w obrazie do [0, 255].
"""
def to_grayscale_uint8(gray01: np.ndarray) -> np.ndarray:
    vir = colormaps.get_cmap("gray")
    # vir = colormaps.get_cmap("inferno")
    rgb = vir(np.clip(gray01, 0, 1))[..., :3]
    return (rgb * 255 + 0.5).astype(np.uint8)

if __name__ == '__main__':
    cv2.imwrite("astronaut.png", skimage.data.astronaut())
    img = Image.open("astronaut.png").convert("L")  # grayscale
    src = np.asarray(img, dtype=np.float32) / 255.0

    down1 = downsample_by2_avgpool(src)      # H/2  x W/2
    down2 = downsample_by2_avgpool(down1)    # H/4  x W/4
    down3 = downsample_by2_avgpool(down2)    # H/8  x W/8
    down4 = downsample_by2_avgpool(down3)    # H/16 x W/16
    down5 = downsample_by2_avgpool(down4)    # H/32 x W/32
    down6 = downsample_by2_avgpool(down5)    # H/64 x W/64

    up_rep1 = upsample_to_src_pixelrep(down1, src)
    up_rep2 = upsample_to_src_pixelrep(down2, src)
    up_rep3 = upsample_to_src_pixelrep(down3, src)
    up_rep4 = upsample_to_src_pixelrep(down4, src)
    up_rep5 = upsample_to_src_pixelrep(down5, src)
    up_rep6 = upsample_to_src_pixelrep(down6, src)

    up_grayscale0 = to_grayscale_uint8(normalize01(src))
    up_grayscale1 = to_grayscale_uint8(normalize01(up_rep1))
    up_grayscale2 = to_grayscale_uint8(normalize01(up_rep2))
    up_grayscale3 = to_grayscale_uint8(normalize01(up_rep3))
    up_grayscale4 = to_grayscale_uint8(normalize01(up_rep4))
    up_grayscale5 = to_grayscale_uint8(normalize01(up_rep5))
    up_grayscale6 = to_grayscale_uint8(normalize01(up_rep6))

    Image.fromarray(up_grayscale1).save("down2_then_rep_grayscale1.png")
    Image.fromarray(up_grayscale2).save("down2_then_rep_grayscale2.png")
    Image.fromarray(up_grayscale3).save("down2_then_rep_grayscale3.png")
    Image.fromarray(up_grayscale4).save("down2_then_rep_grayscale4.png")
    Image.fromarray(up_grayscale5).save("down2_then_rep_grayscale5.png")
    Image.fromarray(up_grayscale6).save("down2_then_rep_grayscale6.png")

    plt.figure(figsize=(32, 7))
    plt.subplot(1,5,1); plt.imshow(src, cmap="gray"); plt.axis("off")
    plt.subplot(1,5,2); plt.imshow(up_grayscale2); plt.axis("off")
    plt.subplot(1,5,3); plt.imshow(up_grayscale3); plt.axis("off")
    plt.subplot(1,5,4); plt.imshow(up_grayscale4); plt.axis("off")
    plt.subplot(1,5,5); plt.imshow(up_grayscale5); plt.axis("off")
    plt.tight_layout()
    plt.show()
