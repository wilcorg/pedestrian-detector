"""
Roman Szlachtun, 272330

Program do demonstracji reprezentacji obrazu .png w przestrzeni kolorów YUV
"""

import cv2
import numpy as np
from pathlib import Path
from DirectoryAnchor import DATASET_DIR

def save_yuv_canonical(png_path, out_dir="yuv_canonical"):
    png_path = Path(png_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(str(png_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Nie udało się odczytać obraz: {png_path}")

    yuv = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV)
    Y, U, V = cv2.split(yuv)

    cv2.imwrite(str(out_dir / "Y.png"), Y)

    u_only = np.empty_like(yuv)
    u_only[:, :, 0] = 128
    u_only[:, :, 1] = U
    u_only[:, :, 2] = 128
    u_rgb = cv2.cvtColor(u_only, cv2.COLOR_YUV2RGB)
    cv2.imwrite(str(out_dir / "U_canonical_rgb.png"), u_rgb)

    v_only = np.empty_like(yuv)
    v_only[:, :, 0] = 128
    v_only[:, :, 1] = 128
    v_only[:, :, 2] = V
    v_rgb = cv2.cvtColor(v_only, cv2.COLOR_YUV2RGB)
    cv2.imwrite(str(out_dir / "V_canonical_rgb.png"), v_rgb)


if __name__ == "__main__":
    p = DATASET_DIR / 'frames' / '2.png'
    save_yuv_canonical(p, out_dir="yuv_out")
