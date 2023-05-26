import cv2
from encoder.color_encoder import color_encoder
import numpy as np

def fill(img, height, width, channel_num):
    img_filled = img.copy()

    if height % 8 != 0:
        filler = np.ones((8 - (height % 8), width, channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img, filler], axis=0)

    if width % 8 != 0:
        filler = np.ones((height + 8 - (height % 8), 8 - (width % 8), channel_num), dtype=np.uint8) * 128
        img_filled = np.concatenate([img_filled, filler], axis=1)

    return img_filled


if __name__ == '__main__':
    img = cv2.imread('test_img/in.pnm', cv2.IMREAD_COLOR)
    height, width = img.shape[:2]
    quality = 90

    img = fill(img, height, width, 3)
    color_encoder('test_img/out.jpeg', img, height, width, quality)
