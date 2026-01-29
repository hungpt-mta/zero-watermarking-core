"""utility.py

Small collection of helpers used by the core zero-watermarking code:

- Arnold transform / inverse (scrambling)
- A few common attacks (JPEG, Gaussian noise, median filter, mean filter)
- Quality / similarity metrics (PSNR, NC, NCC)

This file is derived from the user's consolidated `Utility.py` (cleaned for repo use).
"""

from __future__ import annotations

import math
from typing import Tuple

import cv2
import numpy as np
from skimage.util import random_noise


# ---- Arnold parameters (kept as in the original code) ----
a = 1
b = 1


def arnoldTransform(image: np.ndarray, key: int) -> np.ndarray:
    """Apply Arnold cat map `key` iterations (square images expected)."""
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (a * b * x + x + a * y) % s[0]
    ymap = (b * x + y) % s[0]
    img = image
    for _ in range(key):
        img = img[xmap, ymap]
    return img


def arnoldInverseTransform(image: np.ndarray, key: int) -> np.ndarray:
    """Inverse Arnold cat map `key` iterations (square images expected)."""
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (x - a * y) % s[0]
    ymap = (-b * x + a * b * y + y) % s[0]
    img = image
    for _ in range(key):
        img = img[xmap, ymap]
    return img


# ---- Attacks ----
def gaussian_noise(img: np.ndarray, mean: float = 0.0, var: float = 0.05) -> np.ndarray:
    """Add Gaussian noise using scikit-image (returns float image in [0,1] if input is float)."""
    return random_noise(img, mode="gaussian", mean=mean, var=var, clip=True)


def jpegcompression_attack(filename: str, image: np.ndarray, quality: int = 100) -> np.ndarray:
    """Apply JPEG compression by writing and re-reading the image."""
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    ret = cv2.imread(filename)
    return ret


def meadianfilter_attack(img: np.ndarray, ksize: int) -> np.ndarray:
    """Median filter attack (OpenCV)."""
    return cv2.medianBlur(img, int(ksize))


def meanfilter_attack(img: np.ndarray, ksize: int) -> np.ndarray:
    """Mean filter attack (OpenCV)."""
    return cv2.blur(img, (int(ksize), int(ksize)))


# ---- Metrics ----
def mse(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.mean(np.power(A.astype(np.float32) - B.astype(np.float32), 2)))


def psnr(A: np.ndarray, B: np.ndarray) -> float:
    """PSNR for uint8 images (0..255)."""
    m = mse(A, B)
    if m <= 0:
        return float("inf")
    return float(20 * np.log10(255.0 / math.sqrt(m)))


def calculate_psnr_color(img1: np.ndarray, img2: np.ndarray) -> float:
    """Average PSNR over RGB channels for uint8 images."""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    m = np.mean((img1 - img2) ** 2, axis=(0, 1))
    ps = 10 * np.log10((255.0 ** 2) / m)
    return float(np.mean(ps))


def nc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized Correlation (NC) on binary images (threshold at 127)."""
    original_bin = (img1 <= 127).astype(np.uint8)
    extracted_bin = (img2 <= 127).astype(np.uint8)
    numerator = np.sum(original_bin * extracted_bin)
    denominator = np.sum(original_bin)
    return float(numerator / denominator) if denominator != 0 else 0.0


def ncc(img1: np.ndarray, img2: np.ndarray) -> float:
    """Normalized cross-correlation (NCC) for two same-size arrays."""
    a1 = img1.astype(np.float32).ravel()
    a2 = img2.astype(np.float32).ravel()
    a1 -= a1.mean()
    a2 -= a2.mean()
    denom = (np.linalg.norm(a1) * np.linalg.norm(a2))
    if denom == 0:
        return 0.0
    return float(np.dot(a1, a2) / denom)
