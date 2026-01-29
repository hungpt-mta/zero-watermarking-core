import numpy as np
import math
import cv2
from skimage.metrics import peak_signal_noise_ratio
from skimage import img_as_float
from utils import utils_image as util
from scipy.ndimage.filters import uniform_filter, median_filter
from scipy.ndimage.measurements import variance
import torch
from skimage.util import random_noise
#---- Default parameters ----
a = 1
b = 1
key = 1
#---

def arnoldTransform(image: np.ndarray, key: int, ) -> np.ndarray:
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (a * b * x + x + a * y) % s[0]
    ymap = (b * x + y) % s[0]
    img = image
    for r in range(key):
        img = img[xmap, ymap]
    return img

def arnoldInverseTransform(image: np.ndarray, key: int) -> np.ndarray:
    s = image.shape
    x, y = np.meshgrid(range(s[0]), range(s[0]), indexing="ij")
    xmap = (x - a * y) % s[0]
    ymap = (-b * x + a * b * y + y) % s[0]
    img = image
    for r in range(key):
        img = img[xmap, ymap]
    return img

#----- Attacks-------#
def gaussian_noise(img,mean=0, var=0.05):
	gauss_img = random_noise(img, mode='gaussian', mean=mean, var=var, clip=True)
	return gauss_img



def jpegcompression_attack(filename, image, quality=100):
    cv2.imwrite(filename, image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    ret = cv2.imread(filename)
    return ret


def meadianfilter_attack(img, ksize):
        return cv2.medianBlur(img, ksize)


def meanfilter_attack(img, ksize):
        return cv2.blur(img, (ksize, ksize))

#---- Measures-----
def norm_data(data):
    mead_data = np.mean(data)
    std_data = np.std(data, ddof=1)
    return (data-mead_data)/std_data

def nc(img1, img2):
    # Convert to binary (0 or 1)
    original_bin = (img1 <= 127).astype(np.uint8)
    extracted_bin = (img2 <= 127).astype(np.uint8)
    #print(extracted_bin)
    # Compute NC
    numerator = np.sum(original_bin * extracted_bin)
    denominator = np.sum(original_bin)

    res = numerator / denominator if denominator != 0 else 0
    return res

def mse(A, B):
    return np.mean(np.power(A - B, 2))
def psnr(A, B):
    return 20 * np.log10(255 / np.sqrt(mse(A, B)))

def calculate_psnr_color(img1, img2):
    # Chuyển ảnh về kiểu float32
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)
    
    mse = np.mean((img1 - img2) ** 2, axis=(0, 1))  # MSE trên từng kênh R,G,B
    psnr = 10 * np.log10((255 ** 2) / mse)  # PSNR từng kênh
    psnr_avg = np.mean(psnr)  # Lấy trung bình 3 kênh

    return psnr_avg