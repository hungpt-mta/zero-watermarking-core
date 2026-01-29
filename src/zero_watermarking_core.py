"""zero_watermarking_core.py

Core implementation of the proposed zero-watermarking scheme.

This module is extracted/condensed from the provided notebook and kept focused on:

* Registration: `generate_zero_watermark(...)`
* Authentication (extraction + verification): `authenticate(...)`
* Geometric correction: `correcting_image(...)` using `TempMatcher`

The code intentionally keeps the original algorithmic flow and I/O conventions
(e.g., writing keypoints/descriptors/MS/OS/RC tuples to disk) to match the paper
pipeline and ease editorial review.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import statistics
import time
from dataclasses import dataclass
from typing import List, Tuple

import cv2
import numpy as np
import pywt
from cryptography.hazmat.primitives.asymmetric import dsa
from scipy.fftpack import dct

from .utility import arnoldInverseTransform, nc, calculate_psnr_color


def _rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image around its center (OpenCV), keeping same size."""
    h, w = image.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


# ============================================================
# 1) Signcryption-like primitives (as used in the notebook)
# ============================================================

def int_to_bytes(n: int) -> bytes:
    """Encode a non-negative integer into big-endian bytes (no leading zeros)."""
    if n == 0:
        return b"\x00"
    length = (n.bit_length() + 7) // 8
    return n.to_bytes(length, "big")


def hash_int(n: int) -> int:
    """HASH(n) = SHA-256(int_to_bytes(n)) interpreted as an integer."""
    h = hashlib.sha256(int_to_bytes(n)).digest()
    return int.from_bytes(h, "big")


def hash_pair(R: int, S: int) -> int:
    """HASH(R || S) with length-prefix encoding to avoid ambiguity."""
    rb = int_to_bytes(R)
    sb = int_to_bytes(S)
    data = len(rb).to_bytes(2, "big") + rb + len(sb).to_bytes(2, "big") + sb
    h = hashlib.sha256(data).digest()
    return int.from_bytes(h, "big")


@dataclass
class SystemParams:
    p: int
    q: int
    g: int


@dataclass
class UserKeys:
    sk: int  # x
    pk: int  # y = g^x mod p


def generate_system_params_and_key(lp: int = 2048, lq: int = 256) -> Tuple[SystemParams, UserKeys]:
    """Generate (p,q,g) using DSA params and a user key pair (x,y)."""
    if lp != 2048 or lq != 256:
        raise ValueError("Supported only L=2048, N=256 (DSA) to match the notebook setup.")

    dsa_params = dsa.generate_parameters(key_size=lp)
    nums = dsa_params.parameter_numbers()
    p, q, g = nums.p, nums.q, nums.g
    params = SystemParams(p=p, q=q, g=g)

    x = secrets.randbelow(q - 1) + 1
    y = pow(g, x, p)
    keys = UserKeys(sk=x, pk=y)
    return params, keys


def generate_user_keys(params: SystemParams) -> UserKeys:
    """Generate an additional user key pair (x,y) under the same (p,q,g)."""
    p, q, g = params.p, params.q, params.g
    x = secrets.randbelow(q - 1) + 1
    y = pow(g, x, p)
    return UserKeys(sk=x, pk=y)


def encrypt(params: SystemParams, xs: int, yr: int, P: int) -> Tuple[int, int]:
    """Algorithm 2 (Encryption): returns (R,C)."""
    p, g = params.p, params.g
    if not (0 < P < p):
        raise ValueError("P must be in (0, p).")

    Se = pow(yr, xs, p)
    R = hash_int(P)
    Ke = hash_pair(R, Se)
    C = (P * pow(g, Ke, p)) % p
    return R, C


def decrypt_and_verify(params: SystemParams, xr: int, ys: int, R: int, C: int) -> Tuple[int | None, bool]:
    """Algorithm 3 (Decryption & Authentication): returns (M, ok)."""
    p, g = params.p, params.g
    Sd = pow(ys, xr, p)
    Kd = hash_pair(R, Sd)

    g_pow = pow(g, Kd, p)
    g_pow_inv = pow(g_pow, -1, p)
    M = (C * g_pow_inv) % p

    V = hash_int(M)
    ok = (V == R)
    if not ok:
        return None, False
    return M, True


def os_matrix_to_int(os_matrix: np.ndarray, rows: int = 32, cols: int = 32) -> int:
    """Map a {0,1} OS matrix (rows x cols) into an integer P (row-major, MSB first)."""
    os_matrix = np.asarray(os_matrix)
    if os_matrix.shape != (rows, cols):
        raise ValueError(f"OS size must be ({rows}, {cols}), got {os_matrix.shape}")
    bits = os_matrix.astype(np.uint8)
    if not np.array_equal(bits, bits & 1):
        raise ValueError("OS must contain only bits 0 or 1")

    flat = bits.ravel()
    P = 0
    for b in flat:
        P = (P << 1) | int(b)
    return P


def int_to_os_matrix(P: int, rows: int = 32, cols: int = 32) -> np.ndarray:
    """Inverse of `os_matrix_to_int`: integer -> {0,1} matrix of shape (rows, cols)."""
    total_bits = rows * cols
    if P < 0 or P >= (1 << total_bits):
        raise ValueError("P is out of range for the given matrix size")

    bits = [(P >> (total_bits - 1 - i)) & 1 for i in range(total_bits)]
    return np.array(bits, dtype=np.uint8).reshape(rows, cols)


# ============================================================
# 2) Entropy ROI scoring + DWT-DCT transform
# ============================================================

def dct2(block: np.ndarray) -> np.ndarray:
    return dct(dct(block.T, norm="ortho").T, norm="ortho")


def apply_dwt_dct(img: np.ndarray) -> np.ndarray:
    coeffs = pywt.dwt2(img, "haar")
    LL, (LH, HL, HH) = coeffs
    LL_dct = dct2(LL)
    return np.uint8(LL_dct)


def euclidean_similarity(desc1: np.ndarray, desc2: np.ndarray) -> float:
    v1_norm = desc1 / np.linalg.norm(desc1)
    v2_norm = desc2 / np.linalg.norm(desc2)
    return float(np.dot(v1_norm, v2_norm))


def _to_gray(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def entropy_gradient_roi(gray: np.ndarray, roi: Tuple[int, int, int, int]) -> float:
    """Gradient-orientation entropy for a ROI: used to re-rank candidate keypoints."""
    x, y, w, h = roi
    patch = gray[y : y + h, x : x + w]
    if patch.size == 0:
        return 0.0

    gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=False)

    # histogram over orientations (weighted by magnitude)
    bins = 36
    hist = np.zeros(bins, dtype=np.float32)
    ang = ang.ravel()
    mag = mag.ravel()
    for a, m in zip(ang, mag):
        b = int((a / (2 * math.pi)) * bins) % bins
        hist[b] += m
    s = hist.sum()
    if s <= 0:
        return 0.0
    p = hist / s
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


# ============================================================
# 3) TempMatcher + correcting
# ============================================================

class TempMatcher:
    """Feature-based template matcher (ORB/AKAZE/KAZE/SIFT).

    Loads precomputed keypoints/descriptors for a reference image from disk and
    estimates a RANSAC homography against a queried image. Returns an approximate
    [dx, dy, rotation(deg), scale] parameter tuple as in the notebook.
    """

    def __init__(self, temp, descriptor: str = "SIFT", filename: str = "", numOfFeaturesPoints: int = 50):
        self.detector = self.get_des(descriptor, numOfFeaturesPoints)
        self.bf = self.get_matcher(descriptor)
        self.filename = filename

        if self.detector == 0:
            raise ValueError("Unknown Descriptor!")

        if len(temp.shape) > 2:
            temp = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)

        self.template = temp

        # Build image json name (same convention as the notebook)
        img_base = os.path.basename(filename)
        img_json = os.path.splitext(img_base)[0] + ".json"

        key_dir = "./keypoints"
        des_dir = "./descriptors"

        new_kp_first = os.path.join(key_dir, "keypoint_1_" + img_json)
        new_des_first = os.path.join(des_dir, "des_1_" + img_json)

        kpfile: List[cv2.KeyPoint] = []
        des_list: List[list] = []

        if os.path.exists(new_kp_first) and os.path.exists(new_des_first):
            for pidx in range(1, int(numOfFeaturesPoints) + 1):
                kp_path = os.path.join(key_dir, f"keypoint_{pidx}_" + img_json)
                des_path = os.path.join(des_dir, f"des_{pidx}_" + img_json)

                if not (os.path.exists(kp_path) and os.path.exists(des_path)):
                    break

                with open(kp_path, "r", encoding="utf-8") as f:
                    xy = json.loads(f.read())
                x, y = float(xy[0]), float(xy[1])
                kpt = cv2.KeyPoint(x=x, y=y, size=1.0, angle=-1.0, response=0.0, octave=0, class_id=int(pidx))
                kpfile.append(kpt)

                with open(des_path, "r", encoding="utf-8") as f:
                    dv = json.loads(f.read())
                des_list.append(dv)

            if descriptor in ("ORB", "AKAZE"):
                desfile = np.array(des_list, dtype=np.uint8)
            else:
                desfile = np.array(des_list, dtype=np.float32)
        else:
            # In this minimal repo, stored features are expected.
            raise FileNotFoundError("Precomputed keypoints/descriptors were not found in ./keypoints and ./descriptors")

        self.kp1 = kpfile
        self.des1 = desfile
        self.H = np.eye(3, dtype=np.float32)
        self.center = np.float32([temp.shape[1], temp.shape[0]]).reshape([1, 2]) / 2

    def get_des(self, name, numOfFeaturesPoints):
        return {
            "ORB": cv2.ORB_create(nfeatures=numOfFeaturesPoints, scoreType=cv2.ORB_HARRIS_SCORE),
            "AKAZE": cv2.AKAZE_create(),
            "KAZE": cv2.KAZE_create(extended=False),
            "SIFT": cv2.SIFT_create(),
        }.get(name, 0)

    def get_matcher(self, name):
        return {
            "ORB": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            "AKAZE": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
            "KAZE": cv2.BFMatcher(),
            "SIFT": cv2.BFMatcher(),
        }.get(name, 0)

    def match(self, img, showflag: int = 0):
        # keep signature compatibility; `showflag` is unused in the minimal repo
        if len(img.shape) > 2:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        kp2, des2 = self.detector.detectAndCompute(img, None)
        if kp2 is None or len(kp2) < 5 or des2 is None:
            return [0, 0, 0, 1], 0, 0

        matches = self.bf.knnMatch(self.des1, des2, k=2)

        pts1, pts2 = [], []
        count = 0
        for m, n in matches:
            if m.distance < 0.5 * n.distance:
                if m.queryIdx < len(self.kp1):
                    pts2.append(kp2[m.trainIdx].pt)
                    pts1.append(self.kp1[m.queryIdx].pt)
                    count += 1

        pts1 = np.float32(pts1)
        pts2 = np.float32(pts2)

        inliner = 0
        if count > 4:
            self.H, mask = cv2.findHomography(pts1 - self.center, pts2 - self.center, cv2.RANSAC, 3.0)
            if mask is not None:
                inliner = int(np.count_nonzero(mask))

        param = self.getpoc()
        return param, count, inliner

    def getpoc(self):
        Affine = self.H
        if Affine is None:
            return [0, 0, 0, 1]

        A2 = Affine * Affine
        scale = math.sqrt(np.sum(A2[0:2, 0:2]) / 2.0)
        theta = math.atan2(Affine[0, 1], Affine[0, 0])
        theta = theta * 180.0 / math.pi
        Trans = np.dot(np.linalg.inv(Affine[0:2, 0:2]), Affine[0:2, 2:3])
        return [Trans[0], Trans[1], theta, scale]


def correcting_image(
    folder_path: str,
    attack_path: str,
    reattack_path: str,
    ref_filename: str,
    attack_filename: str,
    numOfFeaturesPoints: int,
    descriptor: str = "SIFT",
):
    """Geometric correction (registration) for an attacked image using TempMatcher.

    This follows the original notebook logic: for files tagged as scaled/rotated/translated,
    it estimates rotation+scale, applies corrections, and saves `re_<attack_filename>`.
    """

    ref_path = os.path.join(folder_path, ref_filename)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference image not found: {ref_path}")
    ref = cv2.imread(ref_path)
    if ref is None:
        raise ValueError(f"Cannot read reference image: {ref_path}")

    comp_path = os.path.join(attack_path, attack_filename)
    if not os.path.exists(comp_path):
        raise FileNotFoundError(f"Attacked image not found: {comp_path}")
    cmp = cv2.imread(comp_path)
    if cmp is None:
        raise ValueError(f"Cannot read attacked image: {comp_path}")

    matcher = TempMatcher(ref, descriptor, ref_filename, numOfFeaturesPoints)

    if re.findall(ref_filename, attack_filename.replace("jpg", "png")) and (
        re.findall("scaled", attack_filename)
        or re.findall("rotated", attack_filename)
        or re.findall("translated", attack_filename)
    ):
        matcho = matcher.match(cmp, 1)
        r_img = _rotate_image(cmp, -1 * matcho[0][2])
        rs_img = cv2.resize(r_img, None, fx=1 / matcho[0][3], fy=1 / matcho[0][3])
        rs_img = cv2.resize(rs_img, (ref.shape[0], ref.shape[1]))

        # Translation step is intentionally disabled in the original notebook (0 > 1)
        matcht = matcher.match(rs_img, 1)
        if isinstance(matcht[0][0], (np.ndarray)) and isinstance(matcht[0][1], (np.ndarray)) and 0 > 1:
            traslated_Matrix = np.float32([[1, 0, -1.0 * matcht[0][0][0]], [0, 1, -1.0 * matcht[0][1][0]]])
            out_img = cv2.warpAffine(rs_img, traslated_Matrix, (ref.shape[1], ref.shape[0]))
        else:
            out_img = rs_img

    elif re.findall(ref_filename, attack_filename.replace("jpg", "png")):
        out_img = cmp
    else:
        raise ValueError(
            f"attack_filename does not match reference filename pattern: {attack_filename} vs {ref_filename}"
        )

    os.makedirs(reattack_path, exist_ok=True)
    new_filename = "re_" + attack_filename
    new_img_path = os.path.join(reattack_path, new_filename)
    cv2.imwrite(new_img_path, out_img)
    return out_img


# ============================================================
# 4) Registration & authentication
# ============================================================

def generate_zero_watermark(filename, folder_path, wm_ar, N, params, xs, yr, enc_times):
    """Register: extract stable patches, build MS/OS, encrypt OS->(R,C), and store tuples."""
    img_enc_times = []
    img_zw_times = []

    img_path = os.path.join(folder_path, filename)
    color_img = cv2.imread(img_path)
    Ycrcb_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2YCrCb)
    Y_split = Ycrcb_img[:, :, 0]
    image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    t_zw = time.perf_counter()

    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(image, None)
    if len(kps) == 0:
        raise ValueError("No SIFT keypoints found!")

    idx = np.argsort([-kp.response for kp in kps])
    kp_sorted = [kps[i] for i in idx]
    des = des[idx] if des is not None else None

    # re-rank by entropy score
    for i in range(len(kp_sorted)):
        kp_sorted[i].response = 0.0
    for i in range(min(100, len(kp_sorted))):
        x, y = int(kp_sorted[i].pt[0]), int(kp_sorted[i].pt[1])
        if (x >= 32) and (x <= width - 32) and (y >= 32) and (y <= height - 32):
            roi = (x - 32, y - 32, 32, 32)
            new_score = entropy_gradient_roi(image, roi)
        else:
            new_score = 0.0
        kp_sorted[i].response = new_score

    idx = np.argsort([-kp.response for kp in kp_sorted])
    kp_sorted = [kp_sorted[i] for i in idx]
    des = des[idx] if des is not None else None

    os.makedirs("./descriptors", exist_ok=True)
    os.makedirs("./keypoints", exist_ok=True)
    os.makedirs("./ms_img", exist_ok=True)
    os.makedirs("./os_img", exist_ok=True)
    os.makedirs("./RCs", exist_ok=True)

    p = 0
    for i in range(min(N, len(kps))):
        x, y = int(kp_sorted[i].pt[0]), int(kp_sorted[i].pt[1])
        if (x >= 32) and (x <= width - 32) and (y >= 32) and (y <= height - 32):
            p += 1

            desfile = "./descriptors/" + "des_" + str(p) + "_" + filename.replace(".png", ".json")
            with open(desfile, "w", encoding="utf-8") as f:
                f.write(json.dumps(des[i].tolist()))

            keypointfile = "./keypoints/" + "keypoint_" + str(p) + "_" + filename.replace(".png", ".json")
            with open(keypointfile, "w", encoding="utf-8") as f:
                f.write(json.dumps([x, y]))

            patch = Y_split[y - 32 : y + 32, x - 32 : x + 32]
            img1 = apply_dwt_dct(patch)

            ms_img = img1.copy()
            for r in range(img1.shape[0]):
                for c in range(img1.shape[1]):
                    ms_img[r][c] = 255 if (img1[r][c] >= 127) else 0
            cv2.imwrite("./ms_img/" + "ms_" + str(p) + "_" + filename, ms_img)

            os_img = img1.copy()
            for r in range(img1.shape[0]):
                for c in range(img1.shape[1]):
                    os_img[r][c] = ms_img[r][c] ^ wm_ar[r][c]
            cv2.imwrite("./os_img/" + "os_" + str(p) + "_" + filename, os_img)

            os_img_bn = os_img / 255
            P = os_matrix_to_int(os_img_bn, rows=32, cols=32)
            if P >= params.p:
                raise ValueError("P >= p. Increase lp or reduce the OS size to ensure P < p.")

            t_enc0 = time.perf_counter()
            R, C = encrypt(params, xs, yr, P)
            t_enc1 = time.perf_counter()
            enc_times.append(t_enc1 - t_enc0)
            img_enc_times.append(t_enc1 - t_enc0)
            img_zw_times.append(t_enc1 - t_zw)

            RCoutname = "./RCs/" + "rc_" + str(p) + "_" + filename + ".txt"
            with open(RCoutname, "w", encoding="utf-8") as f:
                f.write(f"{R} {C} {P}")

    if img_enc_times:
        enc_time = round(statistics.mean(img_enc_times) * 1000, 2)
    else:
        enc_time = 0.0

    if img_zw_times:
        zw_time = round(statistics.mean(img_zw_times) * 1000, 2)
    else:
        zw_time = 0.0

    return enc_time, zw_time


def authenticate(
    orgfilename,
    filename,
    folder_path,
    attacked_img,
    wm_img,
    N,
    T,
    key,
    ex_wms_path,
    params,
    xr,
    ys,
    dec_times,
    success_count,
):
    """Authenticate: extract and verify zero-watermark from an attacked/corrected image."""

    img_dec_times = []

    org_path = os.path.join(folder_path, orgfilename)
    orgimage = cv2.imread(org_path)
    cmp_path = os.path.join("./attacked_imgs", filename.replace("re_", ""))
    cmp_img = cv2.imread(cmp_path)

    image = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2GRAY)
    Ycrcb_img = cv2.cvtColor(attacked_img, cv2.COLOR_BGR2YCrCb)
    Y_split = Ycrcb_img[:, :, 0]
    height, width = image.shape[:2]
    t_ex = time.perf_counter()

    sift = cv2.SIFT_create()
    kps, des = sift.detectAndCompute(image, None)
    if len(kps) == 0:
        return 0, 0

    idx = np.argsort([-kp.response for kp in kps])
    kp_sorted = [kps[i] for i in idx]
    des = des[idx] if des is not None else None

    for i in range(len(kp_sorted)):
        kp_sorted[i].response = 0.0
    for i in range(min(100, len(kp_sorted))):
        x, y = int(kp_sorted[i].pt[0]), int(kp_sorted[i].pt[1])
        if (x >= 32) and (x <= width - 32) and (y >= 32) and (y <= height - 32):
            roi = (x - 32, y - 32, 32, 32)
            new_score = entropy_gradient_roi(image, roi)
        else:
            new_score = 0.0
        kp_sorted[i].response = new_score

    idx = np.argsort([-kp.response for kp in kp_sorted])
    kp_sorted = [kp_sorted[i] for i in idx]
    des = des[idx] if des is not None else None

    q = 0
    W = []
    for i in range(min(N, len(kp_sorted))):
        x, y = int(kp_sorted[i].pt[0]), int(kp_sorted[i].pt[1])
        if (x >= 32) and (x <= width - 32) and (y >= 32) and (y <= height - 32):
            max_similarity = 0
            index = 0

            for desfilename in os.listdir("./descriptors"):
                if re.findall(orgfilename.replace(".png", ".json"), desfilename):
                    with open("./descriptors/" + desfilename, "r", encoding="utf-8") as f:
                        rawDes = json.loads(f.read())
                    desfile = np.array(rawDes, dtype=np.float32)
                    similarity = euclidean_similarity(des[i], desfile)
                    if max_similarity < similarity:
                        max_similarity = similarity
                        index = desfilename.split("_")[1]

            if max_similarity >= T:
                q += 1
                patch = Y_split[y - 32 : y + 32, x - 32 : x + 32]
                img1 = apply_dwt_dct(patch)

                ms_img = img1
                for r in range(img1.shape[0]):
                    for c in range(img1.shape[1]):
                        ms_img[r][c] = 255 if (img1[r][c] >= 127) else 0

                rc_path = "./RCs/" + "rc_" + str(index) + "_" + orgfilename + ".txt"
                with open(rc_path, "r", encoding="utf-8") as f:
                    R, C, P = map(int, f.readline().split())

                t_dec0 = time.perf_counter()
                M, ok = decrypt_and_verify(params, xr, ys, R, C)
                t_dec1 = time.perf_counter()
                dec_times.append(t_dec1 - t_dec0)
                img_dec_times.append(t_dec1 - t_dec0)

                if ok and M == P:
                    success_count += 1

                os_img_bn = int_to_os_matrix(M, rows=32, cols=32) if ok else None
                os_img = 255 * os_img_bn

                wm_ar_xor = img1
                for r in range(img1.shape[0]):
                    for c in range(img1.shape[1]):
                        wm_ar_xor[r][c] = ms_img[r][c] ^ os_img[r][c]

                wm_ar_inv1 = arnoldInverseTransform(wm_ar_xor, key)
                W.append(wm_ar_inv1)

    # Majority vote fusion
    Wq = np.zeros((32, 32), dtype=np.uint8)
    for i in range(Wq.shape[0]):
        for j in range(Wq.shape[1]):
            totalij = 0
            for k in range(len(W)):
                totalij = totalij + W[k][i][j] / 255
            Wq[i][j] = 255 if totalij >= len(W) / 2 else 0

    t_ex1 = time.perf_counter()
    ex_time = round((t_ex1 - t_ex) * 1000, 2)

    if img_dec_times:
        dec_time = round(statistics.mean(img_dec_times) * 1000, 2)
    else:
        dec_time = 0.0

    os.makedirs(ex_wms_path, exist_ok=True)
    ex_filename = "ex_" + filename
    new_ex_path = os.path.join(ex_wms_path, ex_filename)
    cv2.imwrite(new_ex_path, Wq)

    resNC = round(nc(wm_img, Wq), 3)
    resPSNR = round(
        calculate_psnr_color(orgimage, cv2.resize(cmp_img, (orgimage.shape[0], orgimage.shape[1]))), 3
    )

    return resPSNR, resNC, dec_time, ex_time
