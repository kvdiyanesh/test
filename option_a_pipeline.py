#!/usr/bin/env python3
"""Batch photo-to-flat-illustration converter (Option A pipeline).

Pipeline:
1) Foreground extraction with GrabCut.
2) Background replacement.
3) Palette reduction via k-means.
4) Region smoothing and optional skin/hair cleanup.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


def extract_foreground_bgr(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mask = np.zeros((h, w), np.uint8)

    # Start with a centered rectangle; works well for portraits.
    margin_x = max(8, int(w * 0.08))
    margin_y = max(8, int(h * 0.08))
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(img_bgr, mask, rect, bgd_model, fgd_model, 6, cv2.GC_INIT_WITH_RECT)
    fg_mask = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)

    k = np.ones((5, 5), np.uint8)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, k, iterations=1)
    fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, k, iterations=2)
    fg_mask = cv2.GaussianBlur(fg_mask, (0, 0), 1.2)
    return fg_mask


def quantize_kmeans(img_bgr: np.ndarray, n_colors: int = 6) -> np.ndarray:
    data = img_bgr.reshape((-1, 3)).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.8)
    _compact, labels, centers = cv2.kmeans(
        data, n_colors, None, criteria, 3, cv2.KMEANS_PP_CENTERS
    )
    centers = np.uint8(centers)
    return centers[labels.flatten()].reshape(img_bgr.shape)


def normalize_skin_and_hair(img_bgr: np.ndarray) -> np.ndarray:
    out = img_bgr.copy()
    hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV)

    # Broad skin mask (works for many tones; adjust as needed).
    skin = cv2.inRange(hsv, np.array([0, 20, 55], dtype=np.uint8), np.array([30, 255, 255], dtype=np.uint8))
    if np.count_nonzero(skin) > 50:
        skin_pixels = out[skin > 0]
        median_skin = np.median(skin_pixels, axis=0).astype(np.uint8)
        out[skin > 0] = median_skin

    # Very dark pixels become a single hair tone.
    hair = cv2.inRange(hsv, np.array([0, 0, 0], dtype=np.uint8), np.array([180, 255, 55], dtype=np.uint8))
    out[hair > 0] = np.array([10, 10, 10], dtype=np.uint8)
    return out


def stylize(img_bgr: np.ndarray, n_colors: int, bg_color: Tuple[int, int, int]) -> np.ndarray:
    fg_mask = extract_foreground_bgr(img_bgr)

    smoothed = cv2.bilateralFilter(img_bgr, d=9, sigmaColor=40, sigmaSpace=40)
    flat = quantize_kmeans(smoothed, n_colors=n_colors)
    flat = normalize_skin_and_hair(flat)

    result = np.full_like(flat, bg_color, dtype=np.uint8)
    alpha = (fg_mask.astype(np.float32) / 255.0)[:, :, None]
    blended = (flat.astype(np.float32) * alpha + result.astype(np.float32) * (1.0 - alpha)).astype(np.uint8)
    return blended


def process_file(src: Path, dst: Path, n_colors: int, bg_color: Tuple[int, int, int]) -> None:
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Could not read image: {src}")
    out = stylize(img, n_colors=n_colors, bg_color=bg_color)
    dst.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(dst), out)
    if not ok:
        raise RuntimeError(f"Failed writing output: {dst}")


def parse_bg(hex_or_csv: str) -> Tuple[int, int, int]:
    s = hex_or_csv.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 6 and all(c in "0123456789abcdefABCDEF" for c in s):
        r = int(s[0:2], 16)
        g = int(s[2:4], 16)
        b = int(s[4:6], 16)
        return (b, g, r)

    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 3:
        r, g, b = (int(parts[0]), int(parts[1]), int(parts[2]))
        return (b, g, r)
    raise argparse.ArgumentTypeError("Use #RRGGBB or R,G,B for --background")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert photos to flat illustration style in batch.")
    parser.add_argument("input", type=Path, help="Input file or folder")
    parser.add_argument("output", type=Path, help="Output file or folder")
    parser.add_argument("--colors", type=int, default=6, help="Number of color clusters")
    parser.add_argument("--background", type=parse_bg, default=parse_bg("#e6e6e6"), help="Background color")
    args = parser.parse_args()

    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

    if args.input.is_file():
        process_file(args.input, args.output, args.colors, args.background)
        print(f"Wrote {args.output}")
        return

    if not args.input.is_dir():
        raise SystemExit(f"Input path not found: {args.input}")

    files = sorted([p for p in args.input.iterdir() if p.suffix.lower() in exts and p.is_file()])
    if not files:
        raise SystemExit(f"No images found in {args.input}")

    args.output.mkdir(parents=True, exist_ok=True)
    for src in files:
        dst = args.output / f"{src.stem}_flat{src.suffix.lower()}"
        process_file(src, dst, args.colors, args.background)
        print(f"Wrote {dst}")


if __name__ == "__main__":
    main()
