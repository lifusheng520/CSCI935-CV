"""Foreground (leaf) segmentation for white-background rice leaf images.

Pipeline: Lab distance from estimated background colour -> Otsu ->
morphology -> largest component + hole fill -> optional GrabCut refinement.

Usage:
    python seg_leaf.py image.jpg            # writes image_mask.png, image_cutout.png
"""
import sys
import cv2
import numpy as np


def estimate_background(lab):
    """Median Lab colour of the image border (robust to the leaf touching it)."""
    h, w = lab.shape[:2]
    t = max(4, int(0.03 * min(h, w)))
    border = np.concatenate([
        lab[:t].reshape(-1, 3), lab[-t:].reshape(-1, 3),
        lab[:, :t].reshape(-1, 3), lab[:, -t:].reshape(-1, 3),
    ])
    med = np.median(border, axis=0)
    # one refinement pass: drop border pixels far from the median (leaf/edge), re-estimate
    d = np.linalg.norm(border - med, axis=1)
    keep = border[d < np.percentile(d, 80)]
    return np.median(keep, axis=0)


def fit_background_surface(lab, iters=3, deg=2, step=8):
    """Fit a smooth (quadratic) surface per Lab channel to background pixels.

    Handles vignetting / lighting gradients that a single background colour cannot.
    Starts from the image border, then iteratively re-fits on pixels that agree with the fit.
    """
    h, w = lab.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xn, yn = xx / w - 0.5, yy / h - 0.5
    terms = [np.ones_like(xn)]
    for d in range(1, deg + 1):
        for i in range(d + 1):
            terms.append(xn ** (d - i) * yn ** i)
    A = np.stack(terms, axis=-1)  # h,w,T

    t = max(4, int(0.03 * min(h, w)))
    sel = np.zeros((h, w), bool)
    sel[:t] = sel[-t:] = True
    sel[:, :t] = sel[:, -t:] = True

    surf = np.zeros_like(lab)
    for it in range(iters + 1):
        idx = np.argwhere(sel)[::step]
        Ai = A[idx[:, 0], idx[:, 1]]
        for c in range(3):
            coef, *_ = np.linalg.lstsq(Ai, lab[idx[:, 0], idx[:, 1], c], rcond=None)
            surf[..., c] = A @ coef
        res = np.linalg.norm(lab - surf, axis=2)
        sel = res < max(np.percentile(res, 60), 4.0)  # confident-background pixels
    return surf


def largest_component_filled(mask):
    """Keep the largest blob and fill any interior holes (pale lesions)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return mask
    biggest = max(cnts, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [biggest], -1, 255, thickness=cv2.FILLED)
    return out


def segment_leaf(bgr, work_size=800, use_grabcut=True, l_weight=0.0, noise_floor=8):
    h, w = bgr.shape[:2]
    s = min(1.0, work_size / max(h, w))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()

    blur = cv2.GaussianBlur(small, (5, 5), 0)
    lab = cv2.cvtColor(blur, cv2.COLOR_BGR2LAB).astype(np.float32)

    bg = fit_background_surface(lab)
    # Chroma-only distance by default (l_weight=0): cast shadows change mostly lightness L,
    # while leaf tissue (green/tan/cream) differs from the grey wall in a/b. Even l_weight=0.1
    # kept wide shadows in tests. Raise l_weight only for very pale leaves close to the wall colour.
    diff = lab - bg
    dist = np.sqrt((l_weight * diff[..., 0]) ** 2 + diff[..., 1] ** 2 + diff[..., 2] ** 2)
    dist8 = np.clip(dist * (255.0 / max(dist.max(), 1e-6)), 0, 255).astype(np.uint8)

    # Multi-level Otsu: strong green tissue would otherwise dominate a 2-class Otsu
    # and drop pale/dried tan tissue. Use the LOWER of the two thresholds.
    t_otsu, _ = cv2.threshold(dist8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    low = dist8[dist8 < t_otsu]
    t_low = cv2.threshold(low, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0] if low.size else t_otsu
    thr = max(t_low, noise_floor)
    mask = (dist8 > thr).astype(np.uint8) * 255

    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = largest_component_filled(mask)

    if use_grabcut and mask.any():
        gc = np.full(mask.shape, cv2.GC_BGD, np.uint8)
        gc[cv2.dilate(mask, k, iterations=3) > 0] = cv2.GC_PR_BGD
        gc[mask > 0] = cv2.GC_PR_FGD
        gc[cv2.erode(mask, k, iterations=2) > 0] = cv2.GC_FGD
        bgm = np.zeros((1, 65), np.float64)
        fgm = np.zeros((1, 65), np.float64)
        cv2.grabCut(small, gc, None, bgm, fgm, 3, cv2.GC_INIT_WITH_MASK)
        mask = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        mask = largest_component_filled(mask)

    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def cutout_rgba(bgr, mask):
    return np.dstack([bgr, mask])


def composite(bgr, mask, field_bgr):
    """Paste the segmented leaf onto a field background (background-swap augmentation)."""
    h, w = bgr.shape[:2]
    bg = cv2.resize(field_bgr, (w, h))
    a = cv2.GaussianBlur(mask, (7, 7), 0).astype(np.float32)[..., None] / 255.0  # soft edge
    return (a * bgr + (1 - a) * bg).astype(np.uint8)


if __name__ == "__main__":
    path = sys.argv[1]
    img = cv2.imread(path)
    m = segment_leaf(img)
    stem = path.rsplit(".", 1)[0]
    cv2.imwrite(stem + "_mask.png", m)
    cv2.imwrite(stem + "_cutout.png", cutout_rgba(img, m))
    print(f"foreground fraction: {(m > 0).mean():.3f}")