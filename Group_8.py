#!/usr/bin/env python3
"""CSCI935 Group 8 -- MDB-Net: Mask-Guided Domain-Bridging Multi-Task Network for
five-class rice leaf disease classification on the Dhan-Shomadhan dataset.

    image --> ResNet-18 encoder --+--> U-Net decoder --> leaf mask (segmentation head)
                                  |
                                  +--> features pooled over the leaf --> disease class (classification head)

Steps (run from the folder that contains Dhan-Shomadhan/):
    python Group_8.py masks       # 1. resize cache + classical leaf masks for white images + QC
    python Group_8.py train       # 2. five seeds, White / Field / Mixed test scenarios
    python Group_8.py run         # 1 + 2 in one command (reproduces the reported results)

All generated data is written under Dhan-Shomadhan/_interim/:
    cache_448/   resized images            masks_448/   white-background leaf masks
    qc/          mask QC report + sheets   runs/<tag>/  results, predictions, confusion matrices

Required packages: torch, torchvision, numpy, opencv-python-headless, scikit-learn, matplotlib.
Pretrained weights: torchvision ResNet-18 IMAGENET1K_V1, downloaded automatically on first use;
offline: download https://download.pytorch.org/models/resnet18-f37072fd.pth and pass --weights_path.
"""
from __future__ import annotations

import argparse
import copy
import csv
import difflib
import json
import math
import random
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor  # threads, not processes: forking after importing torch can deadlock
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, tv_tensors
from torchvision.transforms import v2


# ----------------------------------------------------------------------------------
# Data discovery
# ----------------------------------------------------------------------------------
# Folder names in the public dataset contain typos ("Leaf Scaled", "Rice Turngo",
# "Steath Blight"), so classes are matched by keyword instead of exact name.
CLASS_KEYS = [
    ("brown_spot", ("brown", "brow", "spot")),
    ("leaf_scald", ("scald", "scal")),
    ("rice_blast", ("blast",)),
    ("rice_tungro", ("tungro", "turngo", "tunro", "tung", "ngro")),
    ("sheath_blight", ("sheath", "steath", "shath")),
]
# Canonical names used for fuzzy matching when no keyword hits (catches new typos like "Browon Spot").
CANON = ["brownspot", "leafscald", "riceblast", "ricetungro", "sheathblight"]


def match_class(part: str):
    """Class index for a folder name: keyword match first, then fuzzy match, else None."""
    s = re.sub(r"[^a-z]", "", part.lower())
    if not s:
        return None
    for i, (_, keys) in enumerate(CLASS_KEYS):
        if any(k in s for k in keys):
            return i
    scores = [difflib.SequenceMatcher(None, s, c).ratio() for c in CANON]
    order = np.argsort(scores)[::-1]
    best, second = int(order[0]), int(order[1])
    return best if scores[best] >= 0.7 and scores[best] - scores[second] > 0.05 else None
CLASS_NAMES = [n for n, _ in CLASS_KEYS]
DOMAINS = ["white", "field"]
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def discover_samples(root: Path, allow_missing: bool = False):
    """Return list of (relative_path, class_index, domain_index). Skips folders starting with '_'."""
    samples, unknown = [], []
    for p in sorted(root.rglob("*")):
        if not p.is_file() or p.suffix.lower() not in IMG_EXT:
            continue
        rel = p.relative_to(root)
        if any(part.startswith("_") for part in rel.parts):  # our own cache/output folders
            continue
        parts = [s.lower() for s in rel.parts[:-1]]
        dom = next((("white" if "white" in s else "field") for s in parts if "white" in s or "field" in s), None)
        cls = next((c for c in (match_class(s) for s in parts) if c is not None), None)
        if dom is None or cls is None:
            unknown.append(str(rel.parent))
            continue
        samples.append((rel, cls, DOMAINS.index(dom)))
    if unknown:
        print(f"[warn] {len(unknown)} images skipped (could not read class/background from folder names). Folders:")
        for folder, n in Counter(unknown).most_common(10):
            print(f"         {n:>4} x {folder}")
    if not samples:
        raise SystemExit(f"No images found under {root}. Expected folders like '<White|Field> Background/<class>/*.jpg'.")
    labels = np.array([s[1] for s in samples])
    domains = np.array([s[2] for s in samples])
    print("Images per class x background:")
    print(f"  {'class':<15}{'white':>7}{'field':>7}")
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name:<15}{int(((labels == c) & (domains == 0)).sum()):>7}{int(((labels == c) & (domains == 1)).sum()):>7}")
    print(f"  total: {len(samples)}")
    empty = [(CLASS_NAMES[c], DOMAINS[d]) for c in range(len(CLASS_NAMES)) for d in range(2)
             if not ((labels == c) & (domains == d)).any()]
    if empty:
        msg = f"No images for {empty}. Some folders were probably skipped (see the warning above)."
        if not allow_missing:
            raise SystemExit(f"[error] {msg}\nFix the folder-name matching, or pass --allow_missing to continue anyway.")
        print(f"[warn] {msg}")
    return samples


def build_cache(root: Path, samples, max_side: int) -> Path:
    """Write resized copies (longest side = max_side) under <root>/_interim/. Big phone photos
    decode slowly, so this makes every later epoch and every seed much faster."""
    cache = root / "_interim" / f"cache_{max_side}"
    todo = [s for s in samples if not (cache / s[0].with_suffix(".jpg")).exists()]
    if todo:
        print(f"Building cache ({len(todo)} images) in {cache} ...")
    for rel, _, _ in todo:
        img = cv2.imread(str(root / rel), cv2.IMREAD_COLOR)  # honours EXIF orientation
        if img is None:
            raise SystemExit(f"Could not read image: {root / rel}")
        h, w = img.shape[:2]
        s = max_side / max(h, w)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        out = cache / rel.with_suffix(".jpg")
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    return cache

def make_split(labels, domains, seed, val_frac=0.15, test_frac=0.15):
    """Stratified by (class, background) so every split has the same mix as the full data."""
    strata = np.array([f"{l}_{d}" for l, d in zip(labels, domains)])
    idx = np.arange(len(labels))
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  stratify=strata[trainval], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


def build_model(num_classes, weights_path=None, dropout=0.2, pretrained=True):
    if not pretrained:  # random (Kaiming) initialisation, no download needed
        model = models.resnet18(weights=None)
    elif weights_path:
        model = models.resnet18(weights=None)
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    else:
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except Exception as e:  # no network etc.
            raise SystemExit(
                "Could not download ImageNet ResNet-18 weights.\n"
                "Download https://download.pytorch.org/models/resnet18-f37072fd.pth manually and run with\n"
                f"  --weights_path <file>\n(original error: {e})")
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def score(y_true, y_pred, n_classes):
    labs = list(range(n_classes))
    present = [c for c in labs if (y_true == c).any()]  # macro-F1 only over classes that occur
    return {
        "n": int(len(y_true)),
        "acc": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)),
        "per_class_f1": f1_score(y_true, y_pred, labels=labs, average=None, zero_division=0).tolist(),
        "cm": confusion_matrix(y_true, y_pred, labels=labs).tolist(),
    }


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_with_warmup(optimizer, warmup_steps, total_steps):
    def f(step):
        if step < warmup_steps:
            return (step + 1) / max(1, warmup_steps)
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, p)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def plot_cm(cm, path, title):
    cmn = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=40, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cmn[i, j]:.2f}", ha="center", va="center",
                    color="white" if cmn[i, j] > 0.5 else "black", fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def aggregate(results, out_dir, tag):
    summary = {}
    sd = lambda a: float(np.std(a, ddof=1)) if len(a) > 1 else 0.0
    for sc in ["a_white", "b_field", "c_all"]:
        rs = [r["scenarios"][sc] for r in results if sc in r["scenarios"]]
        if not rs:
            continue
        acc, f1 = [r["acc"] for r in rs], [r["macro_f1"] for r in rs]
        pcf = np.array([r["per_class_f1"] for r in rs])
        cm = np.sum([np.array(r["cm"]) for r in rs], axis=0)
        summary[sc] = {"runs": len(rs), "acc_mean": float(np.mean(acc)), "acc_std": sd(acc),
                       "macro_f1_mean": float(np.mean(f1)), "macro_f1_std": sd(f1),
                       "per_class_f1_mean": dict(zip(CLASS_NAMES, pcf.mean(0).round(4).tolist())),
                       "per_class_f1_std": dict(zip(CLASS_NAMES, [sd(pcf[:, k]) for k in range(pcf.shape[1])])),
                       "cm_sum": cm.tolist()}
        plot_cm(cm, out_dir / f"confusion_{sc}.png", f"{tag}: {sc} (sum over {len(rs)} runs)")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\n=== {tag}: mean +/- std over {len(results)} runs ===")
    print(f"{'scenario':<10}{'accuracy':>20}{'macro-F1':>20}")
    for sc, s in summary.items():
        print(f"{sc:<10}{s['acc_mean']:>12.4f} +/- {s['acc_std']:.4f}{s['macro_f1_mean']:>12.4f} +/- {s['macro_f1_std']:.4f}")
    return summary


# ==================================================================================
# Classical leaf segmentation for white-background images
# Lab distance from a fitted background surface -> Otsu -> morphology -> largest
# component + hole fill -> GrabCut refinement.
# ==================================================================================
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


# ==================================================================================
# Step 1 (masks): batch mask generation for the white-background images, with QC
# ==================================================================================
FONT = cv2.FONT_HERSHEY_SIMPLEX
WORK = 800  # the segmenter works at this longest side


def mask_metrics(mask: np.ndarray, bgr: np.ndarray) -> dict:
    """Numbers that describe how leaf-like a mask is. All computed at cache resolution."""
    fg = mask > 0
    out = dict(fg_frac=float(fg.mean()), elongation=0.0, solidity=0.0, width_ratio=0.0, edge_contrast=0.0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts or fg.sum() < 20:
        return out
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    out["elongation"] = float(max(rw, rh) / max(min(rw, rh), 1.0))           # long side / short side
    out["solidity"] = float(area / max(cv2.contourArea(cv2.convexHull(c)), 1.0))  # 1 = convex, low = ragged
    # width along the long axis: a shadow block shows up as a bulge (p95 >> median)
    ys, xs = np.where(fg)
    prof = fg.sum(1) if (ys.max() - ys.min()) >= (xs.max() - xs.min()) else fg.sum(0)
    prof = prof[prof > 0]
    out["width_ratio"] = float(np.percentile(prof, 95) / max(np.median(prof), 1.0))
    # colour contrast across the mask boundary (a/b channels only, so soft shadows score low)
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    inner = fg & ~(cv2.erode(mask, k) > 0)
    outer = (cv2.dilate(mask, k) > 0) & ~fg
    if inner.sum() >= 10 and outer.sum() >= 10:
        out["edge_contrast"] = float(np.linalg.norm(lab[inner][:, 1:].mean(0) - lab[outer][:, 1:].mean(0)))
    return out


def _process(job):
    """Worker: segment one image (or reuse an existing mask) and return its metrics."""
    rel, src, cache_file, out_png, l_weight, overwrite = job
    cv2.setNumThreads(1)
    cache = cv2.imread(str(cache_file), cv2.IMREAD_COLOR)
    ch, cw = cache.shape[:2]
    if out_png.exists() and not overwrite:
        mask = cv2.imread(str(out_png), cv2.IMREAD_GRAYSCALE)
        if mask is not None and mask.shape == (ch, cw):
            return rel, mask_metrics(mask, cache)
    img = cv2.imread(str(src), cv2.IMREAD_COLOR)
    s = WORK / max(img.shape[:2])
    small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else img
    m = segment_leaf(small, work_size=WORK, l_weight=l_weight)
    m = cv2.resize(m, (cw, ch), interpolation=cv2.INTER_AREA)
    m = ((m > 127) * 255).astype(np.uint8)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_png), m)
    return rel, mask_metrics(m, cache)


def robust_z(values: np.ndarray) -> np.ndarray:
    med = np.median(values)
    mad = max(1.4826 * np.median(np.abs(values - med)), 0.05 * abs(med), 1e-6)  # floored: avoids tiny-MAD over-flagging
    return (values - med) / mad


def flag_rows(rows: list[dict], z_thr: float) -> None:
    """Adds 'flags' (list of reasons) and 'status' (ok | review | fail) to each row, in place."""
    for r in rows:
        r["flags"] = []
    # hard failures: physically implausible size
    for r in rows:
        if r["fg_frac"] < 0.01:
            r["flags"].append("FAIL:tiny_mask")
        if r["fg_frac"] > 0.40:
            r["flags"].append("FAIL:huge_mask")
        if r["edge_contrast"] < 3.0:   # mask boundary does not follow any colour edge (real leaves: >= 17 in tests)
            r["flags"].append("FAIL:no_edge")
        if r["elongation"] < 3.0:      # rice leaves are long and thin
            r["flags"].append("FAIL:not_leaf_shaped")
    # relative outliers, judged within each disease class
    checks = [("fg_frac", 0, "size"), ("elongation", -1, "not_elongated"), ("solidity", -1, "ragged"),
              ("width_ratio", +1, "bulge"), ("edge_contrast", -1, "weak_edge")]
    for cls in sorted({r["cls"] for r in rows}):
        grp = [r for r in rows if r["cls"] == cls]
        if len(grp) < 8:  # too few to estimate a spread
            continue
        for key, sign, name in checks:
            z = robust_z(np.array([r[key] for r in grp]))
            for r, zi in zip(grp, z):
                if (sign == 0 and abs(zi) > z_thr) or (sign != 0 and sign * zi > z_thr):
                    r["flags"].append(f"{name}(z={zi:+.1f})")
    for r in rows:
        r["status"] = "fail" if any(f.startswith("FAIL") for f in r["flags"]) else ("review" if r["flags"] else "ok")

def _tile(cache_file, mask_file, lines, box=(170, 360)):
    img = cv2.imread(str(cache_file), cv2.IMREAD_COLOR)
    m = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cnts, -1, (0, 0, 255), 3)
    bw, bh = box
    s = min(bw / img.shape[1], bh / img.shape[0])
    im = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    canvas = np.full((bh + 46, bw, 3), 255, np.uint8)
    canvas[: im.shape[0], : im.shape[1]] = im
    for k, line in enumerate(lines[:3]):
        cv2.putText(canvas, line[:34], (2, bh + 12 + 13 * k), FONT, 0.34, (0, 0, 0), 1, cv2.LINE_AA)
    return canvas


def write_sheets(items, prefix, qc_dir, per_row=8, n_rows=2):
    """items: list of (cache_file, mask_file, label_lines). Returns list of written files."""
    per_sheet, written = per_row * n_rows, []
    for si in range(0, len(items), per_sheet):
        tiles = [_tile(*it) for it in items[si: si + per_sheet]]
        blank = np.full_like(tiles[0], 255)
        tiles += [blank] * (per_sheet - len(tiles))
        rows = [np.hstack(tiles[r * per_row: (r + 1) * per_row]) for r in range(n_rows)]
        f = qc_dir / f"{prefix}_{si // per_sheet + 1:02d}.jpg"
        cv2.imwrite(str(f), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
        written.append(f)
    return written

def cmd_masks(a):
    root = a.data_root
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root.resolve()}")
    samples = discover_samples(root, a.allow_missing)
    cache_dir = build_cache(root, samples, a.cache_size)
    white = [s for s in samples if s[2] == 0]  # domain index 0 = white background
    if a.only:
        white = [s for s in white if a.only.lower() in str(s[0]).lower()]
    mask_dir = root / "_interim" / f"masks_{a.cache_size}"
    qc_dir = root / "_interim" / "qc"
    qc_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSegmenting {len(white)} white-background images (workers={a.workers}) ...")

    jobs = [(rel, root / rel, cache_dir / rel.with_suffix(".jpg"), mask_dir / rel.with_suffix(".png"),
             a.l_weight, a.overwrite) for rel, _, _ in white]
    cls_of = {str(rel): CLASS_NAMES[c] for rel, c, _ in white}
    if a.workers > 1:
        with ThreadPoolExecutor(a.workers) as ex:
            results = list(ex.map(_process, jobs))
    else:
        results = [_process(j) for j in jobs]

    rows = []
    for rel, met in results:
        rows.append(dict(image=str(rel), cls=cls_of[str(rel)], **met))
    flag_rows(rows, a.z)
    excluded = set()
    if a.exclude_file and a.exclude_file.exists():
        excluded = {ln.strip() for ln in a.exclude_file.read_text().splitlines() if ln.strip()}
        for r in rows:
            if r["image"] in excluded:
                r["status"] = "excluded"

    # ---- reports -------------------------------------------------------------------------
    cols = ["image", "cls", "status", "fg_frac", "elongation", "solidity", "width_ratio", "edge_contrast", "flags"]
    with open(qc_dir / "qc_report.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in rows:
            w.writerow([r["image"], r["cls"], r["status"]] + [f"{r[k]:.4f}" for k in cols[3:8]] + ["; ".join(r["flags"])])
    with open(qc_dir / "mask_index.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "cache_image", "mask", "class", "status"])
        for r in rows:
            rel = Path(r["image"])
            w.writerow([r["image"], str(cache_dir.relative_to(root) / rel.with_suffix(".jpg")),
                        str(mask_dir.relative_to(root) / rel.with_suffix(".png")), r["cls"], r["status"]])

    # ---- contact sheets: every flagged mask + a random sample of the passing ones ---------
    for old in qc_dir.glob("*.jpg"):
        old.unlink()
    item = lambda r: (cache_dir / Path(r["image"]).with_suffix(".jpg"), mask_dir / Path(r["image"]).with_suffix(".png"),
                      [Path(r["image"]).parent.name[:12] + "/" + Path(r["image"]).name, f"{r['status']} fg={r['fg_frac']:.3f}"]
                      + ([r["flags"][0]] if r["flags"] else []))
    flagged = sorted([r for r in rows if r["status"] in ("fail", "review")], key=lambda r: (r["status"] != "fail", r["image"]))
    passed = [r for r in rows if r["status"] == "ok"]
    random.Random(a.seed).shuffle(passed)
    sheets = write_sheets([item(r) for r in flagged], "flagged", qc_dir) if flagged else []
    sheets += write_sheets([item(r) for r in passed[: a.n_pass_sample]], "sample_pass", qc_dir) if passed else []

    # ---- summary -------------------------------------------------------------------------
    n = len(rows)
    cnt = lambda s: sum(r["status"] == s for r in rows)
    print(f"\n=== QC summary ({n} masks) ===")
    print(f"  ok {cnt('ok')}   review {cnt('review')}   fail {cnt('fail')}   excluded {cnt('excluded')}")
    reasons: dict[str, int] = {}
    for r in rows:
        for fl in r["flags"]:
            reasons[fl.split("(")[0]] = reasons.get(fl.split("(")[0], 0) + 1
    for k, v in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    {k:<22}{v}")
    fg = np.array([r["fg_frac"] for r in rows]) if rows else np.array([0.0])
    print(f"  foreground fraction: median {np.median(fg):.3f}, min {fg.min():.3f}, max {fg.max():.3f}")
    print(f"\nMasks:   {mask_dir}\nReport:  {qc_dir / 'qc_report.csv'}\nSheets:  {qc_dir}  ({len(sheets)} files)")


# ==================================================================================
# Step 2 (train): MDB-Net data pipeline, model, training and evaluation
# ==================================================================================
def load_records(root: Path, samples, mask_status: str, exclude_file: Path):
    """One record per image: relative path, class, background, and a mask path if a trusted mask exists."""
    idx_csv = root / "_interim" / "qc" / "mask_index.csv"
    if not idx_csv.exists():
        raise SystemExit(f"{idx_csv} not found. Run 'python Group_8.py masks' first.")
    info = {r["image"]: r for r in csv.DictReader(open(idx_csv))}
    excl = {ln.strip() for ln in exclude_file.read_text().splitlines() if ln.strip()} if exclude_file.exists() else set()
    allowed = {s.strip() for s in mask_status.split(",")}
    recs = []
    for rel, c, d in samples:
        r = info.get(str(rel))
        ok = r is not None and r["status"] in allowed and str(rel) not in excl
        recs.append(dict(rel=rel, label=c, domain=d, mask=(root / r["mask"]) if ok else None))
    n_m = sum(r["mask"] is not None for r in recs)
    print(f"masks: {n_m} images have a trusted mask (status in {sorted(allowed)}, minus {len(excl)} excluded)")
    return recs


def paste_on_field(img, m, bg_path):
    """Composite the masked leaf onto a random crop of a field image (soft edge).

    A plain cut-and-paste keeps the leaf's studio lighting and sharpness, so the composite
    still looks obviously pasted and the encoder can tell "swapped" from "real field photo"
    from cues that have nothing to do with the leaf itself. Two cheap fixes narrow that gap:
    nudge the leaf's brightness toward the field crop's ambient light, and add camera-like
    blur/noise so the composite's sharpness matches a real field shot.
    """
    H, W = m.shape
    bg = cv2.cvtColor(cv2.imread(str(bg_path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    bh, bw = bg.shape[:2]
    ch = int(bh * random.uniform(0.4, 1.0))
    cw = int(ch * W / H)
    if cw > bw:
        cw, ch = bw, int(bw * H / W)
    y0, x0 = random.randint(0, max(bh - ch, 0)), random.randint(0, max(bw - cw, 0))
    bg = cv2.resize(bg[y0:y0 + ch, x0:x0 + cw], (W, H), interpolation=cv2.INTER_AREA)

    fg = img
    leaf = m > 0
    if leaf.sum() > 50:  # light-match: shift the leaf's L channel toward the field crop's brightness
        lab_fg = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        l_bg = cv2.cvtColor(bg, cv2.COLOR_RGB2LAB).astype(np.float32)[..., 0]
        dl = (l_bg.mean() - lab_fg[..., 0][leaf].mean()) * random.uniform(0.3, 0.7)
        lab_fg[..., 0] = np.clip(lab_fg[..., 0] + dl, 0, 255)
        fg = cv2.cvtColor(lab_fg.astype(np.uint8), cv2.COLOR_LAB2RGB)

    a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 1.0)[..., None]
    out = (a * fg + (1 - a) * bg).astype(np.uint8)

    if random.random() < 0.5:  # field photos are typically softer / noisier than the studio shots
        out = cv2.GaussianBlur(out, (0, 0), random.uniform(0.5, 1.2))
    if random.random() < 0.3:
        noise = np.random.normal(0, random.uniform(3, 8), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return out


def make_tf(size: int, train: bool):
    norm = v2.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:  # geometric ops act on image and mask together, colour ops on the image only
        return v2.Compose([v2.ToDtype(torch.float32, scale=True),
                           v2.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.6, 1.6), antialias=True),
                           v2.RandomHorizontalFlip(), v2.RandomVerticalFlip(), v2.RandomRotation(20),
                           v2.ColorJitter(0.2, 0.2, 0.1, 0.02), norm])
    return v2.Compose([v2.ToDtype(torch.float32, scale=True), v2.Resize((size, size), antialias=True), norm])


class MTDataset(Dataset):
    def __init__(self, cache: Path, recs, idx, size, train, bg_swap=0.0, bg_files=()):
        self.cache, self.recs, self.idx = cache, recs, np.asarray(idx)
        self.train, self.bg_swap, self.bg_files = train, bg_swap, list(bg_files)
        self.tf = make_tf(size, train)
        self.labels = np.array([recs[i]["label"] for i in self.idx])
        self.domains = np.array([recs[i]["domain"] for i in self.idx])

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, k):
        r = self.recs[self.idx[k]]
        img = cv2.cvtColor(cv2.imread(str(self.cache / r["rel"].with_suffix(".jpg")), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        has = r["mask"] is not None
        if has:
            m = cv2.imread(str(r["mask"]), cv2.IMREAD_GRAYSCALE)
            if m.shape != img.shape[:2]:
                m = cv2.resize(m, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            m = (m > 127).astype(np.uint8)
            if self.train and self.bg_files and random.random() < self.bg_swap:
                img = paste_on_field(img, m, random.choice(self.bg_files))
        else:
            m = np.zeros(img.shape[:2], np.uint8)
        x, y = self.tf(tv_tensors.Image(torch.from_numpy(img).permute(2, 0, 1)),
                       tv_tensors.Mask(torch.from_numpy(m)[None]))
        return (x.as_subclass(torch.Tensor), y.as_subclass(torch.Tensor).float(),
                int(r["label"]), float(has), int(r["domain"]))


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class MultiTaskNet(nn.Module):
    """Shared ResNet-18 encoder, U-Net segmentation decoder and mask-guided classification head.

    The reported configuration uses pool="mask". The "global", "mil" and "proto" pooling layers are
    still constructed so that parameter initialisation (and hence the random-number sequence used
    during training) is identical to the code that produced the reported results.
    """
    def __init__(self, weights_path=None, dropout=0.2, pool="mask", pretrained=True, mil_topk_frac=0.2):
        super().__init__()
        r = build_model(NC, weights_path, dropout, pretrained)  # same weight loading / error message as the baseline
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)          # 64  @ H/2
        self.pool0 = r.maxpool
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4  # 64@H/4 128@H/8 256@H/16 512@H/32
        self.d4, self.d3 = conv_block(512 + 256, 256), conv_block(256 + 128, 128)
        self.d2, self.d1 = conv_block(128 + 64, 64), conv_block(64 + 64, 32)
        self.seg_out = nn.Conv2d(32, 1, 1)
        self.cls = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, NC))
        self.cls_map = nn.Conv2d(512, NC, 1)  # per-location disease evidence, used only when pool == "mil"
        # attention-MIL gate (Ilse et al. 2018) + learnable per-class prototypes, used only when pool == "proto"
        self.attn_v, self.attn_u, self.attn_w = nn.Linear(512, 128), nn.Linear(512, 128), nn.Linear(128, 1)
        self.proto = nn.Parameter(torch.randn(NC, 512) * 0.01)
        self.proto_scale = nn.Parameter(torch.tensor(10.0))
        self.pool = pool
        self.mil_topk_frac = mil_topk_frac

    def _mil_pool(self, logit_map, leaf_w):
        """Multiple-instance-learning pooling: one instance per feature-map location inside the leaf.

        A lesion is usually local, so averaging over the whole leaf (the "mask" pool above) dilutes a
        small disease patch with a lot of healthy-looking tissue. Here the image-level logit for each
        class is instead the average of the STRONGEST `mil_topk_frac` leaf locations for that class --
        "even one clear clue is enough" -- which falls back to averaging every location when the
        predicted leaf mask is empty (e.g. early in training, or a bad field prediction).
        """
        B, C, Hf, Wf = logit_map.shape
        flat = logit_map.view(B, C, -1)
        leaf = leaf_w.view(B, -1) > 0.5
        out = flat.new_zeros(B, C)
        for b in range(B):
            idx = leaf[b].nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                out[b] = flat[b].mean(-1)
            else:
                k = max(1, int(round(self.mil_topk_frac * idx.numel())))
                out[b] = flat[b, :, idx].topk(k, dim=-1).values.mean(-1)
        return out

    def _proto_pool(self, f, leaf_w):
        """Attention-pooled feature classified by nearest learnable class prototype, not a linear head.

        The idea: we already know what each disease's leaf tissue looks like from the white-background
        masks, so instead of learning an absolute decision boundary in raw feature space (which shifts
        between white and field lighting/background statistics), classify by which disease's PROTOTYPE
        the pooled feature is most similar to (cosine distance) -- a relative, scale-invariant comparison
        that generalizes across domains better than a linear head does. Prototypes are seeded from white
        images by init_prototypes() and then refined end-to-end. Locations are combined with a learned
        attention gate (a location "votes" more if the gate finds it informative), restricted to the leaf.
        """
        B, C, Hf, Wf = f.shape
        h = f.view(B, C, -1).permute(0, 2, 1)                       # B,N,C
        leaf = leaf_w.view(B, -1) > 0.5                             # B,N
        a = self.attn_w(torch.tanh(self.attn_v(h)) * torch.sigmoid(self.attn_u(h))).squeeze(-1)  # B,N
        empty = ~leaf.any(-1)
        a = a.masked_fill(~leaf & ~empty[:, None], float("-inf"))   # keep all locations for an empty leaf mask
        w = torch.softmax(a, dim=-1)
        pooled = (w.unsqueeze(-1) * h).sum(1)                       # B,C
        sim = F.normalize(pooled, dim=-1) @ F.normalize(self.proto, dim=-1).t()  # B,NC cosine similarity
        return self.proto_scale * sim

    def encoder_modules(self):
        return [self.stem, self.l1, self.l2, self.l3, self.l4]

    @staticmethod
    def _up(x, skip):
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x, gt_mask=None, use_gt=None, return_feat=False):
        H, W = x.shape[-2:]
        x0 = self.stem(x)
        x1 = self.l1(self.pool0(x0))
        x2 = self.l2(x1)
        x3 = self.l3(x2)
        x4 = self.l4(x3)
        d = self.d4(torch.cat([self._up(x4, x3), x3], 1))
        d = self.d3(torch.cat([self._up(d, x2), x2], 1))
        d = self.d2(torch.cat([self._up(d, x1), x1], 1))
        d = self.d1(torch.cat([self._up(d, x0), x0], 1))
        seg = self.seg_out(F.interpolate(d, size=(H, W), mode="bilinear", align_corners=False))

        f = x4.float()
        glob = f.mean((2, 3))
        m = F.adaptive_avg_pool2d(torch.sigmoid(seg.detach().float()), f.shape[-2:])
        if gt_mask is not None and use_gt is not None:  # teacher forcing: sometimes use the true mask while training
            g = F.adaptive_avg_pool2d(gt_mask.float(), f.shape[-2:])
            m = torch.where(use_gt[:, None, None, None], g, m)
        s = m.sum((2, 3))                                   # (B,1) leaf area in feature cells
        # unified masked-average embedding: independent of --pool, used as the representation the
        # SupCon loss shapes (so the same representation-learning term benefits every pooling mode)
        feat_avg = torch.where(s > 0.5, (f * m).sum((2, 3)) / (s + 1e-6), glob)

        if self.pool == "global":
            logits = self.cls(glob)
        elif self.pool == "mil":
            logits = self._mil_pool(self.cls_map(f), m)
        elif self.pool == "proto":
            logits = self._proto_pool(f, m)
        else:  # "mask"
            logits = self.cls(feat_avg)

        if return_feat:
            return seg, logits, feat_avg
        return seg, logits


def seg_loss_fn(logits, target, hm):
    """BCE + Dice, averaged over the images in the batch that have a mask."""
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none").mean((1, 2, 3))
    p = torch.sigmoid(logits)
    inter, tot = (p * target).sum((1, 2, 3)), p.sum((1, 2, 3)) + target.sum((1, 2, 3))
    per = bce + 1 - (2 * inter + 1) / (tot + 1)
    return (per * hm).sum() / hm.sum().clamp(min=1.0)


@torch.no_grad()
def init_prototypes(model, cache, recs, idx, img_size, device, batch_size, workers):
    """Warm-start each class prototype at the mean masked-leaf feature of its WHITE-background images.

    This is the concrete form of "we already know the leaf characteristics of each disease from white-
    background masking": before any gradient step, prototype[c] IS the average known appearance of class
    c on white images. End-to-end training (on white, and on field too with --field_train cls) then only
    has to refine that starting point -- including pulling field images' pooled features toward it -- not
    discover it from scratch.
    """
    white_masked = [i for i in idx if recs[i]["domain"] == 0 and recs[i]["mask"] is not None]
    if not white_masked:
        return
    dl = DataLoader(MTDataset(cache, recs, white_masked, img_size, False), batch_size=batch_size,
                    shuffle=False, num_workers=workers)
    model.eval()
    sums, counts = torch.zeros(NC, 512), torch.zeros(NC)
    for x, m, y, hm, _ in dl:
        x, m = x.to(device), m.to(device)
        f = model.l4(model.l3(model.l2(model.l1(model.pool0(model.stem(x)))))).float()
        mm = F.adaptive_avg_pool2d(m, f.shape[-2:])
        s = mm.sum((2, 3))
        feat = torch.where(s > 0.5, (f * mm).sum((2, 3)) / (s + 1e-6), f.mean((2, 3))).cpu()
        for c in range(NC):
            sel = y == c
            if sel.any():
                sums[c] += feat[sel].sum(0)
                counts[c] += sel.sum()
    for c in range(NC):
        if counts[c] > 0:
            model.proto.data[c] = F.normalize(sums[c] / counts[c], dim=0)
    print(f"  prototypes initialised from {int(counts.sum())} white masked images "
          f"({int((counts == 0).sum())} classes had none -> left at random init)")


def supcon_loss(z, labels, temperature=0.1):
    """Supervised contrastive loss (Khosla et al. 2020) on masked-average leaf embeddings.

    Positive pairs = any two images in the batch with the SAME disease label, regardless of white vs
    field background. This is the "learn representations, then classify" idea applied to the domain
    gap directly: instead of hoping a classifier head generalizes from white to field, the loss pulls
    same-disease embeddings together and different-disease embeddings apart -- across both domains at
    once -- so a field-photo lesion of a disease ends up close to that disease's white-photo cluster
    by construction, not by hope. Skips (with 0 loss) any sample with no same-class positive in the
    batch, which can happen with a small batch size or a rare class -- a larger batch strengthens it.
    """
    z = F.normalize(z.float(), dim=-1)
    B = z.shape[0]
    sim = (z @ z.t()) / temperature
    self_mask = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(self_mask, float("-inf"))
    pos = (labels.view(-1, 1) == labels.view(1, -1)) & ~self_mask
    valid = pos.any(1)
    if not valid.any():
        return z.new_zeros(())
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # log_prob is -inf on the (excluded) diagonal; multiplying by the 0/1 pos mask would give -inf*0 = nan
    # there, so select with `where` instead of multiplying
    per_sample = -torch.where(pos, log_prob, torch.zeros_like(log_prob)).sum(1) / pos.sum(1).clamp(min=1)
    return per_sample[valid].mean()


def set_encoder(model, trainable: bool):
    for mod in model.encoder_modules():
        for p in mod.parameters():
            p.requires_grad = trainable


def freeze_encoder_bn(model):
    for mod in model.encoder_modules():
        for m in mod.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


def train_epoch(model, loader, opt, sched, scaler, device, crit, a, frozen):
    model.train()
    if frozen:
        freeze_encoder_bn(model)
    tot = dict(loss=0.0, cls=0.0, seg=0.0, supcon=0.0, n=0)
    amp = device.type == "cuda"
    for x, m, y, hm, _ in loader:
        x, m, y, hm = x.to(device), m.to(device), y.to(device), hm.to(device)
        use_gt = (hm > 0) & (torch.rand(len(y), device=device) < a.tf_prob)
        with torch.autocast(device_type=device.type, enabled=amp):
            seg, lg, feat = model(x, m, use_gt, return_feat=True)
        l_cls = crit(lg.float(), y)
        l_seg = seg_loss_fn(seg.float(), m, hm) if a.lambda_seg > 0 else torch.zeros((), device=device)
        l_sup = supcon_loss(feat, y, a.supcon_temp) if a.lambda_supcon > 0 else torch.zeros((), device=device)
        loss = l_cls + a.lambda_seg * l_seg + a.lambda_supcon * l_sup
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        if sched is not None:
            sched.step()
        b = len(y)
        tot["loss"] += loss.item() * b
        tot["cls"] += l_cls.item() * b
        tot["seg"] += l_seg.item() * b
        tot["supcon"] += l_sup.item() * b
        tot["n"] += b
    return {k: v / tot["n"] for k, v in tot.items() if k != "n"}


@torch.no_grad()
def evaluate(model, loader, device):
    """Class logits for every image; Dice / IoU on the images that have a mask (predicted mask, threshold 0.5)."""
    model.eval()
    lg_all, ys, doms, dice, iou = [], [], [], [], []
    for x, m, y, hm, dom in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            seg, lg = model(x.to(device))
        p = (torch.sigmoid(seg.float()) > 0.5).float().cpu()
        inter, s = (p * m).sum((1, 2, 3)), p.sum((1, 2, 3)) + m.sum((1, 2, 3))
        keep = hm > 0
        dice.append(((2 * inter + 1e-6) / (s + 1e-6))[keep])
        iou.append(((inter + 1e-6) / (s - inter + 1e-6))[keep])
        lg_all.append(lg.float().cpu())
        ys.append(y)
        doms.append(dom)
    cat = lambda l: torch.cat(l).numpy()
    return cat(lg_all), cat(ys), cat(doms), cat(dice), cat(iou)


@torch.no_grad()
def save_pred_sheet(model, recs, idx, cache, device, size, path, n_field=16, n_white=8):
    """Contact sheet of predicted leaf outlines (red) on test images; green = mask used as label (white images)."""
    tf = make_tf(size, False)
    model.eval()
    fld = [i for i in idx if recs[i]["domain"] == 1][:n_field]
    wht = [i for i in idx if recs[i]["domain"] == 0 and recs[i]["mask"] is not None][:n_white]
    tiles = []
    for i in fld + wht:
        r = recs[i]
        rgb = cv2.cvtColor(cv2.imread(str(cache / r["rel"].with_suffix(".jpg"))), cv2.COLOR_BGR2RGB)
        x = tf(tv_tensors.Image(torch.from_numpy(rgb).permute(2, 0, 1))).as_subclass(torch.Tensor)[None].to(device)
        seg, lg = model(x)
        pm = F.interpolate(torch.sigmoid(seg.float()), size=rgb.shape[:2], mode="bilinear", align_corners=False)[0, 0].cpu().numpy() > 0.5
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if r["mask"] is not None:
            gm = cv2.imread(str(r["mask"]), cv2.IMREAD_GRAYSCALE)
            cv2.drawContours(bgr, cv2.findContours((gm > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 200, 0), 2)
        cv2.drawContours(bgr, cv2.findContours(pm.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], -1, (0, 0, 255), 2)
        pred = int(lg.argmax(1))
        s = min(170 / bgr.shape[1], 360 / bgr.shape[0])
        im = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        can = np.full((360 + 40, 170, 3), 255, np.uint8)
        can[:im.shape[0], :im.shape[1]] = im
        ok = pred == r["label"]
        for k, (txt, col) in enumerate([(f"{DOMAINS[r['domain']]} T:{CLASS_NAMES[r['label']][:12]}", (0, 0, 0)),
                                        (f"P:{CLASS_NAMES[pred][:12]} {'OK' if ok else 'WRONG'}", (0, 130, 0) if ok else (0, 0, 220))]):
            cv2.putText(can, txt, (2, 360 + 14 + 14 * k), cv2.FONT_HERSHEY_SIMPLEX, 0.36, col, 1, cv2.LINE_AA)
        tiles.append(can)
    if not tiles:
        return
    per_row = 8
    tiles += [np.full_like(tiles[0], 255)] * (-len(tiles) % per_row)
    cv2.imwrite(str(path), np.vstack([np.hstack(tiles[j:j + per_row]) for j in range(0, len(tiles), per_row)]),
                [cv2.IMWRITE_JPEG_QUALITY, 90])


def run_one(seed, a, recs, labels, domains, cache, device, out_dir):
    seed_everything(seed)
    tr_all, va, te = make_split(labels, domains, seed)   # same split as train_resnet18.py
    tr = tr_all if a.field_train == "cls" else tr_all[domains[tr_all] == 0]
    bg_files = [cache / recs[i]["rel"].with_suffix(".jpg") for i in tr_all if recs[i]["domain"] == 1] if a.bg_swap > 0 else []
    n_masked = sum(recs[i]["mask"] is not None for i in tr)
    print(f"[seed {seed}] train {len(tr)} ({n_masked} with mask)  val {len(va)}  test {len(te)}  "
          f"| background-swap pool: {len(bg_files)} field images (train split only)")

    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=device.type == "cuda",
              persistent_workers=a.workers > 0)
    gen = torch.Generator().manual_seed(seed)
    tr_ds = MTDataset(cache, recs, tr, a.img_size, True, a.bg_swap, bg_files)
    tl = DataLoader(tr_ds, shuffle=True, drop_last=len(tr) > a.batch_size, generator=gen, **kw)
    vl = DataLoader(MTDataset(cache, recs, va, a.img_size, False), shuffle=False, **kw)
    te_ds = MTDataset(cache, recs, te, a.img_size, False)
    tel = DataLoader(te_ds, shuffle=False, **kw)

    model = MultiTaskNet(a.weights_path, a.dropout, a.pool, not a.from_scratch, a.mil_topk_frac).to(device)
    if a.pool == "proto":
        init_prototypes(model, cache, recs, tr, a.img_size, device, a.batch_size, a.workers)
    w = None
    if a.class_weights:
        cnt = np.bincount(labels[tr], minlength=NC).astype(np.float32)
        w = torch.tensor(cnt.sum() / (NC * np.maximum(cnt, 1)), device=device)
    crit = nn.CrossEntropyLoss(weight=w, label_smoothing=a.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    dec = [p for n, p in model.named_parameters() if n.split(".")[0] in
          ("d4", "d3", "d2", "d1", "seg_out", "cls", "cls_map", "attn_v", "attn_u", "attn_w", "proto", "proto_scale")]
    enc = [p for n, p in model.named_parameters() if n.split(".")[0] in ("stem", "l1", "l2", "l3", "l4")]
    hist = []

    def val_metrics():
        lg, y, dom, dice, _ = evaluate(model, vl, device)
        f1 = f1_score(y, lg.argmax(1), labels=list(range(NC)), average="macro", zero_division=0)
        return float(f1), float((lg.argmax(1) == y).mean()), float(dice.mean()) if len(dice) else float("nan")

    # ---- stage 1: encoder frozen, train decoder + heads --------------------------------------
    if a.head_epochs > 0:
        set_encoder(model, False)
        opt = torch.optim.AdamW(dec, lr=a.lr_head, weight_decay=a.weight_decay)
        for ep in range(a.head_epochs):
            t = train_epoch(model, tl, opt, None, scaler, device, crit, a, frozen=True)
            f1, acc, dice = val_metrics()
            hist.append(dict(stage=1, epoch=ep + 1, **t, val_f1=f1, val_acc=acc, val_dice=dice))
            print(f"  [dec {ep + 1:>2}/{a.head_epochs}] cls {t['cls']:.3f} seg {t['seg']:.3f} supcon {t['supcon']:.3f} | val acc {acc:.3f} F1 {f1:.3f} dice {dice:.3f}")

    # ---- stage 2: fine-tune everything -------------------------------------------------------
    set_encoder(model, True)
    opt = torch.optim.AdamW([{"params": enc, "lr": a.lr_backbone}, {"params": dec, "lr": a.lr_head}], weight_decay=a.weight_decay)
    total = a.epochs * len(tl)
    sched = cosine_with_warmup(opt, min(len(tl), max(total // 10, 1)), total)
    best, best_state, bad = -1.0, None, 0
    for ep in range(a.epochs):
        t0 = time.time()
        t = train_epoch(model, tl, opt, sched, scaler, device, crit, a, frozen=False)
        f1, acc, dice = val_metrics()
        hist.append(dict(stage=2, epoch=ep + 1, **t, val_f1=f1, val_acc=acc, val_dice=dice))
        print(f"  [ft {ep + 1:>2}/{a.epochs}] cls {t['cls']:.3f} seg {t['seg']:.3f} supcon {t['supcon']:.3f} | val acc {acc:.3f} F1 {f1:.3f} dice {dice:.3f} ({time.time() - t0:.0f}s)")
        if f1 > best:  # model selection on validation macro-F1 only
            best, bad, best_state = f1, 0, copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop (no val macro-F1 gain for {a.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # ---- test ------------------------------------------------------------------------------
    lg, y, dom, dice, iou = evaluate(model, tel, device)
    pred = lg.argmax(1)
    res = dict(seed=seed, best_val_macro_f1=float(best), n_train=int(len(tr)), n_train_masked=int(n_masked), history=hist,
              scenarios={}, seg=dict(n=int(len(dice)), dice=float(dice.mean()) if len(dice) else None,
                                     iou=float(iou.mean()) if len(iou) else None))
    for name, msk in [("a_white", dom == 0), ("b_field", dom == 1), ("c_all", np.ones_like(dom, bool))]:
        if msk.any():
            res["scenarios"][name] = score(y[msk], pred[msk], NC)
            s = res["scenarios"][name]
            print(f"  test {name:<8} n={s['n']:<4} acc {s['acc']:.4f}  macroF1 {s['macro_f1']:.4f}")
    if len(dice):
        print(f"  test leaf segmentation (white, vs. mask labels): Dice {dice.mean():.4f}  IoU {iou.mean():.4f}  (n={len(dice)})")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed{seed}_result.json").write_text(json.dumps(res, indent=1))
    # one predicted disease per test image, with the model's probabilities
    prob = torch.softmax(torch.from_numpy(lg), 1).numpy()
    with open(out_dir / f"seed{seed}_predictions.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["image", "background", "true", "predicted", "correct", "confidence"] + [f"p_{c}" for c in CLASS_NAMES])
        for j, i in enumerate(te):
            wr.writerow([str(recs[i]["rel"]), DOMAINS[recs[i]["domain"]], CLASS_NAMES[y[j]], CLASS_NAMES[pred[j]],
                         int(y[j] == pred[j]), f"{prob[j].max():.3f}"] + [f"{p:.3f}" for p in prob[j]])
    save_pred_sheet(model, recs, list(te), cache, device, a.img_size, out_dir / f"seed{seed}_pred_masks.jpg")
    if a.save_models:
        torch.save(best_state, out_dir / f"seed{seed}_model.pt")
    return res


def cmd_train(a):
    if a.img_size % 32:
        raise SystemExit("--img_size must be a multiple of 32")
    root = a.data_root
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root.resolve()}")
    samples = discover_samples(root, a.allow_missing)
    cache = build_cache(root, samples, a.cache_size)
    recs = load_records(root, samples, a.mask_status, a.exclude_file or root / "_interim" / "qc" / "exclude.txt")
    labels = np.array([r["label"] for r in recs])
    domains = np.array([r["domain"] for r in recs])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = a.out_name or f"mt_{a.pool}_swap{a.bg_swap:g}_seg{a.lambda_seg:g}_field-{a.field_train}"
    out_dir = root / "_interim" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(a).items() if k != "func"}, indent=1))
    print(f"device: {device}   run: {tag}")

    results = [run_one(s, a, recs, labels, domains, cache, device, out_dir) for s in a.seeds]
    aggregate(results, out_dir, tag)
    sd = lambda v: float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
    d = [r["seg"]["dice"] for r in results if r["seg"]["dice"] is not None]
    j = [r["seg"]["iou"] for r in results if r["seg"]["iou"] is not None]
    if d:
        seg = dict(dice_mean=float(np.mean(d)), dice_std=sd(d), iou_mean=float(np.mean(j)), iou_std=sd(j), runs=len(d))
        (out_dir / "summary_seg.json").write_text(json.dumps(seg, indent=1))
        print(f"leaf segmentation on white test images: Dice {seg['dice_mean']:.4f} +/- {seg['dice_std']:.4f}   "
              f"IoU {seg['iou_mean']:.4f} +/- {seg['iou_std']:.4f}")
        print("  (measured against masks made by the classical segmenter, so this is agreement with those labels, not with hand-drawn truth)")
    print(f"\nresults and predicted-mask sheets written to {out_dir}")


def cmd_run(a):
    """Step 1 with its default settings, then step 2 with the given settings."""
    masks_args = argparse.Namespace(data_root=a.data_root, cache_size=a.cache_size, l_weight=0.0, workers=4,
                                    only=None, overwrite=False, z=3.5, n_pass_sample=32, exclude_file=None,
                                    seed=0, allow_missing=a.allow_missing)
    cmd_masks(masks_args)
    cmd_train(a)


# ==================================================================================
# Command line
# ==================================================================================
def add_masks_args(p):
    p.add_argument("--data_root", type=Path, default=Path("Dhan-Shomadhan"))
    p.add_argument("--cache_size", type=int, default=448, help="longest side of the cached resized images")
    p.add_argument("--l_weight", type=float, default=0.0, help="lightness weight in the segmenter (0 = ignore shadows)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--only", type=str, default=None, help="only process paths containing this text")
    p.add_argument("--overwrite", action="store_true", help="re-segment even if a mask already exists")
    p.add_argument("--z", type=float, default=3.5, help="outlier threshold (robust z-score)")
    p.add_argument("--n_pass_sample", type=int, default=32, help="passing masks to include in the review sheets")
    p.add_argument("--exclude_file", type=Path, default=None, help="text file, one relative image path per line")
    p.add_argument("--seed", type=int, default=0, help="seed for sampling the passing masks shown in the sheets")
    p.add_argument("--allow_missing", action="store_true")


def add_train_args(p):
    p.add_argument("--data_root", type=Path, default=Path("Dhan-Shomadhan"))
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--img_size", type=int, default=256, help="multiple of 32")
    p.add_argument("--cache_size", type=int, default=448, help="must match the masks step")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--head_epochs", type=int, default=3, help="stage 1: encoder frozen")
    p.add_argument("--epochs", type=int, default=30, help="stage 2 max epochs")
    p.add_argument("--lr_head", type=float, default=1e-3, help="decoder + heads")
    p.add_argument("--lr_backbone", type=float, default=2e-4, help="encoder")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lambda_seg", type=float, default=1.0, help="weight of the mask loss (0 = classification only)")
    p.add_argument("--pool", choices=["mask", "global", "mil", "proto"], default="mask",
                   help="classifier pooling (reported configuration: mask)")
    p.add_argument("--mil_topk_frac", type=float, default=0.2, help="--pool mil: fraction of leaf locations averaged per class")
    p.add_argument("--lambda_supcon", type=float, default=0.0, help="weight of the optional supervised-contrastive loss (0 = off)")
    p.add_argument("--supcon_temp", type=float, default=0.1, help="temperature for --lambda_supcon")
    p.add_argument("--bg_swap", type=float, default=0.5, help="probability of pasting a leaf onto a field background")
    p.add_argument("--tf_prob", type=float, default=0.5, help="probability of pooling over the true mask while training")
    p.add_argument("--field_train", choices=["none", "cls"], default="cls",
                   help="cls (reported): field images also train the class head; none: white images only")
    p.add_argument("--mask_status", default="ok,review", help="QC statuses whose masks are used as labels")
    p.add_argument("--exclude_file", type=Path, default=None)
    p.add_argument("--class_weights", action="store_true")
    p.add_argument("--from_scratch", action="store_true", help="random init instead of ImageNet weights")
    p.add_argument("--weights_path", type=str, default=None, help="local copy of resnet18-f37072fd.pth")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--out_name", type=str, default=None)
    p.add_argument("--allow_missing", action="store_true")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    pm = sub.add_parser("masks", help="step 1: resize cache, white-background leaf masks and QC")
    add_masks_args(pm)
    pm.set_defaults(func=cmd_masks)
    pt = sub.add_parser("train", help="step 2: train and evaluate MDB-Net over the seeds")
    add_train_args(pt)
    pt.set_defaults(func=cmd_train)
    pr = sub.add_parser("run", help="steps 1 and 2 in one command (step 1 with default settings)")
    add_train_args(pr)
    pr.set_defaults(func=cmd_run)
    a = parser.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
