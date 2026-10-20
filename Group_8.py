#!/usr/bin/env python3
"""CSCI935 Group 8 -- MDB-Net: Mask-Guided Domain-Bridging Multi-Task Network for
five-class rice leaf disease classification on the Dhan-Shomadhan dataset.

    image --> ResNet-18 encoder --+--> U-Net decoder --> leaf mask (segmentation head)
                                  |
                                  +--> features pooled over the leaf --> disease class (classification head)

Steps (run from the folder that contains Dhan-Shomadhan/):
    python Group_8.py masks       # 1. resize cache + classical leaf masks for white images + QC
    python Group_8.py train       # 2. five seeds, White / Field / Mixed test scenarios
    python Group_8.py run         # 1 + 2 in one command (reported scientific configuration)
    python Group_8.py verify --help  # independent prediction/metric verification and table export

To save all five selected checkpoints, use: python Group_8.py run --save_models
After training, independently verify/export the prediction-derived report tables:
    python Group_8.py verify --input Dhan-Shomadhan/_interim/runs/mt_mask_swap0.5_seg1_field-cls --output Dhan-Shomadhan/_interim/report_results --expected-seeds 0 1 2 3 4 --allow-partial-export
The output folder must be new. Without the optional original execution-evidence files,
this verifies prediction-derived metrics only, not historical checkpoint/run provenance.

Module origins (functions copied from the project, with entry-point/name/path adaptations):
    src/train_resnet18.py: shared data discovery, splits, weights, metrics and aggregation
    src/seg_leaf.py + src/make_masks.py: classical white-leaf masks and quality control
    src/train_multitask.py: MDB-Net, augmentation, training and test predictions
    src/member5_mdb_results.py: independent result verification and report-table export

Scope: the report's MDB-Net experiment and its result verification. Earlier MALG/U2-Net,
Pilot/test fixtures, the historical unpaired baseline and machine-specific execution/monitor
scripts are not submission entry points. Optional pooling/contrastive internals remain to
preserve the original parameter-initialisation RNG sequence; report defaults disable them.
No src/ files are needed at runtime. External historical audit evidence is optional and
is never manufactured by this program. Re-running training does not reproduce past bytes.

All generated data is written under Dhan-Shomadhan/_interim/:
    cache_448/   resized images            masks_448/   white-background leaf masks
    qc/          mask QC report + sheets   runs/<tag>/  results, predictions, confusion matrices

Required packages: torch, torchvision, numpy, opencv-python-headless, scikit-learn, matplotlib.
Pretrained weights: torchvision ResNet-18 IMAGENET1K_V1, downloaded automatically on first use;
offline: download https://download.pytorch.org/models/resnet18-f37072fd.pth and pass --weights_path.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import io
import copy
import csv
import difflib
import json
import math
import random
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor  # threads, not processes: forking after importing torch can deadlock
from datetime import datetime
from pathlib import Path, PurePosixPath

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
# Progress bar
# ----------------------------------------------------------------------------------
def progress(iterable, total, desc, keep=True, width=30):
    """Yield the items of `iterable` while drawing  desc |######......|  45% (90/200).

    keep=False erases the bar when done (used for per-epoch bars, which are followed by a
    one-line epoch summary). Nothing is drawn when stdout is redirected to a file.
    """
    show = sys.stdout.isatty() and total > 0
    last = -1

    def draw(i):
        nonlocal last
        pct = int(100 * i / total)
        if pct != last:
            last = pct
            fill = int(width * i / total)
            print(f"\r{desc} |{'#' * fill}{'.' * (width - fill)}| {pct:3d}% ({i}/{total})", end="", flush=True)

    if show:
        draw(0)
    i = 0
    for item in iterable:
        yield item
        i += 1
        if show:
            draw(i)
    if show:
        print() if keep else print("\r" + " " * (len(desc) + width + 24) + "\r", end="", flush=True)


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
NC = len(CLASS_NAMES)
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
    for rel, _, _ in progress(todo, len(todo), "  resizing"):
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
            results = list(progress(ex.map(_process, jobs), len(jobs), "  segmenting"))
    else:
        results = [_process(j) for j in progress(jobs, len(jobs), "  segmenting")]

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


def train_epoch(model, loader, opt, sched, scaler, device, crit, a, frozen, desc="  train"):
    model.train()
    if frozen:
        freeze_encoder_bn(model)
    tot = dict(loss=0.0, cls=0.0, seg=0.0, supcon=0.0, n=0)
    amp = device.type == "cuda"
    for x, m, y, hm, _ in progress(loader, len(loader), desc, keep=False):
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
def evaluate(model, loader, device, desc="  evaluate"):
    """Class logits for every image; Dice / IoU on the images that have a mask (predicted mask, threshold 0.5)."""
    model.eval()
    lg_all, ys, doms, dice, iou = [], [], [], [], []
    for x, m, y, hm, dom in progress(loader, len(loader), desc, keep=False):
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
        lg, y, dom, dice, _ = evaluate(model, vl, device, f"  [seed {seed}] validate")
        f1 = f1_score(y, lg.argmax(1), labels=list(range(NC)), average="macro", zero_division=0)
        return float(f1), float((lg.argmax(1) == y).mean()), float(dice.mean()) if len(dice) else float("nan")

    # ---- stage 1: encoder frozen, train decoder + heads --------------------------------------
    if a.head_epochs > 0:
        set_encoder(model, False)
        opt = torch.optim.AdamW(dec, lr=a.lr_head, weight_decay=a.weight_decay)
        for ep in range(a.head_epochs):
            t = train_epoch(model, tl, opt, None, scaler, device, crit, a, frozen=True,
                            desc=f"  [seed {seed}] decoder {ep + 1}/{a.head_epochs}")
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
        t = train_epoch(model, tl, opt, sched, scaler, device, crit, a, frozen=False,
                        desc=f"  [seed {seed}] fine-tune {ep + 1}/{a.epochs}")
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
    lg, y, dom, dice, iou = evaluate(model, tel, device, f"  [seed {seed}] test")
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
# Independent verification and report exports (src/member5_mdb_results.py)
# Training aggregate is separate from verification_aggregate.
# ==================================================================================
CLASSES = ('brown_spot', 'leaf_scald', 'rice_blast', 'rice_tungro', 'sheath_blight')
SCENARIOS = {'a_white': 'White', 'b_field': 'Field', 'c_all': 'Mixed'}
METRIC_TOL = 1e-8
ROUND_TOL = 0.00050001
SCIENCE_KEYS = ('img_size','cache_size','batch_size','head_epochs','epochs','lr_head',
 'lr_backbone','weight_decay','label_smoothing','dropout','patience','lambda_seg','pool',
 'mil_topk_frac','lambda_supcon','supcon_temp','bg_swap','tf_prob','field_train','mask_status',
 'class_weights','from_scratch','allow_missing')
ARG_KEYS = SCIENCE_KEYS + ('data_root','seeds','exclude_file','weights_path','workers','save_models','out_name')
INT_KEYS = {'img_size','cache_size','batch_size','head_epochs','epochs','patience','workers'}
BOOL_KEYS = {'class_weights','from_scratch','allow_missing','save_models'}
FLOAT_KEYS = set(SCIENCE_KEYS)-INT_KEYS-BOOL_KEYS-{'pool','field_train','mask_status'}
CSV_COLUMNS = ('image','background','true','predicted','correct','confidence') + tuple('p_'+c for c in CLASSES)
PROTOCOL = {'model':'MDB-Net','class_names':list(CLASSES),'split_ratios':[.7,.15,.15],
 'stratification':'class_background','selection':'stage2_validation_macro_f1',
 'macro_f1':'fixed_five_zero_division_0','std_ddof':1}
# Original source byte hashes for checking the report's historical four-file evidence.
# These identify historical source versions, not execution of this merged submission.
INTEGRATED_SOURCE_SHA256 = {'train_multitask.py': '5ba9c644e586089cb7579b9104c85dd0ad08c2ad377bc1c382675f6b2a22f982', 'train_resnet18.py': 'b86d3038ea87c7925a0c1c040dbd2ac00b3713b0f1f3c14b03941319a35e04d0', 'make_masks.py': '089a892c1734598e8f1b0a52e234fdc348fc37c0a37f762508acf1c5cc607ec9', 'seg_leaf.py': '8d6f8fcb07a52cf622db29e610242ba536c19ce9ca2a78361d154075daf9f52e'}
# External contracts; this module NEVER manufactures runner records.
APPROVAL_SCHEMA = {'schema_version':1,'status':'APPROVED','purpose':'formal','model':'MDB-Net',
 'approval_reference':'external dated approval reference','expected_seeds':'five integers',
 'protocol':PROTOCOL,'args':'all ARG_KEYS with approved native JSON values'}
EVIDENCE_SCHEMA = {'schema_version':1,'model':'MDB-Net','purpose':'formal',
 'dataset_manifest':'file reference to ordered JSON image/label/background/sha256 records',
 'source_code':'four audited script basenames -> file references',
 'preprocessing':'cache_manifest/mask_index/qc_report/exclude_file/policy -> file references',
 'runs':'five seed/model/purpose/timestamps/config/split/log/environment/result/predictions/checkpoint records',
 'pretrained_weights':'file reference when not from_scratch'}

class VerificationError(ValueError):
    pass

def require(ok, message):
    if not ok:
        raise VerificationError(message)

def number(v, ctx, low=None, high=None):
    require(type(v) in (int,float) and math.isfinite(v), f'{ctx}: finite number required')
    require(low is None or v >= low, f'{ctx}: below {low}')
    require(high is None or v <= high, f'{ctx}: above {high}')
    return float(v)

def integer(v, ctx, low=0):
    require(type(v) is int and v >= low, f'{ctx}: integer >= {low} required')
    return v

def close(v, expected, ctx, tolerance=METRIC_TOL):
    number(v,ctx)
    require(abs(v-expected) <= tolerance, f'{ctx}: saved {v!r} != recomputed {expected!r}')

def normalize_path(v):
    require(isinstance(v,str) and v and v == v.strip(), 'image: nonempty unpadded relative path required')
    require(not any(ord(c)<32 for c in v), 'image: control character')
    v=v.replace('\\','/'); parts=v.split('/')
    require(not v.startswith('/') and ':' not in v and all(p not in ('','.','..') for p in parts), f'image: unsafe relative path {v!r}')
    return PurePosixPath(v).as_posix()

def file_hash(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()

def _unique_object(pairs):
    out={}
    for k,v in pairs:
        require(k not in out, f'JSON duplicate key: {k}')
        out[k]=v
    return out

def _check_json_numbers(value, location=()):
    if isinstance(value, dict):
        for key, item in value.items():
            _check_json_numbers(item, location+(key,))
    elif isinstance(value, list):
        for i, item in enumerate(value):
            _check_json_numbers(item, location+(i,))
    elif type(value) is float and not math.isfinite(value):
        # Upstream explicitly writes val_dice=NaN when there are no validation masks.
        require(math.isnan(value) and len(location)>=3 and location[-1]=='val_dice' and 'history' in location,
                f'JSON nonfinite value outside documented history/val_dice exception: {location}')

def read_json(path):
    try:
        value=json.loads(Path(path).read_text(encoding='utf-8-sig'),object_pairs_hook=_unique_object)
        _check_json_numbers(value)
        return value
    except (OSError,ValueError) as exc:
        raise VerificationError(f'{path}: invalid JSON/unreadable: {exc}') from exc

def seeds_checked(seeds):
    require(isinstance(seeds,(list,tuple)) and len(seeds)==5,'Exactly five externally supplied expected seeds required')
    require(all(type(s) is int and 0<=s<2**32 for s in seeds),'seeds: integers in [0,2**32) required')
    require(len(set(seeds))==5,'Duplicate expected seeds')
    return list(seeds)

def parse_args_record(raw):
    require(isinstance(raw,dict),'args.json: object required')
    require(set(ARG_KEYS)<=set(raw), f'args.json: missing keys {sorted(set(ARG_KEYS)-set(raw))}')
    for k in ('model','variant','model_architecture'):
        require(k not in raw or raw[k]=='MDB-Net', f'args: non-MDB model marker {k}')
    for k in ('pilot','quick','synthetic'):
        require(raw.get(k) not in (True,'True','true'), f'args: {k} cannot count as formal')
    require(raw.get('purpose','formal')=='formal','args: Pilot/nonformal purpose')
    args={}
    for k in ARG_KEYS:
        v=raw[k]
        if isinstance(v,str) and k in INT_KEYS|FLOAT_KEYS|BOOL_KEYS|{'seeds'}:
            try:
                v=ast.literal_eval(v)
            except (SyntaxError,ValueError) as exc:
                raise VerificationError(f'args/{k}: invalid typed value') from exc
        elif v=='None' and k in ('exclude_file','weights_path','out_name'):
            v=None
        if k in INT_KEYS:
            integer(v,f'args/{k}',0 if k in ('head_epochs','patience','workers') else 1)
        elif k in FLOAT_KEYS:
            v=number(v,f'args/{k}',0)
        elif k in BOOL_KEYS:
            require(type(v) is bool,f'args/{k}: boolean required')
        args[k]=v
    args['seeds']=seeds_checked(args['seeds'])
    require(args['pool'] in ('mask','global','mil','proto'),'args/pool invalid')
    require(args['field_train'] in ('none','cls'),'args/field_train invalid')
    require(args['img_size']%32==0,'args/img_size must be multiple of 32')
    for k in ('label_smoothing','dropout','bg_swap','tf_prob','mil_topk_frac'):
        require(args[k]<=1,f'args/{k}: must be <=1')
    for k in ('lr_head','lr_backbone','supcon_temp','mil_topk_frac'):
        require(args[k]>0,f'args/{k}: positive required')
    require(isinstance(args['mask_status'],str) and args['mask_status'] and set(args['mask_status'].split(','))<={'ok','review','fail','excluded'},'args/mask_status invalid')
    require(isinstance(args['data_root'],str) and bool(args['data_root']),'args/data_root invalid')
    for k in ('exclude_file','weights_path','out_name'):
        require(args[k] is None or isinstance(args[k],str),f'args/{k}: string/null required')
    require(not re.search(r'(^|[^a-z])(pilot|smoke|synthetic|malg)([^a-z]|$)',str(args['out_name']).lower()),'args/out_name: Pilot/synthetic/MALG marker')
    return args

def read_predictions(path,warnings):
    try:
        with Path(path).open(encoding='utf-8-sig',newline='') as f:
            reader=csv.DictReader(f); cols=reader.fieldnames or []
            require(len(cols)==len(set(cols)) and set(CSV_COLUMNS)<=set(cols),f'{path}: missing/duplicate MDB CSV columns')
            rows=list(reader)
    except (OSError,csv.Error,UnicodeError) as exc:
        raise VerificationError(f'{path}: unreadable CSV: {exc}') from exc
    require(rows,f'{path}: empty predictions')
    seen,out,tied=set(),[],0
    for i,r in enumerate(rows,2):
        ctx=f'{path.name}:row {i}'
        require(None not in r and all(v is not None for v in r.values()),f'{ctx}: malformed CSV row')
        image=normalize_path(r['image'])
        require(image.casefold() not in seen,f'{ctx}: duplicate image path including Windows aliases')
        seen.add(image.casefold())
        require(r['background'] in ('white','field'),f'{ctx}: invalid background')
        require(r['true'] in CLASSES and r['predicted'] in CLASSES,f'{ctx}: invalid five-class label')
        y,p=CLASSES.index(r['true']),CLASSES.index(r['predicted'])
        require(r['correct'] in ('0','1') and int(r['correct'])==int(y==p),f'{ctx}: incorrect correct flag')
        try:
            prob=[float(r['p_'+c]) for c in CLASSES]; confidence=float(r['confidence'])
        except ValueError as exc:
            raise VerificationError(f'{ctx}: invalid probability') from exc
        for v in prob+[confidence]:
            number(v,f'{ctx}/probability',0,1)
        require(abs(sum(prob)-1)<=5*ROUND_TOL,f'{ctx}: probability sum exceeds three-decimal rounding bound')
        require(max(prob)-prob[p]<=2*ROUND_TOL,f'{ctx}: prediction contradicts rounded probabilities')
        require(abs(confidence-max(prob))<=2*ROUND_TOL,f'{ctx}: confidence differs from rounded maximum')
        if sum(abs(v-max(prob))<=2*ROUND_TOL for v in prob)>1:
            tied+=1
        out.append(dict(image=image,background=r['background'],true=y,predicted=p,correct=int(y==p),confidence=confidence,probabilities=prob))
    if tied:
        warnings.append(f'{path.name}: {tied} rounded/near-tied rows; exact logits/argmax cannot be reconstructed')
    return out

def compute_metrics(rows):
    out={}
    for sc in SCENARIOS:
        sub=[r for r in rows if sc=='c_all' or r['background']==('white' if sc=='a_white' else 'field')]
        require(sub,f'{sc}: empty required scenario')
        y,p=[r['true'] for r in sub],[r['predicted'] for r in sub]
        out[sc]=dict(n=len(y),acc=float(accuracy_score(y,p)),macro_f1=float(f1_score(y,p,labels=list(range(5)),average='macro',zero_division=0)),
            per_class_f1=f1_score(y,p,labels=list(range(5)),average=None,zero_division=0).tolist(),cm=confusion_matrix(y,p,labels=list(range(5))).tolist())
    require(out['a_white']['n']+out['b_field']['n']==out['c_all']['n'],'scenario coverage invalid')
    return out

def compare_metric_record(saved,expected,ctx):
    require(isinstance(saved,dict),f'{ctx}: metric object required')
    require(type(saved.get('n')) is int and saved['n']==expected['n'],f'{ctx}: sample count mismatch')
    for k in ('acc','macro_f1'):
        close(saved.get(k),expected[k],f'{ctx}/{k}')
    pc=saved.get('per_class_f1')
    require(isinstance(pc,list) and len(pc)==5,f'{ctx}: five per-class F1 required')
    for i,(v,e) in enumerate(zip(pc,expected['per_class_f1'])):
        close(v,e,f'{ctx}/per_class_f1/{i}')
    cm=saved.get('cm')
    require(isinstance(cm,list) and len(cm)==5 and all(isinstance(r,list) and len(r)==5 and all(type(v) is int and v>=0 for v in r) for r in cm),f'{ctx}: integer 5x5 confusion matrix required')
    require(cm==expected['cm'],f'{ctx}: confusion matrix mismatch')

def verify_history(result,args,warnings):
    hist=result.get('history')
    require(isinstance(hist,list) and hist,'history: nonempty list required')
    counts,ft,seen_ft={1:0,2:0},[],False
    for h in hist:
        require(isinstance(h,dict) and type(h.get('stage')) is int and h['stage'] in (1,2),'history: invalid stage')
        stage=h['stage']; require(not(seen_ft and stage==1),'history: head after fine-tuning')
        seen_ft |= stage==2; counts[stage]+=1
        require(type(h.get('epoch')) is int and h['epoch']==counts[stage],'history: noncontiguous/duplicate epoch')
        for k in ('loss','cls','seg','supcon'):
            number(h.get(k),f'history/{k}',0)
        for k in ('val_f1','val_acc'):
            number(h.get(k),f'history/{k}',0,1)
        vd=h.get('val_dice')
        if type(vd) in (int,float) and math.isnan(vd):
            warnings.append('history val_dice NaN: source can emit this with no trusted validation masks; segmentation NOT VERIFIED')
        else:
            number(vd,'history/val_dice',0,1)
        close(h['loss'],h['cls']+args['lambda_seg']*h['seg']+args['lambda_supcon']*h['supcon'],'history/weighted_loss',1e-6)
        if stage==2:
            ft.append(h)
    require(counts[1]==args['head_epochs'],'history: incomplete/excess head epochs')
    require(0<len(ft)<=args['epochs'],'history: incomplete/excess fine-tuning epochs')
    best,best_epoch,bad,stop=-1.0,None,0,False
    for h in ft:
        require(not stop,'history: continued after patience early stop')
        if h['val_f1']>best:
            best,best_epoch,bad=h['val_f1'],h['epoch'],0
        else:
            bad+=1; stop=bad>=args['patience']
    require(len(ft)==args['epochs'] or stop,'history: truncated without early-stop condition')
    close(result.get('best_val_macro_f1'),best,'history/best_val_macro_f1')
    if 'best_epoch' in result:
        require(type(result['best_epoch']) is int and result['best_epoch']==best_epoch, 'history: saved best_epoch conflicts with inferred selection')
    integer(result.get('n_train'),'n_train',1); integer(result.get('n_train_masked'),'n_train_masked')
    require(result['n_train_masked']<=result['n_train'],'n_train_masked exceeds n_train')
    if result['n_train_masked']==0 or args['lambda_seg']==0:
        require(all(h['seg']==0 for h in hist), 'history: nonzero segmentation loss without active mask supervision')
    if args['lambda_supcon']==0:
        require(all(h['supcon']==0 for h in hist), 'history: nonzero disabled contrastive loss')
    return dict(head_epochs=counts[1],finetune_epochs=len(ft),inferred_best_epoch=best_epoch,
        best_val_macro_f1=best,checkpoint_status='CHECKPOINT IDENTITY NOT VERIFIED')

def verification_aggregate(runs):
    require(len(runs)==5 and len({r['seed'] for r in runs})==5,'aggregate: five unique seeds required')
    out={}
    for sc in SCENARIOS:
        ms=[r['metrics'][sc] for r in runs]; v={'runs':5}
        for k in ('acc','macro_f1'):
            values=[x[k] for x in ms]; v[k+'_mean'],v[k+'_std']=float(np.mean(values)),float(np.std(values,ddof=1))
        pcs=np.array([x['per_class_f1'] for x in ms])
        v['per_class_f1_mean']=dict(zip(CLASSES,pcs.mean(0).round(4).tolist()))
        v['per_class_f1_std']=dict(zip(CLASSES,pcs.std(0,ddof=1).tolist()))
        v['cm_sum']=np.sum([x['cm'] for x in ms],axis=0).tolist(); out[sc]=v
    return out

def compare_summary(saved,computed):
    require(isinstance(saved,dict) and set(saved)==set(SCENARIOS),'summary.json: three exact scenarios required')
    for sc,v in computed.items():
        s=saved[sc]
        require(isinstance(s,dict) and type(s.get('runs')) is int and s['runs']==5,f'summary/{sc}: five runs required')
        for k in ('acc_mean','acc_std','macro_f1_mean','macro_f1_std'):
            close(s.get(k),v[k],f'summary/{sc}/{k}')
        for k in ('per_class_f1_mean','per_class_f1_std'):
            require(isinstance(s.get(k),dict) and set(s[k])==set(CLASSES),f'summary/{sc}/{k}: five classes required')
            for c in CLASSES:
                close(s[k][c],v[k][c],f'summary/{sc}/{k}/{c}')
        require(s.get('cm_sum')==v['cm_sum'],f'summary/{sc}: cm_sum mismatch')

def segmentation_summary(runs,root):
    valid=[]
    for r in runs:
        seg=r['result'].get('seg'); require(isinstance(seg,dict),f"seed {r['seed']}: missing seg record")
        n=integer(seg.get('n'),'seg/n'); require(n<=len(r['rows']),'seg/n exceeds test count')
        if n:
            for k in ('dice','iou'):
                number(seg.get(k),'seg/'+k,0,1)
            valid.append(seg)
        else:
            require(seg.get('dice') is None and seg.get('iou') is None,'seg: zero count requires null dice/iou')
    out={'interpretation':'Agreement with automatic pseudo masks, NOT hand-annotated segmentation accuracy','runs':len(valid)}
    if valid:
        for k in ('dice','iou'):
            values=[s[k] for s in valid]; out[k+'_mean'],out[k+'_std']=float(np.mean(values)),float(np.std(values,ddof=1)) if len(values)>1 else 0.0
    p=root/'summary_seg.json'
    if p.exists():
        saved=read_json(p)
        require(isinstance(saved,dict) and type(saved.get('runs')) is int and saved['runs']==len(valid) and valid,'summary_seg: count mismatch/stale output')
        for k in ('dice_mean','dice_std','iou_mean','iou_std'):
            close(saved.get(k),out[k],'summary_seg/'+k)
    return out

def _file_ref(ref,base,hashes):
    require(isinstance(ref,dict) and isinstance(ref.get('path'),str) and re.fullmatch(r'[0-9a-fA-F]{64}',str(ref.get('sha256',''))),'evidence: file reference requires path and SHA256')
    p=Path(ref['path']); p=(base/p).resolve() if not p.is_absolute() else p.resolve()
    require(p.is_file() and file_hash(p)==ref['sha256'].lower(),f'evidence: missing file/hash mismatch {p}')
    hashes[str(p)]=file_hash(p); return p

def verify_external(approval_path,evidence_path,root,args,runs,seeds,hashes):
    """Check externally supplied evidence bindings, not record authorship/training truth."""
    limitations=[]; approved=None
    if approval_path:
        approved=read_json(approval_path)
        require(isinstance(approved,dict) and approved.get('schema_version')==1 and approved.get('status')=='APPROVED'
            and approved.get('purpose')=='formal' and approved.get('model')=='MDB-Net','approved-config: invalid approval/model/purpose')
        require(isinstance(approved.get('approval_reference'),str) and approved['approval_reference'].strip(),'approved-config: reference missing')
        require(set(seeds_checked(approved.get('expected_seeds')))==set(seeds),'approved-config: seed conflict')
        require(approved.get('protocol')==PROTOCOL,'approved-config: evaluation/split protocol conflict')
        require(parse_args_record(approved.get('args'))==args,'approved-config: actual args conflict')
        hashes[str(Path(approval_path).resolve())]=file_hash(approval_path)
    else:
        limitations.append('Approved MDB-Net configuration NOT VERIFIED; expected-seeds are not proof of approval')
    if not evidence_path:
        return limitations+['Per-seed configurations/model provenance NOT VERIFIED',
            'Actual train/validation/test manifests, hashes and distinct splits NOT VERIFIED',
            'CHECKPOINT IDENTITY NOT VERIFIED','Logs/environment/preprocessing provenance NOT VERIFIED',
            'Pilot exclusion cannot be proved from shared args/file names alone']
    ev=read_json(evidence_path)
    require(isinstance(ev,dict) and ev.get('schema_version')==1 and ev.get('model')=='MDB-Net' and ev.get('purpose')=='formal','evidence: model/schema/purpose invalid (Pilot/MALG rejected)')
    require(args['save_models'], 'evidence: save_models must be enabled for checkpoint verification')
    base=Path(evidence_path).resolve().parent; hashes[str(Path(evidence_path).resolve())]=file_hash(evidence_path)
    code=ev.get('source_code'); require(isinstance(code,dict),'evidence: source_code missing')
    for name in ('train_multitask.py','train_resnet18.py','make_masks.py','seg_leaf.py'):
        p=_file_ref(code.get(name),base,hashes)
        require(file_hash(p)==INTEGRATED_SOURCE_SHA256[name],f'evidence: audited historical source version mismatch {name}')
    prep=ev.get('preprocessing'); require(isinstance(prep,dict),'evidence: preprocessing missing')
    for name in ('cache_manifest','mask_index','qc_report','exclude_file','policy'):
        _file_ref(prep.get(name),base,hashes)
    policy=read_json(_file_ref(prep['policy'],base,hashes))
    require(isinstance(policy,dict) and policy.get('status')=='APPROVED' and policy.get('approval_reference') and
        all(isinstance(policy.get(k),str) and policy[k].strip() for k in ('mask_qc_scope','test_exposure','near_duplicate_policy')),'evidence: leakage/preprocessing policy approval incomplete')
    manifest_path=_file_ref(ev.get('dataset_manifest'),base,hashes)
    if approved:
        require(approved.get('dataset_manifest_sha256')==file_hash(manifest_path), 'approved-config: frozen dataset manifest hash missing/mismatched')
    if not args['from_scratch']:
        weights=_file_ref(ev.get('pretrained_weights'),base,hashes)
        expected_weight=Path(args['weights_path']).resolve() if args['weights_path'] else (Path.home()/'.cache/torch/hub/checkpoints/resnet18-f37072fd.pth').resolve()
        require(weights==expected_weight, 'evidence: pretrained weight not bound to actual configured source')
        if approved:
            require(approved.get('pretrained_weights_sha256')==file_hash(weights), 'approved-config: pretrained weight hash missing/mismatched')
    manifest=read_json(manifest_path)
    require(isinstance(manifest,list) and manifest,'evidence: ordered dataset manifest required')
    dataset,seen,source_hashes={},set(),set(); rawroot=Path(args['data_root']).resolve()
    for key in ('mask_index','qc_report'):
        require(_file_ref(prep[key],base,hashes)==(rawroot/'_interim'/'qc'/(key+'.csv')).resolve(), f'evidence: {key} not bound to training data root')
    excluded_path=Path(args['exclude_file']).resolve() if args['exclude_file'] else (rawroot/'_interim/qc/exclude.txt').resolve()
    require(_file_ref(prep['exclude_file'],base,hashes)==excluded_path, 'evidence: exclusion file not bound to actual args')
    for row in manifest:
        image=normalize_path(row['image']); require(image.casefold() not in seen,'evidence: duplicate manifest path')
        seen.add(image.casefold())
        require(row['label'] in CLASSES and row['background'] in ('white','field'),'evidence: invalid manifest label/background')
        p=(rawroot/image).resolve()
        require(p.is_relative_to(rawroot) and p.is_file(),'evidence: source path missing/escapes root')
        require(re.fullmatch('[0-9a-f]{64}',str(row.get('sha256',''))) and file_hash(p)==row['sha256'],'evidence: source image SHA mismatch')
        require(row['sha256'] not in source_hashes,'evidence: identical images across paths')
        source_hashes.add(row['sha256']); dataset[image]=row
    # Verify full raw membership and discovery order, not just a self-selected Pilot subset.
    raw_paths={p.relative_to(rawroot).as_posix() for p in rawroot.rglob('*') if p.is_file() and p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.tif','.tiff') and not any(x.startswith('_') for x in p.relative_to(rawroot).parts)}
    require(set(dataset)==raw_paths and list(dataset)==[p.as_posix() for p in sorted(Path(x) for x in dataset)], 'evidence: dataset membership/order differs from raw discovery')
    cache_records=read_json(_file_ref(prep['cache_manifest'],base,hashes))
    require(isinstance(cache_records,list) and len(cache_records)==len(dataset), 'evidence: cache/mask manifest must cover all source images')
    index_path=_file_ref(prep['mask_index'],base,hashes)
    with index_path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        require(set(('image','cache_image','mask','class','status'))<=set(reader.fieldnames or []), 'evidence: mask index columns missing')
        index={}
        for row in reader:
            image=normalize_path(row['image'])
            require(image in dataset and image not in index and row['class']==dataset[image]['label'], 'evidence: mask index identity/label mismatch')
            index[image]=row
    with _file_ref(prep['qc_report'],base,hashes).open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        require(set(('image','cls','status'))<=set(reader.fieldnames or []), 'evidence: QC columns missing')
        qc={}
        for row in reader:
            image=normalize_path(row['image'])
            require(image in index and image not in qc and row['cls']==index[image]['class'] and row['status']==index[image]['status'], 'evidence: mask index/QC inconsistency')
            qc[image]=row
    require(set(qc)==set(index), 'evidence: mask index/QC coverage mismatch')
    trusted=set(); cache_seen=set()
    for row in cache_records:
        image=normalize_path(row['image']); require(image in dataset and image not in cache_seen, 'evidence: cache manifest identity mismatch')
        cache_seen.add(image)
        cache_path=_file_ref(row.get('cache'),base,hashes)
        require(cache_path==(rawroot/'_interim'/f"cache_{args['cache_size']}"/Path(image).with_suffix('.jpg')).resolve(), 'evidence: cache path not bound to actual loader')
        status=row.get('status'); require(status in ('ok','review','fail','excluded','none'), 'evidence: invalid mask status')
        actual=index.get(image)
        if actual is None:
            require(row.get('mask') is None and status=='none', 'evidence: mask manifest invents unindexed mask')
        else:
            require(status==actual['status'] and row.get('mask') is not None, 'evidence: mask index/cache manifest conflict')
            require((rawroot/normalize_path(actual['cache_image'])).resolve()==cache_path, 'evidence: index cache_image mismatch')
            require(_file_ref(row['mask'],base,hashes)==(rawroot/normalize_path(actual['mask'])).resolve(), 'evidence: index mask path mismatch')
        if row.get('mask') is not None:
            _file_ref(row['mask'],base,hashes)
            if status in args['mask_status'].split(','):
                trusted.add(image)
        else:
            require(status=='none', 'evidence: mask status without mask')
    excluded={normalize_path(x.strip()) for x in _file_ref(prep['exclude_file'],base,hashes).read_text(encoding='utf-8-sig').splitlines() if x.strip()}
    trusted-=excluded
    if approved:
        require(approved.get('preprocessing_sha256')=={k:prep[k]['sha256'].lower() for k in prep}, 'approved-config: frozen preprocessing hashes missing/mismatched')
    records=ev.get('runs')
    require(isinstance(records,list) and len(records)==5 and all(isinstance(r,dict) for r in records),'evidence: five run records required')
    require(all(type(r.get('seed')) is int for r in records) and {r['seed'] for r in records}==set(seeds),'evidence: missing/duplicate/unexpected seed')
    fingerprints,counts=set(),[]; ordered=list(dataset)
    from sklearn.model_selection import train_test_split
    strata=np.array([f"{CLASSES.index(dataset[p]['label'])}_{0 if dataset[p]['background']=='white' else 1}" for p in ordered])
    for record in records:
        seed=record['seed']; run=next(r for r in runs if r['seed']==seed)
        require(record.get('purpose')=='formal' and record.get('model')=='MDB-Net','evidence: Pilot/non-MDB run')
        try:
            start,end=[datetime.fromisoformat(record[k].replace('Z','+00:00')) for k in ('started_at','completed_at')]
            require(start.tzinfo is not None and end.tzinfo is not None and start<end,'evidence: invalid timestamps')
        except (KeyError,ValueError,TypeError,AttributeError) as exc:
            raise VerificationError('evidence: timezone-aware run timestamps required') from exc
        cfg=read_json(_file_ref(record.get('config'),base,hashes))
        require(isinstance(cfg,dict) and type(cfg.get('seed')) is int and cfg['seed']==seed and cfg.get('model')=='MDB-Net' and cfg.get('purpose')=='formal','evidence: per-seed identity mismatch')
        require(cfg.get('protocol')==PROTOCOL and parse_args_record(cfg.get('args'))==args,'evidence: per-seed scientific configuration conflict')
        log=_file_ref(record.get('log'),base,hashes); require(log.stat().st_size>0,'evidence: empty log')
        env=read_json(_file_ref(record.get('environment'),base,hashes))
        require(isinstance(env,dict) and all(isinstance(env.get(k),str) and env[k].strip() for k in ('python','torch','torchvision','numpy','scikit-learn','device')),'evidence: environment incomplete')
        for key,name in (('result',f'seed{seed}_result.json'),('predictions',f'seed{seed}_predictions.csv')):
            require(_file_ref(record.get(key),base,hashes)==(root/name).resolve(),f'evidence: {key} not bound to input')
        split=read_json(_file_ref(record.get('split'),base,hashes))
        require(isinstance(split,dict) and all(isinstance(split.get(k),list) and split[k] for k in ('train','val','test','training_images')) and isinstance(split.get('background_swap_images'),list),'evidence: actual split/training/background lists required')
        parts={k:[normalize_path(p) for p in split[k]] for k in ('train','val','test')}
        for k,v in parts.items():
            require(len(v)==len(set(v)),f'evidence: duplicate split path {k}')
        sets={k:set(v) for k,v in parts.items()}
        require(not(sets['train']&sets['val'] or sets['train']&sets['test'] or sets['val']&sets['test']) and set.union(*sets.values())==set(dataset),'evidence: overlapping/incomplete split')
        idx=np.arange(len(ordered)); tv,te=train_test_split(idx,test_size=.15,stratify=strata,random_state=seed)
        tr,va=train_test_split(tv,test_size=.15/.85,stratify=strata[tv],random_state=seed)
        for k,ids in zip(('train','val','test'),(tr,va,te)):
            require(sets[k]=={ordered[i] for i in ids},f'evidence: split does not reproduce MDB algorithm: {k}')
        train=sets['train'] if args['field_train']=='cls' else {p for p in sets['train'] if dataset[p]['background']=='white'}
        bg={p for p in sets['train'] if dataset[p]['background']=='field'} if args['bg_swap']>0 else set()
        require({normalize_path(p) for p in split['training_images']}==train and {normalize_path(p) for p in split['background_swap_images']}==bg,'evidence: training/swap pool mismatch')
        require(run['result']['n_train']==len(train),'evidence: n_train differs from split')
        require(run['result']['n_train_masked']==len(train&trusted), 'evidence: n_train_masked differs from frozen mask records')
        require(run['result']['seg']['n']==len(sets['test']&trusted), 'evidence: segmentation count differs from frozen test masks')
        require({r['image'] for r in run['rows']}==sets['test'],'evidence: predictions do not cover exact test manifest')
        for r in run['rows']:
            require(CLASSES[r['true']]==dataset[r['image']]['label'] and r['background']==dataset[r['image']]['background'],'evidence: prediction truth differs from manifest')
        cp=record.get('checkpoint')
        require(isinstance(cp,dict) and cp.get('stage')==2 and cp.get('epoch')==run['history']['inferred_best_epoch'] and cp.get('selection')==PROTOCOL['selection'],'evidence: checkpoint selection/epoch mismatch')
        p=_file_ref(cp,base,hashes)
        require(p==(root/f'seed{seed}_model.pt').resolve() and p.stat().st_size>0,'evidence: checkpoint not bound to saved input model')
        run['history']['checkpoint_status']='VERIFIED (runner record, hash and epoch binding; weights not deserialized)'
        fingerprints.add(tuple(tuple(sorted(sets[k])) for k in ('train','val','test')))
        counts.append(tuple(len(sets[k]) for k in ('train','val','test')))
    require(len(fingerprints)==5 and len(set(counts))==1,'evidence: distinct splits with constant counts required')
    if not approved:
        limitations.append('Execution records supplied, but formal approval absent')
    limitations.append('Evidence integrity is relative to external runner records, NOT independent proof of authorship/training or absence of near-duplicate/development leakage')
    return limitations

def audit_experiment(input_dir,expected_seeds,approved_config=None,evidence_manifest=None):
    root=Path(input_dir).resolve()
    audit=dict(status='INVALID',validation='INVALID',errors=[],warnings=[],limitations=[],input=str(root),
        expected_seeds=list(expected_seeds),runs=[],summary={},input_hashes={},protocol=PROTOCOL)
    hashes=audit['input_hashes']
    try:
        seeds=seeds_checked(expected_seeds); require(root.is_dir(),'input: directory missing')
        require(not re.search(r'(^|[^a-z])(pilot|smoke|synthetic|malg)([^a-z]|$)',root.name.lower()),'input: Pilot/synthetic/MALG marker')
        require(not any(p.name.startswith('seed_') for p in root.iterdir()),'input: legacy MALG seed_ layout rejected')
        args=parse_args_record(read_json(root/'args.json')); audit['args']=args
        require(set(args['seeds'])==set(seeds),'args: expected seeds conflict')
        expected={f'seed{s}_{suffix}' for s in seeds for suffix in ('result.json','predictions.csv')}
        present={p.name for p in root.iterdir() if p.name.startswith('seed') and (p.name.endswith('_result.json') or p.name.endswith('_predictions.csv'))}
        require(present==expected,f'input: missing/extra seed files; missing={sorted(expected-present)}, extra={sorted(present-expected)}')
        for p in root.iterdir():
            if p.name.startswith('seed'):
                match=re.fullmatch(r'seed(\d+)_(result\.json|predictions\.csv|pred_masks\.jpg|model\.pt)',p.name)
                require(match is not None and int(match[1]) in seeds,f'input: unexpected/stale seed artifact {p.name}')
        for seed in seeds:
            try:
                rp,pp=root/f'seed{seed}_result.json',root/f'seed{seed}_predictions.csv'
                result=read_json(rp)
                require(isinstance(result,dict) and type(result.get('seed')) is int and result['seed']==seed,f'seed {seed}: identity mismatch')
                require(result.get('model','MDB-Net')=='MDB-Net' and result.get('purpose','formal')=='formal' and not result.get('pilot',False),f'seed {seed}: nonformal/non-MDB result')
                rows=read_predictions(pp,audit['warnings']); metrics=compute_metrics(rows)
                require(isinstance(result.get('scenarios'),dict) and set(result['scenarios'])==set(SCENARIOS),f'seed {seed}: scenarios mismatch')
                for sc in SCENARIOS:
                    compare_metric_record(result['scenarios'][sc],metrics[sc],f'seed {seed}/{sc}')
                history=verify_history(result,args,audit['warnings'])
                audit['runs'].append(dict(seed=seed,result=result,rows=rows,metrics=metrics,history=history,predictions_source=str(pp)))
            except (VerificationError,KeyError,TypeError,ValueError,OSError) as exc:
                audit['errors'].append(f'seed {seed}: {exc}')
        require(not audit['errors'],'One or more seeds INVALID; no aggregate produced')
        audit['summary']=verification_aggregate(audit['runs']); compare_summary(read_json(root/'summary.json'),audit['summary'])
        audit['segmentation']=segmentation_summary(audit['runs'],root)
        for p in root.iterdir():
            if p.is_file() and (p.name in ('args.json','summary.json','summary_seg.json') or p.name.startswith('seed')):
                hashes[str(p)]=file_hash(p)
        audit['limitations']=verify_external(approved_config,evidence_manifest,root,args,audit['runs'],seeds,hashes)
        audit['validation']='VERIFIED'
        audit['status']='COMPLETE_VERIFIED' if approved_config is not None and evidence_manifest is not None else 'PARTIALLY VERIFIED'
        audit['verification_scope']='Recorded artifact integrity, prediction metrics and evidence bindings; not independent training attestation'
    except (VerificationError,KeyError,TypeError,ValueError,OSError) as exc:
        audit['errors'].append(str(exc)); audit['summary']={}; audit['status']=audit['validation']='INVALID'
    return audit

def csv_text(rows,columns=None):
    f=io.StringIO(newline=''); w=csv.DictWriter(f,fieldnames=columns or list(rows[0]))
    w.writeheader(); w.writerows(rows); return f.getvalue()

def export_results(audit,output_dir,allow_partial_export=False):
    """Existing output directories are ALWAYS refused; invalid inputs yield diagnostics only."""
    requested=Path(output_dir)
    require(not requested.exists() and not requested.is_symlink(), 'Refusing existing output directory or symlink')
    output=requested.resolve(); root=Path(audit['input']).resolve()
    require(not output.exists(),'Refusing existing output directory')
    require(not output.is_relative_to(root) and not root.is_relative_to(output),'Output must be isolated from input')
    require(output.parent.is_dir(),'Output parent must already exist')
    include=audit['validation']=='VERIFIED' and (audit['status']=='COMPLETE_VERIFIED' or allow_partial_export)
    report=['# MDB-Net verification','',f"Status: **{audit['status']}**",f"Metric validation: {audit['validation']}",'',
        'VERIFIED = directly checked; PARTIALLY VERIFIED = evidence gaps; NOT VERIFIED = no evidence; INVALID = rejected.','',
        'Input-derived metrics do not prove executed formal training. Synthetic tests are not real experiments.','','## Errors']
    report+=['- '+s for s in audit['errors']]+['','## Limitations']+['- '+s for s in audit['limitations']]
    report+=['','## Warnings']+['- '+s for s in sorted(set(audit['warnings']))]
    for r in audit['runs']:
        h=r['history']; report+=['',f"Seed {r['seed']}: head={h['head_epochs']}, fine-tune={h['finetune_epochs']}, inferred best epoch={h['inferred_best_epoch']}; {h['checkpoint_status']}"]
    report+=['','Segmentation: pseudo-mask agreement, NOT hand-annotated truth.',f'Metric tables exported: {include}. Partial exports are descriptive, NOT verified formal results.']
    provenance={k:v for k,v in audit.items() if k!='runs'}
    provenance['runs']=[dict(seed=r['seed'],history=r['history'],predictions_source=r['predictions_source']) for r in audit['runs']]
    payloads={'verification_report.md':'\n'.join(report)+'\n',
        'experiment_provenance.json':json.dumps(provenance,indent=2,ensure_ascii=False,allow_nan=False)}
    if include:
        individual,summary,cases=[],[],[]
        for r in audit['runs']:
            for sc,scenario in SCENARIOS.items():
                m=r['metrics'][sc]
                individual.append(dict(model='MDB-Net',seed=r['seed'],scenario=scenario,n=m['n'],accuracy=m['acc'],macro_f1=m['macro_f1'],verification_status=audit['status']))
                payloads[f"confusion_matrices/seed{r['seed']}_{sc}.csv"]=csv_text([{'true_class':c,**dict(zip(CLASSES,m['cm'][i]))} for i,c in enumerate(CLASSES)])
            for row in r['rows']:
                for scenario in (SCENARIOS['a_white' if row['background']=='white' else 'b_field'],'Mixed'):
                    cases.append(dict(seed=r['seed'],scenario=scenario,image=row['image'],true=CLASSES[row['true']],predicted=CLASSES[row['predicted']],correct=row['correct'],confidence=row['confidence'],predictions_source=r['predictions_source']))
        for sc,scenario in SCENARIOS.items():
            s=audit['summary'][sc]
            summary.append(dict(model='MDB-Net',scenario=scenario,number_of_runs=5,accuracy_mean=s['acc_mean'],accuracy_std=s['acc_std'],macro_f1_mean=s['macro_f1_mean'],macro_f1_std=s['macro_f1_std'],std_ddof=1,verification_status=audit['status']))
            payloads[f'confusion_matrices/sum_{sc}.csv']=csv_text([{'true_class':c,**dict(zip(CLASSES,s['cm_sum'][i]))} for i,c in enumerate(CLASSES)])
        payloads['individual_runs.csv']=csv_text(individual); payloads['summary_results.csv']=csv_text(summary)
        payloads['member6_prediction_cases.csv']=csv_text(cases)
        lines=[r'\begin{tabular}{lcc}',r'\hline',r'Scenario & Accuracy & Macro-F1 \\',r'\hline']
        for s in summary:
            lines.append(f"{s['scenario']} & {s['accuracy_mean']:.4f} $\\pm$ {s['accuracy_std']:.4f} & {s['macro_f1_mean']:.4f} $\\pm$ {s['macro_f1_std']:.4f} "+r'\\')
        payloads['result_table.tex']='% '+audit['status']+'; sample SD ddof=1; MDB-Net; partial=descriptive only\n'+'\n'.join(lines+[r'\hline',r'\end{tabular}'])+'\n'
    # Recheck recorded bytes before publication; never bless inputs changed after audit.
    for name, expected in audit['input_hashes'].items():
        require(Path(name).is_file() and file_hash(name)==expected, f'Input/evidence changed after audit: {name}')
    # Validate/serialize everything before creating any output. Never clean on failure.
    output.mkdir()
    for name,content in payloads.items():
        p=output/name; p.parent.mkdir(exist_ok=True)
        with p.open('x',encoding='utf-8',newline='') as f:
            f.write(content)
    return list(payloads)

def verification_main(argv=None):
    p=argparse.ArgumentParser(prog='Group_8.py verify', description='Independently recompute five-run MDB-Net metrics and export report tables. Historical execution evidence is optional; missing evidence never counts as complete verification.')
    p.add_argument('--input',required=True,type=Path,help='MDB-Net result directory, read only')
    p.add_argument('--output',required=True,type=Path,help='New isolated directory under Dhan-Shomadhan/_interim/; must not exist')
    p.add_argument('--expected-seeds',required=True,nargs=5,type=int,help='Five external approved seeds; no defaults')
    p.add_argument('--approved-config',type=Path,help='External approval JSON schema v1')
    p.add_argument('--evidence-manifest',type=Path,help='Original runner evidence JSON schema v1')
    p.add_argument('--allow-partial-export',action='store_true',help='Descriptive tables despite evidence gaps; NEVER upgrades status')
    p.add_argument('--require-complete',action='store_true',help='Exit 2 unless full evidence verification succeeds')
    a=p.parse_args(argv)
    try:
        audit=audit_experiment(a.input,a.expected_seeds,a.approved_config,a.evidence_manifest)
        export_results(audit,a.output,a.allow_partial_export)
    except (VerificationError,OSError,ValueError) as exc:
        print(f'MDB verification/export rejected: {exc}',file=sys.stderr); return 1
    print(f"MDB-Net: {audit['status']}; metric validation: {audit['validation']}; output: {a.output}")
    for msg in audit['errors']+audit['limitations']:
        print(msg)
    if audit['status']=='INVALID':
        return 1
    return 2 if a.require_complete and audit['status']!='COMPLETE_VERIFIED' else 0



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
    if len(sys.argv) > 1 and sys.argv[1] == "verify":
        raise SystemExit(verification_main(sys.argv[2:]))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("verify", help="independent metric verification and report-table export (verify --help)")
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
