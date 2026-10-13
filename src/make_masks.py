#!/usr/bin/env python3
"""Batch leaf-mask generation for the WHITE-background images, with a QC pass.

For every white-background image it
  1. segments the leaf (classical Lab-distance + Otsu + GrabCut from seg_leaf.py),
  2. saves a binary mask (0/255) aligned pixel-for-pixel with the cached training image,
  3. scores each mask (size, shape, width profile, boundary contrast),
  4. flags suspicious masks (extreme values vs. the other images of the same class),
  5. writes contact sheets so a human can check the flagged masks and a random sample of
     the passing ones in a couple of minutes.

Outputs (all under <data_root>/_interim/):
    masks_<size>/<same relative path>.png     binary masks, same size as cache_<size>/ images
    qc/qc_report.csv                          per-image metrics, flags, status
    qc/flagged_XX.jpg                         contact sheets of flagged masks
    qc/sample_pass_XX.jpg                     random sample of masks that passed
    qc/mask_index.csv                         image / mask / class / status, for later training code

Usage (from the folder that contains Dhan-Shomadhan/):
    python make_masks.py                      # all white images
    python make_masks.py --only "Sheath"      # re-run just paths containing this text
    python make_masks.py --only bad_leaf --l_weight 0.1 --overwrite
    python make_masks.py --exclude_file Dhan-Shomadhan/_interim/qc/exclude.txt   # after review

Needs seg_leaf.py and train_resnet18.py in the same folder (merge into one file for submission).
"""
from __future__ import annotations

import argparse
import csv
import random
import sys
from concurrent.futures import ThreadPoolExecutor  # threads, not processes: forking after importing torch can deadlock
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from seg_leaf import segment_leaf  # noqa: E402
from train_resnet18 import CLASS_NAMES, build_cache, discover_samples  # noqa: E402

FONT = cv2.FONT_HERSHEY_SIMPLEX
WORK = 800  # the segmenter works at this longest side (same as in the tests)


# ----------------------------------------------------------------------------------
# QC metrics
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# Flagging
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
# Contact sheets
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_root", type=Path, default=Path("Dhan-Shomadhan"))
    ap.add_argument("--cache_size", type=int, default=448, help="must match train_resnet18.py")
    ap.add_argument("--l_weight", type=float, default=0.0, help="lightness weight in the segmenter (0 = ignore shadows)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", type=str, default=None, help="only process paths containing this text")
    ap.add_argument("--overwrite", action="store_true", help="re-segment even if a mask already exists")
    ap.add_argument("--z", type=float, default=3.5, help="outlier threshold (robust z-score)")
    ap.add_argument("--n_pass_sample", type=int, default=32, help="passing masks to include in the review sheets")
    ap.add_argument("--exclude_file", type=Path, default=None, help="text file, one relative image path per line")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow_missing", action="store_true")
    a = ap.parse_args()

    root = a.data_root
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
    print("Next: open the flagged_*.jpg sheets, note bad ones, put their relative paths in "
          f"{qc_dir / 'exclude.txt'} (or re-run them with --only ... --l_weight ... --overwrite).")


if __name__ == "__main__":
    main()