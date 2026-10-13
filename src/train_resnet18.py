#!/usr/bin/env python3
"""ResNet-18 classifier for the Dhan-Shomadhan rice leaf dataset.
Default: fine-tune ImageNet-pretrained weights. With --from_scratch: random initialisation.

Five classes (Brown Spot, Leaf Scald, Rice Blast, Rice Tungro, Sheath Blight), two
background types (white, field). One run = one random stratified split. The script runs
several seeds and reports mean +/- std for three test scenarios:
    (a) white-background test images
    (b) field-background test images
    (c) all test images (white or field)

Usage (from the folder that contains Dhan-Shomadhan/):
    python train_resnet18.py                              # joint model, 5 seeds
    python train_resnet18.py --train_domain white         # model trained on white only
    python train_resnet18.py --train_domain field         # model trained on field only
    python train_resnet18.py --from_scratch               # baseline: no pretraining (own defaults)
    python train_resnet18.py --prepare_only               # just build the resized cache

Extra package beyond the permitted list: torchvision (ResNet-18 architecture + ImageNet
weights). Weights: https://download.pytorch.org/models/resnet18-f37072fd.pth
Pass a local copy with --weights_path if the machine cannot download them.
"""
from __future__ import annotations

import argparse
import copy
import difflib
import json
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models
from torchvision import transforms as T

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


class LeafDataset(Dataset):
    def __init__(self, cache: Path, rels, labels, domains, transform):
        self.files = [cache / r.with_suffix(".jpg") for r in rels]
        self.labels = np.asarray(labels)
        self.domains = np.asarray(domains)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = cv2.cvtColor(cv2.imread(str(self.files[i]), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        x = torch.from_numpy(img).permute(2, 0, 1)  # uint8 CxHxW
        return self.transform(x), int(self.labels[i])


def make_transforms(size: int):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    train = T.Compose([
        T.ConvertImageDtype(torch.float32),
        T.RandomResizedCrop(size, scale=(0.4, 1.0), ratio=(0.6, 1.6), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(),
        T.RandomRotation(20),
        # mild: colour is diagnostic for Tungro / Leaf Scald, so keep hue jitter tiny
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.02),
        norm,
    ])
    # Leaves are long and thin; images are squashed to a square here (documented choice).
    evalt = T.Compose([T.ConvertImageDtype(torch.float32), T.Resize((size, size), antialias=True), norm])
    return train, evalt


# ----------------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------------
def make_split(labels, domains, seed, val_frac=0.15, test_frac=0.15):
    """Stratified by (class, background) so every split has the same mix as the full data."""
    strata = np.array([f"{l}_{d}" for l, d in zip(labels, domains)])
    idx = np.arange(len(labels))
    trainval, test = train_test_split(idx, test_size=test_frac, stratify=strata, random_state=seed)
    train, val = train_test_split(trainval, test_size=val_frac / (1 - test_frac),
                                  stratify=strata[trainval], random_state=seed)
    return np.sort(train), np.sort(val), np.sort(test)


# ----------------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------------
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


def set_backbone_trainable(model, flag: bool):
    for name, p in model.named_parameters():
        if not name.startswith("fc."):
            p.requires_grad = flag


def freeze_bn_stats(model):
    """While the backbone is frozen keep BatchNorm running stats fixed as well."""
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()


# ----------------------------------------------------------------------------------
# Train / eval helpers
# ----------------------------------------------------------------------------------
def run_epoch(model, loader, criterion, device, optimizer=None, scheduler=None, scaler=None,
              freeze_bn=False):
    train = optimizer is not None
    model.train(train)
    if train and freeze_bn:
        freeze_bn_stats(model)
    tot_loss, correct, n = 0.0, 0, 0
    use_amp = device.type == "cuda"
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
            tot_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
    return tot_loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            out = model(x.to(device, non_blocking=True))
        logits.append(out.float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


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


# ----------------------------------------------------------------------------------
# One full run (one seed)
# ----------------------------------------------------------------------------------
def run_one(seed, args, rels, labels, domains, cache, device, out_dir):
    seed_everything(seed)
    tr_idx, va_idx, te_idx = make_split(labels, domains, seed)
    if args.train_domain != "both":  # single-background model: train/val from that background only
        d = DOMAINS.index(args.train_domain)
        tr_idx, va_idx = tr_idx[domains[tr_idx] == d], va_idx[domains[va_idx] == d]
    print(f"[seed {seed}] train {len(tr_idx)}  val {len(va_idx)}  test {len(te_idx)}  (train_domain={args.train_domain})")

    train_tf, eval_tf = make_transforms(args.img_size)
    mk = lambda idx, tf: LeafDataset(cache, [rels[i] for i in idx], labels[idx], domains[idx], tf)
    gen = torch.Generator().manual_seed(seed)
    kw = dict(batch_size=args.batch_size, num_workers=args.workers, pin_memory=device.type == "cuda",
              persistent_workers=args.workers > 0)
    train_loader = DataLoader(mk(tr_idx, train_tf), shuffle=True, drop_last=len(tr_idx) > args.batch_size,
                              generator=gen, **kw)
    val_loader = DataLoader(mk(va_idx, eval_tf), shuffle=False, **kw)
    test_ds = mk(te_idx, eval_tf)
    test_loader = DataLoader(test_ds, shuffle=False, **kw)

    model = build_model(len(CLASS_NAMES), args.weights_path, args.dropout, not args.from_scratch).to(device)
    weight = None
    if args.class_weights:
        cnt = np.bincount(labels[tr_idx], minlength=len(CLASS_NAMES)).astype(np.float32)
        w = cnt.sum() / (len(cnt) * np.maximum(cnt, 1))
        weight = torch.tensor(w, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=args.label_smoothing)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []

    # ---- stage 1: train only the new classifier head -------------------------------------
    if args.head_epochs > 0:
        set_backbone_trainable(model, False)
        opt = torch.optim.AdamW(model.fc.parameters(), lr=args.lr_head, weight_decay=args.weight_decay)
        for ep in range(args.head_epochs):
            tl, ta = run_epoch(model, train_loader, criterion, device, opt, None, scaler, freeze_bn=True)
            vl, va = run_epoch(model, val_loader, criterion, device)
            history.append({"stage": 1, "epoch": ep + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl, "val_acc": va})
            print(f"  [head {ep + 1:>2}/{args.head_epochs}] train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}")

    # ---- stage 2: fine-tune everything (lower LR for the pretrained backbone) ---------------
    set_backbone_trainable(model, True)
    bb = [p for n, p in model.named_parameters() if not n.startswith("fc.")]
    opt = torch.optim.AdamW([{"params": bb, "lr": args.lr_backbone},
                             {"params": model.fc.parameters(), "lr": args.lr_head}],
                            weight_decay=args.weight_decay)
    total = args.epochs * len(train_loader)
    sched = cosine_with_warmup(opt, min(len(train_loader), total // 10 or 1), total)
    best_f1, best_state, bad = -1.0, None, 0
    for ep in range(args.epochs):
        t0 = time.time()
        tl, ta = run_epoch(model, train_loader, criterion, device, opt, sched, scaler)
        vl, va = run_epoch(model, val_loader, criterion, device)
        vlog, vy = predict(model, val_loader, device)
        vf1 = f1_score(vy, vlog.argmax(1), labels=list(range(len(CLASS_NAMES))), average="macro", zero_division=0)
        history.append({"stage": 2, "epoch": ep + 1, "train_loss": tl, "train_acc": ta, "val_loss": vl,
                        "val_acc": va, "val_macro_f1": float(vf1)})
        print(f"  [ft {ep + 1:>2}/{args.epochs}] train {tl:.3f}/{ta:.3f}  val {vl:.3f}/{va:.3f}  "
              f"macroF1 {vf1:.3f}  ({time.time() - t0:.0f}s)")
        if vf1 > best_f1:  # model selection uses the validation set only, never the test set
            best_f1, bad = vf1, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= args.patience:
                print(f"  early stop (no val macro-F1 gain for {args.patience} epochs)")
                break
    model.load_state_dict(best_state)

    # ---- test: scenarios (a) white, (b) field, (c) all --------------------------------------
    logits, y = predict(model, test_loader, device)
    pred, dom = logits.argmax(1), test_ds.domains
    res = {"seed": seed, "best_val_macro_f1": float(best_f1), "n_train": int(len(tr_idx)),
           "history": history, "scenarios": {}}
    for name, mask in [("a_white", dom == 0), ("b_field", dom == 1), ("c_all", np.ones_like(dom, bool))]:
        if mask.any():
            res["scenarios"][name] = score(y[mask], pred[mask], len(CLASS_NAMES))
    for name, s in res["scenarios"].items():
        print(f"  test {name:<8} n={s['n']:<4} acc {s['acc']:.4f}  macroF1 {s['macro_f1']:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"seed{seed}_split.json").write_text(json.dumps(
        {"train": [str(rels[i]) for i in tr_idx], "val": [str(rels[i]) for i in va_idx],
         "test": [str(rels[i]) for i in te_idx]}, indent=1))
    (out_dir / f"seed{seed}_result.json").write_text(json.dumps(res, indent=1))
    if args.save_models:
        torch.save(best_state, out_dir / f"seed{seed}_model.pt")
    return res


# ----------------------------------------------------------------------------------
# Aggregation over seeds
# ----------------------------------------------------------------------------------
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


# ----------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data_root", type=Path, default=Path("Dhan-Shomadhan"))
    p.add_argument("--train_domain", choices=["both", "white", "field"], default="both",
                   help="which background(s) to train on; test scenarios a/b/c are always reported")
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--cache_size", type=int, default=448, help="longest side of the cached resized images")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--from_scratch", action="store_true",
                   help="random init instead of ImageNet weights (baseline for the ablation)")
    # None = pick a default that suits the mode (see apply_mode_defaults)
    p.add_argument("--head_epochs", type=int, default=None, help="stage 1: head only (pretrained 3, scratch 0)")
    p.add_argument("--epochs", type=int, default=None, help="stage 2 max epochs (pretrained 25, scratch 80)")
    p.add_argument("--lr_head", type=float, default=1e-3)
    p.add_argument("--lr_backbone", type=float, default=None, help="pretrained 2e-4, scratch 1e-3")
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--label_smoothing", type=float, default=0.1)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--patience", type=int, default=None, help="early-stop patience (pretrained 8, scratch 20)")
    p.add_argument("--class_weights", action="store_true", help="inverse-frequency loss weights")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--weights_path", type=str, default=None, help="local resnet18-f37072fd.pth")
    p.add_argument("--save_models", action="store_true")
    p.add_argument("--out_name", type=str, default=None)
    p.add_argument("--prepare_only", action="store_true", help="only build the resized-image cache")
    p.add_argument("--allow_missing", action="store_true",
                   help="do not stop when a class/background combination has no images")
    return p.parse_args()


def apply_mode_defaults(args):
    """Scratch training needs a bigger backbone LR, many more epochs and no head-only stage,
    because nothing in the network is pretrained. Explicit command-line values always win."""
    d = ({"head_epochs": 0, "epochs": 80, "lr_backbone": 1e-3, "patience": 20} if args.from_scratch
         else {"head_epochs": 3, "epochs": 25, "lr_backbone": 2e-4, "patience": 8})
    for k, v in d.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.from_scratch and args.weights_path:
        print("[warn] --weights_path is ignored with --from_scratch")
    return args


def main():
    args = apply_mode_defaults(parse_args())
    root = args.data_root
    if not root.is_dir():
        raise SystemExit(f"Dataset folder not found: {root.resolve()}")
    samples = discover_samples(root, args.allow_missing)
    cache = build_cache(root, samples, args.cache_size)
    if args.prepare_only:
        return
    rels = [s[0] for s in samples]
    labels = np.array([s[1] for s in samples])
    domains = np.array([s[2] for s in samples])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    tag = args.out_name or f"resnet18_{args.train_domain}" + ("_scratch" if args.from_scratch else "")
    out_dir = root / "_interim" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "args.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    results = [run_one(s, args, rels, labels, domains, cache, device, out_dir) for s in args.seeds]
    aggregate(results, out_dir, tag)
    print(f"\nresults written to {out_dir}")


if __name__ == "__main__":  # guard is required for DataLoader workers on Windows
    main()