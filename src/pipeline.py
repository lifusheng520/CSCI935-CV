"""CLI: python -m src.pipeline {download,run,predict}."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
ARTIFACTS = ROOT / "artifacts"
# Keep generated files separate from the read-only source dataset.
os.environ.setdefault("MPLCONFIGDIR", str(ARTIFACTS / "matplotlib"))

if __name__ == "__main__":
    print("[startup] Loading PyTorch / torchvision; downloads have not started yet...", flush=True)
    _startup_time = time.perf_counter()

import cv2
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

from .data import CLASSES, MEAN, STD, LeafDataset, discover, read_rgb, split_records, subset
from .masks import MaskGenerator, prepare_masks
from .model import MALG, objective

if __name__ == "__main__":
    print(f"[startup] Dependencies ready in {time.perf_counter() - _startup_time:.1f}s.", flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def device_for(name):
    if name != "auto":
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_batch(batch, device):
    return [batch[k].to(device) for k in ("image", "mask", "quality", "label")]


@torch.inference_mode()
def validate(model, loader, device):
    model.eval()
    loss, correct, count = 0., 0, 0
    for batch in loader:
        image, mask, quality, label = load_batch(batch, device)
        out = model(image, mask, quality)
        loss += float(F.nll_loss(out["log_prob"], label, reduction="sum"))
        correct += int((out["log_prob"].argmax(1) == label).sum())
        count += len(label)
    return loss / count, correct / count


def train_one(args, root, splits, masks, mask_index, seed, output, device):
    from .reporting import evaluate, training_curve
    seed_everything(seed)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "split.json", splits)
    config = dict(vars(args), classes=CLASSES, seed=seed, device_used=str(device),
                  data_root=str(root), mask_index=str(mask_index),
                  python=sys.version, packages={k: importlib.metadata.version(k) for k in
                  ("torch", "torchvision", "numpy", "opencv-python-headless", "scikit-learn", "matplotlib")})
    save_json(output / "config.json", config)
    loaders = {name: DataLoader(LeafDataset(root, rows, masks, args.size, name == "train"),
                               batch_size=args.batch_size, shuffle=name == "train", num_workers=0)
               for name, rows in splits.items()}
    model = MALG(pretrained=not args.no_pretrained, variant=args.variant, temperature=args.temperature,
                 quality_threshold=args.quality_threshold, mask_dropout=args.mask_dropout).to(device)
    model_config = dict(variant=args.variant, temperature=args.temperature,
                        quality_threshold=args.quality_threshold, mask_dropout=args.mask_dropout)
    history, best, stale, started = [], float("inf"), 0, time.perf_counter()
    epoch_number = 0
    for stage, epochs in (("heads", args.head_epochs), ("finetune", args.finetune_epochs)):
        if not epochs:
            continue
        if stage == "finetune" and (output / "best.pt").exists():
            model.load_state_dict(torch.load(output / "best.pt", map_location=device, weights_only=True)["model"])
        model.set_stage(stage == "finetune")
        backbone = [p for p in model.backbone.parameters() if p.requires_grad]
        heads = [p for name, p in model.named_parameters() if not name.startswith("backbone.") and p.requires_grad]
        groups = [{"params": heads, "lr": args.head_lr if stage == "heads" else args.finetune_lr}]
        if backbone:
            groups.append({"params": backbone, "lr": args.backbone_lr})
        optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
        stale = 0
        for _ in range(epochs):
            epoch_number += 1
            model.train()
            total, count = 0., 0
            for batch in loaders["train"]:
                image, mask, quality, label = load_batch(batch, device)
                optimizer.zero_grad(set_to_none=True)
                out = model(image, mask, quality)
                loss = objective(out, label, args.auxiliary,
                                 0. if args.variant == "baseline" else args.selection_penalty, args.density)
                if not torch.isfinite(loss):
                    raise RuntimeError("Non-finite training loss")
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(label)
                count += len(label)
            val_loss, val_acc = validate(model, loaders["val"], device)
            row = dict(epoch=epoch_number, stage=stage, train_loss=total/count, val_nll=val_loss,
                       val_accuracy=val_acc, elapsed_seconds=time.perf_counter()-started)
            history.append(row)
            print(f'seed={seed} {stage} epoch={epoch_number} train={total/count:.4f} '
                  f'val_nll={val_loss:.4f} val_acc={val_acc:.3f}', flush=True)
            save_json(output / "history.json", history)
            if val_loss < best - 1e-5:
                best, stale = val_loss, 0
                torch.save(dict(model=model.state_dict(), model_config=model_config, classes=CLASSES,
                                size=args.size, epoch=epoch_number, seed=seed, val_nll=best,
                                mask_config=dict(field_masks=args.field_masks, u2_arch=args.u2_arch,
                                                 u2_weights=args.u2_weights)), output / "best.pt")
            else:
                stale += 1
            if stale >= args.patience:
                print(f"Early stop {stage}: validation did not improve.", flush=True)
                break
    checkpoint = torch.load(output / "best.pt", map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    result = evaluate(model, loaders["test"], device, output)
    training_curve(history, output / "training.png")
    save_json(output / "run_summary.json", dict(best_epoch=checkpoint["epoch"],
              training_seconds=time.perf_counter()-started, parameters=sum(p.numel() for p in model.parameters()),
              scope="smoke_subset" if args.quick else "full_dataset", pretrained=not args.no_pretrained))
    print(json.dumps({k: {m: v[m] for m in ("count", "accuracy", "macro_f1")} for k, v in result.items()}, indent=2), flush=True)
    return result


def run(args):
    if args.head_epochs + args.finetune_epochs < 1:
        raise ValueError("At least one epoch is required")
    root = Path(args.data).resolve()
    torch.hub.set_dir(str(MODELS / "torch"))
    device = device_for(args.device)
    records = discover(root)
    print(f"Found {len(records)} images / {len(CLASSES)} classes; device={device}", flush=True)
    if args.no_pretrained:
        print("DEBUG ONLY: random backbone; results do not measure the proposed pretrained method.", flush=True)
    splits_by_seed = {}
    for seed in args.seeds:
        splits = split_records(records, seed)
        if args.quick:
            splits = {k: subset(v, 4 if k == "train" else 2, seed) for k, v in splits.items()}
        splits_by_seed[seed] = splits
    needed = {r["path"]: r for split in splits_by_seed.values() for rows in split.values() for r in rows}
    if args.u2_weights is None:
        args.u2_weights = str(MODELS / f"{args.u2_arch}.pth")
    generator = MaskGenerator(args.field_masks, args.u2_weights, args.u2_arch, str(device))
    if generator.net is None:
        print("Field masks DISABLED: Global + Local remain active; White masks use HSV.", flush=True)
        # Store the actual route, so inference cannot silently enable masks after training.
        args.field_masks = "none"
    else:
        args.field_masks = "u2net"
    masks, mask_index = prepare_masks(root, list(needed.values()), generator, args.size)
    del generator
    output = Path(args.output).resolve() if args.output else ARTIFACTS / "runs" / (time.strftime("%Y%m%d-%H%M%S") + f"-{args.variant}")
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / "mask_summary.json", {domain: dict(
        count=sum(r["domain"] == domain for r in needed.values()),
        accepted=sum(r["domain"] == domain and masks[r["path"]]["quality"] >= args.quality_threshold for r in needed.values()))
        for domain in ("White", "Field")})
    results = [train_one(args, root, splits_by_seed[seed], masks, mask_index, seed,
                         output / f"seed_{seed}", device) for seed in args.seeds]
    summary = {domain: {metric: dict(mean=float(np.mean([r[domain][metric] for r in results])),
                                    std=float(np.std([r[domain][metric] for r in results], ddof=1)) if len(results)>1 else None)
                        for metric in ("accuracy", "macro_f1")} for domain in ("Mixed", "White", "Field")}
    save_json(output / "summary.json", dict(runs=len(results), seeds=args.seeds, quick=args.quick,
              pretrained=not args.no_pretrained, field_masks=args.field_masks,
              std_definition="sample standard deviation (ddof=1); null for a single run", metrics=summary))
    print(f"Results saved to {output}", flush=True)


def download(args):
    root = MODELS
    root.mkdir(parents=True, exist_ok=True)
    torch.hub.set_dir(str(root / "torch"))
    print("[download 1/2] Checking/downloading MobileNetV2 ImageNet weights...", flush=True)
    mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V2)
    if args.u2net:
        import gdown
        file_id = "1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ" if args.u2_arch == "u2net" else "1rbSTGKAE-MTxBYHd-51l2hMOQPT_7EPy"
        path = root / f"{args.u2_arch}.pth"
        print(f"[download 2/2] Checking/downloading {args.u2_arch}: {path}", flush=True)
        if not path.exists():
            partial = path.with_suffix(".partial")
            if not gdown.download(id=file_id, output=str(partial), quiet=False):
                raise RuntimeError("Download failed; see src/README.md for the official manual download link")
            # Validate before publishing to the stable path.
            print("[download] Validating U2-Net weights...", flush=True)
            from .vendor.u2net import U2NET, U2NETP
            net = (U2NET if args.u2_arch == "u2net" else U2NETP)(3, 1)
            net.load_state_dict(torch.load(partial, map_location="cpu", weights_only=True))
            partial.replace(path)
    print(f"Weights ready: {root}")


def predict(args):
    from .reporting import draw_samples
    device = device_for(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MALG(pretrained=False, **ckpt["model_config"]).to(device).eval()
    model.load_state_dict(ckpt["model"])
    size, classes = ckpt["size"], ckpt["classes"]
    rgb = read_rgb(args.image, size)
    domain = args.background
    if domain == "auto":
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        border = np.concatenate((hsv[:10].reshape(-1,3), hsv[-10:].reshape(-1,3),
                                 hsv[:, :10].reshape(-1,3), hsv[:, -10:].reshape(-1,3)))
        domain = "White" if ((border[:,1] < 50) & (border[:,2] > 190)).mean() > 0.5 else "Field"
    cfg = ckpt["mask_config"]
    weights = args.u2_weights or cfg["u2_weights"]
    # Older checkpoints may contain the previous dataset-local absolute path.
    if not args.u2_weights and (not weights or not Path(weights).is_file()):
        weights = str(MODELS / f'{cfg["u2_arch"]}.pth')
    generator = MaskGenerator(cfg["field_masks"], weights, cfg["u2_arch"], str(device))
    mask, quality, details = generator(rgb, domain)
    x = torch.from_numpy(((rgb.astype(np.float32)/255-MEAN)/STD).transpose(2,0,1).copy())[None].to(device)
    with torch.inference_mode():
        out = model(x, torch.from_numpy(mask.astype(np.float32))[None,None].to(device), torch.tensor([quality], device=device))
    probabilities = out["log_prob"][0].exp().cpu().tolist()
    predicted = int(np.argmax(probabilities))
    row = dict(image=str(Path(args.image).resolve()), domain=domain, prediction_name=classes[predicted],
               confidence=probabilities[predicted], probabilities=dict(zip(classes, probabilities)),
               quality=quality, mask_valid=bool(out["valid"][0]), mask_details=details, target_name="unknown",
               **{f"weight_{b}": float(out["weights"][0,j]) for j,b in enumerate(("global","local","mask"))})
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=False)
    save_json(destination / "prediction.json", row)
    draw_samples([((domain, -1), rgb/255., mask, out["selection"][0,0].cpu().numpy(), row)], destination / "prediction.png")
    print(json.dumps(row, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="Download official pretrained weights")
    dl.add_argument("--data", default=str(ROOT / "Dhan-Shomadhan"))
    dl.add_argument("--u2net", action="store_true")
    dl.add_argument("--u2-arch", choices=("u2net", "u2netp"), default="u2net")
    dl.set_defaults(func=download)
    p = sub.add_parser("run", help="Split -> masks -> two-stage training -> test -> visualizations")
    p.add_argument("--data", default=str(ROOT / "Dhan-Shomadhan"))
    p.add_argument("--output", help="New run directory; never overwrites an existing run")
    p.add_argument("--seeds", type=int, nargs="+", default=[42])
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--variant", choices=("malg", "baseline", "global_local"), default="malg")
    p.add_argument("--quick", action="store_true", help="80-image smoke subset; not report-quality evaluation")
    p.add_argument("--no-pretrained", action="store_true", help="Offline debugging only; random backbone")
    p.add_argument("--field-masks", choices=("auto", "u2net", "none"), default="auto")
    p.add_argument("--u2-weights")
    p.add_argument("--u2-arch", choices=("u2net", "u2netp"), default="u2net")
    p.add_argument("--size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--head-epochs", type=int, default=5)
    p.add_argument("--finetune-epochs", type=int, default=15)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--head-lr", type=float, default=1e-3)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--backbone-lr", type=float, default=1e-5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--temperature", type=float, default=0.5)
    p.add_argument("--quality-threshold", type=float, default=0.45)
    p.add_argument("--mask-dropout", type=float, default=0.2)
    p.add_argument("--auxiliary", type=float, default=0.2)
    p.add_argument("--selection-penalty", type=float, default=0.01)
    p.add_argument("--density", type=float, default=0.25)
    p.set_defaults(func=run)
    pred = sub.add_parser("predict", help="Predict a single image using best.pt")
    pred.add_argument("--checkpoint", required=True)
    pred.add_argument("--image", required=True)
    pred.add_argument("--background", choices=("auto", "White", "Field"), default="auto")
    pred.add_argument("--u2-weights")
    pred.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    pred.add_argument("--output", required=True)
    pred.set_defaults(func=predict)
    args = parser.parse_args()
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    cv2.setNumThreads(0)
    if args.command == "run":
        if args.temperature <= 0 or args.size < 32 or args.batch_size < 1 or args.patience < 1:
            parser.error("temperature > 0, size >= 32, batch-size/patience >= 1 required")
        if min(args.head_epochs, args.finetune_epochs) < 0 or len(set(args.seeds)) != len(args.seeds):
            parser.error("epochs must be nonnegative and seeds distinct")
        for key in ("mask_dropout", "quality_threshold", "density"):
            if not 0 <= getattr(args, key) <= 1:
                parser.error(f"{key} must be in [0, 1]")
    args_dict = vars(args)
    func = args_dict.pop("func")
    func(args)


if __name__ == "__main__":
    main()
