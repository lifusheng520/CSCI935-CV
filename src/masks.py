"""Offline leaf masks. Reliability is a heuristic, not segmentation accuracy."""
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .data import read_rgb, MEAN, STD


def refine(binary):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(binary.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
    # Preserve all substantial components (multiple leaves), not just the largest.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(clean, 8)
    keep = np.zeros_like(clean)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= max(8, clean.size * 0.001):
            keep[labels == i] = 1
    contours, _ = cv2.findContours(keep, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(keep, contours, -1, 1, thickness=cv2.FILLED)
    return keep


def mask_quality(mask, score):
    area = float(mask.mean())
    _, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    connectivity = float(stats[1:, cv2.CC_STAT_AREA].max() / max(mask.sum(), 1)) if len(stats) > 1 else 0.
    area_score = float(np.clip(min(area / 0.05, (1 - area) / 0.10), 0, 1))
    confidence = float(np.abs(score - 0.5).mean() * 2)
    quality = area_score * (0.5 + 0.5 * connectivity) * confidence
    return quality, dict(area=area, connectivity=connectivity, confidence=confidence)


class MaskGenerator:
    def __init__(self, mode="auto", weights=None, architecture="u2net", device="cpu"):
        self.device, self.net = device, None
        self.architecture = architecture
        self.weights = Path(weights) if weights else None
        if mode == "u2net" and (not self.weights or not self.weights.is_file()):
            raise FileNotFoundError("U2-Net weights missing. Run: python -m src.pipeline download --u2net")
        if mode != "none" and self.weights and self.weights.is_file():
            from .vendor.u2net import U2NET, U2NETP
            self.net = (U2NET if architecture == "u2net" else U2NETP)(3, 1)
            self.net.load_state_dict(torch.load(self.weights, map_location="cpu", weights_only=True))
            self.net.to(device).eval()
        self.signature = dict(version=2, field=architecture if self.net else "disabled",
                              weights_sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest() if self.net else None)

    @torch.inference_mode()
    def __call__(self, rgb, domain):
        if domain == "White":
            hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV).astype(np.float32)
            # Paper is often grey under uneven lighting. Estimate its value from
            # low-saturation border pixels instead of requiring absolute near-255.
            width = max(2, min(rgb.shape[:2]) // 20)
            border = np.concatenate((hsv[:width].reshape(-1, 3), hsv[-width:].reshape(-1, 3),
                                     hsv[:, :width].reshape(-1, 3), hsv[:, -width:].reshape(-1, 3)))
            neutral = border[border[:, 1] < 70]
            background_value = float(np.median(neutral[:, 2])) if len(neutral) else 220.
            background_saturation = float(np.median(neutral[:, 1])) if len(neutral) else 15.
            sat_limit = np.clip(background_saturation + 25, 40, 75)
            white = np.minimum(np.clip((hsv[..., 2] - max(30, background_value - 90)) / 40, 0, 1),
                               np.clip((sat_limit + 15 - hsv[..., 1]) / 30, 0, 1))
            score, route = 1 - white, "hsv"
        elif self.net is not None:
            array = cv2.resize(rgb, (320, 320)).astype(np.float32)
            # Match upstream ToTensorLab(flag=0): divide by image maximum, then ImageNet normalization.
            array = (array / max(float(array.max()), 1.) - MEAN) / STD
            x = torch.from_numpy(array.transpose(2, 0, 1).copy())[None].to(self.device)
            raw = self.net(x)[0][0, 0].cpu().numpy()
            span = float(raw.max() - raw.min())
            if span < 1e-6:
                return np.zeros(rgb.shape[:2], np.uint8), 0., {"route": "u2net_constant", "area": 0.}
            score = cv2.resize((raw - raw.min()) / span, (rgb.shape[1], rgb.shape[0]))
            route = self.architecture
        else:
            return np.zeros(rgb.shape[:2], np.uint8), 0., {"route": "field_disabled", "area": 0.}
        mask = refine(score >= 0.5)
        quality, details = mask_quality(mask, score)
        return mask, quality, dict(route=route, **details)


def prepare_masks(root, records, generator, size=224):
    signature = dict(generator.signature, size=size)
    token = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:12]
    cache = Path(__file__).resolve().parents[1] / "artifacts" / "masks" / token
    cache.mkdir(parents=True, exist_ok=True)
    index_file = cache / "index.json"
    index = json.loads(index_file.read_text()) if index_file.exists() else {}
    for i, record in enumerate(records):
        key = record["path"]
        old = index.get(key)
        if old and old.get("sha256") == record["sha256"] and Path(old["mask"]).is_file():
            continue
        rgb = read_rgb(root / key, size)
        mask, quality, details = generator(rgb, record["domain"])
        destination = cache / (record["sha256"] + ".png")
        if not cv2.imwrite(str(destination), mask * 255):
            raise IOError(f"Cannot write mask {destination}")
        index[key] = dict(mask=str(destination.resolve()), quality=quality,
                          sha256=record["sha256"], **details)
        if (i + 1) % 50 == 0:
            print(f"Masks {i + 1}/{len(records)}", flush=True)
            index_file.write_text(json.dumps(index, indent=2))
    index_file.write_text(json.dumps(index, indent=2))
    (cache / "config.json").write_text(json.dumps(signature, indent=2))
    return index, index_file
