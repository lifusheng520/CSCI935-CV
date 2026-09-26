"""Dataset discovery, reproducible stratification, and joint image/mask transforms."""
import hashlib
import random
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

CLASSES = ["Brown Spot", "Leaf Scald", "Rice Blast", "Rice Tungro", "Sheath Blight"]
ALIASES = {"Browon Spot": "Brown Spot", "Leaf Scaled": "Leaf Scald",
           "Rice Turgro": "Rice Tungro", "Shath Blight": "Sheath Blight"}
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def read_rgb(path, size):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Cannot decode image: {path}")
    return cv2.cvtColor(cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)


def discover(root):
    records, seen = [], {}
    for domain in ("White", "Field"):
        folder = root / f"{domain} Background"
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            name = ALIASES.get(path.parent.name, path.parent.name)
            if name not in CLASSES:
                raise ValueError(f"Unknown class directory: {path.parent}")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            # Fail clearly rather than allow identical files into different partitions.
            if digest in seen:
                raise ValueError(f"Duplicate image: {path} and {seen[digest]}; group/deduplicate before splitting")
            seen[digest] = path
            records.append(dict(path=path.relative_to(root).as_posix(), label=CLASSES.index(name),
                                domain=domain, sha256=digest))
    if not records:
        raise ValueError(f"No images found in {root}")
    return records


def split_records(records, seed):
    strata = lambda rows: [f'{r["domain"]}_{r["label"]}' for r in rows]
    train, rest = train_test_split(records, test_size=0.30, random_state=seed, stratify=strata(records))
    val, test = train_test_split(rest, test_size=0.50, random_state=seed, stratify=strata(rest))
    return dict(train=train, val=val, test=test)


def subset(rows, per_stratum, seed):
    if not per_stratum:
        return rows
    rng = random.Random(seed)
    chosen = []
    for domain in ("White", "Field"):
        for label in range(5):
            group = [r for r in rows if r["domain"] == domain and r["label"] == label]
            chosen.extend(rng.sample(group, min(per_stratum, len(group))))
    return chosen


class LeafDataset(Dataset):
    def __init__(self, root, records, masks, size=224, augment=False):
        self.root, self.records, self.masks = root, records, masks
        self.size, self.augment = size, augment

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = read_rgb(self.root / record["path"], self.size)
        item = self.masks[record["path"]]
        mask = cv2.imread(str(Path(item["mask"])), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(Path(item["mask"]))
        mask = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        if self.augment:
            if random.random() < 0.5:
                image, mask = image[:, ::-1], mask[:, ::-1]
            matrix = cv2.getRotationMatrix2D((self.size / 2, self.size / 2), random.uniform(-15, 15), 1)
            image = cv2.warpAffine(image, matrix, (self.size, self.size), borderValue=(128, 128, 128))
            mask = cv2.warpAffine(mask, matrix, (self.size, self.size), flags=cv2.INTER_NEAREST)
            contrast, brightness = random.uniform(0.9, 1.1), random.uniform(-12, 12)
            image = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255)
        tensor = torch.from_numpy(((image.astype(np.float32) / 255 - MEAN) / STD).transpose(2, 0, 1).copy())
        return dict(image=tensor, mask=torch.from_numpy((mask > 127).astype(np.float32)[None]),
                    quality=torch.tensor(item["quality"], dtype=torch.float32), label=record["label"],
                    domain=record["domain"], path=record["path"])
