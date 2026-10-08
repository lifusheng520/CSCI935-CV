"""Create a stratified train/validation/test split from the manifest."""

import csv
from collections import Counter
from pathlib import Path

import cv2
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# config
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "Dhan-Shomadhan"
MANIFEST_PATH = DATASET_ROOT / "generated" / "manifest.csv"
SPLIT_DIRECTORY = DATASET_ROOT / "generated"
RANDOM_SEEDS = [42, 101, 2026, 7, 99]
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MANIFEST_COLUMNS = {
    "path",
    "label",
    "class_name",
    "background",
}

DATASET_COLUMNS = {
    "path",
    "label",
    "split",
}


class RiceLeafDataset(Dataset):
    """Read rice-leaf images and labels from one dataset split."""

    def __init__(self, csv_path: Path, dataset_root: Path, split: str, transform=None) -> None:

        self.dataset_root = dataset_root
        self.transform = transform
        self.records = []

        with csv_path.open(
            "r",
            newline="",
            encoding="utf-8",
        ) as csv_file:
            reader = csv.DictReader(csv_file)
            columns = set(reader.fieldnames or [])

            missing_columns = DATASET_COLUMNS - columns
            if missing_columns:
                raise ValueError(
                    f"Missing columns in {csv_path}: "
                    f"{sorted(missing_columns)}"
                )

            for row in reader:
                if row["split"] == split:
                    self.records.append({
                        "path": row["path"],
                        "label": int(row["label"]),
                    })

        if not self.records:
            raise ValueError(
                f"No records for split '{split}' in {csv_path}"
            )


    def __len__(self) -> int:
        """Return the number of images in this split."""
        return len(self.records)

    def __getitem__(self, index: int):
        """Read one RGB image and return it with its class label."""
        record = self.records[index]
        image_path = self.dataset_root / record["path"]

        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # OpenCV loads colour images as BGR; the model expects RGB.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform is not None:
            image = self.transform(image)

        return image, record["label"]


def build_image_transform():
    """Build the basic preprocessing used by MobileNetV2."""

    return transforms.Compose([
        transforms.ToPILImage(), # Convert the NumPy image to torchvision
        transforms.Resize(IMAGE_SIZE), # resize
        transforms.ToTensor(), # tensor conversion
        transforms.Normalize( #normalization
            mean=IMAGENET_MEAN,
            std=IMAGENET_STD,
        ),
    ])


def build_dataloader(dataset: RiceLeafDataset, batch_size: int = 32, shuffle: bool = False) -> DataLoader:
    """Build a DataLoader from a RiceLeafDataset."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
    )


def load_manifest(manifest_path: Path) -> list[dict]:
    """Read the manifest and return the fields required for splitting."""
    with manifest_path.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        reader = csv.DictReader(csv_file)
        columns = set(reader.fieldnames or [])

        missing_columns = MANIFEST_COLUMNS - columns
        if missing_columns:
            raise ValueError(
                f"Missing columns in {manifest_path}: "
                f"{sorted(missing_columns)}"
            )

        records = []
        for row in reader:
            records.append({
                "path": row["path"],
                "label": int(row["label"]),
                "class_name": row["class_name"],
                "background": row["background"],
            })

    if not records:
        raise ValueError(f"No records found in {manifest_path}")

    return records


def split_records( records: list[dict], random_seed: int = 42) -> list[dict]:
    """Split records into 70% train, 15% validation and 15% test. Default random_seed is 42"""
    indices = list(range(len(records)))

    # combine background and disease so all ten combinations are stratified
    strata = [
        f"{record['background']}__{record['class_name']}"
        for record in records
    ]

    #  70% for training.
    train_indices, remaining_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=random_seed,
        stratify=strata,
    )

    remaining_strata = [
        strata[index]
        for index in remaining_indices
    ]

    # divide the remaining 30% equally: 15% validation and 15% test.
    validation_indices, test_indices = train_test_split(
        remaining_indices,
        test_size=0.50,
        random_state=random_seed,
        stratify=remaining_strata,
    )

    split_by_index = {}
    for index in train_indices:
        split_by_index[index] = "train"

    for index in validation_indices:
        split_by_index[index] = "validation"

    for index in test_indices:
        split_by_index[index] = "test"

    split_records_list = []
    # add split column into the records list, recording train/validation/test dataset
    for index, record in enumerate(records):
        split_record = record.copy()
        split_record["split"] = split_by_index[index]
        split_records_list.append(split_record)

    return split_records_list


def save_split(records: list[dict], output_path: Path) -> None:
    """Save all records and their assigned split to one CSV file."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "path",
        "label",
        "class_name",
        "background",
        "split",
    ]
    # save
    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(records)


def print_split_summary( records: list[dict], random_seed: int) -> None:
    """Print the number of samples assigned to each split."""
    counts = Counter(record["split"] for record in records)

    print(f"Random seed: {random_seed}")
    print(f"Total images: {len(records)}")
    print(f"Training images: {counts['train']}")
    print(f"Validation images: {counts['validation']}")
    print(f"Testing images: {counts['test']}")


def main() -> None:
    records = load_manifest(MANIFEST_PATH)

    for random_seed in RANDOM_SEEDS:
        # split
        split_data = split_records(
            records,
            random_seed=random_seed,
        )

        split_path = (SPLIT_DIRECTORY / f"split_seed_{random_seed}.csv")

        save_split(split_data, split_path)
        print_split_summary(split_data, random_seed)
        print(f"Split saved to: {split_path}\n")

        # transform
        image_transform = build_image_transform()

        #prepare dataset
        train_dataset = RiceLeafDataset(
            csv_path=split_path,
            dataset_root=DATASET_ROOT,
            split="train",
            transform=image_transform,
        )
        validation_dataset = RiceLeafDataset(
            csv_path=split_path,
            dataset_root=DATASET_ROOT,
            split="validation",
            transform=image_transform,
        )
        test_dataset = RiceLeafDataset(
            csv_path=split_path,
            dataset_root=DATASET_ROOT,
            split="test",
            transform=image_transform,
        )

        # prepare data loader
        train_loader = build_dataloader(
            dataset=train_dataset,
            batch_size=32,
            shuffle=True,
        )
        validation_loader = build_dataloader(
            dataset=validation_dataset,
            batch_size=32,
            shuffle=False,
        )
        test_loader = build_dataloader(
            dataset=test_dataset,
            batch_size=32,
            shuffle=False,
        )

        print(
            "DataLoader batches: "
            f"train={len(train_loader)}, "
            f"validation={len(validation_loader)}, "
            f"test={len(test_loader)}\n"
        )


if __name__ == "__main__":
    main()
