"""Build a standard CSV index for the Dhan-Shomadhan image dataset.

The script reads the original background/disease folder structure, normalises
the known spelling variations in folder names, and writes one manifest record
per image.
"""

import csv
from collections import Counter
from pathlib import Path

# configure
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dhan-Shomadhan"

CLASS_NAMES = [
    "Brown Spot",
    "Leaf Scald",
    "Rice Blast",
    "Rice Tungro",
    "Sheath Blight",
]

# convert class name to index
CLASS_TO_INDEX = {
    class_name: index
    for index, class_name in enumerate(CLASS_NAMES)
}

# standardise misspelled class names found in the original dataset folders
CLASS_ALIASES = {
    "Brown Spot": "Brown Spot",
    "Browon Spot": "Brown Spot",

    "Leaf Scald": "Leaf Scald",
    "Leaf Scaled": "Leaf Scald",

    "Rice Blast": "Rice Blast",

    "Rice Tungro": "Rice Tungro",
    "Rice Turgro": "Rice Tungro",

    "Sheath Blight": "Sheath Blight",
    "Shath Blight": "Sheath Blight",
}

BACKGROUND_FOLDERS = {
    "White Background": "White",
    "Field Background": "Field",
}


def build_manifest(dataset_root: Path) -> list[dict]:
    """Scan the dataset and return one standardised record per image."""
    records = []

    # the dataset is organised first by background and then by disease
    for folder_name, background in BACKGROUND_FOLDERS.items():
        background_path = dataset_root / folder_name

        if not background_path.is_dir():
            raise FileNotFoundError(
                f"Folder not found: {background_path}"
            )

        for class_path in sorted(background_path.iterdir()):
            if not class_path.is_dir():
                continue
            # get raw class name
            raw_class_name = class_path.name

            # unknown directories should not silently become new classes
            if raw_class_name not in CLASS_ALIASES:
                raise ValueError(
                    f"Unknown disease folder: {class_path}"
                )
            # mapping
            class_name = CLASS_ALIASES[raw_class_name]
            label = CLASS_TO_INDEX[class_name]

            for image_path in sorted(class_path.glob("*.jpg")):
                relative_path = image_path.relative_to(dataset_root).as_posix()

                # save records
                records.append({
                    "path": relative_path,
                    "label": label,
                    "class_name": class_name,
                    "background": background
                })

    if not records:
        raise ValueError(
            f"No images found under {dataset_root}"
        )

    return records


def validate_manifest(records: list[dict]) -> None:
    """Check that the manifest matches the supplied assignment dataset."""
    # the dataset is expected to contain exactly 1,106 images
    if len(records) != 1106:
        raise ValueError(
            "Expected 1106 images, "
            f"but found {len(records)}."
        )

    observed_classes = {
        record["class_name"]
        for record in records
    }

    # all five disease classes
    if observed_classes != set(CLASS_NAMES):
        raise ValueError(
            f"Unexpected classes: {observed_classes}"
        )

    observed_backgrounds = {
        record["background"]
        for record in records
    }

    # both required evaluation backgrounds must be represented
    if observed_backgrounds != {"White", "Field"}:
        raise ValueError(
            "Expected White and Field backgrounds, "
            f"but found {observed_backgrounds}."
        )


def save_manifest( records: list[dict], output_path: Path) -> None:
    """Write the manifest records to a CSV file."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "path",
        "label",
        "class_name",
        "background",
        "stratum",
    ]
    #save
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


def print_summary(records: list[dict]) -> None:
    """Print image counts for every background-disease combination."""
    counts = Counter(
        (
            record["background"],
            record["class_name"],
        )
        for record in records
    )

    print(f"Total images: {len(records)}")

    for background in ("White", "Field"):
        print(f"\n{background} Background")

        for class_name in CLASS_NAMES:
            count = counts[
                (background, class_name)
            ]

            print(
                f"  {class_name:<15} {count:>4}"
            )


def main() -> None:
    dataset_root = DEFAULT_DATASET_ROOT
    manifest_path = (
        DEFAULT_DATASET_ROOT / "generated" / "manifest.csv"
    )


    records = build_manifest(dataset_root)
    validate_manifest(records)
    save_manifest(records, manifest_path)
    print_summary(records)

    print(f"\nManifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
