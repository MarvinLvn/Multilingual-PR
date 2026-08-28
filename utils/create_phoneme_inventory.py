"""Create a TinyVox phoneme inventory from a metadata CSV file."""

import argparse
import csv
import json
from pathlib import Path


def create_inventory(metadata_path: Path, output_path: Path) -> list[str]:
    """Write the sorted phoneme inventory and return its contents."""
    phonemes = set()
    with metadata_path.open("r", encoding="utf-8", newline="") as metadata_file:
        rows = csv.DictReader(metadata_file)
        if rows.fieldnames is None or "phones" not in rows.fieldnames:
            raise ValueError(f"Missing 'phones' column in {metadata_path}")

        for row in rows:
            phonemes.update(
                phoneme
                for phoneme in (row.get("phones") or "").split()
                if phoneme != "|"
            )

    inventory = sorted(phonemes)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        json.dump(inventory, output_file, ensure_ascii=False, indent=2)

    return inventory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create unique_phonemes.json from TinyVox metadata."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="TinyVox directory containing metadata.csv.",
    )
    parser.add_argument(
        "--metadata",
        default="metadata.csv",
        help="Metadata filename relative to dataset_path (default: metadata.csv).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output JSON path (default: <dataset_path>/unique_phonemes.json).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata_path = args.dataset_path / args.metadata
    output_path = args.output or args.dataset_path / "unique_phonemes.json"

    if not metadata_path.is_file():
        raise FileNotFoundError(f"Metadata CSV not found: {metadata_path}")

    inventory = create_inventory(metadata_path, output_path)
    print(f"Created {output_path.resolve()} with {len(inventory)} phonemes.")


if __name__ == "__main__":
    main()
