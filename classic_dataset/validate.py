"""Validate image, YOLO, COCO, split and sampling invariants."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from PIL import Image


def validate(dataset: str | Path) -> dict[str, Any]:
    root = Path(dataset)
    errors: list[str] = []
    warnings: list[str] = []
    split_maps: dict[str, set[str]] = {}
    counts = Counter()
    for split in ("train", "val", "test"):
        coco_path = root / "annotations" / f"instances_{split}.json"
        if not coco_path.exists():
            errors.append(f"missing {coco_path}")
            continue
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        split_maps[split] = {str(image.get("map_id")) for image in coco["images"]}
        counts[f"images_{split}"] = len(coco["images"])
        counts[f"annotations_{split}"] = len(coco["annotations"])
        by_id = {image["id"]: image for image in coco["images"]}
        annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in by_id}
        for image in coco["images"]:
            path = root / image["file_name"]
            if not path.exists():
                errors.append(f"missing image {path}")
                continue
            with Image.open(path) as bitmap:
                if bitmap.size != (1280, 224):
                    errors.append(f"wrong image size {path}: {bitmap.size}")
            label_path = root / "labels" / split / (Path(image["file_name"]).stem + ".txt")
            if not label_path.exists():
                errors.append(f"missing YOLO label {label_path}")
            else:
                for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                    parts = line.split()
                    if len(parts) != 5 or parts[0] != "0":
                        errors.append(f"invalid YOLO row {label_path}:{line_number}")
                        continue
                    if any(not 0 <= float(value) <= 1 for value in parts[1:]):
                        errors.append(f"YOLO coordinate outside [0,1] {label_path}:{line_number}")
            counts[f"kind_{image.get('sample_kind', 'unknown')}"] += 1
        for annotation in coco["annotations"]:
            image = by_id.get(annotation["image_id"])
            annotations_by_image.setdefault(annotation["image_id"], []).append(annotation)
            x, y, w, h = annotation["bbox"]
            if not image or x < 0 or y < 0 or w < 4 or h < 4 or x + w > 1280 or y + h > 224:
                errors.append(f"illegal COCO bbox annotation {annotation.get('id')}")
            if annotation["area"] != w * h:
                errors.append(f"COCO area mismatch annotation {annotation.get('id')}")
            if any(word in annotation.get("action", "").lower() for word in ("die", "dead", "death")):
                errors.append(f"death action annotation {annotation.get('id')}")
        for image_id, image in by_id.items():
            annotations = annotations_by_image.get(image_id, [])
            instance_count = len(annotations)
            counts[f"instances_{instance_count:02d}"] += 1
            if not 1 <= instance_count <= 20:
                errors.append(f"image {image_id} has {instance_count} boxes; expected 1..20")
            overlapping = False
            for index, first in enumerate(annotations):
                ax, ay, aw, ah = first["bbox"]
                for second in annotations[index + 1:]:
                    bx, by, bw, bh = second["bbox"]
                    intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
                        0, min(ay + ah, by + bh) - max(ay, by)
                    )
                    union = aw * ah + bw * bh - intersection
                    if union and intersection / union >= 0.05:
                        overlapping = True
                        break
                if overlapping:
                    break
            if overlapping:
                counts["images_with_overlap"] += 1
            if image.get("sample_kind") == "dense" and instance_count >= 2 and not overlapping:
                errors.append(f"dense image {image_id} has no overlapping boxes")
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = split_maps.get(a, set()) & split_maps.get(b, set())
        if overlap:
            errors.append(f"map leakage {a}/{b}: {sorted(overlap)}")
    total = sum(counts[f"images_{s}"] for s in ("train", "val", "test"))
    negative_ratio = counts["kind_negative"] / total if total else 0
    overlap_ratio = counts["images_with_overlap"] / total if total else 0
    if total and overlap_ratio < 0.20:
        warnings.append(f"overlap image ratio {overlap_ratio:.3f} below expected 0.20")
    missing_path = root / "missing_assets.json"
    missing_count = len(json.loads(missing_path.read_text(encoding="utf-8"))) if missing_path.exists() else 0
    report = {
        "ok": not errors, "errors": errors, "warnings": warnings, "counts": dict(counts),
        "negative_ratio": negative_ratio, "overlap_image_ratio": overlap_ratio, "missing_assets": missing_count,
        "split_map_counts": {key: len(value) for key, value in split_maps.items()},
    }
    (root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    report = validate(args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


if __name__ == "__main__":
    main()
