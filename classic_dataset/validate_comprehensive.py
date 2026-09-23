"""Strict validation for the quota-driven comprehensive dataset."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import yaml

from .assets import AssetStore
from .comprehensive import _pairwise_ious, build_schedule
from .render import SpriteLibrary


DEATH_WORDS = ("die", "dead", "death")


def _phash(path: Path) -> int:
    with Image.open(path) as image:
        gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    values = cv2.dct(gray)[:8, :8]
    bits = values > np.median(values[1:])
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


def validate(config_path: str | Path, dataset: str | Path) -> dict[str, Any]:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    root = Path(dataset)
    plan = json.loads((root / "generation_plan.json").read_text(encoding="utf-8"))
    schedule = build_schedule(config, int(plan["total"]))
    expected_splits = Counter(item.split for item in schedule)
    expected_scenes = Counter(item.scenario for item in schedule)
    expected_boxes = Counter(item.target_count for item in schedule)
    expected_players = Counter(item.player_count for item in schedule)
    expected_loot = Counter(item.loot_count for item in schedule)
    expected_target_layouts = Counter(item.target_layout for item in schedule)
    expected_loot_layouts = Counter(item.loot_layout for item in schedule)
    records = [json.loads(line) for line in (root / "scenario_manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    errors: list[str] = []
    stats: Counter[str] = Counter()

    def check(condition: bool, message: str) -> None:
        if not condition:
            errors.append(message)

    check(len(records) == len(schedule), f"record count {len(records)} != {len(schedule)}")
    check(len({item["stem"] for item in records}) == len(records), "duplicate manifest stems")
    check(Counter(item["split"] for item in records) == expected_splits, "split quotas differ from plan")
    check(Counter(item["scenario"] for item in records) == expected_scenes, "scenario quotas differ from plan")
    check(Counter(sum(int(label.get("class_id", 0)) == 0 for label in item["labels"]) for item in records) == expected_boxes, "monster box-count quotas differ from plan")
    if config.get("label_players"):
        check(Counter(sum(int(label.get("class_id", 0)) == 1 for label in item["labels"]) for item in records) == expected_players, "player box-count quotas differ from plan")
    check(Counter(int(item.get("loot_count", 0)) for item in records) == expected_loot, "loot-count quotas differ from plan")
    check(Counter(item.get("target_layout", "none") for item in records) == expected_target_layouts, "target-layout quotas differ from plan")
    check(Counter(item.get("loot_layout", "none") for item in records) == expected_loot_layouts, "loot-layout quotas differ from plan")

    selection = json.loads((root / "map_selection.json").read_text(encoding="utf-8"))
    map_sets = {name: set(values) for name, values in selection["splits"].items()}
    check(not (map_sets["train"] & map_sets["val"] or map_sets["train"] & map_sets["test"] or map_sets["val"] & map_sets["test"]), "map split leakage")
    if len(selection["maps"]) == 177:
        check({key: len(value) for key, value in map_sets.items()} == {"train": 141, "val": 18, "test": 18}, "map split is not 141/18/18")

    seen_sha: set[str] = set()
    phashes: list[int] = []
    buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
    near_pairs = 0
    mob_counts: Counter[str] = Counter()
    frame_counts: Counter[tuple[str, str]] = Counter()
    death_counts: Counter[tuple[str, str]] = Counter()
    action_counts: Counter[str] = Counter()
    player_count = 0
    item_counts: Counter[tuple[str, str]] = Counter()
    allowed_classes = set(range(len(config.get("class_names", ["monster"]))))
    for record_index, record in enumerate(records):
        image_path = root / record["image"]
        label_path = root / record["label"]
        check(image_path.exists(), f"missing image {record['image']}")
        check(label_path.exists(), f"missing label {record['label']}")
        if not image_path.exists() or not label_path.exists():
            continue
        payload = image_path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        check(digest == record["sha256"], f"sha mismatch {record['stem']}")
        check(digest not in seen_sha, f"exact duplicate image {record['stem']}")
        seen_sha.add(digest)
        with Image.open(image_path) as image:
            check(image.size == (1280, 224), f"wrong image size {record['stem']}: {image.size}")
            check(image.format == "JPEG", f"not JPEG {record['stem']}")
        value = _phash(image_path)
        candidates: set[int] = set()
        for chunk in range(4):
            candidates.update(buckets[chunk][(value >> (chunk * 16)) & 0xFFFF])
        if any((value ^ phashes[index]).bit_count() <= 2 for index in candidates):
            near_pairs += 1
        phash_index = len(phashes)
        phashes.append(value)
        for chunk in range(4):
            buckets[chunk][(value >> (chunk * 16)) & 0xFFFF].append(phash_index)

        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        check(len(lines) == len(record["labels"]), f"YOLO/manifest label count mismatch {record['stem']}")
        for line in lines:
            fields = line.split()
            check(len(fields) == 5 and fields[0].isdigit() and int(fields[0]) in allowed_classes, f"invalid YOLO row {record['stem']}: {line}")
            if len(fields) == 5:
                check(all(0.0 <= float(value) <= 1.0 for value in fields[1:]), f"YOLO coordinate outside [0,1] {record['stem']}")
        boxes = [(int(label.get("class_id", 0)), tuple(label["bbox"])) for label in record["labels"]]
        check(len(boxes) == len(set(boxes)), f"duplicate boxes {record['stem']}")
        monster_labels = [label for label in record["labels"] if int(label.get("class_id", 0)) == 0]
        player_labels = [label for label in record["labels"] if int(label.get("class_id", 0)) == 1]
        check(len(monster_labels) == int(record["target_count"]), f"monster target count mismatch {record['stem']}")
        check(len(player_labels) == int(record.get("target_player_count", 0)), f"player target count mismatch {record['stem']}")
        if record["scenario"].startswith("negative_"):
            check(not monster_labels, f"negative scene has monster labels {record['stem']}")
        for label in record["labels"]:
            x, y, width, height = label["bbox"]
            check(width >= 4 and height >= 4 and x >= 0 and y >= 0 and x + width <= 1280 and y + height <= 224, f"invalid bbox {record['stem']}")
            check(float(label.get("visible_fraction", 1.0)) >= float(config.get("minimum_visible_fraction", 0.0)), f"target visibility below minimum {record['stem']}")
            if int(label.get("class_id", 0)) == 0:
                check(not any(word in str(label.get("action", "")).lower() for word in DEATH_WORDS), f"death action labeled {record['stem']}")
                check(label.get("ground_error") is None or float(label["ground_error"]) <= 2.0, f"platform error >2px {record['stem']}")
                mob_id, frame_path = str(label["mob_id"]), str(label.get("frame_path", ""))
                mob_counts[mob_id] += 1
                frame_counts[(mob_id, frame_path)] += 1
                action = str(label.get("action", "")).lower()
                family = next((name for name in config["action_weights"] if action.startswith(name)), action)
                action_counts[family] += 1
            elif int(label.get("class_id", 0)) == 1:
                player_count += 1
        if len(monster_labels) >= 2:
            ious = _pairwise_ious(monster_labels)
            maximum_iou = max(ious, default=0.0)
            span = (
                max(label["bbox"][0] + label["bbox"][2] for label in monster_labels)
                - min(label["bbox"][0] for label in monster_labels)
            ) / 1280
            layout = record.get("target_layout")
            if layout == "distributed":
                low_iou_fraction = sum(value < 0.05 for value in ious) / max(1, len(ious))
                check(
                    span >= 0.45 and (maximum_iou < 0.05 if len(monster_labels) <= 5 else low_iou_fraction >= 0.80),
                    f"distributed layout invalid {record['stem']}",
                )
            elif layout == "clustered":
                limit = 0.70 if record.get("multi_platform") else 0.45
                check(maximum_iou <= limit, f"clustered layout invalid {record['stem']}")
            elif layout == "overlap":
                check(any(0.15 <= value <= 0.45 for value in ious), f"overlap layout invalid {record['stem']}")
            elif layout == "severe_overlap":
                check(any(0.35 <= value <= 0.70 for value in ious), f"severe-overlap layout invalid {record['stem']}")
        if record.get("multi_platform"):
            check(len({label.get("foothold_id") for label in monster_labels}) >= 2, f"multi-platform layout invalid {record['stem']}")
        for overlay in record.get("overlay_frames", []):
            if overlay.get("kind") == "death":
                death_counts[(str(overlay.get("source_id", "")), str(overlay.get("frame_path", "")))] += 1
        hidden_objects = record.get("hidden_objects", [])
        loot_objects = [item for item in hidden_objects if item.get("kind") == "loot"]
        check(len(loot_objects) == int(record.get("loot_count", 0)), f"hidden loot count mismatch {record['stem']}")
        for hidden in hidden_objects:
            bbox = hidden.get("bbox", [])
            check(len(bbox) == 4, f"hidden object missing bbox {record['stem']}")
            if len(bbox) == 4:
                x, y, width, height = bbox
                check(width > 0 and height > 0 and x >= 0 and y >= 0 and x + width <= 1280 and y + height <= 224, f"invalid hidden bbox {record['stem']}")
            check(0.0 <= float(hidden.get("visible_fraction", 0.0)) <= 1.0, f"invalid hidden visibility {record['stem']}")
            check(0.0 <= float(hidden.get("occlusion_fraction", 0.0)) <= 1.0, f"invalid hidden occlusion {record['stem']}")
            check(0.0 <= float(hidden.get("max_target_iou", 0.0)) <= 1.0, f"invalid hidden target IoU {record['stem']}")
            if hidden.get("kind") == "loot":
                check(bool(hidden.get("item_id")) and bool(hidden.get("visual_id")), f"loot identity missing {record['stem']}")
                item_counts[(record["split"], str(hidden.get("visual_id")))] += 1

    near_ratio = near_pairs / max(1, len(records))
    check(near_ratio < 0.005, f"near-duplicate pHash ratio {near_ratio:.3%} >= 0.5%")
    visual_limit = max(1, math.ceil(int(config["visual_checks"]) / max(1, len(expected_scenes))))
    expected_visual_checks = sum(min(count, visual_limit) for count in expected_scenes.values())
    check(
        len(list((root / "visual_checks").glob("*.jpg"))) >= expected_visual_checks,
        "insufficient visual checks",
    )
    for split in ("train", "val", "test"):
        check((root / "annotations" / f"instances_{split}.json").exists(), f"missing COCO {split}")
        for family in ("natural", "dense", "edge", "occlusion", "clutter", "negative"):
            check((root / "annotations" / f"{split}_{family}.txt").exists(), f"missing challenge slice {split}/{family}")

    full_total = sum(int(value) for value in config["splits"].values())
    expected_live: set[tuple[str, str]] = set()
    expected_death: set[tuple[str, str]] = set()
    if len(records) == full_total:
        store = AssetStore(config["client_root"], config["cache_dir"])
        store.build_index(include_sprites=True)
        sprites = SpriteLibrary(store, int(config.get("resource_cache", 384)))
        mobs = sorted({mob for item in selection["maps"] for mob in item["mob_ids"]})
        for mob_id in mobs:
            # Some later-client atlas aliases resolve to a 1x1 placeholder.
            # Such frames cannot produce the required 4x4 detection box and
            # must not be treated as independently renderable animations.
            expected_live.update(
                (mob_id, frame.path)
                for frame in sprites.monster_frames(mob_id)
                if (box := frame.image.getchannel("A").getbbox())
                and box[2] - box[0] >= 4 and box[3] - box[1] >= 4
            )
            expected_death.update((mob_id, frame.path) for frame in sprites.monster_death_frames(mob_id))
        store.close()
        check(all(mob_counts[mob] >= int(config["minimum_mob_boxes"]) for mob in mobs), "one or more mobs below minimum box coverage")
        check(all(frame_counts[key] >= int(config["minimum_frame_uses"]) for key in expected_live), "one or more live frames below minimum coverage")
        check(all(death_counts[key] >= int(config["minimum_death_frame_uses"]) for key in expected_death), "one or more death frames below minimum negative coverage")
        if config.get("label_players"):
            check(player_count >= int(config.get("minimum_player_boxes", 0)), "player boxes below minimum coverage")
        if mob_counts:
            check(max(mob_counts.values()) <= 5 * min(mob_counts.values()), "common/rare mob box ratio exceeds 5x")
        if config.get("item_coverage"):
            item_index = json.loads((root / "item_visual_index.json").read_text(encoding="utf-8"))
            visuals = item_index.get("visuals", [])
            check(len(visuals) == int(config.get("item_visual_count", len(visuals))), "item visual catalog size differs from config")
            challenge = {item["visual_id"] for item in visuals if item.get("challenge")}
            check(not any(item_counts[("train", visual_id)] for visual_id in challenge), "challenge item leaked into train")
            for split in ("train", "val", "test"):
                minimum = int(config.get(f"minimum_item_{split}_uses", 0))
                eligible = [item["visual_id"] for item in visuals if split != "train" or not item.get("challenge")]
                below = [visual_id for visual_id in eligible if item_counts[(split, visual_id)] < minimum]
                check(not below, f"{len(below)} item visuals below {split} minimum coverage")

    report = {
        "ok": not errors,
        "errors": errors,
        "records": len(records),
        "annotations": sum(mob_counts.values()) + player_count,
        "monster_annotations": sum(mob_counts.values()),
        "player_annotations": player_count,
        "split_counts": dict(Counter(item["split"] for item in records)),
        "scenario_counts": dict(Counter(item["scenario"] for item in records)),
        "monster_box_counts": dict(sorted(Counter(sum(int(label.get("class_id", 0)) == 0 for label in item["labels"]) for item in records).items())),
        "player_box_counts": dict(sorted(Counter(sum(int(label.get("class_id", 0)) == 1 for label in item["labels"]) for item in records).items())),
        "map_split_counts": {key: len(value) for key, value in map_sets.items()},
        "mob_counts": dict(mob_counts),
        "action_counts": dict(action_counts),
        "live_frame_catalog": len(expected_live),
        "death_frame_catalog": len(expected_death),
        "minimum_live_frame_uses": min(frame_counts.values(), default=0),
        "minimum_death_frame_uses": min(death_counts.values(), default=0),
        "item_visual_counts": {
            split: len({visual_id for record_split, visual_id in item_counts if record_split == split})
            for split in ("train", "val", "test")
        },
        "item_minimum_uses": {
            split: min((value for (record_split, _), value in item_counts.items() if record_split == split), default=0)
            for split in ("train", "val", "test")
        },
        "near_duplicate_phash_count": near_pairs,
        "near_duplicate_phash_ratio": near_ratio,
    }
    (root / "validation_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    args = parser.parse_args()
    report = validate(args.config, args.dataset)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
