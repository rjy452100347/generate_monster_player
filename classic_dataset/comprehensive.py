"""Generate the quota-driven 60K Victoria Island detection dataset."""

from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict, dataclass
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance
import yaml

from .assets import AssetStore
from .distractors import DistractorLibrary
from .generate import camera_bounds, overlapping_pairs
from .item_catalog import ItemCoverageSampler, build_item_visual_index
from .render import MapRenderer, PlacedMonster, SpriteLibrary, build_monsters, foothold_segments


MAP_PATTERN = re.compile(r"/(\d{9})\.wzjson$")
DEATH_WORDS = ("die", "dead", "death")


def _image_phash(image: Image.Image) -> int:
    gray = np.asarray(image.convert("L").resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float32)
    values = cv2.dct(gray)[:8, :8]
    bits = values > np.median(values[1:])
    result = 0
    for bit in bits.flat:
        result = (result << 1) | int(bit)
    return result


@dataclass(frozen=True)
class PlannedSample:
    ordinal: int
    split: str
    split_index: int
    scenario: str
    target_count: int
    player_count: int = 0
    loot_count: int = 0
    target_layout: str = "none"
    loot_layout: str = "none"
    multi_platform: bool = False


def _scaled_quota(source: dict[Any, int], target: int) -> dict[Any, int]:
    total = sum(source.values())
    raw = {key: target * value / total for key, value in source.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    for key, _ in sorted(raw.items(), key=lambda item: (item[1] - result[item[0]], str(item[0])), reverse=True)[: target - sum(result.values())]:
        result[key] += 1
    return result


def _expand(quota: dict[Any, int]) -> list[Any]:
    return [key for key, count in quota.items() for _ in range(count)]


def _fit_quota_total(quota: dict[Any, int], target: int) -> dict[Any, int]:
    result = dict(quota)
    delta = target - sum(result.values())
    if delta:
        key = max(result, key=result.get)
        result[key] += delta
    if any(value < 0 for value in result.values()):
        raise RuntimeError(f"cannot fit quota to {target}: {result}")
    return result


def _scenario_family(name: str) -> str:
    if name in {"dense_moderate", "dense_extreme"} or "dense" in name:
        return "dense"
    if name in {"edge_crop"} or "edge" in name:
        return "edge"
    if name in {"foreground_occlusion"} or "occlusion" in name:
        return "occlusion"
    if name.startswith("negative_"):
        return "negative"
    if name in {"natural_sparse", "natural_standard", "multi_platform"}:
        return "natural"
    return "clutter"


def build_schedule(config: dict[str, Any], limit: int | None = None) -> list[PlannedSample]:
    full_total = sum(int(value) for value in config["box_quotas"].values())
    total = limit or full_total
    box_quota = _scaled_quota({int(key): int(value) for key, value in config["box_quotas"].items()}, total)
    scene_quota = _scaled_quota({key: int(value["count"]) for key, value in config["scenarios"].items()}, total)
    split_quota = _scaled_quota({key: int(value) for key, value in config["splits"].items()}, total)
    player_quota = _scaled_quota(
        {int(key): int(value) for key, value in config.get("player_quotas", {0: full_total}).items()}, total
    )
    zero_count = box_quota.pop(0, 0)
    negative_quota = {key: value for key, value in scene_quota.items() if not config["scenarios"][key]["positive"]}
    positive_quota = {key: value for key, value in scene_quota.items() if config["scenarios"][key]["positive"]}
    # Scaling the two quota tables independently can differ by one. Keep the
    # box-count contract authoritative and adjust the largest scene bucket.
    negative_delta = zero_count - sum(negative_quota.values())
    if negative_delta:
        key = max(negative_quota, key=negative_quota.get)
        negative_quota[key] += negative_delta
    positive_delta = sum(box_quota.values()) - sum(positive_quota.values())
    if positive_delta:
        key = max(positive_quota, key=positive_quota.get)
        positive_quota[key] += positive_delta

    rng = random.Random(int(config["seed"]))
    remaining = Counter(box_quota)
    assignments: list[tuple[str, int]] = []
    restrictive = sorted(
        positive_quota,
        key=lambda name: (
            int(config["scenarios"][name].get("max_instances", 20))
            - int(config["scenarios"][name].get("min_instances", 1)),
            name,
        ),
    )
    for scenario in restrictive:
        spec = config["scenarios"][scenario]
        low, high = int(spec.get("min_instances", 1)), int(spec.get("max_instances", 20))
        for _ in range(positive_quota[scenario]):
            choices = [count for count in range(low, high + 1) if remaining[count] > 0]
            if not choices:
                raise RuntimeError(f"box quota cannot satisfy scenario {scenario} range {low}..{high}")
            count = rng.choices(choices, weights=[remaining[value] for value in choices], k=1)[0]
            remaining[count] -= 1
            assignments.append((scenario, count))
    if any(remaining.values()):
        raise RuntimeError(f"unassigned box quota: {dict(remaining)}")
    assignments.extend((scenario, 0) for scenario, count in negative_quota.items() for _ in range(count))
    # Hard-negative profiles use disjoint player/no-player scenarios. Scaling
    # scenario and player quotas independently can otherwise differ by one.
    player_specs = [config["scenarios"][scenario] for scenario, _ in assignments]
    if all(int(spec.get("max_players", 5)) == 0 or int(spec.get("min_players", 0)) >= 1 for spec in player_specs):
        required_player_images = sum(int(spec.get("min_players", 0)) >= 1 for spec in player_specs)
        nonzero_source = {key: value for key, value in player_quota.items() if key > 0}
        player_quota = {0: len(assignments) - required_player_images, **_scaled_quota(nonzero_source, required_player_images)}
    player_remaining = Counter(player_quota)
    player_values: list[int | None] = [None] * len(assignments)
    indexes = sorted(
        range(len(assignments)),
        key=lambda index: (
            int(config["scenarios"][assignments[index][0]].get("max_players", 5))
            - int(config["scenarios"][assignments[index][0]].get("min_players", 0)),
            assignments[index][0], index,
        ),
    )
    for index in indexes:
        spec = config["scenarios"][assignments[index][0]]
        low, high = int(spec.get("min_players", 0)), int(spec.get("max_players", 5))
        choices = [value for value in range(low, high + 1) if player_remaining[value] > 0]
        if not choices:
            raise RuntimeError(f"player quota cannot satisfy scenario {assignments[index][0]} range {low}..{high}")
        value = rng.choices(choices, weights=[player_remaining[item] for item in choices], k=1)[0]
        player_remaining[value] -= 1
        player_values[index] = value
    loot_values = [0] * len(assignments)
    if config.get("loot_count_quotas"):
        loot_indexes = [
            index for index, (scenario, _) in enumerate(assignments)
            if int(config["scenarios"][scenario].get("max_loot", 0)) > 0
        ]
        loot_quota = _scaled_quota(
            {int(key): int(value) for key, value in config["loot_count_quotas"].items()},
            len(loot_indexes),
        )
        loot_remaining = Counter(_fit_quota_total(loot_quota, len(loot_indexes)))
        for index in sorted(
            loot_indexes,
            key=lambda item: (
                int(config["scenarios"][assignments[item][0]].get("max_loot", 20))
                - int(config["scenarios"][assignments[item][0]].get("min_loot", 1)),
                assignments[item][0], item,
            ),
        ):
            spec = config["scenarios"][assignments[index][0]]
            low, high = int(spec.get("min_loot", 1)), int(spec.get("max_loot", 20))
            choices = [value for value in range(low, high + 1) if loot_remaining[value] > 0]
            if not choices:
                raise RuntimeError(f"loot quota cannot satisfy scenario {assignments[index][0]} range {low}..{high}")
            value = rng.choices(choices, weights=[loot_remaining[item] for item in choices], k=1)[0]
            loot_remaining[value] -= 1
            loot_values[index] = value
        if any(loot_remaining.values()):
            raise RuntimeError(f"unassigned loot quota: {dict(loot_remaining)}")

    loot_layouts = ["none"] * len(assignments)
    if config.get("loot_layout_quotas"):
        loot_indexes = [index for index, value in enumerate(loot_values) if value > 0]
        layout_quota = _scaled_quota(
            {str(key): int(value) for key, value in config["loot_layout_quotas"].items()}, len(loot_indexes)
        )
        values = _expand(_fit_quota_total(layout_quota, len(loot_indexes)))
        rng.shuffle(values)
        for index, value in zip(loot_indexes, values):
            loot_layouts[index] = value

    layout_weights = config.get("target_layout_weights", {})
    combined = []
    for index, (scenario, count) in enumerate(assignments):
        layout = "none"
        multi_platform = False
        if count:
            if count == 1:
                layout = "distributed"
            else:
                bucket = next((
                    key for key in layout_weights
                    if int(key.split("-")[0]) <= count <= int(key.split("-")[1])
                ), None)
                choices = layout_weights.get(bucket, {"distributed": 1.0})
                layout = rng.choices(list(choices), weights=list(choices.values()), k=1)[0]
            multi_platform = count > 4 and rng.random() < float(config.get("multi_platform_fraction", 0.0))
            if multi_platform and layout not in {"distributed", "clustered"}:
                layout = rng.choice(("distributed", "clustered"))
        combined.append((
            scenario, count, int(player_values[index] or 0), loot_values[index],
            layout, loot_layouts[index], multi_platform,
        ))
    rng.shuffle(combined)
    split_names = _expand(split_quota)
    rng.shuffle(split_names)
    if total == full_total and config.get("item_coverage") and config.get("item_visual_count"):
        minimum_instances = {
            split: int(config.get(f"minimum_item_{split}_uses", 0)) * int(config["item_visual_count"])
            + int(config.get("item_coverage_reserve_per_split", 0))
            for split in ("val", "test")
        }
        for split in ("val", "test"):
            while sum(combined[index][3] for index, name in enumerate(split_names) if name == split) < minimum_instances[split]:
                inside = [index for index, name in enumerate(split_names) if name == split]
                train = [index for index, name in enumerate(split_names) if name == "train"]
                low = min(inside, key=lambda index: combined[index][3])
                high = max(train, key=lambda index: combined[index][3])
                if combined[high][3] <= combined[low][3]:
                    raise RuntimeError(f"cannot allocate enough item instances to {split}")
                split_names[low], split_names[high] = split_names[high], split_names[low]
    split_indexes = Counter()
    result = []
    for ordinal, ((scenario, count, player_count, loot_count, target_layout, loot_layout, multi_platform), split) in enumerate(zip(combined, split_names)):
        result.append(PlannedSample(
            ordinal, split, split_indexes[split], scenario, count, player_count,
            loot_count, target_layout, loot_layout, multi_platform,
        ))
        split_indexes[split] += 1
    return result


def select_maps(store: AssetStore, sprites: SpriteLibrary, config: dict[str, Any], output: Path) -> tuple[dict[str, list[str]], dict[str, dict[str, Any]]]:
    path = output / "map_selection.json"
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload["splits"], {item["map_id"]: item for item in payload["maps"]}
    minimum, maximum = (int(value) for value in config["map_range"])
    available_mobs = {
        key.rsplit("/", 1)[-1].removesuffix(".wzspritesheet")
        for key in store.index if "/Mob/" in key and key.endswith(".wzspritesheet")
    }
    records: list[dict[str, Any]] = []
    failures = []
    for key in sorted(store.index):
        match = MAP_PATTERN.search(key)
        if not match:
            continue
        map_id = match.group(1)
        if not minimum <= int(map_id) <= maximum:
            continue
        try:
            root = next(iter(store.read_wzjson(map_id, "/Map/Map/").values()))
            info = root.get("info", {})
            mob_ids = sorted({
                str(value.get("id", ""))
                for value in root.get("life", {}).values()
                if isinstance(value, dict) and value.get("type") == "m"
                and str(value.get("id", "")) in available_mobs
                and int(str(value.get("id", "0"))) < 9_000_000
            })
            if mob_ids and foothold_segments(root) and int(info.get("fieldType", 0)) == 0 and int(info.get("timeLimit", 0)) == 0:
                records.append({"map_id": map_id, "mob_ids": mob_ids, "spawn_count": sum(
                    isinstance(value, dict) and value.get("type") == "m" and str(value.get("id", "")) in mob_ids
                    for value in root.get("life", {}).values()
                )})
        except Exception as exc:
            failures.append({"map_id": map_id, "error": f"{type(exc).__name__}: {exc}"})
    if len(records) < 3:
        raise RuntimeError(f"insufficient normal monster maps: {len(records)}")
    mob_maps: dict[str, list[str]] = defaultdict(list)
    by_id = {record["map_id"]: record for record in records}
    for record in records:
        for mob_id in record["mob_ids"]:
            mob_maps[mob_id].append(record["map_id"])
    forced_train = {maps[0] for maps in mob_maps.values() if len(maps) == 1}
    remaining_counts = Counter({mob_id: len(maps) for mob_id, maps in mob_maps.items()})
    rng = random.Random(int(config["seed"]))
    candidates = [record["map_id"] for record in records if record["map_id"] not in forced_train]
    rng.shuffle(candidates)
    val_test: list[str] = []
    # 177 Victoria Island maps intentionally become 141/18/18. Compute each
    # holdout side independently so Python's floor division cannot yield
    # 143/17/17.
    test_count = round(len(records) * 0.10)
    val_count = round(len(records) * 0.10)
    target_holdout = test_count + val_count
    for map_id in candidates:
        mobs = by_id[map_id]["mob_ids"]
        if len(val_test) < target_holdout and all(remaining_counts[mob_id] > 1 for mob_id in mobs):
            val_test.append(map_id)
            for mob_id in mobs:
                remaining_counts[mob_id] -= 1
    if len(val_test) < target_holdout:
        raise RuntimeError(f"could only allocate {len(val_test)} of {target_holdout} holdout maps while preserving train mobs")
    rng.shuffle(val_test)
    splits = {
        "test": sorted(val_test[:test_count]),
        "val": sorted(val_test[test_count : test_count + val_count]),
    }
    held = set(splits["test"]) | set(splits["val"])
    splits["train"] = sorted(record["map_id"] for record in records if record["map_id"] not in held)
    payload = {
        "seed": config["seed"], "candidate_count": len(records), "forced_train": sorted(forced_train),
        "splits": splits, "maps": records, "failures": failures,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits, by_id


class RootCache:
    def __init__(self, store: AssetStore, limit: int = 32):
        self.store = store
        self.limit = limit
        self.values: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def get(self, map_id: str) -> dict[str, Any]:
        if map_id in self.values:
            self.values.move_to_end(map_id)
            return self.values[map_id]
        root = next(iter(self.store.read_wzjson(map_id, "/Map/Map/").values()))
        self.values[map_id] = root
        while len(self.values) > self.limit:
            self.values.popitem(last=False)
        return root


class ResumableWriter:
    def __init__(
        self, root: Path, width: int, height: int, visual_total: int,
        scenarios: list[str], jpeg_quality: int = 88, class_names: list[str] | None = None,
    ):
        self.root, self.width, self.height = root, width, height
        self.jpeg_quality = jpeg_quality
        self.class_names = class_names or ["monster"]
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "visual_checks").mkdir(parents=True, exist_ok=True)
        self.manifest_path = root / "scenario_manifest.jsonl"
        self.records: dict[str, dict[str, Any]] = {}
        self.hashes: set[str] = set()
        self.phashes: list[int] = []
        self.phash_buckets: list[dict[int, list[int]]] = [defaultdict(list) for _ in range(4)]
        valid_manifest_lines: list[str] = []
        dropped_manifest_records = 0
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    record = json.loads(line)
                    if not (root / record["image"]).exists() or not (root / record["label"]).exists():
                        dropped_manifest_records += 1
                        continue
                    valid_manifest_lines.append(line)
                    self.records[record["stem"]] = record
                    self.hashes.add(record["sha256"])
                    if "phash" in record:
                        value = int(record["phash"], 16)
                    else:
                        image_path = root / record["image"]
                        with Image.open(image_path) as existing:
                            value = _image_phash(existing)
                    self._remember_phash(value)
        if dropped_manifest_records:
            manifest_tmp = self.manifest_path.with_suffix(".jsonl.tmp")
            manifest_tmp.write_text("\n".join(valid_manifest_lines) + "\n", encoding="utf-8")
            manifest_tmp.replace(self.manifest_path)
            print(f"dropped {dropped_manifest_records} manifest records with missing image/label files")
        self.visual_limit = max(1, math.ceil(visual_total / max(1, len(scenarios))))
        self.visual_counts = Counter(record["scenario"] for record in self.records.values() if record.get("visual_check"))

    def has(self, sample: PlannedSample) -> bool:
        return f"{sample.split}_{sample.split_index:06d}" in self.records

    def _remember_phash(self, value: int) -> None:
        index = len(self.phashes)
        self.phashes.append(value)
        for chunk in range(4):
            self.phash_buckets[chunk][(value >> (chunk * 16)) & 0xFFFF].append(index)

    def _phash_is_near(self, value: int) -> bool:
        candidates: set[int] = set()
        for chunk in range(4):
            candidates.update(self.phash_buckets[chunk][(value >> (chunk * 16)) & 0xFFFF])
        return any((value ^ self.phashes[index]).bit_count() <= 2 for index in candidates)

    def add(self, sample: PlannedSample, image: Image.Image, labels: list[dict[str, Any]], metadata: dict[str, Any]) -> bool:
        stem = f"{sample.split}_{sample.split_index:06d}"
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=self.jpeg_quality, subsampling=0, optimize=False)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in self.hashes:
            return False
        phash = _image_phash(image)
        if self._phash_is_near(phash):
            return False
        yolo_lines = []
        for label in labels:
            x, y, w, h = label["bbox"]
            yolo_lines.append(f"{int(label.get('class_id', 0))} {(x+w/2)/self.width:.8f} {(y+h/2)/self.height:.8f} {w/self.width:.8f} {h/self.height:.8f}")
        image_path = self.root / "images" / sample.split / f"{stem}.jpg"
        label_path = self.root / "labels" / sample.split / f"{stem}.txt"
        image_tmp = image_path.with_suffix(".jpg.tmp")
        label_tmp = label_path.with_suffix(".txt.tmp")
        image_tmp.write_bytes(payload)
        label_tmp.write_text("\n".join(yolo_lines), encoding="utf-8")
        image_tmp.replace(image_path)
        label_tmp.replace(label_path)
        visual = self.visual_counts[sample.scenario] < self.visual_limit
        if visual:
            check = image.copy()
            draw = ImageDraw.Draw(check)
            for label in labels:
                x, y, w, h = label["bbox"]
                color = (255, 32, 32) if int(label.get("class_id", 0)) == 0 else (32, 160, 255)
                draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
                draw.text((x, max(0, y - 12)), str(label.get("mob_id", label.get("class_name", "object"))), fill=color)
            check.save(self.root / "visual_checks" / f"{sample.scenario}_{stem}.jpg", quality=92, subsampling=0)
            self.visual_counts[sample.scenario] += 1
        record = {
            "stem": stem, "split": sample.split, "split_index": sample.split_index,
            "scenario": sample.scenario, "scenario_family": _scenario_family(sample.scenario),
            "target_count": sample.target_count, "sha256": digest, "visual_check": visual,
            "target_player_count": sample.player_count,
            "phash": f"{phash:016x}",
            "image": f"images/{sample.split}/{stem}.jpg", "label": f"labels/{sample.split}/{stem}.txt",
            "labels": labels, **metadata,
        }
        with self.manifest_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.records[stem] = record
        self.hashes.add(digest)
        self._remember_phash(phash)
        return True

    def finish(self) -> None:
        ordered = sorted(self.records.values(), key=lambda item: (item["split"], item["split_index"]))
        for split in ("train", "val", "test"):
            images, annotations = [], []
            annotation_id = 1
            subset = [record for record in ordered if record["split"] == split]
            for image_id, record in enumerate(subset, 1):
                images.append({
                    "id": image_id, "file_name": record["image"], "width": self.width, "height": self.height,
                    "map_id": record["map_id"], "sample_kind": record["scenario"],
                    "scenario_family": record["scenario_family"], "distractors": record["distractors"],
                })
                for label in record["labels"]:
                    x, y, w, h = label["bbox"]
                    annotations.append({
                        "id": annotation_id, "image_id": image_id, "category_id": int(label.get("class_id", 0)) + 1,
                        "bbox": [x, y, w, h], "area": w * h, "iscrowd": 0,
                        **{key: value for key, value in label.items() if key not in {"bbox", "area"}},
                    })
                    annotation_id += 1
            coco = {"images": images, "annotations": annotations, "categories": [
                {"id": index + 1, "name": name} for index, name in enumerate(self.class_names)
            ]}
            (self.root / "annotations" / f"instances_{split}.json").write_text(
                json.dumps(coco, ensure_ascii=False, separators=(",", ":")), encoding="utf-8"
            )
            for family in ("natural", "dense", "edge", "occlusion", "clutter", "negative"):
                paths = [record["image"] for record in subset if record["scenario_family"] == family]
                (self.root / "annotations" / f"{split}_{family}.txt").write_text("\n".join(paths), encoding="utf-8")
        (self.root / "data.yaml").write_text(
            f"path: {self.root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n"
            + "".join(f"  {index}: {name}\n" for index, name in enumerate(self.class_names)), encoding="utf-8"
        )


def _write_combined_dataset_yaml(output: Path, config: dict[str, Any], class_names: list[str]) -> None:
    """Write a ready-to-train union of the immutable base set and this extension."""
    base_value = config.get("base_dataset")
    if not base_value:
        return
    base = Path(base_value).resolve()
    current = output.resolve()
    payload = {
        "train": [(base / "images" / "train").as_posix(), (current / "images" / "train").as_posix()],
        "val": [(base / "images" / "val").as_posix(), (current / "images" / "val").as_posix()],
        "test": [(base / "images" / "test").as_posix(), (current / "images" / "test").as_posix()],
        "names": {index: name for index, name in enumerate(class_names)},
    }
    target = output / "data_combined_60k_plus_hard_negative.yaml"
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(target)


def _alpha_height(frame: Any) -> int:
    box = frame.image.getchannel("A").getbbox()
    return box[3] - box[1] if box else 0


def _pairwise_ious(labels: list[dict[str, Any]]) -> list[float]:
    values = []
    for index, first in enumerate(labels):
        ax, ay, aw, ah = first["bbox"]
        for second in labels[index + 1 :]:
            bx, by, bw, bh = second["bbox"]
            intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
            union = aw * ah + bw * bh - intersection
            values.append(intersection / union if union else 0.0)
    return values


def _placement_for(
    scene: str, count: int, rng: random.Random,
    target_layout: str = "none", multi_platform: bool = False,
) -> str:
    if multi_platform:
        return "multi_platform"
    explicit = {
        "distributed": "natural", "clustered": "dense",
        "overlap": "dense", "severe_overlap": "extreme_overlap",
    }
    if target_layout in explicit:
        return explicit[target_layout]
    if scene == "multi_platform":
        return "multi_platform"
    if scene == "dense_moderate":
        return "dense"
    if scene == "dense_extreme":
        return "extreme_overlap"
    if scene.startswith("natural_"):
        return "natural"
    if scene == "foreground_occlusion":
        return "single_platform"
    if scene in {"edge_crop", "size_extreme", "rare_action", "flying_jump"}:
        return "natural"
    if count <= 1:
        return "natural"
    if count <= 5:
        return rng.choices(("natural", "single_platform", "dense"), weights=(60, 25, 15), k=1)[0]
    if count <= 10:
        return rng.choices(("natural", "single_platform", "dense"), weights=(35, 40, 25), k=1)[0]
    if count <= 15:
        return rng.choices(("natural", "single_platform", "dense"), weights=(15, 45, 40), k=1)[0]
    return rng.choices(("natural", "single_platform", "extreme_overlap"), weights=(5, 35, 60), k=1)[0]


def _multi_platform_mobs(root: dict[str, Any], width: int, height: int) -> set[str]:
    """Return mobs whose spawn has another visible spawn on a different foothold."""
    lives = [
        item for item in root.get("life", {}).values()
        if isinstance(item, dict) and item.get("type") == "m" and item.get("id")
    ]
    result: set[str] = set()
    for first in lives:
        first_fh = int(first.get("fh", -1))
        first_x = int(first.get("x", 0))
        first_y = int(first.get("cy", first.get("y", 0)))
        if any(
            int(second.get("fh", -1)) != first_fh
            and abs(int(second.get("x", 0)) - first_x) <= width // 2 - 80
            and abs(int(second.get("cy", second.get("y", 0))) - first_y) <= height - 48
            for second in lives
        ):
            result.add(str(first["id"]))
    return result


def _mob_movement_ranges(root: dict[str, Any]) -> dict[str, int]:
    result: dict[str, int] = {}
    for item in root.get("life", {}).values():
        if not isinstance(item, dict) or item.get("type") != "m" or not item.get("id"):
            continue
        width = abs(int(item.get("rx1", item.get("x", 0))) - int(item.get("rx0", item.get("x", 0))))
        mob_id = str(item["id"])
        result[mob_id] = max(result.get(mob_id, 0), width)
    return result


def _camera(root: dict[str, Any], monsters: list[PlacedMonster], width: int, height: int, rng: random.Random, edge: bool) -> tuple[int, int]:
    left, right, top, bottom = camera_bounds(root, width, height)
    if not monsters:
        return rng.randint(left, right), rng.randint(top, bottom)
    focus_x = round((min(item.world_x for item in monsters) + max(item.world_x for item in monsters)) / 2)
    ground_ys = sorted(item.world_y for item in monsters)
    focus_y = ground_ys[len(ground_ys) // 2]
    camera_x = max(left, min(right, focus_x - width // 2 + rng.randint(-80, 80)))
    camera_y = max(top, min(bottom, focus_y - rng.randint(height * 2 // 3, height - 16)))
    if not edge:
        return camera_x, camera_y
    side = rng.choice(("left", "right", "left", "right", "top", "bottom"))
    # Crop an outermost target. Cropping a middle monster shifts a distributed
    # group out of the opposite side of the 1280px viewport and makes the
    # exact target-count constraint needlessly impossible.
    if side == "left":
        target = min(monsters, key=lambda item: item.world_x)
    elif side == "right":
        target = max(monsters, key=lambda item: item.world_x)
    elif side == "top":
        target = min(monsters, key=lambda item: item.world_y - item.origin_y)
    else:
        target = max(monsters, key=lambda item: item.world_y)
    sprite = target.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if target.flip else target.image
    box = sprite.getchannel("A").getbbox()
    if not box:
        return camera_x, camera_y
    origin_x = sprite.width - target.origin_x if target.flip else target.origin_x
    fraction = rng.uniform(0.10, 0.70)
    if side == "left":
        desired = -round((box[2] - box[0]) * fraction)
        camera_x = round(target.world_x - origin_x + box[0] - desired)
    elif side == "right":
        desired = width - round((box[2] - box[0]) * (1 - fraction))
        camera_x = round(target.world_x - origin_x + box[0] - desired)
    elif side == "top":
        desired = -round((box[3] - box[1]) * fraction)
        camera_y = round(target.world_y - target.origin_y + box[1] - desired)
    else:
        desired = height - round((box[3] - box[1]) * (1 - fraction))
        camera_y = round(target.world_y - target.origin_y + box[1] - desired)
    return max(left, min(right, camera_x)), max(top, min(bottom, camera_y))


def _overlay_kinds(
    scene: str, rng: random.Random, exact_players: int | None = None,
    scene_spec: dict[str, Any] | None = None,
) -> list[str]:
    if scene_spec and "overlays" in scene_spec:
        result = ["player"] * int(exact_players or 0)
        for kind in scene_spec.get("overlays", []):
            if kind == "pet":
                result.extend(["pet"] * rng.randint(
                    int(scene_spec.get("min_pets", 1)), int(scene_spec.get("max_pets", 1))
                ))
            else:
                result.append(str(kind))
        return result
    if scene == "negative_empty":
        return []
    if scene == "negative_actor":
        result = ["player", "pet"]
        return result if exact_players is None else ["pet"] + ["player"] * exact_players
    if scene == "negative_combat":
        result = ["skill", "damage", "loot"]
        return result if exact_players is None else result + ["player"] * exact_players
    if scene == "foreground_occlusion":
        result = ["skill", rng.choice(("player", "damage", "pet"))]
        return result if exact_players is None else [item for item in result if item != "player"] + ["player"] * exact_players
    if scene.startswith("negative_"):
        return [] if exact_players is None else ["player"] * exact_players
    if rng.random() < 0.20:
        return [] if exact_players is None else ["player"] * exact_players
    result = rng.choices(
        (["player", "pet"], ["skill", "damage"], ["loot", "player"], ["player", "pet", "skill", "damage", "loot"]),
        weights=(20, 25, 15, 20), k=1,
    )[0]
    return result if exact_players is None else [item for item in result if item != "player"] + ["player"] * exact_players


def generate(config_path: str | Path, limit: int | None = None, name: str | None = None, overwrite: bool = False) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    width, height = int(config["image"]["width"]), int(config["image"]["height"])
    if (width, height) != (1280, 224):
        raise ValueError("comprehensive profile requires 1280x224")
    output = Path(config["output_root"]) / (name or config["name"])
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    schedule = build_schedule(config, limit)
    plan_payload = {
        "seed": config["seed"], "total": len(schedule), "schedule_sha256": hashlib.sha256(
            json.dumps([asdict(item) for item in schedule], separators=(",", ":")).encode()
        ).hexdigest(), "config_sha256": hashlib.sha256(
            json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    plan_path = output / "generation_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan_payload:
        raise RuntimeError("existing output uses a different generation plan; use another name or --overwrite")
    plan_path.write_text(json.dumps(plan_payload, indent=2), encoding="utf-8")

    store = AssetStore(config["client_root"], config["cache_dir"])
    store.build_index(include_sprites=True)
    shutil.copy2(store.index_path, output / "asset_index.json")
    sprites = SpriteLibrary(store, int(config.get("resource_cache", 384)))
    distractors = DistractorLibrary(sprites)
    item_index_path = output / "item_visual_index.json"
    shared_item_index = Path(config.get("item_index_source", item_index_path))
    if config.get("item_coverage") and shared_item_index != item_index_path and shared_item_index.exists() and not item_index_path.exists():
        shutil.copy2(shared_item_index, item_index_path)
    item_visuals = build_item_visual_index(
        sprites, item_index_path, int(config["seed"])
    ) if config.get("item_coverage") else []
    renderer = MapRenderer(sprites, width, height)
    split_maps, map_records = select_maps(store, sprites, config, output)
    roots = RootCache(store, int(config.get("map_cache", 32)))
    geometry_path = output / "map_geometry_index.json"
    shared_geometry = Path(config.get("map_geometry_source", geometry_path))
    if shared_geometry != geometry_path and shared_geometry.exists() and not geometry_path.exists():
        shutil.copy2(shared_geometry, geometry_path)
    if geometry_path.exists():
        geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
        multi_mobs_by_map = {map_id: set(values) for map_id, values in geometry["multi_mobs_by_map"].items()}
        movement_ranges_by_map = {
            map_id: {mob_id: int(value) for mob_id, value in values.items()}
            for map_id, values in geometry["movement_ranges_by_map"].items()
        }
    else:
        multi_mobs_by_map: dict[str, set[str]] = {}
        movement_ranges_by_map: dict[str, dict[str, int]] = {}
        for maps in split_maps.values():
            for map_id in maps:
                root = roots.get(map_id)
                multi_mobs_by_map[map_id] = _multi_platform_mobs(root, width, height)
                movement_ranges_by_map[map_id] = _mob_movement_ranges(root)
        geometry = {
            "multi_mobs_by_map": {map_id: sorted(values) for map_id, values in multi_mobs_by_map.items()},
            "movement_ranges_by_map": movement_ranges_by_map,
        }
        temporary = geometry_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(geometry, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(geometry_path)
        if shared_geometry != geometry_path and not shared_geometry.exists():
            shared_geometry.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(geometry_path, shared_geometry)
    writer = ResumableWriter(
        output, width, height, int(config.get("visual_checks", 500)),
        list(config["scenarios"]), int(config.get("jpeg_quality", 88)),
        list(config.get("class_names", ["monster"])),
    )
    action_weights = {key: float(value) for key, value in config["action_weights"].items()}
    mob_maps: dict[str, dict[str, list[str]]] = {split: defaultdict(list) for split in split_maps}
    for split, maps in split_maps.items():
        for map_id in maps:
            for mob_id in map_records[map_id]["mob_ids"]:
                mob_maps[split][mob_id].append(map_id)
    all_mobs = sorted(set(mob_maps["train"]))
    frames_by_mob = {mob_id: sprites.monster_frames(mob_id) for mob_id in all_mobs}
    death_frames_by_mob = {mob_id: sprites.monster_death_frames(mob_id) for mob_id in all_mobs}
    frame_counts = Counter()
    death_frame_counts = Counter()
    mob_counts = Counter()
    map_counts = Counter()
    item_counts: Counter[tuple[str, str]] = Counter()
    for record in writer.records.values():
        map_counts[(record["split"], record["map_id"])] += 1
        for label in record["labels"]:
            if int(label.get("class_id", 0)) == 0:
                mob_counts[label["mob_id"]] += 1
                frame_counts[(label["mob_id"], label.get("frame_path", ""))] += 1
        for overlay in record.get("overlay_frames", []):
            if overlay.get("kind") == "death":
                death_frame_counts[(overlay.get("source_id", ""), overlay.get("frame_path", ""))] += 1
        for hidden in record.get("hidden_objects", []):
            if hidden.get("kind") == "loot" and hidden.get("visual_id"):
                item_counts[(record["split"], hidden["visual_id"])] += 1
    item_sampler = ItemCoverageSampler(item_visuals, item_counts) if item_visuals else None
    failure_path = output / "failures.jsonl"
    minimum_free = int(config.get("minimum_free_gb", 5) * 1024**3)
    # Hard validation samples only have 18 maps available. Five failures per
    # map can exhaust that pool before the configured 300 attempts, especially
    # for 13-20 targets plus players, pets and loot. Keep rotating maps, but
    # allow enough retries for the outer attempt budget to remain effective.
    map_retry_limit = 20

    for progress, sample in enumerate(schedule, 1):
        if writer.has(sample):
            continue
        # Per-sample randomness makes a resumed run byte-for-byte equivalent
        # to an uninterrupted run once the manifest state is reconstructed.
        rng = random.Random((int(config["seed"]) << 32) ^ sample.ordinal)
        if progress % 100 == 0 and shutil.disk_usage(output).free < minimum_free:
            store.close()
            raise RuntimeError("free disk space below configured safety threshold; generation stopped resumably")
        scene_spec = config["scenarios"][sample.scenario]
        success = False
        last_error = "constraints not satisfied"
        rejections: Counter[str] = Counter()
        map_rejections: Counter[str] = Counter()
        for attempt in range(int(config.get("max_attempts", 100))):
            if attempt and attempt % 25 == 0:
                print(
                    f"retry {attempt}/{config.get('max_attempts', 100)} for "
                    f"{sample.split}_{sample.split_index:06d} {sample.scenario}: {dict(rejections)}"
                )
            selected_item_visuals: list[str] = []
            def rollback_items() -> None:
                if item_sampler and selected_item_visuals:
                    item_sampler.rollback(sample.split, selected_item_visuals)
                    selected_item_visuals.clear()
            try:
                positive = bool(scene_spec["positive"])
                requested_placement = _placement_for(
                    sample.scenario, sample.target_count, rng,
                    sample.target_layout, sample.multi_platform,
                ) if positive else "natural"
                action_family = None
                size_bucket = None
                target_frame = None
                target_mob = None
                candidate_mobs = sorted(mob_maps[sample.split])
                if positive:
                    if sample.scenario == "rare_action":
                        action_family = rng.choices(("hit", "attack", "skill", "jump", "fly"), weights=(12, 18, 10, 7, 3), k=1)[0]
                    elif sample.scenario == "flying_jump":
                        action_family = rng.choice(("fly", "jump"))
                    if sample.scenario == "size_extreme":
                        size_bucket = rng.choice(("small", "large"))
                    eligible: list[tuple[str, Any]] = []
                    for mob_id in candidate_mobs:
                        frames = frames_by_mob.get(mob_id, [])
                        if action_family:
                            frames = [frame for frame in frames if frame.path.split("/", 1)[0].lower().startswith(action_family)]
                        if size_bucket == "small":
                            frames = [frame for frame in frames if _alpha_height(frame) <= 40]
                        elif size_bucket == "large":
                            frames = [frame for frame in frames if _alpha_height(frame) >= 120]
                        if frames:
                            for frame in frames:
                                deficit = max(0, int(config["minimum_frame_uses"]) - frame_counts[(mob_id, frame.path)])
                                score = (max(0, int(config["minimum_mob_boxes"]) - mob_counts[mob_id]), deficit, -mob_counts[mob_id])
                                eligible.append((mob_id, frame, score))
                    # A resource can be perfectly valid while a particular map
                    # camera is geometrically unable to keep every requested
                    # box visible. After repeated failed render attempts, retain
                    # the coverage target but try one of its other maps; if it
                    # has none, move to the next under-covered frame.
                    eligible = [
                        item for item in eligible
                        if any(map_rejections[map_id] < map_retry_limit for map_id in mob_maps[sample.split][item[0]])
                    ]
                    if not eligible:
                        rejections["no_eligible_frame"] += 1
                        continue
                    eligible.sort(key=lambda item: item[2], reverse=True)
                    if sample.scenario == "foreground_occlusion":
                        possible_maps = split_maps[sample.split]
                        minimum_map_count = min(map_counts[(sample.split, map_id)] for map_id in possible_maps)
                        balanced_maps = [
                            map_id for map_id in possible_maps
                            if map_counts[(sample.split, map_id)] <= minimum_map_count + 2
                            and map_rejections[map_id] < map_retry_limit
                        ]
                        rng.shuffle(balanced_maps)
                        map_id = next((
                            candidate for candidate in balanced_maps
                            if any(item[0] in map_records[candidate]["mob_ids"] for item in eligible)
                        ), None)
                        if map_id is None:
                            rejections["no_balanced_occlusion_map"] += 1
                            continue
                        map_eligible = [item for item in eligible if item[0] in map_records[map_id]["mob_ids"]]
                        target_mob, target_frame, _ = rng.choice(map_eligible[: min(20, len(map_eligible))])
                    elif requested_placement in {"dense", "extreme_overlap"}:
                        factor = 0.18 if requested_placement == "extreme_overlap" else 0.65
                        minimum_spacing = 3 if requested_placement == "extreme_overlap" else 6
                        feasible = []
                        for possible_map in split_maps[sample.split]:
                            if map_rejections[possible_map] >= map_retry_limit:
                                continue
                            ranges = movement_ranges_by_map[possible_map]
                            map_eligible = []
                            for item in eligible:
                                spacing = max(minimum_spacing, min(50, round(item[1].image.width * factor)))
                                if ranges.get(item[0], 0) >= spacing * max(0, sample.target_count - 1) + 4:
                                    map_eligible.append(item)
                            if map_eligible:
                                feasible.append((possible_map, map_eligible))
                        if not feasible:
                            continue
                        minimum_map_count = min(map_counts[(sample.split, item[0])] for item in feasible)
                        map_id, map_eligible = rng.choice([
                            item for item in feasible if map_counts[(sample.split, item[0])] <= minimum_map_count + 2
                        ])
                        target_mob, target_frame, _ = rng.choice(map_eligible[: min(20, len(map_eligible))])
                    elif requested_placement == "multi_platform":
                        feasible = []
                        for possible_map in split_maps[sample.split]:
                            if map_rejections[possible_map] >= map_retry_limit:
                                continue
                            good_mobs = multi_mobs_by_map[possible_map]
                            map_eligible = [item for item in eligible if item[0] in good_mobs]
                            if map_eligible:
                                feasible.append((possible_map, map_eligible))
                        if not feasible:
                            continue
                        minimum_map_count = min(map_counts[(sample.split, item[0])] for item in feasible)
                        map_id, map_eligible = rng.choice([
                            item for item in feasible if map_counts[(sample.split, item[0])] <= minimum_map_count + 2
                        ])
                        top = map_eligible[: min(20, len(map_eligible))]
                        target_mob, target_frame, _ = rng.choice(top)
                    else:
                        top = eligible[: min(20, len(eligible))]
                        target_mob, target_frame, _ = rng.choice(top)
                        possible_maps = [
                            candidate for candidate in mob_maps[sample.split][target_mob]
                            if map_rejections[candidate] < map_retry_limit
                        ]
                        if not possible_maps:
                            rejections["target_maps_exhausted"] += 1
                            continue
                        minimum_map_count = min(map_counts[(sample.split, map_id)] for map_id in possible_maps)
                        map_id = rng.choice([map_id for map_id in possible_maps if map_counts[(sample.split, map_id)] <= minimum_map_count + 2])
                else:
                    maps = split_maps[sample.split]
                    if sample.scenario == "negative_death" or scene_spec.get("death"):
                        death_minimum = int(config.get("minimum_death_frame_uses", 20))
                        scored_maps = []
                        for possible_map in maps:
                            deficits = [
                                max(0, death_minimum - death_frame_counts[(mob_id, frame.path)])
                                for mob_id in map_records[possible_map]["mob_ids"]
                                for frame in death_frames_by_mob.get(mob_id, [])
                            ]
                            scored_maps.append((sum(sorted(deficits, reverse=True)[:5]), -map_counts[(sample.split, possible_map)], possible_map))
                        best_score = max(item[0] for item in scored_maps)
                        best = [item for item in scored_maps if item[0] == best_score]
                        map_id = rng.choice(best)[2]
                    else:
                        minimum_map_count = min(map_counts[(sample.split, map_id)] for map_id in maps)
                        map_id = rng.choice([map_id for map_id in maps if map_counts[(sample.split, map_id)] <= minimum_map_count + 2])
                root = roots.get(map_id)
                edge_mode = bool(scene_spec.get("edge")) or (
                    bool(scene_spec.get("edge_or_occlusion")) and sample.ordinal % 2 == 0
                )
                occlusion_mode = bool(scene_spec.get("occlusion")) or (
                    bool(scene_spec.get("edge_or_occlusion")) and sample.ordinal % 2 == 1
                )
                monsters: list[PlacedMonster] = []
                if positive:
                    monsters = build_monsters(
                        root, sprites, rng, True, max_count=20, target_count=sample.target_count,
                        placement=requested_placement, view_width=width, view_height=height,
                        action_family=action_family, size_bucket=size_bucket,
                        target_mob_id=target_mob, target_frame_path=target_frame.path if target_frame else None,
                        action_weights=action_weights,
                    )
                    if len(monsters) != sample.target_count:
                        rejections[f"placed_{len(monsters)}_wanted_{sample.target_count}"] += 1
                        continue
                # Positive-scene cameras depend only on the target monsters.
                # Resolve the final viewport before placing clutter so loot,
                # players and pets can be kept inside an edge-cropped frame.
                preset_camera = _camera(
                    root, monsters, width, height, rng,
                    sample.scenario == "edge_crop" or edge_mode,
                ) if monsters else None
                overlays = []
                emitted: list[str] = []
                if scene_spec.get("death") or sample.scenario == "negative_death":
                    death_overlays = distractors.death_overlays(
                        root, rng, count=int(scene_spec.get("death_count", 5)), frame_counts=death_frame_counts
                    )
                    if not death_overlays:
                        rejections["missing_death_overlay"] += 1
                        continue
                    overlays.extend(death_overlays)
                    emitted.append("death")
                regular_overlays, regular_emitted = distractors.make_overlays(
                    root, monsters, rng, _overlay_kinds(
                        sample.scenario, rng,
                        sample.player_count if config.get("label_players") else None,
                        scene_spec,
                    ),
                    force_target_overlap=(bool(scene_spec.get("force_target_overlap")) and occlusion_mode)
                    or sample.scenario == "foreground_occlusion",
                    loot_count=sample.loot_count, loot_layout=sample.loot_layout,
                    split=sample.split, item_sampler=item_sampler,
                    viewport=(*preset_camera, width, height) if preset_camera else None,
                )
                overlays.extend(regular_overlays)
                emitted.extend(regular_emitted)
                selected_item_visuals = [item.visual_id for item in overlays if item.visual_id]
                if sum(item.kind == "loot" for item in overlays) != sample.loot_count:
                    rollback_items()
                    rejections["missing_loot_overlay"] += 1
                    continue
                required = set(scene_spec.get("required_overlays", []))
                if required and not required.issubset(emitted):
                    rollback_items()
                    rejections["missing_required_overlay"] += 1
                    continue
                camera_x, camera_y = preset_camera or _camera(
                    root, monsters, width, height, rng,
                    sample.scenario == "edge_crop" or edge_mode,
                )
                if not monsters and overlays:
                    focus_overlays = [item for item in overlays if item.kind in {"player", "loot", "pet", "death"}]
                    if focus_overlays:
                        left, right, top, bottom = camera_bounds(root, width, height)
                        focus_x = round((min(item.world_x for item in focus_overlays) + max(item.world_x for item in focus_overlays)) / 2)
                        ground_ys = sorted(item.world_y for item in focus_overlays)
                        focus_y = ground_ys[len(ground_ys) // 2]
                        camera_x = max(left, min(right, focus_x - width // 2 + rng.randint(-80, 80)))
                        camera_y = max(top, min(bottom, focus_y - rng.randint(height * 2 // 3, height - 16)))
                image, labels, hidden_objects = renderer.render_scene(
                    root, camera_x, camera_y, monsters, overlays=overlays,
                    minimum_clip_fraction=float(config.get("minimum_clip_fraction", 0.30)),
                    minimum_visible_fraction=float(config.get("minimum_visible_fraction", 0.15)),
                    label_players=bool(config.get("label_players", False)),
                    return_hidden_objects=True,
                )
                monster_labels = [label for label in labels if int(label.get("class_id", 0)) == 0]
                player_labels = [label for label in labels if int(label.get("class_id", 0)) == 1]
                visible_hidden = Counter(item.get("kind") for item in hidden_objects)
                expected_pet_count = sum(item.kind == "pet" for item in overlays)
                visibility_ok = visible_hidden["loot"] == sample.loot_count
                if "pet" in required:
                    visibility_ok = visibility_ok and visible_hidden["pet"] == expected_pet_count
                visibility_ok = visibility_ok and all(visible_hidden[kind] > 0 for kind in required - {"loot", "pet"})
                if not visibility_ok:
                    rejections[f"visible_required_{dict(visible_hidden)}"] += 1
                    rollback_items()
                    continue
                if len(monster_labels) != sample.target_count:
                    rejections[f"monster_labels_{len(monster_labels)}_wanted_{sample.target_count}"] += 1
                    map_rejections[map_id] += 1
                    rollback_items()
                    continue
                if config.get("label_players") and len(player_labels) != sample.player_count:
                    rejections[f"player_labels_{len(player_labels)}_wanted_{sample.player_count}"] += 1
                    rollback_items()
                    continue
                label_keys = [(int(label.get("class_id", 0)), tuple(label["bbox"])) for label in labels]
                if len(label_keys) != len(set(label_keys)):
                    rejections["duplicate_class_boxes"] += 1
                    rollback_items()
                    continue
                boxes = [tuple(label["bbox"]) for label in monster_labels]
                if len(boxes) != len(set(boxes)):
                    rejections["duplicate_boxes"] += 1
                    map_rejections[map_id] += 1
                    rollback_items()
                    continue
                pairwise_ious = _pairwise_ious(monster_labels)
                maximum_iou = max(pairwise_ious, default=0.0)
                if sample.scenario == "dense_moderate" and not any(0.05 <= value <= 0.25 for value in pairwise_ious):
                    rollback_items()
                    continue
                if sample.scenario == "dense_extreme" and not any(0.25 <= value <= 0.65 for value in pairwise_ious):
                    rollback_items()
                    continue
                if (sample.scenario == "edge_crop" or edge_mode) and monster_labels and not any(
                    label["edge_sides"] and 0.30 <= label["clip_fraction"] <= 0.90 for label in monster_labels
                ):
                    rollback_items()
                    continue
                if (sample.scenario == "foreground_occlusion" or occlusion_mode) and monster_labels and not any(
                    0.10 <= label["occlusion_fraction"] <= 0.70 for label in monster_labels
                ):
                    rejections["occlusion_ratio"] += 1
                    rollback_items()
                    continue
                if len(monster_labels) >= 2:
                    span = (
                        max(label["bbox"][0] + label["bbox"][2] for label in monster_labels)
                        - min(label["bbox"][0] for label in monster_labels)
                    ) / width
                    layout_ok = True
                    if sample.target_layout == "distributed":
                        low_iou_fraction = sum(value < 0.05 for value in pairwise_ious) / max(1, len(pairwise_ious))
                        layout_ok = span >= 0.45 and (
                            maximum_iou < 0.05 if len(monster_labels) <= 5 else low_iou_fraction >= 0.80
                        )
                    elif sample.target_layout == "clustered":
                        # Multi-platform scenes can project monsters from two
                        # footholds onto the same screen region. Preserve the
                        # global 0.70 visibility-safe overlap ceiling while
                        # avoiding an impossible 0.45 cap for 13-20 targets.
                        layout_ok = maximum_iou <= (0.70 if sample.multi_platform else 0.45)
                    elif sample.target_layout == "overlap":
                        layout_ok = any(0.15 <= value <= 0.45 for value in pairwise_ious)
                    elif sample.target_layout == "severe_overlap":
                        layout_ok = any(0.35 <= value <= 0.70 for value in pairwise_ious)
                    if not layout_ok:
                        rejections[f"layout_{sample.target_layout}"] += 1
                        map_rejections[map_id] += 1
                        rollback_items()
                        continue
                if sample.multi_platform and len({label.get("foothold_id") for label in monster_labels}) < 2:
                    rejections["multi_platform"] += 1
                    map_rejections[map_id] += 1
                    rollback_items()
                    continue
                if sample.scenario == "size_extreme":
                    if size_bucket == "small" and not any(label["bbox"][3] <= 40 for label in monster_labels):
                        rollback_items()
                        continue
                    if size_bucket == "large" and not any(label["bbox"][3] >= 120 for label in monster_labels):
                        rollback_items()
                        continue
                if sample.scenario == "negative_lighting" or scene_spec.get("lighting"):
                    image = ImageEnhance.Brightness(image).enhance(rng.choice((0.55, 0.70, 1.25, 1.40)))
                elif sample.scenario == "negative_transition" or scene_spec.get("transition"):
                    image = Image.blend(image, Image.new("RGB", image.size, (0, 0, 0)), rng.uniform(0.45, 0.90))
                metadata = {
                    "map_id": map_id, "camera": [camera_x, camera_y], "distractors": emitted,
                    "overlay_frames": [
                        {
                            "kind": item.kind, "source_id": item.source_id, "frame_path": item.frame_path,
                            "item_id": item.item_id or None, "visual_id": item.visual_id or None,
                            "resource_path": item.resource_path or item.frame_path, "state": item.state,
                        }
                        for item in overlays
                    ],
                    "hidden_objects": hidden_objects,
                    "max_iou": round(maximum_iou, 4), "overlapping_pairs": overlapping_pairs(monster_labels),
                    "size_target": size_bucket, "action_target": action_family,
                    "target_layout": sample.target_layout, "loot_layout": sample.loot_layout,
                    "loot_count": sample.loot_count, "multi_platform": sample.multi_platform,
                }
                if not writer.add(sample, image, labels, metadata):
                    rejections["duplicate_image_phash"] += 1
                    rollback_items()
                    continue
                map_counts[(sample.split, map_id)] += 1
                for label in labels:
                    if int(label.get("class_id", 0)) == 0:
                        mob_counts[label["mob_id"]] += 1
                        frame_counts[(label["mob_id"], label.get("frame_path", ""))] += 1
                for overlay in overlays:
                    if overlay.kind == "death":
                        death_frame_counts[(overlay.source_id, overlay.frame_path)] += 1
                success = True
                break
            except Exception as exc:
                rollback_items()
                last_error = f"{type(exc).__name__}: {exc}"
        if not success:
            if rejections:
                last_error = f"rejections={dict(rejections)}; map_rejections={dict(map_rejections)}; last={last_error}"
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"sample": asdict(sample), "error": last_error}, ensure_ascii=False) + "\n")
            store.close()
            raise RuntimeError(f"could not generate {sample} after configured attempts: {last_error}")
        if progress % 100 == 0 or progress == len(schedule):
            print(f"generated {len(writer.records)}/{len(schedule)}")
    writer.finish()
    _write_combined_dataset_yaml(output, config, list(config.get("class_names", ["monster"])))
    counts = Counter()
    for record in writer.records.values():
        counts[f"split_{record['split']}"] += 1
        counts[f"scenario_{record['scenario']}"] += 1
        counts[f"boxes_{len(record['labels']):02d}"] += 1
        counts["annotations"] += len(record["labels"])
        counts.update(f"class_{label.get('class_name', 'monster')}" for label in record["labels"])
        counts.update(f"mob_{label['mob_id']}" for label in record["labels"] if int(label.get("class_id", 0)) == 0)
        counts.update(f"action_{label['action']}" for label in record["labels"] if int(label.get("class_id", 0)) == 0)
        counts.update(f"distractor_{kind}" for kind in record["distractors"])
        counts[f"loot_count_{int(record.get('loot_count', 0)):02d}"] += 1
        counts[f"target_layout_{record.get('target_layout', 'none')}"] += 1
        counts[f"loot_layout_{record.get('loot_layout', 'none')}"] += 1
    item_by_id = {item.visual_id: item for item in item_visuals}
    item_category_counts = Counter()
    for (split, visual_id), value in item_counts.items():
        visual = item_by_id.get(visual_id)
        if visual:
            for category in visual.categories:
                item_category_counts[(split, category)] += value
    coverage = {
        "counts": dict(counts), "mob_box_counts": dict(mob_counts),
        "frame_use_min": min(frame_counts.values(), default=0),
        "frame_use_counts": {f"{mob_id}:{path}": value for (mob_id, path), value in frame_counts.items()},
        "death_frame_use_min": min(death_frame_counts.values(), default=0),
        "death_frame_use_counts": {f"{mob_id}:{path}": value for (mob_id, path), value in death_frame_counts.items()},
        "item_visual_count": len(item_visuals),
        "item_challenge_count": sum(item.challenge for item in item_visuals),
        "item_use_counts": {f"{split}:{visual_id}": value for (split, visual_id), value in item_counts.items()},
        "item_category_use_counts": {f"{split}:{category}": value for (split, category), value in item_category_counts.items()},
        "item_minimum_uses": {
            split: min((item_counts[(split, item.visual_id)] for item in item_visuals if split != "train" or not item.challenge), default=0)
            for split in ("train", "val", "test")
        },
        "missing_assets": sprites.missing,
    }
    (output / "coverage_report.json").write_text(json.dumps(coverage, ensure_ascii=False, indent=2), encoding="utf-8")
    store.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--name")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    print(generate(args.config, args.limit, args.name, args.overwrite))


if __name__ == "__main__":
    main()
