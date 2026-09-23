"""Generate COCO and YOLO monster detection data from mxdclassic assets."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re
import shutil
from typing import Any

from PIL import ImageDraw
import yaml

from .assets import AssetStore
from .render import MapRenderer, SpriteLibrary, build_monsters, foothold_segments


SPECIAL_HINTS = ("event", "wedding", "quest", "carnival", "gm", "party", "guild")


def load_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    required = ("client_root", "output_root", "cache_dir", "seed", "image", "splits")
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"missing config keys: {', '.join(missing)}")
    if config["image"] != {"width": 1280, "height": 224}:
        raise ValueError("this dataset profile requires an unscaled 1280x224 image")
    return config


def _map_id_from_key(key: str) -> str | None:
    match = re.search(r"/(\d{9})\.wzjson$", key)
    return match.group(1) if match else None


def select_maps(store: AssetStore, config: dict[str, Any], output: Path) -> dict[str, list[str]]:
    selection_path = output / "map_selection.json"
    if selection_path.exists():
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        return data["splits"]
    selection_source = config.get("map_selection_from")
    if selection_source:
        source = Path(selection_source)
        data = json.loads(source.read_text(encoding="utf-8"))
        splits = data.get("splits", {})
        if set(splits) != {"train", "val", "test"} or any(not splits[name] for name in splits):
            raise ValueError(f"invalid shared map selection: {source}")
        selection_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return splits
    minimum, maximum = config.get("map_range", [100000000, 199999999])
    include = {str(x) for x in config.get("include_maps", [])}
    candidates: list[str] = []
    failures: list[dict[str, str]] = []
    available_mobs = {
        key.rsplit("/", 1)[-1].removesuffix(".wzspritesheet")
        for key in store.index if "/Mob/" in key and key.endswith(".wzspritesheet")
    }
    for key in sorted(store.index):
        map_id = _map_id_from_key(key)
        if not map_id or not (int(minimum) <= int(map_id) <= int(maximum)):
            continue
        if include and map_id not in include:
            continue
        if any(hint in key.lower() for hint in SPECIAL_HINTS):
            continue
        try:
            root = next(iter(store.read_wzjson(map_id, "/Map/Map/").values()))
            lives = [
                v for v in root.get("life", {}).values()
                if isinstance(v, dict) and v.get("type") == "m"
                and str(v.get("id", "")) in available_mobs
                and (not str(v.get("id", "")).isdigit() or int(v["id"]) < 9_000_000)
            ]
            if lives and foothold_segments(root):
                candidates.append(map_id)
        except Exception as exc:
            failures.append({"map_id": map_id, "error": f"{type(exc).__name__}: {exc}"})
    if len(candidates) < 3:
        raise RuntimeError(f"at least 3 valid monster maps are required, found {len(candidates)}")
    rng = random.Random(int(config["seed"]))
    rng.shuffle(candidates)
    test_count = max(1, round(len(candidates) * 0.10))
    val_count = max(1, round(len(candidates) * 0.10))
    splits = {
        "test": sorted(candidates[:test_count]),
        "val": sorted(candidates[test_count : test_count + val_count]),
        "train": sorted(candidates[test_count + val_count :]),
    }
    payload = {"seed": config["seed"], "candidate_count": len(candidates), "splits": splits, "failures": failures}
    selection_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return splits


def camera_bounds(root: dict[str, Any], width: int, height: int) -> tuple[int, int, int, int]:
    segments = list(foothold_segments(root).values())
    xs = [value for segment in segments for value in (segment[0], segment[2])]
    ys = [value for segment in segments for value in (segment[1], segment[3])]
    left, right = min(xs) - 100, max(xs) + 100
    top, bottom = min(ys) - height, max(ys) + 100
    return left, max(left, right - width), top, max(top, bottom - height)


def overlapping_pairs(labels: list[dict[str, Any]], minimum_iou: float = 0.05) -> int:
    """Count pairs whose bounding boxes overlap by at least minimum_iou."""
    count = 0
    for index, first in enumerate(labels):
        ax, ay, aw, ah = first["bbox"]
        for second in labels[index + 1:]:
            bx, by, bw, bh = second["bbox"]
            intersection = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
                0, min(ay + ah, by + bh) - max(ay, by)
            )
            union = aw * ah + bw * bh - intersection
            if union and intersection / union >= minimum_iou:
                count += 1
    return count


class DatasetWriter:
    def __init__(self, root: Path, width: int, height: int):
        self.root = root
        self.width = width
        self.height = height
        self.coco: dict[str, dict[str, Any]] = {}
        self.annotation_id = 1
        for split in ("train", "val", "test"):
            (root / "images" / split).mkdir(parents=True, exist_ok=True)
            (root / "labels" / split).mkdir(parents=True, exist_ok=True)
            self.coco[split] = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "monster"}]}
        (root / "annotations").mkdir(parents=True, exist_ok=True)
        (root / "visual_checks").mkdir(parents=True, exist_ok=True)

    def add(self, split: str, index: int, image, labels: list[dict[str, Any]], map_id: str, sample_kind: str) -> None:
        stem = f"{split}_{index:06d}"
        image_path = self.root / "images" / split / f"{stem}.jpg"
        image.save(image_path, quality=92, subsampling=0)
        yolo_lines = []
        image_id = index + 1
        self.coco[split]["images"].append({
            "id": image_id, "file_name": f"images/{split}/{stem}.jpg", "width": self.width,
            "height": self.height, "map_id": map_id, "sample_kind": sample_kind,
        })
        for label in labels:
            x, y, w, h = label["bbox"]
            yolo_lines.append(f"0 {(x+w/2)/self.width:.8f} {(y+h/2)/self.height:.8f} {w/self.width:.8f} {h/self.height:.8f}")
            self.coco[split]["annotations"].append({
                "id": self.annotation_id, "image_id": image_id, "category_id": 1, "bbox": [x, y, w, h],
                "area": w * h, "iscrowd": 0, "mob_id": label["mob_id"], "action": label["action"],
            })
            self.annotation_id += 1
        (self.root / "labels" / split / f"{stem}.txt").write_text("\n".join(yolo_lines), encoding="utf-8")
        if sum(len(data["images"]) for data in self.coco.values()) <= 100:
            check = image.copy()
            draw = ImageDraw.Draw(check)
            for label in labels:
                x, y, w, h = label["bbox"]
                draw.rectangle((x, y, x + w, y + h), outline=(255, 32, 32), width=2)
                draw.text((x, max(0, y - 12)), label["mob_id"], fill=(255, 32, 32))
            check.save(self.root / "visual_checks" / f"{stem}.jpg", quality=90)

    def finish(self) -> None:
        combined = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "monster"}]}
        image_offset = 0
        for split, data in self.coco.items():
            (self.root / "annotations" / f"instances_{split}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            id_map = {image["id"]: image["id"] + image_offset for image in data["images"]}
            combined["images"].extend({**image, "id": id_map[image["id"]], "split": split} for image in data["images"])
            combined["annotations"].extend({**ann, "image_id": id_map[ann["image_id"]]} for ann in data["annotations"])
            image_offset += len(data["images"])
        (self.root / "annotations" / "coco.json").write_text(json.dumps(combined, ensure_ascii=False, indent=2), encoding="utf-8")
        (self.root / "data.yaml").write_text(
            "path: .\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: monster\n", encoding="utf-8"
        )


def generate(config_path: str | Path, limit: int | None = None, overwrite: bool = False) -> Path:
    config = load_config(config_path)
    output = Path(config["output_root"]) / config.get("name", "pilot")
    if overwrite and output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)
    store = AssetStore(config["client_root"], config["cache_dir"])
    store.build_index(include_sprites=True)
    shutil.copy2(store.index_path, output / "asset_index.json")
    split_maps = select_maps(store, config, output)
    sprites = SpriteLibrary(store)
    width, height = config["image"]["width"], config["image"]["height"]
    renderer = MapRenderer(sprites, width, height)
    writer = DatasetWriter(output, width, height)
    rng = random.Random(int(config["seed"]))
    counts = dict(config["splits"])
    if limit is not None:
        total = sum(counts.values())
        counts = {key: max(1, round(limit * value / total)) for key, value in counts.items()}
        while sum(counts.values()) > limit:
            key = max(counts, key=counts.get)
            counts[key] -= 1
        while sum(counts.values()) < limit:
            counts["train"] += 1
    kinds = config.get("sampling", {"positive": 0.65, "dense": 0.35})
    kind_names, kind_weights = list(kinds), list(kinds.values())
    instance_config = config.get("instances", {"min": 1, "max": 20})
    instance_min, instance_max = int(instance_config.get("min", 1)), int(instance_config.get("max", 20))
    if not (1 <= instance_min <= instance_max <= 20):
        raise ValueError("instances must satisfy 1 <= min <= max <= 20")
    statistics = Counter()
    failed_maps: list[dict[str, str]] = []
    root_cache: dict[str, dict[str, Any]] = {}
    for split in ("train", "val", "test"):
        maps = split_maps[split]
        for index in range(counts[split]):
            for attempt in range(60):
                map_id = rng.choice(maps)
                try:
                    root = root_cache.get(map_id)
                    if root is None:
                        root = next(iter(store.read_wzjson(map_id, "/Map/Map/").values()))
                        root_cache[map_id] = root
                    kind = rng.choices(kind_names, weights=kind_weights, k=1)[0]
                    target_count = rng.randint(instance_min, instance_max)
                    dense = kind == "dense"
                    if dense and instance_max >= 2:
                        target_count = max(2, target_count)
                    monsters = build_monsters(
                        root, sprites, rng, True, max_count=instance_max, target_count=target_count,
                        dense=dense, view_width=width,
                    )
                    if len(monsters) != target_count:
                        continue
                    left, right, top, bottom = camera_bounds(root, width, height)
                    focus_x = round((min(item.world_x for item in monsters) + max(item.world_x for item in monsters)) / 2)
                    ground_ys = sorted(item.world_y for item in monsters)
                    focus_y = ground_ys[len(ground_ys) // 2]
                    camera_x = max(left, min(right, focus_x - width // 2 + rng.randint(-40, 40)))
                    camera_y = max(top, min(bottom, focus_y - rng.randint(height * 3 // 4, height - 20)))
                    image, labels = renderer.render_scene(root, camera_x, camera_y, monsters)
                    if len(labels) != target_count:
                        continue
                    pair_count = overlapping_pairs(labels)
                    if dense and target_count >= 2 and pair_count == 0:
                        continue
                    writer.add(split, index, image, labels, map_id, kind)
                    statistics[f"images_{split}"] += 1
                    statistics[f"kind_{kind}"] += 1
                    statistics["annotations"] += len(labels)
                    statistics[f"instances_{len(labels):02d}"] += 1
                    statistics["overlapping_pairs"] += pair_count
                    if pair_count:
                        statistics["images_with_overlap"] += 1
                    statistics.update(f"mob_{label['mob_id']}" for label in labels)
                    statistics.update(f"action_{label['action']}" for label in labels)
                    break
                except Exception as exc:
                    failed_maps.append({"map_id": map_id, "error": f"{type(exc).__name__}: {exc}"})
            else:
                raise RuntimeError(f"could not generate {split} image {index} after 60 attempts")
    writer.finish()
    combine_with = config.get("combine_with")
    if combine_with:
        combined_yaml = (
            f"path: {Path(config['output_root']).as_posix()}\n"
            f"train:\n  - {combine_with}/images/train\n  - {config.get('name', 'pilot')}/images/train\n"
            f"val: {combine_with}/images/val\n"
            f"test: {combine_with}/images/test\n"
            "names:\n  0: monster\n"
        )
        (output / "data_combined.yaml").write_text(combined_yaml, encoding="utf-8")
    unique_missing = list({(x["asset"], x["group"], x["error"]): x for x in sprites.missing}.values())
    (output / "missing_assets.json").write_text(json.dumps(unique_missing, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "failed_maps.json").write_text(json.dumps(failed_maps, ensure_ascii=False, indent=2), encoding="utf-8")
    stats = {
        "seed": config["seed"], "image_size": [width, height], "class_names": ["monster"],
        "counts": dict(statistics), "missing_assets": len(unique_missing), "failed_render_attempts": len(failed_maps),
    }
    (output / "dataset_stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    store.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--limit", type=int, help="generate a proportional smoke subset")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    output = generate(args.config, args.limit, args.overwrite)
    print(output)


if __name__ == "__main__":
    main()
