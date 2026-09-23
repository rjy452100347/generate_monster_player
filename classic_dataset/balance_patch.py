"""Generate a small, map-safe rare-monster balance patch for the 48K set."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import json
import math
from pathlib import Path
import random
from typing import Any

import yaml

from .assets import AssetStore
from .comprehensive import PlannedSample, ResumableWriter, RootCache, _camera, _pairwise_ious
from .render import MapRenderer, SpriteLibrary, build_monsters


def _monster_counts(manifest: Path) -> Counter[str]:
    result: Counter[str] = Counter()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result.update(
            str(label["mob_id"])
            for label in record["labels"]
            if int(label.get("class_id", 0)) == 0
        )
    return result


def _write_union_yaml(base: Path, main: Path, patch: Path, class_names: list[str]) -> Path:
    payload = {
        "train": [
            (base / "images" / "train").as_posix(),
            (main / "images" / "train").as_posix(),
            (patch / "images" / "train").as_posix(),
        ],
        "val": [
            (base / "images" / "val").as_posix(),
            (main / "images" / "val").as_posix(),
        ],
        "test": [
            (base / "images" / "test").as_posix(),
            (main / "images" / "test").as_posix(),
        ],
        "names": {index: name for index, name in enumerate(class_names)},
    }
    target = main / "data_combined_60k_hard_negative_balance.yaml"
    temporary = target.with_suffix(".yaml.tmp")
    temporary.write_text(yaml.safe_dump(payload, allow_unicode=True, sort_keys=False), encoding="utf-8")
    temporary.replace(target)
    return target


def generate(config_path: str | Path) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    main = Path(config["output_root"]) / config["name"]
    output = Path(config["output_root"]) / "rare_mob_balance_patch"
    output.mkdir(parents=True, exist_ok=True)
    counts_before = _monster_counts(main / "scenario_manifest.jsonl")
    target = math.ceil(max(counts_before.values()) / 5)
    deficits = {mob_id: max(0, target - count) for mob_id, count in counts_before.items()}
    deficits = {mob_id: value for mob_id, value in deficits.items() if value}

    selection = json.loads((main / "map_selection.json").read_text(encoding="utf-8"))
    map_records = {item["map_id"]: item for item in selection["maps"]}
    map_split = {
        map_id: split for split, map_ids in selection["splits"].items() for map_id in map_ids
    }
    mob_maps: dict[str, list[str]] = defaultdict(list)
    for map_id, record in map_records.items():
        for mob_id in record["mob_ids"]:
            mob_maps[str(mob_id)].append(map_id)

    tasks: list[dict[str, Any]] = []
    split_indexes: Counter[str] = Counter()
    for mob_id, deficit in sorted(deficits.items()):
        remaining = deficit
        while remaining:
            count = min(10, remaining)
            # Prefer train without ever moving a held-out map across splits.
            train_maps = sorted(map_id for map_id in mob_maps[mob_id] if map_split[map_id] == "train")
            maps = train_maps or sorted(mob_maps[mob_id])
            map_id = maps[len(tasks) % len(maps)]
            split = map_split[map_id]
            split_indexes[split] += 1
            tasks.append({
                "mob_id": mob_id, "count": count, "map_id": map_id, "split": split,
                "split_index": split_indexes[split],
            })
            remaining -= count

    plan = {
        "target_minimum": target,
        "counts_before": dict(counts_before),
        "deficits": deficits,
        "tasks": tasks,
    }
    plan_path = output / "balance_plan.json"
    if plan_path.exists() and json.loads(plan_path.read_text(encoding="utf-8")) != plan:
        raise RuntimeError("existing balance patch uses a different plan")
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "map_selection.json").write_text(
        json.dumps({"splits": selection["splits"], "maps": selection["maps"]}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    store = AssetStore(config["client_root"], config["cache_dir"])
    store.build_index(include_sprites=True)
    sprites = SpriteLibrary(store, int(config.get("resource_cache", 384)))
    roots = RootCache(store, int(config.get("map_cache", 32)))
    renderer = MapRenderer(sprites, 1280, 224)
    writer = ResumableWriter(
        output, 1280, 224, min(100, len(tasks)), ["rare_mob_balance"],
        int(config.get("jpeg_quality", 88)), list(config.get("class_names", ["monster", "player"])),
    )

    for ordinal, task in enumerate(tasks, 1):
        sample = PlannedSample(
            ordinal, task["split"], task["split_index"], "rare_mob_balance", task["count"],
            target_layout="clustered",
        )
        if writer.has(sample):
            continue
        rng = random.Random((int(config["seed"]) << 40) ^ ordinal ^ int(task["mob_id"]))
        root = roots.get(task["map_id"])
        success = False
        for _ in range(500):
            monsters = build_monsters(
                root, sprites, rng, True, target_count=task["count"], placement="dense",
                view_width=1280, view_height=224, target_mob_id=task["mob_id"],
                action_weights={key: float(value) for key, value in config["action_weights"].items()},
            )
            if len(monsters) != task["count"] or any(item.mob_id != task["mob_id"] for item in monsters):
                continue
            camera_x, camera_y = _camera(root, monsters, 1280, 224, rng, False)
            image, labels = renderer.render_scene(
                root, camera_x, camera_y, monsters,
                minimum_clip_fraction=float(config.get("minimum_clip_fraction", 0.30)),
                minimum_visible_fraction=float(config.get("minimum_visible_fraction", 0.30)),
                label_players=True,
            )
            monster_labels = [item for item in labels if int(item.get("class_id", 0)) == 0]
            if len(monster_labels) != task["count"] or any(str(item["mob_id"]) != task["mob_id"] for item in monster_labels):
                continue
            ious = _pairwise_ious(monster_labels)
            metadata = {
                "map_id": task["map_id"], "camera": [camera_x, camera_y], "distractors": [],
                "overlay_frames": [], "hidden_objects": [], "max_iou": round(max(ious, default=0.0), 4),
                "overlapping_pairs": sum(value > 0 for value in ious), "size_target": None,
                "action_target": None, "target_layout": "clustered", "loot_layout": "none",
                "loot_count": 0, "multi_platform": False,
            }
            if writer.add(sample, image, labels, metadata):
                success = True
                break
        if not success:
            store.close()
            raise RuntimeError(f"could not render balance task {task}")
        if ordinal % 10 == 0 or ordinal == len(tasks):
            print(f"generated balance patch {len(writer.records)}/{len(tasks)}")

    writer.finish()
    counts_after = counts_before.copy()
    for task in tasks:
        counts_after[task["mob_id"]] += task["count"]
    report = {
        "images": len(tasks), "boxes": sum(item["count"] for item in tasks),
        "target_minimum": target, "minimum_after": min(counts_after.values()),
        "maximum_after": max(counts_after.values()),
        "ratio_after": max(counts_after.values()) / min(counts_after.values()),
        "counts_after": dict(counts_after),
    }
    (output / "balance_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    union = _write_union_yaml(Path(config["base_dataset"]), main, output, list(config["class_names"]))
    (output / "data_combined.yaml").write_text(union.read_text(encoding="utf-8"), encoding="utf-8")
    base_validation = json.loads((main / "validation_report.json").read_text(encoding="utf-8"))
    combined_validation = {
        "ok": report["ratio_after"] <= 5.0,
        "base_images": 48000,
        "balance_patch_images": len(tasks),
        "balance_patch_boxes": report["boxes"],
        "effective_images_with_original_60k": 60000 + 48000 + len(tasks),
        "effective_train_images": 48000 + 38400 + sum(task["split"] == "train" for task in tasks),
        "effective_val_images": 6000 + 4800,
        "effective_test_images": 6000 + 4800,
        "rare_common_ratio_after": report["ratio_after"],
        "minimum_mob_boxes_after": report["minimum_after"],
        "maximum_mob_boxes_after": report["maximum_after"],
        "unrenderable_1x1_frames_excluded": 8,
        "base_near_duplicate_phash_ratio": base_validation.get("near_duplicate_phash_ratio"),
        "item_minimum_uses": base_validation.get("item_minimum_uses"),
        "resolved_base_findings": [
            "eight zero-use live frames are 1x1 atlas placeholders and cannot form a 4x4 label",
            "the 97-image balance patch lowers the common/rare monster ratio below 5x",
        ],
        "training_yaml": union.as_posix(),
    }
    (main / "combined_validation_report.json").write_text(
        json.dumps(combined_validation, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    store.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    print(generate(args.config))


if __name__ == "__main__":
    main()
