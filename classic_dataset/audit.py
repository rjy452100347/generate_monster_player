"""Audit client-wide map, monster, action and native sprite-size coverage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
from statistics import median
from typing import Any

import yaml

from .assets import AssetStore
from .render import SpriteLibrary, foothold_segments


MAP_PATTERN = re.compile(r"/(\d{9})\.wzjson$")


def _percentile(values: list[int], quantile: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * quantile)]


def _in_ranges(map_id: str, ranges: list[list[int]]) -> bool:
    value = int(map_id)
    return any(int(low) <= value <= int(high) for low, high in ranges)


def _drawable_count(root: dict[str, Any]) -> int:
    count = len(root.get("back", {}))
    for layer_index in range(8):
        layer = root.get(str(layer_index), {})
        count += len(layer.get("tile", {})) + len(layer.get("obj", {}))
    return count


def audit(config_path: str | Path) -> Path:
    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    output = Path(config["output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    store = AssetStore(config["client_root"], config["cache_dir"])
    store.build_index(include_sprites=True)
    sprites = SpriteLibrary(store)
    ranges = config.get("map_ranges", [[0, 899_999_999]])
    excluded_prefixes = tuple(str(value) for value in config.get("exclude_map_prefixes", ["9"]))
    available_mobs = {
        key.rsplit("/", 1)[-1].removesuffix(".wzspritesheet")
        for key in store.index
        if "/Mob/" in key and key.endswith(".wzspritesheet")
    }

    maps: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    referenced_mobs: Counter[str] = Counter()
    region_counts: Counter[str] = Counter()
    for key in sorted(store.index):
        match = MAP_PATTERN.search(key)
        if not match:
            continue
        map_id = match.group(1)
        if not _in_ranges(map_id, ranges) or map_id.startswith(excluded_prefixes):
            continue
        try:
            root = next(iter(store.read_wzjson(map_id, "/Map/Map/").values()))
            lives = [value for value in root.get("life", {}).values() if isinstance(value, dict)]
            valid_mobs = [
                str(value.get("id", ""))
                for value in lives
                if value.get("type") == "m"
                and str(value.get("id", "")) in available_mobs
                and (not str(value.get("id", "")).isdigit() or int(value["id"]) < 9_000_000)
            ]
            npc_count = sum(value.get("type") == "n" for value in lives)
            footholds = foothold_segments(root)
            drawables = _drawable_count(root)
            info = root.get("info", {})
            record = {
                "map_id": map_id,
                "region": map_id[:3],
                "positive": bool(valid_mobs and footholds),
                "negative_eligible": bool(not valid_mobs and drawables),
                "spawn_count": len(valid_mobs),
                "mob_ids": sorted(set(valid_mobs)),
                "npc_count": npc_count,
                "foothold_count": len(footholds),
                "drawable_count": drawables,
                "field_type": info.get("fieldType", 0),
                "time_limit": info.get("timeLimit", 0),
            }
            maps.append(record)
            if record["positive"]:
                referenced_mobs.update(valid_mobs)
                region_counts[record["region"]] += 1
        except Exception as exc:
            failures.append({"map_id": map_id, "error": f"{type(exc).__name__}: {exc}"})

    mob_records: list[dict[str, Any]] = []
    action_totals: Counter[str] = Counter()
    all_widths: list[int] = []
    all_heights: list[int] = []
    for mob_id, spawn_references in sorted(referenced_mobs.items()):
        frames = sprites.monster_frames(mob_id)
        actions: Counter[str] = Counter()
        widths: list[int] = []
        heights: list[int] = []
        for frame in frames:
            action = frame.path.split("/", 1)[0].lower()
            actions[action] += 1
            alpha_box = frame.image.getchannel("A").getbbox()
            if alpha_box:
                widths.append(alpha_box[2] - alpha_box[0])
                heights.append(alpha_box[3] - alpha_box[1])
        action_totals.update(actions)
        all_widths.extend(widths)
        all_heights.extend(heights)
        mob_records.append(
            {
                "mob_id": mob_id,
                "spawn_references": spawn_references,
                "frame_count": len(frames),
                "actions": dict(sorted(actions.items())),
                "width_min": min(widths, default=0),
                "width_median": round(median(widths)) if widths else 0,
                "width_max": max(widths, default=0),
                "height_min": min(heights, default=0),
                "height_median": round(median(heights)) if heights else 0,
                "height_max": max(heights, default=0),
            }
        )

    positive_maps = [record for record in maps if record["positive"]]
    negative_maps = [record for record in maps if record["negative_eligible"]]
    payload = {
        "client_root": str(config["client_root"]),
        "catalog_version": "1.14.2",
        "filters": {"map_ranges": ranges, "exclude_map_prefixes": list(excluded_prefixes)},
        "summary": {
            "indexed_map_jsons": sum(bool(MAP_PATTERN.search(key)) for key in store.index),
            "audited_maps": len(maps),
            "positive_maps": len(positive_maps),
            "negative_maps": len(negative_maps),
            "referenced_monsters": len(referenced_mobs),
            "available_monster_sheets": len(available_mobs),
            "map_failures": len(failures),
            "missing_sprite_resources": len(sprites.missing),
            "regions": dict(sorted(region_counts.items())),
            "action_frames": dict(sorted(action_totals.items())),
            "native_width_percentiles": {
                "p05": _percentile(all_widths, 0.05),
                "p25": _percentile(all_widths, 0.25),
                "p50": _percentile(all_widths, 0.50),
                "p75": _percentile(all_widths, 0.75),
                "p95": _percentile(all_widths, 0.95),
            },
            "native_height_percentiles": {
                "p05": _percentile(all_heights, 0.05),
                "p25": _percentile(all_heights, 0.25),
                "p50": _percentile(all_heights, 0.50),
                "p75": _percentile(all_heights, 0.75),
                "p95": _percentile(all_heights, 0.95),
            },
        },
        "maps": maps,
        "monsters": mob_records,
        "failures": failures,
        "missing_assets": sprites.missing,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    store.close()
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    output = audit(args.config)
    data = json.loads(output.read_text(encoding="utf-8"))
    print(json.dumps(data["summary"], ensure_ascii=False, indent=2))
    print(output)


if __name__ == "__main__":
    main()
