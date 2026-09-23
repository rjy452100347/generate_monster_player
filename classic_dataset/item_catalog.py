"""Build and sample a visual catalog of client-native dropped-item sprites."""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import heapq
import json
from pathlib import Path
import re
from typing import Any

import numpy as np

from .render import LogicalFrame, SpriteLibrary


SHEET_NAME = re.compile(r"/([^/]+)\.wzspritesheet$")
ITEM_GROUPS = ("Consume", "Etc", "Install", "Special", "Cash")
EQUIP_GROUPS = (
    "Weapon", "Cap", "Longcoat", "Coat", "Pants", "Shoes", "Glove",
    "Shield", "Cape", "Accessory", "Ring", "PetEquip", "TamingMob",
)


@dataclass
class ItemVisual:
    visual_id: str
    category: str
    categories: list[str]
    item_ids: list[str]
    resource_name: str
    resource_group: str
    frame_path: str
    resource_path: str
    width: int
    height: int
    alpha_pixels: int
    saturated_fraction: float
    high_risk: bool
    challenge: bool = False


def _sheet_names(index: dict[str, Any], marker: str) -> list[str]:
    result = []
    for key in index:
        if marker in key and key.endswith(".wzspritesheet"):
            match = SHEET_NAME.search(key)
            if match:
                result.append(match.group(1))
    return sorted(set(result))


def _item_id(frame_path: str, fallback: str) -> str:
    first = frame_path.split("/", 1)[0]
    return first if first.isdigit() and len(first) >= 7 else fallback


def _icon_candidates(frames: dict[str, LogicalFrame], fallback_id: str) -> list[tuple[str, str, LogicalFrame]]:
    by_item: dict[str, list[tuple[str, LogicalFrame]]] = {}
    for path, frame in frames.items():
        lower = path.lower()
        parts = lower.split("/")
        if "iconraw" not in parts and "icon" not in parts:
            continue
        if not frame.image.getchannel("A").getbbox():
            continue
        by_item.setdefault(_item_id(path, fallback_id), []).append((path, frame))
    result = []
    for item_id, values in by_item.items():
        raw = [(path, frame) for path, frame in values if "iconraw" in path.lower().split("/")]
        selected = raw or [(path, frame) for path, frame in values if "icon" in path.lower().split("/")]
        # Animated iconRaw resources keep every real client frame. Ordinary
        # inventory entries contribute one frame.
        if any(path.lower().split("/")[-1].isdigit() for path, _ in selected):
            chosen = selected
        else:
            chosen = selected[:1]
        result.extend((item_id, path, frame) for path, frame in chosen)
    return result


def _visual_digest(frame: LogicalFrame) -> tuple[str, int, int, int, float]:
    box = frame.image.getchannel("A").getbbox()
    if not box:
        raise ValueError("empty item frame")
    cropped = frame.image.crop(box).convert("RGBA")
    array = np.asarray(cropped, dtype=np.uint8)
    alpha = array[..., 3] > 0
    alpha_pixels = int(alpha.sum())
    rgb = array[..., :3].astype(np.float32) / 255.0
    maximum, minimum = rgb.max(axis=2), rgb.min(axis=2)
    saturation = np.divide(maximum - minimum, maximum, out=np.zeros_like(maximum), where=maximum > 0)
    saturated_fraction = float((saturation[alpha] >= 0.55).mean()) if alpha_pixels else 0.0
    payload = cropped.width.to_bytes(2, "little") + cropped.height.to_bytes(2, "little") + cropped.tobytes()
    return hashlib.sha256(payload).hexdigest(), cropped.width, cropped.height, alpha_pixels, saturated_fraction


def build_item_visual_index(sprites: SpriteLibrary, destination: str | Path, seed: int = 83) -> list[ItemVisual]:
    """Scan all supported item/equipment icons and atomically cache the result."""
    destination = Path(destination)
    if destination.exists():
        payload = json.loads(destination.read_text(encoding="utf-8"))
        if payload.get("schema_version") == 1:
            return [ItemVisual(**item) for item in payload["visuals"]]

    sources: list[tuple[str, str, str]] = []
    for category in ITEM_GROUPS:
        group = f"/Item/{category}/"
        marker = f"/SpriteSheet/CN/Item/{category}/"
        sources.extend((category.lower(), name, group) for name in _sheet_names(sprites.store.index, marker))
    for category in EQUIP_GROUPS:
        group = f"/Character/{category}/"
        marker = f"/SpriteSheet/CN/Character/{category}/"
        sources.extend((f"equip_{category.lower()}", name, group) for name in _sheet_names(sprites.store.index, marker))

    merged: dict[str, ItemVisual] = {}
    failures: list[dict[str, str]] = []
    for index, (category, name, group) in enumerate(sources, 1):
        frames = sprites.resource(name, group)
        if not frames:
            failures.append({"category": category, "resource_name": name, "resource_group": group})
            continue
        for item_id, path, frame in _icon_candidates(frames, name):
            try:
                digest, width, height, alpha_pixels, saturation = _visual_digest(frame)
            except ValueError:
                continue
            if width < 2 or height < 2 or width > 256 or height > 256 or alpha_pixels < 4:
                continue
            risk = saturation >= 0.55 and 0.55 <= width / max(1, height) <= 1.8 and max(width, height) >= 20
            resource_path = f"{group.strip('/')}/{name}:{path}"
            if digest not in merged:
                merged[digest] = ItemVisual(
                    digest, category, [category], [item_id], name, group, path,
                    resource_path, width, height, alpha_pixels, round(saturation, 4), risk,
                )
            else:
                record = merged[digest]
                if category not in record.categories:
                    record.categories.append(category)
                if item_id not in record.item_ids:
                    record.item_ids.append(item_id)
        if index % 250 == 0:
            print(f"indexed item resources {index}/{len(sources)}; unique visuals={len(merged)}")

    visuals = sorted(merged.values(), key=lambda item: item.visual_id)
    challenge_count = round(len(visuals) * 0.05)
    challenge_ids = {
        item.visual_id for item in sorted(
            visuals,
            key=lambda item: hashlib.sha256(f"{seed}:{item.visual_id}".encode()).digest(),
        )[:challenge_count]
    }
    for item in visuals:
        item.categories.sort()
        item.item_ids.sort()
        item.challenge = item.visual_id in challenge_ids
    payload = {
        "schema_version": 1,
        "seed": seed,
        "resource_count": len(sources),
        "visual_count": len(visuals),
        "challenge_count": challenge_count,
        "category_counts": dict(Counter(category for item in visuals for category in item.categories)),
        "failures": failures,
        "visuals": [asdict(item) for item in visuals],
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return visuals


class ItemCoverageSampler:
    """Lowest-use-first sampler whose state is reconstructed from manifest counts."""

    def __init__(self, visuals: list[ItemVisual], counts: Counter[tuple[str, str]] | None = None):
        self.visuals = {item.visual_id: item for item in visuals}
        self.counts = counts if counts is not None else Counter()
        self.heaps: dict[tuple[str, bool], list[tuple[int, str]]] = {}
        for split in ("train", "val", "test"):
            allowed = [item for item in visuals if split != "train" or not item.challenge]
            for high_risk in (False, True):
                pool = [item for item in allowed if not high_risk or item.high_risk]
                self.heaps[(split, high_risk)] = [
                    (self.counts[(split, item.visual_id)], item.visual_id) for item in pool
                ]
                heapq.heapify(self.heaps[(split, high_risk)])

    def choose(self, split: str, high_risk: bool = False) -> ItemVisual:
        heap = self.heaps[(split, high_risk)]
        while heap:
            count, visual_id = heapq.heappop(heap)
            if count != self.counts[(split, visual_id)]:
                continue
            self.counts[(split, visual_id)] += 1
            heapq.heappush(heap, (count + 1, visual_id))
            if high_risk:
                heapq.heappush(self.heaps[(split, False)], (count + 1, visual_id))
            elif self.visuals[visual_id].high_risk:
                heapq.heappush(self.heaps[(split, True)], (count + 1, visual_id))
            return self.visuals[visual_id]
        raise RuntimeError(f"no item visuals available for split={split}, high_risk={high_risk}")

    def rollback(self, split: str, visual_ids: list[str]) -> None:
        for visual_id in visual_ids:
            self.counts[(split, visual_id)] -= 1
            count = self.counts[(split, visual_id)]
            heapq.heappush(self.heaps[(split, False)], (count, visual_id))
            if self.visuals[visual_id].high_risk:
                heapq.heappush(self.heaps[(split, True)], (count, visual_id))

    def frame(self, sprites: SpriteLibrary, visual: ItemVisual) -> LogicalFrame | None:
        return sprites.resource(visual.resource_name, visual.resource_group).get(visual.frame_path)
