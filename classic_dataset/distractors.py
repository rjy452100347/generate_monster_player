"""Native client distractors for hard-negative and combat-clutter scenes."""

from __future__ import annotations

from collections import Counter, defaultdict
import random
import re
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont

from .render import LogicalFrame, PlacedMonster, PlacedOverlay, SpriteLibrary, foothold_segments, platform_y


SHEET_NAME = re.compile(r"/([^/]+)\.wzspritesheet$")


def _compose(frames: Iterable[LogicalFrame], required_paths: set[str] | None = None) -> LogicalFrame | None:
    parts = list(frames)
    if not parts:
        return None
    if required_paths and not required_paths.issubset({frame.path for frame in parts}):
        return None
    body = next((frame for frame in parts if frame.z == "body" or frame.path.endswith("/body")), parts[0])
    placements: list[tuple[LogicalFrame, int, int]] = [(body, 0, 0)]
    global_anchors = dict(body.anchors or {})
    pending = [frame for frame in parts if frame is not body]
    while pending:
        changed = False
        for frame in list(pending):
            anchors = frame.anchors or {}
            common = next((name for name in anchors if name in global_anchors), None)
            if common is None:
                continue
            reference_x = global_anchors[common][0] - anchors[common][0]
            reference_y = global_anchors[common][1] - anchors[common][1]
            placements.append((frame, reference_x, reference_y))
            for name, point in anchors.items():
                global_anchors.setdefault(name, (reference_x + point[0], reference_y + point[1]))
            pending.remove(frame)
            changed = True
        if not changed:
            break
    if required_paths and any(frame.path in required_paths for frame in pending):
        return None
    left = min(reference_x - frame.origin_x for frame, reference_x, _ in placements)
    top = min(reference_y - frame.origin_y for frame, _, reference_y in placements)
    right = max(reference_x - frame.origin_x + frame.image.width for frame, reference_x, _ in placements)
    bottom = max(reference_y - frame.origin_y + frame.image.height for frame, _, reference_y in placements)
    if right <= left or bottom <= top or right - left > 512 or bottom - top > 512:
        return None
    canvas = Image.new("RGBA", (right - left, bottom - top), (0, 0, 0, 0))
    z_order = {
        "capeBelowBody": 0, "backHairBelowCap": 1, "body": 10, "pants": 15, "shoes": 16,
        "coat": 20, "armBelowHead": 25, "arm": 30, "lHand": 31, "rHand": 31,
        "weaponBelowArm": 32, "weapon": 35, "head": 40, "face": 45, "hair": 50,
        "hairOverHead": 51, "cap": 60, "capOverHair": 61,
    }
    for frame, reference_x, reference_y in sorted(
        placements, key=lambda item: (z_order.get(item[0].z, 30), item[0].path)
    ):
        canvas.alpha_composite(frame.image, (reference_x - frame.origin_x - left, reference_y - frame.origin_y - top))
    return LogicalFrame("composite", canvas, -left, -top)


class DistractorLibrary:
    def __init__(self, sprites: SpriteLibrary):
        self.sprites = sprites
        index = sprites.store.index
        self.pet_names = self._names(index, "/Item/Pet/")
        self.skill_names = self._names(index, "/Skill/")
        self.consume_names = self._names(index, "/Item/Consume/")
        self.etc_names = self._names(index, "/Item/Etc/")
        self.body_names = [name for name in self._names(index, "/Character/") if name.startswith("00002")]
        self.head_names = [name for name in self._names(index, "/Character/") if name.startswith("00012")]
        self.face_names = self._names(index, "/Character/Face/")[:64]
        self.hair_names = self._names(index, "/Character/Hair/")[:128]
        self.equipment = {
            "/Character/Coat/": self._names(index, "/Character/Coat/")[:64],
            "/Character/Pants/": self._names(index, "/Character/Pants/")[:64],
            "/Character/Shoes/": self._names(index, "/Character/Shoes/")[:64],
            "/Character/Cap/": self._names(index, "/Character/Cap/")[:64],
        }
        self._player_poses: list[LogicalFrame] = []

    @staticmethod
    def _names(index: dict[str, Any], group: str) -> list[str]:
        result = []
        marker = f"/SpriteSheet/CN{group}"
        for key in index:
            if marker in key and key.endswith(".wzspritesheet"):
                match = SHEET_NAME.search(key)
                if match:
                    result.append(match.group(1))
        return sorted(set(result))

    def _build_player(self, rng: random.Random) -> LogicalFrame | None:
        allowed = ("stand1", "stand2", "walk1", "walk2", "jump", "alert")
        skin_index = rng.randrange(min(len(self.body_names), len(self.head_names)))
        body = self.sprites.resource(self.body_names[skin_index], "/Character/")
        pose_keys = sorted({
            "/".join(path.split("/")[:2]) for path in body
            if len(path.split("/")) >= 3 and path.split("/", 1)[0].lower().startswith(allowed)
            and path.endswith("/body")
        })
        if not pose_keys:
            return None
        key = rng.choice(pose_keys)
        parts = [frame for path, frame in body.items() if path.startswith(key + "/") and frame.anchors]
        head = self.sprites.resource(self.head_names[skin_index], "/Character/")
        if key + "/head" in head:
            parts.append(head[key + "/head"])
        if self.face_names:
            face = self.sprites.resource(rng.choice(self.face_names), "/Character/Face/")
            candidate = face.get("default/face") or next((frame for path, frame in face.items() if path.endswith("/face")), None)
            if candidate:
                parts.append(candidate)
        if self.hair_names:
            hair = self.sprites.resource(rng.choice(self.hair_names), "/Character/Hair/")
            parts.extend(frame for path, frame in hair.items() if path.startswith(key + "/") and frame.anchors)
        for group, names in self.equipment.items():
            if not names or rng.random() < 0.20:
                continue
            equipment = self.sprites.resource(rng.choice(names), group)
            parts.extend(frame for path, frame in equipment.items() if path.startswith(key + "/") and frame.anchors)
        pose = _compose(parts)
        if not pose or not pose.image.getchannel("A").getbbox():
            return None
        return LogicalFrame(key, pose.image, pose.origin_x, pose.origin_y)

    def player_frame(self, rng: random.Random) -> LogicalFrame | None:
        if len(self._player_poses) >= 384:
            return rng.choice(self._player_poses)
        for _ in range(12):
            pose = self._build_player(rng)
            if pose:
                self._player_poses.append(pose)
                return pose
        return rng.choice(self._player_poses) if self._player_poses else None

    def pet_frame(self, rng: random.Random) -> LogicalFrame | None:
        result = self.pet_visual(rng)
        return result[1] if result else None

    def pet_visual(self, rng: random.Random) -> tuple[str, LogicalFrame] | None:
        for _ in range(12):
            if not self.pet_names:
                break
            name = rng.choice(self.pet_names)
            frames = self.sprites.resource(name, "/Item/Pet/")
            candidates = [
                frame for path, frame in frames.items()
                if path.lower().startswith(("stand", "move", "jump", "fly"))
                and frame.image.getchannel("A").getbbox()
            ]
            if candidates:
                return name, rng.choice(candidates)
        return None

    def skill_frame(self, rng: random.Random) -> LogicalFrame | None:
        for _ in range(20):
            if not self.skill_names:
                break
            frames = self.sprites.resource(rng.choice(self.skill_names), "/Skill/")
            candidates = []
            for path, frame in frames.items():
                lower = path.lower()
                box = frame.image.getchannel("A").getbbox()
                if not box or "icon" in lower or not any(word in lower for word in ("effect", "hit", "ball", "screen")):
                    continue
                width, height = box[2] - box[0], box[3] - box[1]
                if 8 <= width <= 500 and 8 <= height <= 300:
                    candidates.append(frame)
            if candidates:
                return rng.choice(candidates)
        return None

    def loot_frame(self, rng: random.Random) -> LogicalFrame | None:
        groups = [(self.consume_names, "/Item/Consume/"), (self.etc_names, "/Item/Etc/")]
        for _ in range(16):
            names, group = rng.choice(groups)
            if not names:
                continue
            frames = self.sprites.resource(rng.choice(names), group)
            candidates = [
                frame for path, frame in frames.items()
                if path.lower().endswith(("info/iconraw", "info/icon", "/iconraw", "/icon"))
                and frame.image.getchannel("A").getbbox()
            ]
            if candidates:
                return rng.choice(candidates)
        return None

    @staticmethod
    def damage_frame(rng: random.Random) -> LogicalFrame:
        text = str(rng.randint(10, 99999))
        font = ImageFont.truetype("arialbd.ttf", rng.randint(16, 24))
        probe = Image.new("RGBA", (1, 1))
        box = ImageDraw.Draw(probe).textbbox((0, 0), text, font=font, stroke_width=2)
        image = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 4), (0, 0, 0, 0))
        color = rng.choice(((255, 188, 32, 255), (255, 92, 92, 255), (255, 255, 255, 255)))
        ImageDraw.Draw(image).text((2 - box[0], 2 - box[1]), text, font=font, fill=color, stroke_width=2, stroke_fill=(40, 20, 20, 255))
        return LogicalFrame("damage", image, image.width // 2, image.height)

    @staticmethod
    def ground_points(root: dict[str, Any]) -> list[tuple[int, int]]:
        return [(x, y) for x, y, _ in DistractorLibrary.ground_locations(root)]

    @staticmethod
    def ground_locations(root: dict[str, Any]) -> list[tuple[int, int, int]]:
        footholds = foothold_segments(root)
        points = []
        for life in root.get("life", {}).values():
            if not isinstance(life, dict):
                continue
            x = int(life.get("x", 0))
            foothold_id = int(life.get("fh", -1))
            segment = footholds.get(foothold_id)
            y = platform_y(segment, x) if segment else int(life.get("cy", life.get("y", 0)))
            points.append((x, y, foothold_id))
        for foothold_id, segment in footholds.items():
            if segment[0] == segment[2]:
                continue
            x = round((segment[0] + segment[2]) / 2)
            points.append((x, platform_y(segment, x), foothold_id))
        return list(dict.fromkeys(points))

    def death_overlays(
        self, root: dict[str, Any], rng: random.Random, count: int | None = None,
        frame_counts: Counter[tuple[str, str]] | None = None,
    ) -> list[PlacedOverlay]:
        result = []
        lives = [value for value in root.get("life", {}).values() if isinstance(value, dict) and value.get("type") == "m"]
        target = count or rng.randint(1, 5)
        footholds = foothold_segments(root)
        candidates = []
        for life in lives:
            mob_id = str(life.get("id", ""))
            for frame in self.sprites.monster_death_frames(mob_id):
                candidates.append((life, mob_id, frame))
        if not candidates:
            return []
        local_counts = Counter(frame_counts or {})
        for _ in range(target):
            minimum = min(local_counts[(mob_id, frame.path)] for _, mob_id, frame in candidates)
            pool = [item for item in candidates if local_counts[(item[1], item[2].path)] <= minimum]
            life, mob_id, frame = rng.choice(pool)
            x = int(life.get("x", 0))
            segment = footholds.get(int(life.get("fh", -1)))
            if segment and segment[0] != segment[2]:
                low, high = sorted((segment[0], segment[2]))
                x = rng.randint(low, high)
            y = platform_y(segment, x) if segment else int(life.get("cy", life.get("y", 0)))
            result.append(PlacedOverlay(
                frame.image, x, y, frame.origin_x, frame.origin_y, "death",
                bool(rng.getrandbits(1)), mob_id, frame.path,
            ))
            local_counts[(mob_id, frame.path)] += 1
        return result

    def make_overlays(
        self, root: dict[str, Any], monsters: list[PlacedMonster], rng: random.Random,
        kinds: Iterable[str], force_target_overlap: bool = False,
        loot_count: int = 0, loot_layout: str = "scattered", split: str = "train",
        item_sampler: Any | None = None,
        viewport: tuple[int, int, int, int] | None = None,
    ) -> tuple[list[PlacedOverlay], list[str]]:
        raw_requested = list(kinds)
        requested = [kind for kind in raw_requested if kind != "loot"]
        if "loot" in raw_requested and item_sampler is None:
            requested.append("loot")
        locations = self.ground_locations(root)
        if not locations:
            return [], []
        overlays: list[PlacedOverlay] = []
        emitted: list[str] = []
        footholds = foothold_segments(root)

        monster_points = [(item.world_x, item.world_y, item.foothold_id) for item in monsters]
        scene_anchor = rng.choice(monster_points or locations)
        if viewport:
            view_x, view_y, view_width, view_height = viewport
            visible_locations = [
                item for item in locations
                if view_x + 32 <= item[0] <= view_x + view_width - 32
                and view_y + 12 <= item[1] <= view_y + view_height - 2
            ]
        else:
            visible_locations = locations
        scene_locations = [
            item for item in visible_locations
            if abs(item[0] - scene_anchor[0]) <= 350 and abs(item[1] - scene_anchor[1]) <= 70
        ] or visible_locations or [scene_anchor]

        def clamp_frame_x(frame: LogicalFrame, x: int) -> int:
            if not viewport:
                return x
            box = frame.image.getchannel("A").getbbox()
            if not box:
                return x
            minimum = view_x + 2 + frame.origin_x - box[0]
            maximum = view_x + view_width - 2 + frame.origin_x - box[2]
            return max(minimum, min(maximum, x)) if minimum <= maximum else x

        def point(prefer_monster: bool = False) -> tuple[int, int, int]:
            if prefer_monster and monsters:
                item = rng.choice(monsters)
                return item.world_x, item.world_y, item.foothold_id
            return rng.choice(scene_locations)

        for kind in requested:
            frame: LogicalFrame | None = None
            source_id = kind
            x, y, foothold_id = point(kind in {"skill", "damage"} and force_target_overlap)
            if kind == "player":
                frame = self.player_frame(rng)
                x += rng.randint(-140, 140)
            elif kind == "pet":
                pet = self.pet_visual(rng)
                if pet:
                    source_id, frame = pet
                x += rng.choice((-1, 1)) * rng.randint(25, 70)
            elif kind == "skill":
                frame = self.skill_frame(rng)
            elif kind == "damage":
                frame = self.damage_frame(rng)
                y -= rng.randint(35, 100)
            elif kind == "loot":
                frame = self.loot_frame(rng)
                if frame:
                    frame = LogicalFrame(frame.path, frame.image, frame.image.width // 2, frame.image.height)
                x += rng.randint(-90, 90)
            if frame:
                x = clamp_frame_x(frame, x)
                overlays.append(PlacedOverlay(
                    frame.image, x, y, frame.origin_x, frame.origin_y, kind,
                    bool(rng.getrandbits(1)), source_id, frame.path,
                    resource_path=f"Item/Pet/{source_id}:{frame.path}" if kind == "pet" else frame.path,
                    state=frame.path.split("/", 1)[0], foothold_id=foothold_id,
                ))
                emitted.append(kind)

        if loot_count and item_sampler is not None:
            actors = monster_points + [
                (item.world_x, item.world_y, item.foothold_id) for item in overlays if item.kind == "player"
            ]
            pile_count = {
                "scattered": loot_count,
                "small_piles": max(1, round(loot_count / rng.randint(2, 5))),
                "multi_pile": min(max(2, round(loot_count / 4)), 4),
                "dense_stack": 1,
            }.get(loot_layout, loot_count)
            centers: list[tuple[int, int, int]] = []
            for _ in range(pile_count):
                if actors and rng.random() < 0.50:
                    centers.append(rng.choice(actors))
                else:
                    centers.append(rng.choice(scene_locations))
            used_positions: set[tuple[int, int]] = set()
            for item_index in range(loot_count):
                visual = item_sampler.choose(split, high_risk=(item_index == 0 and rng.random() < 0.35))
                frame = item_sampler.frame(self.sprites, visual)
                if frame is None:
                    item_sampler.rollback(split, [visual.visual_id])
                    continue
                base_x, _, foothold_id = centers[item_index % len(centers)]
                segment = footholds.get(foothold_id)
                spread = {"scattered": 120, "small_piles": 24, "multi_pile": 18, "dense_stack": 10}.get(loot_layout, 90)
                x = base_x + rng.randint(-spread, spread)
                if segment and segment[0] != segment[2]:
                    low, high = sorted((segment[0], segment[2]))
                    if viewport:
                        low, high = max(low, view_x + 32), min(high, view_x + view_width - 32)
                    if low > high:
                        alternate = rng.choice(scene_locations)
                        x, y, foothold_id = alternate
                        segment = footholds.get(foothold_id)
                    else:
                        x = max(low, min(high, x))
                        y = platform_y(segment, x)
                else:
                    _, y, foothold_id = min(scene_locations, key=lambda item: abs(item[0] - x))
                for position_attempt in range(50):
                    if (x, y) not in used_positions:
                        break
                    if position_attempt and position_attempt % 5 == 0:
                        base_x, _, foothold_id = rng.choice(scene_locations)
                        segment = footholds.get(foothold_id)
                    candidate_x = base_x + rng.randint(-spread - position_attempt, spread + position_attempt)
                    if segment and segment[0] != segment[2]:
                        low, high = sorted((segment[0], segment[2]))
                        if viewport:
                            low, high = max(low, view_x + 32), min(high, view_x + view_width - 32)
                        if low <= high:
                            candidate_x = max(low, min(high, candidate_x))
                            candidate_y = platform_y(segment, candidate_x)
                        else:
                            alternate = rng.choice(scene_locations)
                            candidate_x, candidate_y, foothold_id = alternate
                            segment = footholds.get(foothold_id)
                    else:
                        alternate = rng.choice(scene_locations)
                        candidate_x, candidate_y, foothold_id = alternate
                        segment = footholds.get(foothold_id)
                    x, y = candidate_x, candidate_y
                if (x, y) in used_positions:
                    item_sampler.rollback(split, [visual.visual_id])
                    continue
                used_positions.add((x, y))
                item_id = rng.choice(visual.item_ids)
                state = "animated" if frame.path.split("/")[-1].isdigit() else "landed"
                overlays.append(PlacedOverlay(
                    frame.image, x, y, frame.image.width // 2, frame.image.height, "loot", False,
                    item_id, frame.path, item_id=item_id, visual_id=visual.visual_id,
                    resource_path=visual.resource_path, state=state, foothold_id=foothold_id,
                ))
                emitted.append("loot")
        return overlays, emitted
