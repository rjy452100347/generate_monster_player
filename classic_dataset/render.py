"""Offline classic map and actor renderer."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import math
import posixpath
import random
from typing import Any, Iterable

from PIL import Image, ImageChops

from .assets import AssetStore
from .wzjs import SpriteValue


DEATH_WORDS = ("die", "dead", "death")


def _frame_alpha_area(frame: "LogicalFrame") -> int:
    box = frame.image.getchannel("A").getbbox()
    return (box[2] - box[0]) * (box[3] - box[1]) if box else 0


@dataclass
class LogicalFrame:
    path: str
    image: Image.Image
    origin_x: int
    origin_y: int
    anchors: dict[str, tuple[int, int]] | None = None
    z: str = ""


@dataclass
class PlacedMonster:
    mob_id: str
    image: Image.Image
    world_x: int
    world_y: int
    flip: bool
    action: str
    origin_x: int = 0
    origin_y: int = 0
    foothold_id: int = -1
    frame_path: str = ""


@dataclass
class PlacedOverlay:
    image: Image.Image
    world_x: int
    world_y: int
    origin_x: int
    origin_y: int
    kind: str
    flip: bool = False
    source_id: str = ""
    frame_path: str = ""
    item_id: str = ""
    visual_id: str = ""
    resource_path: str = ""
    state: str = "static"
    foothold_id: int = -1


class SpriteLibrary:
    def __init__(self, store: AssetStore, max_cached_resources: int = 384):
        self.store = store
        self.max_cached_resources = max(32, max_cached_resources)
        self._resources: OrderedDict[tuple[str, str], dict[str, LogicalFrame]] = OrderedDict()
        self.missing: list[dict[str, str]] = []
        self._monster_ids = {
            key.rsplit("/", 1)[-1].removesuffix(".wzspritesheet")
            for key in store.index if "/Mob/" in key and key.endswith(".wzspritesheet")
        }

    def resource(self, name: str, group: str) -> dict[str, LogicalFrame]:
        cache_key = (group, name)
        if cache_key in self._resources:
            self._resources.move_to_end(cache_key)
            return self._resources[cache_key]
        try:
            document = self.store.read_wz_document(name, group)
            # Type 12 is a WZ UOL/string alias. It occupies a logical WZSS
            # frame slot just like a direct type-18 sprite node.
            sprite_nodes = [(i, node) for i, node in enumerate(document.nodes) if node.type_code in (12, 18)]
            sheet, textures = self.store.read_spritesheet(name, len(sprite_nodes), group.replace("/Json/", "/SpriteSheet/CN/"))
            frames: dict[str, LogicalFrame] = {}
            aliases: list[tuple[str, str]] = []
            for node_index, node in sprite_nodes:
                path = document.paths[node.path_index]
                if path.rsplit('/', 1)[-1].startswith('$'):
                    continue
                full_path = f"{group.strip('/')}/{name}/{path}"
                try:
                    atlas_index = sheet.index_for_path(full_path)
                except ValueError as exc:
                    self.missing.append({'asset': name, 'group': group, 'error': str(exc)})
                    continue
                value = document._value(node_index)
                origin = value.children.get("origin", {}) if isinstance(value, SpriteValue) else {}
                raw_anchors = value.children.get("map", {}) if isinstance(value, SpriteValue) else {}
                atlas_frame = sheet.resolved_frame(atlas_index)
                frames[path] = LogicalFrame(
                    path=path,
                    image=sheet.crop(atlas_index, textures),
                    origin_x=round(atlas_frame.pivot_x * atlas_frame.width) if atlas_frame.pivot_x is not None else round(float(origin.get("x", 0))),
                    origin_y=round((1 - atlas_frame.pivot_y) * atlas_frame.height) if atlas_frame.pivot_y is not None else round(float(origin.get("y", 0))),
                    anchors={
                        key: (round(float(point.get("x", 0))), round(float(point.get("y", 0))))
                        for key, point in raw_anchors.items() if isinstance(point, dict)
                    },
                    z=str(value.children.get("z", "")) if isinstance(value, SpriteValue) else "",
                )
                if node.type_code == 12 and isinstance(value, str):
                    aliases.append((path, posixpath.normpath(posixpath.join(posixpath.dirname(path), value))))
            unresolved = dict(aliases)
            for _ in range(len(unresolved) + 1):
                changed = False
                for path, target in list(unresolved.items()):
                    if target in frames and target not in unresolved:
                        source = frames[target]
                        frames[path] = LogicalFrame(
                            path, source.image, source.origin_x, source.origin_y,
                            dict(source.anchors or {}), source.z,
                        )
                        unresolved.pop(path)
                        changed = True
                if not changed:
                    break
            self._remember(cache_key, frames)
            return frames
        except Exception as exc:
            self.missing.append({"asset": name, "group": group, "error": f"{type(exc).__name__}: {exc}"})
            self._remember(cache_key, {})
            return {}

    def _remember(self, key: tuple[str, str], frames: dict[str, LogicalFrame]) -> None:
        self._resources[key] = frames
        self._resources.move_to_end(key)
        while len(self._resources) > self.max_cached_resources:
            self._resources.popitem(last=False)

    def monster_frames(self, mob_id: str) -> list[LogicalFrame]:
        if mob_id not in self._monster_ids or (mob_id.isdigit() and int(mob_id) >= 9_000_000):
            return []
        frames = self.resource(mob_id, "/Mob/")
        allowed = ("stand", "move", "hit", "attack", "skill", "jump", "fly")
        return [
            frame for path, frame in frames.items()
            if path.split("/", 1)[0].lower().startswith(allowed)
            and not any(word in path.lower() for word in DEATH_WORDS)
        ]

    def monster_frame_groups(self, mob_id: str) -> dict[str, list[LogicalFrame]]:
        groups: dict[str, list[LogicalFrame]] = {}
        for frame in self.monster_frames(mob_id):
            action = frame.path.split("/", 1)[0].lower()
            family = next((name for name in ("stand", "move", "hit", "attack", "skill", "jump", "fly") if action.startswith(name)), action)
            groups.setdefault(family, []).append(frame)
        return groups

    def monster_death_frames(self, mob_id: str) -> list[LogicalFrame]:
        if mob_id not in self._monster_ids:
            return []
        return [
            frame for path, frame in self.resource(mob_id, "/Mob/").items()
            if any(word in path.lower() for word in DEATH_WORDS)
        ]

    def frame(self, name: str, group: str, candidates: Iterable[str]) -> LogicalFrame | None:
        frames = self.resource(name, group)
        for candidate in candidates:
            if candidate in frames:
                return frames[candidate]
        for candidate in candidates:
            prefix = candidate.rstrip("/") + "/"
            matches = [frame for path, frame in frames.items() if path.startswith(prefix)]
            if matches:
                return matches[0]
        return None


def _alpha_paste(canvas: Image.Image, sprite: Image.Image, x: int, y: int, flip: bool = False) -> None:
    if flip:
        sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    canvas.alpha_composite(sprite, (round(x), round(y)))


class MapRenderer:
    def __init__(self, sprites: SpriteLibrary, width: int = 1280, height: int = 224):
        self.sprites = sprites
        self.width = width
        self.height = height

    def _draw_back(self, canvas: Image.Image, item: dict[str, Any], camera_x: int, camera_y: int) -> None:
        name = item.get("bS")
        if not name:
            return
        number = str(item.get("no", 0))
        animated = int(item.get("ani", 0)) != 0
        candidates = [f"ani/{number}/0", f"back/{number}"] if animated else [f"back/{number}", f"ani/{number}/0"]
        frame = self.sprites.frame(name, "/Map/Back/", candidates)
        if not frame:
            return
        image = frame.image
        flip = bool(item.get('f', 0))
        origin_x = image.width - frame.origin_x if flip else frame.origin_x
        alpha = max(0, min(255, int(item.get('a', 255))))
        if alpha < 255:
            image = image.copy()
            image.putalpha(image.getchannel('A').point(lambda value: value * alpha // 255))
        kind = int(item.get("type", 0))
        # rx/ry are camera parallax percentages except on the automatic-scroll
        # axis, where they are velocity. Freeze that velocity at time zero.
        rx = 0 if kind in (4, 6) else int(item.get('rx', 0))
        ry = 0 if kind in (5, 7) else int(item.get('ry', 0))
        base_x = round(int(item.get('x', 0)) + self.width / 2 + (camera_x + self.width / 2) * rx / 100 - origin_x)
        base_y = round(int(item.get('y', 0)) + self.height / 2 + (camera_y + self.height / 2) * ry / 100 - frame.origin_y)

        def positions(base, step, extent, sprite_extent, tiled):
            if not tiled:
                return [base]
            start = math.floor((-sprite_extent - base) / step) + 1
            end = math.ceil((extent - base) / step)
            return [base + i * step for i in range(start, end)]

        xs = positions(base_x, max(1, abs(int(item.get('cx', 0))) or image.width), self.width, image.width, kind in (1, 3, 4, 6, 7))
        ys = positions(base_y, max(1, abs(int(item.get('cy', 0))) or image.height), self.height, image.height, kind in (2, 3, 5, 6, 7))
        for y in ys:
            for x in xs:
                _alpha_paste(canvas, image, x, y, flip)

    def _draw_tile(self, canvas: Image.Image, item: dict[str, Any], tile_set: str, camera_x: int, camera_y: int) -> None:
        frame = self.sprites.frame(tile_set, "/Map/Tile/", [f"{item.get('u', '')}/{item.get('no', 0)}"])
        if frame:
            x = int(item.get("x", 0)) - frame.origin_x - camera_x
            y = int(item.get("y", 0)) - frame.origin_y - camera_y
            if x < self.width and y < self.height and x + frame.image.width > 0 and y + frame.image.height > 0:
                _alpha_paste(canvas, frame.image, x, y)

    def _draw_object(self, canvas: Image.Image, item: dict[str, Any], camera_x: int, camera_y: int) -> None:
        name = item.get("oS")
        if not name:
            return
        path = f"{item.get('l0', '')}/{item.get('l1', '')}/{item.get('l2', '')}"
        frame = self.sprites.frame(name, "/Map/Obj/", [path + "/0", path])
        if frame:
            flip = bool(item.get('f', 0))
            origin_x = frame.image.width - frame.origin_x if flip else frame.origin_x
            x = int(item.get("x", 0)) - origin_x - camera_x
            y = int(item.get("y", 0)) - frame.origin_y - camera_y
            if x < self.width and y < self.height and x + frame.image.width > 0 and y + frame.image.height > 0:
                _alpha_paste(canvas, frame.image, x, y, flip)

    def render_layers(
        self, root: dict[str, Any], camera_x: int, camera_y: int, layer_min: int, layer_max: int,
        include_back: bool = False, front_back: bool = False,
    ) -> Image.Image:
        canvas = Image.new("RGBA", (self.width, self.height), (184, 222, 245, 255) if include_back else (0, 0, 0, 0))
        if include_back:
            for item in root.get("back", {}).values():
                if isinstance(item, dict) and bool(item.get("front", 0)) == front_back:
                    self._draw_back(canvas, item, camera_x, camera_y)
        for layer_index in range(layer_min, layer_max + 1):
            layer = root.get(str(layer_index), {})
            tile_set = layer.get("info", {}).get("tS")
            drawables: list[tuple[int, int, str, dict[str, Any]]] = []
            if tile_set:
                drawables.extend((int(x.get("zM", 0)), i, "tile", x) for i, x in enumerate(layer.get("tile", {}).values()))
            drawables.extend((int(x.get("z", 0)), i, "obj", x) for i, x in enumerate(layer.get("obj", {}).values()))
            for _, _, kind, item in sorted(drawables):
                if kind == "tile":
                    self._draw_tile(canvas, item, tile_set, camera_x, camera_y)
                else:
                    self._draw_object(canvas, item, camera_x, camera_y)
        return canvas

    def render_scene(
        self, root: dict[str, Any], camera_x: int, camera_y: int, monsters: list[PlacedMonster],
        overlays: list[PlacedOverlay] | None = None,
        minimum_clip_fraction: float = 0.30, minimum_visible_fraction: float = 0.10,
        label_players: bool = False, return_hidden_objects: bool = False,
    ) -> tuple[Image.Image, list[dict[str, Any]]] | tuple[Image.Image, list[dict[str, Any]], list[dict[str, Any]]]:
        footholds = foothold_segments(root)
        canvas = self.render_layers(root, camera_x, camera_y, 0, 5, include_back=True)
        # NPCs intentionally remain in the image without annotations, making
        # them hard negatives for the single monster class.
        for life in root.get("life", {}).values():
            if not isinstance(life, dict) or life.get("type") != "n" or not life.get("id"):
                continue
            frames = self.sprites.resource(str(life["id"]), "/Npc/")
            frame = next((v for k, v in frames.items() if k.lower().startswith(("stand/", "move/"))), None)
            frame = frame or next(iter(frames.values()), None)
            if frame:
                flip = bool(life.get("f", 0))
                origin_x = frame.image.width - frame.origin_x if flip else frame.origin_x
                x = int(life.get("x", 0)) - origin_x - camera_x
                y = int(life.get("cy", life.get("y", 0))) - frame.origin_y - camera_y
                _alpha_paste(canvas, frame.image, x, y, flip)
        foreground = self.render_layers(root, camera_x, camera_y, 6, 7)
        for item in root.get("back", {}).values():
            if isinstance(item, dict) and bool(item.get("front", 0)):
                self._draw_back(foreground, item, camera_x, camera_y)
        overlay_canvas = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        overlay_placements = []
        for overlay in overlays or []:
            sprite = overlay.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if overlay.flip else overlay.image
            origin_x = sprite.width - overlay.origin_x if overlay.flip else overlay.origin_x
            x = round(overlay.world_x - origin_x - camera_x)
            y = round(overlay.world_y - overlay.origin_y - camera_y)
            _alpha_paste(overlay_canvas, sprite, x, y)
            overlay_placements.append((overlay, sprite, x, y))
        later_occluders: list[Image.Image] = [Image.new("L", (1, 1))] * len(overlay_placements)
        running_occluder = foreground.getchannel("A").copy()
        for index in range(len(overlay_placements) - 1, -1, -1):
            later_occluders[index] = running_occluder.copy()
            _, later_sprite, later_x, later_y = overlay_placements[index]
            lx0, ly0 = max(0, later_x), max(0, later_y)
            lx1, ly1 = min(self.width, later_x + later_sprite.width), min(self.height, later_y + later_sprite.height)
            if lx1 > lx0 and ly1 > ly0:
                mask = later_sprite.getchannel("A").crop((
                    max(0, -later_x), max(0, -later_y),
                    min(later_sprite.width, self.width - later_x),
                    min(later_sprite.height, self.height - later_y),
                ))
                running_occluder.paste(
                    ImageChops.lighter(running_occluder.crop((lx0, ly0, lx1, ly1)), mask), (lx0, ly0)
                )
        occluder_alpha = ImageChops.lighter(foreground.getchannel("A"), overlay_canvas.getchannel("A"))
        labels: list[dict[str, Any]] = []
        for monster in monsters:
            sprite = monster.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if monster.flip else monster.image
            origin_x = sprite.width - monster.origin_x if monster.flip else monster.origin_x
            x = round(monster.world_x - origin_x - camera_x)
            y = round(monster.world_y - monster.origin_y - camera_y)
            alpha_box = sprite.getchannel("A").getbbox()
            if not alpha_box:
                continue
            _alpha_paste(canvas, sprite, x, y)
            ax0, ay0, ax1, ay1 = alpha_box
            x0, y0 = max(0, x + ax0), max(0, y + ay0)
            x1, y1 = min(self.width, x + ax1), min(self.height, y + ay1)
            alpha = sprite.getchannel("A")
            original_pixels = max(1, alpha.point(lambda value: 255 if value else 0).histogram()[255])
            sx0, sy0 = max(0, -x), max(0, -y)
            sx1, sy1 = min(sprite.width, self.width - x), min(sprite.height, self.height - y)
            clipped_alpha = alpha.crop((sx0, sy0, sx1, sy1)) if sx1 > sx0 and sy1 > sy0 else Image.new("L", (1, 1))
            clipped_pixels = clipped_alpha.point(lambda value: 255 if value else 0).histogram()[255]
            clip_fraction = clipped_pixels / original_pixels
            occluded_pixels = 0
            if clipped_pixels:
                front_crop = occluder_alpha.crop((max(0, x), max(0, y), min(self.width, x + sprite.width), min(self.height, y + sprite.height)))
                overlap = ImageChops.multiply(clipped_alpha, front_crop)
                occluded_pixels = overlap.point(lambda value: 255 if value else 0).histogram()[255]
            visible_fraction = max(0.0, (clipped_pixels - occluded_pixels) / original_pixels)
            occlusion_fraction = occluded_pixels / clipped_pixels if clipped_pixels else 0.0
            if (
                x1 - x0 >= 4 and y1 - y0 >= 4
                and clip_fraction >= minimum_clip_fraction
                and visible_fraction >= minimum_visible_fraction
            ):
                edge_sides = "".join(("L" if x + ax0 < 0 else "", "R" if x + ax1 > self.width else "", "T" if y + ay0 < 0 else "", "B" if y + ay1 > self.height else ""))
                labels.append({
                    "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                    "class_id": 0, "class_name": "monster",
                    "mob_id": monster.mob_id, "action": monster.action,
                    "foothold_id": monster.foothold_id,
                    "frame_path": monster.frame_path,
                    "world_x": monster.world_x,
                    "world_y": monster.world_y,
                    "ground_error": round(abs(
                        platform_y(footholds[monster.foothold_id], monster.world_x) - monster.world_y
                    ), 3) if monster.foothold_id in footholds else None,
                    "clip_fraction": round(clip_fraction, 4),
                    "visible_fraction": round(visible_fraction, 4),
                    "occlusion_fraction": round(occlusion_fraction, 4),
                    "edge_sides": edge_sides,
                })
        if label_players:
            for overlay_index, (overlay, sprite, x, y) in enumerate(overlay_placements):
                if overlay.kind != "player":
                    continue
                alpha = sprite.getchannel("A")
                alpha_box = alpha.getbbox()
                if not alpha_box:
                    continue
                ax0, ay0, ax1, ay1 = alpha_box
                x0, y0 = max(0, x + ax0), max(0, y + ay0)
                x1, y1 = min(self.width, x + ax1), min(self.height, y + ay1)
                original_pixels = max(1, alpha.point(lambda value: 255 if value else 0).histogram()[255])
                sx0, sy0 = max(0, -x), max(0, -y)
                sx1, sy1 = min(sprite.width, self.width - x), min(sprite.height, self.height - y)
                clipped_alpha = alpha.crop((sx0, sy0, sx1, sy1)) if sx1 > sx0 and sy1 > sy0 else Image.new("L", (1, 1))
                clipped_pixels = clipped_alpha.point(lambda value: 255 if value else 0).histogram()[255]
                occluded_pixels = 0
                if clipped_pixels:
                    front_crop = later_occluders[overlay_index].crop((
                        max(0, x), max(0, y), min(self.width, x + sprite.width), min(self.height, y + sprite.height)
                    ))
                    occluded_pixels = ImageChops.multiply(clipped_alpha, front_crop).point(
                        lambda value: 255 if value else 0
                    ).histogram()[255]
                clip_fraction = clipped_pixels / original_pixels
                visible_fraction = max(0.0, (clipped_pixels - occluded_pixels) / original_pixels)
                if x1 - x0 >= 4 and y1 - y0 >= 4 and clip_fraction >= minimum_clip_fraction and visible_fraction >= 0.30:
                    labels.append({
                        "bbox": [x0, y0, x1 - x0, y1 - y0], "area": (x1 - x0) * (y1 - y0),
                        "class_id": 1, "class_name": "player", "pose": overlay.frame_path,
                        "frame_path": overlay.frame_path, "world_x": overlay.world_x, "world_y": overlay.world_y,
                        "clip_fraction": round(clip_fraction, 4), "visible_fraction": round(visible_fraction, 4),
                        "occlusion_fraction": round(occluded_pixels / clipped_pixels if clipped_pixels else 0.0, 4),
                        "edge_sides": "".join(("L" if x + ax0 < 0 else "", "R" if x + ax1 > self.width else "", "T" if y + ay0 < 0 else "", "B" if y + ay1 > self.height else "")),
                    })
        hidden_objects: list[dict[str, Any]] = []
        target_boxes = [label["bbox"] for label in labels]
        for index, (overlay, sprite, x, y) in enumerate(overlay_placements):
            if overlay.kind == "player":
                continue
            alpha = sprite.getchannel("A")
            alpha_box = alpha.getbbox()
            if not alpha_box:
                continue
            ax0, ay0, ax1, ay1 = alpha_box
            x0, y0 = max(0, x + ax0), max(0, y + ay0)
            x1, y1 = min(self.width, x + ax1), min(self.height, y + ay1)
            if x1 <= x0 or y1 <= y0:
                continue
            original_pixels = max(1, alpha.point(lambda value: 255 if value else 0).histogram()[255])
            sx0, sy0 = max(0, -x), max(0, -y)
            sx1, sy1 = min(sprite.width, self.width - x), min(sprite.height, self.height - y)
            clipped_alpha = alpha.crop((sx0, sy0, sx1, sy1))
            clipped_pixels = clipped_alpha.point(lambda value: 255 if value else 0).histogram()[255]
            occluder = later_occluders[index]
            front_crop = occluder.crop((max(0, x), max(0, y), min(self.width, x + sprite.width), min(self.height, y + sprite.height)))
            occluded_pixels = ImageChops.multiply(clipped_alpha, front_crop).point(
                lambda value: 255 if value else 0
            ).histogram()[255] if clipped_pixels else 0
            bbox = [x0, y0, x1 - x0, y1 - y0]
            max_target_iou = 0.0
            for target in target_boxes:
                tx, ty, tw, th = target
                intersection = max(0, min(x1, tx + tw) - max(x0, tx)) * max(0, min(y1, ty + th) - max(y0, ty))
                union = bbox[2] * bbox[3] + tw * th - intersection
                max_target_iou = max(max_target_iou, intersection / union if union else 0.0)
            hidden_objects.append({
                "kind": overlay.kind, "source_id": overlay.source_id,
                "item_id": overlay.item_id or None, "visual_id": overlay.visual_id or None,
                "resource_path": overlay.resource_path or overlay.frame_path,
                "frame_path": overlay.frame_path, "state": overlay.state,
                "bbox": bbox, "clip_fraction": round(clipped_pixels / original_pixels, 4),
                "visible_fraction": round(max(0.0, (clipped_pixels - occluded_pixels) / original_pixels), 4),
                "occlusion_fraction": round(occluded_pixels / clipped_pixels if clipped_pixels else 0.0, 4),
                "max_target_iou": round(max_target_iou, 4), "foothold_id": overlay.foothold_id,
                "world_x": overlay.world_x, "world_y": overlay.world_y,
            })
        canvas.alpha_composite(overlay_canvas)
        canvas.alpha_composite(foreground)
        result = canvas.convert("RGB")
        return (result, labels, hidden_objects) if return_hidden_objects else (result, labels)


def foothold_segments(root: dict[str, Any]) -> dict[int, tuple[int, int, int, int]]:
    result: dict[int, tuple[int, int, int, int]] = {}
    for page in root.get("foothold", {}).values():
        if not isinstance(page, dict):
            continue
        for group in page.values():
            if not isinstance(group, dict):
                continue
            for key, value in group.items():
                if isinstance(value, dict) and all(k in value for k in ("x1", "y1", "x2", "y2")):
                    result[int(key)] = (int(value["x1"]), int(value["y1"]), int(value["x2"]), int(value["y2"]))
    return result


def platform_y(segment: tuple[int, int, int, int], x: int) -> int:
    x1, y1, x2, y2 = segment
    if x2 == x1:
        return max(y1, y2)
    ratio = (x - x1) / (x2 - x1)
    return round(y1 + ratio * (y2 - y1))


def build_monsters(
    root: dict[str, Any], sprites: SpriteLibrary, rng: random.Random, ensure_one: bool,
    max_count: int = 20, target_count: int | None = None, dense: bool = False,
    view_width: int = 1280, view_height: int = 224, placement: str | None = None,
    action_family: str | None = None, size_bucket: str | None = None,
    target_mob_id: str | None = None, target_frame_path: str | None = None,
    action_weights: dict[str, float] | None = None,
) -> list[PlacedMonster]:
    lives = [x for x in root.get("life", {}).values() if isinstance(x, dict) and x.get("type") == "m"]
    if not lives:
        return []
    footholds = foothold_segments(root)
    available: list[tuple[dict[str, Any], list[LogicalFrame]]] = []
    for life in lives:
        frames = sprites.monster_frames(str(life.get("id", "")))
        if action_family:
            frames = [frame for frame in frames if frame.path.split("/", 1)[0].lower().startswith(action_family)]
        if size_bucket and frames:
            ranked = sorted(frames, key=_frame_alpha_area)
            band = max(1, len(ranked) // 3)
            frames = ranked[:band] if size_bucket == "small" else ranked[-band:]
        if frames:
            available.append((life, frames))
    if not available:
        return []

    count = target_count if target_count is not None else rng.randint(1 if ensure_one else 0, max_count)
    if not 0 <= count <= max_count:
        raise ValueError(f"target monster count {count} outside 0..{max_count}")
    if count == 0:
        return []

    placement = placement or ("dense" if dense else "single_platform")
    if placement not in {"single_platform", "natural", "multi_platform", "dense", "extreme_overlap"}:
        raise ValueError(f"unknown monster placement: {placement}")
    target_available = [item for item in available if str(item[0].get("id", "")) == target_mob_id]
    if placement in {"dense", "extreme_overlap"} and target_frame_path and target_available:
        exact_frames = [frame for _, frames in target_available for frame in frames if frame.path == target_frame_path]
        if exact_frames:
            factor = 0.18 if placement == "extreme_overlap" else 0.65
            minimum_spacing = 3 if placement == "extreme_overlap" else 6
            spacing = max(minimum_spacing, min(50, round(exact_frames[0].image.width * factor)))
            required_range = spacing * max(0, count - 1) + 4
            target_available = [
                item for item in target_available
                if abs(int(item[0].get("rx1", item[0].get("x", 0))) - int(item[0].get("rx0", item[0].get("x", 0)))) >= required_range
            ]
            if not target_available:
                return []
    anchor_life, anchor_frames = rng.choice(target_available or available)
    anchor_x = int(anchor_life.get("x", 0))
    anchor_fh = int(anchor_life.get("fh", -1))
    # A scene remains physically valid by using spawn definitions attached to
    # the same platform. Repetition is intentional: game spawns can contain
    # multiple instances even when WZ stores only one spawn template.
    same_platform = [item for item in available if int(item[0].get("fh", -2)) == anchor_fh] or [
        (anchor_life, anchor_frames)
    ]
    if placement in {"natural", "multi_platform"}:
        anchor_y = int(anchor_life.get("cy", anchor_life.get("y", 0)))
        nearby = [
            item for item in available
            if abs(int(item[0].get("x", 0)) - anchor_x) <= view_width // 2 - 80
            and abs(int(item[0].get("cy", item[0].get("y", 0))) - anchor_y) <= view_height - 48
        ] or same_platform
        if placement == "multi_platform" and count >= 2:
            other_platform = [item for item in nearby if int(item[0].get("fh", -2)) != anchor_fh]
            if not other_platform:
                return []
            selected = [(anchor_life, anchor_frames), rng.choice(other_platform)]
            selected.extend(rng.choice(nearby) for _ in range(count - 2))
        else:
            selected = [(anchor_life, anchor_frames)]
            selected.extend(rng.choice(nearby) for _ in range(count - 1))
    elif placement in {"dense", "extreme_overlap"}:
        selected = [(anchor_life, anchor_frames)] * count
    else:
        selected = [(anchor_life, anchor_frames)]
        selected.extend(rng.choice(same_platform) for _ in range(count - 1))
    chosen_frames = []
    for _, frames in selected:
        if action_weights:
            groups: dict[str, list[LogicalFrame]] = {}
            for frame in frames:
                action = frame.path.split("/", 1)[0].lower()
                family = next((name for name in action_weights if action.startswith(name)), action)
                groups.setdefault(family, []).append(frame)
            families = list(groups)
            family = rng.choices(families, weights=[action_weights.get(name, 0.01) for name in families], k=1)[0]
            chosen_frames.append(rng.choice(groups[family]))
        else:
            chosen_frames.append(rng.choice(frames))
    if target_frame_path:
        exact = next((frame for frame in selected[0][1] if frame.path == target_frame_path), None)
        if exact is not None:
            chosen_frames[0] = exact

    result: list[PlacedMonster] = []
    if placement in {"dense", "extreme_overlap"}:
        # Width-relative spacing preserves genuine overlap but prevents large
        # sprites from almost completely hiding every earlier instance.
        center = max(
            min(anchor_x, max(int(anchor_life.get("rx0", anchor_x)), int(anchor_life.get("rx1", anchor_x)))),
            min(int(anchor_life.get("rx0", anchor_x)), int(anchor_life.get("rx1", anchor_x))),
        )
        widths = sorted(frame.image.width for frame in chosen_frames)
        spacing_factor = 0.18 if placement == "extreme_overlap" else 0.65
        spacing = max(3 if placement == "extreme_overlap" else 6, min(50, round(widths[len(widths) // 2] * spacing_factor)))
        offsets = [round((index - (count - 1) / 2) * spacing) + rng.randint(-2, 2) for index in range(count)]
    else:
        center = anchor_x
        offsets = []

    for index, ((life, _), frame) in enumerate(zip(selected, chosen_frames)):
        mob_id = str(life.get("id", ""))
        left = int(life.get("rx0", life.get("x", 0)))
        right = int(life.get("rx1", life.get("x", 0)))
        low, high = min(left, right), max(left, right)
        if placement in {"dense", "extreme_overlap"}:
            x = max(low, min(high, center + offsets[index]))
        else:
            # Keep all requested instances in one 1280-wide camera while still
            # distributing them over the real platform movement range.
            visible_low = max(low, anchor_x - view_width // 2 + 80)
            visible_high = min(high, anchor_x + view_width // 2 - 80)
            if visible_low > visible_high:
                visible_low = visible_high = max(low, min(high, anchor_x))
            x = rng.randint(visible_low, visible_high)
        segment = footholds.get(int(life.get("fh", -1)))
        y = platform_y(segment, x) if segment else int(life.get("cy", life.get("y", 0)))
        result.append(PlacedMonster(
            mob_id, frame.image, x, y, bool(rng.getrandbits(1)), frame.path.split("/", 1)[0],
            frame.origin_x, frame.origin_y, int(life.get("fh", -1)),
            frame.path,
        ))
    return result
