"""WZSS v4/v5 spritesheet metadata decoder."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Sequence
import uuid

from PIL import Image


class WzssError(ValueError):
    pass


@dataclass(frozen=True)
class AtlasFrame:
    texture_index: int
    x: int
    y: int
    width: int
    height: int
    alias_of: int | None = None
    pivot_x: float | None = None
    pivot_y: float | None = None


class WzssDocument:
    def __init__(self, raw: bytes, sprite_count: int):
        start = raw.find(b"WZSS")
        if start < 0:
            raise WzssError("WZSS magic not found")
        self.blob = raw[start:]
        self.version = struct.unpack_from("<i", self.blob, 4)[0]
        # 1.15.2 adds v5 header metadata; its rectangle and logical-frame
        # tables retain the layout located and checked by _decode().
        if self.version not in (4, 5):
            raise WzssError(f"unsupported WZSS version {self.version}")
        self.sprite_count = sprite_count
        self.name_to_frame: dict[str, int] = {}
        name_length = struct.unpack_from('<i', raw, 28)[0] if start >= 32 else -1
        header_start = (32 + name_length + 3) & ~3
        if name_length >= 0 and header_start + 19 * 4 <= start:
            self.rectangles, self.frames = self._decode_header(raw, header_start)
        else:
            self.rectangles, self.frames = self._decode()

    def _decode_header(self, raw: bytes, header_start: int):
        header = struct.unpack_from('<19i', raw, header_start)

        def table(count: int, offset: int, fmt: str):
            size = struct.calcsize(fmt)
            if count < 0 or offset < 8 or offset + count * size > len(self.blob):
                raise WzssError('WZSS table outside blob')
            return [struct.unpack_from(fmt, self.blob, offset + i * size) for i in range(count)]

        # Logical texture slots are independently sorted by texture GUID.
        # This table maps each slot to the numbered PNG asset, not vice versa.
        pages = [row[0] for row in table(header[0], header[1], '<i')]
        if sorted(pages) != list(range(len(pages))):
            raise WzssError('invalid WZSS texture page permutation')
        pivots = table(header[5], header[6], '<2f')
        rectangles = table(header[7], header[8], '<4i')
        records = table(header[9], header[10], '<5i')
        links = table(header[11], header[12], '<4i')
        name_count, name_data, name_offsets = header[13:16]
        offsets = [row[0] for row in table(name_count + 1, name_offsets, '<i')]
        if offsets[0] != 0 or any(a > b or a < 0 for a, b in zip(offsets, offsets[1:])) or name_data + offsets[-1] > len(self.blob):
            raise WzssError('invalid WZSS name pool')
        names = [self.blob[name_data + offsets[i]:name_data + offsets[i + 1]].decode('utf-8') for i in range(name_count)]
        frames: list[AtlasFrame | None] = [None] * name_count
        for frame_id, name_id, texture_id, rect_id, pivot_id in records:
            if not (0 <= frame_id < name_count and 0 <= name_id < name_count and 0 <= rect_id < len(rectangles) and 0 <= pivot_id < len(pivots) and 0 <= texture_id < len(pages)):
                raise WzssError('invalid WZSS direct frame')
            x, y, w, h = rectangles[rect_id]
            if w <= 0 or h <= 0:
                raise WzssError('invalid WZSS rectangle')
            px, py = pivots[pivot_id]
            frames[frame_id] = AtlasFrame(pages[texture_id], x, y, w, h, pivot_x=px, pivot_y=py)
            self.name_to_frame[names[name_id]] = frame_id
        for frame_id, _, _, target in links:
            if not (0 <= frame_id < name_count and 0 <= target < name_count):
                raise WzssError('invalid WZSS alias')
            frames[frame_id] = AtlasFrame(-1, 0, 0, 0, 0, target)
            self.name_to_frame[names[frame_id]] = frame_id
        if any(frame is None for frame in frames):
            raise WzssError('incomplete WZSS frame table')
        return rectangles, frames

    def index_for_path(self, full_path: str) -> int:
        # The client names sprites with new Guid(MD5(UTF8(path))).ToString("N").
        # Guid reverses the first 4/2/2 byte fields; plain MD5 hex does not match.
        key = uuid.UUID(bytes_le=hashlib.md5(full_path.encode('utf-8')).digest()).hex
        try:
            return self.name_to_frame[key]
        except KeyError as exc:
            raise WzssError(f'no atlas mapping for {full_path}') from exc

    def _decode(self) -> tuple[list[tuple[int, int, int, int]], list[AtlasFrame]]:
        # Atlas/page headers are variable length. Locate the 20-byte logical
        # frame table by its sequential frame-id column, then derive the
        # preceding rectangle pool from its largest referenced rectangle.
        selected: tuple[int, int, list[tuple[int, int, int, int, int]], list[int]] | None = None
        for map_start in range(8, len(self.blob) - 20 + 1, 4):
            first = struct.unpack_from("<5i", self.blob, map_start)
            if first[0] != 0 or first[1] != 0 or first[2] < 0 or first[3] != 0:
                continue
            records: list[tuple[int, int, int, int, int]] = []
            for i in range(self.sprite_count):
                if map_start + (i + 1) * 20 > len(self.blob):
                    break
                record = struct.unpack_from("<5i", self.blob, map_start + i * 20)
                if record[0] != i or record[1] != i or record[2] < 0 or record[3] < 0:
                    break
                records.append(record)
            direct_count = len(records)
            rect_count = max((record[3] for record in records), default=-1) + 1
            rect_start = map_start - rect_count * 16
            if direct_count < 1 or rect_count < 1 or rect_start < 8:
                continue
            rectangles = [struct.unpack_from("<4i", self.blob, rect_start + i * 16) for i in range(rect_count)]
            if not all(w > 0 and h > 0 for _, _, w, h in rectangles):
                continue
            alias_count = self.sprite_count - direct_count
            aliases: list[int] = []
            if alias_count:
                alias_start = map_start + direct_count * 20
                for padding in range(0, 20, 4):
                    candidate = alias_start + padding
                    if candidate + alias_count * 16 > len(self.blob):
                        continue
                    possible = [struct.unpack_from("<4i", self.blob, candidate + i * 16) for i in range(alias_count)]
                    old_prefix = 0
                    for i, row in enumerate(possible):
                        if not (row[0] == direct_count + i and row[1] >= -1 and row[2] == -1 and 0 <= row[3] < self.sprite_count):
                            break
                        old_prefix += 1
                    if old_prefix and direct_count + old_prefix >= round(self.sprite_count * 0.95):
                        aliases = [row[3] for row in possible[:old_prefix]]
                        break
                    # 1.14.4 item atlases store aliases as
                    # (-1, -1, source_frame, logical_frame).
                    new_prefix = 0
                    for i, row in enumerate(possible):
                        if not (row[0] == -1 and row[1] == -1 and 0 <= row[2] < self.sprite_count and row[3] == direct_count + i):
                            break
                        new_prefix += 1
                    if new_prefix and direct_count + new_prefix >= round(self.sprite_count * 0.95):
                        aliases = [row[2] for row in possible[:new_prefix]]
                        break
                else:
                    continue
            selected = rect_start, rect_count, records, aliases
            break
        if selected is None:
            raise WzssError("cannot locate WZSS frame mapping table")
        rect_start, rect_count, records, aliases = selected
        direct_count = len(records)
        rectangles = [struct.unpack_from("<4i", self.blob, rect_start + i * 16) for i in range(rect_count)]
        if any(w <= 0 or h <= 0 for _, _, w, h in rectangles):
            raise WzssError("invalid WZSS crop rectangle")
        frames: list[AtlasFrame] = []
        for record in records:
            _, _, texture_index, rect_index, alias = record
            x, y, width, height = rectangles[rect_index]
            frames.append(AtlasFrame(texture_index, x, y, width, height))
        frames.extend(AtlasFrame(-1, 0, 0, 0, 0, target) for target in aliases)
        return rectangles, frames

    def resolved_frame(self, index: int) -> AtlasFrame:
        seen: set[int] = set()
        while self.frames[index].alias_of is not None:
            if index in seen:
                raise WzssError("cyclic sprite alias")
            seen.add(index)
            index = int(self.frames[index].alias_of)
        return self.frames[index]

    def crop(self, index: int, textures: Sequence[Image.Image]) -> Image.Image:
        frame = self.resolved_frame(index)
        if not 0 <= frame.texture_index < len(textures):
            raise WzssError('texture index outside atlas pages')
        texture = textures[frame.texture_index]
        if frame.x < 0 or frame.y < 0 or frame.x + frame.width > texture.width or frame.y + frame.height > texture.height:
            raise WzssError('crop rectangle outside atlas page')
        # Unity Texture2D rects use a bottom-left origin; Pillow uses top-left.
        top = texture.height - frame.y - frame.height
        return texture.crop((frame.x, top, frame.x + frame.width, top + frame.height)).convert("RGBA")
