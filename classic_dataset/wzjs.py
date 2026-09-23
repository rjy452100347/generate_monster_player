"""Decoder for the WZJS v5 blobs embedded in mxdclassic MonoBehaviours.

The format stores a flat node table followed by typed value pools and three
UTF-8 string pools (keys, full paths and string values).  This implementation
does not rely on Unity MonoBehaviour type trees and therefore also works with
the client's IL2CPP metadata v39.
"""

from __future__ import annotations

from dataclasses import dataclass
import struct
from typing import Any, Iterable


class WzjsError(ValueError):
    pass


@dataclass(frozen=True)
class Node:
    type_code: int
    key_index: int
    value_index: int
    first_child: int
    child_count: int
    parent: int
    path_index: int
    previous_path_index: int


@dataclass(frozen=True)
class SpriteValue:
    """A WZ sprite/link node; children contain spritesheet and frame metadata."""

    children: dict[str, Any]


# Type 19 appeared in later 1.14.x item documents. It is a container-like
# metadata node (not a sprite slot) and can safely be decoded as children.
VALID_TYPES = {2, 5, 6, 8, 11, 12, 14, 18, 19}


def _i32(data: bytes, offset: int) -> int:
    return struct.unpack_from("<i", data, offset)[0]


def _align4(value: int) -> int:
    return (value + 3) & ~3


def _offsets(data: bytes, start: int, count: int) -> tuple[int, ...]:
    if count < 1 or start < 0 or start + count * 4 > len(data):
        raise WzjsError("invalid string offset table")
    values = struct.unpack_from(f"<{count}i", data, start)
    if values[0] != 0 or any(a > b or a < 0 for a, b in zip(values, values[1:])):
        raise WzjsError("non-monotonic string offsets")
    return values


def _strings(data: bytes, start: int, offsets: Iterable[int]) -> list[str]:
    points = list(offsets)
    result: list[str] = []
    for a, b in zip(points, points[1:]):
        raw = data[start + a : start + b]
        result.append(raw.rstrip(b"\0").decode("utf-8", errors="replace"))
    return result


class WzjsDocument:
    def __init__(self, blob: bytes, layout: dict[str, int] | None = None):
        if blob[:4] != b"WZJS":
            raise WzjsError("WZJS magic not found")
        self.blob = blob
        self.layout = layout
        self.version = _i32(blob, 4)
        if self.version != 5:
            raise WzjsError(f"unsupported WZJS version {self.version}")
        self.nodes = self._read_nodes()
        self.logical_path_count = self.nodes[0].path_index
        self.keys, self.paths, self.string_values, self.pool_start = self._read_strings()
        self.bool_values, self.int_values, self.float_values, self.vector_values = self._read_values()

    @classmethod
    def from_unity_raw(cls, raw: bytes) -> "WzjsDocument":
        offset = raw.find(b"WZJS")
        if offset < 0:
            raise WzjsError("MonoBehaviour does not contain a WZJS blob")
        if offset >= 144:
            layout = {
                "node_count": _i32(raw, offset - 144),
                "node_start": _i32(raw, offset - 140),
                "bool_count": _i32(raw, offset - 112),
                "bool_start": _i32(raw, offset - 108),
                "int_count": _i32(raw, offset - 104),
                "int_start": _i32(raw, offset - 100),
                "float_count": _i32(raw, offset - 88),
                "float_start": _i32(raw, offset - 84),
                "vector_count": _i32(raw, offset - 80),
                "vector_start": _i32(raw, offset - 76),
                "key_count": _i32(raw, offset - 40),
                "key_data_start": _i32(raw, offset - 36),
                "key_offsets_start": _i32(raw, offset - 32),
                "path_count": _i32(raw, offset - 28),
                "path_data_start": _i32(raw, offset - 24),
                "path_offsets_start": _i32(raw, offset - 20),
                "value_count": _i32(raw, offset - 16),
                "value_data_start": _i32(raw, offset - 12),
                "value_offsets_start": _i32(raw, offset - 8),
                "blob_length": _i32(raw, offset - 4),
            }
            length = layout["blob_length"]
            if layout["node_start"] == 8 and length >= 8 and offset + length <= len(raw):
                return cls(raw[offset : offset + length], layout)
        return cls(raw[offset:])

    def _read_nodes(self) -> list[Node]:
        if self.layout:
            start = self.layout["node_start"]
            count = self.layout["node_count"]
            if start + count * 32 > len(self.blob):
                raise WzjsError("node table outside WZJS blob")
            nodes = [Node(*struct.unpack_from("<8i", self.blob, start + i * 32)) for i in range(count)]
            if any(node.type_code not in VALID_TYPES for node in nodes):
                raise WzjsError("unknown WZJS node type")
            return nodes
        nodes: list[Node] = []
        pos = 8
        root_limit: int | None = None
        while pos + 32 <= len(self.blob):
            fields = struct.unpack_from("<8i", self.blob, pos)
            if fields[0] not in VALID_TYPES:
                break
            if fields[1] < 0 or fields[4] < 0 or fields[5] < -1:
                break
            if root_limit is None:
                root_limit = max(fields[6], 1)
            if fields[3] < -1 or fields[3] > root_limit + 1 or fields[5] > root_limit + 1:
                break
            nodes.append(Node(*fields))
            pos += 32
        if not nodes or nodes[0].type_code != 2 or nodes[0].parent != -1:
            raise WzjsError("invalid WZJS root node")
        return nodes

    def _find_key_offsets(self, end: int, count: int) -> tuple[int, tuple[int, ...]]:
        # Pools are four-byte aligned and some files contain one padding word.
        for padding in range(0, 20, 4):
            start = end - padding - count * 4
            try:
                values = _offsets(self.blob, start, count)
            except WzjsError:
                continue
            data_start = start - _align4(values[-1])
            if data_start >= 8 + len(self.nodes) * 32:
                return start, values
        raise WzjsError("cannot locate key string offsets")

    def _read_strings(self) -> tuple[list[str], list[str], list[str], int]:
        if self.layout:
            key_count = self.layout["key_count"]
            path_count = self.layout["path_count"]
            value_count = self.layout["value_count"]
            key_offsets = _offsets(self.blob, self.layout["key_offsets_start"], key_count + 1)
            path_offsets = _offsets(self.blob, self.layout["path_offsets_start"], path_count + 1)
            value_offsets = _offsets(self.blob, self.layout["value_offsets_start"], value_count + 1)
            return (
                _strings(self.blob, self.layout["key_data_start"], key_offsets),
                _strings(self.blob, self.layout["path_data_start"], path_offsets),
                _strings(self.blob, self.layout["value_data_start"], value_offsets),
                self.layout["key_data_start"],
            )
        key_count = max(n.key_index for n in self.nodes) + 1
        string_indexes = [n.value_index for n in self.nodes if n.type_code in (11, 12)]
        value_count = max(string_indexes, default=0) + 1
        value_offset_count = value_count + 1
        path_offset_count = self.logical_path_count + 2
        last_error: Exception | None = None
        for trailing_padding in range(0, 20, 4):
            try:
                logical_end = len(self.blob) - trailing_padding
                value_offsets_start = logical_end - value_offset_count * 4
                value_offsets = _offsets(self.blob, value_offsets_start, value_offset_count)
                value_data_start = value_offsets_start - _align4(value_offsets[-1])
                path_offsets_start = value_data_start - path_offset_count * 4
                path_offsets = _offsets(self.blob, path_offsets_start, path_offset_count)
                path_data_start = path_offsets_start - _align4(path_offsets[-1])
                key_offsets_start, key_offsets = self._find_key_offsets(path_data_start, key_count + 1)
                key_data_start = key_offsets_start - _align4(key_offsets[-1])
                keys = _strings(self.blob, key_data_start, key_offsets)
                paths = _strings(self.blob, path_data_start, path_offsets)
                values = _strings(self.blob, value_data_start, value_offsets)
                if len(keys) != key_count:
                    raise WzjsError("key pool size mismatch")
                return keys, paths, values, key_data_start
            except (WzjsError, struct.error) as exc:
                last_error = exc
        raise WzjsError(f"cannot decode WZJS string pools: {last_error}")

    def _read_values(self) -> tuple[list[bool], list[int], list[float], list[tuple[float, float]]]:
        if self.layout:
            bool_count = self.layout["bool_count"]
            int_count = self.layout["int_count"]
            float_count = self.layout["float_count"]
            vector_count = self.layout["vector_count"]
            bools = [bool(value) for value in struct.unpack_from(f"<{bool_count}h", self.blob, self.layout["bool_start"])] if bool_count else []
            ints = list(struct.unpack_from(f"<{int_count}i", self.blob, self.layout["int_start"])) if int_count else []
            floats = list(struct.unpack_from(f"<{float_count}f", self.blob, self.layout["float_start"])) if float_count else []
            vectors = [struct.unpack_from("<2f", self.blob, self.layout["vector_start"] + i * 8) for i in range(vector_count)]
            return bools, ints, floats, vectors
        pos = 8 + len(self.nodes) * 32
        bool_count = max((n.value_index for n in self.nodes if n.type_code == 5), default=-1) + 1
        int_count = max((n.value_index for n in self.nodes if n.type_code == 6), default=-1) + 1
        float_count = max((n.value_index for n in self.nodes if n.type_code == 8), default=-1) + 1
        vector_count = max((n.value_index for n in self.nodes if n.type_code == 14), default=-1) + 1

        bools = [bool(value) for value in struct.unpack_from(f"<{bool_count}h", self.blob, pos)] if bool_count else []
        pos += bool_count * 2
        pos = _align4(pos)
        ints = list(struct.unpack_from(f"<{int_count}i", self.blob, pos)) if int_count else []
        pos += int_count * 4
        floats = list(struct.unpack_from(f"<{float_count}f", self.blob, pos)) if float_count else []
        pos += float_count * 4
        vectors = [struct.unpack_from("<2f", self.blob, pos + i * 8) for i in range(vector_count)]
        pos += vector_count * 8
        if pos > self.pool_start:
            raise WzjsError("typed value pools overlap string data")
        return bools, ints, floats, vectors

    def _value(self, index: int) -> Any:
        node = self.nodes[index]
        if node.type_code == 2:
            return self._children(node)
        if node.type_code == 5:
            return self.bool_values[node.value_index]
        if node.type_code == 6:
            return self.int_values[node.value_index]
        if node.type_code == 8:
            return self.float_values[node.value_index]
        if node.type_code in (11, 12):
            return self.string_values[node.value_index]
        if node.type_code == 14:
            x, y = self.vector_values[node.value_index]
            return {"x": x, "y": y}
        if node.type_code == 18:
            return SpriteValue(self._children(node))
        if node.type_code == 19:
            return self._children(node)
        raise WzjsError(f"unknown node type {node.type_code}")

    def _children(self, node: Node) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if node.first_child < 0:
            return result
        end = node.first_child + node.child_count
        if end > len(self.nodes):
            raise WzjsError("child range outside physical node table")
        for index in range(node.first_child, end):
            child = self.nodes[index]
            key = self.keys[child.key_index]
            value = self._value(index)
            if key in result:
                old = result[key]
                result[key] = old + [value] if isinstance(old, list) else [old, value]
            else:
                result[key] = value
        return result

    def to_python(self) -> dict[str, Any]:
        root = self.nodes[0]
        return {self.keys[root.key_index]: self._value(0)}


def json_default(value: Any) -> Any:
    if isinstance(value, SpriteValue):
        return {"$sprite": value.children}
    raise TypeError(type(value).__name__)
