"""Addressables bundle indexing and selective WZ/Sprite extraction."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import gc
import json
from pathlib import Path
from typing import Any

import UnityPy
from PIL import Image

from .unity_bundle import disk_backed_unity_bundles
from .wzjs import WzjsDocument
from .wzss import WzssDocument


@dataclass(frozen=True)
class AssetLocation:
    bundle: str
    path_id: int
    type_name: str


class AssetStore:
    CLASSIC_1142_BUNDLES = {
        "mob": "spritesheet_d104d5b9c3cb56ae81ab3f15cef159c4.bundle",
        "map": "spritesheet_fa1998bbc1509b70ff5bd2d4f1ae1fde.bundle",
        "npc": "spritesheet_6b47c44b5abc31874d5b2b3e37f12c99.bundle",
        "character": "spritesheet_20e010fab3671bb1401f4b6877884c88.bundle",
        "skill": "spritesheet_8d17345025f1621af57dedfc1d158312.bundle",
        "reactor": "spritesheet_df700bdff698f83c4c00ebdbbc439c27.bundle",
        "effect": "spritesheet_5f5acf932ece444f66a5feae122c8c70.bundle",
        "item": "spritesheet_7c8b1482ddd3557d2f9bb54bf6f60629.bundle",
        "morph": "spritesheet_507cc10179cb538dfe5d8679163167f0.bundle",
        "ui": "spritesheet_34da445088826ac4c4c7aacfd03e1275.bundle",
        "etc": "spritesheet_c886d9f1aceda35a18a6d25b9b384dff.bundle",
    }
    def __init__(self, client_root: str | Path, cache_dir: str | Path):
        self.client_root = Path(client_root)
        self.aa_root = self.client_root / "Maplestory_Classic_Data" / "StreamingAssets" / "aa" / "w"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.cache_dir / "asset_index.json"
        self.index: dict[str, AssetLocation] = {}
        self._environments: dict[str, Any] = {}
        if self.index_path.exists():
            raw = json.loads(self.index_path.read_text(encoding="utf-8"))
            self.index = {key: AssetLocation(**value) for key, value in raw["assets"].items()}
            # Addressables bundle hashes change between client patches.  A
            # cached index that points at an old hash is unusable even though
            # all logical WZ paths are still valid, so rebuild it lazily.
            if any(not (self.aa_root / location.bundle).exists() for location in self.index.values()):
                self.index = {}

    def _load(self, bundle_name: str):
        if bundle_name in self._environments:
            return self._environments[bundle_name]
        path = self.aa_root / bundle_name
        # UnityPy can otherwise materialize several times the compressed size
        # while indexing the 175-588 MB sprite bundles.
        disk = path.stat().st_size > 100 * 1024 * 1024
        with disk_backed_unity_bundles(disk):
            env = UnityPy.load(str(path))
        self._environments[bundle_name] = env
        return env

    def build_index(self, include_sprites: bool = True, force: bool = False) -> dict[str, AssetLocation]:
        if self.index and not force:
            return self.index
        bundle_paths = set(self.aa_root.glob("json_*.bundle"))
        if include_sprites:
            # Discover sprite bundles instead of pinning their content hashes.
            # This keeps the adapter compatible with 1.14.x client updates.
            bundle_paths.update(self.aa_root.glob("spritesheet_*.bundle"))
        bundle_paths = sorted(bundle_paths)
        index: dict[str, AssetLocation] = {}
        failures: list[dict[str, str]] = []
        for bundle_path in bundle_paths:
            try:
                env = self._load(bundle_path.name)
                for asset in env.assets:
                    for key, pointer in asset.container.items():
                        reader = asset.objects.get(pointer.path_id)
                        type_name = reader.type.name if reader else "Unknown"
                        index[key.replace("\\", "/")] = AssetLocation(bundle_path.name, pointer.path_id, type_name)
            except Exception as exc:
                failures.append({"bundle": bundle_path.name, "error": f"{type(exc).__name__}: {exc}"})
            finally:
                self._environments.pop(bundle_path.name, None)
                gc.collect()
        self.index = index
        payload = {
            "client": str(self.client_root),
            "catalog_version": self._catalog_version(),
            "bundle_roles": self.CLASSIC_1142_BUNDLES,
            "assets": {key: asdict(value) for key, value in sorted(index.items())},
            "failures": failures,
        }
        self.index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return index

    def _catalog_version(self) -> str:
        catalogs = sorted(self.aa_root.glob("catalog_*.bin"))
        if not catalogs:
            return "unknown"
        return catalogs[-1].stem.removeprefix("catalog_")

    def find(self, suffix: str, contains: str | None = None) -> tuple[str, AssetLocation]:
        suffix = suffix.lower()
        matches = [
            (key, value)
            for key, value in self.index.items()
            if key.lower().endswith(suffix) and (contains is None or contains.lower() in key.lower())
        ]
        if not matches:
            raise KeyError(f"asset not indexed: *{suffix}")
        matches.sort(key=lambda item: (len(item[0]), item[0]))
        return matches[0]

    def _reader(self, location: AssetLocation):
        env = self._load(location.bundle)
        for asset in env.assets:
            if location.path_id in asset.objects:
                return asset.objects[location.path_id]
        raise KeyError(f"path id {location.path_id} missing from {location.bundle}")

    def read_wzjson(self, name: str, group: str | None = None) -> dict[str, Any]:
        return self.read_wz_document(name, group).to_python()

    def read_wz_document(self, name: str, group: str | None = None) -> WzjsDocument:
        key, location = self.find(name if name.endswith(".wzjson") else f"{name}.wzjson", group)
        return WzjsDocument.from_unity_raw(self._reader(location).get_raw_data())

    def read_image(self, asset_key: str) -> Image.Image:
        location = self.index[asset_key]
        obj = self._reader(location).read()
        image = obj.image
        if image is None:
            raise ValueError(f"image decode failed: {asset_key}")
        return image.convert("RGBA")

    def read_spritesheet(self, name: str, sprite_count: int, group: str | None = None) -> tuple[WzssDocument, list[Image.Image]]:
        suffix = name if name.endswith(".wzspritesheet") else f"{name}.wzspritesheet"
        key, location = self.find(suffix, group)
        sheet = WzssDocument(self._reader(location).get_raw_data(), sprite_count)
        base = key[: -len(".wzspritesheet")]
        page_count = max((frame.texture_index for frame in sheet.frames if frame.texture_index >= 0), default=0) + 1
        textures = []
        for page in range(page_count):
            texture_key = f"{base}_{page}.png"
            if texture_key not in self.index:
                raise KeyError(f"texture page missing: {texture_key}")
            textures.append(self.read_image(texture_key))
        return sheet, textures

    def close(self) -> None:
        self._environments.clear()
        gc.collect()
