"""Disk-backed UnityPy support for very large UnityFS bundles.

UnityPy 1.25.3 concatenates every decompressed block in RAM.  The two classic
client spritesheet bundles expand beyond practical memory limits, so this
module supplies equivalent readers backed by temporary files.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import tempfile

from UnityPy.enums import ArchiveFlags
from UnityPy.helpers import ImportHelper
from UnityPy.streams import EndianBinaryReader

BundleFile = importlib.import_module("UnityPy.files.BundleFile")
File = importlib.import_module("UnityPy.files.File")
SerializedFile = importlib.import_module("UnityPy.files.SerializedFile")


def _read_fs_disk(self, reader):
    reader.read_long()
    compressed_size = reader.read_u_int()
    uncompressed_size = reader.read_u_int()
    dataflags_value = reader.read_u_int()
    if self.signature != "UnityFS":
        reader.read_byte()
    version = self.parse_version()
    if version < (2020,) or (version[0] == 2020 and version < (2020, 3, 34)) or (
        version[0] == 2021 and version < (2021, 3, 2)
    ) or (version[0] == 2022 and version < (2022, 1, 1)):
        self.dataflags = BundleFile.ArchiveFlagsOld(dataflags_value)
    else:
        self.dataflags = ArchiveFlags(dataflags_value)
    if self.dataflags & self.dataflags.UsesAssetBundleEncryption:
        self.decryptor = BundleFile.ArchiveStorageManager.ArchiveStorageDecryptor(reader)
    if self.version >= 7 or (version[0] == 2019 and version >= (2019, 4, 15)):
        reader.align_stream(16)
        self._uses_block_alignment = True
    start = reader.Position
    if self.dataflags & ArchiveFlags.BlocksInfoAtTheEnd:
        reader.Position = reader.Length - compressed_size
        info_bytes = reader.read_bytes(compressed_size)
        reader.Position = start
    else:
        info_bytes = reader.read_bytes(compressed_size)
    info_bytes = self.decompress_data(info_bytes, uncompressed_size, self.dataflags)
    info = EndianBinaryReader(info_bytes, offset=start)
    info.read_bytes(16)
    block_count = info.read_int()
    blocks = [
        BundleFile.BlockInfo(info.read_u_int(), info.read_u_int(), info.read_u_short())
        for _ in range(block_count)
    ]
    node_count = info.read_int()
    directories = [
        BundleFile.DirectoryInfoFS(
            info.read_long(), info.read_long(), info.read_u_int(), info.read_string_to_null()
        )
        for _ in range(node_count)
    ]
    if blocks:
        self._block_info_flags = blocks[0].flags
    if isinstance(self.dataflags, ArchiveFlags) and self.dataflags & ArchiveFlags.BlockInfoNeedPaddingAtStart:
        reader.align_stream(16)

    configured = os.environ.get("CLASSIC_DATASET_TEMP")
    cache_root = Path(configured) if configured else Path("F:/MapleStoryAssets/.classic_cache/unity_bundles")
    if not cache_root.drive or not Path(cache_root.drive + "/").exists():
        cache_root = Path(tempfile.gettempdir()) / "classic_dataset_unitypy"
    cache_root.mkdir(parents=True, exist_ok=True)
    base_offset = info.real_offset()
    safe_name = Path(self.name).name.replace(".bundle", "")
    cache_file = cache_root / f"{safe_name}.decompressed"
    expected_size = base_offset + sum(block.uncompressedSize for block in blocks)
    if not cache_file.exists() or cache_file.stat().st_size != expected_size:
        temp = tempfile.NamedTemporaryFile(prefix=safe_name + "_", suffix=".part", dir=cache_root, delete=False)
        if base_offset:
            temp.write(b"\0" * base_offset)
        for index, block in enumerate(blocks):
            compressed = reader.read_bytes(block.compressedSize)
            temp.write(self.decompress_data(compressed, block.uncompressedSize, block.flags, index))
        temp.flush()
        temp.close()
        Path(temp.name).replace(cache_file)
    blocks_reader = EndianBinaryReader(str(cache_file), offset=base_offset)
    blocks_reader.Position = 0
    self._classic_temp_paths = [str(cache_file)]
    return directories, blocks_reader


def _read_files_streaming(self, reader, files):
    for node in files:
        # Each child reader owns an independent handle. Sharing one handle lets
        # a resource reader's destructor close the stream used by all assets.
        source_path = reader.stream.name
        node_reader = EndianBinaryReader(str(source_path), offset=reader.BaseOffset + node.offset)
        node_reader.Position = 0
        parsed = ImportHelper.parse_file(node_reader, self, node.path, is_dependency=self.is_dependency)
        if isinstance(parsed, (EndianBinaryReader, SerializedFile.SerializedFile)) and self.environment:
            self.environment.register_cab(node.path, parsed)
        parsed.flags = getattr(node, "flags", 0)
        self.files[node.path] = parsed


@contextmanager
def disk_backed_unity_bundles(enabled: bool = True):
    """Temporarily make UnityPy stream decompressed UnityFS data from disk."""
    if not enabled:
        yield
        return
    old_fs = BundleFile.BundleFile.read_fs
    old_files = File.File.read_files
    BundleFile.BundleFile.read_fs = _read_fs_disk
    File.File.read_files = _read_files_streaming
    try:
        yield
    finally:
        BundleFile.BundleFile.read_fs = old_fs
        File.File.read_files = old_files
