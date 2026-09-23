from __future__ import annotations

import random
from pathlib import Path
import unittest

from classic_dataset.assets import AssetStore
from classic_dataset.render import SpriteLibrary, build_monsters, foothold_segments


CLIENT = Path(r"D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic")
CACHE = Path(r"F:\MapleStoryAssets\.classic_cache\index_1.14.2")


@unittest.skipUnless(CLIENT.exists(), "mxdclassic client is not installed")
class ClassicClientIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.store = AssetStore(CLIENT, CACHE)
        cls.store.build_index()
        cls.sprites = SpriteLibrary(cls.store)

    @classmethod
    def tearDownClass(cls):
        cls.store.close()

    def test_map_104000100_and_snail(self):
        root = next(iter(self.store.read_wzjson("104000100", "/Map/Map/").values()))
        self.assertGreater(len(foothold_segments(root)), 0)
        self.assertGreater(len(root["life"]), 0)
        frames = self.sprites.monster_frames("0100100")
        self.assertGreater(len(frames), 0)
        self.assertFalse(any("die" in frame.path.lower() for frame in frames))

    def test_seed_is_deterministic(self):
        root = next(iter(self.store.read_wzjson("104000100", "/Map/Map/").values()))
        first = build_monsters(root, self.sprites, random.Random(83), True)
        second = build_monsters(root, self.sprites, random.Random(83), True)
        signature = lambda items: [
            (item.mob_id, item.world_x, item.world_y, item.flip, item.action, item.image.size) for item in items
        ]
        self.assertEqual(signature(first), signature(second))

    def test_dense_scene_has_exact_requested_count(self):
        root = next(iter(self.store.read_wzjson("104000100", "/Map/Map/").values()))
        monsters = build_monsters(
            root, self.sprites, random.Random(83), True,
            max_count=20, target_count=20, dense=True,
        )
        self.assertEqual(len(monsters), 20)
        self.assertLessEqual(max(item.world_x for item in monsters) - min(item.world_x for item in monsters), 1000)

    def test_later_114x_item_atlas_aliases(self):
        frames = self.sprites.resource("0400", "/Item/Etc/")
        self.assertGreaterEqual(len(frames), 1300)
        self.assertIn("04000000/info/iconRaw", frames)
        self.assertIsNotNone(frames["04000000/info/iconRaw"].image.getchannel("A").getbbox())


if __name__ == "__main__":
    unittest.main()
