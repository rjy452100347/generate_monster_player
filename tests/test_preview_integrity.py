import struct
import unittest
from unittest.mock import Mock
from collections import Counter

from PIL import Image

from classic_dataset.distractors import _compose
from classic_dataset.render import LogicalFrame, MapRenderer
from classic_dataset.wzss import WzssDocument, WzssError
from classic_dataset.scattered_formal import build_schedule


def fixture(version, pages=(0,)):
    # Independently observed client GUID keys, intentionally not path order.
    names = [b'455988fe32c7de6f759a362d11b8b60a', b'56ba3215448792d2f9f8dc56d4befbb2', b'31990500ec7c5bde91131c087f3bba54']
    blob = bytearray(b'WZSS' + struct.pack('<i', version))
    blob.extend(struct.pack('<' + 'i' * len(pages), *pages))
    pivot_offset = len(blob)
    blob.extend(struct.pack('<2f', .25, .75))
    rect_offset = len(blob)
    blob.extend(struct.pack('<8i', 0, 0, 4, 4, 4, 0, 4, 4))
    record_offset = len(blob)
    blob.extend(struct.pack('<10i', 0, 0, 0, 0, 0, 1, 1, 0, 1, 0))
    link_offset = len(blob)
    blob.extend(struct.pack('<4i', 2, -1, -1, 1))
    name_offsets = len(blob)
    blob.extend(struct.pack('<4i', 0, 32, 64, 96))
    name_data = len(blob)
    blob.extend(b''.join(names))
    header = (len(pages), 8, 0, 8, 8, 1, pivot_offset, 2, rect_offset, 2, record_offset, 1, link_offset, 3, name_data, name_offsets, 0, name_data, name_offsets)
    return bytes(32) + struct.pack('<19i', *header) + struct.pack('<i', len(blob)) + blob


class SpriteMappingRegression(unittest.TestCase):
    def test_texture_slot_uses_page_permutation(self):
        doc = WzssDocument(fixture(5, (1, 0)), 3)
        textures = [Image.new('RGBA', (8, 4), 'red'), Image.new('RGBA', (8, 4), 'blue')]
        self.assertEqual(doc.crop(0, textures).getpixel((0, 0)), (0, 0, 255, 255))

    def test_path_guid_order_and_alias_for_both_versions(self):
        for version in (4, 5):
            with self.subTest(version=version):
                doc = WzssDocument(fixture(version), 3)
                self.assertEqual(doc.index_for_path('Mob/0100100/stand/0'), 1)
                self.assertEqual(doc.index_for_path('Mob/0100100/move/0'), 0)
                alias = doc.index_for_path('Mob/0100100/die1/0')
                self.assertEqual(doc.resolved_frame(alias), doc.resolved_frame(1))
                texture = Image.new('RGBA', (8, 4), 'red')
                texture.paste((0, 0, 255, 255), (4, 0, 8, 4))
                self.assertEqual(doc.crop(1, [texture]).getpixel((0, 0)), (0, 0, 255, 255))
                self.assertEqual(doc.resolved_frame(1).pivot_x, .25)

    def test_unknown_path_never_falls_back_to_list_order(self):
        doc = WzssDocument(fixture(5), 3)
        with self.assertRaises(WzssError):
            doc.index_for_path('Mob/0100100/unknown/0')

    def test_reject_out_of_page_crop(self):
        doc = WzssDocument(fixture(5), 3)
        with self.assertRaises(WzssError):
            doc.crop(1, [Image.new('RGBA', (4, 4))])


class MapPlacementRegression(unittest.TestCase):
    def setUp(self):
        self.frame = LogicalFrame('back/0', Image.new('RGBA', (10, 10), 'red'), 2, 5)
        self.renderer = MapRenderer(Mock(frame=Mock(return_value=self.frame)), 100, 50)

    def test_screen_pinned_background_ignores_world_camera(self):
        canvas = Image.new('RGBA', (100, 50))
        self.renderer._draw_back(canvas, {'bS': 'test', 'rx': 0, 'ry': 0}, 900, 800)
        self.assertEqual(canvas.getbbox(), (48, 20, 58, 30))

    def test_world_pinned_background_uses_world_coordinates(self):
        canvas = Image.new('RGBA', (100, 50))
        self.renderer._draw_back(canvas, {'bS': 'test', 'rx': -100, 'ry': -100, 'x': 925, 'y': 820}, 900, 800)
        self.assertEqual(canvas.getbbox(), (23, 15, 33, 25))

    def test_flipped_object_preserves_world_anchor(self):
        canvas = Image.new('RGBA', (100, 50))
        self.renderer._draw_object(canvas, {'oS': 'test', 'f': 1, 'x': 25, 'y': 20}, 0, 0)
        self.assertEqual(canvas.getbbox(), (17, 15, 27, 25))


class FormalScheduleRegression(unittest.TestCase):
    def test_full_schedule_preserves_all_quotas(self):
        config = {
            'seed': 7, 'splits': {'train': 32000, 'val': 4000, 'test': 4000},
            'scenarios': {'player_pet_monster': 28000, 'player_monster': 6000, 'player_pet': 3200, 'monster': 2000, 'background': 800},
            'monster_quotas': {1: 6000, 2: 10000, 3: 10000, 4: 6000, 5: 4000},
            'two_pet_images': 6400,
        }
        plan = build_schedule(config)
        self.assertEqual(len(plan), 40000)
        self.assertEqual(Counter(item.split for item in plan), Counter(config['splits']))
        self.assertEqual(Counter(item.scenario for item in plan), Counter(config['scenarios']))
        self.assertEqual(Counter(item.monster_count for item in plan if item.monster_count), Counter(config['monster_quotas']))
        self.assertEqual(sum(item.pet_count == 2 for item in plan), 6400)
        self.assertEqual(len({(item.split, item.split_index) for item in plan}), 40000)


class CompletePlayerRegression(unittest.TestCase):
    def setUp(self):
        self.body = LogicalFrame('stand1/0/body', Image.new('RGBA', (8, 10), 'red'), 4, 10, {'neck': (4, 0)}, 'body')
        self.head = LogicalFrame('stand1/0/head', Image.new('RGBA', (8, 8), 'blue'), 4, 8, {'neck': (4, 8)}, 'head')
        self.required = {self.body.path, self.head.path}

    def test_missing_head_is_rejected(self):
        self.assertIsNone(_compose([self.body], self.required))

    def test_disconnected_head_is_rejected(self):
        self.head.anchors = {'unrelated': (0, 0)}
        self.assertIsNone(_compose([self.body, self.head], self.required))

    def test_connected_head_and_body_are_kept(self):
        result = _compose([self.body, self.head], self.required)
        self.assertIsNotNone(result)
        colors = set(result.image.getdata())
        self.assertIn((255, 0, 0, 255), colors)
        self.assertIn((0, 0, 255, 255), colors)


if __name__ == '__main__':
    unittest.main()
