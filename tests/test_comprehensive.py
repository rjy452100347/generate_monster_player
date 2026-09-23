from collections import Counter
from pathlib import Path
import unittest

import yaml

from classic_dataset.comprehensive import build_schedule


class ComprehensiveScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load((Path(__file__).parents[1] / "configs" / "comprehensive_60k.yml").read_text(encoding="utf-8"))
        cls.schedule = build_schedule(cls.config)

    def test_exact_global_quotas(self):
        self.assertEqual(len(self.schedule), 60_000)
        self.assertEqual(Counter(item.split for item in self.schedule), Counter({"train": 48_000, "val": 6_000, "test": 6_000}))
        self.assertEqual(Counter(item.target_count for item in self.schedule), Counter({int(k): v for k, v in self.config["box_quotas"].items()}))
        self.assertEqual(Counter(item.scenario for item in self.schedule), Counter({k: v["count"] for k, v in self.config["scenarios"].items()}))

    def test_scaled_smoke_schedule_is_valid(self):
        smoke = build_schedule(self.config, 120)
        self.assertEqual(len(smoke), 120)
        self.assertTrue(all(0 <= item.target_count <= 20 for item in smoke))
        self.assertEqual(sum(item.target_count == 0 for item in smoke), 24)


class HardNegativeScheduleTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = yaml.safe_load(
            (Path(__file__).parents[1] / "configs" / "hard_negative_48k_2class.yml").read_text(encoding="utf-8")
        )
        cls.schedule = build_schedule(cls.config)

    def test_exact_hard_negative_quotas(self):
        self.assertEqual(len(self.schedule), 48_000)
        self.assertEqual(Counter(item.split for item in self.schedule), Counter({"train": 38_400, "val": 4_800, "test": 4_800}))
        self.assertEqual(sum(item.target_count > 0 for item in self.schedule), 26_000)
        self.assertEqual(sum(item.loot_count > 0 for item in self.schedule), 40_000)
        self.assertEqual(Counter(item.loot_count for item in self.schedule if item.loot_count), Counter({int(k): v for k, v in self.config["loot_count_quotas"].items()}))
        self.assertEqual(Counter(item.loot_layout for item in self.schedule if item.loot_count), Counter(self.config["loot_layout_quotas"]))

    def test_item_instance_capacity_and_layouts(self):
        for split in ("train", "val", "test"):
            instances = sum(item.loot_count for item in self.schedule if item.split == split)
            expected = int(self.config["item_visual_count"]) * int(self.config[f"minimum_item_{split}_uses"])
            if split in {"val", "test"}:
                expected += int(self.config["item_coverage_reserve_per_split"])
            self.assertGreaterEqual(instances, expected)
        self.assertTrue(all(item.target_layout == "distributed" for item in self.schedule if item.target_count == 1))
        eligible = [item for item in self.schedule if item.target_count > 4]
        self.assertGreaterEqual(sum(item.multi_platform for item in eligible) / len(eligible), 0.19)


if __name__ == "__main__":
    unittest.main()
