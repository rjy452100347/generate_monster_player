"""Generate the resumable 40K scattered monster/player dataset."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import time
from typing import Any

from PIL import Image, ImageDraw
import yaml

from .assets import AssetStore
from .distractors import DistractorLibrary
from .render import MapRenderer, SpriteLibrary
from .scattered_preview import complete_players, make_scene, pet_catalog


@dataclass(frozen=True)
class PlannedSample:
    ordinal: int
    split: str
    split_index: int
    scenario: str
    monster_count: int
    pet_count: int

    @property
    def stem(self) -> str:
        return f"{self.split}_{self.split_index:06d}"


def _scaled_quota(quota: dict[Any, int], total: int) -> dict[Any, int]:
    source_total = sum(int(value) for value in quota.values())
    raw = {key: int(value) * total / source_total for key, value in quota.items()}
    result = {key: int(value) for key, value in raw.items()}
    for key in sorted(raw, key=lambda item: (raw[item] - result[item], str(item)), reverse=True)[: total - sum(result.values())]:
        result[key] += 1
    return result


def build_schedule(config: dict[str, Any], limit: int | None = None) -> list[PlannedSample]:
    full_total = sum(int(value) for value in config['splits'].values())
    total = limit or full_total
    split_quota = _scaled_quota(config['splits'], total)
    global_scenarios = _scaled_quota(config['scenarios'], total)
    global_monsters = _scaled_quota(config['monster_quotas'], sum(value for key, value in global_scenarios.items() if 'monster' in key))
    global_two_pets = round(int(config['two_pet_images']) * total / full_total)

    rng = random.Random(int(config['seed']) * 1009)
    scenarios = [name for name, count in global_scenarios.items() for _ in range(count)]
    splits = [name for name, count in split_quota.items() for _ in range(count)]
    counts = [int(count) for count, amount in global_monsters.items() for _ in range(amount)]
    rng.shuffle(scenarios)
    rng.shuffle(splits)
    rng.shuffle(counts)
    pet_indexes = [index for index, scenario in enumerate(scenarios) if 'pet' in scenario]
    two_pets = set(rng.sample(pet_indexes, min(global_two_pets, len(pet_indexes))))
    split_indexes = Counter()
    schedule: list[PlannedSample] = []
    for ordinal, (scenario, split) in enumerate(zip(scenarios, splits)):
        monster_count = counts.pop() if 'monster' in scenario else 0
        pet_count = (2 if ordinal in two_pets else 1) if 'pet' in scenario else 0
        schedule.append(PlannedSample(ordinal, split, split_indexes[split], scenario, monster_count, pet_count))
        split_indexes[split] += 1
    if counts:
        raise RuntimeError('unassigned monster quotas')
    return schedule


class Writer:
    def __init__(self, root: Path, config: dict[str, Any]):
        self.root = root
        self.config = config
        self.manifest_path = root / 'scenario_manifest.jsonl'
        self.records: dict[str, dict[str, Any]] = {}
        self.hashes: set[str] = set()
        self.visual_counts: Counter[tuple[str, str]] = Counter()
        for split in ('train', 'val', 'test'):
            (root / 'images' / split).mkdir(parents=True, exist_ok=True)
            (root / 'labels' / split).mkdir(parents=True, exist_ok=True)
        for sub in ('annotations', 'visual_checks', 'catalogs'):
            (root / sub).mkdir(parents=True, exist_ok=True)
        valid_lines: list[str] = []
        if self.manifest_path.exists():
            for line in self.manifest_path.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                record = json.loads(line)
                if not (root / record['image']).exists() or not (root / record['label']).exists():
                    continue
                valid_lines.append(line)
                self.records[record['stem']] = record
                self.hashes.add(record['sha256'])
                if record.get('visual_check'):
                    self.visual_counts[(record['split'], record['scenario'])] += 1
        existing_lines = sum(1 for line in self.manifest_path.read_text(encoding='utf-8').splitlines() if line.strip()) if self.manifest_path.exists() else 0
        if existing_lines != len(valid_lines):
            temp = self.manifest_path.with_suffix('.jsonl.tmp')
            temp.write_text(('\n'.join(valid_lines) + '\n') if valid_lines else '', encoding='utf-8')
            temp.replace(self.manifest_path)

    def add(self, sample: PlannedSample, image: Image.Image, labels: list[dict[str, Any]], metadata: dict[str, Any]) -> bool:
        from io import BytesIO
        buffer = BytesIO()
        image.save(buffer, 'JPEG', quality=int(self.config['jpeg_quality']), subsampling=0, optimize=False)
        payload = buffer.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        if digest in self.hashes:
            return False
        rows = []
        width, height = image.size
        for label in labels:
            x, y, w, h = label['bbox']
            rows.append(f"{label['class_id']} {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}")
        image_rel = Path('images') / sample.split / f'{sample.stem}.jpg'
        label_rel = Path('labels') / sample.split / f'{sample.stem}.txt'
        image_path, label_path = self.root / image_rel, self.root / label_rel
        image_tmp, label_tmp = image_path.with_suffix('.jpg.tmp'), label_path.with_suffix('.txt.tmp')
        image_tmp.write_bytes(payload)
        label_tmp.write_text('\n'.join(rows), encoding='utf-8')
        image_tmp.replace(image_path)
        label_tmp.replace(label_path)

        visual = self.visual_counts[(sample.split, sample.scenario)] < int(self.config['visual_checks_per_split_scenario'])
        if visual:
            check = image.copy()
            draw = ImageDraw.Draw(check)
            for label in labels:
                x, y, w, h = label['bbox']
                color = '#ff4545' if label['class_id'] == 0 else '#399bff'
                draw.rectangle((x, y, x+w, y+h), outline=color, width=2)
                draw.text((x, max(0, y-13)), label['class_name'], fill=color, stroke_width=1, stroke_fill='black')
            check.save(self.root / 'visual_checks' / f'{sample.stem}_{sample.scenario}.jpg', quality=94, subsampling=0)
            self.visual_counts[(sample.split, sample.scenario)] += 1

        record = {
            'stem': sample.stem, 'ordinal': sample.ordinal, 'split': sample.split,
            'split_index': sample.split_index, 'scenario': sample.scenario,
            'monster_count': sample.monster_count,
            'player_count': int('player' in sample.scenario), 'pet_count': sample.pet_count,
            'image': image_rel.as_posix(), 'label': label_rel.as_posix(),
            'sha256': digest, 'visual_check': visual, 'labels': labels, **metadata,
        }
        manifest_line = json.dumps(record, ensure_ascii=False, separators=(',', ':')) + '\n'
        for io_attempt in range(20):
            try:
                with self.manifest_path.open('a', encoding='utf-8') as handle:
                    handle.write(manifest_line)
                    handle.flush()
                    os.fsync(handle.fileno())
                break
            except PermissionError:
                if io_attempt == 19:
                    raise
                time.sleep(0.25)
        self.records[sample.stem] = record
        self.hashes.add(digest)
        return True

    def finish(self) -> dict[str, Any]:
        ordered = sorted(self.records.values(), key=lambda record: record['ordinal'])
        class_names = list(self.config['class_names'])
        for split in ('train', 'val', 'test'):
            subset = [record for record in ordered if record['split'] == split]
            images, annotations, annotation_id = [], [], 1
            for image_id, record in enumerate(subset, 1):
                images.append({'id': image_id, 'file_name': record['image'], 'width': 1280, 'height': 224, 'map_id': record['map_id'], 'scenario': record['scenario']})
                for label in record['labels']:
                    x, y, w, h = label['bbox']
                    annotations.append({'id': annotation_id, 'image_id': image_id, 'category_id': int(label['class_id']) + 1, 'bbox': [x, y, w, h], 'area': w*h, 'iscrowd': 0})
                    annotation_id += 1
            payload = {'images': images, 'annotations': annotations, 'categories': [{'id': i+1, 'name': name} for i, name in enumerate(class_names)]}
            (self.root / 'annotations' / f'instances_{split}.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')), encoding='utf-8')
        (self.root / 'data.yaml').write_text(
            f"path: {self.root.as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + ''.join(f'  {i}: {name}\n' for i, name in enumerate(class_names)), encoding='utf-8')
        report = {
            'ok': True, 'images': len(ordered), 'image_size': [1280, 224],
            'splits': dict(Counter(record['split'] for record in ordered)),
            'scenarios': dict(Counter(record['scenario'] for record in ordered)),
            'monster_count_quotas': dict(Counter(str(record['monster_count']) for record in ordered)),
            'monster_boxes': sum(record['monster_count'] for record in ordered),
            'player_boxes': sum(record['player_count'] for record in ordered),
            'pet_instances': sum(record['pet_count'] for record in ordered),
            'unique_maps': len({record['map_id'] for record in ordered}),
            'unique_player_appearances': len({record['player']['sha256'] for record in ordered if record.get('player')}),
            'pet_ids': sorted({pet['pet_id'] for record in ordered for pet in record.get('pets', [])}),
        }
        (self.root / 'generation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        return report


def validate(root: Path, config: dict[str, Any], schedule: list[PlannedSample], records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    expected = {sample.stem: sample for sample in schedule}
    if set(records) != set(expected):
        errors.append(f'manifest stems differ: expected {len(expected)}, got {len(records)}')
    map_selection = json.loads((root / 'map_selection.json').read_text(encoding='utf-8'))
    map_sets = {split: set(values) for split, values in map_selection['splits'].items()}
    if map_sets['train'] & map_sets['val'] or map_sets['train'] & map_sets['test'] or map_sets['val'] & map_sets['test']:
        errors.append('map split leakage')
    seen_hashes: set[str] = set()
    for stem, record in records.items():
        sample = expected.get(stem)
        if not sample:
            continue
        image_path, label_path = root / record['image'], root / record['label']
        if not image_path.exists() or not label_path.exists():
            errors.append(f'missing files {stem}')
            continue
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        if digest != record['sha256'] or digest in seen_hashes:
            errors.append(f'hash mismatch/duplicate {stem}')
        seen_hashes.add(digest)
        try:
            with Image.open(image_path) as image:
                if image.size != (1280, 224) or image.format != 'JPEG':
                    errors.append(f'image format/size {stem}')
                image.verify()
        except Exception as exc:
            errors.append(f'image decode {stem}: {exc}')
        rows = [line for line in label_path.read_text(encoding='utf-8').splitlines() if line.strip()]
        if len(rows) != len(record['labels']):
            errors.append(f'label count {stem}')
        for row in rows:
            fields = row.split()
            if len(fields) != 5 or fields[0] not in {'0', '1'} or any(not 0 <= float(value) <= 1 for value in fields[1:]):
                errors.append(f'invalid YOLO row {stem}')
                break
        if record['split'] != sample.split or record['scenario'] != sample.scenario or record['monster_count'] != sample.monster_count or record['pet_count'] != sample.pet_count:
            errors.append(f'plan mismatch {stem}')
        if record['map_id'] not in map_sets[record['split']]:
            errors.append(f'map split mismatch {stem}')
        actor_boxes = record['actor_boxes']
        if len(actor_boxes) != sample.monster_count + int('player' in sample.scenario) + sample.pet_count:
            errors.append(f'actor count {stem}')
        if len({pet['pet_id'] for pet in record.get('pets', [])}) != sample.pet_count:
            errors.append(f'repeated pet {stem}')
        if 'player' in sample.scenario and not record['player']['all_required_parts_connected']:
            errors.append(f'incomplete player {stem}')
        if errors and len(errors) >= 100:
            break
    report = {'ok': not errors, 'errors': errors, 'checked_records': len(records)}
    (root / 'validation_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    if errors:
        raise RuntimeError(errors[:10])
    return report


def generate(config_path: Path, limit: int | None = None, name: str | None = None, output_root: str | None = None) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
    if name:
        config['name'] = name
    if output_root:
        config['output_root'] = output_root
    output = Path(config['output_root']) / config['name']
    output.mkdir(parents=True, exist_ok=True)
    config_copy = output / 'generation_config.yml'
    rendered_config = yaml.safe_dump(config, allow_unicode=True, sort_keys=False)
    if config_copy.exists() and config_copy.read_text(encoding='utf-8') != rendered_config:
        raise RuntimeError('Existing dataset was created with a different configuration')
    config_copy.write_text(rendered_config, encoding='utf-8')
    selection_source = Path(config['map_selection'])
    selection_target = output / 'map_selection.json'
    if selection_target.exists() and selection_target.read_bytes() != selection_source.read_bytes():
        raise RuntimeError('Existing dataset uses a different map selection')
    if not selection_target.exists():
        shutil.copy2(selection_source, selection_target)
    selection = json.loads(selection_target.read_text(encoding='utf-8'))
    schedule = build_schedule(config, limit)
    plan = {'total': len(schedule), 'splits': dict(Counter(sample.split for sample in schedule)), 'scenarios': dict(Counter(sample.scenario for sample in schedule)), 'monster_counts': dict(Counter(str(sample.monster_count) for sample in schedule)), 'pet_counts': dict(Counter(str(sample.pet_count) for sample in schedule))}
    plan_path = output / 'generation_plan.json'
    rendered_plan = json.dumps(plan, ensure_ascii=False, indent=2)
    if plan_path.exists() and plan_path.read_text(encoding='utf-8') != rendered_plan:
        raise RuntimeError('Existing dataset has a different generation plan')
    plan_path.write_text(rendered_plan, encoding='utf-8')

    writer = Writer(output, config)
    store = AssetStore(config['client_root'], config['cache_dir'])
    store.build_index()
    sprites = SpriteLibrary(store, int(config.get('resource_cache', 384)))
    library = DistractorLibrary(sprites)
    catalog_rng = random.Random(int(config['seed']))
    print('Building complete player catalog...', flush=True)
    players = complete_players(library, catalog_rng, int(config['player_catalog_size']))
    print('Building pet catalog...', flush=True)
    pets, rejected_pets = pet_catalog(library)
    assets_path = output / 'catalogs' / 'assets.json'
    assets_path.write_text(json.dumps({'players': [p.details for p in players], 'pets': [p.details for p in pets], 'rejected_pets': rejected_pets}, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Catalog ready: {len(players)} players, {len(pets)} pet appearances; resuming at {len(writer.records)}/{len(schedule)}', flush=True)

    roots: dict[str, dict[str, Any]] = {}
    renderer = MapRenderer(sprites, 1280, 224)
    map_uses = Counter((record['split'], record['map_id']) for record in writer.records.values())
    successful_maps = Counter(
        (record['split'], record['scenario'], record['monster_count'], record['pet_count'], record['map_id'])
        for record in writer.records.values()
    )
    ranks = {split: {map_id: index for index, map_id in enumerate(random.Random(int(config['seed']) + offset).sample(values, len(values)))} for offset, (split, values) in enumerate(selection['splits'].items(), 1)}
    try:
        for completed, sample in enumerate(schedule, 1):
            if sample.stem in writer.records:
                continue
            if completed % 100 == 1:
                free_gb = shutil.disk_usage(output).free / (1024 ** 3)
                if free_gb < float(config['minimum_free_gb']):
                    raise RuntimeError(f'Free space {free_gb:.2f} GB below safety threshold')
            rng = random.Random(int(config['seed']) * 1_000_003 + sample.ordinal)
            player = players[(sample.ordinal * 131 + 17) % len(players)] if 'player' in sample.scenario else None
            selected_pets = []
            first_pet = (sample.ordinal * 37 + 11) % len(pets)
            for pet_offset in range(sample.pet_count):
                selected_pets.append(pets[(first_pet + pet_offset * 43) % len(pets)])
            candidates = sorted(
                selection['splits'][sample.split],
                key=lambda map_id: (
                    0 if successful_maps[(sample.split, sample.scenario, sample.monster_count, sample.pet_count, map_id)] else 1,
                    map_uses[(sample.split, map_id)], ranks[sample.split][map_id],
                ),
            )
            chosen = None
            attempt_budget = int(config['max_attempts']) * 5
            for attempt in range(attempt_budget):
                map_id = candidates[attempt % len(candidates)]
                if map_id not in roots:
                    roots[map_id] = next(iter(store.read_wzjson(map_id, '/Map/Map/').values()))
                chosen = make_scene(roots[map_id], sprites, renderer, sample.monster_count, player, selected_pets, rng)
                if chosen is not None:
                    break
            if chosen is None:
                raise RuntimeError(f'Cannot generate {sample.stem} after {attempt_budget} attempts')
            image, labels, metadata = chosen
            metadata.update({'map_id': map_id, 'player': player.details if player else None, 'pets': [pet.details for pet in selected_pets], 'attempts': attempt + 1})
            if not writer.add(sample, image, labels, metadata):
                # Exact duplicates are retried deterministically with a new RNG stream.
                retry_rng = random.Random(int(config['seed']) * 2_000_003 + sample.ordinal)
                for duplicate_attempt in range(int(config['max_attempts'])):
                    map_id = candidates[(attempt + duplicate_attempt + 1) % len(candidates)]
                    if map_id not in roots:
                        roots[map_id] = next(iter(store.read_wzjson(map_id, '/Map/Map/').values()))
                    chosen = make_scene(roots[map_id], sprites, renderer, sample.monster_count, player, selected_pets, retry_rng)
                    if chosen is not None:
                        image, labels, metadata = chosen
                        metadata.update({'map_id': map_id, 'player': player.details if player else None, 'pets': [pet.details for pet in selected_pets], 'attempts': attempt + duplicate_attempt + 2})
                        if writer.add(sample, image, labels, metadata):
                            break
                else:
                    raise RuntimeError(f'Cannot create unique image for {sample.stem}')
            map_uses[(sample.split, map_id)] += 1
            successful_maps[(sample.split, sample.scenario, sample.monster_count, sample.pet_count, map_id)] += 1
            if completed <= 10 or completed % 25 == 0:
                print(f'Generated {completed}/{len(schedule)} {sample.stem} {sample.scenario}; map {map_id}; attempts {metadata["attempts"]}', flush=True)
        report = writer.finish()
        validate(output, config, schedule, writer.records)
        (output / 'asset_loading_report.json').write_text(json.dumps({'client_catalog': store._catalog_version(), 'accepted_pet_visuals': len(pets), 'excluded_pet_ids': rejected_pets, 'resource_warnings': sprites.missing}, ensure_ascii=False, indent=2), encoding='utf-8')
        (output / 'README.txt').write_text(
            '4万张分散人物/怪物两分类数据集，1280×224 JPEG。\n0=monster，1=player；宠物为无标签干扰物。\ntrain/val/test=32000/4000/4000，地图拆分互不交叉。\n使用 data.yaml 训练；scenario_manifest.jsonl 可用于追踪每张图的来源和隐藏宠物。\n重复执行生成命令会从完整的图片、标签和 Manifest 记录处继续。\n', encoding='utf-8')
        print(json.dumps(report, ensure_ascii=False), flush=True)
        print(f'DONE: {output}', flush=True)
        return output
    finally:
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/scattered_40k_2class.yml')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--name')
    parser.add_argument('--output-root')
    args = parser.parse_args()
    generate(Path(args.config), args.limit, args.name, args.output_root)


if __name__ == '__main__':
    main()
