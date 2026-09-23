"""Render ten client-asset scenes at 1280x720, with optional YOLO review labels."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import random
import zipfile

from PIL import Image, ImageDraw, ImageFont

from .assets import AssetStore
from .distractors import DistractorLibrary
from .render import MapRenderer, PlacedMonster, PlacedOverlay, SpriteLibrary, foothold_segments, platform_y

MAP_IDS = [
    '100010000', '100040000', '101010000', '101020000', '102020000',
    '103010000', '104000100', '104010001', '105040000', '106000000',
]
WIDTH, HEIGHT = 1280, 720


class PreviewRenderer(MapRenderer):
    """Static backdrop approximation: viewport-relative parallax and tiled origins."""

    def _draw_back(self, canvas, item, camera_x, camera_y):
        name = item.get('bS')
        if not name:
            return
        number = str(item.get('no', 0))
        candidates = [f'ani/{number}/0', f'back/{number}'] if item.get('ani') else [f'back/{number}', f'ani/{number}/0']
        frame = self.sprites.frame(name, '/Map/Back/', candidates)
        if frame is None:
            return
        sprite = frame.image
        flip = bool(item.get('f', 0))
        if flip:
            sprite = sprite.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        alpha = int(item.get('a', 255))
        if alpha < 255:
            sprite = sprite.copy()
            sprite.putalpha(sprite.getchannel('A').point(lambda value: value * alpha // 255))
        kind = int(item.get('type', 0))
        # Moving backgrounds are frozen at t=0; scrolling speed is not camera parallax.
        rx = 0 if kind in (4, 6) else int(item.get('rx', 0))
        ry = 0 if kind in (5, 7) else int(item.get('ry', 0))
        origin_x = sprite.width - frame.origin_x if flip else frame.origin_x
        base_x = round(int(item.get('x', 0)) + self.width / 2 + (camera_x + self.width / 2) * rx / 100 - origin_x)
        base_y = round(int(item.get('y', 0)) + self.height / 2 + (camera_y + self.height / 2) * ry / 100 - frame.origin_y)

        def positions(base, step, extent, sprite_extent, tiled):
            if not tiled:
                return [base]
            start = math.floor((-sprite_extent - base) / step) + 1
            end = math.ceil((extent - base) / step)
            return [base + i * step for i in range(start, end)]

        xs = positions(base_x, max(1, abs(int(item.get('cx', 0))) or sprite.width), self.width, sprite.width, kind in (1, 3, 4, 6, 7))
        ys = positions(base_y, max(1, abs(int(item.get('cy', 0))) or sprite.height), self.height, sprite.height, kind in (2, 3, 5, 6, 7))
        for y in ys:
            for x in xs:
                canvas.alpha_composite(sprite, (x, y))


def camera_for(root, seed):
    lives = [v for v in root.get('life', {}).values() if v.get('type') == 'm']
    mini = root.get('miniMap', {})
    segments = foothold_segments(root)
    xs = [x for s in segments.values() for x in (s[0], s[2])]
    ys = [y for s in segments.values() for y in (s[1], s[3])]
    left = -int(mini.get('centerX', -min(xs)))
    top = -int(mini.get('centerY', -(min(ys) - 250)))
    right = left + int(mini.get('width', max(xs) - left + 80))
    bottom = top + int(mini.get('height', max(ys) - top + 120))
    rng = random.Random(seed)
    candidates = []
    for life in lives:
        for screen_ground in (440, 510, 570):
            x = round(float(life.get('x', 0)) - WIDTH / 2)
            y = round(float(life.get('cy', life.get('y', 0))) - screen_ground)
            x = max(left, min(right - WIDTH, x)) if right - left >= WIDTH else round((left + right - WIDTH) / 2)
            y = max(top, min(bottom - HEIGHT, y)) if bottom - top >= HEIGHT else bottom - HEIGHT
            visible = [v for v in lives if x + 50 < float(v.get('x', 0)) < x + WIDTH - 50 and y + 130 < float(v.get('cy', v.get('y', 0))) < y + HEIGHT - 60]
            levels = len({round(float(v.get('cy', v.get('y', 0))) / 100) for v in visible})
            score = min(len(visible), 12) + min(levels, 3) * 1.5 + rng.random()
            candidates.append((score, x, y))
    if not candidates:
        raise RuntimeError('Map has no monster camera candidates')
    _, x, y = max(candidates)
    return x, y


def actors_for(root, sprites, people, camera, seed):
    rng = random.Random(seed)
    cx, cy = camera
    segments = foothold_segments(root)
    monsters = []
    for life in root.get('life', {}).values():
        if life.get('type') != 'm':
            continue
        x = int(life.get('x', 0))
        fh = int(life.get('fh', -1))
        y = round(platform_y(segments[fh], x)) if fh in segments else int(life.get('cy', life.get('y', 0)))
        if not (cx + 45 <= x <= cx + WIDTH - 45 and cy + 120 <= y <= cy + HEIGHT - 25):
            continue
        mob_id = str(life.get('id', ''))
        frames = sprites.monster_frames(mob_id)
        frames = [f for f in frames if f.image.getchannel('A').getbbox() and f.image.width >= 4 and f.image.height >= 4]
        natural = [f for f in frames if f.path.lower().startswith(('stand/', 'move/', 'fly/'))]
        if not frames:
            continue
        frame = rng.choice(natural or frames)
        if any(abs(x - m.world_x) < 40 and abs(y - m.world_y) < 30 for m in monsters):
            continue
        monsters.append(PlacedMonster(mob_id, frame.image, x, y, bool(life.get('f', 0)), frame.path.split('/')[0], frame.origin_x, frame.origin_y, fh, frame.path))
    candidates = []
    for fh, s in segments.items():
        low, high = sorted((s[0], s[2]))
        low, high = max(low + 10, cx + 100), min(high - 10, cx + WIDTH - 100)
        if high - low < 20 or abs(s[3] - s[1]) > 100:
            continue
        for fraction in (.25, .5, .75):
            x = round(low + (high - low) * fraction)
            y = round(platform_y(s, x))
            if cy + 200 < y < cy + HEIGHT - 55:
                distance = min((abs(x - m.world_x) + abs(y - m.world_y) for m in monsters), default=200)
                score = -abs(x - cx - 640) / 8 - abs(y - cy - 490) / 5 + min(distance, 120)
                candidates.append((score, x, y, fh))
    overlays = []
    for _, x, y, fh in sorted(candidates, reverse=True):
        frame = people.player_frame(rng)
        if frame is not None:
            overlays.append(PlacedOverlay(frame.image, x, y, frame.origin_x, frame.origin_y, 'player', False, 'composite_player', frame.path, foothold_id=fh))
            break
    return monsters, overlays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--client-root', default=r'D:\Program Files\上海数龙科技有限公司\冒险岛online\mxdclassic')
    parser.add_argument('--cache-dir', default=r'F:\MapleStoryAssets\.classic_cache\preview_index_1.15.2')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=20260920)
    args = parser.parse_args()
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for name in ('images', 'labels', 'visual_checks'):
        (output / name).mkdir(exist_ok=True)
    store = AssetStore(args.client_root, args.cache_dir)
    print('Loading client asset index...', flush=True)
    store.build_index(include_sprites=True)
    sprites = SpriteLibrary(store, 128)
    people = DistractorLibrary(sprites)
    renderer = PreviewRenderer(sprites, WIDTH, HEIGHT)
    records = []
    try:
        for i, mid in enumerate(MAP_IDS, 1):
            print(f'Rendering {i}/10 map {mid}', flush=True)
            root = next(iter(store.read_wzjson(mid, '/Map/Map/').values()))
            camera = camera_for(root, args.seed + i)
            monsters, overlays = actors_for(root, sprites, people, camera, args.seed + i)
            image, labels = renderer.render_scene(root, *camera, monsters, overlays=overlays, minimum_visible_fraction=.30, label_players=True)
            counts = Counter(label['class_id'] for label in labels)
            if not counts[0] or not counts[1]:
                raise RuntimeError(f'{mid}: both classes required, got {counts}')
            stem = f'{i:02d}_{mid}'
            image.save(output / 'images' / f'{stem}.png')
            rows = []
            annotated = image.copy()
            draw = ImageDraw.Draw(annotated)
            for label in labels:
                x, y, w, h = label['bbox']
                rows.append(f"{label['class_id']} {(x+w/2)/WIDTH:.8f} {(y+h/2)/HEIGHT:.8f} {w/WIDTH:.8f} {h/HEIGHT:.8f}")
                color = '#ff5c5c' if label['class_id'] == 0 else '#339cff'
                draw.rectangle((x, y, x+w, y+h), outline=color, width=2)
                draw.text((x, max(0, y-14)), label['class_name'], fill=color, stroke_width=1, stroke_fill='black')
            (output / 'labels' / f'{stem}.txt').write_text('\n'.join(rows) + '\n', encoding='utf-8')
            annotated.save(output / 'visual_checks' / f'{stem}.jpg', quality=95, subsampling=0)
            records.append({'stem': stem, 'map_id': mid, 'camera': camera, 'width': WIDTH, 'height': HEIGHT, 'labels': labels, 'monster_count': counts[0], 'player_count': counts[1], 'backgrounds': sorted({v['bS'] for v in root.get('back', {}).values() if v.get('bS')})})
            print(f"Saved {stem}: {counts[0]} monsters, {counts[1]} player", flush=True)
        (output / 'manifest.json').write_text(json.dumps({'client_catalog': store._catalog_version(), 'seed': args.seed, 'rendering': 'Offline client-asset composition; static backdrop parallax approximation; no game UI; not live screenshots', 'records': records, 'missing_assets': sprites.missing}, ensure_ascii=False, indent=2), encoding='utf-8')
        sheet = Image.new('RGB', (1328, 2072), '#101a27')
        draw = ImageDraw.Draw(sheet)
        font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 20)
        for i, record in enumerate(records):
            with Image.open(output / 'images' / f"{record['stem']}.png") as im:
                thumb = im.resize((640, 360), Image.Resampling.LANCZOS)
            x, y = 16 + (i % 2) * 656, 16 + (i // 2) * 410
            sheet.paste(thumb, (x, y))
            draw.text((x + 3, y + 368), f"{i+1:02d}  地图 {record['map_id']}  |  怪物 {record['monster_count']} · 人物 {record['player_count']}", font=font, fill='#e2e8f0')
        sheet.save(output / 'contact_sheet.jpg', quality=92, subsampling=0)
        (output / 'README.txt').write_text(
            '10 张 1280×720 客户端原始素材离线合成场景（不是实时游戏截图）。\n'
            'images：无框 PNG；labels：同名 YOLO，0=monster，1=player；visual_checks：画框检查图。\n'
            '保留真实地图、图层、平台、怪物出生点；添加组合人物。背景视差为静态近似，无游戏 UI、动态光效或完整交互物。\n'
            '这是预览样本，没有 train/val/test 拆分，不能使用原固定 1280×224 校验器。\n'
            'manifest.json：地图 ID、镜头世界坐标、标注、客户端版本和缺失资源记录。\n', encoding='utf-8')
        archive = output / 'scenes_1280x720_10.zip'
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
            for file in sorted(output.rglob('*')):
                if file.is_file() and file != archive:
                    z.write(file, file.relative_to(output))
        print(f'DONE: {output}', flush=True)
    finally:
        store.close()


if __name__ == '__main__':
    main()
