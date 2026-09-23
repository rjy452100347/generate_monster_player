"""Generate the agreed 100-image review set; never starts the 40K job."""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import html
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import yaml

from .assets import AssetStore
from .distractors import DistractorLibrary, _compose
from .render import LogicalFrame, MapRenderer, PlacedOverlay, SpriteLibrary, build_monsters, foothold_segments, platform_y

W, H = 1280, 224


@dataclass
class ActorVisual:
    identity: str
    frame: LogicalFrame
    details: dict


def digest(image):
    return hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()


def complete_players(library, rng, count):
    sprites = library.sprites
    result, seen = [], set()
    for attempt in range(count * 100):
        body_id = rng.choice(library.body_names[:5])
        head_id = f'{int(body_id) + 10000:08d}'
        body = sprites.resource(body_id, '/Character/')
        poses = sorted({p.rsplit('/', 1)[0] for p in body if p.endswith('/body') and p.startswith(('stand1/', 'stand2/', 'walk1/', 'walk2/', 'alert/'))})
        if not poses:
            continue
        pose = rng.choice(poses)
        parts = [f for p, f in body.items() if p.startswith(pose + '/') and f.anchors]
        head = sprites.resource(head_id, '/Character/').get(pose + '/head')
        if not head or not any(f.path.endswith('/body') for f in parts) or not any(f.path.rsplit('/', 1)[-1] in {'arm', 'lHand', 'rHand'} for f in parts):
            continue
        parts.append(head)
        sources = {'body': body_id, 'head': head_id}
        face_id = rng.choice(library.face_names)
        face = sprites.resource(face_id, '/Character/Face/').get('default/face')
        if not face or not face.anchors:
            continue
        parts.append(face)
        sources['face'] = face_id
        failed = False
        # Hair, torso clothing, trousers and shoes are mandatory for this preview.
        groups = [('/Character/Hair/', library.hair_names), *[(g, library.equipment[g]) for g in ('/Character/Coat/', '/Character/Pants/', '/Character/Shoes/')]]
        if rng.random() < .6:
            groups.append(('/Character/Cap/', library.equipment['/Character/Cap/']))
        for group, names in groups:
            matched = []
            for _ in range(12):
                resource_id = rng.choice(names)
                matched = [f for p, f in sprites.resource(resource_id, group).items() if p.startswith(pose + '/') and f.anchors]
                if matched:
                    sources[group] = resource_id
                    break
            if not matched:
                failed = True
                break
            parts.extend(matched)
        if failed:
            continue
        required = {f.path for f in parts}
        composed = _compose(parts, required_paths=required)
        if not composed or composed.image.height > 180 or composed.image.width > 180:
            continue
        pixels = np.array(composed.image.getchannel('A')) > 0
        # Reject isolated head/body groups separated by an entirely empty row.
        rows = np.flatnonzero(pixels.any(axis=1))
        if len(rows) < 35 or np.any(np.diff(rows) > 3):
            continue
        key = digest(composed.image)
        if key in seen:
            continue
        seen.add(key)
        composed.path = pose
        result.append(ActorVisual(f'player_{len(result):03d}', composed, {'pose': pose, 'sources': sources, 'required_parts': sorted(required), 'all_required_parts_connected': True, 'sha256': key}))
        if len(result) == count:
            return result
    raise RuntimeError(f'Only {len(result)}/{count} complete player composites accepted')


def pet_catalog(library):
    result, rejected, seen = [], [], set()
    for pet_id in library.pet_names:
        frames = library.sprites.resource(pet_id, '/Item/Pet/')
        candidates = sorted((f for p, f in frames.items() if p.split('/', 1)[0].startswith(('stand', 'move', 'fly'))), key=lambda f: (not f.path.startswith('stand'), f.path))
        chosen = None
        for f in candidates:
            box = f.image.getchannel('A').getbbox()
            if box and 8 <= box[2] - box[0] <= 200 and 8 <= box[3] - box[1] <= 160:
                key = digest(f.image)
                if key not in seen:
                    chosen = ActorVisual(pet_id, f, {'pet_id': pet_id, 'frame_path': f.path, 'visual_sha256': key})
                    seen.add(key)
                    break
        if chosen:
            result.append(chosen)
        else:
            rejected.append({'pet_id': pet_id, 'reason': 'No distinct, valid stand/move/fly frame fitting the viewport'})
    if not result:
        raise RuntimeError('No usable pet visuals')
    return result, rejected


def box_for(frame, x, y, cx, cy, flip=False):
    image = frame.image.transpose(Image.Transpose.FLIP_LEFT_RIGHT) if flip else frame.image
    origin = image.width - frame.origin_x if flip else frame.origin_x
    a, b, c, d = image.getchannel('A').getbbox()
    return [round(x - origin - cx + a), round(y - frame.origin_y - cy + b), c-a, d-b]


def inside(box):
    x, y, w, h = box
    return w >= 4 and h >= 4 and x >= 5 and y >= 5 and x+w <= W-5 and y+h <= H-5


def separated(a, b, gap=24):
    x, y, w, h = a
    u, v, q, r = b
    return x+w+gap <= u or u+q+gap <= x or y+h+gap <= v or v+r+gap <= y


def place_visual(root, visual, kind, camera, occupied, rng):
    cx, cy = camera
    choices = []
    for fh, segment in foothold_segments(root).items():
        if segment[0] == segment[2] or abs(segment[3]-segment[1]) > 70:
            continue
        lo, hi = sorted((segment[0], segment[2]))
        lo, hi = max(lo+3, cx+30), min(hi-3, cx+W-30)
        if lo > hi:
            continue
        for x in (lo, (lo+hi)//2, hi):
            y = round(platform_y(segment, x))
            flip = bool(rng.getrandbits(1))
            box = box_for(visual.frame, x, y, cx, cy, flip)
            if inside(box) and all(separated(box, old) for old in occupied):
                choices.append((rng.random(), x, y, fh, flip, box))
    if not choices:
        return None
    _, x, y, fh, flip, box = max(choices)
    occupied.append(box)
    f = visual.frame
    return PlacedOverlay(f.image, x, y, f.origin_x, f.origin_y, kind, flip, visual.identity, f.path, foothold_id=fh)


def make_scene(root, sprites, renderer, count, player, pets, rng):
    monsters = build_monsters(root, sprites, rng, bool(count), max_count=5, target_count=count, placement='natural', view_width=W, view_height=H, action_family=rng.choice(('stand', 'move'))) if count else []
    if len(monsters) != count:
        return None
    # Spawn patrol ranges can cross several footholds. Do not extrapolate the
    # spawn's first slope beyond its actual endpoints into empty sky or cliffs.
    footholds = foothold_segments(root)
    for monster in monsters:
        segment = footholds.get(monster.foothold_id)
        if segment is None or segment[0] == segment[2]:
            return None
        lo, hi = sorted((segment[0], segment[2]))
        monster.world_x = max(lo, min(hi, monster.world_x))
        monster.world_y = platform_y(segment, monster.world_x)
    if monsters:
        cx = round((min(m.world_x for m in monsters)+max(m.world_x for m in monsters))/2 - W/2)
        ground = max(m.world_y for m in monsters)
    else:
        lives = [v for v in root.get('life', {}).values() if v.get('type') == 'm']
        if not lives:
            return None
        life = rng.choice(lives)
        cx = int(life.get('x', 0)) - W//2
        ground = int(life.get('cy', life.get('y', 0)))
    cy = round(ground - rng.randint(167, 195))
    occupied = []
    for m in monsters:
        f = LogicalFrame(m.frame_path, m.image, m.origin_x, m.origin_y)
        box = box_for(f, m.world_x, m.world_y, cx, cy, m.flip)
        if not inside(box) or any(not separated(box, old, 48) for old in occupied):
            return None
        occupied.append(box)
    if count > 1 and max(b[0]+b[2] for b in occupied)-min(b[0] for b in occupied) < W*.45:
        return None
    overlays = []
    for kind, visual in ([('player', player)] if player else []) + [('pet', p) for p in pets]:
        item = place_visual(root, visual, kind, (cx,cy), occupied, rng)
        if item is None:
            return None
        overlays.append(item)
    image, labels, hidden = renderer.render_scene(root, cx, cy, monsters, overlays=overlays, label_players=True, minimum_clip_fraction=1.0, minimum_visible_fraction=.98, return_hidden_objects=True)
    if Counter(x['class_id'] for x in labels) != Counter({0:count, 1:int(player is not None)}):
        return None
    if any(x.get('clip_fraction', 0) < .9999 or x.get('visible_fraction', 0) < .98 for x in labels):
        return None
    pet_objects = [o for o in hidden if o['kind']=='pet']
    if len(pet_objects) != len(pets) or any(o['clip_fraction'] < .9999 or o['visible_fraction'] < .98 for o in pet_objects):
        return None
    return image, labels, {'camera':[cx,cy], 'hidden_objects':hidden, 'actor_boxes':occupied}


def catalog_sheet(visuals, path, title, columns=10):
    cell_w, cell_h = 150, 210
    font = ImageFont.truetype('C:/Windows/Fonts/msyh.ttc', 15)
    canvas=Image.new('RGB',(columns*cell_w,50+((len(visuals)+columns-1)//columns)*cell_h),'#263447')
    draw=ImageDraw.Draw(canvas)
    draw.text((15,12),title,font=font,fill='white')
    for i, visual in enumerate(visuals):
        image=visual.frame.image.copy()
        image.thumbnail((cell_w-10,160),Image.Resampling.NEAREST)
        x=(i%columns)*cell_w; y=50+(i//columns)*cell_h
        canvas.paste(image,(x+(cell_w-image.width)//2,y+165-image.height),image)
        draw.text((x+5,y+170),visual.identity,font=font,fill='white')
        draw.text((x+5,y+189),visual.frame.path,font=font,fill='#b9d9fa')
    canvas.save(path)


def write_views(output, records):
    font=ImageFont.truetype('C:/Windows/Fonts/msyh.ttc',18)
    for page in range(10):
        sheet=Image.new('RGB',(1312,10*258+58),'#182434')
        draw=ImageDraw.Draw(sheet)
        draw.text((16,12),f'验收样本 {page*10+1:03d}–{page*10+10:03d} | 原图 1280×224 | 红框怪物 · 蓝框人物 · 宠物无框',font=font,fill='white')
        for i,record in enumerate(records[page*10:page*10+10]):
            y=52+i*258
            with Image.open(output/'visual_checks'/f"{record['stem']}.jpg") as im:
                sheet.paste(im,(16,y))
            draw.text((16,y+226),f"{record['stem']} | 地图 {record['map_id']} | 怪物 {record['monster_count']} · 人物 {record['player_count']} · 宠物 {record['pet_count']}",font=font,fill='#d7e8ff')
        sheet.save(output/'overviews'/f'page_{page+1:02d}.jpg',quality=94,subsampling=0)
    cards=[]
    for r in records:
        stem=r['stem']
        cards.append(f'<article><h2>{html.escape(stem)} · 怪物 {r["monster_count"]} / 人物 {r["player_count"]} / 宠物 {r["pet_count"]}</h2><a href="images/{stem}.png"><img src="images/{stem}.png" loading="lazy" alt="{stem}"></a><details><summary>查看标注</summary><img src="visual_checks/{stem}.jpg" loading="lazy"></details></article>')
    (output/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>100 张分散场景验收</title><style>body{background:#142030;color:#eef4ff;font:16px system-ui;max-width:1320px;margin:30px auto;padding:16px}h2{font-size:16px}article{margin:24px 0;padding:12px;background:#203047;border-radius:10px}img{display:block;width:100%;height:auto}summary{cursor:pointer;padding:10px}a{color:#8ac5ff}</style><h1>100 张分散场景验收</h1><p>1280×224 · 点击原图查看原始尺寸 · 宠物不标注 · 正式 4 万张尚未生成</p>'+''.join(cards),encoding='utf-8')


def validate_output(output, records):
    errors=[]
    images=list((output/'images').glob('*.png'))
    labels=list((output/'labels').glob('*.txt'))
    if len(images)!=100 or len(labels)!=100:
        errors.append('Image/label file count must be exactly 100')
    expected_scenarios = Counter(player_pet_monster=70, player_monster=15, player_pet=8, monster=5, background=2)
    if len(records) != 100 or Counter(r['scenario'] for r in records) != expected_scenarios:
        errors.append('Scenario quotas do not match the agreed 100-image review plan')
    hashes=set()
    for r in records:
        path=output/'images'/f"{r['stem']}.png"
        sha=hashlib.sha256(path.read_bytes()).hexdigest()
        if sha in hashes:errors.append(f"duplicate {r['stem']}")
        hashes.add(sha)
        with Image.open(path) as im:
            if im.size!=(W,H):errors.append(f"size {r['stem']}")
            im.verify()
        rows=(output/'labels'/f"{r['stem']}.txt").read_text().splitlines()
        if len(rows)!=len(r['labels']):errors.append(f"label count {r['stem']}")
        for row,expected in zip(rows,r['labels']):
            fields=row.split();cid=int(fields[0]);values=list(map(float,fields[1:]));x,y,w,h=expected['bbox']
            target=[(x+w/2)/W,(y+h/2)/H,w/W,h/H]
            if cid!=expected['class_id'] or cid not in (0,1) or len(values)!=4 or any(not 0<=v<=1 for v in values) or any(abs(a-b)>1e-7 for a,b in zip(values,target)):
                errors.append(f"YOLO mismatch {r['stem']}")
        boxes=r['actor_boxes']
        if any(not inside(b) for b in boxes) or any(not separated(a,b) for i,a in enumerate(boxes) for b in boxes[i+1:]):errors.append(f"actor clipping/spacing {r['stem']}")
        monsters=boxes[:r['monster_count']]
        if any(not separated(a,b,48) for i,a in enumerate(monsters) for b in monsters[i+1:]):errors.append(f"monster spacing {r['stem']}")
        if 'monster' in r['scenario'] and not 1 <= r['monster_count'] <= 5:errors.append(f"monster count {r['stem']}")
        if len({p['pet_id'] for p in r['pets']}) != r['pet_count']:errors.append(f"repeated pet in image {r['stem']}")
        if r['player_count'] and not r['player']['all_required_parts_connected']:errors.append(f"incomplete player {r['stem']}")
    report={'ok':not errors,'errors':errors,'images':len(images),'image_size':[W,H],'scenarios':dict(Counter(r['scenario'] for r in records)),'monster_count_quotas':dict(Counter(r['monster_count'] for r in records)),'monster_boxes':sum(r['monster_count'] for r in records),'player_boxes':sum(r['player_count'] for r in records),'pet_instances':sum(r['pet_count'] for r in records),'pet_ids':sorted({p['pet_id'] for r in records for p in r['pets']}),'unique_player_appearances':len({r['player']['sha256'] for r in records if r['player']}),'unique_maps':len({r['map_id'] for r in records})}
    (output/'validation_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    if errors:raise RuntimeError(errors)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/scattered_preview_100.yml')
    args=parser.parse_args()
    config=yaml.safe_load(Path(args.config).read_text(encoding='utf-8'))
    output=Path(config['output_root'])/config['name']
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Output is not empty; this review generator never overwrites existing samples')
    for sub in ('images','labels','visual_checks','overviews','catalogs'):(output/sub).mkdir(parents=True,exist_ok=True)
    (output/'generation_config.yml').write_text(yaml.safe_dump(config,allow_unicode=True,sort_keys=False),encoding='utf-8')
    store=AssetStore(config['client_root'],config['cache_dir']);store.build_index()
    sprites=SpriteLibrary(store,256);library=DistractorLibrary(sprites);rng=random.Random(config['seed'])
    print('Building complete player catalog...',flush=True)
    players=complete_players(library,rng,config['player_catalog_size'])
    print('Building pet catalog...',flush=True)
    pets,rejected_pets=pet_catalog(library)
    catalog_sheet(players,output/'catalogs'/'players.png','完整人物外观目录')
    catalog_sheet(pets,output/'catalogs'/'pets.png','宠物外观目录（不标注）')
    (output/'catalogs'/'assets.json').write_text(json.dumps({'players':[p.details for p in players],'pets':[p.details for p in pets],'rejected_pets':rejected_pets},ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Catalog ready: {len(players)} players, {len(pets)} pet appearances',flush=True)
    schedule=[name for name,n in config['scenarios'].items() for _ in range(n)];rng.shuffle(schedule)
    counts=[int(k) for k,n in config['monster_quotas'].items() for _ in range(n)];rng.shuffle(counts)
    pet_scenes=[i for i,s in enumerate(schedule) if 'pet' in s]
    two_pets=set(rng.sample(pet_scenes,config['two_pet_images']))
    pet_order=list(pets);rng.shuffle(pet_order);pet_index=0
    player_order=list(players);rng.shuffle(player_order);player_index=0
    selection=json.loads(Path(config['map_selection']).read_text(encoding='utf-8'))
    maps=[r['map_id'] for r in selection['maps']];rng.shuffle(maps)
    roots={};records=[];renderer=MapRenderer(sprites,W,H);map_uses=Counter()
    try:
        for i,scenario in enumerate(schedule):
            count=counts.pop() if 'monster' in scenario else 0
            player=player_order[player_index%len(player_order)] if 'player' in scenario else None
            if player:player_index+=1
            selected_pets=[]
            for _ in range((2 if i in two_pets else 1) if 'pet' in scenario else 0):
                selected_pets.append(pet_order[pet_index%len(pet_order)]);pet_index+=1
            chosen=None
            for attempt in range(config['max_attempts']):
                # Prefer unused maps, but rotate quickly when geometry cannot fit all actors.
                candidates=sorted(maps,key=lambda mid:(map_uses[mid],maps.index(mid)))
                mid=candidates[(i+attempt)%len(candidates)]
                if mid not in roots:roots[mid]=next(iter(store.read_wzjson(mid,'/Map/Map/').values()))
                chosen=make_scene(roots[mid],sprites,renderer,count,player,selected_pets,rng)
                if chosen is not None:break
            if chosen is None:raise RuntimeError(f'Cannot place sample {i+1}, {scenario}, {count} monsters')
            image,labels,metadata=chosen;stem=f'{i+1:03d}_{scenario}';map_uses[mid]+=1
            image.save(output/'images'/f'{stem}.png')
            rows=[];review=image.copy();draw=ImageDraw.Draw(review)
            for label in labels:
                x,y,w,h=label['bbox'];cid=label['class_id']
                rows.append(f'{cid} {(x+w/2)/W:.8f} {(y+h/2)/H:.8f} {w/W:.8f} {h/H:.8f}')
                color='#ff4545' if cid==0 else '#399bff'
                draw.rectangle((x,y,x+w,y+h),outline=color,width=2)
                draw.text((x,max(0,y-13)),label['class_name'],fill=color,stroke_width=1,stroke_fill='black')
            (output/'labels'/f'{stem}.txt').write_text('\n'.join(rows),encoding='utf-8')
            review.save(output/'visual_checks'/f'{stem}.jpg',quality=95,subsampling=0)
            record={'stem':stem,'scenario':scenario,'map_id':mid,'monster_count':count,'player_count':int(player is not None),'pet_count':len(selected_pets),'player':player.details if player else None,'pets':[p.details for p in selected_pets],'labels':labels,**metadata}
            records.append(record)
            with (output/'scenario_manifest.jsonl').open('a',encoding='utf-8') as file:file.write(json.dumps(record,ensure_ascii=False)+'\n')
            print(f'Generated {i+1}/100: {scenario}; map {mid}; attempts {attempt+1}',flush=True)
        report=validate_output(output,records)
        write_views(output,records)
        (output/'asset_loading_report.json').write_text(json.dumps({'client_catalog':store._catalog_version(),'pet_resource_ids':len(library.pet_names),'accepted_pet_visuals':len(pets),'excluded_pet_ids':rejected_pets,'resource_warnings':sprites.missing},ensure_ascii=False,indent=2),encoding='utf-8')
        (output/'README.txt').write_text('100 张独立验收样本，1280×224，无 train/val/test 拆分。\n0=monster，1=player；宠物无标签。\n打开 index.html 浏览原图与标注，overviews 为每页10张的分页总览，catalogs 为完整人物和宠物素材目录。\n正式4万张尚未开始；等待用户鉴定。\n所有图像来自客户端素材离线合成，并非真实游戏截图。\nvalidation_report.json 为自动检查结果，视觉标准仍以用户验收为准。\n复现命令：python -m classic_dataset.scattered_preview --config configs/scattered_preview_100.yml\n已有非空输出目录会被拒绝覆盖。\n',encoding='utf-8')
        print(json.dumps(report,ensure_ascii=False),flush=True)
        print(f'DONE: {output}',flush=True)
    finally:store.close()


if __name__=='__main__':main()
