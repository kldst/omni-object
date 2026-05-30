"""DIAGNOSTIC B: Reference-view discriminative-power audit.

For each housecat6d test instance in the problematic categories (box, tube,
remote, cutlery) AND a control set of good categories (bottle, glass, can, cup,
teapot, shoe), grab the 4 reference views that the model actually consumes
(views 1, 5, 10, 15) and lay them out as a strip:
    [view 1 | view 5 | view 10 | view 15]
labeled with the instance name.

Then stack many instances vertically per category to produce a per-category
PNG. Visually inspecting tells us whether the 4 fixed reference views give the
model enough angular coverage / discriminative texture for each category.

Output: outputs/diag_B_ref_views/{<category>.png, ...}

Also produces a single side-by-side "problematic vs good" image for each
(bad_cat, good_cat) pair to make the contrast obvious.
"""
from __future__ import annotations
import multiprocessing as mp
from pathlib import Path
from typing import List, Tuple
import numpy as np
from PIL import Image, ImageDraw, ImageFont

FREEPOSE = Path('/mnt/train-data-4-hdd/yian/freepose')
REF_ROOT = FREEPOSE / 'housecat6d/housecat6d_aligned_object_refs'
OUT_DIR = FREEPOSE / 'omni-object_clone/outputs/diag_B_ref_views'
OUT_DIR.mkdir(parents=True, exist_ok=True)

VIEWS = (1, 5, 10, 15)
TILE_W, TILE_H = 220, 220
LABEL_H = 28
PROBLEMATIC_CATS = ('box', 'tube', 'remote', 'cutlery')
GOOD_CATS = ('bottle', 'glass', 'can', 'cup', 'teapot', 'shoe')

# We focus on TEST instances (because they are the ones failing). Test instance
# names come from housecat6d_aligned_object_refs_test/ if available, else fall
# back to scanning test_scene*/labels for unique model_list entries.
import json

def collect_test_instances(category: str):
    """Use test_scene*/labels to enumerate test-time instances of given category."""
    hc_root = FREEPOSE / 'housecat6d'
    import pickle
    seen = set()
    for d in sorted(hc_root.glob('test_scene*')):
        for lp in sorted((d / 'labels').glob('*_label.pkl')):
            try:
                with open(lp, 'rb') as h: lab = pickle.load(h)
            except Exception: continue
            for n in lab.get('model_list', []):
                n = str(n)
                if n.startswith(f'{category}-'):
                    seen.add(n)
            break  # one label per scene is enough to enumerate
    return sorted(seen)


def collect_train_instances(category: str, limit: int = 6):
    """Sample train instances of the category from the ref root."""
    cands = sorted(p.name for p in REF_ROOT.iterdir()
                   if p.is_dir() and p.name.startswith(f'{category}-'))
    test_set = set(collect_test_instances(category))
    train_only = [n for n in cands if n not in test_set]
    return train_only[:limit]


def make_strip(instance: str) -> Image.Image | None:
    """Horizontal strip of view 1, 5, 10, 15 + label header."""
    rgb_dir = REF_ROOT / instance / 'rgb'
    if not rgb_dir.is_dir():
        return None
    tiles = []
    for v in VIEWS:
        img_path = rgb_dir / f'{int(v):06d}.png'
        if not img_path.is_file():
            return None
        im = Image.open(img_path).convert('RGB').resize((TILE_W, TILE_H), Image.LANCZOS)
        tiles.append(im)
    strip_w = TILE_W * len(VIEWS) + 2 * (len(VIEWS) - 1)
    full = Image.new('RGB', (strip_w, TILE_H + LABEL_H), color=(30, 30, 30))
    x = 0
    for i, t in enumerate(tiles):
        full.paste(t, (x, LABEL_H))
        x += TILE_W + (2 if i < len(tiles)-1 else 0)
    d = ImageDraw.Draw(full)
    try:
        font = ImageFont.truetype('DejaVuSans-Bold.ttf', 16)
    except IOError:
        font = ImageFont.load_default()
    d.text((6, 5), f'{instance}    (views {VIEWS[0]}, {VIEWS[1]}, {VIEWS[2]}, {VIEWS[3]})',
           fill=(255, 220, 60), font=font)
    return full


def build_category_grid(category: str):
    test_insts = collect_test_instances(category)
    train_insts = collect_train_instances(category, limit=8)
    sections = [
        ('TEST', test_insts, (60, 200, 80)),
        ('TRAIN', train_insts, (60, 140, 220)),
    ]
    strips = []
    headers = []
    for label, insts, color in sections:
        for inst in insts:
            s = make_strip(inst)
            if s is None: continue
            strips.append((label, color, s))
        headers.append((label, len(insts)))
    if not strips:
        print(f'[skip] {category}: no strips')
        return
    strip_w = strips[0][2].size[0]
    total_h = sum(s[2].size[1] + 4 for s in strips) + 60
    final = Image.new('RGB', (strip_w, total_h), color=(15, 15, 15))
    d = ImageDraw.Draw(final)
    try:
        font_big = ImageFont.truetype('DejaVuSans-Bold.ttf', 24)
    except IOError:
        font_big = ImageFont.load_default()
    d.text((10, 6), f'[ {category.upper()} ]  reference views fed to model — TEST instances (green) | TRAIN instances (blue)',
           fill=(255, 255, 255), font=font_big)
    y = 50
    last_label = None
    for label, color, strip in strips:
        if label != last_label:
            d.rectangle((0, y - 2, strip_w, y + strip.size[1] + 2), outline=color, width=2)
            last_label = label
        final.paste(strip, (0, y))
        y += strip.size[1] + 4
    out_path = OUT_DIR / f'{category}.png'
    final.save(out_path, quality=92)
    print(f'[ok] {category}: {len(strips)} strips -> {out_path}  ({final.size[0]}x{final.size[1]})')


def main():
    cats = list(PROBLEMATIC_CATS) + list(GOOD_CATS)
    with mp.Pool(min(8, len(cats))) as pool:
        pool.map(build_category_grid, cats)


if __name__ == '__main__':
    main()
