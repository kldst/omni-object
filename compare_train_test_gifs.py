"""Build per-category train-vs-test comparison grids by sampling the middle
frame of each rendered GIF.

For each problematic category (box / tube / remote / cutlery):
  Top section  = TEST instances (test_scene1..4), thin green border
  Lower rows   = TRAIN instances (scene01..34), thin blue border
Each tile is labeled with scene/model_name. Red bbox in tile = GT pose,
green bbox = Predicted pose (using GT size; isolates R+t error).

Outputs one PNG per category at
  gif_outputs/0521_14000_compare/<category>.png
"""
from __future__ import annotations
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_DIR = PROJECT_ROOT / "gif_outputs/0521_14000_train_problematic/housecat6d"
TEST_DIR  = PROJECT_ROOT / "gif_outputs/0521_14000_problematic/housecat6d"
OUT_DIR   = PROJECT_ROOT / "gif_outputs/0521_14000_compare"

CATEGORIES = ["box", "tube", "remote", "cutlery"]
TILE_W, TILE_H = 420, 340
COLS = 5
BORDER = 4
HEADER = 22

def get_middle_frame(gif_path: Path) -> np.ndarray:
    reader = imageio.get_reader(str(gif_path))
    n = reader.get_length()
    if n in (0, float("inf")):
        # fallback: read all then pick middle
        frames = list(reader)
        return np.asarray(frames[len(frames) // 2])
    return np.asarray(reader.get_data(n // 2))

def make_tile(frame: np.ndarray, scene: str, model: str, border_rgb) -> Image.Image:
    img = Image.fromarray(frame).convert("RGB").resize((TILE_W, TILE_H - HEADER), Image.LANCZOS)
    full = Image.new("RGB", (TILE_W, TILE_H), color=border_rgb)
    full.paste(img, (0, HEADER))
    d = ImageDraw.Draw(full)
    d.rectangle((0, HEADER, TILE_W - 1, TILE_H - 1), outline=border_rgb, width=BORDER)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except IOError:
        font = ImageFont.load_default()
    d.text((6, 3), f"{scene} | {model}", fill=(255, 255, 255), font=font)
    return full

def collect(base_dir: Path, category: str):
    """Return sorted [(scene, model, gif_path)] for one category."""
    out = []
    if not base_dir.is_dir():
        return out
    for scene_dir in sorted(base_dir.iterdir()):
        if not scene_dir.is_dir():
            continue
        for gif in sorted(scene_dir.glob(f"{category}-*.gif")):
            out.append((scene_dir.name, gif.stem, gif))
    return out

def build_grid(category: str) -> Path:
    test_items  = collect(TEST_DIR, category)
    train_items = collect(TRAIN_DIR, category)
    n_test = len(test_items)
    n_train = len(train_items)
    if n_test == 0 and n_train == 0:
        print(f"[skip] {category}: no GIFs found")
        return None

    rows_test  = (n_test  + COLS - 1) // COLS
    rows_train = (n_train + COLS - 1) // COLS
    rows_total = rows_test + rows_train + 2     # +2 for two label bars
    grid_w = COLS * TILE_W
    grid_h = rows_total * TILE_H + 40

    grid = Image.new("RGB", (grid_w, grid_h), color=(20, 20, 20))
    d_grid = ImageDraw.Draw(grid)
    try:
        font_big = ImageFont.truetype("DejaVuSans-Bold.ttf", 22)
    except IOError:
        font_big = ImageFont.load_default()

    y = 6
    d_grid.text((10, y), f"[ {category.upper()} ]  TEST (red=GT, green=Pred)  —  {n_test} instances",
                fill=(120, 230, 130), font=font_big)
    y += 34

    # Test rows (green border)
    for idx, (scene, model, gif) in enumerate(test_items):
        try:
            frame = get_middle_frame(gif)
        except Exception as e:
            print(f"[err] {gif}: {e}"); continue
        tile = make_tile(frame, scene, model, (60, 200, 80))
        r, c = divmod(idx, COLS)
        grid.paste(tile, (c * TILE_W, y + r * TILE_H))
    y += rows_test * TILE_H + 6

    d_grid.text((10, y), f"[ {category.upper()} ]  TRAIN (red=GT, green=Pred)  —  {n_train} instances",
                fill=(120, 170, 230), font=font_big)
    y += 34

    # Train rows (blue border)
    for idx, (scene, model, gif) in enumerate(train_items):
        try:
            frame = get_middle_frame(gif)
        except Exception as e:
            print(f"[err] {gif}: {e}"); continue
        tile = make_tile(frame, scene, model, (60, 140, 220))
        r, c = divmod(idx, COLS)
        grid.paste(tile, (c * TILE_W, y + r * TILE_H))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"{category}.png"
    grid.save(out_path, quality=92)
    print(f"[ok] {category}: {n_test} test + {n_train} train tiles -> {out_path}  ({grid_w}x{grid_h})")
    return out_path

if __name__ == "__main__":
    for cat in CATEGORIES:
        build_grid(cat)
