"""Compare cutoop SOPE metrics across models -> text + side-by-side PNG/JSON.

Usage:
  python compare_metrics.py \
      "FT_5000=outputs/eval_omni6dpose_0604/FT_5000/cutoop_metrics.json" \
      "6000=outputs/eval_omni6dpose_cutoop/sope_test_00_05/cutoop_metrics.json" \
      --out outputs/eval_omni6dpose_0604/comparison.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IOU_TH = ["0.25", "0.50", "0.75"]
POSE_TH = ["5°2cm", "5°5cm", "10°2cm", "10°5cm"]


def load(path):
    cm = json.loads(Path(path).read_text())["class_means"]
    return {
        "iou": [float(x) * 100 for x in cm["iou_acc"]],
        "pose": [float(x) * 100 for x in cm["pose_acc"]],
        "vus": [float(a["auc"]) * 100 for a in cm["pose_auc"]],
        "miou": float(np.mean(cm["iou_mean"])) * 100,
        "deg": float(np.mean(cm["deg_mean"])),
        "sht": float(np.mean(cm["sht_mean"])),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="label=path/to/cutoop_metrics.json")
    ap.add_argument("--out", type=Path, default=Path("comparison.png"))
    args = ap.parse_args()

    items = []
    for spec in args.models:
        label, path = spec.split("=", 1)
        items.append((label, load(path)))

    sub = IOU_TH + POSE_TH + POSE_TH + ["mIoU", "Rₑ°", "Tₑcm"]
    groups = [("3D IoU ↑", 0, 3, "#d9e8fb"), ("Pose Acc ↑", 3, 7, "#dbf0db"),
              ("VUS ↑", 7, 11, "#fbe7cf"), ("Error", 11, 14, "#ececec")]

    def row_vals(m):
        return ([f"{v:.1f}" for v in m["iou"]] + [f"{v:.1f}" for v in m["pose"]] +
                [f"{v:.1f}" for v in m["vus"]] + [f"{m['miou']:.1f}", f"{m['deg']:.2f}", f"{m['sht']:.2f}"])

    # ---- text ----
    print(f"\n{'model':14s} | " + " ".join(f"{s:>7s}" for s in sub))
    for label, m in items:
        print(f"{label:14s} | " + " ".join(f"{v:>7s}" for v in row_vals(m)))
    # delta vs last (baseline)
    if len(items) == 2:
        a, b = items[0][1], items[1][1]
        d = [f"{(x-y):+.1f}" for x, y in zip(
            a["iou"]+a["pose"]+a["vus"]+[a["miou"]], b["iou"]+b["pose"]+b["vus"]+[b["miou"]])]
        d += [f"{a['deg']-b['deg']:+.2f}", f"{a['sht']-b['sht']:+.2f}"]
        print(f"{'Δ (1st-2nd)':14s} | " + " ".join(f"{v:>7s}" for v in d))

    # ---- PNG ----
    ncol = 1 + len(sub)
    nrow = len(items)
    fig, ax = plt.subplots(figsize=(max(9, ncol * 0.92), 1.4 + 0.5 * nrow))
    ax.axis("off")
    cw = 1.0 / ncol
    h = 0.7 / (nrow + 2)
    y_grp = 1.0 - h
    y_sub = y_grp - h
    for gname, c0, c1, col in groups:
        x0 = (1 + c0) * cw
        ax.add_patch(plt.Rectangle((x0, y_grp), (c1 - c0) * cw, h, facecolor=col, edgecolor="white"))
        ax.text(x0 + (c1 - c0) * cw / 2, y_grp + h / 2, gname, ha="center", va="center", fontsize=11, fontweight="bold")
    ax.text(cw / 2, y_sub + h / 2, "model", ha="center", va="center", fontsize=9.5, fontweight="bold")
    for j, s in enumerate(sub):
        ax.add_patch(plt.Rectangle(((1 + j) * cw, y_sub), cw, h, facecolor="#f6f6f6", edgecolor="#ccc"))
        ax.text((1 + j) * cw + cw / 2, y_sub + h / 2, s, ha="center", va="center", fontsize=9)
    for r, (label, m) in enumerate(items):
        y = y_sub - (r + 1) * h
        ax.add_patch(plt.Rectangle((0, y), cw, h, facecolor="#fff", edgecolor="#ccc"))
        ax.text(cw / 2, y + h / 2, label, ha="center", va="center", fontsize=9.5, fontweight="bold")
        for j, v in enumerate(row_vals(m)):
            ax.add_patch(plt.Rectangle(((1 + j) * cw, y), cw, h, facecolor="#fff", edgecolor="#eee"))
            ax.text((1 + j) * cw + cw / 2, y + h / 2, v, ha="center", va="center", fontsize=9.5)
    ax.text(0.0, 1.0, "SOPE/test 00-05 · class-mean · values in % (Rₑ deg / Tₑ cm)",
            ha="left", va="bottom", fontsize=9, color="#444")
    ax.set_xlim(0, 1); ax.set_ylim(y_sub - (nrow + 0.5) * h, 1.04)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=200, bbox_inches="tight", facecolor="white")
    json.dump({label: m for label, m in items}, open(args.out.with_suffix(".json"), "w"), indent=2)
    print(f"\nsaved -> {args.out}  and  {args.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
