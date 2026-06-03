"""Render cutoop SOPE metrics (class-mean) as a paper-style results-table PNG.

Reads a cutoop_metrics.json (dumped by eval_omni6dpose_cutoop.py) and draws a table
with 3D IoU acc, pose acc (deg-cm), and VUS, plus mean rot/trans error.

Usage:
  python render_metrics_table.py outputs/eval_omni6dpose_cutoop/scene_0096/cutoop_metrics.json \
      --label "0603 · SOPE/test 0096" --out /tmp/metrics_table.png
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("metrics_json", type=Path)
    ap.add_argument("--label", default="Ours")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cm = json.loads(Path(args.metrics_json).read_text())["class_means"]
    iou_acc = [float(x) * 100 for x in cm["iou_acc"]]
    pose_acc = [float(x) * 100 for x in cm["pose_acc"]]
    vus = [float(a["auc"]) * 100 for a in cm["pose_auc"]]
    deg = float(np.mean(cm["deg_mean"]))
    sht = float(np.mean(cm["sht_mean"]))
    miou = float(np.mean(cm["iou_mean"])) * 100

    # ---- grouped table ----
    groups = [("3D IoU ↑", IOU_TH, iou_acc),
              ("Pose Acc ↑", POSE_TH, pose_acc),
              ("VUS ↑", POSE_TH, vus)]
    sub_headers, values, group_spans = [], [], []
    for gname, subs, vals in groups:
        group_spans.append((gname, len(sub_headers), len(sub_headers) + len(subs)))
        sub_headers += subs
        values += [f"{v:.1f}" for v in vals]
    sub_headers += ["mIoU", "Rₑ(°)", "Tₑ(cm)"]
    values += [f"{miou:.1f}", f"{deg:.2f}", f"{sht:.2f}"]

    ncol = len(sub_headers)
    fig_w = max(8.0, ncol * 0.95)
    fig, ax = plt.subplots(figsize=(fig_w, 2.3))
    ax.axis("off")

    cell_w = 1.0 / ncol
    y_grp, y_sub, y_val = 0.60, 0.38, 0.16
    h = 0.22
    gcolors = {"3D IoU ↑": "#d9e8fb", "Pose Acc ↑": "#dbf0db", "VUS ↑": "#fbe7cf"}

    # group header band
    for gname, c0, c1 in group_spans:
        x0 = c0 * cell_w
        ax.add_patch(plt.Rectangle((x0, y_grp), (c1 - c0) * cell_w, h,
                                   facecolor=gcolors.get(gname, "#eeeeee"), edgecolor="white"))
        ax.text(x0 + (c1 - c0) * cell_w / 2, y_grp + h / 2, gname,
                ha="center", va="center", fontsize=12, fontweight="bold")
    # trailing summary group
    x0 = group_spans[-1][2] * cell_w
    ax.add_patch(plt.Rectangle((x0, y_grp), (ncol - group_spans[-1][2]) * cell_w, h,
                               facecolor="#ececec", edgecolor="white"))
    ax.text(x0 + (ncol - group_spans[-1][2]) * cell_w / 2, y_grp + h / 2, "Error",
            ha="center", va="center", fontsize=12, fontweight="bold")

    # sub-header + value cells
    for j, (sh, vv) in enumerate(zip(sub_headers, values)):
        x = j * cell_w
        ax.add_patch(plt.Rectangle((x, y_sub), cell_w, h, facecolor="#f6f6f6", edgecolor="#cccccc"))
        ax.text(x + cell_w / 2, y_sub + h / 2, sh, ha="center", va="center", fontsize=10.5)
        ax.add_patch(plt.Rectangle((x, y_val), cell_w, h, facecolor="white", edgecolor="#cccccc"))
        ax.text(x + cell_w / 2, y_val + h / 2, vv, ha="center", va="center", fontsize=11.5, fontweight="bold")

    ax.text(0.0, 0.92, args.label, ha="left", va="center", fontsize=12, fontweight="bold")
    ax.text(0.0, 0.03, "values in %, except Rₑ(deg) / Tₑ(cm).  class-mean over SOPE/test.",
            ha="left", va="center", fontsize=8.5, color="#555555")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    out = args.out or args.metrics_json.with_suffix(".table.png")
    fig.savefig(out, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
