from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .src.utils import ensure_layout, load_config, read_json


def _read_csv(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write(path, rows):
    if not rows: path.write_text("", encoding="utf-8"); return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config); figures = root / "figures"
    design = [{"stage": 1, "name": "prefix pool"}, {"stage": 2, "name": "common M/V candidates"}, {"stage": 3, "name": "forced one-token forward"}, {"stage": 4, "name": "split-local double centering"}, {"stage": 5, "name": "Local/Global/Matched/Content EV"}, {"stage": 6, "name": "rank-0 functional recovery"}]
    _write(figures / "figure1_design.csv", design)
    geometry = _read_csv(root / "metrics/geometry_rows.csv")
    _write(figures / "figure2_layer_rank_ev.csv", [{key: row[key] for key in ("layer", "fold", "rank", "ev_local", "ev_global", "ev_matched", "ev_content")} for row in geometry])
    _write(figures / "figure3_delta_distributions.csv", [{key: row[key] for key in ("problem_id", "prefix_id", "layer", "fold", "delta_global", "delta_matched", "delta_content")} for row in geometry])
    functional = _read_csv(root / "functional/distribution_rows.csv")
    _write(figures / "figure4_output_distances.csv", [{key: row[key] for key in ("problem_id", "prefix_id", "condition", "layer", "fold", "js", "kl")} for row in functional])
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return
    if geometry:
        labels = ["global", "matched", "content"]
        data = [[float(row[f"delta_{label}"]) for row in geometry] for label in labels]
        fig, ax = plt.subplots(figsize=(6, 4)); ax.boxplot(data, labels=labels); ax.axhline(0, color="black", linewidth=.8); ax.set_ylabel("Local EV minus control EV"); fig.tight_layout(); fig.savefig(figures / "figure3_deltas.png", dpi=180); plt.close(fig)
    if functional:
        conditions = ["Oracle", "Rank-0", "Local", "Matched", "Global"]
        data = [[float(row["js"]) for row in functional if row["condition"] == condition] for condition in conditions]
        if all(data):
            fig, ax = plt.subplots(figsize=(7, 4)); ax.boxplot(data, labels=conditions); ax.set_ylabel("JS distance from original output"); fig.tight_layout(); fig.savefig(figures / "figure4_functional.png", dpi=180); plt.close(fig)


if __name__ == "__main__": main()
