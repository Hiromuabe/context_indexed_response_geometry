from __future__ import annotations

import argparse
import json

from .src.utils import atomic_json, ensure_layout, load_config, read_json


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config)
    geometry = read_json(root / "metrics/geometry_summary.json")
    functional_path = root / "functional/summary.json"; functional = read_json(functional_path) if functional_path.exists() else {"status": "not_run_or_gated"}
    atomic_json(root / "tables/main_results.json", {"geometry": geometry, "functional": functional})
    lines = ["# Main results", "", "| Metric | Mean | 95% CI |", "|---|---:|---:|"]
    for key in ("delta_global", "delta_matched", "delta_matched_good_matches", "delta_content"):
        row = geometry[key]; lines.append(f"| {key} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    (root / "tables/main_results.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


if __name__ == "__main__": main()
