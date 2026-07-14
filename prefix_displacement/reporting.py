from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


def _json_cell(value: Any) -> str:
    return json.dumps(value, sort_keys=True).replace("|", "\\|")


def write_cache_report(summary: Mapping[str, Any], path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    source = summary.get("source", {})
    source_kind = source.get("kind", "UNKNOWN") if isinstance(source, Mapping) else "UNKNOWN"
    model = summary.get("model", {})
    margin = summary.get("margin", {})

    lines = [
        "# Transition cache report",
        "",
        f"- Source kind: `{source_kind}`",
        f"- Manifest: `{summary.get('manifest_path', 'UNKNOWN')}`",
        f"- Records: {summary.get('num_records', 'UNKNOWN')}",
        f"- Problems: {summary.get('num_problems', 'UNKNOWN')}",
        f"- Trajectories: {summary.get('num_trajectories', 'UNKNOWN')}",
        f"- Shards: {summary.get('num_shards', 'UNKNOWN')}",
        "",
    ]
    if source_kind != "gsm8k":
        lines.extend(
            [
                "> **Not a production GSM8K cache.** This report describes a synthetic",
                "> acceptance artifact only. Real model/checkpoint/hook/margin values remain",
                "> blocked by Task 0 and must not be inferred from this report.",
                "",
            ]
        )

    lines.extend(["## Scientific metadata", ""])
    lines.extend(["| Section | Value |", "|---|---|"])
    lines.append(f"| model | {_json_cell(model)} |")
    lines.append(f"| margin | {_json_cell(margin)} |")
    lines.append(f"| source | {_json_cell(source)} |")
    lines.extend(["", "## Counts", "", "### Records by split", ""])
    lines.extend(["| Split | Records |", "|---|---:|"])
    for split, count in summary.get("records_by_split", {}).items():
        lines.append(f"| {split} | {count} |")
    lines.extend(["", "### Boundary classes", ""])
    lines.extend(["| Boundary class | Records |", "|---|---:|"])
    for boundary, count in summary.get("boundary_counts", {}).items():
        lines.append(f"| {boundary} | {count} |")

    lines.extend(["", "## Tensor schema", ""])
    lines.extend(["| Field | Shape suffix | Dtype(s) | Semantics |", "|---|---|---|---|"])
    semantics = {
        "h_departure": r"`[N, d]` departure/origin/basepoint representation $h_t^\ell$",
        "h_arrival": r"`[N, d]` arrival representation $h_{t+1}^\ell$",
        "delta": r"`[N, d]` prefix-extension displacement $h_{t+1}^\ell-h_t^\ell$",
        "g_arrival": r"`[N, d]` arrival-point gradient $\nabla_{h_{t+1}}m$",
        "baseline_margin": "`[N]` unmodified answer margin",
    }
    dtypes = summary.get("tensor_dtypes", {})
    shapes = summary.get("tensor_shape_suffixes", {})
    for field in semantics:
        lines.append(
            f"| {field} | {_json_cell(shapes.get(field, []))} | "
            f"{_json_cell(dtypes.get(field, []))} | {semantics[field]} |"
        )

    lines.extend(["", "## Metadata completeness", ""])
    lines.extend(
        [
            "| Field | Observed JSON type(s) | Missing | Missing rate |",
            "|---|---|---:|---:|",
        ]
    )
    missing_counts = summary.get("missing_counts", {})
    missing_rates = summary.get("missing_rates", {})
    metadata_types = summary.get("metadata_types", {})
    for field in sorted(missing_counts):
        lines.append(
            f"| {field} | {_json_cell(metadata_types.get(field, []))} | "
            f"{missing_counts[field]} | {100.0 * missing_rates[field]:.4f}% |"
        )

    lines.extend(
        [
            "",
            "## Integrity checks",
            "",
            f"- Split leakage: **{summary.get('split_leakage_check', 'UNKNOWN')}**",
            f"- `delta == h_arrival - h_departure`: **{summary.get('delta_identity_check', 'UNKNOWN')}**",
            f"- Tensor finite check: **{summary.get('finite_tensor_check', 'UNKNOWN')}**",
            f"- Shard checksum check: **{summary.get('checksum_check', 'UNKNOWN')}**",
            f"- Zero-norm delta rows: {summary.get('zero_norm_delta_count', 'UNKNOWN')}",
            "",
            "## Data-quality interpretation",
            "",
            "Problem ID is the split grain. Every cached row is checked against the immutable",
            "registry, trajectory IDs may not move between problems, and duplicate composite",
            "transition IDs fail immediately. Tensor shards are loaded one at a time on CPU for",
            "validation, so validation does not accumulate activations or gradients on GPU 0.",
            "",
        ]
    )
    with output.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))
    return output
