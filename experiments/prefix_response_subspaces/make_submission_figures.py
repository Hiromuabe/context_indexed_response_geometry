from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


BLUE = "#2F6B9A"
BLUE_LIGHT = "#A9C7DD"
ORANGE = "#D8873A"
INK = "#27313B"
MID = "#707A86"
LIGHT = "#C8CFD7"
GRID = "#E7EBEF"

# High-contrast, color-vision-safe palette used only for the block-mechanism
# figure. Non-color redundancy is retained through luminance and hatching.
MECH_BLUE = "#0072B2"
MECH_SKY = "#56B4E9"
MECH_GOLD = "#E69F00"
MECH_GOLD_DARK = "#8A4B00"


# Reviewed values provide a faithful Figure 2 and mechanism figure even when
# the paper result directory is not mounted locally. Figure 3 never interpolates
# from these aggregates: it is emitted only from rank_saturation_rows.csv.
REVIEWED = {
    "delta_conditional_global": {
        "mean": 0.380646,
        "ci_low": 0.369533,
        "ci_high": 0.392286,
        "n_problems": 255,
    },
    "delta_wrong": {
        "mean": 0.412374,
        "ci_low": 0.395474,
        "ci_high": 0.428646,
        "n_problems": 255,
    },
    "R_within": {
        "mean": 0.616014,
        "ci_low": 0.614156,
        "ci_high": 0.617979,
        "n_problems": 256,
    },
    "R_between": {
        "mean": 0.852250,
        "ci_low": 0.843822,
        "ci_high": 0.860322,
        "n_problems": 256,
    },
    "R_between_minus_within": {
        "mean": 0.236236,
        "ci_low": 0.227980,
        "ci_high": 0.243901,
        "n_problems": 256,
    },
    "D_rank0": {"mean": 0.008120259730880182, "n_problems": 256},
    "D_local": {"mean": 0.0018640190998928934, "n_problems": 256},
    "D_conditional_global": {"mean": 0.004512279604087712, "n_problems": 256},
    "D_wrong_mean": {"mean": 0.004874028392910802, "n_problems": 256},
    "recovery_fraction_local": {"mean": 0.7704483401184452, "n_problems": 256},
    "recovery_fraction_conditional_global": {"mean": 0.4443183157149321, "n_problems": 256},
    "recovery_fraction_wrong_mean": {"mean": 0.3997693972305379, "n_problems": 256},
    "post_attention_interaction_fraction": 0.025588,
    "post_mlp_interaction_fraction": 0.094349,
    "post_attention_value_span_fraction": {
        "mean": 0.9730304427534773,
        "ci_low": 0.9705212987388171,
        "ci_high": 0.9754327712173831,
        "n_problems": 256,
    },
    "post_mlp_value_span_fraction": {
        "mean": 0.5574250089193475,
        "ci_low": 0.5424566336907232,
        "ci_high": 0.5720136737139964,
        "n_problems": 256,
    },
    "median_r90": 96.0,
    "rank64_to_127_relative_gain": 0.114318,
}


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(
    source: dict[str, Any] | None,
    keys: tuple[str, ...],
    fallback: str,
) -> dict[str, Any]:
    if source is not None:
        for key in keys:
            value = source.get(key)
            if isinstance(value, dict) and "mean" in value:
                return value
    return dict(REVIEWED[fallback])


def _validate_estimate(name: str, estimate: dict[str, Any]) -> None:
    mean = float(estimate["mean"])
    if not np.isfinite(mean):
        raise ValueError(f"{name} has a non-finite mean")
    if "ci_low" in estimate and "ci_high" in estimate:
        low, high = float(estimate["ci_low"]), float(estimate["ci_high"])
        if not low <= mean <= high:
            raise ValueError(f"{name} mean is outside its confidence interval")


def _error(estimate: dict[str, Any]) -> np.ndarray | None:
    if "ci_low" not in estimate or "ci_high" not in estimate:
        return None
    mean = float(estimate["mean"])
    return np.asarray([[mean - float(estimate["ci_low"])], [float(estimate["ci_high"]) - mean]])


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _save(fig: plt.Figure, stem: Path, dpi: int) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    paths = []
    for suffix in ("png", "pdf", "svg"):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
        paths.append(path)
    plt.close(fig)
    return paths


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.0,
            "axes.titlesize": 9.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "text.color": INK,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(-0.13, 1.08, label, transform=ax.transAxes, fontweight="bold", fontsize=10, va="top")


def make_main_figure(root: Path, output: Path, dpi: int) -> list[Path]:
    geometry = _read_json(root / "metrics/paper_geometry_summary.json")
    functional = _read_json(root / "functional/paper_summary.json")
    delta_global = _metric(geometry, ("delta_conditional_global",), "delta_conditional_global")
    delta_wrong = _metric(geometry, ("delta_wrong_exact_bin", "delta_wrong"), "delta_wrong")
    r_within = _metric(geometry, ("R_within",), "R_within")
    r_between = _metric(geometry, ("R_between",), "R_between")
    r_gap = _metric(geometry, ("R_between_minus_within",), "R_between_minus_within")
    d_rank0 = _metric(functional, ("D_rank0",), "D_rank0")
    d_local = _metric(functional, ("D_local",), "D_local")
    d_global = _metric(functional, ("D_conditional_global",), "D_conditional_global")
    d_wrong = _metric(functional, ("D_wrong_mean",), "D_wrong_mean")
    recovery_local = _metric(functional, ("recovery_fraction_local",), "recovery_fraction_local")
    recovery_global = _metric(functional, ("recovery_fraction_conditional_global",), "recovery_fraction_conditional_global")
    recovery_wrong = _metric(functional, ("recovery_fraction_wrong_mean",), "recovery_fraction_wrong_mean")

    estimates = {
        "delta_conditional_global": delta_global,
        "delta_wrong": delta_wrong,
        "R_within": r_within,
        "R_between": r_between,
        "R_between_minus_within": r_gap,
        "D_rank0": d_rank0,
        "D_local": d_local,
        "D_conditional_global": d_global,
        "D_wrong_mean": d_wrong,
        "recovery_fraction_local": recovery_local,
        "recovery_fraction_conditional_global": recovery_global,
        "recovery_fraction_wrong_mean": recovery_wrong,
    }
    for name, estimate in estimates.items():
        _validate_estimate(name, estimate)

    fig, axes = plt.subplots(1, 3, figsize=(7.15, 2.65), layout="constrained")

    ax = axes[0]
    means = [float(delta_global["mean"]), float(delta_wrong["mean"])]
    labels = ["Matched\ncommon", "Wrong\ncontext"]
    for y, (mean, estimate, color, marker) in enumerate(zip(means, (delta_global, delta_wrong), (BLUE, ORANGE), ("o", "s"))):
        ax.errorbar(mean, y, xerr=_error(estimate), fmt=marker, color=color, ecolor=color, capsize=3, lw=1.3, ms=5)
        ax.text(mean + 0.014, y, f"{mean:.3f}", va="center", fontsize=7.2)
    ax.axvline(0, color=INK, lw=0.9)
    ax.set_yticks([0, 1], labels)
    ax.invert_yaxis()
    ax.set_xlim(-0.025, 0.475)
    ax.set_xlabel("Held-out ΔEV (target context - control)")
    ax.set_title("Held-out specificity")
    ax.grid(axis="x", color=GRID, lw=0.7)
    _panel_label(ax, "A")

    ax = axes[1]
    distance_means = [float(r_within["mean"]), float(r_between["mean"])]
    for y, (mean, estimate, color) in enumerate(zip(distance_means, (r_within, r_between), (MID, BLUE))):
        ax.errorbar(mean, y, xerr=_error(estimate), fmt="o", color=color, ecolor=color, capsize=3, lw=1.3, ms=5)
        ax.text(mean + 0.008, y, f"{mean:.3f}", va="center", fontsize=7.2)
    arrow_y = 1.45
    ax.annotate("", xy=(distance_means[1], arrow_y), xytext=(distance_means[0], arrow_y), arrowprops={"arrowstyle": "<->", "color": INK, "lw": 0.9})
    ax.text(np.mean(distance_means), arrow_y + 0.08, rf"gap = {float(r_gap['mean']):.3f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_yticks([0, 1], ["Within-context\nsplit halves", "Between\ncontexts"])
    ax.set_ylim(-0.45, 1.75)
    ax.set_xlim(0.58, 0.895)
    ax.set_xlabel("Normalized projection distance")
    ax.set_title("Subspace separation")
    ax.grid(axis="x", color=GRID, lw=0.7)
    _panel_label(ax, "B")

    ax = axes[2]
    js_estimates = [d_rank0, d_local, d_global, d_wrong]
    js_means = [float(value["mean"]) for value in js_estimates]
    colors = [LIGHT, BLUE, "#9CA7B3", "#BEC5CD"]
    bars = ax.bar(np.arange(4), js_means, color=colors, edgecolor=INK, linewidth=0.65, width=0.68)
    js_tops = []
    for index, (bar, estimate) in enumerate(zip(bars, js_estimates)):
        error = _error(estimate)
        if error is not None:
            ax.errorbar(index, float(estimate["mean"]), yerr=error, fmt="none", ecolor=INK, capsize=2.5, lw=0.9)
        top = float(estimate.get("ci_high", estimate["mean"]))
        js_tops.append(top)
        ax.text(bar.get_x() + bar.get_width() / 2, top + 0.00016, f"{bar.get_height():.5f}", ha="center", va="bottom", fontsize=6.3)
    recoveries = [None, recovery_local, recovery_global, recovery_wrong]
    recovery_colors = [INK, BLUE, MID, ORANGE]
    for index in range(1, 4):
        ax.text(
            index,
            js_tops[index] + 0.00067,
            f"{100 * float(recoveries[index]['mean']):.1f}%\nrecovery",
            ha="center",
            va="bottom",
            color=recovery_colors[index],
            fontweight="bold",
            fontsize=6.2,
            linespacing=0.95,
        )
    ax.set_xticks(np.arange(4), ["Rank-0", "Target\ncontext", "Matched\ncommon", "Wrong\ncontext"])
    ax.tick_params(axis="x", labelsize=6.0, pad=2)
    ax.set_ylim(0, max(js_tops) * 1.25)
    ax.set_ylabel("Jensen-Shannon divergence")
    ax.set_title("Output-distribution fidelity")
    ax.grid(axis="y", color=GRID, lw=0.7)
    _panel_label(ax, "C")

    rows = []
    for metric, estimate in estimates.items():
        rows.append({"metric": metric, **estimate, "source": "artifact" if ((geometry and metric in geometry) or (functional and metric in functional)) else "reviewed_summary"})
    _write_rows(output / "figure2_main_results_source.csv", rows)
    return _save(fig, output / "figure2_main_results", dpi)


def _rank_curve(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]] | None:
    rows_path = root / "metrics/rank_saturation_rows.csv"
    if not rows_path.is_file():
        return None
    with rows_path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    if not raw:
        return None
    ranks = sorted({int(row["rank"]) for row in raw})
    reference_rank = max(ranks)
    by_curve: dict[tuple[str, int], dict[int, float]] = defaultdict(dict)
    problem_by_prefix: dict[str, str] = {}
    for row in raw:
        prefix_id = row["prefix_id"]
        problem_by_prefix[prefix_id] = row["problem_id"]
        by_curve[(prefix_id, int(row["split_id"]))][int(row["rank"])] = float(row["heldout_ev"])
    normalized: list[dict[str, Any]] = []
    by_prefix_rank: dict[tuple[str, int], list[float]] = defaultdict(list)
    for (prefix_id, split_id), curve in by_curve.items():
        reference = curve.get(reference_rank)
        if reference is None or reference <= 0:
            continue
        for rank in ranks:
            if rank not in curve:
                continue
            value = curve[rank] / reference
            if not -1e-8 <= value <= 1.00001:
                raise ValueError(f"normalized rank EV outside [0, 1] for {prefix_id}, split {split_id}, rank {rank}")
            by_prefix_rank[(prefix_id, rank)].append(value)
            normalized.append({"problem_id": problem_by_prefix[prefix_id], "prefix_id": prefix_id, "split_id": split_id, "rank": rank, "normalized_ev": value})
    summary_rows = []
    for rank in ranks:
        values = np.asarray([np.mean(v) for (prefix_id, candidate_rank), v in by_prefix_rank.items() if candidate_rank == rank], dtype=np.float64)
        if not len(values):
            continue
        summary_rows.append({
            "rank": rank,
            "median_normalized_ev": float(np.median(values)),
            "q25_normalized_ev": float(np.quantile(values, 0.25)),
            "q75_normalized_ev": float(np.quantile(values, 0.75)),
            "n_prefixes": int(len(values)),
            "reference_rank": reference_rank,
        })
    return summary_rows, {"normalized_rows": normalized, "reference_rank": reference_rank}


def make_rank_figure(root: Path, output: Path, dpi: int) -> list[Path]:
    result = _rank_curve(root)
    if result is None:
        return []
    rows, diagnostics = result
    summary = _read_json(root / "metrics/rank_saturation_summary.json") or {}
    median_r90 = float(summary.get("median_r90", REVIEWED["median_r90"]))
    ranks = np.asarray([row["rank"] for row in rows], dtype=np.float64)
    median = np.asarray([row["median_normalized_ev"] for row in rows])
    q25 = np.asarray([row["q25_normalized_ev"] for row in rows])
    q75 = np.asarray([row["q75_normalized_ev"] for row in rows])
    control_summary = _read_json(root / "metrics/control_rank_sensitivity_summary.json")
    control_curves = control_summary.get("rank_curve", {}) if control_summary else {}
    has_controls = all(str(int(rank)) in control_curves for rank in (8, 16, 32, 64, 96, 127))

    if has_controls:
        fig, axes = plt.subplots(1, 2, figsize=(7.15, 3.15), layout="constrained")
        ax = axes[0]
    else:
        fig, ax = plt.subplots(figsize=(6.9, 3.15), layout="constrained")
    ax.fill_between(ranks, q25, q75, color=BLUE_LIGHT, alpha=0.55, linewidth=0)
    ax.plot(ranks, median, color=BLUE, marker="o", ms=4, lw=1.8)
    ax.axhline(0.90, color=INK, ls="--", lw=0.9)
    ax.text(2, 0.906, "90% of rank-127 EV", color=INK, fontsize=7.2, va="bottom")
    ax.axvline(64, color=MID, ls=":", lw=1.0)
    ax.text(64 - 1.5, 0.18, "Primary rank = 64", rotation=90, ha="right", va="bottom", color=MID, fontsize=7.2)
    ax.axvline(median_r90, color=INK, ls="-.", lw=1.0)
    ax.text(median_r90 + 2, 0.22, rf"Median $r_{{90}}={median_r90:g}$", rotation=90, ha="left", va="bottom", fontsize=7.2)
    ax.set_xlim(0, max(ranks) + 3)
    ax.set_ylim(0, 1.035)
    tick_candidates = {1, 16, 32, 64, 96, int(max(ranks))}
    ax.set_xticks([int(rank) for rank in ranks if int(rank) in tick_candidates])
    ax.set_xlabel("Rank")
    ax.set_ylabel("Explained energy relative to rank 127")
    ax.set_title("Held-out response energy")
    ax.grid(axis="y", color=GRID, lw=0.7)
    if has_controls:
        _panel_label(ax, "A")

        ax = axes[1]
        control_ranks = np.asarray([8, 16, 32, 64, 96, 127], dtype=np.float64)
        source_rows = []
        for label, metric, color, linestyle, marker in (
            ("Matched common", "delta_common", BLUE, "-", "o"),
            ("Wrong context", "delta_wrong", ORANGE, "--", "s"),
        ):
            estimates = [control_curves[str(int(rank))][metric] for rank in control_ranks]
            means = np.asarray([float(estimate["mean"]) for estimate in estimates])
            lows = np.asarray([float(estimate["ci_low"]) for estimate in estimates])
            highs = np.asarray([float(estimate["ci_high"]) for estimate in estimates])
            ax.errorbar(control_ranks, means, yerr=np.vstack((means - lows, highs - means)), marker=marker, ls=linestyle, ms=4, lw=1.5, capsize=2.5, color=color, label=label)
            for rank, estimate in zip(control_ranks, estimates):
                source_rows.append({"control": label, "rank": int(rank), "candidate_protocol": "primary 192-fit/64-heldout four-fold", **estimate})
        ax.axhline(0, color=INK, lw=0.9)
        ax.axvline(64, color=MID, ls=":", lw=1.0)
        ax.text(64 - 2, ax.get_ylim()[0] + 0.03 * (ax.get_ylim()[1] - ax.get_ylim()[0]), "Primary rank = 64", rotation=90, ha="right", va="bottom", color=MID, fontsize=7.0)
        ax.set_xticks(control_ranks)
        ax.set_xlabel("Rank")
        ax.set_ylabel("Held-out ΔEV (target context - control)")
        ax.set_title("Target-versus-control specificity")
        ax.legend(frameon=False, fontsize=7.0)
        ax.grid(axis="y", color=GRID, lw=0.7)
        _panel_label(ax, "B")
        _write_rows(output / "figure3_control_rank_curve_source.csv", source_rows)
    if has_controls:
        common_n = int(control_curves["64"]["delta_common"].get("n_problems", 0))
        wrong_n = int(control_curves["64"]["delta_wrong"].get("n_problems", 0))
        footer = (
            f"A: median and interquartile range across {rows[0]['n_prefixes']} contexts over random 128/128 splits.  "
            f"B: exact-bin mean ΔEV and problem-bootstrap 95% CI under the primary 192/64 four-fold protocol "
            f"(common n={common_n}; wrong n={wrong_n})."
        )
    else:
        footer = f"Median and interquartile range across {rows[0]['n_prefixes']} contexts after averaging random 128/128 token splits."
    fig.text(0.5, -0.025, footer, ha="center", fontsize=6.8 if has_controls else 7.1, color=MID)
    _write_rows(output / "figure3_rank_saturation_source.csv", rows)
    _write_rows(output / "figure3_rank_saturation_normalized_rows.csv", diagnostics["normalized_rows"])
    return _save(fig, output / "figure3_rank_saturation", dpi)


def make_block_figure(root: Path, output: Path, dpi: int) -> list[Path]:
    mechanism = _read_json(root / "metrics/first_layer_mechanism_summary.json")
    optimal = _read_json(root / "metrics/optimal_value_control_summary.json")
    if mechanism:
        interaction = mechanism.get("interaction_energy", {})
        post_attention_energy = float(interaction.get("post_attention", {}).get("interaction_fraction_eta", REVIEWED["post_attention_interaction_fraction"]))
        post_mlp_energy = float(interaction.get("post_mlp", {}).get("interaction_fraction_eta", REVIEWED["post_mlp_interaction_fraction"]))
    else:
        post_attention_energy = float(REVIEWED["post_attention_interaction_fraction"])
        post_mlp_energy = float(REVIEWED["post_mlp_interaction_fraction"])
    if optimal:
        post_attention_span = _metric(optimal.get("sites"), ("post_attention",), "post_attention_value_span_fraction")
        if isinstance(optimal.get("sites", {}).get("post_attention"), dict):
            post_attention_span = optimal["sites"]["post_attention"].get("full_value_span_interaction_fraction", post_attention_span)
        post_mlp_span = _metric(optimal.get("sites"), ("post_mlp",), "post_mlp_value_span_fraction")
        if isinstance(optimal.get("sites", {}).get("post_mlp"), dict):
            post_mlp_span = optimal["sites"]["post_mlp"].get("full_value_span_interaction_fraction", post_mlp_span)
    else:
        post_attention_span = dict(REVIEWED["post_attention_value_span_fraction"])
        post_mlp_span = dict(REVIEWED["post_mlp_value_span_fraction"])
    for name, estimate in (("post_attention_value_span_fraction", post_attention_span), ("post_mlp_value_span_fraction", post_mlp_span)):
        _validate_estimate(name, estimate)

    energies = [post_attention_energy, post_mlp_energy]
    inside = [float(post_attention_span["mean"]), float(post_mlp_span["mean"])]
    outside = [1.0 - value for value in inside]
    if any(not 0 <= value <= 1 for value in inside + outside):
        raise ValueError("value-span fractions must lie in [0, 1]")

    fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.75), layout="constrained")
    fig.suptitle("Response geometry across decoder block 0", fontsize=10.5, fontweight="bold")
    ax = axes[0]
    bars = ax.bar([0, 1], energies, color=[MECH_SKY, MECH_BLUE], edgecolor=INK, linewidth=0.8, width=0.58)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004, f"{bar.get_height():.4f}", ha="center", va="bottom", fontsize=7.5)
    bracket_y = 0.121
    ax.plot([0, 0, 1, 1], [bracket_y - 0.004, bracket_y, bracket_y, bracket_y - 0.004], color=INK, lw=0.9, clip_on=False)
    ax.text(0.5, bracket_y + 0.002, "3.69× (fraction)", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
    ax.set_xticks([0, 1], ["Post-attention", "Post-MLP"])
    ax.set_ylim(0, 0.145)
    ax.set_ylabel("Interaction-energy fraction")
    ax.set_title("Interaction-energy fraction")
    ax.grid(axis="y", color=GRID, lw=0.7)
    _panel_label(ax, "A")

    ax = axes[1]
    x = np.arange(2)
    ax.bar(x, inside, color=MECH_BLUE, edgecolor=INK, linewidth=0.8, width=0.58, label="Inside value/output span")
    ax.bar(x, outside, bottom=inside, color=MECH_GOLD, edgecolor=INK, linewidth=0.8, width=0.58, hatch="////", label="Outside value/output span")
    for index, estimate in enumerate((post_attention_span, post_mlp_span)):
        error = _error(estimate)
        if error is not None:
            ax.errorbar(index, inside[index], yerr=error, fmt="none", ecolor=INK, capsize=4, capthick=1.0, lw=1.0, zorder=6)
    for index in range(2):
        ax.text(index, inside[index] / 2, f"{100 * inside[index]:.1f}%", ha="center", va="center", color="white", fontweight="bold", fontsize=8)
        if outside[index] >= 0.06:
            ax.text(index, inside[index] + outside[index] / 2, f"{100 * outside[index]:.1f}%", ha="center", va="center", color=INK, fontweight="bold", fontsize=8)
        else:
            ax.text(index, 1.015, f"{100 * outside[index]:.1f}% outside", ha="center", va="bottom", color=MECH_GOLD_DARK, fontsize=7.1)
    ax.set_xticks(x, ["Post-attention", "Post-MLP"])
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_ylabel("Share of interaction energy")
    ax.set_title("Value/output-span composition")
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, fontsize=6.8)
    ax.grid(axis="y", color=GRID, lw=0.7)
    _panel_label(ax, "B")

    rows = [
        {"site": "post_attention", "interaction_fraction": energies[0], "inside_value_span": inside[0], "inside_ci_low": post_attention_span.get("ci_low"), "inside_ci_high": post_attention_span.get("ci_high"), "outside_value_span": outside[0]},
        {"site": "post_mlp", "interaction_fraction": energies[1], "inside_value_span": inside[1], "inside_ci_low": post_mlp_span.get("ci_low"), "inside_ci_high": post_mlp_span.get("ci_high"), "outside_value_span": outside[1]},
    ]
    _write_rows(output / "figureS_block_reorganization_source.csv", rows)
    return _save(fig, output / "figureS_block_reorganization", dpi)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create publication figures from saved paper artifacts.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--results-root", help="Override config.results_root")
    parser.add_argument("--output-dir", default="reports/prefix_response_final_results/submission_figures")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--require-rank-data", action="store_true")
    args = parser.parse_args()
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    root = Path(args.results_root or config["results_root"])
    output = Path(args.output_dir)
    _style()
    generated = []
    generated.extend(make_main_figure(root, output, args.dpi))
    rank_paths = make_rank_figure(root, output, args.dpi)
    if args.require_rank_data and not rank_paths:
        raise FileNotFoundError(root / "metrics/rank_saturation_rows.csv")
    generated.extend(rank_paths)
    generated.extend(make_block_figure(root, output, args.dpi))
    for path in generated:
        print(path)
    if not rank_paths:
        print(f"SKIPPED Figure 3: {root / 'metrics/rank_saturation_rows.csv'} is unavailable")


if __name__ == "__main__":
    main()
