from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .src.utils import load_config, read_json


SUMMARY_FILES = {
    "candidate_distribution": "candidate_distribution_transfer_summary.json",
    "context_controls": "context_control_summary.json",
    "jacobian_alignment": "jacobian_alignment_summary.json",
    "subspace_stability": "subspace_stability_summary.json",
    "response_law_state": "response_law_state_summary.json",
}


def review_result_root(config: dict[str, Any], quick_check: bool, formal_check: bool = False) -> Path:
    root = Path(str(config["results_root"]))
    if quick_check and not bool(config.get("quick_check")):
        root = root.with_name(f"{root.name}_quick_check")
    elif formal_check and not bool(config.get("formal_check")):
        root = root.with_name(f"{root.name}_formal_check")
    return root


def collect_review_results(root: Path) -> dict[str, Any]:
    metrics = root / "metrics"
    result: dict[str, Any] = {
        "output_root": str(root),
        "pilot_only": root.name.endswith("_quick_check"),
        "formal_check": root.name.endswith("_formal_check"),
        "available": {},
        "missing": [],
    }
    for key, filename in SUMMARY_FILES.items():
        path = metrics / filename
        if path.is_file():
            result["available"][key] = read_json(path)
        else:
            result["missing"].append(str(path))
    return result


def _interval(value: dict[str, Any] | None) -> str:
    if not value or "mean" not in value:
        return "未計算"
    mean = float(value["mean"])
    if "ci_low" in value and "ci_high" in value:
        return f"{mean:.4f} [{float(value['ci_low']):.4f}, {float(value['ci_high']):.4f}]"
    return f"{mean:.4f}"


def print_review_results(result: dict[str, Any]) -> None:
    available = result["available"]
    if result["pilot_only"]:
        label = "QUICK CHECK（傾向確認のみ）"
    elif result["formal_check"]:
        label = "FORMAL CHECK（中規模レビュー検証）"
    else:
        label = "FULL REVIEW EXPERIMENT"
    print(f"=== {label} ===")
    print(f"output: {result['output_root']}")

    candidate = available.get("candidate_distribution")
    print("\n[候補分布 transfer]")
    if not candidate:
        print("未実行")
    else:
        pairs = candidate.get("pairs", {})
        order = ("high_to_low", "low_to_high", "independent_A_to_B", "independent_B_to_A")
        for name in order:
            row = pairs.get(name)
            if not row:
                reason = candidate.get("skipped_pairs", {}).get(name, "未計算")
                print(f"{name}: SKIP ({reason})")
                continue
            print(
                f"{name}: transfer/reference={float(row['transfer_fraction_of_target_reference']):.3f}, "
                f"transfer-reference={_interval(row.get('transfer_minus_reference_problem_bootstrap'))}, "
                f"rank={row['rank']}"
            )
        print("見方: transfer/referenceが1に近いほど候補分布を越えてtransferし、低いほど候補分布依存が強い。")

    context = available.get("context_controls")
    print("\n[文脈 control]")
    if not context:
        print("未実行")
    else:
        for name, row in sorted(context.get("controls", {}).items()):
            print(
                f"{name}: target-control="
                f"{_interval(row.get('delta_target_minus_control_problem_bootstrap'))}, "
                f"n_problem={row.get('unique_target_problems')}"
            )
        print("見方: target-controlが正なら、control文脈より元の文脈部分空間がheld-out応答をよく説明する。")

    stability = available.get("subspace_stability")
    print("\n[部分空間 stability]")
    if not stability:
        print("未実行")
    else:
        print(f"within distance: {_interval(stability.get('within_distance'))}")
        print(f"between distance: {_interval(stability.get('between_distance'))}")
        print(f"between-within: {_interval(stability.get('between_minus_within_distance'))}")
        print(f"reliability-corrected between: {_interval(stability.get('reliability_corrected_between_distance'))}")
        print("見方: between-withinが正なら、推定ノイズを超えた文脈間回転がある。")

    jacobian = available.get("jacobian_alignment")
    print("\n[Jacobian alignment]")
    if not jacobian:
        print("未実行（quick-checkでは意図的に省略）" if result["pilot_only"] else "未実行")
    else:
        print(f"projection distance: {_interval(jacobian.get('projection_distance'))}")
        print(f"finite EV by Jacobian: {_interval(jacobian.get('finite_response_ev_by_jacobian_subspace'))}")
        print(f"scaled linearization error: {_interval(jacobian.get('scaled_linearization_relative_squared_error'))}")
        print("見方: projection distanceと線形化誤差が低いほど、有限応答は局所Jacobianで説明される。")

    response_state = available.get("response_law_state")
    print("\n[出力分布の非十分性 / 応答場の一段先contrast伝播]")
    if not response_state:
        print("未実行")
    else:
        print(f"current matched JS: {_interval(response_state.get('current_distribution_match'))}")
        print(f"post-x future JS: {_interval(response_state.get('future_divergence'))}")
        print(f"post-x minus current JS: {_interval(response_state.get('post_x_minus_current_js'))}")
        geometry = response_state.get("contrast_geometry")
        if not geometry:
            print("旧level検定: 構造的に不適切なため解釈しない。contrast-geometry版を再実行してください。")
        else:
            alignment = geometry["candidate_identity_alignment"]
            print(
                f"hidden→future candidate CKA: {_interval(alignment.get('observed_pair_bootstrap'))}, "
                f"permuted={float(alignment['candidate_label_permutation_mean']):.4f}, "
                f"observed-permuted={float(alignment['observed_minus_permutation_mean']):.4f}, "
                f"p={float(alignment['positive_alignment_permutation_p']):.4f}"
            )
            for label, key in (
                ("response-law contrast distance", "response_law_distance_correspondence"),
                ("candidate-kernel distance", "candidate_kernel_distance_correspondence"),
                ("subspace contrast distance", "subspace_distance_correspondence"),
            ):
                block = geometry[key]
                print(
                    f"{label}: Pearson={_interval(block.get('pearson_pair_bootstrap'))}, "
                    f"Spearman={_interval(block.get('spearman_pair_bootstrap'))}, "
                    f"partial beta={_interval(block.get('partial_standardized_beta_controlling_current_js'))}, "
                    f"permutation p={float(block['positive_association_permutation_p']):.4f}, "
                    f"future-distance sd={float(block['outcome_sd']):.4f}"
                )
            print(
                "見方: CKAが候補ID置換nullを上回れば同一文脈内の候補対比幾何が一段先へ伝播する。"
                "距離対応が正なら、文脈間の応答場の差も一段先の対比場の差へ伝播する。"
            )

    if result["missing"]:
        print("\n未生成ファイル:")
        for path in result["missing"]:
            print(f"- {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    scale = parser.add_mutually_exclusive_group()
    scale.add_argument("--quick-check", action="store_true")
    scale.add_argument("--formal-check", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the collected raw summaries as JSON")
    args = parser.parse_args()
    config = load_config(args.config)
    root = review_result_root(config, args.quick_check, args.formal_check)
    result = collect_review_results(root)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        print_review_results(result)


if __name__ == "__main__":
    main()
