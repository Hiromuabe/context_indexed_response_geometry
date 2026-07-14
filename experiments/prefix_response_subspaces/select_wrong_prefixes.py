from __future__ import annotations

import argparse

from .src.utils import atomic_json, atomic_jsonl, ensure_layout, file_sha256, load_config, read_jsonl, stable_hash, stage_is_complete


def select_wrong_prefix_controls(prefixes: list[dict], count: int, prefer_same_last_token: bool = True) -> list[dict]:
    # Keep the conditional-Global training set fixed, but allow the dedicated
    # matching pool to supply Wrong-prefix donors.  This makes five controls per
    # target feasible without leaking development/test targets into a basis.
    donor_groups = {"analysis_train", "matching_pool"}
    donors = [row for row in prefixes if row["problem_group"] in donor_groups]
    targets = [row for row in prefixes if row["problem_group"] in {"analysis_dev", "analysis_test"}]
    rows = []
    for target in targets:
        eligible = [row for row in donors if row["problem_id"] != target["problem_id"] and int(row["prefix_length_bin"]) == int(target["prefix_length_bin"]) and int(row["reasoning_progress_bin"]) == int(target["reasoning_progress_bin"])]
        eligible.sort(key=lambda row: (0 if prefer_same_last_token and int(row["last_token_id"]) == int(target["last_token_id"]) else 1, abs(int(row["prefix_length"])-int(target["prefix_length"])), abs(float(row["prefix_position_fraction"])-float(target["prefix_position_fraction"])), str(row["prefix_id"])))
        selected = eligible[:count]
        # A rare joint bin can be empty in the dedicated donor pool.  Fill only
        # those shortages from already-extracted prefixes in the exact same bin.
        # Development selection never borrows evaluation prefixes.
        fallback_groups = {"analysis_dev"} if target["problem_group"] == "analysis_dev" else {"analysis_dev", "analysis_test"}
        selected_ids = {row["prefix_id"] for row in selected}
        fallback = [row for row in prefixes if row["problem_group"] in fallback_groups and row["prefix_id"] != target["prefix_id"] and row["prefix_id"] not in selected_ids and row["problem_id"] != target["problem_id"] and int(row["prefix_length_bin"]) == int(target["prefix_length_bin"]) and int(row["reasoning_progress_bin"]) == int(target["reasoning_progress_bin"])]
        fallback.sort(key=lambda row: (0 if row["problem_group"] == "analysis_dev" else 1, 0 if prefer_same_last_token and int(row["last_token_id"]) == int(target["last_token_id"]) else 1, abs(int(row["prefix_length"])-int(target["prefix_length"])), abs(float(row["prefix_position_fraction"])-float(target["prefix_position_fraction"])), str(row["prefix_id"])))
        used_fallback = fallback[:max(0,count-len(selected))]; selected.extend(used_fallback)
        selected_ids.update(row["prefix_id"] for row in used_fallback)
        relaxed_groups={"analysis_train","analysis_dev"} if target["problem_group"]=="analysis_dev" else {"analysis_train","analysis_dev","analysis_test"}
        relaxed=[row for row in prefixes if row["problem_group"] in relaxed_groups and row["prefix_id"] != target["prefix_id"] and row["prefix_id"] not in selected_ids and row["problem_id"] != target["problem_id"] and int(row["reasoning_progress_bin"]) == int(target["reasoning_progress_bin"])]
        relaxed.sort(key=lambda row:(abs(int(row["prefix_length_bin"])-int(target["prefix_length_bin"])),0 if prefer_same_last_token and int(row["last_token_id"])==int(target["last_token_id"]) else 1,abs(int(row["prefix_length"])-int(target["prefix_length"])),str(row["prefix_id"])))
        used_relaxed=relaxed[:max(0,count-len(selected))]; selected.extend(used_relaxed)
        rows.append({"prefix_id": target["prefix_id"], "problem_id": target["problem_id"], "split": "development" if target["problem_group"] == "analysis_dev" else "evaluation", "wrong_prefix_ids": [row["prefix_id"] for row in selected], "wrong_problem_ids": [row["problem_id"] for row in selected], "wrong_problem_groups": [row["problem_group"] for row in selected], "same_last_token_count": sum(int(row["last_token_id"]) == int(target["last_token_id"]) for row in selected), "primary_eligible_pool_size": len(eligible), "fallback_eligible_pool_size":len(fallback), "fallback_wrong_prefixes":len(used_fallback), "relaxed_length_wrong_prefixes":len(used_relaxed), "maximum_length_bin_distance":max([abs(int(row["prefix_length_bin"])-int(target["prefix_length_bin"])) for row in used_relaxed],default=0), "requested_wrong_prefixes": count, "selected_wrong_prefixes": len(selected), "complete": len(selected) == count})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", required=True); parser.add_argument("--force",action="store_true"); args = parser.parse_args()
    config = load_config(args.config); root = ensure_layout(config); prefix_path = root / "prefix_pool/prefixes.jsonl"
    prefix_hash = file_sha256(prefix_path); manifest_path = root / "manifests/wrong_prefixes.json"
    if not args.force and stage_is_complete(manifest_path, config, {"prefixes_sha256": prefix_hash}): print(manifest_path); return
    rows = select_wrong_prefix_controls(read_jsonl(prefix_path), int(config["controls"]["wrong_prefixes_per_target"]), bool(config["controls"].get("prefer_same_last_token", True)))
    output = root / "controls/wrong_prefixes.jsonl"; output.parent.mkdir(parents=True, exist_ok=True); atomic_jsonl(output, rows)
    evaluation = [row for row in rows if row["split"] == "evaluation"]
    diagnostics = {"requested_targets": len(evaluation), "complete_targets": sum(row["complete"] for row in evaluation), "exact_bin_complete_targets":sum(row["complete"] and row["relaxed_length_wrong_prefixes"]==0 for row in evaluation), "exact_bin_complete_fraction":sum(row["complete"] and row["relaxed_length_wrong_prefixes"]==0 for row in evaluation)/max(1,len(evaluation)), "targets_with_any_wrong_prefix": sum(row["selected_wrong_prefixes"] > 0 for row in evaluation), "mean_wrong_prefixes_per_target": sum(row["selected_wrong_prefixes"] for row in evaluation)/max(1, len(evaluation)), "all_evaluation_targets_complete": bool(evaluation) and all(row["complete"] for row in evaluation), "evaluation_targets_using_fallback":sum(row["fallback_wrong_prefixes"]>0 for row in evaluation), "evaluation_fallback_wrong_prefixes":sum(row["fallback_wrong_prefixes"] for row in evaluation), "evaluation_targets_using_relaxed_length_fallback":sum(row["relaxed_length_wrong_prefixes"]>0 for row in evaluation), "evaluation_relaxed_length_wrong_prefixes":sum(row["relaxed_length_wrong_prefixes"] for row in evaluation), "donor_groups":["analysis_train","matching_pool","analysis_dev","analysis_test"], "primary_donor_groups": ["analysis_train", "matching_pool"], "fallback_donor_groups":["analysis_dev","analysis_test"], "relaxed_length_donor_groups":["analysis_train","analysis_dev","analysis_test"], "selection_variables": ["same reasoning-progress bin always", "same length bin for primary/exact fallback", "nearest length bin only when exact bin is structurally empty", "different problem", "prefer same last token", "nearest length/progress metadata"]}
    atomic_json(manifest_path, {"complete": True, "config_hash": stable_hash(config), "prefixes_sha256": prefix_hash, "wrong_prefixes": str(output), "wrong_prefixes_sha256": file_sha256(output), "diagnostics": diagnostics})
    if str(config.get("profile","")).startswith("paper_full") and not diagnostics["all_evaluation_targets_complete"]:
        raise RuntimeError("Full paper run has an evaluation prefix without five exact-bin Wrong controls; diagnostics were saved")
    print(manifest_path)


if __name__ == "__main__": main()
