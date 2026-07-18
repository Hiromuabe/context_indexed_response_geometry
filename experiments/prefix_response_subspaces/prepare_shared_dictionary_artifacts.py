from __future__ import annotations

import argparse
import csv
import json
import math
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any


REQUIRED_FILES = (
    "paired_summary.csv",
    "optimization_diagnostics.json",
    "heldout_ev_all.csv",
)

BASELINE_METHODS = (
    "target_context_pca",
    "matched_common",
    "wrong_context",
)
SHARED_METHODS = (
    "pooled_pca_context_selection",
    "truncated_cpc",
    "shared_nonorthogonal_dictionary",
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no", ""}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _as_number_key(value: Any) -> str:
    text = str(value).strip()
    if not text:
        return ""
    return format(float(text), ".17g")


def _setting_key(row: dict[str, Any]) -> tuple[str, str, int, str, bool]:
    return (
        str(row["method"]),
        str(row.get("dictionary_size", "")).strip(),
        int(row["rank"]),
        _as_number_key(row.get("coherence_penalty", "")),
        _as_bool(row.get("leave_one_context_out", False)),
    )


def _expected_summary_settings(run: dict[str, Any]) -> set[tuple[str, str, int, str, bool]]:
    sizes = sorted(set(map(int, run.get("dictionary_sizes", []))))
    ranks = sorted(set(map(int, run.get("ranks", []))))
    betas = sorted(set(map(float, run.get("coherence_penalties", []))))
    expected: set[tuple[str, str, int, str, bool]] = set()
    for rank in ranks:
        for method in BASELINE_METHODS:
            expected.add((method, "", rank, "", False))
    for size in sizes:
        for rank in ranks:
            if rank > size:
                continue
            expected.add(("pooled_pca_context_selection", str(size), rank, "", False))
            expected.add(("truncated_cpc", str(size), rank, "", False))
            for beta in betas:
                expected.add(("shared_nonorthogonal_dictionary", str(size), rank, _as_number_key(beta), False))
            if _as_bool(run.get("leave_one_context_out", False)):
                expected.add(("truncated_cpc", str(size), rank, "", True))
                for beta in betas:
                    expected.add(("shared_nonorthogonal_dictionary", str(size), rank, _as_number_key(beta), True))
    return expected


def audit_paired_summary(rows: list[dict[str, str]], run: dict[str, Any]) -> dict[str, Any]:
    keys = [_setting_key(row) for row in rows]
    counts: dict[tuple[str, str, int, str, bool], int] = defaultdict(int)
    for key in keys:
        counts[key] += 1
    duplicates = [key for key, count in counts.items() if count > 1]
    expected = _expected_summary_settings(run)
    observed = set(keys)
    missing = sorted(expected - observed)
    numeric_fields = (
        "heldout_ev_mean", "heldout_ev_ci_low", "heldout_ev_ci_high",
        "target_minus_method_mean", "target_minus_method_ci_low", "target_minus_method_ci_high",
    )
    nonfinite_values: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        for field in numeric_fields:
            if field not in row:
                continue
            try:
                value = float(row[field])
            except (TypeError, ValueError):
                value = float("nan")
            if not math.isfinite(value):
                nonfinite_values.append({
                    "csv_row": row_number, "setting": list(_setting_key(row)),
                    "field": field, "value": row.get(field),
                })
    return {
        "row_count": len(rows),
        "expected_setting_count": len(expected),
        "observed_setting_count": len(observed),
        "missing_setting_count": len(missing),
        "missing_settings": [list(item) for item in missing],
        "duplicate_setting_count": len(duplicates),
        "duplicate_settings": [list(item) for item in sorted(duplicates)],
        "nonfinite_value_count": len(nonfinite_values),
        "nonfinite_values": nonfinite_values,
    }


def _fit_identity(fit: dict[str, Any]) -> dict[str, Any]:
    optimization = fit.get("optimization", {})
    return {
        "fold": fit.get("fold"),
        "excluded_prefix_id": fit.get("excluded_prefix_id"),
        "kind": fit.get("kind"),
        "dictionary_size": fit.get("dictionary_size"),
        "coherence_penalty": optimization.get("coherence_penalty"),
        "selected_restart": fit.get("selected_restart"),
    }


def audit_optimization_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    fits = list(payload.get("fit_diagnostics", []))
    failed_restarts: list[dict[str, Any]] = []
    maximum_step_restarts: list[dict[str, Any]] = []
    cpc_rows: list[dict[str, Any]] = []
    dictionary_rows: list[dict[str, Any]] = []
    finite_cpc_values: list[float] = []
    finite_dictionary_values: list[float] = []
    nonfinite_diagnostic_count = 0
    successful_restart_count = 0
    restart_count = 0
    for fit in fits:
        identity = _fit_identity(fit)
        maximum_steps = int(fit.get("optimization", {}).get("maximum_steps", 0) or 0)
        kind = str(fit.get("kind", ""))
        if kind == "cpc":
            value = float(fit.get("orthogonality_error_frobenius", float("nan")))
            if math.isfinite(value):
                finite_cpc_values.append(value)
            else:
                nonfinite_diagnostic_count += 1
            cpc_rows.append({**identity, "orthogonality_error_frobenius": value if math.isfinite(value) else None})
        elif kind == "dictionary":
            value = float(fit.get("maximum_column_coherence", float("nan")))
            if math.isfinite(value):
                finite_dictionary_values.append(value)
            else:
                nonfinite_diagnostic_count += 1
            dictionary_rows.append({**identity, "maximum_column_coherence": value if math.isfinite(value) else None})
        for restart in fit.get("restarts", []):
            restart_count += 1
            row = {**identity, **restart}
            if str(restart.get("status", "")) != "success":
                failed_restarts.append(row)
                continue
            successful_restart_count += 1
            steps = int(restart.get("steps_executed", 0) or 0)
            if maximum_steps and steps >= maximum_steps:
                maximum_step_restarts.append({**row, "maximum_steps": maximum_steps})
    return {
        "fit_count": len(fits),
        "restart_count": restart_count,
        "successful_restart_count": successful_restart_count,
        "failed_restart_count": len(failed_restarts),
        "failed_restarts": failed_restarts,
        "maximum_step_restart_count": len(maximum_step_restarts),
        "maximum_step_restarts": maximum_step_restarts,
        "cpc_fit_count": len(cpc_rows),
        "cpc_orthogonality_error_max": max(finite_cpc_values, default=None),
        "cpc_orthogonality": cpc_rows,
        "dictionary_fit_count": len(dictionary_rows),
        "dictionary_coherence_max": max(finite_dictionary_values, default=None),
        "dictionary_coherence": dictionary_rows,
        "nonfinite_diagnostic_count": nonfinite_diagnostic_count,
    }


def audit_heldout_rows(
    rows: list[dict[str, str]], summary_rows: list[dict[str, str]], expected_fold_ids: list[int],
) -> dict[str, Any]:
    if not rows:
        return {
            "row_count": 0,
            "setting_count": 0,
            "expected_fold_ids": [],
            "nonfinite_heldout_ev_count": 0,
            "nonfinite_rows": [],
            "duplicate_grain_count": 0,
            "duplicate_grains": [],
            "missing_problem_setting_count": 0,
            "missing_problem_settings": [],
            "missing_fold_count": 0,
            "missing_folds": [],
        }
    target_rows = [
        row for row in rows
        if row.get("method") == "target_context_pca" and not _as_bool(row.get("leave_one_context_out", False))
    ]
    primary_problem_ids = sorted({str(row["problem_id"]) for row in target_rows})
    observed: dict[tuple[str, str, int, str, bool], dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))
    grains: dict[tuple[Any, ...], int] = defaultdict(int)
    nonfinite_rows: list[dict[str, Any]] = []
    for row_number, row in enumerate(rows, start=2):
        setting = _setting_key(row)
        problem_id = str(row["problem_id"])
        fold = int(row["fold"])
        observed[setting][problem_id].add(fold)
        grain = (problem_id, str(row.get("prefix_id", "")), fold, *setting)
        grains[grain] += 1
        try:
            value = float(row["heldout_ev"])
        except (KeyError, TypeError, ValueError):
            value = float("nan")
        if not math.isfinite(value):
            nonfinite_rows.append({"csv_row": row_number, "problem_id": problem_id, "fold": fold, "setting": list(setting), "heldout_ev": row.get("heldout_ev")})
    duplicate_grains = [grain for grain, count in grains.items() if count > 1]
    missing_problem_settings: list[dict[str, Any]] = []
    missing_folds: list[dict[str, Any]] = []
    summary_settings = {_setting_key(row) for row in summary_rows}
    for setting in sorted(summary_settings):
        problems = primary_problem_ids if not setting[-1] else sorted(observed.get(setting, {}))
        for problem_id in problems:
            folds = observed.get(setting, {}).get(problem_id, set())
            if not folds:
                missing_problem_settings.append({"problem_id": problem_id, "setting": list(setting)})
                continue
            absent = sorted(set(expected_fold_ids) - folds)
            if absent:
                missing_folds.append({"problem_id": problem_id, "setting": list(setting), "missing_fold_ids": absent})
    return {
        "row_count": len(rows),
        "setting_count": len(observed),
        "expected_fold_ids": expected_fold_ids,
        "primary_problem_count": len(primary_problem_ids),
        "nonfinite_heldout_ev_count": len(nonfinite_rows),
        "nonfinite_rows": nonfinite_rows,
        "duplicate_grain_count": len(duplicate_grains),
        "duplicate_grains": [list(item) for item in sorted(duplicate_grains, key=str)],
        "missing_problem_setting_count": len(missing_problem_settings),
        "missing_problem_settings": missing_problem_settings,
        "missing_fold_count": len(missing_folds),
        "missing_folds": missing_folds,
    }


def _configured_fold_ids(run_dir: Path, diagnostics: dict[str, Any]) -> tuple[list[int], str]:
    resolved_path = run_dir / "config_resolved.json"
    if resolved_path.is_file():
        resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
        configured = resolved.get("paper_config", {}).get("candidates", {}).get("folds")
        if isinstance(configured, int):
            return list(range(configured)), "config_resolved.json"
        if isinstance(configured, list):
            ids = [int(row.get("fold_id", index)) if isinstance(row, dict) else int(row) for index, row in enumerate(configured)]
            return sorted(set(ids)), "config_resolved.json"
    manifest_path = run_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        candidate_path = Path(str(manifest.get("inputs", {}).get("candidates", {}).get("path", "")))
        if candidate_path.is_file():
            candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
            ids = [int(row["fold_id"]) for row in candidates.get("folds", [])]
            if ids:
                return sorted(set(ids)), "candidate_tokens.json"
    configured = diagnostics.get("run", {}).get("candidate_fold_ids")
    if isinstance(configured, list) and configured:
        return sorted(set(map(int, configured))), "optimization_diagnostics.json"
    return [], "unavailable"


def audit_run(run_dir: Path) -> dict[str, Any]:
    paths = {name: run_dir / name for name in REQUIRED_FILES}
    missing_files = [str(path) for path in paths.values() if not path.is_file()]
    if missing_files:
        return {"status": "FAIL", "run_dir": str(run_dir), "missing_files": missing_files}
    summary_rows = _read_csv(paths["paired_summary.csv"])
    heldout_rows = _read_csv(paths["heldout_ev_all.csv"])
    diagnostics = json.loads(paths["optimization_diagnostics.json"].read_text(encoding="utf-8"))
    expected_fold_ids, fold_source = _configured_fold_ids(run_dir, diagnostics)
    paired_audit = audit_paired_summary(summary_rows, diagnostics.get("run", {}))
    optimization_audit = audit_optimization_diagnostics(diagnostics)
    heldout_audit = audit_heldout_rows(heldout_rows, summary_rows, expected_fold_ids)
    heldout_audit["expected_fold_source"] = fold_source
    blockers = {
        "paired_missing_settings": paired_audit["missing_setting_count"],
        "paired_duplicate_settings": paired_audit["duplicate_setting_count"],
        "paired_nonfinite_values": paired_audit["nonfinite_value_count"],
        "heldout_nonfinite": heldout_audit["nonfinite_heldout_ev_count"],
        "heldout_duplicate_grains": heldout_audit["duplicate_grain_count"],
        "heldout_missing_problem_settings": heldout_audit["missing_problem_setting_count"],
        "heldout_missing_folds": heldout_audit["missing_fold_count"],
        "expected_fold_definition_missing": int(not expected_fold_ids),
        "nonfinite_optimization_diagnostics": optimization_audit["nonfinite_diagnostic_count"],
    }
    return {
        "status": "PASS" if not any(blockers.values()) else "FAIL",
        "run_dir": str(run_dir),
        "required_files": {name: str(path) for name, path in paths.items()},
        "blockers": blockers,
        "paired_summary_audit": paired_audit,
        "optimization_audit": optimization_audit,
        "heldout_ev_audit": heldout_audit,
    }


def _write_archive(run_dir: Path, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(output, "w:gz") as archive:
        for name in REQUIRED_FILES:
            archive.add(run_dir / name, arcname=name, recursive=False)


def _print_report(audit: dict[str, Any], archive: Path | None) -> None:
    print(f"ARTIFACT AUDIT: {audit['status']}")
    if audit.get("missing_files"):
        for path in audit["missing_files"]:
            print(f"MISSING {path}")
        return
    paired = audit["paired_summary_audit"]
    optimization = audit["optimization_audit"]
    heldout = audit["heldout_ev_audit"]
    print(
        "paired_summary: "
        f"rows={paired['row_count']} settings={paired['observed_setting_count']}/"
        f"{paired['expected_setting_count']} missing={paired['missing_setting_count']} "
        f"duplicates={paired['duplicate_setting_count']}"
    )
    print(
        "optimization: "
        f"fits={optimization['fit_count']} restarts={optimization['restart_count']} "
        f"failed={optimization['failed_restart_count']} "
        f"reached_max_steps={optimization['maximum_step_restart_count']}"
    )
    print(
        "cpc: "
        f"fits={optimization['cpc_fit_count']} "
        f"max_orthogonality_error={optimization['cpc_orthogonality_error_max']}"
    )
    print(
        "dictionary: "
        f"fits={optimization['dictionary_fit_count']} "
        f"max_column_coherence={optimization['dictionary_coherence_max']}"
    )
    print(
        "heldout_ev: "
        f"rows={heldout['row_count']} settings={heldout['setting_count']} "
        f"folds={heldout['expected_fold_ids']} missing_folds={heldout['missing_fold_count']} "
        f"missing_problem_settings={heldout['missing_problem_setting_count']} "
        f"nonfinite={heldout['nonfinite_heldout_ev_count']} duplicates={heldout['duplicate_grain_count']}"
    )
    if archive is not None:
        print(f"archive: {archive}")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Audit and package the three reviewer-facing shared-dictionary artifacts",
    )
    value.add_argument("--run-dir", required=True, type=Path)
    value.add_argument(
        "--archive",
        type=Path,
        help="tar.gz output; defaults to RUN_DIR/shared_dictionary_review_artifacts.tar.gz",
    )
    value.add_argument("--no-archive", action="store_true")
    value.add_argument("--json-report", type=Path, help="optional full machine-readable audit report")
    return value


def run(args: argparse.Namespace) -> int:
    run_dir = args.run_dir.resolve()
    audit = audit_run(run_dir)
    archive: Path | None = None
    if audit["status"] == "PASS" and not args.no_archive:
        archive = (args.archive or run_dir / "shared_dictionary_review_artifacts.tar.gz").resolve()
        _write_archive(run_dir, archive)
    if args.json_report:
        args.json_report.parent.mkdir(parents=True, exist_ok=True)
        args.json_report.write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _print_report(audit, archive)
    return 0 if audit["status"] == "PASS" else 1


def main() -> None:
    raise SystemExit(run(parser().parse_args()))


if __name__ == "__main__":
    main()
