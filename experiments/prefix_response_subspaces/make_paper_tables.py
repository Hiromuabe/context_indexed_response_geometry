from __future__ import annotations

import argparse

from .src.utils import atomic_json, ensure_layout, load_config, read_json


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); root=ensure_layout(config); geometry=read_json(root/"metrics/paper_geometry_summary.json"); functional=read_json(root/"functional/paper_summary.json") if (root/"functional/paper_summary.json").exists() else {"status":"not run"}
    additional={name:(read_json(path) if path.exists() else {"status":"not run"}) for name,path in {"rank_saturation":root/"metrics/rank_saturation_summary.json","shared_backbone":root/"metrics/shared_backbone_summary.json","first_layer_mechanism":root/"metrics/first_layer_mechanism_summary.json","optimal_value_control":root/"metrics/optimal_value_control_summary.json","value_space_functional":root/"functional/value_space_summary.json","optimal_value_functional":root/"functional/optimal_value_summary.json"}.items()}; atomic_json(root/"tables/paper_results.json",{"geometry":geometry,"functional":functional,"additional":additional})
    lines=["# Paper results","","| Metric | Mean | CI |","|---|---:|---:|"]
    for key in ("delta_conditional_global","delta_wrong","delta_wrong_exact_bin",f"delta_conditional_global_top{int(config['analysis']['high_probability_primary_top_k'])}",f"delta_wrong_top{int(config['analysis']['high_probability_primary_top_k'])}",f"delta_wrong_top{int(config['analysis']['high_probability_primary_top_k'])}_exact_bin","d_rotation_local_conditional_global","d_rotation_local_wrong_mean","R_within","R_between","R_between_minus_within"):
        row=geometry[key]; lines.append(f"| {key} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    for key in ("G_local","G_local_minus_conditional_global_exact_bin","G_local_minus_wrong_mean_exact_bin","G_local_minus_conditional_global","G_local_minus_wrong_mean"):
        if key in functional:
            row=functional[key]; lines.append(f"| {key} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    for key in ("D_rank0","D_local","D_conditional_global","D_wrong_mean","recovery_fraction_local","recovery_fraction_conditional_global","recovery_fraction_wrong_mean"):
        if key in functional:
            row=functional[key]; lines.append(f"| {key} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    saturation=additional["rank_saturation"]
    if saturation.get("status")!="not run":
        row=saturation["relative_gain_after_compact_rank"]; lines.append(f"| rank64_to_rank127_relative_EV_gain | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
        lines.append(f"| saturation_median_r90 | {saturation['median_r90']:.6g} | — |")
    shared=additional["shared_backbone"]
    if shared.get("status")!="not run":
        row=shared["delta_specific_local_wrong"]; lines.append(f"| shared_removed_Local_minus_Wrong | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    mechanism=additional["first_layer_mechanism"]
    if mechanism.get("status")!="not run":
        for site in ("pre_attention","post_attention","post_mlp"):
            row=mechanism["sites"][site]["delta_local_value_output"]; lines.append(f"| {site}_Local_minus_leading_value_output_rank64_EV | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    optimal_value=additional["optimal_value_control"]
    if optimal_value.get("status")!="not run":
        for site in ("post_attention","post_mlp"):
            row=optimal_value["sites"][site]["delta_EV_local_minus_optimal_value"]; lines.append(f"| {site}_Local_minus_optimal_within_value_span_rank64_EV | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
            for key in ("full_value_span_interaction_fraction","outside_value_span_interaction_fraction"):
                if key in optimal_value["sites"].get(site,{}):
                    row=optimal_value["sites"][site][key]; lines.append(f"| {site}_{key} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    value_functional=additional["value_space_functional"]
    if value_functional.get("status")!="not run":
        for key in ("G_value_output","G_local_minus_value_output"):
            label={"G_value_output":"G_leading_value_output_rank64","G_local_minus_value_output":"G_local_minus_leading_value_output_rank64"}[key]
            row=value_functional[key]; lines.append(f"| {label} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    optimal_functional=additional["optimal_value_functional"]
    if optimal_functional.get("status")!="not run":
        for key in ("G_optimal_value","G_local_minus_optimal_value"):
            label={"G_optimal_value":"G_optimal_within_value_span_rank64","G_local_minus_optimal_value":"G_local_minus_optimal_within_value_span_rank64"}[key]
            row=optimal_functional[key]; lines.append(f"| {label} | {row['mean']:.6g} | [{row['ci_low']:.6g}, {row['ci_high']:.6g}] |")
    (root/"tables/paper_results.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__": main()
