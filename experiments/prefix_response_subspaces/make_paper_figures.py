from __future__ import annotations

import argparse
import csv

import numpy as np

from .src.utils import ensure_layout, load_config, read_json


def _read(path):
    if not path.exists(): return []
    with path.open(encoding="utf-8") as handle: return list(csv.DictReader(handle))


def _write(path,rows):
    if not rows: path.write_text("",encoding="utf-8"); return
    with path.open("w",newline="",encoding="utf-8") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True); args=parser.parse_args(); config=load_config(args.config); root=ensure_layout(config); figures=root/"figures"; geometry=_read(root/"metrics/paper_geometry_rows.csv"); rotation=_read(root/"metrics/paper_rotation_rows.csv"); pooled_top_k=_read(root/"metrics/paper_topk_pooled_rows.csv"); energy=_read(root/"metrics/interaction_energy.csv"); summary=read_json(root/"metrics/paper_geometry_summary.json"); selected_layer=int(summary["selected_layer"])
    _write(figures/"figure1_design.csv",[{"step":1,"name":"common forced-token branches"},{"step":2,"name":"split-local interaction contrast"},{"step":3,"name":"Local / conditional Global / Wrong-prefix"},{"step":4,"name":"rank-0 functional reconstruction"}])
    rank_rows=[]
    for layer in sorted({int(row["layer"]) for row in geometry}):
        for rank in sorted({int(row["rank"]) for row in geometry}):
            selected=[row for row in geometry if row["split"]=="evaluation" and int(row["layer"])==layer and int(row["rank"])==rank]; rank_rows.append({"layer":layer,"rank":rank,"mean_local_ev":float(np.mean([float(row["ev_local"]) for row in selected])) if selected else float("nan")})
    _write(figures/"figure2_interaction_energy.csv",energy); _write(figures/"figure2_rank_ev.csv",rank_rows)
    primary=[row for row in geometry if row["split"]=="evaluation" and int(row["layer"])==selected_layer and int(row["rank"])==int(summary["selected_rank"])]; top_k=int(config["analysis"]["high_probability_primary_top_k"])
    per_prefix=[]
    pooled_by_prefix={row["prefix_id"]:row for row in pooled_top_k}
    for prefix_id in sorted({row["prefix_id"] for row in primary}):
        selected=[row for row in primary if row["prefix_id"]==prefix_id]; pooled=pooled_by_prefix.get(prefix_id,{})
        per_prefix.append({"problem_id":selected[0]["problem_id"],"prefix_id":prefix_id,"delta_conditional_global":float(np.mean([float(row["delta_conditional_global"]) for row in selected])),"delta_wrong":float(np.mean([float(row["delta_wrong"]) for row in selected])),f"delta_conditional_global_top{top_k}":pooled.get(f"delta_conditional_global_top{top_k}",float("nan")),f"delta_wrong_top{top_k}":pooled.get(f"delta_wrong_top{top_k}",float("nan"))})
    _write(figures/"figure3_prefix_specificity.csv",per_prefix)
    _write(figures/"figure3_rotation_reliability.csv",rotation)
    functional=_read(root/"functional/paper_cell_summary.csv"); _write(figures/"figure4_functional_distances.csv",functional)
    saturation=_read(root/"metrics/rank_saturation_rows.csv"); shared_spectrum=_read(root/"metrics/shared_backbone_spectrum.csv"); shared_rows=_read(root/"metrics/shared_backbone_rows.csv"); mechanism_rows=_read(root/"metrics/first_layer_mechanism_rows.csv"); mechanism_energy=_read(root/"metrics/first_layer_mechanism_energy.csv"); value_functional=_read(root/"functional/value_space_rows.csv")
    _write(figures/"figure5_rank_saturation.csv",saturation); _write(figures/"figure6_shared_spectrum.csv",shared_spectrum); _write(figures/"figure6_shared_specificity.csv",shared_rows); _write(figures/"figure7_first_layer_mechanism.csv",mechanism_rows); _write(figures/"figure7_first_layer_energy.csv",mechanism_energy); _write(figures/"figure7_value_functional.csv",value_functional)
    try: import matplotlib.pyplot as plt
    except ImportError: return
    fig,ax=plt.subplots(figsize=(9,3.4)); ax.axis("off")
    boxes=[(.08,.58,"prefix $p_i$"),(.34,.78,"force $x_1$"),(.34,.58,"force $x_2$"),(.34,.38,"force $x_j$"),(.63,.58,"$r_{ij}$\ninteraction"),(.88,.58,"Local / Global /\nWrong-prefix")]
    for x,y,label in boxes:
        ax.text(x,y,label,ha="center",va="center",transform=ax.transAxes,bbox={"boxstyle":"round,pad=.35","facecolor":"white","edgecolor":"black"})
    for y in (.78,.58,.38): ax.annotate("",xy=(.285,y),xytext=(.15,.58),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    for y in (.78,.58,.38): ax.annotate("",xy=(.56,.58),xytext=(.41,y),xycoords="axes fraction",arrowprops={"arrowstyle":"->"})
    ax.annotate("",xy=(.80,.58),xytext=(.70,.58),xycoords="axes fraction",arrowprops={"arrowstyle":"->"}); fig.tight_layout(); fig.savefig(figures/"figure1_design.png",dpi=180,bbox_inches="tight"); plt.close(fig)
    if rank_rows:
        fig,axes=plt.subplots(1,2,figsize=(11,4)); ax=axes[0]
        energy_sorted=sorted(energy,key=lambda row:int(row["layer"]))
        ax.plot([int(row["layer"]) for row in energy_sorted],[float(row["interaction_fraction_eta"]) for row in energy_sorted],marker="o"); ax.set_xlabel("layer"); ax.set_ylabel("interaction fraction $\\eta$"); ax.set_title("Interaction energy")
        ax=axes[1]
        for layer in sorted({row["layer"] for row in rank_rows}):
            rows=[row for row in rank_rows if row["layer"]==layer]; ax.plot([row["rank"] for row in rows],[row["mean_local_ev"] for row in rows],marker="o",label=f"layer {layer}")
        ax.set_xscale("log",base=2); ax.set_xlabel("rank"); ax.set_ylabel("held-out EV"); ax.set_title("Held-out low dimensionality"); ax.legend(); fig.tight_layout(); fig.savefig(figures/"figure2_geometry.png",dpi=180); plt.close(fig)
    if primary:
        all_fields=["delta_conditional_global","delta_wrong"]; top_fields=[f"delta_conditional_global_top{top_k}",f"delta_wrong_top{top_k}"]; values=[[float(row[field]) for row in primary if np.isfinite(float(row[field]))] for field in all_fields]+[[float(row[field]) for row in pooled_top_k if np.isfinite(float(row[field]))] for field in top_fields]; fig,axes=plt.subplots(1,2,figsize=(12,4)); ax=axes[0]; ax.boxplot(values,tick_labels=["Global\nall","Wrong\nall",f"Global\nTop-{top_k}",f"Wrong\nTop-{top_k}"]); ax.axhline(0,color="black",linewidth=.8); ax.set_ylabel("Local EV minus control EV"); ax.set_title("Held-out specificity"); ax=axes[1]; rotation_fields=["R_within","R_between","d_rotation_local_conditional_global","d_rotation_local_wrong_mean"]; rotation_values=[[float(row[field]) for row in rotation if np.isfinite(float(row[field]))] for field in rotation_fields]; ax.boxplot(rotation_values,tick_labels=["within","between","Local–Global","Local–Wrong"]); ax.set_ylabel("normalized projection distance"); ax.set_title("Rotation beyond split-half noise"); fig.tight_layout(); fig.savefig(figures/"figure3_specificity.png",dpi=180); plt.close(fig)
    if functional:
        fields=["D_oracle","D_rank0","D_local","D_conditional_global","D_wrong_mean"]; fig,ax=plt.subplots(figsize=(7,4)); ax.boxplot([[float(row[field]) for row in functional] for field in fields],tick_labels=["Oracle","Rank-0","Local","cond. Global","Wrong"]); ax.set_ylabel("JS distance"); fig.tight_layout(); fig.savefig(figures/"figure4_functional.png",dpi=180); plt.close(fig)
    if saturation:
        fig,ax=plt.subplots(figsize=(6,4))
        for split in sorted({int(row["split_id"]) for row in saturation}):
            split_rows=[row for row in saturation if int(row["split_id"])==split]
            ranks=sorted({int(row["rank"]) for row in split_rows}); ax.plot(ranks,[float(np.mean([float(row["heldout_ev"]) for row in split_rows if int(row["rank"])==rank])) for rank in ranks],alpha=.45)
        ax.set_xscale("log",base=2); ax.set_xlabel("rank"); ax.set_ylabel("held-out EV"); ax.set_title("128/128 rank saturation"); fig.tight_layout(); fig.savefig(figures/"figure5_rank_saturation.png",dpi=180); plt.close(fig)
    if shared_spectrum:
        fig,ax=plt.subplots(figsize=(6,4)); directions=sorted({int(row["direction"]) for row in shared_spectrum}); ax.plot(directions,[float(np.mean([float(row["eigenvalue"]) for row in shared_spectrum if int(row["direction"])==direction])) for direction in directions]); ax.axhline(.5,color="black",linestyle="--",linewidth=.8); ax.set_xlabel("mean-projector direction"); ax.set_ylabel("eigenvalue"); ax.set_title("Shared response-space spectrum"); fig.tight_layout(); fig.savefig(figures/"figure6_shared_backbone.png",dpi=180); plt.close(fig)
    if mechanism_energy:
        fig,axes=plt.subplots(1,2,figsize=(10,4)); sites=[row["site"] for row in mechanism_energy]; axes[0].bar(sites,[float(row["interaction_fraction_eta"]) for row in mechanism_energy]); axes[0].set_ylabel("interaction fraction"); axes[0].tick_params(axis="x",rotation=20); axes[0].set_title("First-layer localization")
        if mechanism_rows:
            axes[1].boxplot([[float(row["delta_local_value_output"]) for row in mechanism_rows if row["site"]==site] for site in sites],tick_labels=sites); axes[1].axhline(0,color="black",linewidth=.8); axes[1].set_ylabel("Local EV - value/output EV"); axes[1].tick_params(axis="x",rotation=20)
        fig.tight_layout(); fig.savefig(figures/"figure7_first_layer_mechanism.png",dpi=180); plt.close(fig)


if __name__=="__main__": main()
