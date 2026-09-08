#!/usr/bin/env python3
"""Export the selected clustering instance and block distinctiveness to text."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "weighted_blocked_pca_kmeans_k24_58_42_balanced20pct_recombined"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--top-blocks", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir
    config = json.loads((input_dir / "run_config.json").read_text(encoding="utf-8"))
    manifest = json.loads(
        (input_dir / "hierarchy_weight_manifest.json").read_text(encoding="utf-8")
    )
    metrics = pd.read_csv(input_dir / "k_stability_metrics.csv")
    assignments = pd.read_csv(input_dir / "selected_assignments.csv").sort_values("row_index")
    embedding = np.load(input_dir / "weighted_embedding.npy")
    centroids = np.load(input_dir / "selected_centroids.npy")

    selected_k = int(config["selected_k"])
    selected_metric = metrics.loc[metrics["k"] == selected_k].iloc[0]
    if len(assignments) != len(embedding):
        raise ValueError("Assignment and embedding row counts do not match.")
    if not np.array_equal(assignments["row_index"].to_numpy(), np.arange(len(assignments))):
        raise ValueError("Assignments are not aligned to embedding row indexes.")
    if centroids.shape != (selected_k, embedding.shape[1]):
        raise ValueError("Centroid shape does not match selected k and embedding dimensions.")

    block_indexes: dict[str, list[int]] = defaultdict(list)
    block_weights: dict[str, float] = {}
    block_domains: dict[str, str] = {}
    for subgroup in manifest["subgroups"]:
        name = subgroup["effective_block"]
        block_indexes[name].extend(range(subgroup["slice_start"], subgroup["slice_stop"]))
        block_weights[name] = float(subgroup["effective_block_weight"])
        block_domains[name] = subgroup["domain"]

    cluster_block_rows: list[dict[str, float | int | str]] = []
    for cluster_raw in range(selected_k):
        center = centroids[cluster_raw]
        energies = {
            name: float(np.square(center[indexes]).sum())
            for name, indexes in block_indexes.items()
        }
        total_energy = sum(energies.values())
        for name, energy in energies.items():
            cluster_block_rows.append(
                {
                    "cluster": cluster_raw + 1,
                    "block": name,
                    "domain": block_domains[name],
                    "centroid_energy": energy,
                    "allocated_weight": block_weights[name],
                    "normalized_distinctiveness": energy / block_weights[name],
                    "weighted_signal_share": energy / total_energy if total_energy else 0.0,
                }
            )
    cluster_blocks = pd.DataFrame(cluster_block_rows)

    cluster_sizes = assignments["cluster"].value_counts().sort_index()
    global_rows = []
    for block, rows in cluster_blocks.groupby("block", sort=False):
        weights = rows["cluster"].map(cluster_sizes).to_numpy(dtype=float)
        global_rows.append(
            {
                "block": block,
                "domain": rows["domain"].iloc[0],
                "between_cluster_distinctiveness": float(
                    np.average(rows["normalized_distinctiveness"], weights=weights)
                ),
            }
        )
    global_blocks = pd.DataFrame(global_rows).sort_values(
        ["between_cluster_distinctiveness", "block"], ascending=[False, True]
    )

    output = args.output or input_dir / f"best_clustering_instance_k{selected_k}.txt"
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"BEST WEIGHTED BLOCKED-PCA K-MEANS++ CLUSTERING: k={selected_k}",
        "=" * 78,
        "",
        f"Player-seasons: {len(assignments)}",
        f"Embedding dimensions: {embedding.shape[1]}",
        f"Clustering domain: {config.get('domain', 'All')}",
        f"Mean AMI: {selected_metric['mean_ami']:.6f}",
        f"Mean ARI: {selected_metric['mean_ari']:.6f}",
        f"AMI/ARI selection score: {selected_metric['selection_score']:.6f}",
        f"Cluster-size target: {len(assignments) / selected_k:.2f}",
        f"Allowed cluster-size range: {config.get('balanced_min_cluster_size', 'n/a')}–"
        f"{config.get('balanced_max_cluster_size', 'n/a')}",
        "Guarding compositions: raw normalized shares; no Minutes/prior shrinkage",
        "",
        "DISTINCTIVENESS DEFINITION",
        "-" * 78,
        "Centroid energy is the squared distance from the league-wide origin contributed",
        "by an effective mega block. Normalized distinctiveness divides that energy by",
        "the block's allocated variance weight, preventing larger weights from winning",
        "automatically. Weighted signal share states the block's share of the cluster",
        "centroid's actual model-space squared distance.",
        "",
        "GLOBAL BLOCK DISTINCTIVENESS",
        "-" * 78,
    ]
    for rank, row in enumerate(global_blocks.itertuples(index=False), start=1):
        lines.append(
            f"{rank:>2}. {row.block} [{row.domain}] — normalized score "
            f"{row.between_cluster_distinctiveness:.4f}"
        )

    for cluster in range(1, selected_k + 1):
        cluster_members = assignments.loc[assignments["cluster"] == cluster].copy()
        center = centroids[cluster - 1]
        member_indexes = cluster_members["row_index"].to_numpy(dtype=int)
        cluster_members["distance_to_centroid"] = np.linalg.norm(
            embedding[member_indexes] - center, axis=1
        )
        cluster_members = cluster_members.sort_values(
            ["distance_to_centroid", "player", "season"]
        )
        ranked_blocks = cluster_blocks.loc[cluster_blocks["cluster"] == cluster].sort_values(
            ["normalized_distinctiveness", "block"], ascending=[False, True]
        )
        lines.extend(
            [
                "",
                "=" * 78,
                f"CLUSTER {cluster} — {len(cluster_members)} PLAYER-SEASONS",
                "=" * 78,
                "Most distinctive blocks:",
            ]
        )
        for rank, row in enumerate(
            ranked_blocks.head(args.top_blocks).itertuples(index=False), start=1
        ):
            lines.append(
                f"  {rank}. {row.block} [{row.domain}] — normalized "
                f"{row.normalized_distinctiveness:.4f}; weighted signal share "
                f"{100.0 * row.weighted_signal_share:.2f}%"
            )
        lines.extend(["", "Members (centroid-nearest first):"])
        for row in cluster_members.itertuples(index=False):
            lines.append(
                f"  - {row.player} — {row.season} ({row.position}); "
                f"centroid distance {row.distance_to_centroid:.4f}"
            )

    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    cluster_blocks.to_csv(input_dir / "cluster_block_distinctiveness.csv", index=False)
    global_blocks.to_csv(input_dir / "global_block_distinctiveness.csv", index=False)
    print(output)


if __name__ == "__main__":
    main()
