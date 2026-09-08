#!/usr/bin/env python3
"""Run the isolated weighted blocked-PCA KMeans++ experiment."""

from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.weighted_blocked_pca import (  # noqa: E402
    DEFAULT_EFFECTIVE_BLOCKS,
    build_weighted_subgroups,
    engineer_notebook_features,
    evaluate_k_stability,
    file_sha256,
    fit_size_balanced_kmeans,
    fit_weighted_embedding,
    parse_hierarchy,
    restrict_weighted_subgroups_to_domain,
    save_model_bundle,
    select_k,
)


DEFAULT_DATASET = Path(
    "/Users/harsha/Desktop/PickPocketProjectOfficial/bballindex_complete_dataset.csv"
)
DEFAULT_HIERARCHY = Path("/Users/harsha/Downloads/hiarchy.txt")
DEFAULT_OUTPUT = (
    PROJECT_ROOT
    / "outputs"
    / "weighted_blocked_pca_kmeans_k24_58_42_balanced20pct_recombined"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Cluster player-seasons with exact weighted subgroup PCA whitening and "
            "select k by resampled AMI/ARI stability."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--hierarchy", type=Path, default=DEFAULT_HIERARCHY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--domain",
        choices=("all", "offense", "defense"),
        default="all",
        help="Cluster with all blocks or one independently renormalized domain.",
    )
    parser.add_argument("--k-min", type=int, default=24)
    parser.add_argument("--k-max", type=int, default=24)
    parser.add_argument("--resamples", type=int, default=100)
    parser.add_argument("--subsample-fraction", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--stability-n-init", type=int, default=50)
    parser.add_argument("--final-n-init", type=int, default=200)
    parser.add_argument(
        "--balance-tolerance",
        type=float,
        default=0.20,
        help="Maximum cluster-size deviation from n/k (0.20 means +/-20%%).",
    )
    parser.add_argument("--balanced-max-iter", type=int, default=100)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument(
        "--expected-row-count",
        type=int,
        default=4162,
        help="Set to 0 to disable the dataset snapshot row-count check.",
    )
    return parser.parse_args()


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.k_min < 2 or args.k_max < args.k_min:
        raise ValueError("Require 2 <= k-min <= k-max.")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    raw = pd.read_csv(args.dataset, low_memory=False).reset_index(drop=True)
    if args.expected_row_count and len(raw) != args.expected_row_count:
        raise ValueError(
            f"Expected {args.expected_row_count} player-seasons, found {len(raw)}. "
            "Pass --expected-row-count 0 to accept a changed snapshot."
        )
    required_identity = {"player", "season"}
    if not required_identity.issubset(raw.columns):
        raise KeyError(f"Dataset lacks identity columns: {sorted(required_identity - set(raw.columns))}")
    duplicate_identity = raw.duplicated(["player", "season"], keep=False)
    if duplicate_identity.any():
        examples = raw.loc[duplicate_identity, ["player", "season"]].head(20)
        raise ValueError(f"Duplicate player-season rows: {examples.to_dict('records')}")

    print(f"Parsing hierarchy: {args.hierarchy}")
    hierarchy = parse_hierarchy(args.hierarchy)
    if hierarchy.ignored_feature_like_lines:
        print("WARNING: ignored malformed feature-like hierarchy lines:")
        for row in hierarchy.ignored_feature_like_lines:
            print(f"  line {row['line_number']}: {row['text']!r}")
    weighted_subgroups = build_weighted_subgroups(hierarchy)
    active_domain = None if args.domain == "all" else args.domain.title()
    if active_domain is not None:
        weighted_subgroups = restrict_weighted_subgroups_to_domain(
            weighted_subgroups, active_domain
        )

    print("Engineering raw guarding compositions and AST/TOV features")
    data, engineering_manifest = engineer_notebook_features(raw)
    missing = sorted(set(hierarchy.features) - set(data.columns))
    if missing:
        raise KeyError(f"Hierarchy features unavailable after engineering: {missing}")

    print("Fitting per-subgroup full-variance PCA and exact whitening weights")
    embedding = fit_weighted_embedding(data, weighted_subgroups, random_state=args.seed)
    np.save(args.output_dir / "weighted_embedding.npy", embedding.matrix)
    embedding.contribution_audit.to_csv(
        args.output_dir / "variance_contribution_audit.csv", index=False
    )
    pd.DataFrame(embedding.standardization_manifest).to_csv(
        args.output_dir / "within_season_standardization.csv.gz",
        index=False,
        compression="gzip",
    )
    pd.DataFrame(engineering_manifest["composition_audit"]).to_csv(
        args.output_dir / "guarding_composition_audit.csv", index=False
    )

    row_index = raw[["player", "season"]].copy()
    row_index.insert(0, "row_index", np.arange(len(raw), dtype=int))
    if "position" in raw.columns:
        row_index["position"] = raw["position"].astype(str)
    row_index.to_csv(args.output_dir / "row_index.csv", index=False)

    active_block_names = {row.effective_block for row in weighted_subgroups}
    active_features = {
        feature for row in weighted_subgroups for feature in row.features
    }
    domain_scale = 1.0 if active_domain is None else sum(
        spec.final_weight for spec in DEFAULT_EFFECTIVE_BLOCKS if spec.domain == active_domain
    )
    hierarchy_manifest = {
        "source_mega_block_count": len(hierarchy.blocks),
        "subgroup_count": len(weighted_subgroups),
        "feature_count": len(active_features),
        "full_hierarchy_feature_count": len(hierarchy.features),
        "active_domain": active_domain or "All",
        "ignored_feature_like_lines": list(hierarchy.ignored_feature_like_lines),
        "effective_blocks": [
            {
                "name": spec.name,
                "domain": spec.domain,
                "requested_domain_weight": spec.requested_domain_weight,
                "final_weight": spec.final_weight / domain_scale,
                "source_blocks": list(spec.source_blocks),
                "selected_subgroups": (
                    None if spec.selected_subgroups is None else list(spec.selected_subgroups)
                ),
            }
            for spec in DEFAULT_EFFECTIVE_BLOCKS
            if spec.name in active_block_names
        ],
        "subgroups": embedding.subgroup_manifest,
    }
    write_json(args.output_dir / "hierarchy_weight_manifest.json", hierarchy_manifest)

    print(
        f"Evaluating k={args.k_min}..{args.k_max} with "
        f"{args.resamples} season-stratified resamples each"
    )
    metrics, pairwise, candidates = evaluate_k_stability(
        embedding.matrix,
        raw["season"].to_numpy(),
        k_values=tuple(range(args.k_min, args.k_max + 1)),
        resamples=args.resamples,
        fraction=args.subsample_fraction,
        seed=args.seed,
        n_init=args.stability_n_init,
        max_iter=args.max_iter,
        tolerance=args.tol,
        jobs=args.jobs,
        balance_tolerance=args.balance_tolerance,
    )
    selected_k = select_k(metrics)
    metrics["selected"] = metrics["k"].eq(selected_k)
    metrics.to_csv(args.output_dir / "k_stability_metrics.csv", index=False)
    pairwise.to_csv(
        args.output_dir / "ami_ari_pairwise.csv.gz", index=False, compression="gzip"
    )
    for k, candidate in candidates.items():
        np.savez_compressed(
            args.output_dir / f"candidate_assignments_k{k}.npz", **candidate
        )

    print(f"Selected k={selected_k} by mean AMI/ARI score")
    final_model = fit_size_balanced_kmeans(
        embedding.matrix,
        k=selected_k,
        seed=args.seed,
        n_init=args.final_n_init,
        max_iter=args.balanced_max_iter,
        tolerance=args.tol,
        balance_tolerance=args.balance_tolerance,
    )
    labels = final_model.labels_.astype(int)
    assignments = row_index.copy()
    assignments["cluster_raw"] = labels
    assignments["cluster"] = labels + 1
    assignments.to_csv(args.output_dir / "selected_assignments.csv", index=False)
    np.save(args.output_dir / "selected_centroids.npy", final_model.cluster_centers_)
    cluster_sizes = (
        assignments.groupby(["cluster_raw", "cluster"], as_index=False)
        .size()
        .rename(columns={"size": "player_seasons"})
    )
    cluster_sizes.to_csv(args.output_dir / "selected_cluster_sizes.csv", index=False)

    run_config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "dataset": str(args.dataset.resolve()),
        "dataset_sha256": file_sha256(args.dataset),
        "hierarchy": str(args.hierarchy.resolve()),
        "hierarchy_sha256": file_sha256(args.hierarchy),
        "output_dir": str(args.output_dir.resolve()),
        "rows": len(raw),
        "embedding_dimensions": int(embedding.matrix.shape[1]),
        "k_min": args.k_min,
        "k_max": args.k_max,
        "selected_k": selected_k,
        "selection_rule": "max((mean_AMI + mean_ARI) / 2), exact ties use smaller k",
        "resamples": args.resamples,
        "subsample_fraction": args.subsample_fraction,
        "seed": args.seed,
        "stability_n_init": args.stability_n_init,
        "final_n_init": args.final_n_init,
        "balance_tolerance": args.balance_tolerance,
        "balanced_max_iter": args.balanced_max_iter,
        "balanced_iterations": final_model.n_iter_,
        "balanced_min_cluster_size": final_model.min_cluster_size,
        "balanced_max_cluster_size": final_model.max_cluster_size,
        "unbalanced_inertia": final_model.base_inertia_,
        "balanced_inertia": final_model.inertia_,
        "balance_inertia_increase_fraction": (
            final_model.inertia_ / final_model.base_inertia_ - 1.0
        ),
        "max_iter": args.max_iter,
        "tolerance": args.tol,
        "jobs": args.jobs,
        "domain": active_domain or "All",
    }
    write_json(args.output_dir / "run_config.json", run_config)
    model_bundle = {
        **embedding.model_bundle,
        "feature_engineering_manifest": engineering_manifest,
        "hierarchy_manifest": hierarchy_manifest,
        "run_config": run_config,
        "kmeans": final_model,
    }
    save_model_bundle(model_bundle, args.output_dir / "selected_model.joblib")
    print(metrics.to_string(index=False))
    print(f"Artifacts written to: {args.output_dir}")


if __name__ == "__main__":
    main()
