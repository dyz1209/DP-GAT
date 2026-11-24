"""Prepare clean-data FFT features, candidate graphs and split files."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fft_preprocessing import (
    build_fft_features, fit_feature_standardizer,
    apply_feature_standardizer,
)
from adaptive_pooling import apply_adaptive_average_pooling
from data_split import (
    blocked_train_val_test_indices, prepare_data_for_training,
    build_labeled_training_mask,
)
from graph_construction import build_inductive_feature_knn_graph


HERE = Path(__file__).resolve().parent
DATASETS = {
    "A": ("dataset_A.csv", 1024, 1023),
    "B": ("dataset_B.csv", 1024, 1023),
    "C": ("dataset_C.csv", 256, 62),
    "D": ("dataset_D.csv", 1024, 200),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=DATASETS, default="A")
    parser.add_argument("--csv", type=Path, help="override the dataset CSV path")
    parser.add_argument("--step", type=int, help="override raw window length")
    parser.add_argument("--nodes-per-class", type=int, help="override windows per class before gaps")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pooling-dim", type=int, default=64)
    parser.add_argument("--label-rate", type=float, default=0.2)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--test-anchor-k", type=int, default=5)
    parser.add_argument("--std-floor", type=float, default=1e-6)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    filename, preset_step, preset_nodes = DATASETS[args.dataset]
    step = preset_step if args.step is None else args.step
    nodes = preset_nodes if args.nodes_per_class is None else args.nodes_per_class
    csv_path = (HERE.parent / filename) if args.csv is None else args.csv
    output_dir = (HERE / "outputs" / f"{args.dataset}_seed_{args.seed}"
                  if args.output_dir is None else args.output_dir)
    if (step < 4 or nodes < 1 or not 1 <= args.pooling_dim <= step // 2
            or args.k < 1 or args.test_anchor_k < 0 or args.cpu_threads < 1
            or not 0 < args.label_rate <= 1 or not 0 <= args.seed < 2 ** 32):
        parser.error("invalid window/count/pooling/neighbor/thread/label-rate/seed setting")
    if not np.isfinite(args.std_floor) or args.std_floor <= 0:
        parser.error("standard deviation floors must be positive and finite")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    df = pd.read_csv(csv_path)
    raw, labels, train_mask, val_mask, test_mask = prepare_data_for_training(
        df, step=step, nodes_per_class=nodes,
    )
    spectrum, _ = build_fft_features(raw)
    model_stats = fit_feature_standardizer(spectrum[train_mask], args.std_floor)
    features = apply_adaptive_average_pooling(
        apply_feature_standardizer(spectrum, model_stats), args.pooling_dim,
    )
    split_masks = {"train": train_mask, "validation": val_mask, "test": test_mask}
    split_features = {name: features[mask] for name, mask in split_masks.items()}
    labeled = build_labeled_training_mask(
        int(train_mask.sum()), df.shape[1], args.label_rate,
    )
    train_graph = build_inductive_feature_knn_graph(
        split_features["train"], train_k=args.k,
    )
    # Export each heldout graph separately for later evaluation. This script does
    # not train; these graphs must not be passed to the training optimizer.
    heldout_graphs = {
        name: build_inductive_feature_knn_graph(
            split_features["train"], split_features[name], train_k=args.k,
            test_anchor_k=args.test_anchor_k, test_test_k=0,
        )
        for name in ("validation", "test")
    }
    local_indices = blocked_train_val_test_indices(nodes)
    indices_by_split = dict(zip(("train", "validation", "test"), local_indices))
    metadata = {
        "dataset": args.dataset, "seed": args.seed,
        "step": step, "nodes_per_class_before_gaps": nodes,
        "pooling_dim": args.pooling_dim, "label_rate": args.label_rate,
        "k": args.k, "test_anchor_k": args.test_anchor_k, "test_test_k": 0,
        "std_floor": args.std_floor,
        "pipeline": "fft_standardization_pooling_euclidean_knn_purged_split",
        "split_ratios": {"train": 0.6, "gap_each": 0.05, "validation": 0.1, "test": "remainder"},
        "counts_per_class": {name: idx.numel() for name, idx in indices_by_split.items()},
        "discarded_per_class": nodes - sum(idx.numel() for idx in local_indices),
        "class_names": list(map(str, df.columns)),
        "noise": "none", "protocol": "inductive_purged_60_5_10_5_20",
        "edge_direction": "source_to_target; heldout links train_to_query_only",
        "index_convention": "zero-based window indices within each class recording",
        "label_usage": "CSV column identifies class for splitting/loss metadata; graph functions receive features only",
        "labeled_train_nodes": int(labeled.sum()),
        "torch_version": str(torch.__version__),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "features": split_features,
        "labels": {name: labels[mask] for name, mask in split_masks.items()},
        "split_masks_in_retained_class_major_order": split_masks,
        "labeled_train_mask": labeled,
        "window_indices_per_class": indices_by_split,
        "standardizer": model_stats,
        "train_edge_index": train_graph,
        "validation_edge_index": heldout_graphs["validation"],
        "test_edge_index": heldout_graphs["test"], "metadata": metadata,
    }, output_dir / "prepared_inputs.pt")
    (output_dir / "split_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(json.dumps(metadata["counts_per_class"], ensure_ascii=False))
    print(f"Saved inputs and split metadata: {output_dir}")


if __name__ == "__main__":
    main()
