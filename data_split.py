"""Non-overlapping windows, chronological purged split, and labeled subset.

Per class: train 60% / discarded gap 5% / validation 10% / discarded gap 5%
/ remaining test (~20%). Counts use Python round, with remainder assigned to
test. CSV columns represent class-specific recordings; no label is used for
KNN construction. The partition is fixed; RNG affects labeled-subset selection.
"""
import numpy as np
import torch


def blocked_train_val_test_indices(n_samples, *, train_ratio=0.6, val_ratio=0.1,
                                   gap_ratio=0.05, device="cpu"):
    """Split chronologically; assign the remainder to test after rounding."""
    train_size = int(round(n_samples * train_ratio))
    gap_size = int(round(n_samples * gap_ratio))
    val_size = int(round(n_samples * val_ratio))
    test_size = n_samples - train_size - (2 * gap_size) - val_size
    if min(train_size, val_size, test_size, gap_size) <= 0:
        raise ValueError("purged split produced an empty partition or gap")

    train_start = 0
    val_start = train_start + train_size + gap_size
    test_start = val_start + val_size + gap_size
    train_idx = torch.arange(
        train_start, train_start + train_size, device=device
    )
    val_idx = torch.arange(
        val_start, val_start + val_size, device=device
    )
    test_idx = torch.arange(
        test_start, test_start + test_size, device=device
    )
    return train_idx, val_idx, test_idx

def prepare_data_for_training(df, *, step=256, nodes_per_class=62,
                              train_ratio=0.6, val_ratio=0.1, gap_ratio=0.05,
                              preprocessing="torch", device="cpu"):
    n_classes = df.shape[1]
    n_per_class = nodes_per_class
    train_idx, val_idx, test_idx = blocked_train_val_test_indices(
        n_per_class, train_ratio=train_ratio, val_ratio=val_ratio,
        gap_ratio=gap_ratio, device=device
    )

    if preprocessing == "torch":
        blocks = []
        y_parts = []
        train_mask_parts = []
        val_mask_parts = []
        test_mask_parts = []
        for class_idx, col in enumerate(df.columns):
            values = df[col].dropna().to_numpy(dtype=np.float32)
            required_len = step * n_per_class
            if len(values) < required_len:
                raise ValueError(f"Insufficient data for class {col}: {required_len} samples required, {len(values)} available")
            samples = torch.as_tensor(values[:required_len], device=device).reshape(n_per_class, step)
            class_samples = torch.cat((
                samples[train_idx], samples[val_idx], samples[test_idx]
            ), dim=0)
            blocks.append(class_samples)
            y_parts.append(torch.full((class_samples.size(0),), class_idx, dtype=torch.long, device=device))
            train_mask_parts.append(torch.cat((
                torch.ones(train_idx.numel(), dtype=torch.bool, device=device),
                torch.zeros(val_idx.numel() + test_idx.numel(), dtype=torch.bool, device=device),
            )))
            val_mask_parts.append(torch.cat((
                torch.zeros(train_idx.numel(), dtype=torch.bool, device=device),
                torch.ones(val_idx.numel(), dtype=torch.bool, device=device),
                torch.zeros(test_idx.numel(), dtype=torch.bool, device=device),
            )))
            test_mask_parts.append(torch.cat((
                torch.zeros(train_idx.numel() + val_idx.numel(), dtype=torch.bool, device=device),
                torch.ones(test_idx.numel(), dtype=torch.bool, device=device),
            )))
        x_full = torch.cat(blocks)
        y_full = torch.cat(y_parts)
        train_mask = torch.cat(train_mask_parts)
        val_mask = torch.cat(val_mask_parts)
        test_mask = torch.cat(test_mask_parts)
        discarded = n_per_class - train_idx.numel() - val_idx.numel() - test_idx.numel()
        print(
            f"purged split per class: train={train_idx.numel()}, "
            f"validation={val_idx.numel()}, test={test_idx.numel()}, "
            f"discarded={discarded}"
        )
        return x_full, y_full, train_mask, val_mask, test_mask

    x_list = []
    y_list = []
    train_mask_list = []
    val_mask_list = []
    test_mask_list = []

    for class_idx, col in enumerate(df.columns):
        series_data = df[col].dropna().values

        # Check that this class has enough data for non-overlapping windows.
        required_len = step * n_per_class
        if len(series_data) < required_len:
            raise ValueError(f"Insufficient data for class {col}: {required_len} samples required, {len(series_data)} available")

        class_samples = []
        for i in range(n_per_class):
            start = i * step
            sample = series_data[start:start + step]
            class_samples.append(sample)

        class_samples = np.asarray(class_samples, dtype=np.float32)
        kept_samples = np.concatenate((
            class_samples[train_idx.cpu().numpy()],
            class_samples[val_idx.cpu().numpy()],
            class_samples[test_idx.cpu().numpy()],
        ))
        x_list.extend(kept_samples)
        y_list.extend([class_idx] * len(kept_samples))
        train_mask_list.extend(
            [True] * train_idx.numel()
            + [False] * (val_idx.numel() + test_idx.numel())
        )
        val_mask_list.extend(
            [False] * train_idx.numel()
            + [True] * val_idx.numel()
            + [False] * test_idx.numel()
        )
        test_mask_list.extend(
            [False] * (train_idx.numel() + val_idx.numel())
            + [True] * test_idx.numel()
        )

    x_full = torch.tensor(np.array(x_list), dtype=torch.float32)
    y_full = torch.tensor(y_list, dtype=torch.long)
    train_mask = torch.tensor(train_mask_list, dtype=torch.bool)
    val_mask = torch.tensor(val_mask_list, dtype=torch.bool)
    test_mask = torch.tensor(test_mask_list, dtype=torch.bool)

    discarded = n_per_class - train_idx.numel() - val_idx.numel() - test_idx.numel()
    print(
        f"purged split per class: train={train_idx.numel()}, "
        f"validation={val_idx.numel()}, test={test_idx.numel()}, "
        f"discarded={discarded}"
    )
    return x_full, y_full, train_mask, val_mask, test_mask

def build_labeled_training_mask(
        n_train_nodes, n_classes, label_rate=0.2, eligible_mask=None, *, device="cpu"
):
    """Select a random, class-balanced labeled subset.

    For each class, round training_count * label_rate and select at least one
    labeled node. If supplied, eligible_mask restricts the selectable nodes.
    """
    if not 0 < label_rate <= 1:
        raise ValueError("label_rate must be in the interval (0, 1]")
    if n_train_nodes <= 0 or n_train_nodes % n_classes != 0:
        raise ValueError("training nodes must form equal-sized class blocks")
    if eligible_mask is None:
        eligible_mask = torch.ones(n_train_nodes, dtype=torch.bool, device=device)
    else:
        eligible_mask = eligible_mask.to(device=device, dtype=torch.bool)
        if eligible_mask.numel() != n_train_nodes:
            raise ValueError("eligible_mask must match n_train_nodes")

    nodes_per_class = n_train_nodes // n_classes
    labeled_mask = torch.zeros(n_train_nodes, dtype=torch.bool, device=device)
    for class_id in range(n_classes):
        class_idx = torch.arange(
            class_id * nodes_per_class,
            (class_id + 1) * nodes_per_class,
            device=device,
        )
        eligible_class_idx = class_idx[eligible_mask[class_idx]]
        count = max(1, int(round(nodes_per_class * label_rate)))
        if count > eligible_class_idx.numel():
            raise ValueError(
                "requested labeled count exceeds the eligible training nodes"
            )
        permutation = torch.randperm(eligible_class_idx.numel(), device=device)
        labeled_mask[eligible_class_idx[permutation[:count]]] = True
    return labeled_mask
