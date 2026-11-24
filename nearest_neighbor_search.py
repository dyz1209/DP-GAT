"""Exact Euclidean KNN and union symmetrization; no class labels are read.

The training graph excludes self-neighbors and stores both edge directions.
"""
import torch


def knn_distance_matrix(query, reference):
    """Compute Euclidean distances directly between the supplied features."""
    return torch.cdist(query, reference, compute_mode="donot_use_mm_for_euclid_dist")

def build_knn_graph(nodes, k=5, *, device="cpu"):
    if not isinstance(nodes, torch.Tensor):
        nodes = torch.as_tensor(nodes, dtype=torch.float32, device=device)
    n = nodes.shape[0]
    if k >= n:
        raise ValueError("K must be smaller than the number of graph nodes")
    rows, cols = [], []
    for start in range(0, n, 512):
        ids = torch.arange(start, min(start + 512, n), device=nodes.device)
        distances = knn_distance_matrix(nodes[ids], nodes)
        distances[torch.arange(ids.numel(), device=nodes.device), ids] = float("inf")
        _, neighbors = distances.topk(k, largest=False, sorted=True)
        rows.append(ids[:, None].expand(-1, k).reshape(-1))
        cols.append(neighbors.reshape(-1))
    rows, cols = torch.cat(rows), torch.cat(cols)
    ids = rows * n + cols
    reverse = cols * n + rows
    ids = torch.unique(torch.cat((ids, reverse)), sorted=True)
    edge_index = torch.stack((ids // n, ids % n))
    return edge_index
