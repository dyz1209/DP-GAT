"""Label-free training graph, heldout attachment, and GCN edge weights.

edge_index[0] = source; edge_index[1] = target (PyG convention).
Training KNN uses union symmetrization, not mutual-KNN. Heldout graphs add
training -> query links only; query -> training and query-query are absent
by default. Self-loops are added later by GCN/GAT, not in this candidate pool.
"""
import torch
from nearest_neighbor_search import build_knn_graph, knn_distance_matrix


def _deduplicate_edges(edge_index, n_nodes):
    if edge_index.numel() == 0:
        return edge_index
    ids = edge_index[0] * n_nodes + edge_index[1]
    ids = torch.unique(ids, sorted=True)
    return torch.stack((ids // n_nodes, ids % n_nodes), dim=0)

@torch.no_grad()
def build_inductive_feature_knn_graph(
        x_train, x_test=None, train_k=5, test_anchor_k=5,
        test_test_k=0, *, device="cpu"):
    """Build a label-free graph for inductive semi-supervised evaluation.

    Build the training KNN subgraph from training features only. When query
    features are supplied through x_test, attach them to training anchors.
    With test_test_k=0, all additional edges run from train to query.
    This function neither trains a model nor reads labels.
    """
    x_train = x_train.to(device)
    x_test = None if x_test is None else x_test.to(device)
    n_train = x_train.size(0)
    n_test = 0 if x_test is None else x_test.size(0)
    n_nodes = n_train + n_test
    edge_parts = []

    if n_train > 1:
        local_train_k = min(train_k, n_train - 1)
        edge_parts.append(build_knn_graph(x_train, k=local_train_k, device=device))

    if n_test > 0:
        local_anchor_k = min(test_anchor_k, n_train)
        if local_anchor_k > 0:
            distances = knn_distance_matrix(x_test, x_train)
            _, nearest = distances.topk(local_anchor_k, largest=False, sorted=True)
            test_nodes = torch.arange(n_train, n_nodes, device=device)
            # PyG convention: edge_index[0] is the source and edge_index[1]
            # is the target. Test nodes receive messages from train anchors;
            # no test-to-train edge is created.
            src = nearest.reshape(-1)
            dst = test_nodes[:, None].expand(-1, local_anchor_k).reshape(-1)
            edge_parts.append(torch.stack((src, dst), dim=0))

        if test_test_k > 0 and n_test > 1:
            local_test_k = min(test_test_k, n_test - 1)
            test_edges = build_knn_graph(x_test, k=local_test_k, device=device) + n_train
            edge_parts.append(test_edges)

    if not edge_parts:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return _deduplicate_edges(torch.cat(edge_parts, dim=1).to(device), n_nodes)

@torch.no_grad()
def build_transductive_feature_knn_graph(x, k=5, *, device="cpu"):
    """Build a standard label-free KNN graph over all available nodes."""
    x = x.to(device)
    if x.size(0) <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return build_knn_graph(x, k=min(k, x.size(0) - 1), device=device)

def compute_rbf_edge_weights(x, edge_index, eps=1e-12, *, sigma_scale=2.0):
    """RBF bandwidth: sigma_scale times the median positive edge distance."""
    if edge_index.numel() == 0:
        return x.new_empty((0,))
    src, dst = edge_index
    distances = torch.linalg.vector_norm(x[src] - x[dst], dim=1)
    positive = distances[distances > eps]
    if positive.numel() == 0:
        sigma = distances.new_tensor(1.0)
    else:
        sigma = (positive.median() * sigma_scale).clamp_min(eps)
    return torch.exp(-(distances ** 2) / (2.0 * sigma ** 2))

def compute_local_edge_weights(x, edge_index, eps=1e-12, *, sigma_scale=2.0):
    """Local Gaussian weights using the existing graph's incoming neighbors.

    Each node's bandwidth is the farthest positive incoming-edge distance,
    multiplied by sigma_scale. For symmetrized training KNN this is the
    farthest retained neighbor, not necessarily the original k-th neighbor.
    Query nodes use incoming training anchors (and optional query neighbors).
    Training bandwidths do not depend on outgoing train-to-query edges.
    Isolated nodes or nodes with only zero-distance neighbors use bandwidth 1.
    No labels, clean query features, or noise/SNR flags are consulted.
    """
    if edge_index.numel() == 0:
        return x.new_empty((0,))
    src, dst = edge_index
    distances = torch.linalg.vector_norm(x[src] - x[dst], dim=1)
    local_radius = x.new_zeros(x.size(0))
    local_radius.scatter_reduce_(0, dst, distances, reduce="amax", include_self=True)
    sigma = torch.where(
        local_radius > eps,
        (local_radius * sigma_scale).clamp_min(eps),
        torch.ones_like(local_radius),
    )
    return torch.exp(-(distances ** 2) / (2.0 * sigma[src] * sigma[dst]))

def compute_edge_weights(x, edge_index, eps=1e-12, *, mode="rbf", sigma_scale=2.0):
    """Select the fixed GCN's edge-weight rule; KNN topology is unchanged."""
    if mode == "rbf":
        return compute_rbf_edge_weights(x, edge_index, eps=eps, sigma_scale=sigma_scale)
    if mode == "local":
        return compute_local_edge_weights(x, edge_index, eps=eps, sigma_scale=sigma_scale)
    raise ValueError(f"Unsupported edge-weight mode: {mode}")
