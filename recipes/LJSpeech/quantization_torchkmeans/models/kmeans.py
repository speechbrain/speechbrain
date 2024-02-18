"""K-means clustering.

Authors
 * Luca Della Libera 2024
"""

# Adapted from:
# https://github.com/jokofa/torch_kmeans/tree/be7d2b78664e81a985ddfa6d21d94917a8b49fe6

import logging
from typing import List, Optional, Tuple, Union

import torch
from torch import Tensor, nn


__all__ = ["KMeans", "MultiKMeans"]


_LOGGER = logging.getLogger(__file__)


class KMeans(nn.Module):
    """K-means clustering.

    Parameters
    ----------
    num_features:
        The number of features.
    num_clusters:
        The number of clusters.
    init:
        Method to initialize cluster centroids. One of ["random"].
    normalize:
        Method to use to normalize input. One of [None, "mean", "minmax", "unit"].
    p_norm:
        The order of norm/distance.

    Examples
    --------
    >>> import torch
    >>>
    >>> batch_size = 8
    >>> seq_length = 200
    >>> num_features = 64
    >>> num_clusters = 4
    >>> kmeans = KMeans(num_features, num_clusters)
    >>> input = torch.randn(batch_size, seq_length, num_features)
    >>> labels, centroids = kmeans(input)
    >>> drift = kmeans.step(input, labels)

    """

    _INIT_METHODS = ["random"]
    _NORM_METHODS = ["mean", "minmax", "unit"]

    def __init__(
        self,
        num_features: "int",
        num_clusters: "int",
        init: "str" = "random",
        normalize: "Optional[Union[str, bool]]" = None,
        p_norm: "float" = 2.0,
    ) -> "None":
        super().__init__()
        self.num_features = num_features
        self.num_clusters = num_clusters
        self.init = init.lower()
        self.normalize = normalize
        self.p_norm = p_norm
        self._check_params()

        # Register centroids as a buffer, with "inf" indicating uninitialized centroids
        self.register_buffer(
            "centroids",
            torch.full((self.num_clusters, self.num_features), float("inf")),
        )

    def reset_parameters(self, x: "Tensor") -> "None":
        x = x.reshape(-1, self.num_features)
        if x.shape[0] >= self.num_clusters:
            self.centroids = _init_centroids(x, self.num_clusters, self.init)
            return
        _LOGGER.warning(
            "The first batch contains less samples than centroids, skipping initialization..."
        )

    def forward(
        self, x: "Tensor", return_centroids: "bool" = True
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        """Forward pass.

        Parameters
        ----------
        x:
            The input features, shape (*batch_shape, num_features)
        return_centroids:
            True to return the assigned centroids, False otherwise.

        Returns
        -------
            - The cluster assignments, shape (*batch_shape,).
            - If `return_centroids=True`, the corresponding centroids.

        """
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.num_features)

        if self.centroids[0, 0].isinf():
            # Initialize centroids
            self.reset_parameters(x)

        if self.normalize is not None:
            x = _normalize(x, self.normalize)

        dist = _compute_pairwise_distance(x, self.centroids, self.p_norm)

        # Get cluster assignments (centroid with minimal distance)
        labels = dist.argmin(dim=-1)
        if return_centroids:
            assigned_centroids = self.centroids.gather(
                0, labels[:, None].expand(-1, self.num_features)
            ).clone()
            labels = labels.reshape(batch_shape)
            assigned_centroids = assigned_centroids.reshape(
                *batch_shape, self.num_features
            )
            return labels, assigned_centroids

        labels = labels.reshape(batch_shape)
        return labels, None

    def step(
        self,
        x: "Tensor",
        labels: "Optional[Tensor]" = None,
        return_drift: "bool" = True,
    ) -> "Optional[Tensor]":
        """"Lloyd K-means update.

        Parameters
        ----------
        x:
            The input features, shape (*batch_shape, num_features)
        labels:
            The corresponding labels, shape (*batch_shape,).
        return_drift:
            True to return the drift between current and previous centroids, False otherwise.

        Returns
        -------
            If `return_drift=True`, the drift between current and previous centroids.

        """
        x = x.reshape(-1, self.num_features)

        if x.shape[0] < self.num_clusters:
            _LOGGER.warning(
                f"Number of samples ({x.shape[0]}) is less than the number "
                f"of clusters ({self.num_clusters}), skipping this batch",
            )
            return torch.zeros(1, device=x.device) if return_drift else None

        if labels is None:
            labels, _ = self.forward(x, return_centroids=False)
        labels = labels.flatten()

        # Update cluster centroids
        old_centroids = self.centroids.clone()
        self.centroids = _group_by_label_mean(x, labels, self.num_clusters)

        if return_drift:
            # Compute centroid drift
            drift = _compute_drift(self.centroids, old_centroids, p=self.p_norm)
            return drift

    def evaluate(
        self, x: "Tensor", labels: "Optional[Tensor]" = None
    ) -> "Tensor":
        """Compute inertia for the current batch."""
        batch_shape = x.shape[:-1]
        x = x.reshape(-1, self.num_features)
        if labels is None:
            labels, _ = self.forward(x, return_centroids=False)
        inertia = _compute_inertia(x, self.centroids, labels, p=self.p_norm)
        inertia = inertia.reshape(batch_shape)
        return inertia

    def _check_params(self):
        """Check initialization parameters."""
        if self.num_features < 1:
            raise ValueError(
                f"<num_features> should be > 0, but got {self.num_features}."
            )
        if self.num_clusters < 2:
            raise ValueError(
                f"<num_clusters> should be > 1, but got {self.num_clusters}."
            )
        if self.init not in self._INIT_METHODS:
            raise ValueError(
                f"unknown <init>: {self.init}. "
                f"Please choose one of {self._INIT_METHODS}"
            )
        if isinstance(self.normalize, bool):
            if self.normalize:
                self.normalize = "mean"
            else:
                self.normalize = None
        if (
            self.normalize is not None
            and self.normalize not in self._NORM_METHODS
        ):
            raise ValueError(
                f"unknown <normalize> method: {self.normalize}. "
                f"Please choose one of {self._NORM_METHODS}"
            )

    def __repr__(self):
        return (
            f"{type(self).__name__}("
            f"num_features: {self.num_features}, "
            f"num_clusters: {self.num_clusters}, "
            f"init: {self.init}, "
            f"normalize: {self.normalize}, "
            f"p_norm: {self.p_norm})"
        )


class MultiKMeans(nn.Module):
    """K-means clustering with multiple instances, each one with its own args and kwargs."""

    def __init__(self, *args, **kwargs) -> "None":
        super().__init__()
        max_length = max(
            len(v)
            for v in args + tuple(kwargs.values())
            if isinstance(v, (list, tuple))
        )
        args = [
            v if isinstance(v, (list, tuple)) else [v] * max_length
            for v in args
        ]
        kwargs = {
            k: v if isinstance(v, (list, tuple)) else [v] * max_length
            for k, v in kwargs.items()
        }
        all_args = list(zip(*args))
        all_kwargs_values = list(zip(*kwargs.values()))
        all_kwargs = [
            dict(zip(kwargs.keys(), values)) for values in all_kwargs_values
        ]
        if not all_args:
            all_args = [[] for _ in range(len(all_kwargs))]
        if not all_kwargs:
            all_kwargs = [{} for _ in range(len(all_args))]
        assert len(all_args) == len(all_kwargs)

        kmeanss = [
            KMeans(*args, **kwargs)
            for args, kwargs in zip(all_args, all_kwargs)
        ]
        self.kmeanss = nn.ModuleList(kmeanss)

    @property
    def num_features(self) -> "List[int]":
        return [kmeans.num_features for kmeans in self.kmeanss]

    @property
    def num_clusters(self) -> "List[int]":
        return [kmeans.num_clusters for kmeans in self.kmeanss]

    @property
    def init(self) -> "List[str]":
        return [kmeans.init for kmeans in self.kmeanss]

    @property
    def normalize(self) -> "List[str]":
        return [kmeans.normalize for kmeans in self.kmeanss]

    @property
    def p_norm(self) -> "List[float]":
        return [kmeans.p_norm for kmeans in self.kmeanss]

    @property
    def centroids(self) -> "Tensor":
        if len(self.kmeanss) == 1:
            # Fast path
            return self.kmeanss[0].centroids[..., None]
        centroids_list = [kmeans.centroids for kmeans in self.kmeanss]
        centroids = torch.stack(centroids_list).movedim(0, -1)
        return centroids

    def reset_parameters(self, x: "Tensor") -> "None":
        for i, kmeans in enumerate(self.kmeanss):
            kmeans.reset_parameters(x[..., i])

    def forward(
        self, x: "Tensor", return_centroids: "bool" = True
    ) -> "Tuple[Tensor, Optional[Tensor]]":
        assert x.shape[-1] == len(self.kmeanss)

        if len(self.kmeanss) == 1:
            # Fast path
            labels, assigned_centroids = self.kmeanss[0](
                x[..., 0], return_centroids
            )
            labels = labels[..., None]
            if return_centroids:
                assigned_centroids = assigned_centroids[..., None]
                return labels, assigned_centroids
            return labels

        labels_list, assigned_centroids_list = [], []
        for i, kmeans in enumerate(self.kmeanss):
            labels, assigned_centroids = kmeans(x[..., i], return_centroids)
            labels_list.append(labels)
            assigned_centroids_list.append(assigned_centroids)
        labels = torch.stack(labels_list).movedim(0, -1)
        if return_centroids:
            assigned_centroids = torch.stack(assigned_centroids_list).movedim(
                0, -1
            )
            return labels, assigned_centroids
        return labels

    def step(
        self,
        x: "Tensor",
        labels: "Optional[Tensor]" = None,
        return_drift: "bool" = True,
    ) -> "Optional[Tensor]":
        total_drift = 0.0
        for i, kmeans in enumerate(self.kmeanss):
            drift = kmeans.step(
                x[..., i],
                labels[..., i] if labels is not None else None,
                return_drift,
            )
            if return_drift:
                total_drift += drift
        if return_drift:
            return total_drift / len(self.kmeanss)

    def evaluate(
        self, x: "Tensor", labels: "Optional[Tensor]" = None
    ) -> "Tensor":
        total_inertia = 0.0
        for i, kmeans in enumerate(self.kmeanss):
            inertia = kmeans.evaluate(
                x[..., i], labels[..., i] if labels is not None else None
            )
            total_inertia += inertia
        return total_inertia / len(self.kmeanss)


@torch.jit.script
def _init_centroids(x: "Tensor", k: "int", init: "str" = "random") -> "Tensor":
    """Apply different methods for initialization of initial centroids (centroids)."""
    if init == "random":
        b = x.shape[0]
        rnd_idx = torch.multinomial(
            torch.full((b,), 1 / b, device=x.device), k, replacement=k > b
        )
        return x[rnd_idx].reshape(k, -1)
    else:
        raise ValueError(f"Unknown initialization method: {init}.")


@torch.jit.script
def _normalize(
    x: "Tensor", normalize: "str" = "mean", eps: "float" = 1e-8
) -> "Tensor":
    """Normalize input samples x according to specified method:

    - mean: subtract sample mean
    - minmax: min-max normalization subtracting sample min and divide by sample max
    - unit: normalize x to lie on D-dimensional unit sphere

    """
    if normalize == "mean":
        x -= x.mean(dim=0)[None]
    elif normalize == "minmax":
        x -= x.min(dim=-1).values[:, None]
        x /= x.max(dim=-1).values[:, None]
    elif normalize == "unit":
        # Normalize x to unit sphere
        z_msk = x == 0
        x = x.clone()
        x[z_msk] = eps
        x = (1.0 / (x.norm(p=2, dim=-1))).diag_embed() @ x
    else:
        raise ValueError(f"Unknown normalization type {normalize}.")
    return x


@torch.jit.script
def _compute_pairwise_distance(
    x: "Tensor", centroids: "Tensor", p: "float" = 2.0
) -> "Tensor":
    """Compute pairwise distances between samples in x and all centroids."""
    b, d = x.shape
    k, d = centroids.shape
    x = x[:, None].expand(b, k, d).reshape(-1, d)
    centroids = centroids.expand(b, k, d).reshape(-1, d)
    return nn.functional.pairwise_distance(x, centroids, p=p).reshape(b, k)


@torch.jit.script
def _group_by_label_mean(x: "Tensor", labels: "Tensor", k: "int") -> "Tensor":
    """Group samples in x by label and compute grouped mean."""
    M = nn.functional.one_hot(labels, num_classes=k).T.to(x.dtype)
    M = nn.functional.normalize(M, p=1.0, dim=-1)
    return M @ x


@torch.jit.script
def _compute_drift(
    centroids: "Tensor", old_centroids: "Tensor", p: "float" = 2.0
) -> "Tensor":
    """Compute centroid drift w.r.t. old centroids."""
    dist = (centroids - old_centroids).norm(p=p, dim=-1)
    dist[dist.isinf()] = 0
    # sum(dist, dim=-1)**2 -> use mean to be independent of number of points
    return dist.mean(dim=-1)


@torch.jit.script
def _compute_inertia(
    x: "Tensor", centroids: "Tensor", labels: "Tensor", p: "float" = 2.0
) -> "Tensor":
    """Compute sum of squared distances of samples to their closest cluster centroid."""
    b, d = x.shape
    # Select assigned centroid by label and compute squared distance
    assigned_centroids = centroids.gather(0, labels[:, None].expand(-1, d))
    # Squared distance to closest centroid
    dist = (x - assigned_centroids).norm(p=p, dim=-1) ** 2
    dist[dist.isinf()] = 0
    return dist


# Test
if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.datasets import make_blobs
    from sklearn.metrics.pairwise import pairwise_distances_argmin

    np.random.seed(0)
    torch.manual_seed(0)

    n_samples = 30000
    batch_size = 1024
    centers = [[1, 1], [-1, -1], [1, -1]]
    n_clusters = len(centers)
    max_iter = 100
    X, labels_true = make_blobs(
        n_samples=n_samples, centers=centers, cluster_std=0.7
    )

    # PyTorch
    k_means_torch = KMeans(2, n_clusters)
    X_torch = torch.from_numpy(X)
    t0 = time.time()
    for epoch in range(max_iter):
        for i in range(n_samples // batch_size):
            batch = X_torch[i * batch_size : (i + 1) * batch_size]
            k_means_torch.step(batch)
    t_batch = time.time() - t0

    # Scikit-learn
    mbk = MiniBatchKMeans(
        init="random",
        n_clusters=n_clusters,
        batch_size=batch_size,
        max_iter=max_iter,
        n_init=1,
        max_no_improvement=10000,
        reassignment_ratio=0.0,
        verbose=0,
    )
    t0 = time.time()
    mbk.fit(X)
    t_mini_batch = time.time() - t0

    k_means_torch_cluster_centers = k_means_torch.centroids.numpy()
    order = pairwise_distances_argmin(
        k_means_torch_cluster_centers, mbk.cluster_centers_
    )
    mbk_means_cluster_centers = mbk.cluster_centers_[order]

    k_means_labels = pairwise_distances_argmin(X, k_means_torch_cluster_centers)
    mbk_means_labels = pairwise_distances_argmin(X, mbk_means_cluster_centers)

    fig = plt.figure(figsize=(8, 3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.9)
    colors = ["#4EACC5", "#FF9C34", "#4E9A06"]

    # PyTorch
    ax = fig.add_subplot(1, 3, 1)
    for k, col in zip(range(n_clusters), colors):
        my_members = k_means_labels == k
        cluster_center = k_means_torch_cluster_centers[k]
        ax.plot(
            X[my_members, 0],
            X[my_members, 1],
            "w",
            markerfacecolor=col,
            marker=".",
        )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("MiniBatchKMeans PyTorch")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(
        -3.5,
        1.8,
        "train time: %.2fs\ninertia: %f"
        % (t_batch, k_means_torch.evaluate(X_torch).sum().item()),
    )

    # Scikit-learn
    ax = fig.add_subplot(1, 3, 2)
    for k, col in zip(range(n_clusters), colors):
        my_members = mbk_means_labels == k
        cluster_center = mbk_means_cluster_centers[k]
        ax.plot(
            X[my_members, 0],
            X[my_members, 1],
            "w",
            markerfacecolor=col,
            marker=".",
        )
        ax.plot(
            cluster_center[0],
            cluster_center[1],
            "o",
            markerfacecolor=col,
            markeredgecolor="k",
            markersize=6,
        )
    ax.set_title("MiniBatchKMeans Scikit-learn")
    ax.set_xticks(())
    ax.set_yticks(())
    plt.text(
        -3.5,
        1.8,
        "train time: %.2fs\ninertia: %f" % (t_mini_batch, mbk.inertia_),
    )

    # Initialize the different array to all False
    different = mbk_means_labels == 4
    ax = fig.add_subplot(1, 3, 3)

    for k in range(n_clusters):
        different += (k_means_labels == k) != (mbk_means_labels == k)

    identical = np.logical_not(different)
    ax.plot(
        X[identical, 0],
        X[identical, 1],
        "w",
        markerfacecolor="#bbbbbb",
        marker=".",
    )
    ax.plot(
        X[different, 0], X[different, 1], "w", markerfacecolor="m", marker="."
    )
    ax.set_title("Difference")
    ax.set_xticks(())
    ax.set_yticks(())

    plt.show()
