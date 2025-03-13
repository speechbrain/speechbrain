"""K-means implementation.

Authors
* Luca Della Libera 2024
"""

import joblib
import torch


class MiniBatchKMeansSklearn(torch.nn.Module):
    """A wrapper for scikit-learn MiniBatchKMeans, providing integration with PyTorch tensors.

    See https://scikit-learn.org/stable/modules/generated/sklearn.cluster.MiniBatchKMeans.html.

    Arguments
    ---------
    *args : tuple
        Positional arguments passed to scikit-learn `MiniBatchKMeans`.
    **kwargs : dict
        Keyword arguments passed to scikit-learn `MiniBatchKMeans`.

    Example
    -------
    >>> import torch
    >>> device = "cpu"
    >>> n_clusters = 20
    >>> batch_size = 8
    >>> seq_length = 100
    >>> hidden_size = 256
    >>> model = MiniBatchKMeansSklearn(n_clusters).to(device)
    >>> input = torch.randn(batch_size, seq_length, hidden_size, device=device)
    >>> model.partial_fit(input)
    >>> labels = model(input)
    >>> labels.shape
    torch.Size([8, 100])
    >>> centers = model.cluster_centers
    >>> centers.shape
    torch.Size([20, 256])
    >>> len(list(model.buffers()))
    1
    >>> model.n_steps
    1
    >>> inertia = model.inertia(input)
    """

    def __init__(self, *args, **kwargs):
        try:
            from sklearn.cluster import MiniBatchKMeans
        except ImportError:
            err_msg = "The optional dependency `scikit-learn` must be installed to use this module.\n"
            err_msg += "Install using `pip install scikit-learn`.\n"
            raise ImportError(err_msg)

        super().__init__()
        self.kmeans = MiniBatchKMeans(*args, **kwargs)
        self.device = torch.device("cpu")
        self.register_buffer(
            "cluster_centers", self.cluster_centers_, persistent=False
        )

    def to(self, device=None, **kwargs):
        """See documentation of `torch.nn.Module.to`."""
        self.device = device
        return super().to(device)

    def save(self, path):
        """Saves the model to the specified file.

        Arguments
        ---------
        path : str
            The file path to save the model.
        """
        joblib.dump(self.kmeans, path)

    def load(self, path, end_of_epoch):
        """Loads the model from the specified file.

        Arguments
        ---------
        path : str
            The file path from which to load the model.
        end_of_epoch : bool
            Indicates if this load is triggered at the end of an epoch.
        """
        self.kmeans = joblib.load(path)
        self.cluster_centers = self.cluster_centers_

    def fit(self, input):
        """Fits the model to the input data.

        Arguments
        ---------
        input : torch.Tensor
            The input data tensor of shape (..., n_features).
        """
        numpy_input = input.detach().flatten(end_dim=-2).cpu().numpy()
        self.kmeans.fit(numpy_input)
        self.cluster_centers = self.cluster_centers_

    def partial_fit(self, input):
        """Performs an incremental fit of the model on the input data.

        Arguments
        ---------
        input : torch.Tensor
            The input data tensor of shape (..., n_features).
        """
        numpy_input = input.detach().flatten(end_dim=-2).cpu().numpy()
        self.kmeans.partial_fit(numpy_input)
        self.cluster_centers = self.cluster_centers_

    def forward(self, input):
        """Predicts cluster indices for the input data.

        Arguments
        ---------
        input : torch.Tensor
            The input data tensor of shape (..., n_features).

        Returns
        -------
        torch.Tensor
            Predicted cluster indices of shape (...,).
        """
        numpy_input = input.detach().flatten(end_dim=-2).cpu().numpy()
        cluster_idxes = self.kmeans.predict(numpy_input)
        cluster_idxes = torch.tensor(cluster_idxes, device=self.device).long()
        cluster_idxes = cluster_idxes.reshape(input.shape[:-1])
        return cluster_idxes

    def inertia(self, input):
        """Returns the inertia of the clustering.

        Arguments
        ---------
        input : torch.Tensor
            The input data tensor of shape (..., n_features).

        Returns
        -------
        torch.Tensor
            Inertia (sum of squared distances to the cluster centers).
        """
        numpy_input = input.detach().flatten(end_dim=-2).cpu().numpy()
        score = self.kmeans.score(numpy_input)
        inertia = -torch.tensor(score, device=self.device).float()
        return inertia

    @property
    def n_steps(self):
        """Returns the number of minibatches processed.

        Returns
        -------
        int
            Number of minibatches processed.
        """
        return self.kmeans.n_steps_

    @property
    def cluster_centers_(self):
        """Returns the cluster centers.

        Returns
        -------
        torch.Tensor
            Cluster centers of shape (n_clusters, n_features).
        """
        if hasattr(self.kmeans, "cluster_centers_"):
            cluster_centers = self.kmeans.cluster_centers_
            cluster_centers = torch.tensor(
                cluster_centers, device=self.device
            ).float()
        else:
            cluster_centers = torch.tensor(0.0, device=self.device)
        return cluster_centers
