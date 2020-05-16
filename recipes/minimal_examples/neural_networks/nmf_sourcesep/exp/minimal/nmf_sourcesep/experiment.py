#!/usr/bin/python
import speechbrain as sb
import torch

params_file = "params.yaml"  # "recipes/minimal_examples/neural_networks/nmf_sourcesep/params.yaml"
with open(params_file) as fin:
    params = sb.yaml.load_extended_yaml(fin)

sb.core.create_experiment_directory(
    experiment_directory=params.output_folder, params_to_save=params_file,
)


class NMF_Brain(sb.core.Brain):

    r"""Brain class abstracts away the details of data loops.

    The primary purpose of the `Brain` class is the implementation of
    the `fit()` method, which iterates epochs and datasets for the
    purpose of "fitting" a set of modules to a set of data.

    In order to use the `fit()` method, one should sub-class the `Brain` class
    and override any methods for which the default behavior does not match
    the use case. For a simple use case (e.g. training a single model with
    a single dataset) the only methods that need to be overridden are:

    * `forward()`
    * `compute_objectives()`

    The example below illustrates how overriding these two methods is done.

    For more complicated use cases, such as multiple modules that need to
    be updated, the following methods can be overridden:

    * `fit_batch()`
    * `evaluate_batch()`

    If there is more than one objective (either for training or evaluation),
    the method for summarizing the losses (e.g. averaging) can be specified
    by overriding the `summarize()` method.

    Arguments
    ---------
    modules : list of torch.Tensors
        The modules that will be updated using the optimizer.
    optimizer : optimizer
        The class to use for updating the modules' parameters.
    scheduler : scheduler
        An object that changes the learning rate based on performance.
    saver : Checkpointer
        This is called by default at the end of each epoch to save progress.

    Example
    -------
    >>> from speechbrain.nnet.optimizers import Optimize
    >>> class SimpleBrain(Brain):
    ...     def forward(self, x, init_params=False):
    ...         return self.modules[0](x)
    ...     def compute_objectives(self, predictions, targets, train=True):
    ...         return torch.nn.functional.l1_loss(predictions, targets)
    >>> tmpdir = getfixture('tmpdir')
    >>> model = torch.nn.Linear(in_features=10, out_features=10)
    >>> brain = SimpleBrain([model], Optimize('sgd', 0.01))
    >>> brain.fit(
    ...     train_set=([torch.rand(10, 10)], [torch.rand(10, 10)]),
    ... )
    """

    def forward(self, X, init_params=False):
        """Forward pass, to be overridden by sub-classes.

        Arguments
        ---------
        x : torch.Tensor or list of tensors
            The input tensor or tensors for processing.
        init_params : bool
            Whether this pass should initialize parameters rather
            than return the results of the forward pass.
        """

        X = params.compute_features(X[0][1], init_params)
        # concatenate all the inputs
        X = X.permute(0, 2, 1)
        X = X.reshape(-1, X.size(-1)).t()

        m = X.shape[0]
        n = X.shape[1]
        eps = 1e-20

        # Normalize input
        g = X.sum(dim=0) + eps
        z = X / g

        # initialize
        w = torch.rand(m, params.K) + 10
        w /= torch.sum(w, dim=0)

        h = torch.rand(params.K, n) + 10
        h /= torch.sum(h, dim=0) + eps

        for ep in range(params.N_epochs):
            v = z / (torch.matmul(w, h) + eps)

            nw = w * torch.matmul(v, h.t())
            w = nw / (torch.sum(nw, dim=0) + eps)

            nh = h * torch.matmul(w.t(), v)
            h = nh / (torch.sum(nh, dim=0) + eps)

        h *= g
        return torch.matmul(w, h), w, h / g

    def add_stats(self, *args):
        pass

    def fit_batch(self, batch):
        """Fit one batch, override to do multiple updates.

        The default impementation depends on three methods being defined
        with a particular behavior:

        * `forward()`
        * `compute_objectives()`
        * `optimizer()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for training. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs = batch
        predictions = self.forward(inputs)
        self.training_out = predictions

    def evaluate_batch(self, batch):
        """Evaluate one batch, override for different procedure than train.

        The default impementation depends on two methods being defined
        with a particular behavior:

        * `forward()`
        * `compute_objectives()`

        Arguments
        ---------
        batch : list of torch.Tensors
            batch of data to use for evaluation. Default implementation assumes
            this batch has two elements: inputs and targets.
        """
        inputs, targets = batch
        output = self.forward(inputs)
        loss, stats = self.compute_objectives(output, targets, train=False)
        stats["loss"] = loss.detach()
        return stats

    def summarize(self, stats, write=False):
        """Take a list of stats from a pass through data and summarize it.

        By default, averages the loss and returns the average.

        Arguments
        ---------
        stats : list of dicts
            A list of stats to summarize.
        """
        return {"loss": float(sum(s["loss"] for s in stats) / len(stats))}

    def on_epoch_end(self, *args):
        """Gets called at the end of each epoch.

        Arguments
        ---------
        summary : mapping
            This dict defines summary info about the validation pass, if
            the validation data was passed, otherwise training pass. The
            output of the `summarize` method is directly passed.
        max_keys : list of str
            A sequence of strings that match keys in the summary. Highest
            value is the relevant value.
        min_keys : list of str
            A sequence of strings that match keys in the summary. Lowest
            value is the relevant value.
        """
        print("NMF training has finished")


def separate(Whats, mixture_loader):

    W1, W2 = Whats
    X = list(mixture_loader[0])[0][1]
    X = params.compute_features(X, init_params=None)

    # concatenate all the inputs
    X = X.permute(0, 2, 1)
    X = X.reshape(-1, X.size(-1)).t()

    n = X.shape[1]
    eps = 1e-20

    # Normalize input
    g = X.sum(dim=0) + eps
    z = X / g

    # initialize
    w = torch.cat([W1, W2], dim=1)
    K = w.size(1)
    K1 = W1.size(1)
    K2 = W2.size(1)

    h = torch.rand(K, n) + 10
    h /= torch.sum(h, dim=0) + eps

    for ep in range(100):
        v = z / (torch.matmul(w, h) + eps)

        nh = h * torch.matmul(w.t(), v)
        h = nh / (torch.sum(nh, dim=0) + eps)

    h *= g
    Xhat1 = torch.matmul(w[:, :K1], h[:K1, :])
    Xhat1 = torch.split(Xhat1.unsqueeze(0), Xhat1.size(1) // 4, dim=2)
    Xhat1 = torch.cat(Xhat1, dim=0)

    Xhat2 = torch.matmul(w[:, K2:], h[K2:, :])
    Xhat2 = torch.split(Xhat2.unsqueeze(0), Xhat2.size(1) // 4, dim=2)
    Xhat2 = torch.cat(Xhat2, dim=0)

    return Xhat1, Xhat2


NMF1 = NMF_Brain([], None)
print("fitting model 1")
NMF1.fit(
    train_set=params.train_loader1(), valid_set=None, epoch_counter=range(1)
)
W1hat = NMF1.training_out[1]

NMF2 = NMF_Brain([], None)
print("fitting model 2")
NMF2.fit(
    train_set=params.train_loader2(), valid_set=None, epoch_counter=range(1)
)
W2hat = NMF2.training_out[1]

mixture_loader = params.test_loader()
Xhat1, Xhat2 = separate([W1hat, W2hat], mixture_loader)

# all_shat1s = []
# all_shat2s = []
# for i in range(Xhat1.shape[0]):
#    Xhat1_stft = spectr_mix[i].numpy() * np.exp(np.complex(0,1)*phase_mix[i].numpy())
#    Xhat2_stft = Xhat2[i].numpy() * np.exp(1j*phase_mix[i]).numpy()
#
#    shat1 = torch.from_numpy(lr.core.istft(Xhat1_stft)).unsqueeze(0)
#    shat2 = torch.from_numpy(lr.core.istft(Xhat2_stft)).unsqueeze(0)
# pdb.set_trace()

# asr_brain.evaluate(params.train_loader())
