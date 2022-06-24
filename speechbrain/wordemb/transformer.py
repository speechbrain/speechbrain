"""
A convenience wrapper for word embeddings retrieved out of
HuggingFace transformers (e.g. BERT)

Authors
* Artem Ploujnikov 2021
"""


import torch
import numpy as np
from torch import nn


def _last_n_layers(count):
    return range(-count, 0)


class TransformerWordEmbeddings(nn.Module):
    """A wrapper to retrieve word embeddings out of a pretrained Transformer model
    from HuggingFace Transformers (e.g. BERT)

    Arguments
    ---------
    model: str|nn.Module
        the underlying model instance or the name of the model
        to download

    tokenizer: str|transformers.tokenization_utils_base.PreTrainedTokenizerBase
        a pretrained tokenizer - or the identifier to retrieve
        one from HuggingFace

    layers: int|list
        a list of layer indexes from which to construct an embedding or the number of layers

    device:
        a torch device identifier. If provided, the model
        will be transferred onto that device

    Example
    -------
    NOTE: Doctests are disabled because the dependency on the
    HuggingFace transformer library is optional.

    >>> from transformers import AutoTokenizer, AutoModel # doctest: +SKIP
    >>> from speechbrain.wordemb.transformer import TransformerWordEmbeddings
    >>> model_name = "bert-base-uncased" # doctest: +SKIP
    >>> tokenizer = AutoTokenizer.from_pretrained(
    ...    model_name, return_tensors='pt') # doctest: +SKIP
    >>> model = AutoModel.from_pretrained(
    ...     model_name,
    ...     output_hidden_states=True) # doctest: +SKIP
    >>> word_emb = TransformerWordEmbeddings(
    ...     model=model,
    ...     layers=4,
    ...     tokenizer=tokenizer
    ... ) # doctest: +SKIP
    >>> embedding = word_emb.embedding(
    ...     sentence="THIS IS A TEST SENTENCE",
    ...     word="TEST"
    ... ) # doctest: +SKIP
    >>> embedding[:8] # doctest: +SKIP
    tensor([ 3.4332, -3.6702,  0.5152, -1.9301,  0.9197,  2.1628, -0.2841, -0.3549])
    >>> embeddings = word_emb.embeddings("This is cool") # doctest: +SKIP
    >>> embeddings.shape # doctest: +SKIP
    torch.Size([3, 768])
    >>> embeddings[:, :3] # doctest: +SKIP
    tensor([[-2.9078,  1.2496,  0.7269],
        [-0.9940, -0.6960,  1.4350],
        [-1.2401, -3.8237,  0.2739]])
    >>> sentences = [
    ...     "This is the first test sentence",
    ...     "This is the second test sentence",
    ...     "A quick brown fox jumped over the lazy dog"
    ... ]
    >>> batch_embeddings = word_emb.batch_embeddings(sentences) # doctest: +SKIP
    >>> batch_embeddings.shape # doctest: +SKIP
    torch.Size([3, 9, 768])
    >>> batch_embeddings[:, :2, :3] # doctest: +SKIP
    tensor([[[-5.0935, -1.2838,  0.7868],
             [-4.6889, -2.1488,  2.1380]],

            [[-4.4993, -2.0178,  0.9369],
             [-4.1760, -2.4141,  1.9474]],

            [[-1.0065,  1.4227, -2.6671],
             [-0.3408, -0.6238,  0.1780]]])
    """

    MSG_WORD = "'word' should be either a word or the index of a word"
    DEFAULT_LAYERS = 4

    def __init__(self, model, tokenizer=None, layers=None, device=None):
        super().__init__()
        if not layers:
            layers = self.DEFAULT_LAYERS
        layers = _last_n_layers(layers) if isinstance(layers, int) else layers
        self.layers = list(layers)

        if isinstance(model, str):
            if tokenizer is None:
                tokenizer = model
            model = _get_model(model)
            if isinstance(tokenizer, str):
                tokenizer = _get_tokenizer(tokenizer)
        elif tokenizer is None:
            raise ValueError(self.MSG_)

        self.model = model
        self.tokenizer = tokenizer
        if device is not None:
            self.device = device
            self.model = self.model.to(device)
        else:
            self.device = self.model.device

    def forward(self, sentence, word=None):
        """Retrieves a word embedding for the specified word within
        a given sentence, if a word is provided, or all word embeddings
        if only a sentence is given

        Arguments
        ---------
        sentence: str
            a sentence
        word: str|int
            a word or a word's index within the sentence. If a word
            is given, and it is encountered multiple times in a
            sentence, the first occurrence is used

        Returns
        -------
        emb: torch.Tensor
            the word embedding
        """
        return (
            self.embedding(sentence, word)
            if word
            else self.embeddings(sentence)
        )

    def embedding(self, sentence, word):
        """Retrieves a word embedding for the specified word within
        a given sentence

        Arguments
        ---------
        sentence: str
            a sentence
        word: str|int
            a word or a word's index within the sentence. If a word
            is given, and it is encountered multiple times in a
            sentence, the first occurrence is used

        Returns
        -------
        emb: torch.Tensor
            the word embedding
        """
        encoded = self.tokenizer.encode_plus(sentence, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**self._to_device(encoded))

        if isinstance(word, str):
            idx = self._get_word_idx(sentence, word)
        elif isinstance(word, int):
            idx = word
        else:
            raise ValueError(self.MSG_WORD)

        states = torch.stack(output.hidden_states)
        word_embedding = self._get_word_vector(encoded, states, idx).mean(dim=0)
        return word_embedding

    def embeddings(self, sentence):
        """
        Returns the model embeddings for all words
        in a sentence

        Arguments
        ---------
        sentence: str
            a sentence

        Returns
        -------
        emb: torch.Tensor
            a tensor of all word embeddings

        """
        encoded = self.tokenizer.encode_plus(sentence, return_tensors="pt")

        with torch.no_grad():
            output = self.model(**self._to_device(encoded))

        token_ids_word = torch.tensor(
            [
                idx
                for idx, word_id in enumerate(encoded.word_ids())
                if word_id is not None
            ],
            device=self.device,
        )
        states = torch.stack(output.hidden_states)
        return self._get_hidden_states(states, token_ids_word)

    def batch_embeddings(self, sentences):
        """Returns embeddings for a collection of sentences

        Arguments
        ---------
        sentences: List[str]
            a list of strings corresponding to a batch of
            sentences

        Returns
        -------
        emb: torch.Tensor
            a (B x W x E) tensor
            B - the batch dimensions (samples)
            W - the word dimension
            E - the embedding dimension
        """
        encoded = self.tokenizer.batch_encode_plus(
            sentences, padding=True, return_tensors="pt"
        )

        with torch.no_grad():
            output = self.model(**self._to_device(encoded))

        states = torch.stack(output.hidden_states)
        return self._get_hidden_states(states)

    def _to_device(self, encoded):
        return {
            key: self._tensor_to_device(value) for key, value in encoded.items()
        }

    def _tensor_to_device(self, value):
        return (
            value.to(self.device) if isinstance(value, torch.Tensor) else value
        )

    def _get_word_idx(self, sent, word):
        return sent.split(" ").index(word)

    def _get_hidden_states(self, states, token_ids_word=None):
        output = states[self.layers].sum(0).squeeze()
        if token_ids_word is not None:
            output = output[token_ids_word]
        else:
            output = output[:, 1:-1, :]
        return output

    def _get_word_vector(self, encoded, states, idx):
        token_ids_word = torch.from_numpy(
            np.where(np.array(encoded.word_ids()) == idx)[0]
        ).to(self.device)
        return self._get_hidden_states(states, token_ids_word)

    def to(self, device):
        """Transfers the model to the specified PyTorch device"""
        self.device = device
        self.model = self.model.to(device)
        return self


class MissingTransformersError(Exception):
    """Thrown when HuggingFace Transformers is not installed"""

    MESSAGE = "This module requires HuggingFace Transformers"

    def __init__(self):
        super().__init__(self.MESSAGE)


def _get_model(identifier):
    """Tries to retrieve a pretrained model from Huggingface"""
    try:
        from transformers import AutoModel  # noqa

        return AutoModel.from_pretrained(identifier, output_hidden_states=True)
    except ImportError:
        raise MissingTransformersError()


def _get_tokenizer(identifier):
    """Tries to retreive a pretrained tokenizer from HuggingFace"""
    try:
        from transformers import AutoTokenizer  # noqa

        return AutoTokenizer.from_pretrained(identifier)
    except ImportError:
        raise MissingTransformersError()
