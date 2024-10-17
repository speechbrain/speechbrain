"""Tokenizer for semantic tokens.

Author
 * Pooneh Mousavi 2024
"""

import numpy as np
import torch


class DiscreteSSLTokenizer:
    """This class is tokenizer for DiscreteSSL models that apply post-processing on the semnatic tokens extracted from DiscreteSSL model.
    It makes the token ids of each layer to be unique by adding the token IDs of each layer by layer_num*sunmber_of _cluster.
    It applies deduplication for each layer independently if the field is set to true for the layer and padded all items with zero.
    It applies subwording for each layer independently if the sentence piece tokenizer is set to for the layer and padded all items with zero.
    If subwording is not applied, all token IDs are incremented by one to avoid conflict between pad_id(0) and cluster with centroid zero.

    Arguments
    ---------
    num_clusters: List[int]
        determine the number of clusters of the  kmeans models. It could be varying for each layer.

    Example
    -------
    >>> import torch
    >>> inputs = torch.randint(0,1000,(3, 6, 2))
    >>> ssl_layer_num = [7,23]
    >>> deduplicate =[False, True]
    >>> bpe_tokenizers=[None, None]
    >>> num_clusters = [1000,2000]
    >>> tokenizer = DiscreteSSLTokenizer(num_clusters=num_clusters)
    >>> tokens= tokenizer.encode(inputs,SSL_layers=ssl_layer_num, deduplicates=deduplicate, bpe_tokenizers=bpe_tokenizers)
    >>> print(tokens.shape)
    torch.Size([3, 6, 2])
    """

    def __init__(self, num_clusters):
        self.num_clusters = num_clusters

    def textify(self, tokens):
        """Convert token ID to char to be used for training sentencepiece tokenizer.

        Arguments
        ---------
        tokens : torch.Tensor
            A (Batch x Seq ) tensor of audio tokens

        Returns
        -------
        processed_tokens : list
            A (Batch x Seq) list of corresponding char for each token ID.
        """
        tokens_char = []
        # tokens = [row - layer *  self.num_clusters for row in input]
        for row in tokens:
            tokens_char.append(" ".join([chr((token) + 97) for token in row]))
        return tokens_char

    def encode(
        self, input, SSL_layers=[7], deduplicates=[False], bpe_tokenizers=[None]
    ):
        """Takes an input tokenized wavform and return its corresponding processed tokens.

        Arguments
        ---------
        input : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens.
        SSL_layers: List[int] (default: [7]):
            determine which layers of SSL should be used to extract information.
        deduplicates: List[boolean] (default: [False]):
            determine to apply deduplication(remove duplicate subsequent tokens) on the tokens extracted for the corresponding layer.
        bpe_tokenizers: List[int] (default: [None]):
            determine to apply subwording on the tokens extracted for the corresponding layer if the sentencePiece tokenizer is trained for that layer.

        Returns
        -------
        processed_tokens : torch.Tensor
            A (Batch x Seq x num_SSL_layers) tensor of audio tokens after applying deduplication and subwording if necessary.
        """
        assert input.shape[2] == len(
            SSL_layers
        ), f"input shape:{input.shape} has conflicts with the length of provided SSL_layers: {len(SSL_layers)}. The second dimension of input should be the same  as number of layers!!!"
        token_ids = []
        for i, duplicate in enumerate(deduplicates):
            tokens = []
            if duplicate:
                unique_token_ids = [
                    row[np.diff(row, prepend=np.nan).astype(bool)]
                    for row in input[:, :, i].cpu()
                ]
                layer_token_ids = [
                    row.clone().detach() for row in unique_token_ids
                ]
                tokens.extend(layer_token_ids)

            else:
                tokens.extend(input[:, :, i])

            if bpe_tokenizers[i] is not None:
                token_char = self.textify(tokens)
                token_ids.extend(
                    [
                        torch.LongTensor(bpe_tokenizers[i].encode_as_ids(row))
                        + SSL_layers[i] * self.num_clusters[i]
                        for row in token_char
                    ]
                )
            else:
                token_ids.extend(
                    [
                        row + SSL_layers[i] * self.num_clusters[i] + 1
                        for row in tokens
                    ]
                )

        return torch.stack(
            torch.split(
                torch.nn.utils.rnn.pad_sequence(token_ids, batch_first=True),
                input.shape[0],
            ),
            dim=2,
        )
