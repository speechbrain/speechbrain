"""WavLM + LSTM model with character-level SentencePiece tokenizer.

Authors
 * Luca Della Libera 2023
"""

import os

from torch import nn

from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
from speechbrain.nnet.RNN import LSTM as SBLSTM
from speechbrain.tokenizers.SentencePiece import SentencePiece
from speechbrain.utils.data_utils import download_file


__all__ = [
    "ProgressiveWavLM",
]


_TOKENIZER_URL = (
    "https://www.dropbox.com/sh/gxzzr2znd9z8tu1/AACQgjzSVG1PgoyIK_Og8Brda?dl=1"
)

_TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "tokenizer")


class Decoder(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        hidden_size=1024,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                SBLSTM(
                    hidden_size,
                    input_size=input_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    bidirectional=bidirectional,
                ),
            ]
        )
        self.out_proj = nn.Linear(
            (2 if bidirectional else 1) * hidden_size, output_size,
        )

    def forward(self, input, lengths=None):
        output, state = self.layers[0](input, lengths=lengths)
        for layer in self.layers[1:]:
            output, state = layer(output, state, lengths=lengths)
        output = self.out_proj(output)
        return output


class Model(nn.Module):
    def __init__(
        self,
        source,
        save_path,
        vocab_size,
        encoder_kwargs=None,
        decoder_kwargs=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.encoder = HuggingFaceWav2Vec2(
            source, save_path, **(encoder_kwargs or {}),
        )
        self.decoder = Decoder(
            self.encoder.model.config.hidden_size,
            vocab_size,
            **(decoder_kwargs or {}),
        )
        self.config = self.encoder.model.config

    def forward(self, wav, wav_lens=None):
        output = self.encoder(wav, wav_lens)
        output = self.decoder(output, wav_lens)
        return output


class ProgressiveWavLM(nn.Module):
    def __init__(
        self,
        # Encoder (WavLM)
        source,
        save_path,
        output_norm=False,
        freeze=False,
        freeze_encoder=False,
        freeze_feature_extractor=False,
        apply_spec_augment=False,
        # Decoder (LSTM)
        hidden_size=1024,
        num_layers=1,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__()
        download_file(
            _TOKENIZER_URL,
            f"{_TOKENIZER_PATH}.zip",
            unpack=True,
            dest_unpack=_TOKENIZER_PATH,
        )
        self.tokenizer = SentencePiece(
            model_dir=_TOKENIZER_PATH, vocab_size=4887, model_type="char",
        ).sp
        vocab_size = self.tokenizer.vocab_size()
        encoder_kwargs = {
            "output_norm": output_norm,
            "freeze": freeze_encoder or freeze,
            "freeze_feature_extractor": freeze_feature_extractor,
            "apply_spec_augment": apply_spec_augment,
            "output_all_hiddens": False,
        }
        decoder_kwargs = {
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "dropout": dropout,
            "bidirectional": bidirectional,
        }
        self.model = Model(
            source, save_path, vocab_size, encoder_kwargs, decoder_kwargs,
        )
        if freeze:
            self.model.requires_grad_(False)

    def forward(self, wav, wav_lens=None):
        output = self.model(wav, wav_lens)
        return output
