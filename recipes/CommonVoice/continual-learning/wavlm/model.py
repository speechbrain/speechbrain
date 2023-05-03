"""WavLM + LSTM model with Whisper's tokenizer.

Authors
 * Luca Della Libera 2023
"""

from torch import nn
from transformers.models.whisper.tokenization_whisper import WhisperTokenizer

from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2


__all__ = [
    "ProgressiveWavLM",
]


class Decoder(nn.Module):
    def __init__(
        self, input_size, output_size, hidden_size=1024, **lstm_kwargs
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.LSTM(input_size, hidden_size, batch_first=True, **lstm_kwargs)]
        )
        self.out_proj = nn.Linear(hidden_size, output_size)

    def forward(self, input, lengths=None):
        # TODO: forward lengths argument?
        output, state = self.layers[0](input)
        for layer in self.layers[1:]:
            output, state = layer(output, state)
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

    def resize_out_proj(self, new_num_tokens):
        old_out_proj = self.decoder.out_proj
        n = min(old_out_proj.out_features, new_num_tokens)
        has_bias = old_out_proj.bias is not None
        new_out_proj = nn.Linear(
            old_out_proj.in_features,
            new_num_tokens,
            bias=has_bias,
            device=old_out_proj.weight.device,
            dtype=old_out_proj.weight.dtype,
        )
        new_out_proj.weight.data[:n, :] = old_out_proj.weight.data[:n, :]
        new_out_proj.requires_grad_(old_out_proj.weight.requires_grad)
        if has_bias:
            new_out_proj.bias.data[:n] = old_out_proj.bias.data[:n]
            new_out_proj.requires_grad_(old_out_proj.bias.requires_grad)
        self.decoder.out_proj = new_out_proj
        self.vocab_size = new_num_tokens
        return new_out_proj

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
        self.tokenizer = WhisperTokenizer.from_pretrained(
            "openai/whisper-tiny",
            language=None,
            task="transcribe",
            predict_timestamps=False,
        )
        vocab_size = len(self.tokenizer.get_vocab())
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
