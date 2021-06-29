"""
Transformer Encoder Manager
"""
from speechbrain.lobes.models.transformer.TransformerUtilities import TransformerEncoder
from .Longformer import LongformerEncoder
from .Linformer import LinformerEncoder
from .conformer import ConformerEncoder
from .Reformer import ReformerEncoder

class EncoderManager:
    def __init__(
            self,
            encoder_module='transformer'
    ):
        self.encoder_module = encoder_module

    def build(self, nhead, num_encoder_layers, d_ffn, d_model, dropout, activation, normalize_before, **encoder_arguments):
        if self.encoder_module == "transformer":
            self.encoder = TransformerEncoder(
                nhead=nhead,
                num_layers=num_encoder_layers,
                d_ffn=d_ffn,
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )
        elif self.encoder_module == "reformer":
            self.encoder = ReformerEncoder(
                d_ffn=d_ffn,
                num_layers=num_encoder_layers,
                nhead=nhead,
                n_hashes=encoder_arguments.get('ref_n_hashes'),
                bucket_size=encoder_arguments.get('ref_bucket_size'),
                attn_chunks=encoder_arguments.get('ref_attn_chunks'),
                d_model=d_model,
                dropout=dropout,
                activation=activation,
                normalize_before=normalize_before,
            )
