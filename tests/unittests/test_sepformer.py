import torch


def test_JITability():
    from speechbrain.pretrained.interfaces import SepformerSeparation
    from speechbrain.lobes.models.dual_path import (
        Encoder,
        Decoder,
        Dual_Path_Model,
        SBTransformerBlock,
    )

    encoder = Encoder(kernel_size=16, out_channels=256)
    decoder = Decoder(
        in_channels=256, out_channels=1, kernel_size=16, stride=8, bias=False
    )
    intra_model = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0.0,
        use_positional_encoding=True,
        norm_before=True,
    )
    inter_model = SBTransformerBlock(
        num_layers=8,
        d_model=256,
        nhead=8,
        d_ffn=1024,
        dropout=0.0,
        use_positional_encoding=True,
        norm_before=True,
    )
    masknet = Dual_Path_Model(
        num_spks=2,
        in_channels=256,
        out_channels=256,
        num_layers=2,
        K=250,
        intra_model=intra_model,
        inter_model=inter_model,
        norm="ln",
        linear_layer_after_inter_intra=False,
        skip_around_intra=True,
    )

    modules = dict(encoder=encoder, decoder=decoder, masknet=masknet)
    hparams = dict(modules=modules, num_spks=2)

    sepformer = SepformerSeparation(modules=modules, hparams=hparams)

    class JITableSepformer(torch.nn.Module):
        def __init__(self, encoder, decoder, masknet):
            super().__init__()
            self.encoder = encoder
            self.decoder = decoder
            self.masknet = masknet

        def forward(self, mix):
            mix_w = self.encoder(mix)
            est_mask = self.masknet(mix_w)
            mix_w = torch.stack([mix_w] * 2)
            sep_h = mix_w * est_mask

            # Decoding
            est_source = torch.cat(
                [self.decoder(sep_h[i]).unsqueeze(-1) for i in range(2)], dim=-1
            )

            # T changed after conv1d in encoder, fix it here
            T_origin = mix.size(1)
            T_est = est_source.size(1)
            if T_origin > T_est:
                est_source = torch.nn.functional.pad(
                    est_source, (0, 0, 0, T_origin - T_est)
                )
            else:
                est_source = est_source[:, :T_origin, :]
            return est_source

    sepformer_jit = torch.jit.script(
        JITableSepformer(encoder, decoder, masknet)
    )
    input = torch.rand(size=(2, 8000), dtype=torch.float32)
    with torch.no_grad():
        assert torch.allclose(
            sepformer.separate_batch(input), sepformer_jit(input)
        )
