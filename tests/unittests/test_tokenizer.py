import torch


def test_tokenizer():
    from speechbrain.tokenizers.SentencePiece import SentencePiece

    dict_int2lab = {1: "HELLO", 2: "MORNING"}

    spm = SentencePiece(
        "tokenizer_data/",
        2000,
        csv_train="tests/unittests/tokenizer_data/dev-clean.csv",
        csv_read="wrd",
        model_type="bpe",
    )
    encoded_seq_ids, encoded_seq_pieces = spm(
        torch.Tensor([[1, 2, 2, 1], [1, 2, 1, 0]]),
        torch.Tensor([1.0, 0.75]),
        dict_int2lab,
        task="encode",
        init_params=True,
    )
    # decode from torch tensors (batch, batch_lens)
    spm(encoded_seq_ids, encoded_seq_pieces, task="decode")
    # decode from a list of bpe sequence (without padding)
    hyps_list = [
        encoded_seq_ids[0].int().tolist(),
        encoded_seq_ids[1][:-1].int().tolist(),
    ]
    spm(hyps_list, task="decode_from_list")
