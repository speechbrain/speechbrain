import os
import tempfile

import torch


def test_tokenizer():
    from speechbrain.tokenizers.SentencePiece import SentencePiece

    gt = [
        ["HELLO", "MORNING", "MORNING", "HELLO"],
        ["HELLO", "MORNING", "HELLO"],
    ]

    # Word-level input test
    dict_int2lab = {1: "HELLO", 2: "MORNING"}

    spm = SentencePiece(
        os.path.abspath("tests/tmp/tokenizer_data"),
        100,
        annotation_train=os.path.abspath(
            "tests/samples/annotation/tokenizer.csv"
        ),
        annotation_read="wrd",
        model_type="bpe",
    )
    encoded_seq_ids, encoded_seq_pieces = spm(
        torch.Tensor([[1, 2, 2, 1], [1, 2, 1, 0]]),
        torch.Tensor([1.0, 0.75]),
        dict_int2lab,
        task="encode",
    )
    lens = (encoded_seq_pieces * encoded_seq_ids.shape[1]).round().int()
    # decode from torch tensors (batch, batch_lens)
    words_seq = spm(encoded_seq_ids, encoded_seq_pieces, task="decode")
    assert words_seq == gt, "output not the same"
    # decode from a list of bpe sequence (without padding)
    hyps_list = [
        encoded_seq_ids[0].int().tolist(),
        encoded_seq_ids[1][: lens[1]].int().tolist(),
    ]
    words_seq = spm(hyps_list, task="decode_from_list")
    assert words_seq == gt, "output not the same"

    # Char-level input test
    dict_int2lab = {
        1: "H",
        2: "E",
        3: "L",
        4: "O",
        5: "M",
        6: "R",
        7: "N",
        8: "I",
        9: "G",
        10: "_",
    }

    spm = SentencePiece(
        os.path.abspath("tests/tmp/tokenizer_data"),
        100,
        annotation_train=os.path.abspath(
            "tests/sample/annotation/tokenizer.csv"
        ),
        annotation_read="char",
        char_format_input=True,
        model_type="bpe",
    )
    encoded_seq_ids, encoded_seq_pieces = spm(
        torch.Tensor(
            [
                [
                    1,
                    2,
                    3,
                    3,
                    4,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    1,
                    2,
                    3,
                    3,
                    4,
                ],
                [
                    1,
                    2,
                    3,
                    3,
                    4,
                    10,
                    5,
                    4,
                    6,
                    7,
                    8,
                    7,
                    9,
                    10,
                    1,
                    2,
                    3,
                    3,
                    4,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                ],
            ]
        ),
        torch.Tensor([1.0, 0.7037037037037037]),
        dict_int2lab,
        task="encode",
    )
    lens = (encoded_seq_pieces * encoded_seq_ids.shape[1]).round().int()
    # decode from torch tensors (batch, batch_lens)
    words_seq = spm(encoded_seq_ids, encoded_seq_pieces, task="decode")
    assert words_seq == gt, "output not the same"
    # decode from a list of bpe sequence (without padding)
    hyps_list = [
        encoded_seq_ids[0].int().tolist(),
        encoded_seq_ids[1][: lens[1]].int().tolist(),
    ]
    words_seq = spm(hyps_list, task="decode_from_list")
    assert words_seq == gt, "output not the same"


def test_tokenizer_text_file():
    """Test that custom text_file parameter is properly assigned and used."""
    from speechbrain.tokenizers.SentencePiece import SentencePiece

    # Create a temporary directory for test outputs
    with tempfile.TemporaryDirectory() as temp_dir:
        # Define custom text_file path
        custom_text_file = os.path.join(temp_dir, "custom_text_file.txt")

        # Create tokenizer with custom text_file parameter
        spm = SentencePiece(
            os.path.join(temp_dir, "tokenizer_data"),
            100,
            annotation_train=os.path.abspath(
                "tests/samples/annotation/tokenizer.csv"
            ),
            annotation_read="wrd",
            model_type="bpe",
            text_file=custom_text_file,
        )

        # Verify that the custom text_file path was assigned correctly
        assert spm.text_file == custom_text_file, (
            f"Expected text_file to be {custom_text_file}, "
            f"but got {spm.text_file}"
        )

        # Verify that the text file was created at the custom location
        assert os.path.isfile(custom_text_file), (
            f"Custom text file was not created at {custom_text_file}"
        )
