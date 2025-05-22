import os
import shutil
import tempfile
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
import torch

from speechbrain.k2_integration import k2
from speechbrain.utils.logger import get_logger

logger = get_logger(__name__)


@pytest.fixture
def tmp_csv_file(tmp_path):
    csv_file = tmp_path / "train.csv"
    with open(csv_file, "w", encoding="utf-8") as f:
        f.write("ID,duration,wav,spk_id,wrd\n")
        f.write("1,1,1,1,hello world\n")
        f.write("2,0.5,1,1,hello\n")
    return csv_file


def test_get_lexicon(tmp_path, tmp_csv_file):
    # Define the inputs
    lang_dir = tmp_path
    csv_files = [tmp_csv_file]
    vocab_files = []  # This list is empty for simplicity in this test.

    # Call the function
    from speechbrain.k2_integration.lexicon import prepare_char_lexicon

    prepare_char_lexicon(
        lang_dir, vocab_files, csv_files, add_word_boundary=False
    )

    # Read the output and assert its content
    with open(lang_dir / "lexicon.txt", "r", encoding="utf-8") as f:
        assert f.read() == "<UNK> <unk>\nhello h e l l o\nworld w o r l d\n"


def test_get_lexicon_with_boundary(tmp_path, tmp_csv_file):
    # Define the inputs
    lang_dir = tmp_path
    csv_files = [tmp_csv_file]
    vocab_files = []

    # Call the function with word boundaries
    from speechbrain.k2_integration.lexicon import prepare_char_lexicon

    prepare_char_lexicon(
        lang_dir, vocab_files, csv_files, add_word_boundary=True
    )

    # Read the output and assert its content
    with open(lang_dir / "lexicon.txt", "r", encoding="utf-8") as f:
        assert (
            f.read()
            == "<UNK> <unk>\nhello h e l l o <eow>\nworld w o r l d <eow>\n"
        )


@pytest.fixture
def mock_lexicon_file(tmp_path):
    lexicon_content = "hello h e l l o\nworld w o r l d\n"
    lexicon_file = tmp_path / "mock_lexicon.txt"
    with open(lexicon_file, "w", encoding="utf-8") as f:
        f.write(lexicon_content)
    return lexicon_file


def test_read_lexicon(mock_lexicon_file):
    expected_output = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]

    from speechbrain.k2_integration.lexicon import read_lexicon

    output = read_lexicon(mock_lexicon_file)
    assert output == expected_output


def test_write_lexicon(tmp_path):
    # Sample lexicon data.
    lexicon_data = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]

    # Path to save the lexicon file.
    lexicon_file = tmp_path / "test_lexicon.txt"

    # Use the function to write lexicon to the file.
    from speechbrain.k2_integration.lexicon import write_lexicon

    write_lexicon(lexicon_file, lexicon_data)

    # Expected content of the lexicon file.
    expected_content = "hello h e l l o\nworld w o r l d\n"

    # Read back the content of the file and assert its correctness.
    with open(lexicon_file, "r", encoding="utf-8") as f:
        assert f.read() == expected_content


def test_get_tokens_basic():
    # Prepare a mock lexicon
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]
    from speechbrain.k2_integration.prepare_lang import get_tokens

    tokens = get_tokens(lexicon)
    expected_tokens = ["d", "e", "h", "l", "o", "r", "w"]
    assert tokens == expected_tokens


def test_get_tokens_with_sil():
    # Prepare a mock lexicon
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d", "SIL"]),
    ]
    with pytest.raises(AssertionError):
        from speechbrain.k2_integration.prepare_lang import get_tokens

        get_tokens(lexicon)


def test_get_tokens_manually_add_sil():
    # Prepare a mock lexicon
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]
    from speechbrain.k2_integration.prepare_lang import get_tokens

    tokens = get_tokens(lexicon, manually_add_sil_to_tokens=True)
    expected_tokens = ["SIL", "d", "e", "h", "l", "o", "r", "w"]
    assert tokens == expected_tokens


def test_unique_pronunciations():
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]
    from speechbrain.k2_integration.prepare_lang import add_disambig_symbols

    new_lexicon, max_disambig = add_disambig_symbols(lexicon)
    assert new_lexicon == lexicon
    assert max_disambig == 0


def test_repeated_pronunciations():
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("greeting", ["h", "e", "l", "l", "o"]),
    ]
    from speechbrain.k2_integration.prepare_lang import add_disambig_symbols

    new_lexicon, max_disambig = add_disambig_symbols(lexicon)
    assert new_lexicon == [
        ("hello", ["h", "e", "l", "l", "o", "#1"]),
        ("greeting", ["h", "e", "l", "l", "o", "#2"]),
    ]
    assert max_disambig == 2


def test_prefix_pronunciations():
    lexicon = [("he", ["h", "e"]), ("hello", ["h", "e", "l", "l", "o"])]
    from speechbrain.k2_integration.prepare_lang import add_disambig_symbols

    new_lexicon, max_disambig = add_disambig_symbols(lexicon)
    assert new_lexicon == [
        ("he", ["h", "e", "#1"]),
        ("hello", ["h", "e", "l", "l", "o"]),
    ]
    assert max_disambig == 1


def test_mixed_pronunciations():
    lexicon = [
        ("he", ["h", "e"]),
        ("hello", ["h", "e", "l", "l", "o"]),
        ("hey", ["h", "e"]),
        ("world", ["h", "e", "l", "l", "o"]),
    ]
    from speechbrain.k2_integration.prepare_lang import add_disambig_symbols

    new_lexicon, max_disambig = add_disambig_symbols(lexicon)
    # Correct the expected output based on function behavior
    assert new_lexicon == [
        ("he", ["h", "e", "#1"]),
        ("hello", ["h", "e", "l", "l", "o", "#1"]),
        ("hey", ["h", "e", "#2"]),
        ("world", ["h", "e", "l", "l", "o", "#2"]),
    ]
    assert max_disambig == 2


def test_lexicon_to_fst():
    # Sample lexicon: Each word maps to a list of tokens
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]

    # Maps from token to ID and word to ID
    token2id = {
        "<eps>": 0,
        "h": 1,
        "e": 2,
        "l": 3,
        "o": 4,
        "w": 5,
        "r": 6,
        "d": 7,
        "SIL": 8,
        "#0": 9,  # for self-loop
    }

    word2id = {"<eps>": 0, "hello": 1, "world": 2, "#0": 3}  # for self-loop

    from speechbrain.k2_integration.prepare_lang import lexicon_to_fst

    fsa = lexicon_to_fst(
        lexicon=lexicon,
        token2id=token2id,
        word2id=word2id,
        sil_token="SIL",
        sil_prob=0.5,
        need_self_loops=True,  # Assuming you have the add_self_loops function implemented
    )

    # Ensure fsa is a valid k2 FSA
    assert isinstance(fsa, k2.Fsa)


def test_lexicon_to_fst_no_sil():
    # Sample lexicon: Each word maps to a list of tokens
    lexicon = [
        ("hello", ["h", "e", "l", "l", "o"]),
        ("world", ["w", "o", "r", "l", "d"]),
    ]

    # Maps from token to ID and word to ID
    token2id = {
        "<eps>": 0,
        "h": 1,
        "e": 2,
        "l": 3,
        "o": 4,
        "w": 5,
        "r": 6,
        "d": 7,
        "#0": 8,  # for self-loop
    }

    word2id = {"<eps>": 0, "hello": 1, "world": 2, "#0": 3}  # for self-loop

    from speechbrain.k2_integration.prepare_lang import lexicon_to_fst_no_sil

    fsa = lexicon_to_fst_no_sil(
        lexicon=lexicon,
        token2id=token2id,
        word2id=word2id,
        need_self_loops=True,  # Assuming you have the add_self_loops function implemented
    )

    # Ensure fsa is a valid k2 FSA
    assert isinstance(fsa, k2.Fsa)


def test_prepare_lang():
    # Step 1: Setup
    temp_dir = tempfile.mkdtemp()

    # Create a simple lexicon for testing
    lexicon_content = """
    hello h e l l o
    world w o r l d
    """
    with open(
        os.path.join(temp_dir, "lexicon.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(lexicon_content.strip())

    # Step 2: Run prepare_lang
    from speechbrain.k2_integration.prepare_lang import prepare_lang

    prepare_lang(temp_dir, sil_token="SIL", sil_prob=0.5)

    # Step 3: Check the output
    # Check if the expected files are present
    for expected_file in [
        "tokens.txt",
        "words.txt",
        "L.pt",
        "L_disambig.pt",
        "Linv.pt",
    ]:
        assert os.path.exists(os.path.join(temp_dir, expected_file))

    # Step 4: Cleanup
    shutil.rmtree(temp_dir)


def test_lexicon_loading_and_conversion():
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a small lexicon containing only two words.
        lexicon_sample = """<UNK> <unk>
hello h e l l o
world w o r l d"""
        lexicon_file = tmpdir_path.joinpath("lexicon.txt")
        with open(lexicon_file, "w", encoding="utf-8") as f:
            f.write(lexicon_sample)

        # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt using prepare_lang
        from speechbrain.k2_integration.prepare_lang import prepare_lang

        prepare_lang(tmpdir_path)

        # Create a lexicon object
        from speechbrain.k2_integration.lexicon import Lexicon

        lexicon = Lexicon(tmpdir_path)

        # Assert instance types
        assert isinstance(lexicon.token_table, k2.SymbolTable)
        assert isinstance(lexicon.word_table, k2.SymbolTable)
        assert isinstance(lexicon.L, k2.Fsa)

        # Test conversion from texts to token IDs
        hello_tids = lexicon.word_table["hello"]
        world_tids = lexicon.word_table["world"]
        expected_tids = [hello_tids] + [world_tids]
        assert lexicon.texts_to_word_ids(["hello world"])[0] == expected_tids

        # Test out-of-vocabulary words
        # Assuming that <UNK> exists in the tokens:
        unk_tid = lexicon.word_table["<UNK>"]
        hello_tids = lexicon.word_table["hello"]
        expected_oov_tids = [hello_tids] + [unk_tid]
        assert (
            lexicon.texts_to_word_ids(["hello universe"])[0]
            == expected_oov_tids
        )

        # Test with sil_token as separator
        # Assuming that SIL exists in the tokens:
        sil_tid = lexicon.token_table["SIL"]
        hello_tids = lexicon.word_table["hello"]
        world_tids = lexicon.word_table["world"]
        expected_sil_tids = [hello_tids] + [sil_tid] + [world_tids]
        assert (
            lexicon.texts_to_word_ids(
                ["hello world"],
                add_sil_token_as_separator=True,
                sil_token_id=sil_tid,
            )[0]
            == expected_sil_tids
        )


def test_ctc_k2_loss():
    # Create a random batch of log-probs
    batch_size = 4
    log_probs = torch.randn(batch_size, 100, 30).requires_grad_(True)
    log_probs = torch.nn.functional.log_softmax(log_probs, dim=-1)
    input_lens = torch.tensor([1, 0.9, 0.8, 0.7])

    # Create a temporary directory for lexicon and other files
    with TemporaryDirectory() as tmpdir:
        # Create a small lexicon containing only two words and write it to a file.
        lexicon_sample = """<UNK> <unk>
hello h e l l o
world w o r l d"""
        lexicon_file_path = f"{tmpdir}/lexicon.txt"
        with open(lexicon_file_path, "w", encoding="utf-8") as f:
            f.write(lexicon_sample)

        # Create a lang directory with the lexicon and L.pt, L_inv.pt, L_disambig.pt
        from speechbrain.k2_integration.prepare_lang import prepare_lang

        prepare_lang(tmpdir)

        # Create a lexicon object
        from speechbrain.k2_integration.lexicon import Lexicon

        lexicon = Lexicon(tmpdir)

        # Create a graph compiler
        from speechbrain.k2_integration.graph_compiler import CtcGraphCompiler

        graph_compiler = CtcGraphCompiler(
            lexicon,
            device=log_probs.device,
        )

        # Create a random batch of texts
        texts = ["hello world", "world hello", "hello", "world"]

        # Compute the loss
        from speechbrain.k2_integration.losses import ctc_k2

        loss = ctc_k2(
            log_probs=log_probs,
            input_lens=input_lens,
            graph_compiler=graph_compiler,
            texts=texts,
            reduction="mean",
            beam_size=10,
            use_double_scores=True,
            is_training=True,
        )

        # Assertions
        assert loss.requires_grad
        assert loss.item() >= 0  # Loss should be non-negative
