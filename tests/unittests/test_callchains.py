def test_lengths_arg_exists():
    from speechbrain.utils.callchains import lengths_arg_exists

    def non_len_func(x):
        return x + 1

    def len_func(x, lengths):
        return x + lengths

    assert not lengths_arg_exists(non_len_func)
    assert lengths_arg_exists(len_func)


def test_lengths_arg_name():
    from speechbrain.utils.callchains import lengths_arg_name

    def non_len_func(x):
        return x + 1

    def len_func(x, lengths):
        return x + lengths

    def wav_lens_func(x, wav_lens=None):
        return x + wav_lens

    assert lengths_arg_name(non_len_func) is None
    assert lengths_arg_name(len_func) == "lengths"
    assert lengths_arg_name(wav_lens_func) == "wav_lens"


def test_lengths_capable_chain():
    from speechbrain.utils.callchains import LengthsCapableChain

    def non_len_func(x):
        return x + 1

    def len_func(x, lengths):
        return x + lengths

    def tuple_func(x):
        return x, x + 1

    chain = LengthsCapableChain(non_len_func, len_func)
    assert chain(1, 2) == 4
    assert chain(lengths=2, x=1) == 4
    chain.append(non_len_func)
    assert chain(1, 2) == 5
    chain.append(tuple_func)
    assert chain(1, 2) == 5


def test_lengths_capable_chain_wav_lens():
    from speechbrain.utils.callchains import LengthsCapableChain

    def wav_lens_func(x, wav_lens=None):
        assert wav_lens is not None
        return x + wav_lens

    chain = LengthsCapableChain(wav_lens_func)
    assert chain(1, 2) == 3
