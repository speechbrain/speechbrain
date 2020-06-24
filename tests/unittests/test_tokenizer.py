


def test_tokenizer():
    from speechbrain.tokenizers.BPE import BPE
    def int2lab(sequence):
        dict_int2lab={1:"hello", 2:"morning", 3:" "}
        output=""
        for id_lab in sequence:
            output+=dict_int2lab[id_lab]
        return output
    bpe=BPE("tokenizer_data/", 2000, text_file="tokenizer_data/botchan.txt", model_type="bpe")
    encoded_seq_ids, encoded_seq_pieces = bpe([[1, 3, 2, 3, 1, 3, 2],[2, 3, 2, 3, 2]], int2lab, task="encode", init_params=True)
    print(encoded_seq_ids)
    print(encoded_seq_pieces)
    decode_sequence = bpe(encoded_seq_ids, task="decode")
    print(decode_sequence)


test_tokenizer()
