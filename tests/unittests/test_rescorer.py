def test_rnnlmrescorer(tmpdir, device):
    import torch
    from sentencepiece import SentencePieceProcessor

    from speechbrain.lobes.models.RNNLM import RNNLM
    from speechbrain.utils.parameter_transfer import Pretrainer

    source = "speechbrain/asr-crdnn-rnnlm-librispeech"
    lm_model_path = source + "/lm.ckpt"
    tokenizer_path = source + "/tokenizer.ckpt"

    # Define your tokenizer and RNNLM from the HF hub
    tokenizer = SentencePieceProcessor()
    lm_model = RNNLM(
        output_neurons=1000,
        embedding_dim=128,
        activation=torch.nn.LeakyReLU,
        dropout=0.0,
        rnn_layers=2,
        rnn_neurons=2048,
        dnn_blocks=1,
        dnn_neurons=512,
        return_hidden=True,
    )

    pretrainer = Pretrainer(
        collect_in=tmpdir,
        loadables={"lm": lm_model, "tokenizer": tokenizer},
        paths={"lm": lm_model_path, "tokenizer": tokenizer_path},
    )

    pretrainer.collect_files()
    pretrainer.load_collected()

    from speechbrain.decoders.scorer import RescorerBuilder, RNNLMRescorer

    rnnlm_rescorer = RNNLMRescorer(
        language_model=lm_model,
        tokenizer=tokenizer,
        temperature=1.0,
        bos_index=0,
        eos_index=0,
        pad_index=0,
    )

    # Define a rescorer builder
    rescorer = RescorerBuilder(
        rescorers=[rnnlm_rescorer], weights={"rnnlm": 1.0}
    )

    # Topk hypotheses
    topk_hyps = [["HELLO", "HE LLO", "H E L L O"]]
    topk_scores = [[-2, -2, -2]]
    rescored_hyps, rescored_scores = rescorer.rescore(topk_hyps, topk_scores)

    # check all hyps are still there
    for hyp in topk_hyps[0]:
        assert hyp in rescored_hyps[0]

    # check rescored_scores are sorted
    for i in range(len(rescored_scores[0]) - 1):
        assert rescored_scores[0][i] >= rescored_scores[0][i + 1]

    # check normalized_text is working
    text = "hello"
    normalized_text = rnnlm_rescorer.normalize_text(text)
    assert normalized_text == text.upper()

    # check lm is on the right device
    rnnlm_rescorer.to_device(device)
    assert rnnlm_rescorer.lm.parameters().__next__().device.type == device

    # check preprocess_func
    padded_hyps, enc_hyps_length = rnnlm_rescorer.preprocess_func(topk_hyps)
    assert padded_hyps.shape[0] == 3
    assert len(padded_hyps) == 3


def test_transformerlmrescorer(tmpdir, device):
    import torch
    from sentencepiece import SentencePieceProcessor

    from speechbrain.lobes.models.transformer.TransformerLM import TransformerLM
    from speechbrain.utils.parameter_transfer import Pretrainer

    source = "speechbrain/asr-transformer-transformerlm-librispeech"
    lm_model_path = source + "/lm.ckpt"
    tokenizer_path = source + "/tokenizer.ckpt"
    tokenizer = SentencePieceProcessor()

    lm_model = TransformerLM(
        vocab=5000,
        d_model=768,
        nhead=12,
        num_encoder_layers=12,
        num_decoder_layers=0,
        d_ffn=3072,
        dropout=0.0,
        activation=torch.nn.GELU,
        normalize_before=False,
    )

    pretrainer = Pretrainer(
        collect_in=tmpdir,
        loadables={"lm": lm_model, "tokenizer": tokenizer},
        paths={"lm": lm_model_path, "tokenizer": tokenizer_path},
    )

    _ = pretrainer.collect_files()
    pretrainer.load_collected()

    from speechbrain.decoders.scorer import (
        RescorerBuilder,
        TransformerLMRescorer,
    )

    transformerlm_rescorer = TransformerLMRescorer(
        language_model=lm_model,
        tokenizer=tokenizer,
        temperature=1.0,
        bos_index=1,
        eos_index=2,
        pad_index=0,
    )

    rescorer = RescorerBuilder(
        rescorers=[transformerlm_rescorer], weights={"transformerlm": 1.0}
    )

    # Topk hypotheses
    topk_hyps = [["HELLO", "HE LLO", "H E L L O"]]
    topk_scores = [[-2, -2, -2]]
    rescored_hyps, rescored_scores = rescorer.rescore(topk_hyps, topk_scores)

    # check all hyps are still there
    for hyp in topk_hyps[0]:
        assert hyp in rescored_hyps[0]

    # check rescored_scores are sorted
    for i in range(len(rescored_scores[0]) - 1):
        assert rescored_scores[0][i] >= rescored_scores[0][i + 1]

    # check normalized_text is working
    text = "hello"
    normalized_text = transformerlm_rescorer.normalize_text(text)
    assert normalized_text == text.upper()

    # check lm is on the right device
    transformerlm_rescorer.to_device(device)
    assert (
        transformerlm_rescorer.lm.parameters().__next__().device.type == device
    )

    # check preprocess_func
    padded_hyps, enc_hyps_length = transformerlm_rescorer.preprocess_func(
        topk_hyps
    )
    assert padded_hyps.shape[0] == 3
    assert len(padded_hyps) == 3


def test_huggingfacelmrescorer(device):
    from speechbrain.decoders.scorer import (
        HuggingFaceLMRescorer,
        RescorerBuilder,
    )

    source = "gpt2-medium"

    huggingfacelm_rescorer = HuggingFaceLMRescorer(model_name=source)

    rescorer = RescorerBuilder(
        rescorers=[huggingfacelm_rescorer], weights={"huggingfacelm": 1.0}
    )

    # Topk hypotheses
    topk_hyps = [["HELLO", "HE LLO", "H E L L O"]]
    topk_scores = [[-2, -2, -2]]
    rescored_hyps, rescored_scores = rescorer.rescore(topk_hyps, topk_scores)

    # check all hyps are still there
    for hyp in topk_hyps[0]:
        assert hyp in rescored_hyps[0]

    # check rescored_scores are sorted
    for i in range(len(rescored_scores[0]) - 1):
        assert rescored_scores[0][i] >= rescored_scores[0][i + 1]

    # check normalized_text is working
    text = "hello"
    normalized_text = huggingfacelm_rescorer.normalize_text(text)
    assert normalized_text == text

    # check lm is on the right device
    huggingfacelm_rescorer.to_device(device)
    assert huggingfacelm_rescorer.lm.device.type == device

    # check preprocess_func
    padded_hyps = huggingfacelm_rescorer.preprocess_func(topk_hyps)
    assert padded_hyps.input_ids.shape[0] == 3


def test_rescorerbuilder(tmpdir, device):
    import torch
    from sentencepiece import SentencePieceProcessor

    from speechbrain.lobes.models.RNNLM import RNNLM
    from speechbrain.utils.parameter_transfer import Pretrainer

    source = "speechbrain/asr-crdnn-rnnlm-librispeech"
    lm_model_path = source + "/lm.ckpt"
    tokenizer_path = source + "/tokenizer.ckpt"

    # Define your tokenizer and RNNLM from the HF hub
    tokenizer = SentencePieceProcessor()
    lm_model = RNNLM(
        output_neurons=1000,
        embedding_dim=128,
        activation=torch.nn.LeakyReLU,
        dropout=0.0,
        rnn_layers=2,
        rnn_neurons=2048,
        dnn_blocks=1,
        dnn_neurons=512,
        return_hidden=True,
    )

    pretrainer = Pretrainer(
        collect_in=tmpdir,
        loadables={"lm": lm_model, "tokenizer": tokenizer},
        paths={"lm": lm_model_path, "tokenizer": tokenizer_path},
    )

    pretrainer.collect_files()
    pretrainer.load_collected()

    from speechbrain.decoders.scorer import (
        HuggingFaceLMRescorer,
        RescorerBuilder,
        RNNLMRescorer,
    )

    rnnlm_rescorer = RNNLMRescorer(
        language_model=lm_model,
        tokenizer=tokenizer,
        temperature=1.0,
        bos_index=0,
        eos_index=0,
        pad_index=0,
    )

    source = "gpt2-medium"

    huggingfacelm_rescorer = HuggingFaceLMRescorer(model_name=source)

    # check combine both rescorers
    rescorer = RescorerBuilder(
        rescorers=[rnnlm_rescorer, huggingfacelm_rescorer],
        weights={"rnnlm": 1.0, "huggingfacelm": 1.0},
    )
    rescorer.move_rescorers_to_device(device)
    # check lm is on the right device
    assert rnnlm_rescorer.lm.parameters().__next__().device.type == device
    assert huggingfacelm_rescorer.lm.device.type == device

    # Topk hypotheses
    topk_hyps = [["HELLO", "HE LLO", "H E L L O"]]
    topk_scores = [[-2, -2, -2]]
    rescored_hyps, rescored_scores = rescorer.rescore(topk_hyps, topk_scores)

    # check all hyps are still there
    for hyp in topk_hyps[0]:
        assert hyp in rescored_hyps[0]

    # check rescored_scores are sorted
    for i in range(len(rescored_scores[0]) - 1):
        assert rescored_scores[0][i] >= rescored_scores[0][i + 1]
