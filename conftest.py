def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


collect_ignore = ["setup.py"]
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/loss/transducer_loss.py")
try:
    import fairseq  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/fairseq_wav2vec.py")
try:
    from transformers import Wav2Vec2Model  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/huggingface_wav2vec.py")
try:
    from transformers import WhisperModel  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/lobes/models/huggingface_whisper.py")
try:
    import sacrebleu  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/bleu.py")
