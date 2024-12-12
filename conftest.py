def pytest_addoption(parser):
    parser.addoption("--device", action="store", default="cpu")


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.device
    if "device" in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("device", [option_value])


collect_ignore = ["setup.py", "speechbrain/integrations/"]
try:
    import numba  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/nnet/loss/transducer_loss.py")
try:
    import kenlm  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/decoders/language_model.py")
try:
    import sklearn  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/kmeans.py")
    collect_ignore.append(
        "speechbrain/integrations/huggingface/discrete_ssl.py"
    )
try:
    import sacrebleu  # noqa: F401
except ModuleNotFoundError:
    collect_ignore.append("speechbrain/utils/bleu.py")
