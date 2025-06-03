import hyperpyyaml
import pytest


@pytest.mark.filterwarnings(
    "ignore:Module 'speechbrain.pretrained' was deprecated"
)
def test_lazy_pretrained_hparams():
    """Test the lazy redirection for `pretrained` through a YAML to ensure that
    `hyperpyyaml`'s magic does not break there"""

    yaml = hyperpyyaml.load_hyperpyyaml(
        """\
test_pretrained: !name:speechbrain.pretrained.interfaces.Pretrained
"""
    )
    assert yaml["test_pretrained"] is not None
