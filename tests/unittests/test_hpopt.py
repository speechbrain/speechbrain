import pytest


def test_hpopt_generic():
    import json
    from io import StringIO

    from speechbrain.utils import hpopt as hp

    output = StringIO()

    reporter = hp.GenericHyperparameterOptimizationReporter(
        objective_key="per", output=output
    )
    result = {"train_loss": 0.9, "valid_loss": 1.2, "per": 0.10}
    reporter.report_objective(result)
    output.seek(0)
    output_result = json.load(output)
    assert output_result["train_loss"] == pytest.approx(0.9)
    assert output_result["valid_loss"] == pytest.approx(1.2)
    assert output_result["per"] == pytest.approx(0.10)
    assert output_result["objective"] == pytest.approx(0.10)


def test_hpopt_orion():
    from speechbrain.utils import hpopt as hp

    results = {}

    class MockOrion:
        def report_objective(self, value):
            results["value"] = value

    mock_orion = MockOrion()

    reporter = hp.OrionHyperparameterOptimizationReporter(
        objective_key="valid_loss"
    )
    reporter.orion_client = mock_orion

    result = {"train_loss": 0.9, "valid_loss": 1.2, "per": 0.10}
    reporter.report_objective(result)
    assert results["value"] == pytest.approx(1.2)


def test_hpopt_context():
    import json
    from io import StringIO

    from speechbrain.utils import hpopt as hp

    output = StringIO()
    reporter = hp.GenericHyperparameterOptimizationReporter(
        objective_key="per", output=output
    )

    with hp.hyperparameter_optimization() as hp_ctx:
        hp_ctx.reporter = reporter
        result = {"per": 10, "loss": 1.2}
        hp.report_result(result)

        result = {"per": 3, "loss": 1.3}
        hp.report_result(result)

    output.seek(0)
    output_result = json.load(output)
    assert output_result["per"] == 3
    assert hp.get_trial_id() == hp.DEFAULT_TRIAL_ID


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("objective_key", ["error", "objective"])
def test_hpopt_context_without_report(enabled, objective_key):
    from io import StringIO

    from speechbrain.utils import hpopt as hp

    output = StringIO()
    with hp.hyperparameter_optimization(
        objective_key=objective_key, output=output
    ) as hp_ctx:
        args = ["hparams.yaml"] + (["--hpopt", "True"] if enabled else [])
        hp_ctx.parse_arguments(args)

    assert output.getvalue() == ""
    assert hp.get_trial_id() == hp.DEFAULT_TRIAL_ID


@pytest.mark.parametrize("enabled", [False, True])
def test_hpopt_context_reports_zero(enabled):
    import json
    from io import StringIO

    from speechbrain.utils import hpopt as hp

    output = StringIO()
    with hp.hyperparameter_optimization(
        objective_key="error", output=output
    ) as hp_ctx:
        args = ["hparams.yaml"] + (["--hpopt", "True"] if enabled else [])
        hp_ctx.parse_arguments(args)
        hp.report_result({"error": 0.0})

    assert json.loads(output.getvalue()) == {"error": 0.0, "objective": 0.0}
    assert hp.get_trial_id() == hp.DEFAULT_TRIAL_ID
