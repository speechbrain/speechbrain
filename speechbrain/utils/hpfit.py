"""Utilities for hyperparameter fitting.
This wrapper has an optional dependency on
Oríon

https://orion.readthedocs.io/en/stable/
https://github.com/Epistimio/orion

Authors
 * Artem Ploujnikov 2021
"""
import importlib
import logging
import json
import os
import speechbrain as sb

from hyperpyyaml import load_hyperpyyaml


logger = logging.getLogger(__name__)

MODUE_ORION = "orion.client"


class HyperparameterFitReporter:
    """A base class for hyperparameter fit reporters

    Arguments
    ---------
    objective_key: str
        the key from the result dictionary to be used as the objective
    """

    def __init__(self, objective_key):
        self.objective_key = objective_key

    def report_objective(self, result):
        """Reports the objective for hyperparameter fitting.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.
        """
        return NotImplemented


class GenericHyperparameterFitReporter(HyperparameterFitReporter):
    """
    A generic hyperparameter fit reporter that outputs the result as
    JSON to an arbitrary data stream, which may be read as a third-party
    tool

    Arguments
    ---------
    objective_key: str
        the key from the result dictionary to be used as the objective

    """

    def __init__(self, output=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.output = output

    def report_objective(self, result):
        """Reports the objective for hyperparameter fitting.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.
        """
        json.dump(
            dict(result, objective=result[self.objective_key]), self.output
        )


class OrionHyperparameterFitReporter(HyperparameterFitReporter):
    """A result reporter implementation based on Orion

    Arguments
    ---------
    orion_client: module
        the Python module for Orion
    """

    def __init__(self, orion_client, objective_key):
        super().__init__(objective_key=objective_key)
        self.orion_client = orion_client

    def _format_message(self, result):
        return ", ".join(f"{key} = {value}" for key, value in result.items())

    def report_objective(self, result):
        """Reports the objective for hyperparameter fitting.

        Arguments
        ---------
        result: dict
            a dictionary with the run result.
        """
        message = self._format_message(result)
        logger.info(f"Hyperparameter fit: {message}")
        objective_value = result[self.objective_key]
        self.orion_client.report_objective(objective_value)


def get_reporter(*args, **kwargs):
    """Gets the reporter appropriate for the system.
    """
    try:
        client = importlib.import_module(MODUE_ORION)
        reporter = OrionHyperparameterFitReporter(client, *args, **kwargs)
    except ImportError:
        logger.info("Orion is not available, using a generic reporter")
        reporter = GenericHyperparameterFitReporter(*args, **kwargs)
    return reporter


_context = {"current": None}


class HyperparameterFittingContext:
    """
    A convenience context manager that makes it possible to conditionally
    enable hyperparameter fitting for a recipe.
    """

    def __init__(self, reporter):
        self.reporter = reporter
        self.enabled = False
        self.result = {"objective": 0.0}

    def parse_arguments(self, arg_list):
        """A version of speechbrain.parse_arguments enhanced for hyperparameter
        fitting.

        If a parameter named 'hpfit' is provided, hyperparameter
        fitting and reporting will be enabled.

        If the parameter value corresponds to a filename, it will
        be read as a hyperpyaml file, and the contents will be added
        to "overrides". This is useful for cases where the values of
        certain hyperparameters are different during hyperparameter
        fitting vs during full training (e.g. number of epochs, saving
        files, etc)
        """
        hparams_file, run_opts, overrides_yaml = sb.parse_arguments(arg_list)
        overrides = load_hyperpyyaml(overrides_yaml)
        hpfit = overrides.get("hpfit", False)
        if hpfit:
            self.enabled = True
            if isinstance(hpfit, str) and os.path.exists(hpfit):
                with open(hpfit) as hpfit_file:
                    hpfit_overrides = load_hyperpyyaml(hpfit_file)
                    overrides.update(hpfit_overrides)

        return hparams_file, run_opts, overrides

    def __enter__(self):
        _context["current"] = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.result is not None:
            self.reporter.report_objective(self.result)
        _context["current"] = None


def hyperparameter_fitting(reporter=None, *args, **kwargs):
    """Initializes the hyperparameter fitting context"""
    if reporter is None:
        reporter = get_reporter(*args, **kwargs)
    hpfit = HyperparameterFittingContext(reporter=reporter)
    return hpfit


def report_result(result):
    """Reports the result using the current reporter, if available.
    When not in hyperparameter fitting mode, this function does nothing.

    Arguments
    ---------
    result: dict
        A dictionary of stats to be reported
    """
    ctx = _context["current"]
    if ctx:
        ctx.result = result
