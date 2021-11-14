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
import sys

from datetime import datetime
from hyperpyyaml import load_hyperpyyaml


logger = logging.getLogger(__name__)

MODULE_ORION = "orion.client"
FORMAT_TIMESTAMP = "%Y%m%d%H%M%S%f"
DEFAULT_TRIAL_ID = "hpfit"
DEFAULT_REPORTER = "generic"
ORION_TRIAL_ID_ENV = [
    "ORION_EXPERIMENT_NAME",
    "ORION_EXPERIMENT_VERSION",
    "ORION_TRIAL_ID",
]
KEY_HPFIT = "hpfit"
KEY_HPFIT_MODE = "hpfit_mode"

_hpfit_modes = {}


def hpfit_mode(mode):
    """A decorator to register a reporter implementation for
    a hyperparameter fitting mode

    Arguments
    ---------
    mode: str
        the mode to register

    Returns
    -------
    f: callable
        a callable function that registers and returns the
        reporter class
    """

    def f(cls):
        _hpfit_modes[mode] = cls
        return cls

    return f


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

    @property
    def is_available(self):
        """Determines whether this reporter is available"""
        return True

    @property
    def trial_id(self):
        """The unique ID of this trial (used for folder naming)"""
        return DEFAULT_TRIAL_ID


@hpfit_mode("generic")
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
        self.output = output or sys.stdout
        self._trial_id = None

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

    @property
    def trial_id(self):
        """The unique ID of this trial (used mainly for folder naming)"""
        if self._trial_id is None:
            self._trial_id = datetime.now().strftime(FORMAT_TIMESTAMP)
        return self._trial_id


@hpfit_mode("orion")
class OrionHyperparameterFitReporter(HyperparameterFitReporter):
    """A result reporter implementation based on Orion

    Arguments
    ---------
    orion_client: module
        the Python module for Orion
    """

    def __init__(self, objective_key):
        super().__init__(objective_key=objective_key)
        self.orion_client = None
        self._trial_id = None
        self._check_client()

    def _check_client(self):
        try:
            self.orion_client = importlib.import_module(MODULE_ORION)
        except ImportError:
            logger.warn("Orion is not available")
            self.orion_client = None

    def _format_message(self, result):
        """Formats the log message for output

        Arguments
        ---------
        result: dict
            the result dictionary

        Returns
        -------
        message: str
            a formatted message"""
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
        if self.orion_client is not None:
            objective_value = result[self.objective_key]
            self.orion_client.report_objective(objective_value)

    @property
    def trial_id(self):
        """The unique ID of this trial (used mainly for folder naming)"""
        if self._trial_id is None:
            self._trial_id = "-".join(
                os.getenv(name) or "" for name in ORION_TRIAL_ID_ENV
            )
        return self._trial_id

    @property
    def is_available(self):
        """Determines if Orion is available. In order for it to
        be available, the library needs to be installed, and at
        least one of ORION_EXPERIMENT_NAME, ORION_EXPERIMENT_VERSION,
        ORION_TRIAL_ID needs to be set"""
        return self.orion_client is not None and any(
            os.getenv(name) for name in ORION_TRIAL_ID_ENV
        )


def get_reporter(mode, *args, **kwargs):
    """Attempts to get the reporter specified by the mode
    and reverts to a generic one if it is not available

    Arguments
    ---------
    mode: str
        a string identifier for a registered hyperparametr
        fitting mode, corresponding to a specific reporter
        instance

    Returns
    -------
    reporter: HyperparameterFitReporter
        a reporter instance
    """
    reporter_cls = _hpfit_modes.get(mode)
    if reporter_cls is None:
        logger.warn(f"hpfit_mode {mode} is not supported, reverting to generic")
        reporter_cls = _hpfit_modes[DEFAULT_REPORTER]
    reporter = reporter_cls(*args, **kwargs)
    if not reporter.is_available:
        logger.warn("Reverting to a generic reporter")
        reporter_cls = _hpfit_modes[DEFAULT_REPORTER]
        reporter = reporter_cls(*args, **kwargs)
    return reporter


_context = {"current": None}


class HyperparameterFittingContext:
    """
    A convenience context manager that makes it possible to conditionally
    enable hyperparameter fitting for a recipe.

    Arguments
    ---------
    reporter_args: list
        arguments to the reporter class
    reporter_kwargs: dict
        keyword arguments to the reporter class
    """

    def __init__(self, reporter_args, reporter_kwargs):
        self.reporter_args = reporter_args
        self.reporter_kwargs = reporter_kwargs
        self.reporter = None
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

        Arguments
        ---------
        arg_list: a list of arguments

        Returns
        -------
        param_file : str
            The location of the parameters file.
        run_opts : dict
            Run options, such as distributed, device, etc.
        overrides : dict
            The overrides to pass to ``load_hyperpyyaml``.
        """
        hparams_file, run_opts, overrides_yaml = sb.parse_arguments(arg_list)
        overrides = load_hyperpyyaml(overrides_yaml)
        hpfit = overrides.get(KEY_HPFIT, False)
        hpfit_mode = overrides.get(KEY_HPFIT_MODE) or DEFAULT_REPORTER
        if hpfit:
            self.enabled = True
            self.reporter = get_reporter(
                hpfit_mode, *self.reporter_args, **self.reporter_kwargs
            )
            if isinstance(hpfit, str) and os.path.exists(hpfit):
                with open(hpfit) as hpfit_file:
                    trial_id = get_trial_id()
                    hpfit_overrides = load_hyperpyyaml(
                        hpfit_file,
                        overrides={"trial_id": trial_id},
                        overrides_must_match=False,
                    )
                    overrides = dict(hpfit_overrides, **overrides)
                    for key in [KEY_HPFIT, KEY_HPFIT_MODE]:
                        if key in overrides:
                            del overrides[key]
        return hparams_file, run_opts, overrides

    def __enter__(self):
        _context["current"] = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if (
            exc_type is None
            and self.result is not None
            and self.reporter is not None
        ):
            self.reporter.report_objective(self.result)
        _context["current"] = None


def hyperparameter_fitting(*args, **kwargs):
    """Initializes the hyperparameter fitting context"""
    hpfit = HyperparameterFittingContext(args, kwargs)
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


def get_trial_id():
    """
    Returns the ID of the current hyperparameter fitting trial,
    used primarily for the name of experiment folders

    Returns
    -------
    trial_id: str
        the trial identifier
    """
    ctx = _context["current"]
    trial_id = ctx.reporter.trial_id if ctx else DEFAULT_TRIAL_ID
    return trial_id
