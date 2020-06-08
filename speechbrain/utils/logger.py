"""
Managing the logger.

Author
    Fang-Pen Lin 2012 and Peter Plantinga 2020
    https://fangpenlin.com/posts/2012/08/26/good-logging-practice-in-python/
    Aku Rouhe
"""

import os
import yaml
import logging
import logging.config
import math
from speechbrain.utils.data_utils import recursive_update

ORDERS_ABBREV = {
    -24: "y",
    -21: "z",
    -18: "a",
    -15: "f",
    -12: "p",
    -9: "n",
    -6: "µ",
    -3: "m",
    0: "",
    3: "k",
    6: "M",
    9: "G",
    12: "T",
    15: "P",
    18: "E",
    21: "Z",
    24: "Y",
}

# Short scale
# Negative powers of ten in lowercase, positive in uppercase
ORDERS_WORDS = {
    -24: "septillionth",
    -21: "sextillionth",
    -18: "quintillionth",
    -15: "quadrillionth",
    -12: "trillionth",
    -9: "billionth",
    -6: "millionth",
    -3: "thousdanth",
    0: "",
    3: "Thousand",
    6: "Million",
    9: "Billion",
    12: "Trillion",
    15: "Quadrillion",
    18: "Quintillion",
    21: "Sextillion",
    24: "Septillion",
}


def setup_logging(
    config_path="log-config.yaml", overrides={}, default_level=logging.INFO,
):
    """Setup logging configuration

    Arguments
    ---------
    config_path : str
        the path to a logging config file
    default_level : int
        the level to use if config file is not found
    overrides : dict
        a dictionary of the same structure as the config dict
        with any updated values that need to be applied
    """
    if os.path.exists(config_path):
        with open(config_path, "rt") as f:
            config = yaml.safe_load(f)
        recursive_update(config, overrides)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)


def format_order_of_magnitude(number, style=ORDERS_ABBREV):
    """
    Formats number to appropriate order of magnitude for printing

    Arguments
    ---------
    number : int, float
        The number to format
    style : dict
        A mapping from order of magnitude (power of 3) to name / abbreviation.
        By default, SI prefixes.

    Returns
    -------
    str
        The reformatted number
    str
        The order of magnitude name / abbreviation

    Example
    -------
    >>> num, mag = format_order_of_magnitude(123456.7)
    >>> print(num+mag)
    123.5k
    """
    precision = "{num:3.1f}"
    order = 3 * math.floor(math.log(math.fabs(number), 1000))
    # Fallback for very large numbers:
    while order not in ORDERS_ABBREV and order != 0:
        order = order - math.copysign(3, order)  # Bring 3 units towards 0
    order_token = ORDERS_ABBREV[order]
    if order != 0:
        formatted_number = precision.format(num=number / 10 ** order)
    else:
        if isinstance(number, int):
            formatted_number = str(number)
        else:
            formatted_number = precision.format(num=number)
    return formatted_number, order_token
