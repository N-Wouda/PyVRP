__doc__ = """
The tests in this file check that Python objects convertible to integers are
handled appropriately on the C++ side - this includes checking for potential
overflows and meaningful responses in that case.
"""

import numpy as np
import pytest
from numpy.testing import assert_equal, assert_raises

from pyvrp import Client


@pytest.mark.parametrize(
    ("input", "expected"),
    [
        (1, 1),
        (0, 0),
        ("1", 1),
        (1.0, 1),
        (1.5, 1),
        (2**31 - 1, 2**31 - 1),
        (2**31, 2**31),
        (2**63 - 1, 2**63 - 1),  # largest supported 64-bit signed value
    ],
)
def test_measure_convertible_to_int(input, expected):
    """
    Tests that input argument for which int() succeeds are also properly
    handled on the C++ side.
    """
    client = Client(1, 1, tw_late=input)
    assert_equal(client.tw_late, expected)


@pytest.mark.parametrize(
    "input", [np.iinfo(np.int64).max + 1, np.finfo(np.float64).max]
)
def test_larger_than_max_size(input):
    """
    Tests that arguments larger than the maximum value for measure arguments
    raise an overflow error to warn the user.
    """
    with assert_raises(OverflowError):
        Client(1, 1, tw_late=input)
