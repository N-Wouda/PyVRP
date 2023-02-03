from numpy.testing import assert_equal, assert_raises

from pyvrp.Matrix import IntMatrix


def test_dimension_constructor():
    square = IntMatrix(10)
    assert_equal(square.size(), 10 * 10)
    assert_equal(square.max(), 0)  # matrix initialises all zero

    rectangle = IntMatrix(10, 20)
    assert_equal(rectangle.size(), 10 * 20)
    assert_equal(rectangle.max(), 0)  # matrix initialises all zero


def test_data_constructor():
    empty = IntMatrix([[], []])
    assert_equal(empty.size(), 0)

    non_empty = IntMatrix([[1, 2], [1, 3]])
    assert_equal(non_empty.size(), 4)  # matrix has four elements
    assert_equal(non_empty.max(), 3)


def test_data_constructor_throws():
    with assert_raises(ValueError):
        IntMatrix([[1, 2], []])  # second row shorter than first

    with assert_raises(ValueError):
        IntMatrix([[1, 2], [1, 2, 3]])  # second row longer than first


def test_element_access():
    mat = IntMatrix(10)

    for i in range(10):
        for j in range(10):
            mat[i, j] = i + j

    assert_equal(mat.max(), 9 + 9)  # maximum value

    assert_equal(mat[1, 1], 1 + 1)  # several elements
    assert_equal(mat[2, 1], 2 + 1)
    assert_equal(mat[1, 2], 1 + 2)
