import numpy
from numpy import arccos, sqrt, fabs
from numpy import array
from numpy import dot, cross
from numpy.linalg import det
from scipy.interpolate import splrep, splev


def norm(x):
    """Returns the magnitude of the vector x."""
    return sqrt(numpy.sum(array(x) ** 2, axis=-1))


def point_line_distance(x0, x1, x2):
    """Finds the shortest distance between the point x0 and the line x1 to x2.
    Ref: http://mathworld.wolfram.com/Point-LineDistance3-Dimensional.html"""
    assert x1.shape == x2.shape == (2,)
    return fabs(cross(x0 - x1, x0 - x2)) / norm(x2 - x1)


def angle(x0, x1, x2):
    """Returns angle between three points.
    Ref: https://stackoverflow.com/questions/1211212"""
    assert x1.shape == x2.shape == (2,)
    a, b = x1 - x0, x1 - x2
    return arccos(dot(a, b) / (norm(a) * norm(b)))


def is_left(x0, x1, x2):
    """Returns True if x0 is left of the line between x1 and x2,
    False otherwise.  Ref: https://stackoverflow.com/questions/1560492"""
    assert x1.shape == x2.shape == (2,)
    matrix = array([x1 - x0, x2 - x0])
    if len(x0.shape) == 2:
        matrix = matrix.transpose((1, 2, 0))
    return det(matrix) > 0


def lininterp2(x1, y1, x):
    """Linear interpolation at points x between numpy arrays (x1, y1).
    Only y1 is allowed to be two-dimensional.  The x1 values should be sorted
    from low to high.  Returns a numpy.array of y values corresponding to
    points x.
    """
    return splev(x, splrep(x1, y1, s=0, k=1))
