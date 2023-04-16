import numpy
from matplotlib import pyplot
from numpy import array
from numpy import sqrt, fabs

from charges.custom_math import norm


class PointCharge:
    """A point charge."""

    R = 0.01  # The effective radius of the charge

    def __init__(self, q, x):
        """Initializes the quantity of charge 'q' and position vector 'x'."""
        self.q, self.x = q, array(x)

    def E(self, x):  # pylint: disable=invalid-name
        """Electric field vector."""
        if self.q == 0:
            return 0
        dx = x - self.x
        return (self.q * dx.T / numpy.sum(dx ** 2, axis=-1) ** 1.5).T

    def V(self, x):  # pylint: disable=invalid-name
        """Potential."""
        return self.q / norm(x - self.x)

    def is_close(self, x):
        """Returns True if x is close to the charge; false otherwise."""
        return norm(x - self.x) < self.R

    def plot(self):
        """Plots the charge."""
        color = 'b' if self.q < 0 else 'r' if self.q > 0 else 'k'
        r = 0.1 * (sqrt(fabs(self.q)) / 2 + 1)
        circle = pyplot.Circle(self.x, r, color=color, zorder=10)
        pyplot.gca().add_artist(circle)
