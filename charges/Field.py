import matplotlib
import numpy
from matplotlib import pyplot
from numpy import alltrue, isclose
from numpy import array, arange, linspace, meshgrid, zeros_like, ones_like
from numpy import insert
from numpy import log10, sin, cos, arctan2, cumsum
from numpy import radians
from scipy.integrate import ode

from charges.custom_math import norm, lininterp2

XMIN, XMAX = None, None
YMIN, YMAX = None, None
ZOOM = None
XOFFSET = None


def init(xmin, xmax, ymin, ymax, zoom=1, xoffset=0):
    """Initializes the domain."""
    # pylint: disable=global-statement
    global XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET
    XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET = \
        xmin, xmax, ymin, ymax, zoom, xoffset


class FieldLine:
    """A Field Line."""

    def __init__(self, x):
        "Initializes the field line points 'x'."""
        self.x = x

    def plot(self, linewidth=None, linestyle='-',
             startarrows=True, endarrows=True):
        """Plots the field line and arrows."""

        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']

        x, y = zip(*self.x)
        pyplot.plot(x, y, '-k', linewidth=linewidth, linestyle=linestyle)

        n = int(len(x) / 2) if len(x) < 225 else 75
        if startarrows:
            pyplot.arrow(x[n], y[n], (x[n + 1] - x[n]) / 100., (y[n + 1] - y[n]) / 100.,
                         fc="k", ec="k",
                         head_width=0.1 * linewidth, head_length=0.1 * linewidth)

        if len(x) < 225 or not endarrows:
            return

        pyplot.arrow(x[-n], y[-n],
                     (x[-n + 1] - x[-n]) / 100., (y[-n + 1] - y[-n]) / 100.,
                     fc="k", ec="k",
                     head_width=0.1 * linewidth, head_length=0.1 * linewidth)


class ElectricField:
    """The electric field owing to a collection of charges."""

    dt0 = 0.01  # The time step for integrations

    def __init__(self, charges):
        """Initializes the field given 'charges'."""
        self.charges = charges

    def vector(self, x):
        """Returns the field vector."""
        return numpy.sum([charge.E(x) for charge in self.charges], axis=0)

    def magnitude(self, x):
        """Returns the magnitude of the field vector."""
        return norm(self.vector(x))

    def angle(self, x):
        """Returns the field vector's angle from the x-axis (in radians)."""
        return arctan2(*(self.vector(x).T[::-1]))  # arctan2 gets quadrant right

    def direction(self, x):
        """Returns a unit vector pointing in the direction of the field."""
        v = self.vector(x)
        return (v.T / norm(v)).T

    def projection(self, x, a):
        """Returns the projection of the field vector on a line at given angle
        from x-axis."""
        return self.magnitude(x) * cos(a - self.angle(x))

    def line(self, x0):
        """Returns the field line passing through x0.
        Refs: http://folk.uib.no/fcihh/seminar/lec1.pdf and lect2.pdf
              http://numbercrunch.de/blog/2013/05/visualizing-streamlines/
        and especially: "Electric field lines don't work",
        http://scitation.aip.org/content/aapt/journal/ajp/64/6/10.1119/1.18237
        """

        if None in [XMIN, XMAX, YMIN, YMAX]:
            raise ValueError('Domain must be set using init().')

        # Set up integrator for the field line
        streamline = lambda t, y: list(self.direction(y))
        solver = ode(streamline).set_integrator('vode')

        # Initialize the coordinate lists
        x = [x0]

        # Integrate in both the forward and backward directions
        dt = 0.008

        # Solve in both the forward and reverse directions
        for sign in [1, -1]:

            # Set the starting coordinates and time
            solver.set_initial_value(x0, 0)

            # Integrate field line over successive time steps
            while solver.successful():

                # Find the next step
                solver.integrate(solver.t + sign * dt)

                # Save the coordinates
                if sign > 0:
                    x.append(solver.y)
                else:
                    x.insert(0, solver.y)

                # Check if line connects to a charge
                flag = False
                for c in self.charges:
                    if c.is_close(solver.y):
                        flag = True
                        break

                # Terminate line at charge or if it leaves the area of interest
                if flag or not (XMIN < solver.y[0] < XMAX) or \
                        not YMIN < solver.y[1] < YMAX:
                    break

        return FieldLine(x)

    def plot(self, nmin=-3.5, nmax=1.5):
        """Plots the field magnitude."""
        x, y = meshgrid(
            linspace(XMIN / ZOOM + XOFFSET, XMAX / ZOOM + XOFFSET, 200),
            linspace(YMIN / ZOOM, YMAX / ZOOM, 200))
        z = zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # pylint: disable=unsupported-assignment-operation
                z[i, j] = log10(self.magnitude([x[i, j], y[i, j]]))
        levels = arange(nmin, nmax + 0.2, 0.2)
        cmap = pyplot.cm.get_cmap('plasma')
        pyplot.contourf(x, y, numpy.clip(z, nmin, nmax),
                        10, cmap=cmap, levels=levels, extend='both')


class Potential:
    """The potential owing to a collection of charges."""

    def __init__(self, charges):
        """Initializes the field given 'charges'."""
        self.charges = charges

    def magnitude(self, x):
        """Returns the magnitude of the potential."""
        return sum(charge.V(x) for charge in self.charges)

    def plot(self, zmin=-1.5, zmax=1.5, step=0.25, linewidth=1, linestyle=':'):
        """Plots the field magnitude."""

        if linewidth is None:
            linewidth = matplotlib.rcParams['lines.linewidth']

        x, y = meshgrid(
            linspace(XMIN / ZOOM + XOFFSET, XMAX / ZOOM + XOFFSET, 200),
            linspace(YMIN / ZOOM, YMAX / ZOOM, 200))
        z = zeros_like(x)
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # pylint: disable=unsupported-assignment-operation
                z[i, j] = self.magnitude([x[i, j], y[i, j]])
        # levels = arange(nmin, nmax+0.2, 0.2)
        # cmap = pyplot.cm.get_cmap('plasma')
        pyplot.contour(x, y, z, numpy.arange(zmin, zmax + step, step),
                       linewidths=linewidth, linestyles=linestyle, colors='k')


# pylint: disable=too-few-public-methods
class GaussianCircle:
    """A Gaussian circle with radius r."""

    def __init__(self, x, r, a0=0):
        """Initializes the Gaussian surface at position vector 'x'
        and given radius 'r'.  'a0' defines an offset angle (in radians) CCW
        from the x-axis.  Use this to identify the axis around which flux
        points should be symmetric."""
        self.x = x
        self.r = r
        self.a0 = a0

    def fluxpoints(self, field, n, uniform=False):
        """Returns points where field lines should enter/exit the surface.
        The flux points are usually chosen so that they are equally separated
        in electric field flux.  However, if 'uniform' is True then the points
        are equispaced.
        This method requires that the flux be in xor out everywhere on the
        circle (unless 'uniform' is True)."""

        # Create a dense array of points around the circle
        a = radians(linspace(0, 360, 1001)) + self.a0
        assert len(a) % 4 == 1
        x = self.r * array([cos(a), sin(a)]).T + self.x

        if uniform:
            flux = ones_like(a)

        else:
            # Get the flux through each point.  Ensure the fluxes are either
            # all in or all out.
            flux = field.projection(x, a)

            if numpy.sum(flux) < 0:
                flux *= -1
            assert alltrue(flux > 0)

        # Create an integrated flux curve
        intflux = insert(cumsum((flux[:-1] + flux[1:]) / 2), 0, 0)
        assert isclose(intflux[-1], numpy.sum(flux[:-1]))

        # Divide the integrated flux curve into n+1 portions, and calculate
        # the corresponding angles.
        v = linspace(0, intflux[-1], n + 1)
        a = lininterp2(intflux, a, v)[:-1]

        return self.r * array([cos(a), sin(a)]).T + self.x


def finalize_plot():
    """Finalizes the plot."""
    ax = pyplot.gca()
    ax.set_xticks([])
    ax.set_yticks([])
    pyplot.xlim(XMIN / ZOOM + XOFFSET, XMAX / ZOOM + XOFFSET)
    pyplot.ylim(YMIN / ZOOM, YMAX / ZOOM)
    pyplot.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
