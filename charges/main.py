import numpy
from matplotlib import pyplot

from charges.Charge import PointCharge
from charges.Field import init, ElectricField, Potential, GaussianCircle, finalize_plot
from charges.custom_math import norm

XMIN, XMAX, XOFFSET = 0, 10, 0
YMIN, YMAX = 0, 10
ZOOM = 10


def main():
    XMIN, XMAX = -40, 40
    YMIN, YMAX = -30, 30
    ZOOM = 6
    XOFFSET = 0

    init(XMIN, XMAX, YMIN, YMAX, ZOOM, XOFFSET)

    # Set up the charges, electric field, and potential
    charges = [PointCharge(1, [-1, 0]),
               PointCharge(-1, [1, 0])]
    field = ElectricField(charges)
    potential = Potential(charges)

    # Set up the Gaussian surface
    g = GaussianCircle(charges[0].x, 0.1)

    # Create the field lines
    fieldlines = []
    for x in g.fluxpoints(field, 12):
        fieldlines.append(field.line(x))
    fieldlines.append(field.line([10, 0]))

    # Create the vector grid
    x, y = numpy.meshgrid(numpy.linspace(XMIN / ZOOM + XOFFSET, XMAX / ZOOM + XOFFSET, 41),
                          numpy.linspace(YMIN / ZOOM, YMAX / ZOOM, 31))
    u, v = numpy.zeros_like(x), numpy.zeros_like(y)
    n, m = x.shape
    for i in range(n):
        for j in range(m):
            if any(numpy.isclose(norm(charge.x - [x[i, j], y[i, j]]),
                                 0) for charge in charges):
                u[i, j] = v[i, j] = None
            else:
                mag = field.magnitude([x[i, j], y[i, j]]) ** (1 / 5)
                a = field.angle([x[i, j], y[i, j]])
                u[i, j], v[i, j] = mag * numpy.cos(a), mag * numpy.sin(a)

    ## Plotting ##

    # Electric field lines and potential contours
    fig = pyplot.figure(figsize=(6, 4.5))
    potential.plot()
    field.plot()
    for fieldline in fieldlines:
        fieldline.plot()
    for charge in charges:
        charge.plot()
    finalize_plot()
    # fig.savefig('dipole-field-lines.pdf', transparent=True)

    # Field vectors
    fig = pyplot.figure(figsize=(6, 4.5))
    cmap = pyplot.cm.get_cmap('plasma')
    pyplot.quiver(x, y, u, v, pivot='mid', cmap=cmap, scale=35)
    for charge in charges:
        charge.plot()
    finalize_plot()
    # fig.savefig('dipole-field-vectors.pdf', transparent=True)

    pyplot.show()


if __name__ == "__main__":
    main()
