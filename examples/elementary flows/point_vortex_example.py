#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show how to use point vortex code.

The examples shows how to draw streamlines, potential lines, and velocity
vectors for a point vortex.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyPC.vortex_flow import PointVortex2D


def draw_point_vortex_flow_field() -> None:
    """Draws the streamlines, potential lines, and the velocity vectors."""
    vortex = PointVortex2D(xo=1, yo=2, strength=1.0)

    # set mesh
    nptsx = 100
    nptsy = 100
    xg, yg = np.meshgrid(np.linspace(-1, 5, nptsx), np.linspace(-1, 5, nptsy))

    stream_function = vortex.stream_function(xg, yg)
    potential = vortex.potential(xg, yg, True)
    ug, vg = vortex.velocity(xg, yg)

    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(16)
    gs = GridSpec(3, 1, figure=fig)
    stream_function_axis = fig.add_subplot(gs[0, 0])
    potential_axis = fig.add_subplot(gs[1, 0])
    velocity_axis = fig.add_subplot(gs[2, 0])

    ax = stream_function_axis
    ax.contour(xg, yg, stream_function, levels=20, colors="blue",
               linestyles="solid")
    ax = potential_axis
    ax.contour(xg, yg, potential, levels=20, colors="blue",
               linestyles="solid")
    ax = velocity_axis
    ax.quiver(xg, yg, ug, vg, color="blue")
    ax.set_xlim([0.5, 1.5])
    ax.set_ylim([1.5, 2.5])


if __name__ == "__main__":
    draw_point_vortex_flow_field()
