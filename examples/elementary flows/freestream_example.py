#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show how to use point source code.

The examples shows how to draw streamlines, potential lines, and velocity
vectors for a point source.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyPC.freestream_flow import FreestreamFlow2D


def draw_freestream_flow_field() -> None:
    """Draws the streamlines, potential lines, and the velocity vectors."""
    freestream = FreestreamFlow2D(U_inf=10, alpha=0.2)

    # set mesh
    nptsx = 100
    nptsy = 100
    xg, yg = np.meshgrid(np.linspace(-1, 5, nptsx), np.linspace(-1, 5, nptsy))

    stream_function = freestream.stream_function(xg, yg, True)
    potential = freestream.potential(xg, yg)
    ug, vg = freestream.velocity(xg, yg)

    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(16)
    gs = GridSpec(3, 1, figure=fig)
    stream_function_axis = fig.add_subplot(gs[0, 0])
    potential_axis = fig.add_subplot(gs[1, 0])
    velocity_axis = fig.add_subplot(gs[2, 0])

    ax = stream_function_axis
    ax.contour(xg, yg, stream_function, levels=10, colors="blue",
               linestyles="solid")
    ax = potential_axis
    ax.contour(xg, yg, potential, levels=10, colors="blue",
               linestyles="solid")
    ax = velocity_axis
    ax.quiver(xg, yg, ug, vg, color="blue")
    ax.set_xlim([0.5, 1.5])
    ax.set_ylim([1.5, 2.5])


if __name__ == "__main__":
    draw_freestream_flow_field()
