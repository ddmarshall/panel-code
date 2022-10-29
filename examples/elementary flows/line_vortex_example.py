#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Show how to use line vortex code.

The examples shows how to draw streamlines, potential lines, and velocity
vectors for a line vortex.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyPC.vortex_flow import LineVortexConstant2D


def draw_approximate_point_vortex_flow_field():
    """Draws the streamlines, potential lines, and the velocity vectors."""
    # set mesh
    xg, yg = np.meshgrid(np.linspace(-1, 5, 100), np.linspace(-1, 5, 100))

    vortex = LineVortexConstant2D([1, 2], [2, 4], strength=1)
    ug, vg = vortex.velocity(xg, yg)
    potential = vortex.potential(xg, yg)
    stream_function = vortex.stream_function(xg, yg)

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
    ax.plot(vortex.x0, vortex.y0, linewidth=3,
            color="red", marker="o")
    ax = potential_axis
    ax.contour(xg, yg, potential, levels=20, colors="blue",
               linestyles="solid")
    ax.plot(vortex.x0, vortex.y0, linewidth=3,
            color="red", marker="o")
    ax = velocity_axis
    ax.quiver(xg, yg, ug, vg, color="blue")
    ax.plot(vortex.x0, vortex.y0, linewidth=3,
            color="red", marker="o")
    ax.set_xlim([0.8, 2.2])
    ax.set_ylim([1.8, 4.2])


if __name__ == "__main__":
    draw_approximate_point_vortex_flow_field()
