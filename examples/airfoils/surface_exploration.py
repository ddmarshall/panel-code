#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 15:31:03 2023

@author: ddmarshall
"""

from typing import Tuple

import numpy as np
import numpy.typing as nptype
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyPC.airfoil import Naca4DigitAirfoilClassic


def draw_surface_vectors(af: Naca4DigitAirfoilClassic) -> None:
    n_sample = 5
    xi_start = -0.1
    xi_end = 0.1
    le_offset = 1e-7

    # get points needed to draw the airfoil curve
    xi_l = np.linspace(xi_start, -le_offset, 200)
    xi_u = np.linspace(le_offset, xi_end, 200)
    xi = np.concatenate((xi_l, xi_u))
    x_curve, y_curve = af.xy_from_xi(xi)

    # get the camber and chord info
    x_camber = np.linspace(0, 1, 200)
    y_camber = af.camber(x_camber)

    xle, yle = af.leading_edge()
    xte, yte = af.trailing_edge()

    # sample the airfoil to visualize local properties
    xi_sample_l = np.linspace(xi_start, -0.0025, n_sample)
    xi_sample_u = np.linspace(0.0025, xi_end, n_sample)
    xi_sample = np.concatenate((xi_sample_l, xi_sample_u))
    x_sample, y_sample = af.xy_from_xi(xi_sample)

    xp_sample, yp_sample = af.xy_p(xi_sample)
    tx_sample = xp_sample/np.sqrt(xp_sample**2 + yp_sample**2)
    ty_sample = yp_sample/np.sqrt(xp_sample**2 + yp_sample**2)

    x_camber_sample = xi_sample_u
    y_camber_sample = af.camber(x_camber_sample)
    thick_sample = af.thickness(x_camber_sample)

    xmin_idx = np.argmin(x_curve)
    print(xi[xmin_idx])

    fig = plt.figure()
    ax = plt.gca()
    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.plot(xle, yle, 'om')
    plt.plot(x_camber, y_camber, '--m')
    plt.plot([xle, xte], [yle, yte], '-.m')
    plt.plot(x_curve, y_curve, '-b')
    plt.plot(x_sample, y_sample, 'or')
    plt.quiver(x_sample, y_sample, tx_sample, ty_sample, color="red")
    for i in range(n_sample):
        plt.plot([x_sample[i], x_sample[-i-1]],
                 [y_sample[i], y_sample[-i-1]], "--g")
    plt.axis("equal")
    plt.xlim(right=0.2)
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.text(0.095, 0.075, f"Min. $x$ at ({x_curve[xmin_idx]:.2e}, "
             f"{y_curve[xmin_idx]:.2e})", fontsize="small")
    plt.show()


def draw_surface_slope(af: Naca4DigitAirfoilClassic) -> None:
    n_curve = 200
    xi_min = -0.1
    xi_max = 0.1
    le_offset = 1e-7

    # get points needed to draw the airfoil curve
    xi_curve_l = np.linspace(xi_min, -le_offset, n_curve//2)
    xi_curve_u = np.linspace(le_offset, xi_max, n_curve//2)
    xi_curve = np.concatenate((xi_curve_l, xi_curve_u))
    idx_curve_le = len(xi_curve_l)
    x_curve, y_curve = af.xy_from_xi(xi_curve)
    xp_curve, yp_curve = af.xy_p(xi_curve)
    slope_curve = yp_curve/xp_curve
    k_curve = af.k(xi_curve)

    # find vertical slope location
    idx_slope_disc = 1+np.where(np.diff(np.sign(xp_curve)) != 0)[0][0]

    # get camber transition location
    xi_trans = [-af.p/10, af.p/10]
    x_trans, y_trans = af.xy_from_xi(xi_trans)
    xp_trans, yp_trans = af.xy_p(xi_trans)
    slope_trans = yp_trans/xp_trans
    k_trans = af.k(xi_trans)

    # plot figures
    def calc_limits(v: nptype.NDArray) -> Tuple[float, float]:
        rel_limit = 0.01

        v_min = np.min(v)
        v_max = np.max(v)
        delta_v = v_max - v_min
        if abs(v_min/delta_v) < 0.05:
            v_min -= delta_v/10
        elif v_min > 0:
            v_min *= 0.9
        else:
            v_min *= 1.1

        if abs(v_max/delta_v) < 0.05:
            v_max += delta_v/10
        elif v_max < 0:
            v_max *= 0.9
        else:
            v_max *= 1.1
        return v_min, v_max


    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(12)
    gs = GridSpec(4, 1, figure=fig)
    axis_x = fig.add_subplot(gs[0, 0])
    axis_y = fig.add_subplot(gs[1, 0])
    axis_slope = fig.add_subplot(gs[2, 0])
    axis_curv = fig.add_subplot(gs[3, 0])

    x_min, x_max = calc_limits(xi_curve)

    ax = axis_x
    y_min, y_max = calc_limits(x_curve)
    ax.plot(xi_trans, x_trans, 'og')
    ax.plot(xi_curve, x_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$x$")
    ax.grid(True)

    ax = axis_y
    y_min, y_max = calc_limits(y_curve)
    ax.plot(xi_trans, y_trans, 'og')
    ax.plot(xi_curve, y_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$y$")
    ax.grid(True)

    ax = axis_slope
    y_min, y_max = calc_limits(slope_curve)
    ax.plot(xi_trans, slope_trans, 'og')
    ax.plot(xi_curve[0:idx_slope_disc], slope_curve[0:idx_slope_disc], '-b')
    ax.plot(xi_curve[idx_slope_disc:-1], slope_curve[idx_slope_disc:-1], '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(-5, y_min), min(5, y_max))
    ax.set_ylabel(r"$\frac{dy}{dx}$")
    ax.grid(True)

    ax = axis_curv
    y_min, y_max = calc_limits(k_curve)
    ax.plot(xi_trans, k_trans, 'og')
    ax.plot(xi_curve, k_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("$k$")
    ax.grid(True)

if __name__ == "__main__":
    af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=2,
                                  max_thickness=21, scale=1.0)
    # af = Naca4DigitAirfoilClassic(max_camber=1, max_camber_location=4,
    #                               max_thickness=8, scale=1.0)
    draw_surface_vectors(af)
    draw_surface_slope(af)
