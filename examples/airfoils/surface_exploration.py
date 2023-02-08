#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Example showing how to interact with the airfoil class."""

from typing import Tuple

import numpy as np
import numpy.typing as nptype
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from pyPC.airfoil.camber import Naca4DigitCamber
from pyPC.airfoil.thickness import Naca45DigitThickness
from pyPC.airfoil.airfoil import OrthogonalAirfoil
from pyPC.airfoil.airfoil import Airfoil


def draw_surface_vectors(af: OrthogonalAirfoil) -> None:
    """Draw surface vectors in the leading edge region."""
    n_sample = 5
    xi_start = -0.1
    xi_end = 0.1

    # get points needed to draw the airfoil curve
    xi_l = np.linspace(xi_start, 0, 200)
    xi_u = np.linspace(0, xi_end, 200)
    xi = np.concatenate((xi_l, xi_u[1:]))
    x_curve, y_curve = af.xy(xi)

    # get the camber and chord info
    x_camber = np.linspace(0, 1, 200)
    y_camber = af.camber_value(x_camber)

    # TODO: Need to use the tangent and normal methods of airfoil
    xle, yle = af.leading_edge()
    xte, yte = af.trailing_edge()
    # tx_le, ty_le = af.tangent(0)

    # sample the airfoil to visualize local properties
    xi_sample_l = np.linspace(xi_start, -0.0025, n_sample)
    xi_sample_u = np.linspace(0.0025, xi_end, n_sample)
    xi_sample = np.concatenate((xi_sample_l, xi_sample_u))
    x_sample, y_sample = af.xy(xi_sample)

    xp_sample, yp_sample = af.xy_p(xi_sample)
    tx_sample = xp_sample/np.sqrt(xp_sample**2 + yp_sample**2)
    ty_sample = yp_sample/np.sqrt(xp_sample**2 + yp_sample**2)

    # get x-minimum point info
    xi_xmin = af.xi_xmin
    x_xmin, y_xmin = af.xy(xi_xmin)

    # plot leading edge curves
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.plot(xle, yle, 'om')
    plt.plot(x_camber, y_camber, '--m')
    plt.plot([xle, xte], [yle, yte], '-.m')
    plt.plot(x_curve, y_curve, '-b')
    plt.plot(xle, yle, 'or')
    plt.plot(x_sample, y_sample, 'or')
    # plt.quiver(xle, yle, tx_le, ty_le, color="red")
    plt.quiver(x_sample, y_sample, tx_sample, ty_sample, color="red")
    for i in range(n_sample):
        plt.plot([x_sample[i], x_sample[-i-1]],
                 [y_sample[i], y_sample[-i-1]], "--g")
    plt.axis("equal")
    plt.xlim(right=0.2)
    plt.xlabel("$x$")
    plt.ylabel("$z$")
    plt.text(0.1, 0.1, f"Min. $x$ at $\\xi=${xi_xmin:.2e} \n({x_xmin:.2e}, "
             f"{y_xmin:.2e})", fontsize="small")
    plt.show()


def draw_surface_slope(af: Airfoil) -> None:
    """Draw surface properties as a function of parameterization."""
    n_curve = 300
    xi_min = -1
    xi_max = 1
    offset = 1e-8

    # find joints and vertical slope location
    xi_joints = af.joints()
    xi_xmin = af.xi_xmin

    # get points needed to draw the airfoil curve
    xi_curve_l = np.linspace(xi_min, xi_xmin-2*offset, n_curve//2)
    xi_curve_u = np.linspace(xi_xmin+offset, xi_max, n_curve//2)
    xi_curve = np.concatenate((xi_curve_l, np.array([xi_xmin-offset]),
                               xi_curve_u))
    xi_curve_end_idx = xi_curve_l.shape[0]+1
    for xi in xi_joints[1:-1]:
        if (xi > xi_min) and (xi < xi_max):
            idx = (np.abs(xi_curve-xi)).argmin()
            xi_curve[idx] = xi

    x_curve, y_curve = af.xy(xi_curve)
    slope_curve = af.dydx(xi_curve)
    k_curve = af.k(xi_curve)

    # get camber transition location
    interior_joints = len(xi_joints) > 2
    if interior_joints:
        xi_trans = xi_joints[1:-1]
        x_trans, y_trans = af.xy(xi_trans)
        slope_trans = af.dydx(xi_trans)
        k_trans = af.k(xi_trans)
    else:
        xi_trans = None
        x_trans = None
        y_trans = None
        slope_trans = None
        k_trans = None

    # plot figures
    def calc_limits(v: nptype.NDArray) -> Tuple[float, float]:
        v_min = np.ma.masked_invalid(v).min()
        v_max = np.ma.masked_invalid(v).max()
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
    fig.set_figwidth(8)
    fig.set_figheight(12)
    gs = GridSpec(4, 1, figure=fig)
    axis_x = fig.add_subplot(gs[0, 0])
    axis_y = fig.add_subplot(gs[1, 0])
    axis_slope = fig.add_subplot(gs[2, 0])
    axis_curv = fig.add_subplot(gs[3, 0])

    x_min, x_max = calc_limits(xi_curve)

    ax = axis_x
    y_min, y_max = calc_limits(x_curve)
    if interior_joints:
        ax.plot(xi_trans, x_trans, 'og')
    ax.plot(xi_curve, x_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$x$")
    ax.grid(True)

    ax = axis_y
    y_min, y_max = calc_limits(y_curve)
    if interior_joints:
        ax.plot(xi_trans, y_trans, 'og')
    ax.plot(xi_curve, y_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$y$")
    ax.grid(True)

    ax = axis_slope
    y_min, y_max = calc_limits(slope_curve)
    if interior_joints:
        ax.plot(xi_trans, slope_trans, 'og')
    ax.plot(xi_curve[0:xi_curve_end_idx], slope_curve[0:xi_curve_end_idx],
            '-b')
    ax.plot(xi_curve[xi_curve_end_idx:], slope_curve[xi_curve_end_idx:], '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(-5, y_min), min(5, y_max))
    ax.set_ylabel(r"$\frac{dy}{dx}$")
    ax.grid(True)

    ax = axis_curv
    y_min, y_max = calc_limits(k_curve)
    if interior_joints:
        ax.plot(xi_trans, k_trans, 'og')
    ax.plot(xi_curve, k_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$\xi$")
    ax.set_ylabel("$k$")
    ax.grid(True)


def create_naca_4digit(max_camber_index: float, loc_max_camber_index: float,
                       max_thickness_index: float) -> OrthogonalAirfoil:
    """Create NACA 4-digit airfoil manually."""
    camber = Naca4DigitCamber(mci=max_camber_index, lci=loc_max_camber_index)
    thickness = Naca45DigitThickness(mti=max_thickness_index)
    return OrthogonalAirfoil(camber, thickness)


if __name__ == "__main__":
    af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                            max_thickness_index=21)
    # af = create_naca_4digit(max_camber_index=1, loc_max_camber_index=4,
    #                         max_thickness_index=8)
    # af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
    #                         max_thickness_index=12)
    draw_surface_vectors(af)
    draw_surface_slope(af)
