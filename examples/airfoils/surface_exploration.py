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
    n_sample = 8
    t_start = -0.4
    t_end = -t_start

    # get points needed to draw the airfoil curve
    t_l = np.linspace(t_start, 0, 200)
    t_u = np.linspace(0, t_end, 200)
    t = np.concatenate((t_l, t_u[1:]))
    x_curve, y_curve = af.xy(t)

    # get the camber and chord info
    t_camber = np.linspace(0, 1, 200)
    x_camber, y_camber = af.camber_location(t_camber)

    xle, yle = af.leading_edge()
    xte, yte = af.trailing_edge()
    sx_le, sy_le = af.tangent(0)

    # sample the airfoil to visualize local properties
    t_sample = np.linspace(t_start, t_end, n_sample)
    t_sample[n_sample//2-1:n_sample//2+1] *= 1.5  # move points away from l.e.
    x_sample, y_sample = af.xy(t_sample)
    sx_sample, sy_sample = af.tangent(t_sample)

    # get x-minimum point info
    t_xmin = af.xmin_parameter
    x_xmin, y_xmin = af.xy(t_xmin)

    # plot leading edge curves
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(4)
    plt.plot(x_camber, y_camber, '--m')
    plt.plot([xle, xte], [yle, yte], '-.m')
    plt.plot(x_curve, y_curve, '-b')
    plt.plot(x_sample, y_sample, 'or')
    plt.plot(xle, yle, 'or')
    plt.quiver(xle, yle, sx_le, sy_le, color="red")
    plt.quiver(x_sample, y_sample, sx_sample, sy_sample, color="red")
    for i in range(n_sample):
        plt.plot([x_sample[i], x_sample[-i-1]],
                 [y_sample[i], y_sample[-i-1]], "--g")
    plt.axis("equal")
    plt.xlim(right=0.2)
    plt.xlabel("$x$")
    plt.ylabel("$y$")
    plt.text(-0.05, 0.1, f"Min. $x$ at $t=${t_xmin:.2e} \n({x_xmin:.2e}, "
             f"{y_xmin:.2e})", fontsize="small")
    plt.show()


def draw_surface_curves(af: Airfoil) -> None:
    """Draw surface and related properties."""
    n_curve = 300
    t_min = -1
    t_max = 1
    offset = 1e-8

    # find joints and vertical slope location
    t_joints = af.joints()
    t_xmin = af.xmin_parameter

    # get points needed to draw the airfoil curve
    t_curve_l = np.linspace(t_min, t_xmin-2*offset, n_curve//2)
    t_curve_u = np.linspace(t_xmin+offset, t_max, n_curve//2)
    t_curve = np.concatenate((t_curve_l, np.array([t_xmin-offset]),
                              t_curve_u))
    t_curve_end_idx = t_curve_l.shape[0]+1
    for t in t_joints[1:-1]:
        if (t > t_min) and (t < t_max):
            idx = (np.abs(t_curve-t)).argmin()
            t_curve[idx] = t

    x_curve, y_curve = af.xy(t_curve)
    slope_curve = af.dydx(t_curve)
    k_curve = af.k(t_curve)

    # get camber transition location
    interior_joints = len(t_joints) > 2
    if interior_joints:
        t_trans = t_joints[1:-1]
        x_trans, y_trans = af.xy(t_trans)
        slope_trans = af.dydx(t_trans)
        k_trans = af.k(t_trans)
    else:
        t_trans = None
        x_trans = None
        y_trans = None
        slope_trans = None
        k_trans = None

    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(12)
    gs = GridSpec(3, 1, figure=fig)
    axis_xy = fig.add_subplot(gs[0, 0])
    axis_slope = fig.add_subplot(gs[1, 0])
    axis_curv = fig.add_subplot(gs[2, 0])

    axy = axis_xy
    x_min, x_max = calc_limits(x_curve)
    y_min, y_max = calc_limits(y_curve)
    if interior_joints:
        axy.plot(x_trans, y_trans, 'og')
    axy.plot(x_curve, y_curve, '-b')
    axy.set_xlim(x_min, x_max)
    axy.set_ylim(y_min, y_max)
    axy.set_xlabel("$x$")
    axy.set_ylabel("$y$")
    axy.grid(True)
    axy.axis("equal")

    ax = axis_slope
    x_min, x_max = calc_limits(t_curve)
    y_min, y_max = calc_limits(slope_curve)
    if interior_joints:
        ax.plot(t_trans, slope_trans, 'og')
    ax.plot(t_curve[0:t_curve_end_idx], slope_curve[0:t_curve_end_idx],
            '-b')
    ax.plot(t_curve[t_curve_end_idx:], slope_curve[t_curve_end_idx:], '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(-5, y_min), min(5, y_max))
    ax.set_ylabel(r"$\frac{dy}{dx}$")
    ax.grid(True)

    ax = axis_curv
    x_min, x_max = calc_limits(t_curve)
    y_min, y_max = calc_limits(k_curve)
    if interior_joints:
        ax.plot(t_trans, k_trans, 'og')
    ax.plot(t_curve, k_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("$k$")
    ax.grid(True)


def draw_parameter_variation(af: Airfoil) -> None:
    """Draw surface properties as a function of parameterization."""
    n_curve = 300
    t_min = -1
    t_max = 1
    offset = 1e-8

    # find joints and vertical slope location
    t_joints = af.joints()
    t_xmin = af.xmin_parameter

    # get points needed to draw the airfoil curve
    t_curve_l = np.linspace(t_min, t_xmin-2*offset, n_curve//2)
    t_curve_u = np.linspace(t_xmin+offset, t_max, n_curve//2)
    t_curve = np.concatenate((t_curve_l, np.array([t_xmin-offset]),
                              t_curve_u))
    t_curve_end_idx = t_curve_l.shape[0]+1
    for t in t_joints[1:-1]:
        if (t > t_min) and (t < t_max):
            idx = (np.abs(t_curve-t)).argmin()
            t_curve[idx] = t

    x_curve, y_curve = af.xy(t_curve)
    slope_curve = af.dydx(t_curve)
    k_curve = af.k(t_curve)

    # get camber transition location
    interior_joints = len(t_joints) > 2
    if interior_joints:
        t_trans = t_joints[1:-1]
        x_trans, y_trans = af.xy(t_trans)
        slope_trans = af.dydx(t_trans)
        k_trans = af.k(t_trans)
    else:
        t_trans = None
        x_trans = None
        y_trans = None
        slope_trans = None
        k_trans = None

    fig = plt.figure()
    fig.set_figwidth(8)
    fig.set_figheight(12)
    gs = GridSpec(4, 1, figure=fig)
    axis_x = fig.add_subplot(gs[0, 0])
    axis_y = fig.add_subplot(gs[1, 0])
    axis_slope = fig.add_subplot(gs[2, 0])
    axis_curv = fig.add_subplot(gs[3, 0])

    x_min, x_max = calc_limits(t_curve)

    ax = axis_x
    y_min, y_max = calc_limits(x_curve)
    if interior_joints:
        ax.plot(t_trans, x_trans, 'og')
    ax.plot(t_curve, x_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$x$")
    ax.grid(True)

    ax = axis_y
    y_min, y_max = calc_limits(y_curve)
    if interior_joints:
        ax.plot(t_trans, y_trans, 'og')
    ax.plot(t_curve, y_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_ylabel("$y$")
    ax.grid(True)

    ax = axis_slope
    y_min, y_max = calc_limits(slope_curve)
    if interior_joints:
        ax.plot(t_trans, slope_trans, 'og')
    ax.plot(t_curve[0:t_curve_end_idx], slope_curve[0:t_curve_end_idx],
            '-b')
    ax.plot(t_curve[t_curve_end_idx:], slope_curve[t_curve_end_idx:], '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(max(-5, y_min), min(5, y_max))
    ax.set_ylabel(r"$\frac{dy}{dx}$")
    ax.grid(True)

    ax = axis_curv
    y_min, y_max = calc_limits(k_curve)
    if interior_joints:
        ax.plot(t_trans, k_trans, 'og')
    ax.plot(t_curve, k_curve, '-b')
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"$t$")
    ax.set_ylabel("$k$")
    ax.grid(True)


def calc_limits(v: nptype.NDArray) -> Tuple[float, float]:
    """Calculate the limits for plotting."""
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
    draw_surface_curves(af)
    # draw_parameter_variation(af)
