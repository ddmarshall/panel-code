#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  2 22:05:01 2022

@author: ddmarshall
"""

from typing import Tuple, List
from dataclasses import dataclass

import numpy as np
import numpy.typing as np_type
from scipy import linalg

import matplotlib.pyplot as plt

from pyPC.element_flow import LineElement2D
from pyPC.freestream_flow import FreestreamFlow2D
from pyPC.source_flow import LineSourceConstant2D


@dataclass
class FreestreamCondition:
    """Representation of freestream flow conditions."""

    U_inf: float = 1
    alpha: float = 0


@dataclass
class Geometry:
    """Representation of basic geometry terms."""

    name: str
    x: np_type.NDArray
    y: np_type.NDArray

    def number_panels(self) -> int:
        """
        Return number of panels in this geometry.

        Returns
        -------
        int
            Number of panels.
        """
        return self.x.size - 1


def hess_smith_solver(freestream: FreestreamCondition,
                      airfoil:List[Geometry]) -> Tuple[FreestreamFlow2D,
                                                       List[List[LineElement2D]]]:
    """
    Solves for the panel strengths for Hess & Smith method.

    Parameters
    ----------
    freestream : FreestreamCondition
        Freestream conditions for this case.
    airfoil : List[Geometry]
        Geometry in the flow field.

    Returns
    -------
    List[LineElement2D]
        Elements representing the geometry.
    """
    # create freestream and surface elements
    fs = FreestreamFlow2D(U_inf=freestream.U_inf, alpha=freestream.alpha)
    src: List[LineSourceConstant2D] = []
    j = 0
    npan = airfoil[j].number_panels()
    for i in range(npan):
        src.append(LineSourceConstant2D(xo=(airfoil[j].x[i],
                                            airfoil[j].x[i+1]),
                                        yo=(airfoil[j].y[i],
                                            airfoil[j].y[i+1])))

    # construct solution
    A = np.empty((npan, npan))
    r = np.empty((npan, 1))
    for i in range(npan):  # each collocation point
        xc, yc = src[i].get_panel_collo_point()
        nx, ny = src[i].get_panel_normal()
        u_inf, v_inf = fs.velocity(xc, yc)
        r[i] = -(u_inf*nx + v_inf*ny)
        for j in range(npan):  # each panel
            u, v = src[j].velocity(xc, yc, True)
            A[i, j] = u*nx + v*ny

    # get solution and set panel strengths
    sol = linalg.solve(A, r)
    for i in range(npan):
        src[i].set_strength(sol[i])

    return fs, src


def main() -> None:
    """Run main function."""
    # set freestream conditions
    freestream = FreestreamCondition(U_inf=10, alpha=0)

    # get airfoil geometry
    npan = 101
    c = 2
    theta = np.linspace(2*np.pi, 0, npan+1)
    af = Geometry(name="Cylinder", x=0.5*c*(np.cos(theta)+1),
                  y=0.5*c*np.sin(theta))

    # solve for panel strengths
    fs, elements = hess_smith_solver(freestream, [af])

    # analyze results
    xc, yc = np.zeros(af.number_panels()), np.zeros(af.number_panels())
    nx, ny = np.zeros(af.number_panels()), np.zeros(af.number_panels())
    sx, sy = np.zeros(af.number_panels()), np.zeros(af.number_panels())
    for i, el in enumerate(elements):
        xc[i], yc[i] = el.get_panel_collo_point()
        nx[i], ny[i] = el.get_panel_normal()
        sx[i], sy[i] = el.get_panel_tangent()
    u_inf, v_inf = fs.velocity(0, 0)
    u, v = u_inf*np.ones(af.number_panels()), v_inf*np.ones(af.number_panels())

    # set mesh
    nptsx = 100
    nptsy = 100
    xg, yg = np.meshgrid(np.linspace(-3, 5, nptsx), np.linspace(-4, 4, nptsy))
    stream_function = fs.stream_function(xg, yg)
    potential = fs.potential(xg, yg)
    ug, vg = fs.velocity(xg, yg)
    # doublet = PointDoublet2D(xo=0.5*c, yo=0, angle=np.pi,
    #                          strength=2*np.pi*fs.U_inf*(0.5*c)**2)
    # stream_function = stream_function + doublet.stream_function(xg, yg)
    for el in elements:
        uc, vc = el.velocity(xc, yc, True)
        u = u + uc
        v = v + vc
        stream_function = stream_function + el.stream_function(xg, yg, True)
        potential = potential + el.potential(xg, yg, True)
        ut, vt = el.velocity(xg, yg, True)
        ug = ug + ut
        vg = vg + vt
    cp = 1-(u**2 + v**2)/fs.U_inf**2
    theta = np.arctan2(yc, xc-0.5*c)
    theta[theta < 0] = 2*np.pi + theta[theta < 0]

    # # reference results
    # doublet = PointDoublet2D(xo=0.5*c, yo=0, angle=np.pi,
    #                          strength=2*np.pi*fs.U_inf*(0.5*c)**2)
    # u_ref, v_ref = (u_inf*np.ones(af.number_panels()),
    #                 v_inf*np.ones(af.number_panels()))
    # ud, vd = doublet.velocity(xc, yc, True)
    # u_ref = u_ref + ud
    # v_ref = v_ref + vd
    # cp_ref = 1-(u_ref**2 + v_ref**2)/fs.U_inf**2

    # plot geometry
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(af.x, af.y, "-ro")
    plt.plot(xc, yc, "rs")
    plt.quiver(xc, yc, nx, ny, scale=10, color="blue")
    plt.quiver(xc, yc, sx, sy, scale=10, color="blue")
    plt.title(f"{af.name} with {af.number_panels()} panels")

    # plot the velocity vectors on surface
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(af.x, af.y, "-r")
    plt.quiver(xc, yc, u, v, scale=100, color="orange")

    # plot the pressure profile
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(theta*(180/np.pi), cp, "-b")
    plt.xticks(np.linspace(0, 360, 9))
    plt.xlabel(r"$\theta\ (^\circ)$")
    plt.ylabel(r"$c_p$")

    # plot the potentials
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.contour(xg, yg, potential, levels=60, colors="blue",
                linestyles="solid")
    plt.plot(af.x, af.y, "-r")

    # plot the stream function
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.contour(xg, yg, stream_function, levels=60, colors="blue",
                linestyles="solid")
    plt.plot(af.x, af.y, "-r")

    # plot the flow field
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.streamplot(xg, yg, ug, vg, broken_streamlines=False,
                   color=np.sqrt(ug**2+vg**2),
                   start_points=np.array([np.linspace(-3, -3, 40),
                                          np.linspace(-4, 4, 40)]).T,
                   linewidth=0.5, cmap="jet")
    plt.plot(af.x, af.y, "-k", linewidth=0.5)


if __name__ == "__main__":
    main()
