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
from pyPC.airfoil.cylinder import Cylinder


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


class PanelSolution2D():
    """
    Results to solution to 2D panel code.

    Attributes
    ----------
    freestream: FreestreamFlow2D
        Elementary flow representing the freestream conditions.
    body: List[List[LineElement2D]])
        List of elementary flows for each body in flow.
    """

    def __init__(self, freestream: FreestreamFlow2D,
                 body: List[List[LineElement2D]]) -> None:
        self.freestream = freestream
        self.body = body

    def get_surface_geometry(self) -> List[Tuple[np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray,
                                                 np_type.NDArray]]:
        """
        Return the geometry terms associated with each surface.

        Returns
        -------
        List[Tuple[numpy.ndarray,...]]
            A list with the panel end points, panel collocation points, panel
            unit normal vector, and panel unit tangent vector for each body
            in the flow.
        """
        surf = []
        for bod in self.body:
            num_panel = len(bod)
            xp, yp = np.zeros(num_panel + 1), np.zeros(num_panel + 1)
            xc, yc = np.zeros(num_panel), np.zeros(num_panel)
            nx, ny = np.zeros(num_panel), np.zeros(num_panel)
            sx, sy = np.zeros(num_panel), np.zeros(num_panel)
            for i, el in enumerate(bod):
                xp[i], yp[i] = el.get_panel_start()
                xc[i], yc[i] = el.get_panel_collo_point()
                nx[i], ny[i] = el.get_panel_normal()
                sx[i], sy[i] = el.get_panel_tangent()
            xp[-1], yp[-1] = bod[-1].get_panel_end()
            surf.append((xp, yp, xc, yc, nx, ny, sx, sy))

        return surf

    def get_surface_properties(self) -> List[Tuple[np_type.NDArray,
                                                   np_type.NDArray,
                                                   np_type.NDArray,
                                                   np_type.NDArray,
                                                   np_type.NDArray]]:
        """
        Return the flow properties associated with each surface.

        Returns
        -------
        List[Tuple[numpy.ndarray,...]]
            A list with the panel collocation points, the velocity vector, and
            the pressure coefficient for each body in the flow.
        """
        surf = []
        u_inf, v_inf = self.freestream.velocity(0, 0)
        for bod in self.body:
            num_panel = len(bod)
            xc, yc = np.zeros(num_panel), np.zeros(num_panel)
            for i, el in enumerate(bod):
                xc[i], yc[i] = el.get_panel_collo_point()

            u, v = u_inf*np.ones(num_panel), v_inf*np.ones(num_panel)
            for el in bod:
                uc, vc = el.velocity(xc, yc, True)
                u = u + uc
                v = v + vc
            cp = 1-(u**2 + v**2)/self.freestream.U_inf**2
            surf.append((xc, yc, u, v, cp))

        return surf

    def get_potential(self, x: np_type.NDArray,
                      y: np_type.NDArray) -> np_type.NDArray:
        """
        Return the potential value at the points provided.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinates of the points of interest.
        y : numpy.ndarray
            Y-coordinates of the points of interest.

        Returns
        -------
        numpy.ndarray
            Potential value at each point.
        """
        potential = self.freestream.potential(x, y)
        for bod in self.body:
            for el in bod:
                potential = potential + el.potential(x, y, True)

        return potential

    def get_stream_function(self, x: np_type.NDArray,
                            y: np_type.NDArray) -> np_type.NDArray:
        """
        Return the stream function value at the points provided.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinates of the points of interest.
        y : numpy.ndarray
            Y-coordinates of the points of interest.

        Returns
        -------
        numpy.ndarray
            Stream function value at each point.
        """
        stream_function = self.freestream.stream_function(x, y)
        for bod in self.body:
            for el in bod:
                stream_function = (stream_function
                                   + el.stream_function(x, y, True))

        return stream_function

    def get_velocity(self, x: np_type.NDArray,
                     y: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Return the velocity vector at the points provided.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinates of the points of interest.
        y : numpy.ndarray
            Y-coordinates of the points of interest.

        Returns
        -------
        Tuple[numpy.ndarray, numpy.ndarray]
            Stream function value at each point.
        """
        u, v = self.freestream.velocity(x, y)
        for bod in self.body:
            for el in bod:
                ut, vt = el.velocity(x, y, True)
                u = u + ut
                v = v + vt

        return u, v


def hess_smith_solver(freestream: FreestreamCondition,
                      airfoil:List[Geometry]) -> PanelSolution2D:
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
    # TODO: this needs to be modified to handle multiple surfaces
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

    return PanelSolution2D(fs, [src])


def main() -> None:
    """Run main function."""
    # set freestream conditions
    freestream = FreestreamCondition(U_inf=10, alpha=0)

    # get airfoil geometry
    npan = 101
    c = 2
    theta = np.linspace(2*np.pi, 0, npan+1)
    xr = 0.5*c*(np.cos(theta)+1)
    yr = 0.5*c*np.sin(theta)
    surf = Cylinder(radius=0.5*c)
    xb, yb = surf.xy_from_xi(np.linspace(-1, 1, npan+1))
    af = Geometry(name="Cylinder", x=xr, y=yr)

    # solve for panel strengths
    pan_sol = hess_smith_solver(freestream, [af])

    # # # reference results
    # # doublet = PointDoublet2D(xo=0.5*c, yo=0, angle=np.pi,
    # #                          strength=2*np.pi*fs.U_inf*(0.5*c)**2)
    # # u_ref, v_ref = (u_inf*np.ones(af.number_panels()),
    # #                 v_inf*np.ones(af.number_panels()))
    # # ud, vd = doublet.velocity(xc, yc, True)
    # # u_ref = u_ref + ud
    # # v_ref = v_ref + vd
    # # cp_ref = 1-(u_ref**2 + v_ref**2)/fs.U_inf**2

    # create mesh
    nptsx = 100
    nptsy = 100
    xg, yg = np.meshgrid(np.linspace(-3, 5, nptsx), np.linspace(-4, 4, nptsy))

    # plot geometry
    xp, yp, xc, yc, nx, ny, sx, sy = pan_sol.get_surface_geometry()[0]
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(xp, yp, "-ro")
    plt.plot(xc, yc, "rs")
    plt.quiver(xc, yc, nx, ny, scale=10, color="blue")
    plt.quiver(xc, yc, sx, sy, scale=10, color="blue")
    plt.title(f"{af.name} with {af.number_panels()} panels")
    plt.show()

    # # plot the velocity vectors on surface
    xc, yc, u, v, cp = pan_sol.get_surface_properties()[0]
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(xp, yp, "-r")
    plt.quiver(xc, yc, u, v, scale=100, color="orange")
    plt.show()

    # plot the pressure profile
    theta = np.arctan2(yc, xc-0.5*c)
    theta[theta < 0] = 2*np.pi + theta[theta < 0]
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.plot(theta*(180/np.pi), cp, "-b")
    plt.xticks(np.linspace(0, 360, 9))
    plt.xlabel(r"$\theta\ (^\circ)$")
    plt.ylabel(r"$c_p$")
    plt.show()

    # # plot the potentials
    potential = pan_sol.get_potential(xg, yg)
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.contour(xg, yg, potential, levels=60, colors="blue",
                linestyles="solid")
    plt.plot(xp, yp, "-r")
    plt.show()

    # plot the stream function
    stream_function = pan_sol.get_stream_function(xg, yg)
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.contour(xg, yg, stream_function, levels=60, colors="blue",
                linestyles="solid")
    plt.plot(xp, yp, "-r")
    plt.show()

    # plot the flow field
    ug, vg = pan_sol.get_velocity(xg, yg)
    fig = plt.figure()
    fig.set_figwidth(5)
    fig.set_figheight(5)
    plt.streamplot(xg, yg, ug, vg, broken_streamlines=False,
                   color=np.sqrt(ug**2+vg**2),
                   start_points=np.array([np.linspace(-3, -3, 40),
                                          np.linspace(-4, 4, 40)]).T,
                   linewidth=0.5, cmap="jet")
    plt.plot(xp, yp, "-k", linewidth=0.5)
    plt.show()


if __name__ == "__main__":
    main()
