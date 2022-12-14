#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Classes that use point elements to approximate behaviour of elements.

These classes use point elements to construct the approximate behaviour of
exact higher dimensional elements.
"""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.source_flow import PointSource2D
from pyPC.vortex_flow import PointVortex2D
from pyPC.doublet_flow import PointDoublet2D


class ApproxLineSourceConstant2D():
    """Approximate constant strength 2D line source."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 sigma: float, num_elements: int) -> None:

        self.xo = xo
        self.yo = yo
        self.sigma = sigma
        self.ne = num_elements

    def potential(self, x: np_type.NDArray,
                  y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the potential.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        xpan, ypan, source = self._setup_source()

        potential = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            source.xo = xp
            source.yo = yp
            potential = potential + source.potential(x, y)

        return potential

    def stream_function(self, x: np_type.NDArray,
                        y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the stream function.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        xpan, ypan, source = self._setup_source()

        stream_function = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            source.xo = xp
            source.yo = yp
            stream_function = stream_function + source.stream_function(x, y)

        return stream_function

    def velocity(self, x: np_type.NDArray,
                 y: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the velocity vector components.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        xpan, ypan, source = self._setup_source()

        u = np.zeros_like(x)
        v = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            source.xo = xp
            source.yo = yp
            utemp, vtemp = source.velocity(x, y)
            u = u + utemp
            v = v + vtemp

        return u, v

    def _setup_source(self) -> Tuple[np_type.NDArray, np_type.NDArray,
                                     PointSource2D]:
        """
        Set up the point source for calculations.

        Returns
        -------
        numpy.ndarray
            X-coordinates of the point elements.
        numpy.ndarray
            Y-coordinates of the point elements.
        PointSource2D
            Point source class configured for analysis.
        """
        xpan = np.linspace(self.xo[0], self.xo[1], self.ne)
        ypan = np.linspace(self.yo[0], self.yo[1], self.ne)

        ell = np.sqrt((self.xo[1]-self.xo[0])**2 + (self.yo[1]-self.yo[0])**2)
        angle = np.arctan2(self.yo[1]-self.yo[0], self.xo[1]-self.xo[0])
        strength = self.sigma*ell/(self.ne-1)

        source = PointSource2D(xo=self.xo[0], yo=self.yo[0],
                               strength=strength, angle=angle)

        return xpan, ypan, source


class ApproxLineVortexConstant2D():
    """Approximate constant strength 2D line vortex."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 gamma: float, num_elements: int) -> None:

        self.xo = xo
        self.yo = yo
        self.gamma = gamma
        self.ne = num_elements

    def potential(self, x: np_type.NDArray,
                  y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the potential.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        xpan, ypan, vortex = self._setup_vortex()

        potential = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            vortex.xo = xp
            vortex.yo = yp
            potential = potential + vortex.potential(x, y)

        return potential

    def stream_function(self, x: np_type.NDArray,
                        y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the stream function.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        xpan, ypan, vortex = self._setup_vortex()

        stream_function = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            vortex.xo = xp
            vortex.yo = yp
            stream_function = stream_function + vortex.stream_function(x, y)

        return stream_function

    def velocity(self, x: np_type.NDArray,
                 y: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the velocity vector components.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        xpan, ypan, vortex = self._setup_vortex()

        u = np.zeros_like(x)
        v = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            vortex.xo = xp
            vortex.yo = yp
            utemp, vtemp = vortex.velocity(x, y)
            u = u + utemp
            v = v + vtemp

        return u, v

    def _setup_vortex(self) -> Tuple[np_type.NDArray, np_type.NDArray,
                                     PointVortex2D]:
        """
        Set up the point vortex for calculations.

        Returns
        -------
        numpy.ndarray
            X-coordinates of the point elements.
        numpy.ndarray
            Y-coordinates of the point elements.
        PointVortex2D
            Point vortex class configured for analysis.
        """
        xpan = np.linspace(self.xo[0], self.xo[1], self.ne)
        ypan = np.linspace(self.yo[0], self.yo[1], self.ne)

        ell = np.sqrt((self.xo[1]-self.xo[0])**2 + (self.yo[1]-self.yo[0])**2)
        angle = np.arctan2(self.yo[1]-self.yo[0], self.xo[1]-self.xo[0])
        strength = self.gamma*ell/(self.ne-1)

        vortex = PointVortex2D(xo=self.xo[0], yo=self.yo[0],
                               strength=strength, angle=angle)

        return xpan, ypan, vortex


class ApproxLineDoubletConstant2D():
    """Approximate constant strength 2D line doublet."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 mu: float, num_elements: int) -> None:

        self.xo = xo
        self.yo = yo
        self.mu = mu
        self.ne = num_elements

    def potential(self, x: np_type.NDArray,
                  y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the potential.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        xpan, ypan, doublet = self._setup_doublet()

        potential = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            doublet.xo = xp
            doublet.yo = yp
            potential = potential + doublet.potential(x, y)

        return potential

    def stream_function(self, x: np_type.NDArray,
                        y: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the stream function.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        xpan, ypan, doublet = self._setup_doublet()

        stream_function = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            doublet.xo = xp
            doublet.yo = yp
            stream_function = stream_function + doublet.stream_function(x, y)

        return stream_function

    def velocity(self, x: np_type.NDArray,
                 y: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the velocity vector components.

        Parameters
        ----------
        x : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        xpan, ypan, doublet = self._setup_doublet()

        u = np.zeros_like(x)
        v = np.zeros_like(x)
        for xp, yp in zip(xpan, ypan):
            doublet.xo = xp
            doublet.yo = yp
            utemp, vtemp = doublet.velocity(x, y)
            u = u + utemp
            v = v + vtemp

        return u, v

    def _setup_doublet(self) -> Tuple[np_type.NDArray, np_type.NDArray,
                                      PointDoublet2D]:
        """
        Set up the point doublet for calculations.

        Returns
        -------
        numpy.ndarray
            X-coordinates of the point elements.
        numpy.ndarray
            Y-coordinates of the point elements.
        PointDoublet2D
            Point doublet class configured for analysis.
        """
        xpan = np.linspace(self.xo[0], self.xo[1], self.ne)
        ypan = np.linspace(self.yo[0], self.yo[1], self.ne)

        # doublet is orientied in the xi-direction, so need to rotate angle
        angle = 0.5*np.pi+np.arctan2(self.yo[1]-self.yo[0],
                                     self.xo[1]-self.xo[0])
        ell = np.sqrt((self.xo[1]-self.xo[0])**2 + (self.yo[1]-self.yo[0])**2)
        strength = self.mu*ell/(self.ne-1)

        doublet = PointDoublet2D(xo=self.xo[0], yo=self.yo[0],
                                 strength=strength, angle=angle)

        return xpan, ypan, doublet
