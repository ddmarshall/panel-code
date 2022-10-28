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


class ApproximateLineSourceConstant2D():
    """Approximate constant strength 2D line source."""

    def __init__(self, x0: Tuple[float, float], y0: Tuple[float, float],
                 sigma: float, num_elements: int) -> None:

        self.x0 = x0
        self.y0 = y0
        self.sigma = sigma
        self.ne = num_elements

    def _setup_source(self) -> Tuple[np_type.NDArray, np_type.NDArray,
                                     PointSource2D]:
        xpan = np.linspace(self.x0[0], self.x0[1], self.ne)
        ypan = np.linspace(self.y0[0], self.y0[1], self.ne)

        ell = np.sqrt((self.x0[1]-self.x0[0])**2 + (self.y0[1]-self.y0[0])**2)
        angle = np.arctan2(self.y0[1]-self.y0[0], self.x0[1]-self.x0[0])
        strength = self.sigma*ell/(self.ne-1)

        source = PointSource2D(x0=self.x0[0], y0=self.y0[0],
                               strength=strength, angle=angle)

        return xpan, ypan, source

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
            source.x0 = xp
            source.y0 = yp
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
            source.x0 = xp
            source.y0 = yp
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
            source.x0 = xp
            source.y0 = yp
            utemp, vtemp = source.velocity(x, y)
            u = u + utemp
            v = v + vtemp

        return u, v
