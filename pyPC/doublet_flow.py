# -*- coding: utf-8 -*-
"""
All of the doublet type flows for panel methods.

This module contains all of the available doublet types for panel methods.
"""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import PointElement2D


class PointDoublet2D(PointElement2D):
    """Represents a point vortex in 2 dimensions."""

    def __init__(self, x0: float = 0, y0: float = 0,
                 strength: float = 1, angle: float = 0) -> None:
        super().__init__(x0=x0, y0=y0, angle=angle)
        self.set_strength(strength)

    def potential(self, xp: np_type.NDArray,
                  yp: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        rx, ry, rmag2 = self._r_terms(xp, yp)
        nx, ny = self._orientation()

        return -self._strength_over_2pi*(rx*nx+ry*ny)/rmag2

    def stream_function(self, xp: np_type.NDArray,
                        yp: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        rx, ry, rmag2 = self._r_terms(xp, yp)
        nx, ny = self._orientation()

        return self._strength_over_2pi*(ry*nx-rx*ny)/rmag2

    def velocity(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
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
        rx, ry, rmag2 = self._r_terms(xp, yp)
        nx, ny = self._orientation()

        coef = self._strength_over_2pi/rmag2**2
        term1 = rx**2-ry**2
        term2 = 2*rx*ry

        return (coef*(nx*term1+ny*term2), coef*(nx*term2-ny*term1))

    def _orientation(self) -> Tuple[float, float]:
        return np.cos(self.angle), np.sin(self.angle)
