# -*- coding: utf-8 -*-
"""
All of the source type flows for panel methods.

This module contains all of the available source types for panel methods.
"""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import PointElement2D


class PointSource2D(PointElement2D):
    """Represents a point source in 2 dimensions."""

    def __init__(self, x0: float = 0, y0: float = 0,
                 strength: float = 1) -> None:
        super().__init__(x0=x0, y0=y0)
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
        _, _, rmag2 = self._r_terms(xp, yp)

        return 0.5*self._strength_over_2pi*np.log(rmag2)

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
        rx, ry, _ = self._r_terms(xp, yp)

        return self._strength_over_2pi*np.arctan2(ry, rx)

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

        coef = self._strength_over_2pi/rmag2
        return coef*rx, coef*ry
