# -*- coding: utf-8 -*-
"""
All of the doublet type flows for panel methods.

This module contains all of the available doublet types for panel methods.
"""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import PointElement2D, LineElementConstant2D


class PointDoublet2D(PointElement2D):
    """Represents a point vortex in 2 dimensions."""

    def __init__(self, xo: float = 0, yo: float = 0,
                 strength: float = 1, angle: float = 0) -> None:
        super().__init__(xo=xo, yo=yo, angle=angle)
        self.set_strength(strength)

    def potential(self, xp: np_type.NDArray, yp: np_type.NDArray,
                  top: bool = True) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Branch cut flag that does not affect this class.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        rx, ry, rmag2 = self._r_terms(xp, yp)
        nx, ny = self._orientation()

        return -self._strength_over_2pi*(rx*nx+ry*ny)/rmag2

    def stream_function(self, xp: np_type.NDArray, yp: np_type.NDArray,
                        top: bool = True) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Branch cut flag that does not affect this class.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        rx, ry, rmag2 = self._r_terms(xp, yp)
        nx, ny = self._orientation()

        return self._strength_over_2pi*(ry*nx-rx*ny)/rmag2

    def velocity(self, xp: np_type.NDArray, yp: np_type.NDArray,
                 top: bool = True) -> Tuple[np_type.NDArray,
                                            np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Branch cut flag that does not affect this class.

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


class LineDoubletConstant2D(LineElementConstant2D):
    """Represents a constant strength line doublet in 2 dimensions."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float],
                 strength:float = 1) -> None:
        super().__init__(xo=xo, yo=yo)
        self.set_strength(strength)

    def potential(self, xp: np_type.NDArray, yp: np_type.NDArray,
                  top: bool) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """
        xip, etap = self._get_xi_eta(xp, yp)
        _, _, beta_i, beta_ip1 = self._get_I_terms(xip, etap, top)
        return -self._strength_over_2pi*self._get_I01(beta_i, beta_ip1)

    def stream_function(self, xp: np_type.NDArray, yp: np_type.NDArray,
                        top: bool) -> np_type.NDArray:
        """
        Calculate the stream function at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """
        xip, etap = self._get_xi_eta(xp, yp)
        r2_i, r2_ip1, _, _ = self._get_I_terms(xip, etap, top)
        return -self._strength_over_2pi*self._get_I00(r2_i, r2_ip1)

    def velocity(self, xp: np_type.NDArray, yp: np_type.NDArray,
                 top: bool) -> Tuple[np_type.NDArray, np_type.NDArray]:
        """
        Calculate the induced velocity at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """
        xip, etap = self._get_xi_eta(xp, yp)
        r2_i, r2_ip1, _, _ = self._get_I_terms(xip, etap, top)
        uxi = self._strength_over_2pi*self._get_I04(etap, r2_i, r2_ip1)
        ueta = self._strength_over_2pi*self._get_I05(xip, r2_i, r2_ip1)
        u, v = self._get_u_v(uxi, ueta)
        return u, v
