# -*- coding: utf-8 -*-
"""
All of the vortex type flows for panel methods.

This module contains all of the available vortex types for panel methods.
"""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.element_flow import PointElement2D, LineElementConstant2D


class PointVortex2D(PointElement2D):
    """Represents a point vortex in 2 dimensions."""

    def __init__(self, xo: float = 0, yo: float = 0,
                 strength: float = 1, angle:float = 0) -> None:
        super().__init__(xo=xo, yo=yo, angle=angle)
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
        rx, ry, _ = self._r_terms(xp, yp)
        angle = np.asarray(np.arctan2(ry, rx)-self.angle)
        angle[angle > np.pi] = 2*np.pi - angle[angle > np.pi]
        angle[angle <= -np.pi] = 2*np.pi + angle[angle <= -np.pi]

        return -self._strength_over_2pi*angle

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
        _, _, rmag2 = self._r_terms(xp, yp)

        return 0.5*self._strength_over_2pi*np.log(rmag2)

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
        return coef*ry, -coef*rx


class LineVortexConstant2D(LineElementConstant2D):
    """Represents a constant strength line vortex in 2 dimensions."""

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
        r2_i, r2_ip1, beta_i, beta_ip1 = self._get_I_terms(xip, etap, top)
        return -self._strength_over_2pi*self._get_I03(xip, etap, r2_i, r2_ip1,
                                                      beta_i, beta_ip1)

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
        r2_i, r2_ip1, beta_i, beta_ip1 = self._get_I_terms(xip, etap, top)
        return self._strength_over_2pi*self._get_I02(xip, etap, r2_i, r2_ip1,
                                                     beta_i, beta_ip1)

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
        r2_i, r2_ip1, beta_i, beta_ip1 = self._get_I_terms(xip, etap, top)
        uxi = self._strength_over_2pi*self._get_I01(beta_i, beta_ip1)
        ueta = -self._strength_over_2pi*self._get_I00(r2_i, r2_ip1)
        u, v = self._get_u_v(uxi, ueta)
        return u, v
