#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils modeled in panel codes."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type

from pyPC.geometry import Geometry
from pyPC.camber import Naca4DigitCamber


class Naca4DigitAirfoilClassic(Geometry):
    """
    Classic NACA 4-digit airfoil.

    Attributes
    ----------
    max_camber : int
        Maximum camber parameter. 100 time the actual maximum camber per chord.
    max_camber_location : int
        Location of maximum camber parameter. 10 times the actual location per
        chord.
    max_thickness : int
        Maximum thickness parameter. 100 times the actual thickness per chord.
    scale : float
        Amount to scale the final airfoil.
    """

    def __init__(self, max_camber: int, max_camber_location: int,
                 max_thickness: int, scale: float) -> None:
        self._scale = scale
        self._max_camber = max_camber
        self._max_camber_loc = max_camber_location
        self._max_thickness = max_thickness
        self._delta_t = Naca4DigitThicknessClassic(max_thickness/100.0)
        self._yc = Naca4DigitCamber(m=max_camber/100.0,
                                    p=max_camber_location/10.0)

    @property
    def m(self) -> int:
        """Maximum camber parameter."""
        return self.max_camber

    @m.setter
    def m(self, m: int) -> None:
        self.max_camber = m

    @property
    def p(self) -> int:
        """Location of maximum camber parameter."""
        return self.max_camber_location

    @p.setter
    def p(self, p: int) -> None:
        self.max_camber_location = p

    @property
    def t(self) -> int:
        """Maximum thickness parameter."""
        return self.max_thickness

    @t.setter
    def t(self, t: int) -> None:
        self.max_thickness = t

    @property
    def max_camber(self) -> int:
        """Maximum camber parameter."""
        return self._max_camber

    @max_camber.setter
    def max_camber(self, max_camber: int) -> None:
        self._max_camber = max_camber
        self._yc.m = max_camber/100.0

    @property
    def max_camber_location(self) -> int:
        """Location of maximum camber parameter."""
        return self._max_camber_loc

    @max_camber_location.setter
    def max_camber_location(self, max_camber_loc: int) -> None:
        self._max_camber = max_camber_loc
        self._yc.p = max_camber_loc/10.0

    @property
    def max_thickness(self) -> int:
        """Maximum thickness parameter."""
        return self._max_thickness

    @max_thickness.setter
    def max_thickness(self, max_thickness: int) -> None:
        if max_thickness <= 0:
            raise ValueError("Maximum thickness must be non-zero and "
                             "positive.")
        self._delta_t.thickness(thickness=max_thickness/100.0)
        self._max_Dthickness = max_thickness

    @property
    def scale(self) -> float:
        """Scale factor for airfiol."""
        return self._scale

    @scale.setter
    def scale(self, scale: float) -> None:
        self._scale = scale

    @staticmethod
    def _convert_xi(xi: np_type.NDArray) -> np_type.NDArray:
        xic = np.asarray(xi).copy()
        sgn = np.ones_like(xic)
        sgn[xic < 0] = -1.0
        xic[xic < 0] = -xic[xic < 0]
        return xic, sgn

    def camber(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        xic, _ = self._convert_xi(xi)
        return self.scale*self._yc.y(xic)

    def thickness(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the amount of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        xic, _ = self._convert_xi(xi)
        return self.scale*self._delta_t.y(xic)

    def xy_from_xi(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                       np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            X-coordinate of point.
        numpy.ndarray
            Y-coordinate of point.
        """
        xic, sgn = self._convert_xi(xi)

        delta_t = self._delta_t.y(xic)
        yc = self._yc.y(xic)
        yc_p = self._yc.y_p(xic)
        denom = np.sqrt(1+yc_p**2)
        x = xic - sgn*delta_t*yc_p/denom
        y = yc + sgn*delta_t/denom
        return self.scale*x, self.scale*y

    def xy_p(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric rate of change of the x-coordinate of point.
        numpy.ndarray
            Parametric rate of change of the y-coordinate of point.
        """
        xic, sgn = self._convert_xi(xi)

        delta_t = self._delta_t.y(xic)
        delta_tp = self._delta_t.y_p(xic)
        yc_p = self._yc.y_p(xic)
        yc_pp = self._yc.y_pp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_p = 1.0 - sgn/denom*(delta_tp*yc_p + delta_t*yc_pp/denom**2)
        y_p = yc_p + sgn/denom*(delta_tp - delta_t*yc_p*yc_pp/denom**2)
        return sgn*self.scale*x_p, sgn*self.scale*y_p

    def xy_pp(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Return second derivative of the coordinates at parameter location.

        Notes
        -----
        Parameter goes from -1 (trailing edge lower surface) to +1 (trailing
        edge upper surface) with 0 representing the leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric second derivative of the x-coordinate of point.
        numpy.ndarray
            Parametric second derivative of the y-coordinate of point.
        """
        xic, sgn = self._convert_xi(xi)

        delta_t = self._delta_t.y(xic)
        delta_tp = self._delta_t.y_p(xic)
        delta_tpp = self._delta_t.y_pp(xic)
        yc_p = self._yc.y_p(xic)
        yc_pp = self._yc.y_pp(xic)
        yc_ppp = self._yc.y_ppp(xic)
        denom = np.sqrt(1+yc_p**2)
        x_pp = -sgn/denom*(delta_tpp*yc_p + (2*delta_tp*yc_pp
                                             + delta_t*yc_ppp)/denom**2
                           - 3*delta_t*yc_p*yc_pp**2/denom**4)
        y_pp = yc_pp + sgn/denom*(delta_tpp - (yc_p*(2*delta_tp*yc_pp
                                                     + delta_t*yc_ppp)
                                               - 2*delta_t*yc_pp**2)/denom**2
                                  - 3*delta_t*yc_pp**2/denom**4)
        return self.scale*x_pp, self.scale*y_pp
