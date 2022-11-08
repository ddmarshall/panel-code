#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
All of the base element classes for panel methods.

This module contains all of the base element classes for panel methods.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Tuple

import numpy as np
import numpy.typing as np_type


class Element2D(ABC):
    """Fundamental interface for any 2-dimensional element."""

    @abstractmethod
    def potential(self, xp: np_type.NDArray,
                  yp: np_type.NDArray, top: bool) -> np_type.NDArray:
        """
        Calculate the velocity potential at given point.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coorindate of point to evaluate potential.
        yp : numpy.ndarray
            Y-coorindate of point to evaluate potential.
        top : bool
            Flag indicating whether the top or bottom of the branch cut (if
            one exists) should be returned when the input point is on the
            branch cut.

        Returns
        -------
        numpy.ndarray
            Value of the velocity potential.
        """

    @abstractmethod
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
            Flag indicating whether the top or bottom of the branch cut (if
            one exists) should be returned when the input point is on the
            branch cut.

        Returns
        -------
        numpy.ndarray
            Value of the stream function.
        """

    @abstractmethod
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
            Flag indicating whether the top or bottom of the branch cut (if
            one exists) should be returned when the input point is on the
            branch cut.

        Returns
        -------
        numpy.ndarray
            Value of the x-velocity.
        numpy.ndarray
            Value of the y-velocity.
        """


@dataclass
class PointElement2D(Element2D):
    """
    Base class for 2D point elements.

    Attributes
    ----------
    xo : float
        X-coorinate of source origin.
    yo : float
        Y-coorinate of source origin.
    angle : float
        Orientation angle of element in radians.
    """

    xo: float
    yo: float
    angle: float
    _strength_over_2pi: float = field(init=False)

    def __post_init__(self) -> None:
        """Configure the strength base parameter."""
        self._strength_over_2pi = 1/(2*np.pi)

    def get_strength(self) -> float:
        """
        Return the strength of the point element.

        Returns
        -------
        float
            Point element strength.
        """
        return (2*np.pi)*self._strength_over_2pi

    def set_strength(self, strength: float) -> None:
        """
        Set the strength of the point element.

        Parameters
        ----------
        strength : float
            Point element strength.
        """
        self._strength_over_2pi = strength/(2*np.pi)

    def _r_terms(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the radius terms needed for many calculations.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of the point to calculate terms.
        yp : numpy.ndarray
            Y-coordinate of the point to calculate terms.

        Returns
        -------
        rx : numpy.ndarray
            X-component of the vector from element to point.
        ry : numpy.ndarray
            Y-component of the vector from element to point.
        rmag2 : numpy.ndarray
            Square of the distance from element to point.
        """
        rx = xp - self.xo
        ry = yp - self.yo
        rmag2 = rx**2 + ry**2
        return rx, ry, rmag2


class LineElement2D(Element2D):
    """Base class for 2D point elements."""

    def __init__(self, xo: Tuple[float, float], yo: Tuple[float, float]):
        self.set_panel_coordinates(xo, yo)

    def set_panel_coordinates(self, xo: Tuple[float, float],
                              yo: Tuple[float, float]) -> None:
        """
        Set the coordinates for the end points of the panel.

        Parameters
        ----------
        xo : Tuple[float, float]
            X-coordinates of the start and end of panel.
        yo : Tuple[float, float]
            Y-coordinates of the start and end of panel.
        """
        self._xo = xo
        self._yo = yo
        self._sx = xo[1] - xo[0]
        self._sy = yo[1] - yo[0]
        self._ell = np.sqrt(self._sx**2 + self._sy**2)
        self._sx = self._sx/self._ell
        self._sy = self._sy/self._ell
        self._nx = -self._sy
        self._ny = self._sx
        self._xc = 0.5*(xo[1]+xo[0])
        self._yc = 0.5*(yo[1]+yo[0])

    def get_panel_xo(self) -> Tuple[float, float]:
        """
        Return the x-coordinates of the panel end points.

        Returns
        -------
        float
            X-coordinates of the panel end points.
        """
        return self._xo

    def get_panel_yo(self) -> Tuple[float, float]:
        """
        Return the y-coordinates of the panel end points.

        Returns
        -------
        float
            Y-coordinates of the panel end points.
        """
        return self._yo

    def get_panel_start(self) -> Tuple[float, float]:
        """
        Return the coordinates of the start of the panel.

        Returns
        -------
        float
            X-coordinate of the panel start.
        float
            Y-coordinate of the panel start.
        """
        return self._xo[0], self._yo[0]

    def get_panel_end(self) -> Tuple[float, float]:
        """
        Return the coordinates of the end of the panel.

        Returns
        -------
        float
            X-coordinate of the panel end.
        float
            Y-coordinate of the panel end.
        """
        return self._xo[1], self._yo[1]

    def get_panel_tangent(self) -> Tuple[float, float]:
        """
        Return the unit normal to the panel.

        Returns
        -------
        float
            X-component of the panel normal.
        float
            Y-component of the panel normal.
        """
        return self._sx, self._sy

    def get_panel_normal(self) -> Tuple[float, float]:
        """
        Return the unit normal to the panel.

        Returns
        -------
        float
            X-component of the panel normal.
        float
            Y-component of the panel normal.
        """
        return self._nx, self._ny

    def get_panel_collo_point(self) -> Tuple[float, float]:
        """
        Return the collocation point for the panel.

        Returns
        -------
        float
            X-component of the panel collocation point.
        float
            Y-component of the panel collocation point.
        """
        return self._xc, self._yc

    def get_panel_length(self) -> float:
        """
        Return the length of the panel.

        Returns
        -------
        float
            Panel length.
        """
        return self._ell

    def get_panel_angle(self) -> float:
        """
        Return the angle of the panel.

        Returns
        -------
        float
            Angle of panel.
        """
        return np.arctan2(self._yo[1]-self._yo[0], self._xo[1]-self._xo[0])

    def _get_xi_eta(self, xp: np_type.NDArray,
                    yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Return the xi and eta coordinates for line elements.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            xi-coordinate of points in panel coordinate system
        numpy.ndarray
            eta-coordinate of points in panel coordinate system
        """
        # calculate the computational coordinates
        x1p = xp - self._xo[0]
        y1p = yp - self._yo[0]
        xip = x1p*self._sx + y1p*self._sy
        etap = x1p*self._nx + y1p*self._ny

        return xip, etap

    def _get_u_v(self, uxi: np_type.NDArray,
                 ueta: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate the Cartesian velocity components.

        Parameters
        ----------
        uxi : numpy.ndarray
            Velocity in the xi-coordinate direction.
        ueta : numpy.ndarray
            Velocity in the eta-coordinate direction.

        Returns
        -------
        u : numpy.ndarray
            Velocity in the x-coordinate direction.
        v : numpy.ndarray
            Velocity in the y-coordinate direction.
        """
        u = uxi*self._sx + ueta*self._nx
        v = uxi*self._sy + ueta*self._ny
        return u, v

    def _get_I_terms(self, xip: np_type.NDArray,
                     etap: np_type.NDArray,
                     top: bool) -> Tuple[np_type.NDArray, np_type.NDArray,
                                         np_type.NDArray, np_type.NDArray]:
        """
        Return the basic integral terms needed for line elements.

        Parameters
        ----------
        xip : numpy.ndarray
            Xi-coordinate of point in panel coordinate system
        etap : numpy.ndarray
            Y-coordinate of point to evaluate terms.
        top : bool
            Flag indicating whether the top (eta>0) or bottom (eta<0) should
            be returned when the input point is collinear with panel.

        Returns
        -------
        numpy.ndarray
            Square of the distance from panel start to point
        numpy.ndarray
            Square of the distance from panel end to point
        numpy.ndarray
            Angle (in radians) between panel and panel start point
        numpy.ndarray
            Angle (in radians) between panel and panel end point
        """
        # calculate the terms need to be returned
        r1_sqr = np.asarray(xip**2 + etap**2)
        beta1 = np.asarray(np.arctan2(etap, xip))
        r2_sqr = np.asarray((xip-self._ell)**2 + etap**2)
        beta2 = np.asarray(np.arctan2(etap, xip-self._ell))

        # need to handle branch cut.
        # Find indexes that represent on branch cut, spec_idx
        # Find indexes that represent points on the panel, p_idx
        spec_idx = np.asarray(np.abs(etap) < 1e-10)
        l0_idx = np.logical_and(spec_idx, np.asarray(xip < 0))
        p_idx = np.logical_and(np.logical_and(np.logical_not(l0_idx),
                                              np.asarray(xip < self._ell)),
                               spec_idx)
        beta1[p_idx] = 0
        if top:
            beta1[l0_idx] = np.pi
            beta2[l0_idx] = np.pi
            beta2[p_idx] = np.pi
        else:
            beta1[l0_idx] = -np.pi
            beta2[l0_idx] = -np.pi
            beta2[p_idx] = -np.pi

        return r1_sqr, r2_sqr, beta1, beta2

    def _get_I00(self, r2_i: np_type.NDArray,
                 r2_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,0 integral values.

        Parameters
        ----------
        r2_i : numpy.ndarray
            Squared distance from panel start to point of interest.
        r2_ip1 : numpy.ndarray
            Squared distance from panel end to point of interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,0 integral
        """
        return 0.5*np.log(r2_i/r2_ip1)

    def _get_I01(self, beta_i: np_type.NDArray,
                 beta_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,1 integral values.

        Parameters
        ----------
        beta_i : numpy.ndarray
            Angle (in radians) of vector connecting panel start to point of
            interest.
        beta_ip1 : numpy.ndarray
            Angle (in radians) of vector connecting panel end to point of
            interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,1 integral
        """
        return beta_ip1 - beta_i

    # pylint: disable=too-many-arguments
    def _get_I02(self, xip: np_type.NDArray, etap: np_type.NDArray,
                 r2_i: np_type.NDArray, r2_ip1: np_type.NDArray,
                 beta_i: np_type.NDArray,
                 beta_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,2 integral values.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.
        r2_i : numpy.ndarray
            Squared distance from panel start to point of interest.
        r2_ip1 : numpy.ndarray
            Squared distance from panel end to point of interest.
        beta_i : numpy.ndarray
            Angle (in radians) of vector connecting panel start to point of
            interest.
        beta_ip1 : numpy.ndarray
            Angle (in radians) of vector connecting panel end to point of
            interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,2 integral
        """
        I00 = self._get_I00(r2_i, r2_ip1)
        I01 = self._get_I01(beta_i, beta_ip1)
        return xip*I00 + etap*I01 + self._ell*(0.5*np.log(r2_ip1)-1)

    # pylint: disable=too-many-arguments
    def _get_I03(self, xip: np_type.NDArray, etap: np_type.NDArray,
                 r2_i: np_type.NDArray, r2_ip1: np_type.NDArray,
                 beta_i: np_type.NDArray,
                 beta_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,3 integral values.

        Parameters
        ----------
        xip : numpy.ndarray
            Xi-coordinate of point to evaluate velocity.
        etap : numpy.ndarray
            Eta-coordinate of point to evaluate velocity.
        r2_i : numpy.ndarray
            Squared distance from panel start to point of interest.
        r2_ip1 : numpy.ndarray
            Squared distance from panel end to point of interest.
        beta_i : numpy.ndarray
            Angle (in radians) of vector connecting panel start to point of
            interest.
        beta_ip1 : numpy.ndarray
            Angle (in radians) of vector connecting panel end to point of
            interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,3 integral
        """
        I00 = self._get_I00(r2_i, r2_ip1)
        I01 = self._get_I01(beta_i, beta_ip1)
        return etap*I00 - xip*I01 + self._ell*beta_ip1

    def _get_I04(self, etap: np_type.NDArray, r2_i: np_type.NDArray,
                 r2_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,4 integral values.

        Parameters
        ----------
        etap : numpy.ndarray
            Eta-coordinate of point to evaluate velocity.
        r2_i : numpy.ndarray
            Squared distance from panel start to point of interest.
        r2_ip1 : numpy.ndarray
            Squared distance from panel end to point of interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,4 integral
        """
        return etap*(1/r2_ip1 - 1/r2_i)

    def _get_I05(self, xip: np_type.NDArray, r2_i: np_type.NDArray,
                 r2_ip1: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I0,5 integral values.

        Parameters
        ----------
        xip : numpy.ndarray
            Xi-coordinate of point to evaluate velocity.
        r2_i : numpy.ndarray
            Squared distance from panel start to point of interest.
        r2_ip1 : numpy.ndarray
            Squared distance from panel end to point of interest.

        Returns
        -------
        numpy.ndarray
            Value of the I0,5 integral
        """
        return xip/r2_i - (xip-self._ell)/r2_ip1


class LineElementConstant2D(LineElement2D):
    """Base class for constant strength line elements in 2 dimensions."""

    def __init__(self, xo: Tuple[float, float],
                 yo: Tuple[float, float], strength: float = 1) -> None:
        self._strength_over_2pi = strength/(2*np.pi)
        super().__init__(xo, yo)

    def get_strength(self) -> float:
        """
        Return the strength of the point element.

        Returns
        -------
        float
            Point element strength.
        """
        return (2*np.pi)*self._strength_over_2pi

    def set_strength(self, strength: float) -> None:
        """
        Set the strength of the point element.

        Parameters
        ----------
        strength : float
            Point element strength.
        """
        self._strength_over_2pi = strength/(2*np.pi)
