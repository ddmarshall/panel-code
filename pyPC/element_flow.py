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


@dataclass
class PointElement2D(ABC):
    """
    Base class for 2D point elements.

    Attributes
    ----------
    x0 : float
        X-coorinate of source origin.
    y0 : float
        Y-coorinate of source origin.
    angle : float
        Orientation angle of element in radians.
    """

    x0: float
    y0: float
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

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def _r_terms(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the radius terms needed for many calculations.

        Parameters
        ----------
        xp : np_type.NDArray
            X-coordinate of the point to calculate terms.
        yp : np_type.NDArray
            Y-coordinate of the point to calculate terms.

        Returns
        -------
        rx : np_type.NDArray
            X-component of the vector from element to point.
        ry : np_type.NDArray
            Y-component of the vector from element to point.
        rmag2 : np_type.NDArray
            Square of the distance from element to point.
        """
        rx = xp - self.x0
        ry = yp - self.y0
        rmag2 = rx**2 + ry**2
        return rx, ry, rmag2


@dataclass
class LineElement2D(ABC):
    """
    Base class for 2D point elements.

    Attributes
    ----------
    x0: float
        X-coorinate of source origin.
    y0: float
        Y-coorinate of source origin.
    """

    x0: Tuple[float, float]
    y0: Tuple[float, float]

    @abstractmethod
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

    @abstractmethod
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

    @abstractmethod
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

    def _getI0I1(self, xp: np_type.NDArray,
                 yp: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray,
                                               np_type.NDArray,
                                               float]:
        """
        Return the basic integrals and terms needed for line elements.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the I0 integral
        numpy.ndarray
            Value of the I1 integral
        numpy.ndarray
            Square of the distance from panel end to point
        numpy.ndarray
            Angle (in radians) between panel and panel end point
        numpy.ndarray
            xi-coordinate of points relative to the panel length
        numpy.ndarray
            eta-coordinate of points relative to the panel length
        float
            panel length
        """
        # pylint: disable=too-many-locals

        # calculate the panel geometry terms
        dxp = self.x0[1]-self.x0[0]
        dyp = self.y0[1]-self.y0[0]
        ell = np.sqrt(dxp**2 + dyp**2)

        # calculate the computational coordinates
        x1p = xp - self.x0[0]
        y1p = yp - self.y0[0]
        xip = (x1p*dxp + y1p*dyp)/ell
        etap = (-x1p*dyp + y1p*dxp)/ell

        # calculate the terms need to be returned
        r1_sqr = xip**2 + etap**2
        beta1 = np.arctan2(etap, xip)
        r2_sqr = (xip-ell)**2 + etap**2
        beta2 = np.arctan2(etap, xip-ell)
        I0 = 0.5*np.log(r1_sqr/r2_sqr)
        I1 = beta2 - beta1

        return I0, I1, r2_sqr, beta2, xip, etap, ell

    def _getI2(self, xp: np_type.NDArray,
               yp: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I2 itegral values.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the I2 integral
        """
        I0, I1, r2_sqr, _, xip, etap, ell = self._getI0I1(xp, yp)
        return ell*(xip*I0/ell + etap*I1/ell + 0.5*np.log(r2_sqr)-1)

    def _getI3(self, xp: np_type.NDArray,
               yp: np_type.NDArray) -> np_type.NDArray:
        """
        Return the I3 itegral values.

        Parameters
        ----------
        xp : numpy.ndarray
            X-coordinate of point to evaluate velocity.
        yp : numpy.ndarray
            Y-coordinate of point to evaluate velocity.

        Returns
        -------
        numpy.ndarray
            Value of the I3 integral
        """
        I0, I1, _, beta2, xip, etap, ell = self._getI0I1(xp, yp)
        return ell*(etap*I0/ell - xip*I1/ell + beta2)


@dataclass
class LineElementConstant2D(LineElement2D):
    """Base class for constant strength line elements in 2 dimensions."""

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
