#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoil camber lines."""

from abc import abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type

from scipy.optimize import root_scalar

from pyPC.airfoil.airfoil import Curve


class Camber(Curve):
    """
    Base class for camber lines.

    The camber lines will have a parameterization from 0 to 1 and are not
    defined outside of that range.
    """

    def xy(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                               np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

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
        return xi, self.y(xi)

    def xy_p(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

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
        return np.ones_like(xi), self.y_p(xi)

    def xy_pp(self, xi: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at parameter location.

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
        return np.zeros_like(xi), self.y_pp(xi)

    @abstractmethod
    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """

    @abstractmethod
    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """

    @abstractmethod
    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """

    @abstractmethod
    def y_ppp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """

    @abstractmethod
    def max_camber(self) -> Tuple[float, float]:
        """
        Return chord location of maximum camber and the maximum camber.

        Returns
        -------
        float
            Chord location of maximum camber.
        float
            Maximum camber.
        """


class Naca4DigitCamber(Camber):
    """
    Camber for the NACA 4-digit airfoils.

    Attributes
    ----------
    m : float
        Maximum amount of camber per chord length times 100.
    p : float
        Relative chord location of maximum camber times 10.
    """

    def __init__(self, m: float, p: float) -> None:
        self._m = m/100.0
        self._p = p/10.0

    @property
    def m(self) -> float:
        """Maximum amount of camber."""
        return 100.0*self._m

    @m.setter
    def m(self, m: float) -> None:
        self._m = m/100.0

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return 10.0*self._p

    @p.setter
    def p(self, p: float) -> None:
        self._p = p/10.0

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return (self._m/self._p**2)*(2*self._p*xi - xi**2)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return (self._m/(1-self._p)**2)*(1
                                                 + 2*self._p*(xi - 1) - xi**2)

        return np.piecewise(xi, [xi <= self._p, xi > self._p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return 2*(self._m/self._p**2)*(self._p - xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return 2*(self._m/(1-self._p)**2)*(self._p - xi)

        return np.piecewise(xi, [xi <= self._p, xi > self._p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return -2*(self._m/self._p**2)*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(xi)
            else:
                return -2*(self._m/(1-self._p)**2)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= self._p, xi > self._p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def y_ppp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self._p, xi > self._p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, self._p, 1.0]

    def max_camber(self) -> Tuple[float, float]:
        """
        Return chord location of maximum camber and the maximum camber.

        Returns
        -------
        float
            Chord location of maximum camber.
        float
            Maximum camber.
        """
        return self._p, self._m
