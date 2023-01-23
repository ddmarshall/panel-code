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


class Naca5DigitCamber(Camber):
    """
    Base class for the NACA 5-digit airfoil camber.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber times 20.
    ci : float
        Ideal lift coefficient times 20/3.
    m : float
        Camber transition relative chord location.
    k1 : float
        Scale factor for camber.
    """

    def __init__(self, ci: float, p: float, m: float, k1: float) -> None:
        self._ci = (3.0/20.0)*ci
        self._p = p/20.0
        self._m = m
        self._k1 = k1

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return 20.0*self._p

    @p.setter
    def p(self, p: float) -> None:
        self._p_setter(p)

    @abstractmethod
    def _p_setter(self, p: float) -> None:
        pass

    @property
    def ci(self) -> float:
        """Ideal lift coefficient term."""
        return (20.0/3.0)*self._ci

    @ci.setter
    def ci(self, ci: float) -> None:
        self._ci_setter(ci)

    @abstractmethod
    def _ci_setter(self, ci: float) -> None:
        pass

    @property
    def m(self) -> float:
        """Camber transition relative chord location."""
        return self._m

    @property
    def k1(self) -> float:
        """Scale factor for camber."""
        return self._k1

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
            m = self.m
            return (self.k1/6)*(xi**3 - 3*m*xi**2 + m**2*(3-m)*xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1*self.m**3/6)*(1 - xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
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

        m = self.m

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.k1/6)*(3*xi**2 - 6*m*xi + m**2*(3-m))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -(self.k1*m**3/6)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= m, xi > m],
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
            return (self.k1)*(xi - self.m)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
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
            return self.k1*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.m, xi > self.m],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, self.m, 1.0]

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
        return self._p, self.y(self._p)


class Naca5DigitCamberClassic(Naca5DigitCamber):
    """
    Camber for the classic NACA 5-digit airfoils.

    The only valid values for the relative chord location of maximum camber
    term are 1, 2, 3, 4, and 5. The only valid value for the ideal lift
    coefficient term is 2.
    """

    def __init__(self, p: int) -> None:
        self._p_setter(p)
        super().__init__(2, p, self._m, self._k1)

    def _p_setter(self, p: int) -> None:
        if p == 1:
            self._m = 0.0580
            self._k1 = 361.4
        elif p == 2:
            self._m = 0.1260
            self._k1 = 51.64
        elif p == 3:
            self._m = 0.2025
            self._k1 = 15.957
        elif p == 4:
            self._m = 0.2900
            self._k1 = 6.643
        elif p == 5:
            self._m = 0.3910
            self._k1 = 3.230
        else:
            raise ValueError("Invalid NACA 5-Digit max. camber location: {p}.")

        self._p = p/20.0

    def _ci_setter(self, ci: int) -> None:
        if ci != 2:
            raise ValueError("Invalid NACA 5-Digit ideal lift coefficient: "
                             f"{ci}.")
        self._ci = (3.0/20.0)*ci


class Naca5DigitCamberEnhanced(Naca5DigitCamber):
    """
    Camber for the regular NACA 5-digit airfoils.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber.
    Cl_ideal : float
        Ideal lift coefficient.
    """

    def __init__(self, p: float, ci: float) -> None:
        # Need to bootstrap initialization
        self._ci = 0.3
        self._p_setter(p)
        super().__init__(ci, p, self._m, self._k1)

    def _p_setter(self, p: float) -> None:
        if p < 1 or p >= 6:
            raise ValueError("Invalid NACA 5-Digit max. camber location: "
                             f"{p}.")

        self._p = p/20.0

        def camber_slope(m: float) -> float:
            return 3*self._p**2 - 6*m*self._p + m**2*(3-m)

        root = root_scalar(lambda x: camber_slope(x),
                           bracket=[self._p, 2*self._p])
        self._m = root.root
        self._p = p
        self._determine_k1()

    def _ci_setter(self, ci: int) -> None:
        if ci < 1 or ci >= 4:
            raise ValueError("Invalid NACA 5-Digit ideal lift coefficient: "
                             f"{ci}.")
        self._ci = (3.0/20.0)*ci
        self._determine_k1()

    def _determine_k1(self):
        self._k1 = 6*self._ci/(-3/2*(1-2*self._m)*np.arccos(1-2*self._m)
                               + (4*self._m**2-4*self._m
                                  + 3)*np.sqrt(self._m*(1-self._m)))
