#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoil camber lines."""

from abc import ABC, abstractmethod
from typing import List

import numpy as np
import numpy.typing as np_type

from scipy.optimize import root_scalar


class CamberBase(ABC):
    """Base class for camber lines."""

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
    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """


class Naca5DigitCamberBase(CamberBase):
    """
    Base class for the NACA 5-digit airfoil camber.

    Attributes
    ----------
    m : float
        Camber transition relative chord location.
    k1 : float
        Scale factor for camber.
    """

    def __init__(self, m: float, k1: float) -> None:
        self._m = m
        self._k1 = k1

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
        return [self.m]


class Naca5DigitCamberClassic(Naca5DigitCamberBase):
    """
    Camber for the classic NACA 5-digit airfoils.

    Attributes
    ----------
    camber_location : int
        Relative chord location of maximum camber. The only valid values are 1,
        2, 3, 4, and 5.
    """

    def __init__(self, camber_loc: int) -> None:
        self.camber_location = camber_loc
        super().__init__(m=self.m, k1=self.k1)

    @property
    def camber_location(self) -> int:
        """Relative chord location of maximum camber."""
        return self._camber_location

    @camber_location.setter
    def camber_location(self, camber_loc: int) -> None:
        if camber_loc == 1:
            self._m = 0.0580
            self._k1 = 361.4
        elif camber_loc == 2:
            self._m = 0.1260
            self._k1 = 51.64
        elif camber_loc == 3:
            self._m = 0.2025
            self._k1 = 15.957
        elif camber_loc == 4:
            self._m = 0.2900
            self._k1 = 6.643
        elif camber_loc == 5:
            self._m = 0.3910
            self._k1 = 3.230
        else:
            raise ValueError("Invalid NACA 5-Digit max. camber location: "
                             f"{camber_loc}.")

        self._camber_location = camber_loc


class Naca5DigitCamberEnhanced(Naca5DigitCamberBase):
    """
    Camber for the regular NACA 5-digit airfoils.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber.
    Cl_ideal : float
        Ideal lift coefficient.
    """

    def __init__(self, p: float, Cl_ideal: float) -> None:
        self._m = 0.0
        self._k1 = 0.0
        self._Cl_ideal = 1  # Need to bootstrap initialization
        self.p = p
        self.Cl_ideal = Cl_ideal

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

    @p.setter
    def p(self, p) -> float:
        root = root_scalar(lambda x: self._camber_slope(p, -6, x),
                           bracket=[p, 2*p])
        self._m = root.root
        self._p = p
        # This sets the new k1 since m changed
        self.Cl_ideal = self.Cl_ideal

    @property
    def Cl_ideal(self) -> float:
        """Ideal lift coefficient."""
        return self._Cl_ideal

    @Cl_ideal.setter
    def Cl_ideal(self, Cl_ideal: float) -> float:
        self._Cl_ideal = Cl_ideal
        self._k1 = self._determine_k1(Cl_ideal, self._m)

    @staticmethod
    def _camber_slope(xi: float, k1: float, m: float) -> float:
        return k1/6*(3*xi**2 - 6*m*xi + m**2*(3-m))

    @staticmethod
    def _determine_k1(Cl_ideal: float, m: float) -> float:
        return 6*Cl_ideal/(-3/2*(1-2*m)*np.arccos(1-2*m)
                           + (4*m**2-4*m+3)*np.sqrt(m*(1-m)))


class Naca5DigitCamberReflexedBase:
    """
    Base class for the NACA 5-digit reflexed airfoil camber.

    Attributes
    ----------
    m : float
        Camber transition relative chord location.
    k1 : float
        First scale factor for camber.
    k2 : float
        Second scale factor for camber.
    """

    def __init__(self, m: float, k1: float, k2: float) -> None:
        self._m = m
        self._k1 = k1
        self._k2 = k2

    @property
    def m(self) -> float:
        """Camber transition relative chord location."""
        return self._m

    @property
    def k1(self) -> float:
        """First scale factor for camber."""
        return self._k1

    @property
    def k2(self) -> float:
        """Second scale factor for camber."""
        return self._k2

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

        m = self.m
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*((xi-m)**3 - k2ok1*(1-m)**3*xi + m**3*(1-xi))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(k2ok1*(xi-m)**3 - k2ok1*(1-m)**3*xi + m**3*(1-xi))

        return np.piecewise(xi, [xi <= m, xi > m],
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
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*(xi-m)**2 - k2ok1*(1-m)**3 - m**3)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*k2ok1*(xi-m)**2 - k2ok1*(1-m)**3 - m**3)

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
            return (self.k2)*(xi - self.m)

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
            return self.k2*np.ones_like(xi)

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
        return [self.m]


class Naca5DigitCamberReflexedClassic(Naca5DigitCamberReflexedBase):
    """
    Camber for the classic NACA 5-digit reflexed airfoils.

    Attributes
    ----------
    camber_location : int
        Relative chord location of maximum camber. The only valid values are 1,
        2, 3, 4, and 5.
    """

    def __init__(self, camber_loc: int) -> None:
        self.camber_location = camber_loc
        super().__init__(m=self.m, k1=self.k1, k2=self.k2)

    @property
    def camber_location(self) -> int:
        """Relative chord location of maximum camber."""
        return self._camber_location

    @camber_location.setter
    def camber_location(self, camber_loc: int) -> None:
        if camber_loc == 2:
            self._m = 0.1300
            self._k1 = 51.99
            self._k2 = 0.03972036
        elif camber_loc == 3:
            self._m = 0.2170
            self._k1 = 15.793
            self._k2 = 0.10691861
        elif camber_loc == 4:
            self._m = 0.3180
            self._k1 = 6.520
            self._k2 = 0.197556
        elif camber_loc == 5:
            self._m = 0.4410
            self._k1 = 3.191
            self._k2 = 0.4323805
        else:
            raise ValueError("Invalid NACA 5-Digit reflexed max. camber "
                             f"location: {camber_loc}.")

        self._camber_location = camber_loc


class Naca5DigitCamberReflexedEnhanced(Naca5DigitCamberReflexedBase):
    """
    Camber for the regular NACA 5-digit reflexed airfoils.

    Attributes
    ----------
    p : float
        Relative chord location of maximum camber.
    Cl_ideal : float
        Ideal lift coefficient.
    """

    def __init__(self, p: float, Cl_ideal: float) -> None:
        self._m = 0.0
        self._k1 = 0.0
        self._k2 = 1.0
        self._Cl_ideal = 1  # Need to bootstrap initialization
        self.p = p
        self.Cl_ideal = Cl_ideal
        super().__init__(m=self.m, k1=self.k1, k2=self.k2)

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

    @p.setter
    def p(self, p) -> float:
        def m_fun(m: float) -> float:
            k1 = 1.0
            k2ok1 = self._k2ok1(m, p)
            Cl_id = self._Cl_id(m, k1, k2ok1)
            return self._Cmc4(m, k1, k2ok1, Cl_id)

        root = root_scalar(m_fun, bracket=[p, 3*p])
        self._m = root.root
        self._p = p
        # This sets the new k1 and k2 since m and p changed
        self.Cl_ideal = self.Cl_ideal

    @property
    def Cl_ideal(self) -> float:
        """Ideal lift coefficient."""
        return self._Cl_ideal

    @Cl_ideal.setter
    def Cl_ideal(self, Cl_ideal: float) -> float:
        k2ok1 = self._k2ok1(self.m, self.p)
        self._k1 = Cl_ideal/self._Cl_id(self.m, 1, k2ok1)
        self._k2 = k2ok1*self._k1
        self._Cl_ideal = Cl_ideal

    @staticmethod
    def _Cl_id(m: float, k1: float, k2ok1: float) -> float:
        return (k1/12)*(3*k2ok1*(2*m-1)*np.pi
                        + 3*(1-k2ok1)*(2*m-1)*np.arccos(1-2*m)
                        + 2*(1-k2ok1)*(4*m**2-4*m+3)*np.sqrt(m*(1-m)))

    @staticmethod
    def _k2ok1(m: float, p: float) -> float:
        return (3*(p-m)**2 - m**3)/(1 - m)**3

    @staticmethod
    def _Cmc4(m: float, k1: float, k2ok1: float, Cl_id) -> float:
        return -0.25*Cl_id + (k1/192)*(3*k2ok1*np.pi
                                       + 3*(1-k2ok1)*np.arccos(1-2*m)
                                       + 2*(1-k2ok1)*(1-2*m)*(8*m**2-8*m-3)
                                       * np.sqrt(m*(1-m)))
