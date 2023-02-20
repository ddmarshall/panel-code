#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoil camber lines."""

from abc import abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type

from scipy.optimize import root_scalar

from pyPC.airfoil.curve import Curve


class Camber(Curve):
    """
    Base class for camber lines.

    The camber lines will have a parameterization from 0 to 1 and are not
    defined outside of that range.
    """

    def xy(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                              np_type.NDArray]:
        """
        Calculate the coordinates of geometry at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            X-coordinate of point.
        numpy.ndarray
            Y-coordinate of point.
        """
        return t, self._y(t)

    def xy_t(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                np_type.NDArray]:
        """
        Calculate rates of change of the coordinates at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric rate of change of the x-coordinate of point.
        numpy.ndarray
            Parametric rate of change of the y-coordinate of point.
        """
        return np.ones_like(t), self._y_t(t)

    def xy_tt(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                 np_type.NDArray]:
        """
        Calculate second derivative of the coordinates at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric second derivative of the x-coordinate of point.
        numpy.ndarray
            Parametric second derivative of the y-coordinate of point.
        """
        return np.zeros_like(t), self._y_tt(t)

    def xy_ttt(self, t: np_type.NDArray) -> Tuple[np_type.NDArray,
                                                  np_type.NDArray]:
        """
        Calculate third derivative of the coordinates at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Parametric third derivative of the x-coordinate of point.
        numpy.ndarray
            Parametric third derivative of the y-coordinate of point.
        """
        return np.zeros_like(t), self._y_ttt(t)

    def k_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the rate of change of curvature at parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Rate of change of curvature of surface at point.
        """
        xt, yt = self.xy_t(t)
        xtt, ytt = self.xy_tt(t)
        xttt, yttt = self.xy_ttt(t)
        k = self.k(t)
        term1 = xt**2 + yt**2
        p = xt/np.sqrt(term1)
        q = yt/np.sqrt(term1)

        # return ((yttt*p - yttt*q)/term1
        #         - k*(xtt*xt + ytt*yt)/term1**2)*(k*term1 - 2*(xtt*q - ytt*p))
        return (yttt*p - xttt*q)/term1 - 3*k*(xtt*xt + ytt*yt)/term1

    @abstractmethod
    def max_camber_parameter(self) -> float:
        """
        Return parameter where the camber is maximum.

        Returns
        -------
        float
            Parameter where camber is maximum.
        """

    @abstractmethod
    def _y(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """

    @abstractmethod
    def _y_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """

    @abstractmethod
    def _y_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """

    @abstractmethod
    def _y_ttt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """


class NoCamber(Camber):
    """Representation of the case where there is no camber."""

    def __init__(self) -> None:
        pass

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Parameter coordinates of any discontinuities.
        """
        return [0.0, 1.0]

    def max_camber_parameter(self) -> float:
        """
        Return parameter where the camber is maximum.

        Returns
        -------
        float
            Parameter where camber is maximum.
        """
        return 0.0

    def _y(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        return np.zeros_like(t)

    def _y_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        return np.zeros_like(t)

    def _y_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """
        return np.zeros_like(t)

    def _y_ttt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """
        return np.zeros_like(t)


class Naca4DigitCamber(Camber):
    """
    Camber for the NACA 4-digit airfoils.

    Attributes
    ----------
    max_camber_index : float
        Maximum amount of camber per chord length times 100.
    loc_max_camber_index : float
        Relative chord location of maximum camber times 10.
    """

    def __init__(self, mci: float, lci: float) -> None:
        # bootstrap values
        self._m = 2
        self._p = 2
        self.max_camber_index = mci
        self.loc_max_camber_index = lci

    @property
    def max_camber_index(self) -> float:
        """Maximum amount of camber."""
        return 100.0*self._m

    @max_camber_index.setter
    def max_camber_index(self, mci: float) -> None:
        if mci == 0:
            self._m = 0.0
            self._p = 0.0
        elif (mci < 0) or (mci > 9):
            raise ValueError("Invalid NACA 4-Digit max. camber: {mci}.")
        else:
            # check to see if currently no camber and force valid value
            if self._p == 0:
                self._p = 0.2

            self._m = mci/100.0

    @property
    def loc_max_camber_index(self) -> float:
        """Location of maximum camber."""
        return 10.0*self._p

    @loc_max_camber_index.setter
    def loc_max_camber_index(self, lci: float) -> None:
        if lci == 0:
            self._m = 0.0
            self._p = 0.0
        elif (lci < 0) or (lci > 9):
            raise ValueError("Invalid NACA 4-Digit max. camber location: "
                             f"{lci}.")
        else:
            # check to see if currently no camber and force valid value
            if self._m == 0:
                self._m = 0.2

            self._p = lci/10.0

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        if (self._m == 0) or (self._p == 0):
            return [0.0, 1.0]

        return [0.0, self._p, 1.0]

    def max_camber_parameter(self) -> float:
        """
        Return parameter where the camber is maximum.

        Returns
        -------
        float
            Parameter where camber is maximum.
        """
        return self._m

    def _y(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        if (self._m == 0) or (self._p == 0):
            return np.zeros_like(t)

        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)

            return (self._m/self._p**2)*(2*self._p*t - t**2)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)

            return (self._m/(1-self._p)**2)*(1 + 2*self._p*(t - 1) - t**2)

        return np.piecewise(t, [t <= self._p, t > self._p], [fore, aft])

    def _y_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        if (self._m == 0) or (self._p == 0):
            return np.zeros_like(t)

        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)

            return 2*(self._m/self._p**2)*(self._p - t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)

            return 2*(self._m/(1-self._p)**2)*(self._p - t)

        return np.piecewise(t, [t <= self._p, t > self._p], [fore, aft])

    def _y_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """
        if (self._m == 0) or (self._p == 0):
            return np.zeros_like(t)

        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)
            return -2*(self._m/self._p**2)*np.ones_like(t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            if self._m == 0:
                return np.zeros_like(t)
            return -2*(self._m/(1-self._p)**2)*np.ones_like(t)

        return np.piecewise(t, [t <= self._p, t > self._p], [fore, aft])

    def _y_ttt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """
        if (self._m == 0) or (self._p == 0):
            return np.zeros_like(t)

        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(t)

        return np.piecewise(t, [t <= self._p, t > self._p], [fore, aft])


class Naca5DigitCamber(Camber):
    """
    Class for the classic NACA 5-digit airfoil camber.

    Attributes
    ----------
    lift_coefficient_index : float
        Ideal lift coefficient times 20/3.
    max_camber_index : float
        Relative chord location of maximum camber times 20.
    m : float
        Camber transition relative chord location.
    k1 : float
        Scale factor for camber.

    Notes
    -----
    The only valid values for the relative chord location of maximum camber
    term are 1, 2, 3, 4, and 5. The only valid value for the ideal lift
    coefficient term is 2.
    """

    def __init__(self, lci: float, mci: float) -> None:
        self._m = 0.2
        self._k1 = 0.0
        self._p_setter(mci)
        self._lci_setter(lci)

    @property
    def lift_coefficient_index(self) -> float:
        """Ideal lift coefficient term."""
        return (20.0/3.0)*self._lci

    @lift_coefficient_index.setter
    def lift_coefficient_index(self, lci: float) -> None:
        self._lci_setter(lci)

    @property
    def max_camber_index(self) -> float:
        """Location of maximum camber."""
        return 20.0*self._p

    @max_camber_index.setter
    def max_camber_index(self, mci: float) -> None:
        self._p_setter(mci)

    @property
    def m(self) -> float:
        """Camber transition relative chord location."""
        return self._m

    @property
    def k1(self) -> float:
        """Scale factor for camber."""
        return self._k1

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, self.m, 1.0]

    def max_camber_parameter(self) -> float:
        """
        Return parameter where the camber is maximum.

        Returns
        -------
        float
            Parameter where camber is maximum.
        """
        return self._p

    def _y(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            m = self.m
            return (self.k1/6)*(t**3 - 3*m*t**2 + m**2*(3-m)*t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return (self.k1*self.m**3/6)*(1 - t)

        return np.piecewise(t, [t <= self.m, t > self.m], [fore, aft])

    def _y_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)
        m = self.m

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return (self.k1/6)*(3*t**2 - 6*m*t + m**2*(3-m))

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return -(self.k1*m**3/6)*np.ones_like(t)

        return np.piecewise(t, [t <= m, t > m], [fore, aft])

    def _y_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return (self.k1)*(t - self.m)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(t)

        return np.piecewise(t, [t <= self.m, t > self.m], [fore, aft])

    def _y_ttt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return self.k1*np.ones_like(t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(t)

        return np.piecewise(t, [t <= self.m, t > self.m], [fore, aft])

    def _p_setter(self, mci: int) -> None:
        if mci == 1:
            self._m = 0.0580
            self._k1 = 361.4
        elif mci == 2:
            self._m = 0.1260
            self._k1 = 51.64
        elif mci == 3:
            self._m = 0.2025
            self._k1 = 15.957
        elif mci == 4:
            self._m = 0.2900
            self._k1 = 6.643
        elif mci == 5:
            self._m = 0.3910
            self._k1 = 3.230
        else:
            raise ValueError("Invalid NACA 5-Digit max. camber location:"
                             f" {mci}.")

        self._p = mci/20.0

    def _lci_setter(self, lci: int) -> None:
        if lci != 2:
            raise ValueError("Invalid NACA 5-Digit ideal lift coefficient "
                             f"parameter: {lci}.")
        self._lci = (3.0/20.0)*lci


class Naca5DigitCamberEnhanced(Naca5DigitCamber):
    """
    Camber for the regular NACA 5-digit airfoils.

    The valid range of the relative chord location of maximum camber term is
    [1, 6). The valid range for the ideal lift coefficient term is [1, 4).
    """

    def __init__(self, lci: float, mci: float) -> None:
        # Need to bootstrap initialization
        self._lci = (3.0/20.0)*lci
        super().__init__(lci=lci, mci=mci)

    def _p_setter(self, mci: float) -> None:
        if mci < 1 or mci >= 6:
            raise ValueError("Invalid NACA 5-Digit max. camber location "
                             f"parameter: {mci}.")

        self._p = mci/20.0

        def camber_slope(m: float) -> float:
            return 3*self._p**2 - 6*m*self._p + m**2*(3-m)

        root = root_scalar(camber_slope, bracket=[self._p, 2*self._p])
        self._m = root.root
        self._determine_k1()

    def _lci_setter(self, lci: float) -> None:
        if lci < 1 or lci >= 4:
            raise ValueError("Invalid NACA 5-Digit ideal lift coefficient "
                             f"paremeter: {lci}.")

        self._lci = (3.0/20.0)*lci
        self._determine_k1()

    def _determine_k1(self) -> None:
        self._k1 = 6*self._lci/(-3/2*(1-2*self._m)*np.arccos(1-2*self._m)
                                + (4*self._m**2-4*self._m
                                   + 3)*np.sqrt(self._m*(1-self._m)))


class Naca5DigitCamberReflexed(Camber):
    """
    Class for the classic NACA 5-digit reflexed camber airfoil.

    Attributes
    ----------
    lift_coefficient_index : float
        Ideal lift coefficient times 20/3.
    max_camber_index : float
        Relative chord location of maximum camber times 20.
    m : float
        Camber transition relative chord location.
    k1 : float
        First scale factor for camber.
    k2 : float
        Second scale factor for camber.

    Notes
    -----
    The only valid values for the relative chord location of maximum camber
    term are 1, 2, 3, 4, and 5. The only valid value for the ideal lift
    coefficient term is 2.
    """

    def __init__(self, lci: float, mci: float,) -> None:
        self._m = 0.2
        self._k1 = 0.0
        self._k2 = 0.0
        self._p_setter(mci)
        self._lci_setter(lci)

    @property
    def lift_coefficient_index(self) -> float:
        """Ideal lift coefficient term."""
        return (20.0/3.0)*self._lci

    @lift_coefficient_index.setter
    def lift_coefficient_index(self, lci: float) -> None:
        self._lci_setter(lci)

    @property
    def max_camber_index(self) -> float:
        """Location of maximum camber."""
        return 20.0*self._p

    @max_camber_index.setter
    def max_camber_index(self, mci: float) -> None:
        self._p_setter(mci)

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

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the camber line.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, self.m, 1.0]

    def max_camber_parameter(self) -> Tuple[float, float]:
        """
        Return parameter where the camber is maximum.

        Returns
        -------
        float
            Parameter where camber is maximum.
        """
        return self._p

    def _y(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the camber location at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)
        m = self.m
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*((t-m)**3 - k2ok1*(1-m)**3*t + m**3*(1-t))

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(k2ok1*(t-m)**3 - k2ok1*(1-m)**3*t + m**3*(1-t))

        return np.piecewise(t, [t <= m, t > m], [fore, aft])

    def _y_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)
        m = self.m
        k1 = self.k1
        k2ok1 = self.k2/k1

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*(t-m)**2 - k2ok1*(1-m)**3 - m**3)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return (k1/6)*(3*k2ok1*(t-m)**2 - k2ok1*(1-m)**3 - m**3)

        return np.piecewise(t, [t <= m, t > m], [fore, aft])

    def _y_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return (self.k1)*(t - self.m)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return (self.k2)*(t - self.m)

        return np.piecewise(t, [t <= self.m, t > self.m], [fore, aft])

    def _y_ttt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return third derivative of camber at specified chord location.

        Parameters
        ----------
        t : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Third derivative of camber at specified point.
        """
        t = np.asarray(t, dtype=np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            return self.k1*np.ones_like(t)

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            return self.k2*np.ones_like(t)

        return np.piecewise(t, [t <= self.m, t > self.m], [fore, aft])

    def _p_setter(self, mci: int) -> None:
        if mci == 2:
            self._m = 0.1300
            self._k1 = 51.99
            self._k2 = 0.03972036
        elif mci == 3:
            self._m = 0.2170
            self._k1 = 15.793
            self._k2 = 0.10691861
        elif mci == 4:
            self._m = 0.3180
            self._k1 = 6.520
            self._k2 = 0.197556
        elif mci == 5:
            self._m = 0.4410
            self._k1 = 3.191
            self._k2 = 0.4323805
        else:
            raise ValueError("Invalid NACA 5-Digit reflexed max. camber "
                             f"location: {mci}.")

        self._p = mci/20.0

    def _lci_setter(self, lci: int) -> None:
        if lci != 2:
            raise ValueError("Invalid NACA 5-Digit reflexed ideal lift "
                             f"coefficient parameter: {lci}.")
        self._lci = (3.0/20.0)*lci


class Naca5DigitCamberReflexedEnhanced(Naca5DigitCamberReflexed):
    """
    Camber for the regular NACA 5-digit reflexed airfoils.

    The valid range of the relative chord location of maximum camber term is
    [1, 6). The valid range for the ideal lift coefficient term is [1, 4).
    """

    def __init__(self, lci: float, mci: float) -> None:
        # Need to bootstrap initialization
        self._lci = (3.0/20.0)*lci
        super().__init__(lci=lci, mci=mci)

    def _p_setter(self, mci) -> float:
        if mci < 1 or mci >= 6:
            raise ValueError("Invalid NACA 5-Digit reflexed max. camber "
                             f"location: {mci}")

        self._p = mci/20.0

        def m_fun(m: float) -> float:
            k1 = 1.0
            k2ok1 = self._k2ok1(m, self._p)
            Cl_id = self._Cl_id(m, k1, k2ok1)
            return -0.25*Cl_id + (k1/192)*(3*k2ok1*np.pi
                                           + 3*(1-k2ok1)*np.arccos(1-2*m)
                                           + 2*(1-k2ok1)*(1-2*m)*(8*m**2-8*m-3)
                                           * np.sqrt(m*(1-m)))

        root = root_scalar(m_fun, bracket=[self._p, 3*self._p])
        self._m = root.root
        self._determine_k1k2()

    def _lci_setter(self, lci: float) -> float:
        if lci < 1 or lci >= 4:
            raise ValueError("Invalid NACA 5-Digit reflexed ideal lift "
                             f"coefficient parameter: {lci}.")

        self._lci = (3.0/20.0)*lci
        self._determine_k1k2()

    def _determine_k1k2(self) -> None:
        k2ok1 = self._k2ok1(self.m, self._p)
        self._k1 = self._lci/self._Cl_id(self.m, 1, k2ok1)
        self._k2 = k2ok1*self._k1

    @staticmethod
    def _Cl_id(m: float, k1: float, k2ok1: float) -> float:
        return (k1/12)*(3*k2ok1*(2*m-1)*np.pi
                        + 3*(1-k2ok1)*(2*m-1)*np.arccos(1-2*m)
                        + 2*(1-k2ok1)*(4*m**2-4*m+3)*np.sqrt(m*(1-m)))

    @staticmethod
    def _k2ok1(m: float, p: float) -> float:
        return (3*(p-m)**2 - m**3)/(1 - m)**3
