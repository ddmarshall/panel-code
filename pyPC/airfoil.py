#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoils modeled in panel codes."""

from typing import Tuple

import numpy as np
import numpy.typing as np_type
from scipy.optimize import root_scalar

from pyPC.geometry import Geometry


class Naca4DigitCamber:
    """
    Camber for the NACA 4-digit airfoils.

    Attributes
    ----------
    m : float
        Maximum amount of camber per chord length.
    p : float
        Relative chord location of maximum camber.
    """

    def __init__(self, m: float, p: float) -> None:
        self._m = m
        self._p = p

    @property
    def m(self) -> float:
        """Maximum amount of camber."""
        return self._m

    @property
    def p(self) -> float:
        """Location of maximum camber."""
        return self._p

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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/self.p**2)*(2*self.p*xi - xi**2)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/(1-self.p)**2)*(1 + 2*self.p*(xi - 1) - xi**2)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/self.p**2)*(self.p - xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/(1-self.p)**2)*(self.p - xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/self.p**2)*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/(1-self.p)**2)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])


class Naca4DigitThicknessBase:
    """
    Base class for the NACA 4-digit airfoil thickness.

    Attributes
    ----------
    thickness : float
        Maximum thickness per chord length.
    """

    def __init__(self, thickness: float, a: np_type.NDArray) -> None:
        self._t = thickness
        self._a = a

    @property
    def thickness(self) -> float:
        """Maximum thickness."""
        return self._t

    @thickness.setter
    def thickness(self, thickness: float) -> None:
        self._t = thickness

    def y(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified point.
        """
        return (self.thickness/0.20)*(self._a[0]*np.sqrt(xi)
                                      + xi*(self._a[1]
                                            + xi*(self._a[2]
                                                  + xi*(self._a[3]
                                                        + xi*self._a[4]))))

    def y_p(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified point.
        """
        return (self.thickness/0.20)*(0.5*self._a[0]/np.sqrt(xi)
                                      + (self._a[1]
                                         + xi*(2*self._a[2]
                                               + xi*(3*self._a[3]
                                                     + 4*xi*self._a[4]))))

    def y_pp(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified chord location.

        Parameters
        ----------
        xi : numpy.ndarray
            Chord location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified point.
        """
        return (self.thickness/0.20)*(-0.25*self._a[0]/(xi*np.sqrt(xi))
                                      + 2*(self._a[2]
                                           + 3*xi*(self._a[3]
                                                   + 2*xi*self._a[4])))


class Naca4DigitThicknessClassic(Naca4DigitThicknessBase):
    """Classic NACA 4-digit airfoil thickness."""

    def __init__(self, thickness: float) -> None:
        super().__init__(thickness=thickness, a=np.array([0.29690, -0.12600,
                                                          -0.35160, 0.28430,
                                                          -0.10150]))


class Naca4DigitThicknessEnhanced(Naca4DigitThicknessBase):
    """
    Enhanced NACA 4-digit airfoil thickness relation.

    This class extends the standard thickness distribution relations by
    - Solving for the coefficients based on the original constraints used to
      describe the thickness
    - Allowing the trailing edge to be closed instead of the default thickness
    - Allowing the setting of the leading edge radius value instead of the
      approximate way it is set in the original formulations.

    Attributes
    ----------
    closed_te : bool
        True if the thickness should be zero at the trailing edge
    le_radius : bool
        True if the leading edge radius should be same as classic airfoil or
        False if original method of setting the thickness near the leading
        edge should be used.
    """

    def __init__(self, thickness: float, closed_te: bool,
                 le_radius: bool) -> None:
        self.reset(closed_te, le_radius)
        super().__init__(thickness=thickness, a=self._a)

    def reset(self, closed_te: bool, le_radius: bool) -> None:
        """
        Reset the flags for the leading edge and trailing edge shape.

        Parameters
        ----------
        closed_te : bool
            True if the thickness should be zero at the trailing edge
        le_radius : bool
            True if the leading edge radius should be same as classic airfoil
            or False if original method of setting the thickness near the
            leading edge should be used.
        """
        self._closed_te = closed_te
        self._le_radius = le_radius

        # solve for new values of the coefficients
        B = np.zeros([5,5])
        r = np.zeros([5,1])

        # first row is leading edge condition
        i = 0
        if le_radius:
            B[i, :] = [1, 0, 0, 0, 0]
            r[i] = 0.29690
        else:
            xi_1c = 0.1
            t_1c = 0.078
            B[i, :] = [np.sqrt(xi_1c), xi_1c, xi_1c**2, xi_1c**3, xi_1c**4]
            r[i] = t_1c

        # second row is the maximum thickness at 0.3c
        i = 1
        xi_max = 0.3
        B[i, :] = [0.5/np.sqrt(xi_max), 1, 2*xi_max, 3*xi_max**2, 4*xi_max**3]
        r[i] = 0.0

        # third row is the maximum thickness of 0.2c
        i = 2
        t_max = 0.1
        B[i, :] = [np.sqrt(xi_max), xi_max, xi_max**2, xi_max**3, xi_max**4]
        r[i] = t_max

        # fourth row is trailing edge slope
        i = 3
        te_slope = -0.234
        B[i, :] = [0.5, 1, 2, 3, 4]
        r[i] = te_slope

        # fith row is trailing edge thickness
        i = 4
        if closed_te:
            t_te = 0
        else:
            t_te = 0.002
        B[i, :] = [1, 1, 1, 1, 1]
        r[i] = t_te

        self._a = np.linalg.solve(B, r).transpose()[0]


class Naca5DigitCamberRegular:
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
        self.Cl_ideal = self.Cl_ideal

    @property
    def Cl_ideal(self) -> float:
        """Ideal lift coefficient."""
        return self._Cl_ideal

    @Cl_ideal.setter
    def Cl_ideal(self, Cl_ideal) -> float:
        self._Cl_ideal = Cl_ideal
        self._k1 = self._determine_k1(Cl_ideal, self._m)

    @staticmethod
    def _camber_slope(xi: float, k1: float, m: float) -> float:
        return k1/6*(3*xi**2 - 6*m*xi + m**2*(3-m))

    @staticmethod
    def _determine_k1(Cl_ideal: float, m: float) -> float:
        return 6*Cl_ideal/(-3/2*(1-2*m)*np.arccos(1-2*m)
                           + (4*m**2-4*m+3)*np.sqrt(m*(1-m)))

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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/self.p**2)*(2*self.p*xi - xi**2)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return (self.m/(1-self.p)**2)*(1 + 2*self.p*(xi - 1) - xi**2)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/self.p**2)*(self.p - xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*(self.m/(1-self.p)**2)*(self.p - xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/self.p**2)*np.ones_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -2*(self.m/(1-self.p)**2)*np.ones_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
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

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return np.zeros_like(xi)

        return np.piecewise(xi, [xi <= self.p, xi > self.p],
                            [lambda xi: fore(xi), lambda xi: aft(xi)])


class Airfoil(Geometry):
    """
    Cylinderical shaped 2D airfoil.

    Attributes
    ----------
    radius : float
        Radius of the cylinder.
    """

    def __init__(self, radius) -> None:
        self._r = radius

    @property
    def radius(self) -> float:
        """Radius of the cylinder."""
        return self._r

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
        return np.zeros_like(xi)

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
        _, th = self.xy_from_xi(xi)
        return th

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
        theta = np.pi*(1-xi)
        x = self.radius*(1+np.cos(theta))
        y = self.radius*np.sin(theta)
        return x, y

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
        theta = np.pi*(1-xi)
        x_p = np.pi*self.radius*np.sin(theta)
        y_p = -np.pi*self.radius*np.cos(theta)
        return x_p, y_p

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
        theta = np.pi*(1-xi)
        x_pp = -np.pi**2*self.radius*np.cos(theta)
        y_pp = -np.pi**2*self.radius*np.sin(theta)
        return x_pp, y_pp
