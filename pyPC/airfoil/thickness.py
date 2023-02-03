#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoil thickness distributions."""

from abc import abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type

from pyPC.airfoil.curve import Curve


class Thickness(Curve):
    """
    Base class for thickness distribution.

    The thickness will have a parameterization from 0 to 1 and is not
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

    def k(self, xi: np_type.NDArray) -> np_type.NDArray:
        """
        Calculate the curvature at parameter location.

        Special treatment is needed for many formulations of the typical
        rounded leading edge.

        Parameters
        ----------
        xi : numpy.ndarray
            Parameter for desired locations.

        Returns
        -------
        numpy.ndarray
            Curvature of surface at point.

        Raises
        ------
        ValueError
            If there is no surface point at the given x-location.
        """
        eps = 1e-6
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        return np.piecewise(xi, [np.abs(xi) < eps, np.abs(xi) >= eps],
                            [lambda xi: self.le_k(),
                             lambda xi: super(Thickness, self).k(xi)])

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
    def max_thickness(self) -> Tuple[float, float]:
        """
        Return chord location of maximum thickness and the maximum thickness.

        Returns
        -------
        float
            Chord location of maximum thickness.
        float
            Maximum thickness.
        """

    @abstractmethod
    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """


class NoThickness(Thickness):
    """Reprentation of the case where there is no thickness."""

    def __init__(self) -> None:
        pass

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
        return np.zeros_like(xi)

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
        return np.zeros_like(xi)

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
        return np.zeros_like(xi)

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the thickness.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, 1.0]

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return chord location of maximum thickness and the maximum thickness.

        Returns
        -------
        float
            Chord location of maximum thickness.
        float
            Maximum thickness.
        """
        return 0.0, 0.0

    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """
        return 0.0


class Naca45DigitThickness(Thickness):
    """
    Class for the classic NACA 4-digit and 5-digit airfoil thickness.

    Attributes
    ----------
    tmax : float
        Maximum thickness per chord length times 100.
    a : numpy.ndarray
        Coefficients for equation.

    Notes
    -----
    To obtain a classic thickness profile the maximum thickness should be set
    to an integer value, i.e. 12. However, any floating point value can be
    passed in, i.e. 12.3, if a more accurate maximum thickness is needed to
    be specified.
    """

    def __init__(self, tmax: float) -> None:
        self._a = np.array([0.29690, -0.12600, -0.35160, 0.28430, -0.10150])
        self.tmax = tmax

    @property
    def tmax(self) -> float:
        """Maximum thickness."""
        return 100*self._tmax

    @tmax.setter
    def tmax(self, tmax: float) -> None:
        if tmax < 0 or tmax >= 100:
            raise ValueError(f"Invalid NACA 4/5-digit max. thicknes: {tmax}")

        self._tmax = tmax/100.0

    @property
    def a(self) -> float:
        """Equation coefficients."""
        return self._a

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
        return (self._tmax/0.20)*(self.a[0]*np.sqrt(xi)
                                  + xi*(self.a[1]
                                        + xi*(self.a[2]
                                              + xi*(self.a[3]
                                                    + xi*self.a[4]))))

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
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fun(xi: np_type.NDArray) -> np_type.NDArray:
            return (self._tmax/0.20)*(0.5*self.a[0]/np.sqrt(xi)
                                      + (self.a[1]
                                         + xi*(2*self.a[2]
                                               + xi*(3*self.a[3]
                                                     + 4*xi*self.a[4]))))

        return np.piecewise(xi, [xi == 0, xi != 0],
                            [lambda xi: np.inf, lambda xi: fun(xi)])

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
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fun(xi: np_type.NDArray) -> np_type.NDArray:
            return (self._tmax/0.20)*(-0.25*self.a[0]/(xi*np.sqrt(xi))
                                      + 2*(self.a[2]
                                           + 3*xi*(self.a[3]
                                                   + 2*xi*self.a[4])))

        return np.piecewise(xi, [xi == 0, xi != 0],
                            [lambda xi: -np.inf, lambda xi: fun(xi)])

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the thickness.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, 1.0]

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return chord location of maximum thickness and the maximum thickness.

        Returns
        -------
        float
            Chord location of maximum thickness.
        float
            Maximum thickness.
        """
        return 0.3, self.y(0.3)

    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """
        return -2/((self._tmax/0.20)*self.a[0])**2


class Naca45DigitThicknessEnhanced(Naca45DigitThickness):
    """
    Enhanced NACA 4-digit and 5-digit airfoil thickness.

    This class extends the standard thickness distribution relations by
    - Solving for the coefficients based on the original constraints used to
      describe the thickness even with non-integer parameters,
    - Allowing the trailing edge to be closed instead of the default
      thickness, and
    - Allowing the setting of the leading edge radius value instead of the
      approximate way it is set in the original formulations.

    Attributes
    ----------
    closed_trailing_edge : bool
        True if the thickness should be zero at the trailing edge
    use_leading_edge_radius : bool
        True if the leading edge radius should be same as classic airfoil or
        False if original method of setting the thickness near the leading
        edge should be used.

    Notes
    -----
    Specifying the same parameters as the classic thickness profile will not
    result in an identical thickness profile because the cannonical
    coefficients do not match the stated constraints in the original source
    from Jacobs, Ward, and Pinkerton (1933).
    """

    def __init__(self, tmax: float, closed_te: bool, use_radius: bool) -> None:
        super().__init__(tmax=tmax)
        self._closed_te = closed_te
        self._use_radius = use_radius
        self._calculate_a()

    @property
    def closed_trailing_edge(self) -> bool:
        """Return state of trailing edge condition."""
        return self._closed_te

    @closed_trailing_edge.setter
    def closed_trailing_edge(self, closed_te: bool) -> None:
        self._closed_te = closed_te
        self._calculate_a()

    @property
    def use_leading_edge_radius(self) -> bool:
        """Return state of leading edge treatment."""
        return self._use_radius

    @use_leading_edge_radius.setter
    def use_leading_edge_radius(self, use_radius: bool) -> None:
        self._use_radius = use_radius
        self._calculate_a()

    def _calculate_a(self) -> None:
        """Reset the flags for the leading edge and trailing edge shape."""
        # solve for new values of the coefficients
        B = np.zeros([5,5])
        r = np.zeros([5,1])

        # first row is leading edge condition
        i = 0
        if self._use_radius:
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
        if self._closed_te:
            t_te = 0
        else:
            t_te = 0.002
        B[i, :] = [1, 1, 1, 1, 1]
        r[i] = t_te

        self._a = np.linalg.solve(B, r).transpose()[0]


class Naca45DigitModifiedThickness(Thickness):
    """
    Base class for the NACA modified 4-digit and 5-digit airfoil thickness.

    Attributes
    ----------
    tmax : float
        Maximum thickness per chord length times 100.
    leading_edge_index: float
        Parameter to specify the radius of the leading edge.
    max_thickness_index : float
        Location of end of fore section and start of aft section times 10.
    a : numpy.ndarray
        Coefficients for fore equation.
    d : numpy.ndarray
        Coefficients for aft equation.
    """

    def __init__(self, tmax: float, lei: float, xi_m: float) -> None:
        # start with valid defaults for setters to work
        self._closed_te = False
        self._lei = 4
        self._xi_m = 4
        self._a = np.zeros(4)
        self._d = np.zeros(4)

        # use settters to ensure valid data
        self.tmax = tmax
        self.max_thickness_index = xi_m
        self.leading_edge_index = lei

    @property
    def tmax(self) -> float:
        """Maximum thickness."""
        return 100*self._tmax

    @tmax.setter
    def tmax(self, tmax: float) -> None:
        if tmax < 0 or tmax >= 100:
            raise ValueError("Invalid NACA modified 4/5-digit max. thickness: "
                             f"{tmax}")
        self._tmax = tmax/100.0

    @property
    def leading_edge_index(self) -> float:
        """Leading edge index parameter."""
        return self._lei

    @leading_edge_index.setter
    def leading_edge_index(self, lei: float) -> None:
        if lei < 1 or lei >= 10:
            raise ValueError("Invalid NACA modified 4/5-digit leading edge "
                             f"parameter: {lei}")
        self._lei = lei
        self._calculate_coefficients()

    @property
    def max_thickness_index(self) -> float:
        """Location where fore and aft equations meet."""
        return 10*self._xi_m

    @max_thickness_index.setter
    def max_thickness_index(self, xi_m: float) -> None:
        if xi_m < 1 or xi_m >= 10:
            raise ValueError("Invalid NACA modified 4/5-digit max. thickness "
                             f"location parameter: {xi_m}")
        self._xi_m = xi_m/10.0
        self._calculate_coefficients()

    @property
    def a(self) -> float:
        """Fore equation coefficients."""
        return self._a

    @property
    def d(self) -> float:
        """Aft equation coefficients."""
        return self._d

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
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return self.a[0]*np.sqrt(xi) + xi*(self.a[1]
                                               + xi*(self.a[2]
                                                     + (xi*self.a[3])))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return self.d[0] + (1-xi)*(self.d[1]
                                       + (1-xi)*(self.d[2]
                                                 + ((1-xi)*self.d[3])))

        return self._tmax*np.piecewise(xi, [xi <= self._xi_m, xi > self._xi_m],
                                       [lambda xi: fore(xi),
                                        lambda xi: aft(xi)])

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
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return 0.5*self.a[0]/np.sqrt(xi) + (self.a[1]
                                                + xi*(2*self.a[2]
                                                      + (3*xi*self.a[3])))

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return -self.d[1] + (1-xi)*(-2*self.d[2] + (-3*(1-xi)*self.d[3]))

        return self._tmax*np.piecewise(xi,
                                       [xi == 0, (xi > 0) & (xi <= self._xi_m),
                                        xi > self._xi_m],
                                       [lambda xi: np.inf,
                                        lambda xi: fore(xi),
                                        lambda xi: aft(xi)])

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
        xi = np.asarray(xi)
        if issubclass(xi.dtype.type, np.integer):
            xi = xi.astype(np.float64)

        def fore(xi: np_type.NDArray) -> np_type.NDArray:
            return (-0.25*self.a[0]/(xi*np.sqrt(xi)) + 2*self.a[2]
                    + 6*xi*self.a[3])

        def aft(xi: np_type.NDArray) -> np_type.NDArray:
            return 2*self.d[2] + 6*(1-xi)*self.d[3]

        return self._tmax*np.piecewise(xi,
                                       [xi == 0, (0 < xi) & (xi <= self._xi_m),
                                        xi > self._xi_m],
                                       [lambda xi: -np.inf,
                                        lambda xi: fore(xi),
                                        lambda xi: aft(xi)])

    def joints(self) -> List[float]:
        """
        Return the locations of any joints/discontinuities in the thickness.

        Returns
        -------
        List[float]
            Xi-coordinates of any discontinuities.
        """
        return [0.0, self._xi_m, 1.0]

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return chord location of maximum thickness and the maximum thickness.

        Returns
        -------
        float
            Chord location of maximum thickness.
        float
            Maximum thickness.
        """
        return self._xi_m, self.y(self._xi_m)

    def le_k(self) -> float:
        """
        Return the curvature of the leading edge.

        Returns
        -------
        float
            Leading edge curvature.
        """
        return -2/(self._tmax*self.a[0])**2

    def _calculate_coefficients(self):
        # Pade approximation that goes through all Stack and von Doenhoff
        # (1935) values. Improves upon Riegels (1961) fit.
        p = [1.0310900853, -2.7171508529, 4.8594083156]
        q = [1.0, -1.8252487562, 1.1771499645]
        tau = ((p[0] + p[1]*self._xi_m + p[2]*self._xi_m**2)
               / (q[0] + q[1]*self._xi_m + q[2]*self._xi_m**2))

        # calculate the d coefficients
        if self._closed_te:
            eta = 0.0
        else:
            eta = 0.02

        self._d[0] = 0.5*eta
        self._d[1] = tau
        self._d[2] = (2*tau*(self._xi_m-1)-1.5*(eta-1))/(self._xi_m-1)**2
        self._d[3] = (tau*(self._xi_m-1)-(eta-1))/(self._xi_m-1)**3

        # calculate the a coefficients
        Q = 25*0.08814961
        if self._lei < 9:
            Q /= 72
        else:
            Q /= 54

        k_m = (3*(eta-1)-2*tau*(self._xi_m-1))/(self._xi_m-1)**2
        sqrt_term = np.sqrt(2*Q)
        sqrt_term2 = np.sqrt(2*Q*self._xi_m)
        self._a[0] = self._lei*sqrt_term
        self._a[1] = (0.5*self._xi_m
                      * (k_m + (3-3.75*self._lei*sqrt_term2)/self._xi_m**2))
        self._a[2] = -(k_m + (1.5-1.25*self._lei*sqrt_term2)/self._xi_m**2)
        self._a[3] = (0.5/self._xi_m
                      * (k_m + (1-0.75*self._lei*sqrt_term2)/self._xi_m**2))


class Naca45DigitModifiedThicknessEnhanced(Naca45DigitModifiedThickness):
    """
    Enhanced NACA modified 4-digit and 5-digit airfoil thickness relation.

    This class extends the standard modified thickness distribution relations
    by
    - Solving for the coefficients based on the original constraints used to
      describe the thickness for non-integer values of parameters
    - Allowing the trailing edge to be closed instead of the default thickness

    Attributes
    ----------
    closed_te : bool
        True if the thickness should be zero at the trailing edge
    """

    def __init__(self, tmax: float,  lei: float, xi_m: float,
                 closed_te: bool) -> None:
        super().__init__(tmax=tmax, xi_m=xi_m, lei=lei)
        self.closed_te = closed_te

    @property
    def closed_te(self) -> bool:
        """Flag whether trailing edge should be closed."""
        return self._closed_te

    @closed_te.setter
    def closed_te(self, closed_te: bool) -> None:
        if self._closed_te != closed_te:
            self._closed_te = closed_te
            self._calculate_coefficients()
