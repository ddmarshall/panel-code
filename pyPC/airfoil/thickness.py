#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Classes associated with airfoil thickness distributions."""

from abc import ABC, abstractmethod
from typing import Tuple, List

import numpy as np
import numpy.typing as np_type


class Thickness(ABC):
    """
    Base class for thickness distribution.

    The thickness will have a parameterization from 0 to 1 and is not
    defined outside of that range. This parameterization should be chosen so
    that all derivatives are finite over the entire parameterization range.
    """

    @abstractmethod
    def discontinuities(self) -> List[float]:
        """
        Return the locations of any discontinuities in the thickness.

        Returns
        -------
        List[float]
            Parametrics coordinates of any discontinuities.
        """

    @abstractmethod
    def max_thickness(self) -> Tuple[float, float]:
        """
        Return parameter location and value of maximum thickness.

        Returns
        -------
        float
            Parameter location of maximum thickness.
        float
            Maximum thickness.
        """

    @abstractmethod
    def delta(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

        Returns
        -------
        numpy.ndarray
            Thickness at specified parameter.
        """

    @abstractmethod
    def delta_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified parameter.
        """

    @abstractmethod
    def delta_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified parameter.
        """


class NoThickness(Thickness):
    """Reprentation of the case where there is no thickness."""

    def __init__(self) -> None:
        pass

    def discontinuities(self) -> List[float]:
        """
        Return the locations of any discontinuities in the thickness.

        Returns
        -------
        List[float]
            Parametric coordinates of any discontinuities.
        """
        return []

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return parameter location and value of maximum thickness.

        Returns
        -------
        float
            Parameter location of maximum thickness.
        float
            Maximum thickness.
        """
        return 0.0, 0.0

    def delta(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Thickness at specified parameter.
        """
        return np.zeros_like(t)

    def delta_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified parameter.
        """
        return np.zeros_like(t)

    def delta_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified parameter.
        """
        return np.zeros_like(t)


class Naca45DigitThickness(Thickness):
    """
    Class for the classic NACA 4-digit and 5-digit airfoil thickness.

    The thickness is parameterized on the square root of the chord location
    where the thickness is desired. This is to remove singularities that
    occur at the leading edge for the typical chord length parameterization.

    Attributes
    ----------
    max_thickness_index : float
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

    def __init__(self, mti: float) -> None:
        self._a = np.array([0.29690, -0.12600, -0.35160, 0.28430,
                            -0.10150])/0.20
        self.max_thickness_index = mti

    @property
    def max_thickness_index(self) -> float:
        """Maximum thickness."""
        return 100*self._tmax

    @max_thickness_index.setter
    def max_thickness_index(self, mti: float) -> None:
        if mti < 0 or mti >= 100:
            raise ValueError(f"Invalid NACA 4/5-digit max. thickness: {mti}")

        self._tmax = mti/100.0

    @property
    def a(self) -> float:
        """Equation coefficients."""
        return self._a

    def discontinuities(self) -> List[float]:
        """
        Return the locations of any discontinuities in the thickness.

        Returns
        -------
        List[float]
            Parametric coordinates of any discontinuities.
        """
        return []

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return parameter location and value of maximum thickness.

        Returns
        -------
        float
            Parameter location of maximum thickness.
        float
            Maximum thickness.
        """
        return 0.3, self.delta(np.sqrt(0.3))

    def delta(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Thickness at specified parameter.
        """
        t2 = t**2
        return self._tmax*t*(self.a[0] + t*(self.a[1]
                                            + t2*(self.a[2]
                                                  + t2*(self.a[3]
                                                        + t2*self.a[4]))))

    def delta_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified parameter.
        """
        t2 = t**2
        return self._tmax*(self.a[0]
                           + 2*t*(self.a[1]
                                  + t2*(2*self.a[2]
                                        + t2*(3*self.a[3]
                                              + 4*t2*self.a[4]))))

    def delta_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified parameter.
        """
        t2 = t**2
        return 2*self._tmax*(self.a[1]
                             + t2*(6*self.a[2]
                                   + t2*(15*self.a[3]
                                         + 28*self.a[4]*t2)))


class Naca45DigitThicknessEnhanced(Naca45DigitThickness):
    """
    Enhanced NACA 4-digit and 5-digit airfoil thickness.

    The thickness is parameterized on the square root of the chord location
    where the thickness is desired. This is to remove singularities that
    occur at the leading edge for the typical chord length parameterization.

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

    def __init__(self, mti: float, closed_te: bool, use_radius: bool) -> None:
        super().__init__(mti=mti)
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
        # solve for new values of the coefficients using chord based
        # parameterization
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

        self._a = (np.linalg.solve(B, r).transpose()[0])/0.20


class Naca45DigitModifiedThickness(Thickness):
    """
    Base class for the NACA modified 4-digit and 5-digit airfoil thickness.

    The thickness is parameterized on the square root of the chord location
    where the thickness is desired. This is to remove singularities that
    occur at the leading edge for the typical chord length parameterization.

    Attributes
    ----------
    max_thickness_index : float
        Maximum thickness per chord length times 100.
    leading_edge_index: float
        Parameter to specify the radius of the leading edge.
    loc_max_thickness_index : float
        Location of end of fore section and start of aft section times 10.
    a : numpy.ndarray
        Coefficients for fore equation.
    d : numpy.ndarray
        Coefficients for aft equation.
    """

    def __init__(self, mti: float, lei: float, lmti: float) -> None:
        # start with valid defaults for setters to work
        self._closed_te = False
        self._lei = 4
        self._t_m = 4
        self._a = np.zeros(4)
        self._d = np.zeros(4)

        # use settters to ensure valid data
        self.max_thickness_index = mti
        self.loc_max_thickness_index = lmti
        self.leading_edge_index = lei

    @property
    def max_thickness_index(self) -> float:
        """Maximum thickness."""
        return 100*self._tmax

    @max_thickness_index.setter
    def max_thickness_index(self, mti: float) -> None:
        if mti < 0 or mti >= 100:
            raise ValueError("Invalid NACA modified 4/5-digit max. thickness: "
                             f"{mti}")
        self._tmax = mti/100.0

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
    def loc_max_thickness_index(self) -> float:
        """Location where fore and aft equations meet."""
        return 10*self._t_m**2

    @loc_max_thickness_index.setter
    def loc_max_thickness_index(self, lmti: float) -> None:
        if lmti < 1 or lmti >= 10:
            raise ValueError("Invalid NACA modified 4/5-digit max. thickness "
                             f"location parameter: {lmti}")
        self._t_m = np.sqrt(lmti/10.0)
        self._calculate_coefficients()

    @property
    def a(self) -> float:
        """Fore equation coefficients."""
        return self._a

    @property
    def d(self) -> float:
        """Aft equation coefficients."""
        return self._d

    def discontinuities(self) -> List[float]:
        """
        Return the locations of any discontinuities in the thickness.

        Returns
        -------
        List[float]
            Parametric coordinates of any discontinuities.
        """
        return [self._t_m]

    def max_thickness(self) -> Tuple[float, float]:
        """
        Return parameter location and value of maximum thickness.

        Returns
        -------
        float
            Parameter location of maximum thickness.
        float
            Maximum thickness.
        """
        return self._t_m, self.delta(self._t_m)

    def delta(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return the thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Thickness at specified parameter.
        """
        t = np.asarray(t, np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            return t*(self.a[0]
                      + t*(self.a[1] + t2*(self.a[2] + t2*self.a[3])))

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            term = 1-t2
            return self.d[0] + term*(self.d[1]
                                     + term*(self.d[2] + term*self.d[3]))

        return self._tmax*np.piecewise(t, [t <= self._t_m, t > self._t_m],
                                       [lambda t: fore(t), lambda t: aft(t)])

    def delta_t(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return first derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            First derivative of thickness at specified parameter.
        """
        t = np.asarray(t, np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            return self.a[0] + 2*t*(self.a[1] + t2*(2*self.a[2]
                                                    + 3*self.a[3]*t2))

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            return -2*t*(self.d[1] + (1-t2)*(2*self.d[2] + 3*(1-t2)*self.d[3]))

        return self._tmax*np.piecewise(t, [t <= self._t_m, t > self._t_m],
                                       [lambda t: fore(t), lambda t: aft(t)])

    def delta_tt(self, t: np_type.NDArray) -> np_type.NDArray:
        """
        Return second derivative of thickness at specified parameter location.

        Parameters
        ----------
        t : numpy.ndarray
            Parameter location of interest. Equal to the square root of the
            desired chord location.

        Returns
        -------
        numpy.ndarray
            Second derivative of thickness at specified parameter.
        """
        t = np.asarray(t, np.float64)

        def fore(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            return 2*(self.a[1] + 3*t2*(2*self.a[2] + 5*t2*self.a[3]))

        def aft(t: np_type.NDArray) -> np_type.NDArray:
            t2 = t**2
            return -2*(self.d[1] + 2*(1-3*t2)*self.d[2]
                       + 3*(1-t2)*(1-5*t2)*self.d[3])

        return self._tmax*np.piecewise(t, [t <= self._t_m, t > self._t_m],
                                       [lambda t: fore(t), lambda t: aft(t)])

    def _calculate_coefficients(self):
        # Pade approximation that goes through all Stack and von Doenhoff
        # (1935) values. Improves upon Riegels (1961) fit.
        p = [1.0310900853, -2.7171508529, 4.8594083156]
        q = [1.0, -1.8252487562, 1.1771499645]
        xi_m = self._t_m**2
        tau = ((p[0] + p[1]*xi_m + p[2]*xi_m**2)
               / (q[0] + q[1]*xi_m + q[2]*xi_m**2))

        # calculate the d coefficients
        if self._closed_te:
            eta = 0.0
        else:
            eta = 0.02

        self._d[0] = 0.5*eta
        self._d[1] = tau
        self._d[2] = (2*tau*(xi_m-1)-1.5*(eta-1))/(xi_m-1)**2
        self._d[3] = (tau*(xi_m-1)-(eta-1))/(xi_m-1)**3

        # calculate the a coefficients
        Q = 25*0.08814961
        if self._lei < 9:
            Q /= 72
        else:
            Q /= 54

        k_m = (3*(eta-1)-2*tau*(xi_m-1))/(xi_m-1)**2
        sqrt_term = np.sqrt(2*Q)
        sqrt_term2 = np.sqrt(2*Q*xi_m)
        self._a[0] = self._lei*sqrt_term
        self._a[1] = (0.5*xi_m*(k_m + (3-3.75*self._lei*sqrt_term2)/xi_m**2))
        self._a[2] = -(k_m + (1.5-1.25*self._lei*sqrt_term2)/xi_m**2)
        self._a[3] = (0.5/xi_m*(k_m + (1-0.75*self._lei*sqrt_term2)/xi_m**2))


class Naca45DigitModifiedThicknessEnhanced(Naca45DigitModifiedThickness):
    """
    Enhanced NACA modified 4-digit and 5-digit airfoil thickness relation.

    The thickness is parameterized on the square root of the chord location
    where the thickness is desired. This is to remove singularities that
    occur at the leading edge for the typical chord length parameterization.

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

    def __init__(self, mti: float,  lei: float, lmti: float,
                 closed_te: bool) -> None:
        super().__init__(mti=mti, lei=lei, lmti=lmti)
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
