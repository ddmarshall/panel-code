#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:37:52 2022

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil import Naca4DigitCamber, Naca4DigitThicknessBase
from pyPC.airfoil import Naca4DigitThicknessClassic
from pyPC.airfoil import Naca4DigitThicknessEnhanced


class TestNaca4Digit(unittest.TestCase):
    """Class to test the NACA 4-digit airfoil geometry."""

    def testCamber(self) -> None:
        """Test the camber relations."""
        af = Naca4DigitCamber(m=0.03, p=0.4)

        def compare_values(xi: np_type.NDArray, af: Naca4DigitCamber) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    if xir <= af.p:
                        yr[...] = (af.m/af.p**2)*(2*af.p*xir - xir**2)
                    else:
                        yr[...] = (af.m/(1-af.p)**2)*(1-2*af.p + 2*af.p*xir
                                                      - xir**2)

            # compare point values
            y = af.y(xi)
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            ypl = af.y(xi+eps)
            ymi = af.y(xi-eps)
            yp_ref = 0.5*(ypl-ymi)/eps
            yp = af.y_p(xi)
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            ypl = af.y_p(xi+eps)
            ymi = af.y_p(xi-eps)
            ypp_ref = 0.5*(ypl-ymi)/eps
            ypp = af.y_pp(xi)
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

            # compare third derivatives
            ypl = af.y_pp(xi+eps)
            ymi = af.y_pp(xi-eps)
            yppp_ref = 0.5*(ypl-ymi)/eps
            yppp = af.y_ppp(xi)
            self.assertIsNone(npt.assert_allclose(yppp, yppp_ref, atol=1e-7))

        # test point on front
        xi = 0.25
        compare_values(xi, af)

        # test point on back
        xi = 0.6
        compare_values(xi, af)

        # test points on lower and upper surface
        xi = np.linspace(0, 1, 12)
        compare_values(xi, af)

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca4DigitThicknessEnhanced(thickness=0.20, closed_te=True,
                                         le_radius=False)
        xi_max = 0.3
        xi_1c = 0.1
        xi_te = 1.0

        # test open trailing edge, original leading edge shape
        af.reset(closed_te=False, le_radius=False)
        y_max_ref = 0.1
        y_te_ref = 0.002
        y_pte_ref = -0.234
        y_1c_ref = 0.078
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        y_1c = af.y(xi_1c)
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test open trailing edge, leading edge radius
        af.reset(closed_te=False, le_radius=True)
        y_max_ref = 0.1
        y_te_ref = 0.002
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af._a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

        # test closed trailing edge, original leading edge shape
        af.reset(closed_te=True, le_radius=False)
        y_max_ref = 0.1
        y_te_ref = 0.0
        y_pte_ref = -0.234
        y_1c_ref = 0.078
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        y_1c = af.y(xi_1c)
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test closed trailing edge, leading edge radius
        af.reset(closed_te=True, le_radius=True)
        y_max_ref = 0.1
        y_te_ref = 0.0
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af._a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca4DigitThicknessClassic(thickness=0.3)
        af_closed = Naca4DigitThicknessEnhanced(thickness=0.3, closed_te=True,
                                                le_radius=False)

        def compare_values(xi: np_type.NDArray,
                           af: Naca4DigitThicknessBase) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    yr[...] = (af.t_max/0.20)*(af._a[0]*np.sqrt(xir)
                                               + af._a[1]*xir
                                               + af._a[2]*xir**2
                                               + af._a[3]*xir**3
                                               + af._a[4]*xir**4)

            # compare point values
            y = af.y(xi)
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            ypl = af.y(xi+eps)
            ymi = af.y(xi-eps)
            yp_ref = 0.5*(ypl-ymi)/eps
            yp = af.y_p(xi)
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            ypl = af.y_p(xi+eps)
            ymi = af.y_p(xi-eps)
            ypp_ref = 0.5*(ypl-ymi)/eps
            ypp = af.y_pp(xi)
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        # test point on front
        xi = 0.25
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test point on back
        xi = 0.6
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        xi = np.linspace(0.001, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)


if __name__ == "__main__":
    unittest.main(verbosity=1)