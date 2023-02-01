#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 22:37:52 2022

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.thickness import (Naca45DigitModifiedThickness,
                                    Naca45DigitModifiedThicknessEnhanced)

from theory_of_wing_sections import thickness_data


class TestNaca45DigitModifiedThickness(unittest.TestCase):
    """Class to test the NACA modified 4-digit thickness geometry."""

    def testSetters(self) -> None:
        """Test the setting of thickness parameters."""
        af_open = Naca45DigitModifiedThickness(tmax=20, lei=4, xi_m=5)

        self.assertAlmostEqual(af_open.tmax, 20, delta=1e-7)
        self.assertAlmostEqual(af_open.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(af_open.max_thickness_index, 5, delta=1e-7)

        af_closed = Naca45DigitModifiedThicknessEnhanced(tmax=20, lei=4,
                                                         xi_m=5,
                                                         closed_te=True)

        self.assertAlmostEqual(af_closed.tmax, 20, delta=1e-7)
        self.assertAlmostEqual(af_closed.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(af_closed.max_thickness_index, 5, delta=1e-7)
        self.assertTrue(af_closed.closed_te)

        # test initializing enhanced parameters
        # sharp trailing edge
        I_ref = 3
        M_ref = 4
        d_ref = [0, 1.575, -1.0833333333, -0.2546296297]
        a_ref = [0.74225, 0.9328327771, -2.6241657396, 1.2077076380]
        af_closed.tmax = 12
        af_closed.max_thickness_index = M_ref
        af_closed.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(af_closed.a, a_ref))
        self.assertIsNone(npt.assert_allclose(af_closed.d, d_ref))

        # test initializing classic parameters
        I_ref = 3
        M_ref = 4
        d_ref = [0.01, 1.575, -1.1666666667, -0.1620370371]
        a_ref = [0.74225, 0.9661661252, -2.7908324797, 1.4160410631]
        af_open.tmax = 12
        af_open.max_thickness_index = M_ref
        af_open.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(af_open.a, a_ref))
        self.assertIsNone(npt.assert_allclose(af_open.d, d_ref))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0008-34
        af = Naca45DigitModifiedThickness(tmax=8, lei=3, xi_m=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=3e-5))

        # NACA 0010-34
        af = Naca45DigitModifiedThickness(tmax=10, lei=3, xi_m=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=4e-5))

        # NACA 0010-35
        af = Naca45DigitModifiedThickness(tmax=10, lei=3, xi_m=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=4e-5))

        # NACA 0010-64
        af = Naca45DigitModifiedThickness(tmax=10, lei=6, xi_m=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=2e-5))

        # NACA 0010-65
        af = Naca45DigitModifiedThickness(tmax=10, lei=6, xi_m=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=2e-5))

        # NACA 0010-66
        af = Naca45DigitModifiedThickness(tmax=10, lei=6, xi_m=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=2e-5))

        # NACA 0012-34
        af = Naca45DigitModifiedThickness(tmax=12, lei=3, xi_m=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0012-64
        af = Naca45DigitModifiedThickness(tmax=12, lei=6, xi_m=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.tmax):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=8e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca45DigitModifiedThicknessEnhanced(tmax=20, lei=4.3, xi_m=5.6,
                                                  closed_te=False)

        # test the settings with open trailing edge
        xi_m = af.max_thickness_index/10.0
        xi_te = 1.0
        y_m = af.y(xi_m)
        yp_m = af.y_p(xi_m)
        y_te = af.y(xi_te)
        self.assertIsNone(npt.assert_allclose(y_m, 0.005*af.tmax))
        self.assertIsNone(npt.assert_allclose(yp_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, 0.02*0.10))

        # test the settings with close trailing edge
        af.closed_te = True
        y_m = af.y(xi_m)
        yp_m = af.y_p(xi_m)
        y_te = af.y(xi_te)
        self.assertIsNone(npt.assert_allclose(y_m, 0.005*af.tmax))
        self.assertIsNone(npt.assert_allclose(yp_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, 0, atol=1e-7))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca45DigitModifiedThickness(tmax=10, lei=6, xi_m=4)
        af_closed = Naca45DigitModifiedThicknessEnhanced(tmax=15.2, lei=4.3,
                                                         xi_m=5.6,
                                                         closed_te=True)

        def compare_values(xi: np_type.NDArray,
                           af: Naca45DigitModifiedThickness) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    if xir <= af.max_thickness_index/10.0:
                        yr[...] = af.tmax/100.0*(af.a[0]*np.sqrt(xir)
                                                 + af.a[1]*xir
                                                 + af.a[2]*xir**2
                                                 + af.a[3]*xir**3)
                    else:
                        yr[...] = af.tmax/100.0*(af.d[0] + af.d[1]*(1-xir)
                                                 + af.d[2]*(1-xir)**2
                                                 + af.d[3]*(1-xir)**3)

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
        xi = 0.60
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        xi = np.linspace(0.001, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        af = Naca45DigitModifiedThickness(tmax=12, lei=4, xi_m=6)

        # reference values
        y_ref = [0, 0.0012]
        yp_ref = [np.inf, -0.4200]
        ypp_ref = [-np.inf, -1.995]
        k_ref = [af.le_k(), -1.5635449681]

        # test leading edge
        xi = 0
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        xi = 1
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        xi = np.array([0, 1])
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        k = af.k(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref))
        self.assertIsNone(npt.assert_allclose(k, k_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        af = Naca45DigitModifiedThickness(tmax=24, lei=3, xi_m=5)

        self.assertListEqual([0.0, 0.5, 1.0], af.joints())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        af = Naca45DigitModifiedThickness(tmax=24, lei=3, xi_m=5)

        xi_max, y_max = af.max_thickness()
        self.assertAlmostEqual(0.5, xi_max, delta=1e-7)
        self.assertAlmostEqual(0.12, y_max, delta=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=1)
