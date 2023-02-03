#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 13:32:30 2023

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.thickness import (Naca45DigitThickness,
                                    Naca45DigitThicknessEnhanced)

from theory_of_wing_sections import thickness_data


class TestNaca45DigitThickness(unittest.TestCase):
    """Class to test the NACA 4-digit thickness geometry."""

    def testSetters(self) -> None:
        """Test the setting of thickness parameters."""
        af = Naca45DigitThickness(mti=14)

        self.assertAlmostEqual(af.max_thickness_index, 14, delta=1e-7)

        af = Naca45DigitThicknessEnhanced(mti=18.5, closed_te=True,
                                          use_radius=True)
        self.assertAlmostEqual(af.max_thickness_index, 18.5, delta=1e-7)
        self.assertTrue(af.closed_trailing_edge)
        self.assertTrue(af.use_leading_edge_radius)

        # Note: The published values from Jacobs, Ward, and Pinkerton (1933)
        #       are only accurate to 3 sig. figs. These reference values come
        #       from previous Matlab implementation.
        # a from ref.    [0.29690,  -0.12600,  -0.35160,  0.28430,  -0.10150]
        a_ref = np.array([0.296676, -0.125834, -0.350607, 0.282350, -0.100585])

        af.closed_trailing_edge = False
        af.use_leading_edge_radius = False
        self.assertIsNone(npt.assert_allclose(af._a, a_ref, rtol=1e-6))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        af = Naca45DigitThickness(mti=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=5e-5))

        # NACA 0008
        af = Naca45DigitThickness(mti=8)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=6e-5))

        # NACA 0009
        af = Naca45DigitThickness(mti=9)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=3e-5))

        # NACA 0010
        af = Naca45DigitThickness(mti=10)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=2e-5))

        # NACA 0012
        af = Naca45DigitThickness(mti=12)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=7e-5))

        # NACA 0015
        af = Naca45DigitThickness(mti=15)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0018
        af = Naca45DigitThickness(mti=18)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0021
        af = Naca45DigitThickness(mti=21)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0024
        af = Naca45DigitThickness(mti=24)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=4e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca45DigitThicknessEnhanced(mti=20, closed_te=True,
                                          use_radius=False)
        xi_max = 0.3
        xi_1c = 0.1
        xi_te = 1.0

        # test open trailing edge, original leading edge shape
        af.closed_trailing_edge = False
        af.use_leading_edge_radius = False
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
        af.closed_trailing_edge = False
        af.use_leading_edge_radius = True
        y_max_ref = 0.1
        y_te_ref = 0.002
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af.a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

        # test closed trailing edge, original leading edge shape
        af.closed_trailing_edge = True
        af.use_leading_edge_radius = False
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
        af.closed_trailing_edge = True
        af.use_leading_edge_radius = True
        y_max_ref = 0.1
        y_te_ref = 0.0
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af.a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca45DigitThickness(mti=30)
        af_closed = Naca45DigitThicknessEnhanced(mti=24.2, closed_te=True,
                                                 use_radius=False)

        def compare_values(xi: np_type.NDArray,
                           af: Naca45DigitThickness) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    tmax = af.max_thickness_index/20.0
                    yr[...] = tmax*(af.a[0]*np.sqrt(xir) + af.a[1]*xir
                                    + af.a[2]*xir**2 + af.a[3]*xir**3
                                    + af.a[4]*xir**4)

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

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        af = Naca45DigitThickness(mti=12)

        # reference values
        y_ref = [0, 0.00126]
        yp_ref = [np.inf, -0.14031]
        ypp_ref = [-np.inf, -0.173775]
        k_ref = [af.le_k(), -0.1687668093]

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
        af = Naca45DigitThickness(mti=24)

        self.assertListEqual([0.0, 1.0], af.joints())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        af = Naca45DigitThickness(mti=24)

        xi_max, y_max = af.max_thickness()
        self.assertAlmostEqual(0.3, xi_max, delta=1e-7)
        self.assertAlmostEqual(0.12, y_max, delta=4e-5)


if __name__ == "__main__":
    unittest.main(verbosity=1)
