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
        # a from ref.    [0.29690,  -0.12600,  -0.35160,  0.28430,
        #                 -0.10150]/0.20
        a_ref = np.array([0.296676, -0.125834, -0.350607, 0.282350,
                          -0.100585])/0.20

        af.closed_trailing_edge = False
        af.use_leading_edge_radius = False
        self.assertIsNone(npt.assert_allclose(af.a, a_ref, rtol=1e-6))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        af = Naca45DigitThickness(mti=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=5e-5))

        # NACA 0008
        af = Naca45DigitThickness(mti=8)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=6e-5))

        # NACA 0009
        af = Naca45DigitThickness(mti=9)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=3e-5))

        # NACA 0010
        af = Naca45DigitThickness(mti=10)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0012
        af = Naca45DigitThickness(mti=12)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=7e-5))

        # NACA 0015
        af = Naca45DigitThickness(mti=15)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0018
        af = Naca45DigitThickness(mti=18)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0021
        af = Naca45DigitThickness(mti=21)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0024
        af = Naca45DigitThickness(mti=24)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca45DigitThicknessEnhanced(mti=20, closed_te=True,
                                          use_radius=False)
        xi_max = 0.3
        t_max = np.sqrt(xi_max)
        xi_1c = 0.1
        t_1c = np.sqrt(xi_1c)
        xi_te = 1.0
        t_te = 1.0

        # test open trailing edge, original leading edge shape
        af.closed_trailing_edge = False
        af.use_leading_edge_radius = False
        y_max_ref = 0.1
        y_te_ref = 0.002
        dydx_te_ref = -0.234
        y_1c_ref = 0.078
        x_max, y_max = af.xy(t_max)
        x_pmax, y_pmax = af.xy_p(t_max)
        x_te, y_te = af.xy(t_te)
        x_pte, y_pte = af.xy_p(t_te)
        dydx_te = y_pte/x_pte
        x_1c, y_1c = af.xy(t_1c)
        self.assertIsNone(npt.assert_allclose(x_max, xi_max))
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(x_pmax, 2*t_max))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(dydx_te, dydx_te_ref))
        self.assertIsNone(npt.assert_allclose(x_1c, xi_1c))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test open trailing edge, leading edge radius
        af.closed_trailing_edge = False
        af.use_leading_edge_radius = True
        y_max_ref = 0.1
        y_te_ref = 0.002
        dydx_te_ref = -0.234
        r_le_ref = -0.5*0.29690**2
        x_max, y_max = af.xy(t_max)
        x_pmax, y_pmax = af.xy_p(t_max)
        x_te, y_te = af.xy(t_te)
        x_pte, y_pte = af.xy_p(t_te)
        dydx_te = y_pte/x_pte
        r_le = 1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x_max, xi_max))
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(x_pmax, 2*t_max))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(dydx_te, dydx_te_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

        # test closed trailing edge, original leading edge shape
        af.closed_trailing_edge = True
        af.use_leading_edge_radius = False
        y_max_ref = 0.1
        y_te_ref = 0.0
        dydx_te_ref = -0.234
        y_1c_ref = 0.078
        x_max, y_max = af.xy(t_max)
        x_pmax, y_pmax = af.xy_p(t_max)
        x_te, y_te = af.xy(t_te)
        x_pte, y_pte = af.xy_p(t_te)
        dydx_te = y_pte/x_pte
        x_1c, y_1c = af.xy(t_1c)
        self.assertIsNone(npt.assert_allclose(x_max, xi_max))
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(x_pmax, 2*t_max))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(dydx_te, dydx_te_ref))
        self.assertIsNone(npt.assert_allclose(x_1c, xi_1c))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test closed trailing edge, leading edge radius
        af.closed_trailing_edge = True
        af.use_leading_edge_radius = True
        y_max_ref = 0.1
        y_te_ref = 0.0
        dydx_te_ref = -0.234
        r_le_ref = -0.5*0.29690**2
        x_max, y_max = af.xy(t_max)
        x_pmax, y_pmax = af.xy_p(t_max)
        x_te, y_te = af.xy(t_te)
        x_pte, y_pte = af.xy_p(t_te)
        dydx_te = y_pte/x_pte
        r_le = 1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x_max, xi_max))
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(x_pmax, 2*t_max))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(dydx_te, dydx_te_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca45DigitThickness(mti=30)
        af_closed = Naca45DigitThicknessEnhanced(mti=24.2, closed_te=True,
                                                 use_radius=False)

        def compare_values(xi: np_type.NDArray,
                           af: Naca45DigitThickness) -> None:
            eps = 1e-7

            t = np.sqrt(np.asarray(xi))
            y_ref = np.zeros_like(t)
            it = np.nditer([t, y_ref], op_flags=[["readonly"], ["writeonly"]])
            with it:
                for tr, yr in it:
                    tmax = af.max_thickness_index/100.0
                    yr[...] = tmax*(af.a[0]*tr + af.a[1]*tr**2 + af.a[2]*tr**4
                                    + af.a[3]*tr**6 + af.a[4]*tr**8)

            # compare point values
            x, y = af.xy(np.sqrt(xi))
            self.assertIsNone(npt.assert_allclose(x, xi))
            self.assertIsNone(npt.assert_allclose(y, y_ref, atol=1e-7))

            # compare first derivatives
            xpl, ypl = af.xy(t+eps)
            xmi, ymi = af.xy(t-eps)
            xp_ref = 0.5*(xpl-xmi)/eps
            yp_ref = 0.5*(ypl-ymi)/eps
            xp, yp = af.xy_p(t)
            self.assertIsNone(npt.assert_allclose(xp, xp_ref))
            self.assertIsNone(npt.assert_allclose(yp, yp_ref))

            # compare second derivatives
            xpl, ypl = af.xy_p(xi+eps)
            xmi, ymi = af.xy_p(xi-eps)
            xpp_ref = 0.5*(xpl-xmi)/eps
            ypp_ref = 0.5*(ypl-ymi)/eps
            xpp, ypp = af.xy_pp(xi)
            self.assertIsNone(npt.assert_allclose(xpp, xpp_ref))
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref))

        # test point on front
        xi = 0.25
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test point on back
        xi = 0.6
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on fore and aft
        xi = np.linspace(0, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        af = Naca45DigitThickness(mti=12)

        # reference values
        x_ref = [0, 1]
        y_ref = [0, 0.00126]
        xp_ref = [0, 2]
        yp_ref = [0.17814, -0.28062]
        xpp_ref = [2, 2]
        ypp_ref = [-0.1512, -0.97572]
        k_ref = [-63.02416489, -0.1687668093]

        # test leading edge
        t = 0
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref[0]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        t = 1
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        t = np.array([0, 1])
        x, y = af.xy(t)
        xp, yp = af.xy_p(t)
        xpp, ypp = af.xy_pp(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xp, xp_ref))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
        self.assertIsNone(npt.assert_allclose(xpp, xpp_ref))
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
