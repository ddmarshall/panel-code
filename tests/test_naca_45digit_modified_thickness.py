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
        af_open = Naca45DigitModifiedThickness(mti=20, lei=4, lmti=5)

        self.assertAlmostEqual(af_open.max_thickness_index, 20, delta=1e-7)
        self.assertAlmostEqual(af_open.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(af_open.loc_max_thickness_index, 5, delta=1e-7)

        af_closed = Naca45DigitModifiedThicknessEnhanced(mti=20, lei=4,
                                                         lmti=5,
                                                         closed_te=True)

        self.assertAlmostEqual(af_closed.max_thickness_index, 20, delta=1e-7)
        self.assertAlmostEqual(af_closed.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(af_closed.loc_max_thickness_index, 5,
                               delta=1e-7)
        self.assertTrue(af_closed.closed_te)

        # test initializing enhanced parameters
        # sharp trailing edge
        I_ref = 3
        M_ref = 4
        d_ref = [0, 1.575, -1.0833333333, -0.2546296297]
        a_ref = [0.74225, 0.9328327771, -2.6241657396, 1.2077076380]
        af_closed.max_thickness_index = 12
        af_closed.loc_max_thickness_index = M_ref
        af_closed.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(af_closed.a, a_ref))
        self.assertIsNone(npt.assert_allclose(af_closed.d, d_ref))

        # test initializing classic parameters
        I_ref = 3
        M_ref = 4
        d_ref = [0.01, 1.575, -1.1666666667, -0.1620370371]
        a_ref = [0.74225, 0.9661661252, -2.7908324797, 1.4160410631]
        af_open.max_thickness_index = 12
        af_open.loc_max_thickness_index = M_ref
        af_open.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(af_open.a, a_ref))
        self.assertIsNone(npt.assert_allclose(af_open.d, d_ref))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0008-34
        af = Naca45DigitModifiedThickness(mti=8, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=3e-5))

        # NACA 0010-34
        af = Naca45DigitModifiedThickness(mti=10, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

        # NACA 0010-35
        af = Naca45DigitModifiedThickness(mti=10, lei=3, lmti=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

        # NACA 0010-64
        af = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0010-65
        af = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0010-66
        af = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0012-34
        af = Naca45DigitModifiedThickness(mti=12, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0012-64
        af = Naca45DigitModifiedThickness(mti=12, lei=6, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(af.max_thickness_index):02d}"
                    f"-{int(af.leading_edge_index)}"
                    f"{int(af.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        x, y = af.xy(t)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=8e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca45DigitModifiedThicknessEnhanced(mti=20, lei=4.3, lmti=5.6,
                                                  closed_te=False)

        # test the settings with open trailing edge
        xi_m = af.loc_max_thickness_index/10.0
        t_m = np.sqrt(xi_m)
        xi_te = 1.0
        t_te = 1.0
        y_m_ref = 0.005*af.max_thickness_index
        x_m, y_m = af.xy(t_m)
        xt_m, yt_m = af.xy_t(t_m)
        x_te, y_te = af.xy(t_te)
        self.assertIsNone(npt.assert_allclose(x_m, xi_m))
        self.assertIsNone(npt.assert_allclose(y_m, y_m_ref))
        self.assertIsNone(npt.assert_allclose(xt_m, 2*t_m))
        self.assertIsNone(npt.assert_allclose(yt_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, 0.02*0.10))

        # test the settings with close trailing edge
        af.closed_te = True
        x_m, y_m = af.xy(t_m)
        xt_m, yt_m = af.xy_t(t_m)
        x_te, y_te = af.xy(t_te)
        self.assertIsNone(npt.assert_allclose(x_m, xi_m))
        self.assertIsNone(npt.assert_allclose(y_m, y_m_ref))
        self.assertIsNone(npt.assert_allclose(xt_m, 2*t_m))
        self.assertIsNone(npt.assert_allclose(yt_m, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(x_te, xi_te))
        self.assertIsNone(npt.assert_allclose(y_te, 0))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=4)
        af_closed = Naca45DigitModifiedThicknessEnhanced(mti=15.2, lei=4.3,
                                                         lmti=5.6,
                                                         closed_te=True)

        def compare_values(xi: np_type.NDArray,
                           af: Naca45DigitModifiedThickness) -> None:
            eps = 1e-7

            t = np.sqrt(np.asarray(xi))
            y_ref = np.zeros_like(t)
            it = np.nditer([t, y_ref], op_flags=[["readonly"], ["writeonly"]])
            with it:
                for tr, yr in it:
                    tmax = af.max_thickness_index/100.0
                    if tr**2 <= af.loc_max_thickness_index/10.0:
                        yr[...] = tmax*(af.a[0]*tr + af.a[1]*tr**2
                                        + af.a[2]*tr**4 + af.a[3]*tr**6)
                    else:
                        yr[...] = tmax*(af.d[0] + af.d[1]*(1-tr**2)
                                        + af.d[2]*(1-tr**2)**2
                                        + af.d[3]*(1-tr**2)**3)

            # compare point values
            x, y = af.xy(np.sqrt(xi))
            self.assertIsNone(npt.assert_allclose(x, xi))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xpl, ypl = af.xy(t+eps)
            xmi, ymi = af.xy(t-eps)
            xt_ref = 0.5*(xpl-xmi)/eps
            yt_ref = 0.5*(ypl-ymi)/eps
            xt, yt = af.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xt, xt_ref))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref))

            # compare second derivatives
            xpl, ypl = af.xy_t(xi+eps)
            xmi, ymi = af.xy_t(xi-eps)
            xtt_ref = 0.5*(xpl-xmi)/eps
            ytt_ref = 0.5*(ypl-ymi)/eps
            xtt, ytt = af.xy_tt(xi)
            self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

        # test point on front
        xi = 0.25
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test point on back
        xi = 0.60
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on fore and aft
        xi = np.linspace(0, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        af = Naca45DigitModifiedThickness(mti=12, lei=4, lmti=6)

        # reference values
        x_ref = [0, 1]
        y_ref = [0, 0.0012]
        xt_ref = [0, 2]
        yt_ref = [0.11876, -0.8400]
        xtt_ref = [2, 2]
        ytt_ref = [-0.0379443778, -8.82]
        k_ref = [-141.804371, -1.5635449681]

        # test leading edge
        t = 0
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[0]))

        # test trailing edge
        t = 1
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))
        self.assertIsNone(npt.assert_allclose(k, k_ref[1]))

        # test both
        t = np.array([0, 1])
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        k = af.k(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))
        self.assertIsNone(npt.assert_allclose(k, k_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        af = Naca45DigitModifiedThickness(mti=24, lei=3, lmti=5)

        self.assertListEqual([0.0, np.sqrt(0.5), 1.0], af.joints())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        af = Naca45DigitModifiedThickness(mti=24, lei=3, lmti=5)

        xi_max, y_max = af.max_thickness()
        self.assertAlmostEqual(0.5, xi_max, delta=1e-7)
        self.assertAlmostEqual(0.12, y_max, delta=1e-7)


if __name__ == "__main__":
    unittest.main(verbosity=1)
