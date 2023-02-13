#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 01:23:04 2023

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.camber import Naca5DigitCamber, Naca5DigitCamberEnhanced

from theory_of_wing_sections import camber_data


class TestNaca5DigitCamber(unittest.TestCase):
    """Class to test the NACA 5-digit camber geometry."""

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamber(lci=2, mci=4)

        self.assertEqual(af.max_camber_index, 4)
        self.assertEqual(af.lift_coefficient_index, 2)

        af = Naca5DigitCamberEnhanced(lci=2.4, mci=1.2)

        self.assertEqual(af.max_camber_index, 1.2)
        self.assertEqual(af.lift_coefficient_index, 2.4)

        # Note: The published data from Jacobs and Pinkerton (1936) are
        #       noticably off from actual values. These reference values come
        #       from previous Matlab implementation.
        p = 20*np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        lci = (20/3.0)*0.3
        # m from ref.    [0.0580,       0.1260,       0.2025,
        #                 0.2900,       0.3910]
        m_ref = np.array([0.0580815972, 0.1257435084, 0.2026819782,
                          0.2903086448, 0.3913440104])
        # k1 from ref.    [361.4000,       51.6400,       15.9570,
        #                  6.6430,       3.2300]
        k1_ref = np.array([350.3324130671, 51.5774898103, 15.9196523168,
                           6.6240446646, 3.2227606900])

        # test the initialization of camber
        af.lift_coefficient_index = lci
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            af.max_camber_index = pit
            self.assertIsNone(npt.assert_allclose(af._m, mit))
            self.assertIsNone(npt.assert_allclose(af._k1, k1it))

    def testClassicCamber(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 210xx
        af = Naca5DigitCamber(lci=2, mci=1)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.max_camber_index):1d}0.dat")
        tows.change_case_data(filename=filename)
        x, y = af.xy(tows.x)
        xt, yt = af.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=3e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=2e-5))

        # NACA 220xx
        af = Naca5DigitCamber(lci=2, mci=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.max_camber_index):1d}0.dat")
        tows.change_case_data(filename=filename)
        x, y = af.xy(tows.x)
        xt, yt = af.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 230xx
        af = Naca5DigitCamber(lci=2, mci=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.max_camber_index):1d}0.dat")
        tows.change_case_data(filename=filename)
        x, y = af.xy(tows.x)
        xt, yt = af.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 240xx
        af = Naca5DigitCamber(lci=2, mci=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.max_camber_index):1d}0.dat")
        tows.change_case_data(filename=filename)
        x, y = af.xy(tows.x)
        xt, yt = af.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 250xx
        af = Naca5DigitCamber(lci=2, mci=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.max_camber_index):1d}0.dat")
        tows.change_case_data(filename=filename)
        x, y = af.xy(tows.x)
        xt, yt = af.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=2e-5))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af_classic = Naca5DigitCamber(lci=2, mci=2)
        af_enhanced = Naca5DigitCamberEnhanced(lci=3.7, mci=2.4)

        def compare_values(xi: np_type.NDArray, af: Naca5DigitCamber) -> None:
            eps = 1e-7

            m = af.m
            k1 = af.k1
            xi = np.asarray(xi)
            it = np.nditer([xi, None])
            with it:
                for xit, yit in it:
                    if xit <= m:
                        yit[...] = (k1/6)*(xit**3
                                           - 3*m*xit**2 + m**2*(3-m)*xit)
                    else:
                        yit[...] = (k1*m**3/6)*(1-xit)
                y_ref = it.operands[1]

            # compare point values
            x, y = af.xy(xi)
            self.assertIsNone(npt.assert_allclose(x, xi))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xpl, ypl = af.xy(xi+eps)
            xmi, ymi = af.xy(xi-eps)
            xt_ref = 0.5*(xpl-xmi)/eps
            yt_ref = 0.5*(ypl-ymi)/eps
            xt, yt = af.xy_t(xi)
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

            # compare third derivatives
            xpl, ypl = af.xy_tt(xi+eps)
            xmi, ymi = af.xy_tt(xi-eps)
            xttt_ref = 0.5*(xpl-xmi)/eps
            yttt_ref = 0.5*(ypl-ymi)/eps
            xttt, yttt = af.xy_ttt(xi)
            self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
            self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

        # test point on front
        xi = 0.125
        compare_values(xi, af_classic)
        compare_values(xi, af_enhanced)

        # test point on back
        xi = 0.6
        compare_values(xi, af_classic)
        compare_values(xi, af_enhanced)

        # test points on lower and upper surface
        xi = np.linspace(0, 1, 12)
        compare_values(xi, af_classic)
        compare_values(xi, af_enhanced)

    def testEndpoints(self) -> None:
        """Test accessing the end points of camber with integers."""
        af = Naca5DigitCamber(lci=2, mci=3)

        # reference values
        coef = [af.k1/6, af.k1*af.m**3/6]
        x_ref = [0, 1]
        y_ref = [0, 0]
        xt_ref = [1, 1]
        yt_ref = [coef[0]*af.m**2*(3-af.m), -coef[1]]
        xtt_ref = [0, 0]
        ytt_ref = [-6*coef[0]*af.m, 0]
        xttt_ref = [0, 0]
        yttt_ref = [6*coef[0], 0]

        # test leading edge
        xi = 0
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[0]))

        # test trailing edge
        xi = 1
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref[1]))

        # test both
        xi = np.array([0, 1])
        x, y = af.xy(xi)
        xt, yt = af.xy_t(xi)
        xtt, ytt = af.xy_tt(xi)
        xttt, yttt = af.xy_ttt(xi)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))
        self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
        self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        af = Naca5DigitCamber(lci=2, mci=3)

        self.assertListEqual([0.0, 0.2025, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        """Test maximum camber."""
        af = Naca5DigitCamber(lci=2, mci=3)

        self.assertTupleEqual((0.15, af.xy(0.15)[1]), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
