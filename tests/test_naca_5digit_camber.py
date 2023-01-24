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

from pyPC.airfoil.camber import (Naca5DigitCamber,
                                 Naca5DigitCamberClassic,
                                 Naca5DigitCamberEnhanced)

from theory_of_wing_sections import camber_data


class TestNaca5DigitCamber(unittest.TestCase):
    """Class to test the NACA 5-digit camber geometry."""

    def testCamberClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 210xx
        af = Naca5DigitCamberClassic(p=1)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.p):1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=3e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=2e-5))

        # NACA 220xx
        af = Naca5DigitCamberClassic(p=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.p):1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 230xx
        af = Naca5DigitCamberClassic(p=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.p):1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 240xx
        af = Naca5DigitCamberClassic(p=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.p):1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 250xx
        af = Naca5DigitCamberClassic(p=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{int(af.p):1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=2e-5))

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamberClassic(p=4)

        self.assertEqual(af.p, 4)
        self.assertEqual(af.ci, 2)

        af = Naca5DigitCamberEnhanced(ci=2.4, p=1.2)

        self.assertEqual(af.p, 1.2)
        self.assertEqual(af.ci, 2.4)

        # Note: while published data from Jacobs and Pinkerton (1936) has
        #       values, they are noticable off from actual values. These
        #       reference values come from previous Matlab implementation.
        p = 20*np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        ci = (20/3.0)*0.3
        # m from ref.    [0.0580,       0.1260,       0.2025,
        #                 0.2900,       0.3910]
        m_ref = np.array([0.0580815972, 0.1257435084, 0.2026819782,
                          0.2903086448, 0.3913440104])
        # k1 from ref.    [361.4000,       51.6400,       15.9570,
        #                  6.6430,       3.2300]
        k1_ref = np.array([350.3324130671, 51.5774898103, 15.9196523168,
                           6.6240446646, 3.2227606900])

        # test the initialization of camber
        af.ci = ci
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            af.p = pit
            self.assertIsNone(npt.assert_allclose(af._m, mit))
            self.assertIsNone(npt.assert_allclose(af._k1, k1it))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af_classic = Naca5DigitCamberClassic(p=2)
        af_enhanced = Naca5DigitCamberEnhanced(ci=3.7, p=2.4)

        def compare_values(xi: np_type.NDArray, af: Naca5DigitCamber) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    m = af.m
                    k1 = af.k1
                    if xir <= m:
                        yr[...] = (k1/6)*(xir**3 - 3*m*xir**2 + m**2*(3-m)*xir)
                    else:
                        yr[...] = (k1*m**3/6)*(1-xir)

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
        af = Naca5DigitCamberClassic(p=3)

        # reference values
        coef = [af.k1/6, af.k1*af.m**3/6]
        y_ref = [0, 0]
        yp_ref = [coef[0]*af.m**2*(3-af.m), -coef[1]]
        ypp_ref = [-6*coef[0]*af.m, 0]
        yppp_ref = [6*coef[0], 0]

        # test leading edge
        xi = 0
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[0]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[0]))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref[0]))

        # test trailing edge
        xi = 1
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref[1]))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref[1]))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref[1]))

        # test both
        xi = np.array([0, 1])
        y = af.y(xi)
        yp = af.y_p(xi)
        ypp = af.y_pp(xi)
        yppp = af.y_ppp(xi)
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(yp, yp_ref))
        self.assertIsNone(npt.assert_allclose(ypp, ypp_ref))
        self.assertIsNone(npt.assert_allclose(yppp, yppp_ref))

    def testJoints(self) -> None:
        af = Naca5DigitCamberClassic(p=3)

        self.assertListEqual([0.0, 0.2025, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        af = Naca5DigitCamberClassic(p=3)

        self.assertTupleEqual((0.15, af.y(0.15)), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
