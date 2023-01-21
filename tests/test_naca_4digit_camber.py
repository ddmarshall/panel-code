#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 20 17:04:28 2023

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.camber import Naca4DigitCamber

from theory_of_wing_sections import camber_data


class TestNaca4DigitCamber(unittest.TestCase):
    """Class to test the NACA 4-digit camber geometry."""

    def testClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 62xx
        af = Naca4DigitCamber(m=6, p=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 63xx
        af = Naca4DigitCamber(m=6, p=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 64xx
        af = Naca4DigitCamber(m=6, p=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 65xx
        af = Naca4DigitCamber(m=6, p=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 66xx
        af = Naca4DigitCamber(m=6, p=6)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 67xx
        af = Naca4DigitCamber(m=6, p=7)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m):1d}{int(af.p):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af = Naca4DigitCamber(m=3, p=4)

        def compare_values(xi: np_type.NDArray, af: Naca4DigitCamber) -> None:
            eps = 1e-7

            p, m = af.max_camber()
            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    if xir <= p:
                        yr[...] = (m/p**2)*(2*p*xir - xir**2)
                    else:
                        yr[...] = (m/(1-p)**2)*(1-2*p + 2*p*xir - xir**2)

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

    def testEndpoints(self) -> None:
        af = Naca4DigitCamber(m=4, p=2)
        p, m = af.max_camber()

        # reference values
        coef = [m/p**2, m/(1-p)**2]
        y_ref = [0, 0]
        yp_ref = [2*coef[0]*p, 2*coef[1]*(p - 1)]
        ypp_ref = [-2*coef[0], -2*coef[1]]
        yppp_ref = [0, 0]

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
        af = Naca4DigitCamber(m=3, p=4.0)

        self.assertListEqual([0.0, 0.4, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        af = Naca4DigitCamber(m=6.1, p=3.0)

        self.assertTupleEqual((0.3, 0.061), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
