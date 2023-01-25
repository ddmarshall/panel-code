#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:09:35 2023

@author: ddmarshall
"""

import unittest

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.airfoil.camber import (Naca5DigitCamberReflexed,
                                 Naca5DigitCamberReflexedClassic,
                                 Naca5DigitCamberReflexedEnhanced)


class TestNaca5DigitReflexed(unittest.TestCase):
    """Class to test the NACA 5-digit reflexed camber geometry."""

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamberReflexedClassic(p=3)

        self.assertEqual(af.p, 3)
        self.assertEqual(af.ci, 2)

        af = Naca5DigitCamberReflexedEnhanced(ci=2.3, p=3.2)

        self.assertEqual(af.p, 3.2)
        self.assertEqual(af.ci, 2.3)

        # Note: while published data from Jacobs and Pinkerton (1936) has
        #       values, they are noticable off from actual values. These
        #       reference values come from previous Matlab implementation.
        p = 20*np.array([0.10, 0.15, 0.20, 0.25])
        ci = (20/3.0)*0.3
        # m from ref.    [0.1300,       0.2170,       0.3180,
        #                 0.4410]
        m_ref = np.array([0.1307497584, 0.2160145029, 0.3179188983,
                          0.4408303366])
        # k1 from ref.    [51.99,         15.793,        6.520,
        #                  3.191]
        k1_ref = np.array([51.1202497358, 15.6909755022, 6.5072929322,
                           3.1755242567])
        # k2 from ref.    [0.03972036,   0.10691861,   0.197556,
        #                  0.4323805]
        k2_ref = np.array([0.0468090174, 0.0974945233, 0.1964888189,
                           0.4283075436])

        # test the static methods
        for pit, mit, k1it, k2it in np.nditer([p, m_ref, k1_ref, k2_ref]):
            Cl_id = af._Cl_id(m=mit, k1=k1it, k2ok1=k2it/k1it)
            k2ok1 = af._k2ok1(m=mit, p=pit/20)
            k2ok1_ref = k2it/k1it
            self.assertIsNone(npt.assert_allclose(Cl_id, 3.0*ci/20))
            self.assertIsNone(npt.assert_allclose(k2ok1, k2ok1_ref))

        # test the initialization of camber
        af.ci = ci
        for pit, mit, k1it, k2it in np.nditer([p, m_ref, k1_ref, k2_ref]):
            af.p = pit
            self.assertIsNone(npt.assert_allclose(af.m, mit))
            self.assertIsNone(npt.assert_allclose(af.k1, k1it))
            self.assertIsNone(npt.assert_allclose(af.k2, k2it))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af_classic = Naca5DigitCamberReflexedClassic(p=2)
        af_enhanced = Naca5DigitCamberReflexedEnhanced(ci=3.7, p=2.4)

        def compare_values(xi: np_type.NDArray,
                           af: Naca5DigitCamberReflexed) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    m = af.m
                    k1 = af.k1
                    k2ok1 = af.k2/k1
                    if xir <= m:
                        yr[...] = (k1/6)*((xir-m)**3 - k2ok1*(1-m)**3*xir
                                          - m**3*xir + m**3)
                    else:
                        yr[...] = (k1/6)*(k2ok1*(xir-m)**3 - k2ok1*(1-m)**3*xir
                                          - m**3*xir + m**3)

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
        """Test accessing the end points of camber with integers."""
        af = Naca5DigitCamberReflexedClassic(p=3)

        # reference values
        coef = [af.k1/6, af.k1/6]
        y_ref = [0, 0]
        yp_ref = [coef[0]*(3*af.m**2-(af.k2/af.k1)*(1-af.m)**3-af.m**3),
                  coef[1]*((af.k2/af.k1)*(3*(1-af.m)**2-(1-af.m)**3)-af.m**3)]
        ypp_ref = [-6*coef[0]*af.m, 6*coef[1]*(af.k2/af.k1)*(1-af.m)]
        yppp_ref = [6*coef[0], 6*coef[1]*(af.k2/af.k1)]

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
        """Test correct joints are being reported."""
        af = Naca5DigitCamberReflexedClassic(p=3)

        self.assertListEqual([0.0, 0.2170, 1.0], af.joints())

    def testMaxCamber(self) -> None:
        """Test maximum camber."""
        af = Naca5DigitCamberReflexedClassic(p=3)

        self.assertTupleEqual((0.15, af.y(0.15)), af.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
