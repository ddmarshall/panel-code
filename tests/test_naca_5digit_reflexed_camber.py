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
                                 Naca5DigitCamberReflexedEnhanced)


class TestNaca5DigitReflexed(unittest.TestCase):
    """Class to test the NACA 5-digit reflexed camber geometry."""

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        ca = Naca5DigitCamberReflexed(lci=2, mci=3)

        self.assertEqual(ca.max_camber_index, 3)
        self.assertEqual(ca.lift_coefficient_index, 2)

        ca = Naca5DigitCamberReflexedEnhanced(lci=2.3, mci=3.2)

        self.assertEqual(ca.max_camber_index, 3.2)
        self.assertEqual(ca.lift_coefficient_index, 2.3)

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
            Cl_id = ca._Cl_id(m=mit, k1=k1it, k2ok1=k2it/k1it)
            k2ok1 = ca._k2ok1(m=mit, p=pit/20)
            k2ok1_ref = k2it/k1it
            self.assertIsNone(npt.assert_allclose(Cl_id, 3.0*ci/20))
            self.assertIsNone(npt.assert_allclose(k2ok1, k2ok1_ref))

        # test the initialization of camber
        ca.lift_coefficient_index = ci
        for pit, mit, k1it, k2it in np.nditer([p, m_ref, k1_ref, k2_ref]):
            ca.max_camber_index = pit
            self.assertIsNone(npt.assert_allclose(ca.m, mit))
            self.assertIsNone(npt.assert_allclose(ca.k1, k1it))
            self.assertIsNone(npt.assert_allclose(ca.k2, k2it))

    def testCamber(self) -> None:
        """Test the camber relations."""
        ca_classic = Naca5DigitCamberReflexed(lci=2, mci=2)
        ca_enhanced = Naca5DigitCamberReflexedEnhanced(lci=3.7, mci=2.4)

        def compare_values(t: np_type.NDArray,
                           ca: Naca5DigitCamberReflexed) -> None:
            eps = 1e-7

            m = ca.m
            k1 = ca.k1
            k2ok1 = ca.k2/k1
            t = np.asarray(t)
            it = np.nditer([t, None])
            with it:
                for xit, yit in it:
                    if xit <= m:
                        yit[...] = (k1/6)*((xit-m)**3 - k2ok1*(1-m)**3*xit
                                           - m**3*xit + m**3)
                    else:
                        yit[...] = (k1/6)*(k2ok1*(xit-m)**3
                                           - k2ok1*(1-m)**3*xit
                                           - m**3*xit + m**3)
                y_ref = it.operands[1]

            # compare point values
            x, y = ca.xy(t)
            self.assertIsNone(npt.assert_allclose(x, t))
            self.assertIsNone(npt.assert_allclose(y, y_ref))

            # compare first derivatives
            xpl, ypl = ca.xy(t+eps)
            xmi, ymi = ca.xy(t-eps)
            xt_ref = 0.5*(xpl-xmi)/eps
            yt_ref = 0.5*(ypl-ymi)/eps
            xt, yt = ca.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xt, xt_ref))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref))

            # compare second derivatives
            xpl, ypl = ca.xy_t(t+eps)
            xmi, ymi = ca.xy_t(t-eps)
            xtt_ref = 0.5*(xpl-xmi)/eps
            ytt_ref = 0.5*(ypl-ymi)/eps
            xtt, ytt = ca.xy_tt(t)
            self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

            # compare third derivatives
            xpl, ypl = ca.xy_tt(t+eps)
            xmi, ymi = ca.xy_tt(t-eps)
            xttt_ref = 0.5*(xpl-xmi)/eps
            yttt_ref = 0.5*(ypl-ymi)/eps
            xttt, yttt = ca.xy_ttt(t)
            self.assertIsNone(npt.assert_allclose(xttt, xttt_ref))
            self.assertIsNone(npt.assert_allclose(yttt, yttt_ref))

        # test point on front
        t = 0.125
        compare_values(t, ca_classic)
        compare_values(t, ca_enhanced)

        # test point on back
        t = 0.6
        compare_values(t, ca_classic)
        compare_values(t, ca_enhanced)

        # test points on lower and upper surface
        t = np.linspace(0, 1, 12)
        compare_values(t, ca_classic)
        compare_values(t, ca_enhanced)

    def testEndpoints(self) -> None:
        """Test accessing the end points of camber with integers."""
        ca = Naca5DigitCamberReflexed(lci=2, mci=3)

        # reference values
        coef = [ca.k1/6, ca.k1/6]
        x_ref = [0, 1]
        y_ref = [0, 0]
        xt_ref = [1, 1]
        yt_ref = [coef[0]*(3*ca.m**2-(ca.k2/ca.k1)*(1-ca.m)**3-ca.m**3),
                  coef[1]*((ca.k2/ca.k1)*(3*(1-ca.m)**2-(1-ca.m)**3)-ca.m**3)]
        xtt_ref = [0, 0]
        ytt_ref = [-6*coef[0]*ca.m, 6*coef[1]*(ca.k2/ca.k1)*(1-ca.m)]
        xttt_ref = [0, 0]
        yttt_ref = [6*coef[0], 6*coef[1]*(ca.k2/ca.k1)]

        # test leading edge
        xi = 0
        x, y = ca.xy(xi)
        xt, yt = ca.xy_t(xi)
        xtt, ytt = ca.xy_tt(xi)
        xttt, yttt = ca.xy_ttt(xi)
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
        x, y = ca.xy(xi)
        xt, yt = ca.xy_t(xi)
        xtt, ytt = ca.xy_tt(xi)
        xttt, yttt = ca.xy_ttt(xi)
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
        x, y = ca.xy(xi)
        xt, yt = ca.xy_t(xi)
        xtt, ytt = ca.xy_tt(xi)
        xttt, yttt = ca.xy_ttt(xi)
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
        ca = Naca5DigitCamberReflexed(lci=2, mci=3)

        self.assertListEqual([0.0, 0.2170, 1.0], ca.joints())

    def testMaxCamber(self) -> None:
        """Test maximum camber."""
        ca = Naca5DigitCamberReflexed(lci=2, mci=3)

        self.assertAlmostEqual(0.15, ca.max_camber_parameter())


if __name__ == "__main__":
    unittest.main(verbosity=1)
