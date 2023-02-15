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

    def testSetters(self) -> None:
        """Test the setting of the max. camber and its location."""
        ca = Naca4DigitCamber(mci=6, lci=2)

        self.assertEqual(ca.max_camber_index, 6)
        self.assertEqual(ca.loc_max_camber_index, 2)

        # test setting non-zero to zero camber
        ca.max_camber_index = 0

        self.assertEqual(ca.max_camber_index, 0)
        self.assertEqual(ca.loc_max_camber_index, 0)

        # test setting zero to non-zero camber
        ca.max_camber_index = 3
        ca.loc_max_camber_index = 4

        self.assertEqual(ca.max_camber_index, 3)
        self.assertEqual(ca.loc_max_camber_index, 4)

        # test setting non-zero to zero camber
        ca.loc_max_camber_index = 0

        self.assertEqual(ca.max_camber_index, 0)
        self.assertEqual(ca.loc_max_camber_index, 0)

    def testClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 62xx
        ca = Naca4DigitCamber(mci=6, lci=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 63xx
        ca = Naca4DigitCamber(mci=6, lci=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 64xx
        ca = Naca4DigitCamber(mci=6, lci=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 65xx
        ca = Naca4DigitCamber(mci=6, lci=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 66xx
        ca = Naca4DigitCamber(mci=6, lci=6)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

        # NACA 67xx
        ca = Naca4DigitCamber(mci=6, lci=7)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(ca.max_camber_index):1d}"
                    + f"{int(ca.loc_max_camber_index):1d}.dat")
        tows.change_case_data(filename=filename)
        x, y = ca.xy(tows.x)
        xt, yt = ca.xy_t(tows.x)
        self.assertIsNone(npt.assert_allclose(x, tows.x))
        self.assertIsNone(npt.assert_allclose(y, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yt/xt, tows.dydx, atol=1e-5))

    def testCamber(self) -> None:
        """Test the camber relations."""
        ca = Naca4DigitCamber(mci=3, lci=4)
        ca_flat = Naca4DigitCamber(mci=0, lci=0)

        def compare_values(xi: np_type.NDArray, af: Naca4DigitCamber) -> None:
            eps = 1e-7

            p, m = af.max_camber()
            xi = np.asarray(xi)
            it = np.nditer([xi, None])
            with it:
                for xit, yit in it:
                    if p == 0:
                        yit[...] = np.zeros_like(xit)
                    else:
                        if xit <= p:
                            yit[...] = (m/p**2)*(2*p*xit - xit**2)
                        else:
                            yit[...] = (m/(1-p)**2)*(1-2*p + 2*p*xit - xit**2)
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
        t = 0.25
        compare_values(t, ca)
        compare_values(t, ca_flat)

        # test point on back
        t = 0.6
        compare_values(t, ca)
        compare_values(t, ca_flat)

        # test points on lower and upper surface
        t = np.linspace(0, 1, 12)
        compare_values(t, ca)
        compare_values(t, ca_flat)

    def testEndpoints(self) -> None:
        af = Naca4DigitCamber(mci=4, lci=2)
        p, m = af.max_camber()

        # reference values
        coef = [m/p**2, m/(1-p)**2]
        x_ref = [0, 1]
        y_ref = [0, 0]
        xt_ref = [1, 1]
        yt_ref = [2*coef[0]*p, 2*coef[1]*(p - 1)]
        xtt_ref = [0, 0]
        ytt_ref = [-2*coef[0], -2*coef[1]]
        xttt_ref = [0, 0]
        yttt_ref = [0, 0]

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
        ca = Naca4DigitCamber(mci=3, lci=4.0)
        ca_flat = Naca4DigitCamber(mci=0, lci=0)

        self.assertListEqual([0.0, 0.4, 1.0], ca.joints())
        self.assertListEqual([0.0, 1.0], ca_flat.joints())

    def testMaxCamber(self) -> None:
        ca = Naca4DigitCamber(mci=6.1, lci=3.0)
        ca_flat = Naca4DigitCamber(mci=0, lci=0)

        self.assertTupleEqual((0.3, 0.061), ca.max_camber())
        self.assertTupleEqual((0, ca_flat.max_camber())


if __name__ == "__main__":
    unittest.main(verbosity=1)
