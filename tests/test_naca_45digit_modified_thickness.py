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

from theory_of_wing_sections import thickness_data

from pyPC.airfoil.thickness import (Naca45DigitModifiedThickness,
                                    Naca45DigitModifiedThicknessEnhanced)


def _calculate_leading_edge_radius(th: Naca45DigitModifiedThickness) -> float:
    return 0.5*((th.max_thickness_index/100.0)*th.a[0])**2


class TestNaca45DigitModifiedThickness(unittest.TestCase):
    """Class to test the NACA modified 4-digit thickness geometry."""

    def testSetters(self) -> None:
        """Test the setting of thickness parameters."""
        th_open = Naca45DigitModifiedThickness(mti=20, lei=4, lmti=5)

        self.assertAlmostEqual(th_open.max_thickness_index, 20, delta=1e-7)
        self.assertAlmostEqual(th_open.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(th_open.loc_max_thickness_index, 5, delta=1e-7)

        th_closed = Naca45DigitModifiedThicknessEnhanced(mti=20, lei=4,
                                                         lmti=5,
                                                         closed_te=True)

        self.assertAlmostEqual(th_closed.max_thickness_index, 20, delta=1e-7)
        self.assertAlmostEqual(th_closed.leading_edge_index, 4, delta=1e-7)
        self.assertAlmostEqual(th_closed.loc_max_thickness_index, 5,
                               delta=1e-7)
        self.assertTrue(th_closed.closed_te)

        # test initializing enhanced parameters
        # sharp trailing edge
        I_ref = 3
        M_ref = 4
        d_ref = [0, 1.575, -1.0833333333, -0.2546296297]
        a_ref = [0.74225, 0.9328327771, -2.6241657396, 1.2077076380]
        th_closed.max_thickness_index = 12
        th_closed.loc_max_thickness_index = M_ref
        th_closed.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(th_closed.a, a_ref))
        self.assertIsNone(npt.assert_allclose(th_closed.d, d_ref))

        # test initializing classic parameters
        I_ref = 3
        M_ref = 4
        d_ref = [0.01, 1.575, -1.1666666667, -0.1620370371]
        a_ref = [0.74225, 0.9661661252, -2.7908324797, 1.4160410631]
        th_open.max_thickness_index = 12
        th_open.loc_max_thickness_index = M_ref
        th_open.leading_edge_index = I_ref
        self.assertIsNone(npt.assert_allclose(th_open.a, a_ref))
        self.assertIsNone(npt.assert_allclose(th_open.d, d_ref))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        # pylint: disable=too-many-statements
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0008-34
        th = Naca45DigitModifiedThickness(mti=8, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=3e-5))

        # NACA 0010-34
        th = Naca45DigitModifiedThickness(mti=10, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

        # NACA 0010-35
        th = Naca45DigitModifiedThickness(mti=10, lei=3, lmti=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

        # NACA 0010-64
        th = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0010-65
        th = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=5)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0010-66
        th = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0012-34
        th = Naca45DigitModifiedThickness(mti=12, lei=3, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=6e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0012-64
        th = Naca45DigitModifiedThickness(mti=12, lei=6, lmti=4)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    f"NACA00{int(th.max_thickness_index):02d}"
                    f"-{int(th.leading_edge_index)}"
                    f"{int(th.loc_max_thickness_index)}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=8e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        th = Naca45DigitModifiedThicknessEnhanced(mti=20, lei=4.3, lmti=5.6,
                                                  closed_te=False)

        # test the settings with open trailing edge
        xi_max = th.loc_max_thickness_index/10.0
        t_max = np.sqrt(xi_max)
        xi_te = 1.0
        t_te = np.sqrt(xi_te)
        delta_max_ref = 0.005*th.max_thickness_index
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, 0.02*0.10))

        # test the settings with close trailing edge
        th.closed_te = True
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, 0))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        th_open = Naca45DigitModifiedThickness(mti=10, lei=6, lmti=4)
        th_closed = Naca45DigitModifiedThicknessEnhanced(mti=15.2, lei=4.3,
                                                         lmti=5.6,
                                                         closed_te=True)

        def compare_values(t: np_type.NDArray,
                           th: Naca45DigitModifiedThickness) -> None:
            eps = 1e-7

            t = np.sqrt(np.asarray(t))
            it = np.nditer([t, None])
            with it:
                for tr, deltar in it:
                    tmax = th.max_thickness_index/100.0
                    if tr**2 <= th.loc_max_thickness_index/10.0:
                        deltar[...] = tmax*(th.a[0]*tr + th.a[1]*tr**2
                                            + th.a[2]*tr**4 + th.a[3]*tr**6)
                    else:
                        deltar[...] = tmax*(th.d[0] + th.d[1]*(1-tr**2)
                                            + th.d[2]*(1-tr**2)**2
                                            + th.d[3]*(1-tr**2)**3)
                delta_ref = it.operands[1]

            # compare point values
            delta = th.delta(t)
            self.assertIsNone(npt.assert_allclose(delta, delta_ref))

            # compare first derivatives
            deltat_ref = 0.5*(th.delta(t+eps)-th.delta(t-eps))/eps
            deltat = th.delta_t(t)
            self.assertIsNone(npt.assert_allclose(deltat, deltat_ref))

            # compare second derivatives
            deltatt_ref = 0.5*(th.delta_t(t+eps)-th.delta_t(t-eps))/eps
            deltatt = th.delta_tt(t)
            self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref))

        # test point on front
        t = np.sqrt(0.25)
        compare_values(t, th_closed)
        compare_values(t, th_open)

        # test point on back
        t = np.sqrt(0.60)
        compare_values(t, th_closed)
        compare_values(t, th_open)

        # test points on fore and aft
        t = np.linspace(0, 1, 12)
        compare_values(t, th_closed)
        compare_values(t, th_open)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        th = Naca45DigitModifiedThickness(mti=12, lei=4, lmti=6)

        # reference values
        delta_ref = [0, 0.0012]
        deltat_ref = [0.11876, -0.8400]
        deltatt_ref = [-0.0379443778, -8.82]

        # test leading edge
        t = 0
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref[0]))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref[0]))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref[0]))

        # test trailing edge
        t = 1
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref[1]))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref[1]))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref[1]))

        # test both
        t = np.array([0, 1])
        delta = th.delta(t)
        deltat = th.delta_t(t)
        deltatt = th.delta_tt(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref))
        self.assertIsNone(npt.assert_allclose(deltat, deltat_ref))
        self.assertIsNone(npt.assert_allclose(deltatt, deltatt_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        th = Naca45DigitModifiedThickness(mti=24, lei=3, lmti=5)

        self.assertListEqual([np.sqrt(0.5)], th.discontinuities())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        th = Naca45DigitModifiedThickness(mti=24, lei=3, lmti=5)

        self.assertAlmostEqual(np.sqrt(0.5), th.max_thickness_parameter())


if __name__ == "__main__":
    unittest.main(verbosity=1)
