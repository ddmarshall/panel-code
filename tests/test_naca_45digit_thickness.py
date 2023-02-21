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

from theory_of_wing_sections import thickness_data

from pyPC.airfoil.thickness import (Naca45DigitThickness,
                                    Naca45DigitThicknessEnhanced)


def _calculate_leading_edge_radius(th: Naca45DigitThickness) -> float:
    return 0.5*((th.max_thickness_index/100.0)*th.a[0])**2


class TestNaca45DigitThickness(unittest.TestCase):
    """Class to test the NACA 4-digit thickness geometry."""

    def testSetters(self) -> None:
        """Test the setting of thickness parameters."""
        th = Naca45DigitThickness(mti=14)

        self.assertAlmostEqual(th.max_thickness_index, 14, delta=1e-7)

        th = Naca45DigitThicknessEnhanced(mti=18.5, closed_te=True,
                                          use_radius=True)
        self.assertAlmostEqual(th.max_thickness_index, 18.5, delta=1e-7)
        self.assertTrue(th.closed_trailing_edge)
        self.assertTrue(th.use_leading_edge_radius)

        # Note: The published values from Jacobs, Ward, and Pinkerton (1933)
        #       are only accurate to 3 sig. figs. These reference values come
        #       from previous Matlab implementation.
        # a from ref.    [0.29690,  -0.12600,  -0.35160,  0.28430,
        #                 -0.10150]/0.20
        a_ref = np.array([0.296676, -0.125834, -0.350607, 0.282350,
                          -0.100585])/0.20

        th.closed_trailing_edge = False
        th.use_leading_edge_radius = False
        self.assertIsNone(npt.assert_allclose(th.a, a_ref, rtol=1e-6))

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        # pylint: disable=too-many-statements
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        th = Naca45DigitThickness(mti=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=5e-5))

        # NACA 0008
        th = Naca45DigitThickness(mti=8)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=6e-5))

        # NACA 0009
        th = Naca45DigitThickness(mti=9)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=3e-5))

        # NACA 0010
        th = Naca45DigitThickness(mti=10)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=2e-5))

        # NACA 0012
        th = Naca45DigitThickness(mti=12)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=7e-5))

        # NACA 0015
        th = Naca45DigitThickness(mti=15)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0018
        th = Naca45DigitThickness(mti=18)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0021
        th = Naca45DigitThickness(mti=21)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=1e-5))

        # NACA 0024
        th = Naca45DigitThickness(mti=24)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(th.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t = np.sqrt(tows.x)
        delta = th.delta(t)
        le_radius = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=4e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        # pylint: disable=too-many-locals
        # pylint: disable=too-many-statements
        th = Naca45DigitThicknessEnhanced(mti=20, closed_te=True,
                                          use_radius=False)
        xi_max = 0.3
        t_max = np.sqrt(xi_max)
        xi_1c = 0.1
        t_1c = np.sqrt(xi_1c)
        xi_te = 1.0
        t_te = np.sqrt(xi_te)

        # test open trailing edge, original leading edge shape
        th.closed_trailing_edge = False
        th.use_leading_edge_radius = False
        delta_max_ref = 0.1
        delta_te_ref = 0.002
        deltat_te_ref = -0.468
        delta_1c_ref = 0.078
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        deltat_te = th.delta_t(t_te)
        delta_1c = th.delta(t_1c)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, delta_te_ref))
        self.assertIsNone(npt.assert_allclose(deltat_te, deltat_te_ref))
        self.assertIsNone(npt.assert_allclose(delta_1c, delta_1c_ref))

        # test open trailing edge, leading edge radius
        th.closed_trailing_edge = False
        th.use_leading_edge_radius = True
        delta_max_ref = 0.1
        delta_te_ref = 0.002
        deltat_te_ref = -0.468
        r_le_ref = 0.5*0.29690**2
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        deltat_te = th.delta_t(t_te)
        r_le = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, delta_te_ref))
        self.assertIsNone(npt.assert_allclose(deltat_te, deltat_te_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

        # test closed trailing edge, original leading edge shape
        th.closed_trailing_edge = True
        th.use_leading_edge_radius = False
        delta_max_ref = 0.1
        delta_te_ref = 0.0
        deltat_te_ref = -0.468
        delta_1c_ref = 0.078
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        deltat_te = th.delta_t(t_te)
        delta_1c = th.delta(t_1c)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, delta_te_ref))
        self.assertIsNone(npt.assert_allclose(deltat_te, deltat_te_ref))
        self.assertIsNone(npt.assert_allclose(delta_1c, delta_1c_ref))

        # test closed trailing edge, leading edge radius
        th.closed_trailing_edge = True
        th.use_leading_edge_radius = True
        delta_max_ref = 0.1
        delta_te_ref = 0.0
        deltat_te_ref = -0.468
        r_le_ref = 0.5*0.29690**2
        delta_max = th.delta(t_max)
        deltat_max = th.delta_t(t_max)
        delta_te = th.delta(t_te)
        deltat_te = th.delta_t(t_te)
        r_le = _calculate_leading_edge_radius(th)
        self.assertIsNone(npt.assert_allclose(delta_max, delta_max_ref))
        self.assertIsNone(npt.assert_allclose(deltat_max, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(delta_te, delta_te_ref))
        self.assertIsNone(npt.assert_allclose(deltat_te, deltat_te_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        th_open = Naca45DigitThickness(mti=30)
        th_closed = Naca45DigitThicknessEnhanced(mti=24.2, closed_te=True,
                                                 use_radius=False)

        def compare_values(t: np_type.NDArray,
                           th: Naca45DigitThickness) -> None:
            eps = 1e-7

            t = np.sqrt(np.asarray(t))
            it = np.nditer([t, None])
            with it:
                for tr, deltar in it:
                    tmax = th.max_thickness_index/100.0
                    deltar[...] = tmax*(th.a[0]*tr + th.a[1]*tr**2
                                        + th.a[2]*tr**4 + th.a[3]*tr**6
                                        + th.a[4]*tr**8)
                delta_ref = it.operands[1]

            # compare point values
            delta = th.delta(t)
            self.assertIsNone(npt.assert_allclose(delta, delta_ref, atol=1e-7))

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
        t = np.sqrt(0.6)
        compare_values(t, th_closed)
        compare_values(t, th_open)

        # test points on fore and aft
        t = np.linspace(0, 1, 12)
        compare_values(t, th_closed)
        compare_values(t, th_open)

    def testEndPoints(self) -> None:
        """Test accessing the end points of thickness with integers."""
        th = Naca45DigitThickness(mti=12)

        # reference values
        delta_ref = [0, 0.00126]
        deltat_ref = [0.17814, -0.28062]
        deltatt_ref = [-0.1512, -0.97572]

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
        th = Naca45DigitThickness(mti=24)

        self.assertListEqual([], th.discontinuities())

    def testMaxThickness(self) -> None:
        """Test maximum thickness."""
        th = Naca45DigitThickness(mti=24)

        self.assertAlmostEqual(np.sqrt(0.3), th.max_thickness_parameter())


if __name__ == "__main__":
    unittest.main(verbosity=1)
