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
from numpy.random import default_rng

from pyPC.airfoil.camber import Naca4DigitCamber
from pyPC.airfoil.thickness import Naca45DigitThickness
from pyPC.airfoil.airfoil import OrthogonalAirfoil

from theory_of_wing_sections import thickness_data, airfoil_data


def create_naca_4digit(max_camber_index: float, loc_max_camber_index: float,
                       max_thickness_index: float) -> OrthogonalAirfoil:
    camber = Naca4DigitCamber(mci=max_camber_index, lci=loc_max_camber_index)
    thickness = Naca45DigitThickness(mti=max_thickness_index)
    return OrthogonalAirfoil(camber, thickness)


class TestNaca4Digit(unittest.TestCase):
    """Class to test the NACA 4-digit airfoil geometry."""

    def testAirfoilSetters(self) -> None:
        """Test setters for airfoil."""
        af = create_naca_4digit(max_camber_index=1, loc_max_camber_index=4,
                                max_thickness_index=8)
        self.assertEqual(af.camber.max_camber_index, 1)
        self.assertEqual(af.camber.loc_max_camber_index, 4)
        self.assertEqual(af.thickness.max_thickness_index, 8)

        # What about scaling and translating?

        af.camber.max_camber_index = 3.125
        af.camber.loc_max_camber_index = 2.5
        af.thickness.max_thickness_index = 6.25
        self.assertEqual(af.camber.max_camber_index, 3.125)
        self.assertEqual(af.camber.loc_max_camber_index, 2.5)
        self.assertEqual(af.thickness.max_thickness_index, 6.25)

        af.camber.max_camber_index = 0
        af.camber.loc_max_camber_index = 0
        self.assertEqual(af.camber.max_camber_index, 0)
        self.assertEqual(af.camber.loc_max_camber_index, 0)
        self.assertEqual(af.thickness.max_thickness_index, 6.25)

    def testCamberTransformation(self) -> None:
        """Test the camber coordinates."""
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                                max_thickness_index=18)

        t = np.linspace(0, 1, 12)
        x, y = af.camber_location(t)
        x_ref, y_ref = af.camber.xy(t**2)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))

    def testThicknessTransformation(self) -> None:
        """Test the thickness values."""
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                                max_thickness_index=18)

        t = np.linspace(0, 1, 12)
        delta = af.thickness_value(t)
        delta_ref = af.thickness.delta(t)
        self.assertIsNone(npt.assert_allclose(delta, delta_ref))

    def testClassicSymmetricAirfoil(self) -> None:
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=6)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              atol=5e-5))

        # NACA 0008
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=8)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=6e-5)

        # NACA 0009
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=9)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=3e-5)

        # NACA 0010
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=10)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1.2e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-5)

        # NACA 0012
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=12)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=7e-5)

        # NACA 0015
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=15)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=1e-5)

        # NACA 0018
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=18)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=1e-5)

        # NACA 0021
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=21)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=1e-5)

        # NACA 0024
        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=24)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x, upper=True)
        t_l = af.t_from_x(x=tows.x, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x))
        self.assertIsNone(npt.assert_allclose(yu, tows.y, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, atol=1e-5))
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=4e-5)

    def testClassicCamberedAirfoil(self) -> None:
        """Test classic airfoil against published data."""
        directory = dirname(abspath(__file__))
        tows = airfoil_data(filename=None)

        # NACA 1408
        af = create_naca_4digit(max_camber_index=1, loc_max_camber_index=4,
                                max_thickness_index=8)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=1.5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=5e-5)

        # NACA 1410
        af = create_naca_4digit(max_camber_index=1, loc_max_camber_index=4,
                                max_thickness_index=10)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=2e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=1e-5)

        # NACA 1412
        af = create_naca_4digit(max_camber_index=1, loc_max_camber_index=4,
                                max_thickness_index=12)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=1.5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=5e-5)

        # NACA 2408
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=8)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=1.5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-5)

        # NACA 2410
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=10)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=1.5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=4e-5)

        # NACA 2412
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=12)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=1.5e-5)

        # NACA 2415
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=15)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-4)

        # NACA 2418
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=18)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=8e-5)

        # NACA 2421
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=21)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-4)

        # NACA 2424
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=4,
                                max_thickness_index=24)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=2e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-4)

        # NACA 4412
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=4,
                                max_thickness_index=12)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=7e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=3e-4)

        # NACA 4415
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=4,
                                max_thickness_index=15)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=5e-4)

        # NACA 4418
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=4,
                                max_thickness_index=18)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=6e-4)

        # NACA 4421
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=4,
                                max_thickness_index=21)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=5e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=9e-4)

        # NACA 4424
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=4,
                                max_thickness_index=24)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{int(af.camber.max_camber_index):1d}"
                    + f"{int(af.camber.loc_max_camber_index):1d}"
                    + f"{int(af.thickness.max_thickness_index):02d}.dat")
        tows.change_case_data(filename=filename)
        t_u = af.t_from_x(x=tows.x_upper, upper=True)
        t_l = af.t_from_x(x=tows.x_lower, upper=False)
        xu, yu = af.xy(t_u)
        xl, yl = af.xy(t_l)
        le_rad_slope_u = -1/af.dydx(0)
        le_rad_slope_l = -1/af.dydx(0)
        le_radius = -1/af.k(0)
        self.assertIsNone(npt.assert_allclose(xu, tows.x_upper))
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, atol=3e-5))
        self.assertIsNone(npt.assert_allclose(xl, tows.x_lower))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, atol=3e-5))
        self.assertAlmostEqual(le_rad_slope_u, tows.le_radius_slope)
        self.assertAlmostEqual(le_rad_slope_l, tows.le_radius_slope)
        self.assertAlmostEqual(le_radius, tows.le_radius, delta=2e-3)

    def testAirfoilParametricDerivatives(self) -> None:
        """Test calculations of parameteric derivatives."""
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                                max_thickness_index=12)

        def compare_values(t: np_type.NDArray, af: OrthogonalAirfoil) -> None:
            eps = 1e-7

            # compare first derivatives
            xpl, ypl = af.xy(t+eps)
            xmi, ymi = af.xy(t-eps)
            xt_ref = 0.5*(xpl-xmi)/eps
            yt_ref = 0.5*(ypl-ymi)/eps
            xt, yt = af.xy_t(t)
            self.assertIsNone(npt.assert_allclose(xt, xt_ref))
            self.assertIsNone(npt.assert_allclose(yt, yt_ref))

            # compare second derivatives
            xtp, ytp = af.xy_t(t+eps)
            xtm, ytm = af.xy_t(t-eps)
            xtt_ref = 0.5*(xtp-xtm)/eps
            ytt_ref = 0.5*(ytp-ytm)/eps
            xtt, ytt = af.xy_tt(t)
            self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
            self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

        rg = default_rng(42)
        t = 2*rg.random((20,))-1
        compare_values(t, af)


    # def testAirfoilArclengthDerivatives(self) -> None:
    #     """Test calculations of arc-length derivatives."""
    #     # TODO: Implement
    #     af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
    #                             max_thickness_index=14)

    #     # test calculation of entire length
    #     xi_min = 0.001
    #     xi_max = 1
    #     s_max = af.arc_length(xi_min, xi_max)
    #     s_max_ref = 2
    #     self.assertAlmostEqual(s_max, s_max_ref, 1e-7)

    #     def compare_values(xi: np_type.NDArray, af: OrthogonalAirfoil) -> None:
    #         if np.abs(xi) < 1e-7:
    #             sx_ref = -af.camber.y_p(xi)
    #             sy_ref = 1.0
    #         else:
    #             sx_ref, sy_ref = af.xy_p(xi)
    #         tmp = np.sqrt(sx_ref**2 + sy_ref**2)
    #         sx_ref /= tmp
    #         sy_ref /= tmp
    #         nx_ref = -sy_ref
    #         ny_ref = sx_ref

    #         sx, sy = af.tangent(xi)
    #         nx, ny = af.normal(xi)
    #         self.assertIsNone(npt.assert_allclose(sx, sx_ref, atol=1e-7))
    #         self.assertIsNone(npt.assert_allclose(sy, sy_ref, atol=1e-7))
    #         self.assertIsNone(npt.assert_allclose(nx, nx_ref, atol=1e-7))
    #         self.assertIsNone(npt.assert_allclose(ny, ny_ref, atol=1e-7))

    def testAirfoilNormalTangentVectors(self) -> None:
        """Test calculations of unit normal and tangent vectors."""
        af = create_naca_4digit(max_camber_index=2, loc_max_camber_index=2,
                                max_thickness_index=12)

        def compare_values(t: np_type.NDArray, af: OrthogonalAirfoil) -> None:
            sx_ref, sy_ref = af.xy_t(t)
            tmp = np.sqrt(sx_ref**2 + sy_ref**2)
            sx_ref /= tmp
            sy_ref /= tmp
            nx_ref = -sy_ref
            ny_ref = sx_ref

            sx, sy = af.tangent(t)
            nx, ny = af.normal(t)
            self.assertIsNone(npt.assert_allclose(sx, sx_ref))
            self.assertIsNone(npt.assert_allclose(sy, sy_ref))
            self.assertIsNone(npt.assert_allclose(nx, nx_ref))
            self.assertIsNone(npt.assert_allclose(ny, ny_ref))

        t = -1
        compare_values(t, af)

        t = -0.2
        compare_values(t, af)

        t = 0
        compare_values(t, af)

        t = 0.4
        compare_values(t, af)

        t = 1
        compare_values(t, af)

    def testEndpoints(self) -> None:
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                                max_thickness_index=21)

        # reference values
        t_ref = np.array([-1.0, 1.0])
        x_ref, y_ref = af.xy(t_ref)
        xt_ref, yt_ref = af.xy_t(t_ref)
        xtt_ref, ytt_ref = af.xy_tt(t_ref)

        # test lower trailing edge
        t = -1
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[0]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[0]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[0]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[0]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[0]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[0]))

        # test upper trailing edge
        t = 1
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref[1]))
        self.assertIsNone(npt.assert_allclose(y, y_ref[1]))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref[1]))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref[1]))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref[1]))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref[1]))

        # test both
        t = np.array([-1, 1])
        x, y = af.xy(t)
        xt, yt = af.xy_t(t)
        xtt, ytt = af.xy_tt(t)
        self.assertIsNone(npt.assert_allclose(x, x_ref))
        self.assertIsNone(npt.assert_allclose(y, y_ref))
        self.assertIsNone(npt.assert_allclose(xt, xt_ref))
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))
        self.assertIsNone(npt.assert_allclose(xtt, xtt_ref))
        self.assertIsNone(npt.assert_allclose(ytt, ytt_ref))

    def testJoints(self) -> None:
        """Test correct joints are being reported."""
        af = create_naca_4digit(max_camber_index=4, loc_max_camber_index=2,
                                max_thickness_index=21)

        self.assertListEqual([-1.0, -np.sqrt(0.2), 0.0, np.sqrt(0.2), 1.0],
                             af.joints())

        af = create_naca_4digit(max_camber_index=0, loc_max_camber_index=0,
                                max_thickness_index=21)

        self.assertListEqual([-1.0, 0.0, 1.0], af.joints())


if __name__ == "__main__":
    unittest.main(verbosity=1)
