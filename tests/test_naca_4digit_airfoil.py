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

from pyPC.camber import Naca4DigitCamber

from pyPC.airfoil import Naca4DigitThicknessClassic
from pyPC.airfoil import Naca4DigitAirfoilClassic

from theory_of_wing_sections import thickness_data, airfoil_data


class TestNaca4Digit(unittest.TestCase):
    """Class to test the NACA 4-digit airfoil geometry."""

    def testAirfoilSetters(self) -> None:
        """Test setters for airfoil."""
        af_c = Naca4DigitAirfoilClassic(max_camber=1, max_camber_location=4,
                                        max_thickness=8, scale=1.8)
        # af_e = Naca4DigitEnhanced(max_camber=1.2, max_camber_location=4.3,
        #                           max_thickness=12.6, scale=2.1)

        self.assertIs(af_c.max_camber, 1)
        self.assertIs(int(100*af_c._yc.m), af_c.max_camber)
        self.assertIs(af_c.max_camber_location, 4)
        self.assertIs(int(10*af_c._yc.p), af_c.max_camber_location)
        self.assertIs(af_c.max_thickness, 8)
        self.assertIs(int(100*af_c._delta_t.thickness), af_c.max_thickness)
        self.assertIsNone(npt.assert_allclose(af_c.scale, 1.8))

    def testAirfoilCamber(self) -> None:
        """Test camber calculation for airfoil."""
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=2,
                                      max_thickness=12, scale=1.4)

        def camber_calc(xi: np_type.NDArray) -> np_type.NDArray:
            cam = Naca4DigitCamber(m=af.m/100.0, p=af.p/10.0)
            y = np.zeros_like(xi)
            it = np.nditer([xi, y], op_flags=[["readonly"], ["writeonly"]])
            with it:
                for xir, yr in it:
                    if (xir < 0):
                        yr[...] = cam.y(-xir)
                    else:
                        yr[...] = cam.y(xir)
            return y

        rg = default_rng(42)
        xi = 2*rg.random((20,))-1
        yc_ref = af.scale*camber_calc(xi)
        yc = af.camber(xi)
        self.assertIsNone(npt.assert_allclose(yc, yc_ref))

    def testAirfoilThickness(self) -> None:
        """Test thickness calculation for airfoil."""
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=2,
                                      max_thickness=12, scale=1.4)

        def thickness_calc(xi: np_type.NDArray) -> np_type.NDArray:
            thick = Naca4DigitThicknessClassic(thickness=af.t/100.0)
            y = np.zeros_like(xi)
            it = np.nditer([xi, y], op_flags=[["readonly"], ["writeonly"]])
            with it:
                for xir, yr in it:
                    if (xir < 0):
                        yr[...] = thick.y(-xir)
                    else:
                        yr[...] = thick.y(xir)
            return y

        rg = default_rng(42)
        xi = 2*rg.random((20,))-1
        yt_ref = af.scale*thickness_calc(xi)
        yt = af.thickness(xi)
        self.assertIsNone(npt.assert_allclose(yt, yt_ref))

    def testClassicSymmetricAirfoil(self) -> None:
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=6, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0008
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=8, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0009
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=9, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0010
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=10, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0,
                                              atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0,
                                              atol=1.2e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0012
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=12, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0015
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=15, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0018
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=18, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0021
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=21, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

        # NACA 0024
        af = Naca4DigitAirfoilClassic(max_camber=0, max_camber_location=0,
                                      max_thickness=24, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        # le_radius = -1/af.le_k()
        yu = af.y(x=tows.x, upper=True)
        yl = af.y(x=tows.x, upper=False)
        self.assertIsNone(npt.assert_allclose(yu, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yl, -tows.y, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
        #                                       rtol=0, atol=1e-5))

    def testClassicCamberedAirfoil(self) -> None:
        """Test classic airfoil against published data."""
        directory = dirname(abspath(__file__))
        tows = airfoil_data(filename=None)

        # NACA 1408
        af = Naca4DigitAirfoilClassic(max_camber=1, max_camber_location=4,
                                      max_thickness=8, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=1e-7, upper=True)  # FIX: should be at x=0
        yp0l = af.dydx(x=1e-7, upper=False)
        le_radius = -1/af.k(xi=1e-7)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=1.5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 1410
        af = Naca4DigitAirfoilClassic(max_camber=1, max_camber_location=4,
                                      max_thickness=10, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=2e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 1412
        af = Naca4DigitAirfoilClassic(max_camber=1, max_camber_location=4,
                                      max_thickness=12, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=1.5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2408
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=8, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=1.5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=1.5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2410
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=10, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=1.5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2412
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=12, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2415
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=15, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2418
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=18, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2421
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=21, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 2424
        af = Naca4DigitAirfoilClassic(max_camber=2, max_camber_location=4,
                                      max_thickness=24, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=2e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 4412
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=4,
                                      max_thickness=12, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 4415
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=4,
                                      max_thickness=15, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 4418
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=4,
                                      max_thickness=18, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 4421
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=4,
                                      max_thickness=21, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

        # NACA 4424
        af = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=4,
                                      max_thickness=24, scale=1.0)
        filename = (directory + "/data/Theory of Wing Sections/Airfoil/"
                    + f"NACA{af.max_camber:1d}{af.max_camber_location:1d}"
                    + f"{af.max_thickness:02d}.dat")
        tows.change_case_data(filename=filename)
        yu = af.y(x=tows.x_upper, upper=True)
        yl = af.y(x=tows.x_lower, upper=False)
        yp0u = af.dydx(x=0.0, upper=True)
        yp0l = af.dydx(x=0.0, upper=False)
        le_radius = af.k(xi=0.0)
        self.assertIsNone(npt.assert_allclose(yu, tows.y_upper, rtol=0, atol=2.5e-5))
        self.assertIsNone(npt.assert_allclose(yl, tows.y_lower, rtol=0, atol=2.5e-5))
        # self.assertIsNone(npt.assert_allclose(yp0u, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(yp0l, tows.le_slope, rtol=0, atol=1e-5))
        # self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius, rtol=0, atol=1e-5))

    def testAirfoilParametricDerivatives(self) -> None:
        """Test camber calculation for airfoil."""
        af_c = Naca4DigitAirfoilClassic(max_camber=4, max_camber_location=2,
                                        max_thickness=12, scale=1.4)

        def compare_values(xi: np_type.NDArray,
                           af: Naca4DigitAirfoilClassic) -> None:
            eps = 1e-7

            # compare first derivatives
            xpl, ypl = af.xy_from_xi(xi+eps)
            xmi, ymi = af.xy_from_xi(xi-eps)
            xp_ref = 0.5*(xpl-xmi)/eps
            yp_ref = 0.5*(ypl-ymi)/eps
            xp, yp = af.xy_p(xi)
            self.assertIsNone(npt.assert_allclose(xp, xp_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(yp, yp_ref, atol=1e-7))

            # compare second derivatives
            xtp, ytp = af.xy_p(xi+eps)
            xtm, ytm = af.xy_p(xi-eps)
            xpp_ref = 0.5*(xtp-xtm)/eps
            ypp_ref = 0.5*(ytp-ytm)/eps
            xpp, ypp = af.xy_pp(xi)
            self.assertIsNone(npt.assert_allclose(xpp, xpp_ref, atol=1e-7))
            self.assertIsNone(npt.assert_allclose(ypp, ypp_ref, atol=1e-7))

        rg = default_rng(42)
        xi = 2*rg.random((20,))-1
        compare_values(xi, af_c)


if __name__ == "__main__":
    unittest.main(verbosity=1)
