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

from pyPC.airfoil import Naca4DigitThicknessBase
from pyPC.airfoil import Naca4DigitThicknessClassic
from pyPC.airfoil import Naca4DigitThicknessEnhanced
from pyPC.airfoil import Naca4DigitAirfoilClassic

from theory_of_wing_sections import thickness_data, airfoil_data


class TestNaca4Digit(unittest.TestCase):
    """Class to test the NACA 4-digit airfoil geometry."""

    def testClassicThickness(self) -> None:
        """Test the classic thickness coordinates to published data."""
        directory = dirname(abspath(__file__))
        tows = thickness_data(filename=None)

        # NACA 0006
        af = Naca4DigitThicknessClassic(thickness=0.06)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=5e-5))

        # NACA 0008
        af = Naca4DigitThicknessClassic(thickness=0.08)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=6e-5))

        # NACA 0009
        af = Naca4DigitThicknessClassic(thickness=0.09)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=3e-5))

        # NACA 0010
        af = Naca4DigitThicknessClassic(thickness=0.10)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1.2e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=2e-5))

        # NACA 0012
        af = Naca4DigitThicknessClassic(thickness=0.12)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=7e-5))

        # NACA 0015
        af = Naca4DigitThicknessClassic(thickness=0.15)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0018
        af = Naca4DigitThicknessClassic(thickness=0.18)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0021
        af = Naca4DigitThicknessClassic(thickness=0.21)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=1e-5))

        # NACA 0024
        af = Naca4DigitThicknessClassic(thickness=0.24)
        filename = (directory + "/data/Theory of Wing Sections/Thickness/"
                    + f"NACA00{int(af.thickness*100):02d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        le_radius = -1/af.le_k()
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(le_radius, tows.le_radius,
                                              rtol=0, atol=4e-5))

    def testEnhancedThickness(self) -> None:
        """Test the enhanced thickness coefficient calculation."""
        af = Naca4DigitThicknessEnhanced(thickness=0.20, closed_te=True,
                                         le_radius=False)
        xi_max = 0.3
        xi_1c = 0.1
        xi_te = 1.0

        # test open trailing edge, original leading edge shape
        af.reset(closed_te=False, le_radius=False)
        y_max_ref = 0.1
        y_te_ref = 0.002
        y_pte_ref = -0.234
        y_1c_ref = 0.078
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        y_1c = af.y(xi_1c)
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test open trailing edge, leading edge radius
        af.reset(closed_te=False, le_radius=True)
        y_max_ref = 0.1
        y_te_ref = 0.002
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af.a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

        # test closed trailing edge, original leading edge shape
        af.reset(closed_te=True, le_radius=False)
        y_max_ref = 0.1
        y_te_ref = 0.0
        y_pte_ref = -0.234
        y_1c_ref = 0.078
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        y_1c = af.y(xi_1c)
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(y_1c, y_1c_ref))

        # test closed trailing edge, leading edge radius
        af.reset(closed_te=True, le_radius=True)
        y_max_ref = 0.1
        y_te_ref = 0.0
        y_pte_ref = -0.234
        r_le_ref = 0.5*0.29690**2
        y_max = af.y(xi_max)
        y_pmax = af.y_p(xi_max)
        y_te = af.y(xi_te)
        y_pte = af.y_p(xi_te)
        r_le = 0.5*af.a[0]**2
        self.assertIsNone(npt.assert_allclose(y_max, y_max_ref))
        self.assertIsNone(npt.assert_allclose(y_pmax, 0, atol=1e-7))
        self.assertIsNone(npt.assert_allclose(y_te, y_te_ref))
        self.assertIsNone(npt.assert_allclose(y_pte, y_pte_ref))
        self.assertIsNone(npt.assert_allclose(r_le, r_le_ref))

    def testThickness(self) -> None:
        """Test the thickness relations."""
        af_open = Naca4DigitThicknessClassic(thickness=0.3)
        af_closed = Naca4DigitThicknessEnhanced(thickness=0.3, closed_te=True,
                                                le_radius=False)

        def compare_values(xi: np_type.NDArray,
                           af: Naca4DigitThicknessBase) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    yr[...] = (af.thickness/0.20)*(af.a[0]*np.sqrt(xir)
                                                   + af.a[1]*xir
                                                   + af.a[2]*xir**2
                                                   + af.a[3]*xir**3
                                                   + af.a[4]*xir**4)

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

        # test point on front
        xi = 0.25
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test point on back
        xi = 0.6
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

        # test points on lower and upper surface (avoid leading edge because
        # the derivatives are infinite)
        xi = np.linspace(0.001, 1, 12)
        compare_values(xi, af_closed)
        compare_values(xi, af_open)

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
