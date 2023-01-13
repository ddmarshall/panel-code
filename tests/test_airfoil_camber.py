#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:11:28 2023

@author: ddmarshall
"""

import unittest

from os.path import abspath, dirname

import numpy as np
import numpy.typing as np_type
import numpy.testing as npt

from pyPC.camber import Naca4DigitCamber
from pyPC.camber import Naca5DigitCamberBase
from pyPC.camber import Naca5DigitCamberClassic
from pyPC.camber import Naca5DigitCamberEnhanced
from pyPC.camber import Naca5DigitCamberReflexedBase
from pyPC.camber import Naca5DigitCamberReflexedClassic
from pyPC.camber import Naca5DigitCamberReflexedEnhanced

from theory_of_wing_sections import camber_data


class TestNaca4DigitCamber(unittest.TestCase):
    """Class to test the NACA 4-digit camber geometry."""

    def testClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 62xx
        af = Naca4DigitCamber(m=0.06, p=0.2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 63xx
        af = Naca4DigitCamber(m=0.06, p=0.3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 64xx
        af = Naca4DigitCamber(m=0.06, p=0.3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 65xx
        af = Naca4DigitCamber(m=0.06, p=0.3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 66xx
        af = Naca4DigitCamber(m=0.06, p=0.3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 67xx
        af = Naca4DigitCamber(m=0.06, p=0.3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA{int(af.m*100):1d}{int(af.p*10):1d}.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af = Naca4DigitCamber(m=0.03, p=0.4)

        def compare_values(xi: np_type.NDArray, af: Naca4DigitCamber) -> None:
            eps = 1e-7

            xi_a = np.asarray(xi)
            y_ref = np.zeros_like(xi_a)
            it = np.nditer([xi_a, y_ref], op_flags=[["readonly"],
                                                    ["writeonly"]])
            with it:
                for xir, yr in it:
                    if xir <= af.p:
                        yr[...] = (af.m/af.p**2)*(2*af.p*xir - xir**2)
                    else:
                        yr[...] = (af.m/(1-af.p)**2)*(1-2*af.p + 2*af.p*xir
                                                      - xir**2)

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
        af = Naca4DigitCamber(m=0.04, p=0.2)

        # reference values
        coef = [af.m/af.p**2, af.m/(1-af.p)**2]
        y_ref = [0, 0]
        yp_ref = [2*coef[0]*af.p, 2*coef[1]*(af.p - 1)]
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
        af = Naca4DigitCamber(m=0.03, p=0.4)

        self.assertListEqual([0.4], af.joints())


class TestNaca5DigitCamber(unittest.TestCase):
    """Class to test the NACA 5-digit camber geometry."""

    def testCamberClassic(self) -> None:
        """Test the camber coordinates and slope against published data."""
        directory = dirname(abspath(__file__))
        tows = camber_data(filename=None)

        # NACA 210xx
        af = Naca5DigitCamberClassic(camber_loc=1)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=3e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=2e-5))

        # NACA 220xx
        af = Naca5DigitCamberClassic(camber_loc=2)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=2e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 230xx
        af = Naca5DigitCamberClassic(camber_loc=3)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 240xx
        af = Naca5DigitCamberClassic(camber_loc=4)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=1e-5))

        # NACA 250xx
        af = Naca5DigitCamberClassic(camber_loc=5)
        filename = (directory + "/data/Theory of Wing Sections/Camber/"
                    + f"NACA2{af.camber_location:1d}0.dat")
        tows.change_case_data(filename=filename)
        y = af.y(tows.x)
        yp = af.y_p(tows.x)
        self.assertIsNone(npt.assert_allclose(y, tows.y, rtol=0, atol=1e-5))
        self.assertIsNone(npt.assert_allclose(yp, tows.dydx, rtol=0,
                                              atol=2e-5))

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamberEnhanced(0.05, 0.30)

        # Note: while published data from Jacobs and Pinkerton (1936) has
        #       values, they are noticable off from actual values. These
        #       reference values come from previous Matlab implementation.
        p = np.array([0.05, 0.10, 0.15, 0.20, 0.25])
        Cl_ideal = 0.3
        # m from ref.    [0.0580,       0.1260,       0.2025,
        #                 0.2900,       0.3910]
        m_ref = np.array([0.0580815972, 0.1257435084, 0.2026819782,
                          0.2903086448, 0.3913440104])
        # k1 from ref.    [361.4000,       51.6400,       15.9570,
        #                  6.6430,       3.2300]
        k1_ref = np.array([350.3324130671, 51.5774898103, 15.9196523168,
                           6.6240446646, 3.2227606900])

        # test the static methods
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            k1 = af._determine_k1(Cl_ideal, mit)
            y_p = af._camber_slope(pit, k1it, mit)
            self.assertIsNone(npt.assert_allclose(k1, k1it))
            self.assertIsNone(npt.assert_allclose(y_p, 0, atol=1e-7))

        # test the initialization of camber
        af.Cl_ideal = Cl_ideal
        for pit, mit, k1it in np.nditer([p, m_ref, k1_ref]):
            af.p = pit
            self.assertIsNone(npt.assert_allclose(af.m, mit))
            self.assertIsNone(npt.assert_allclose(af.k1, k1it))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af_classic = Naca5DigitCamberClassic(camber_loc=2)
        af_enhanced = Naca5DigitCamberEnhanced(p=0.25, Cl_ideal=0.35)

        def compare_values(xi: np_type.NDArray,
                           af: Naca5DigitCamberBase) -> None:
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
        af = Naca5DigitCamberClassic(camber_loc=3)

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
        af = Naca5DigitCamberClassic(camber_loc=3)

        self.assertListEqual([0.2025], af.joints())


class TestNaca5DigitReflexed(unittest.TestCase):
    """Class to test the NACA 5-digit reflexed camber geometry."""

    def testSetters(self) -> None:
        """Test the setting of the max. camber location and ideal lift coef."""
        af = Naca5DigitCamberReflexedEnhanced(0.05, 0.30)

        # Note: while published data from Jacobs and Pinkerton (1936) has
        #       values, they are noticable off from actual values. These
        #       reference values come from previous Matlab implementation.
        p = np.array([0.10, 0.15, 0.20, 0.25])
        Cl_ideal = 0.3
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
            k2ok1 = af._k2ok1(m=mit, p=pit)
            k2ok1_ref = k2it/k1it
            Cmc4 = af._Cmc4(m=mit, k1=k1it, k2ok1=k2it/k1it, Cl_id=Cl_ideal)
            self.assertIsNone(npt.assert_allclose(Cl_id, Cl_ideal))
            self.assertIsNone(npt.assert_allclose(k2ok1, k2ok1_ref))
            self.assertIsNone(npt.assert_allclose(Cmc4, 0, atol=1e-7))

        # test the initialization of camber
        af.Cl_ideal = Cl_ideal
        for pit, mit, k1it, k2it in np.nditer([p, m_ref, k1_ref, k2_ref]):
            af.p = pit
            self.assertIsNone(npt.assert_allclose(af.m, mit))
            self.assertIsNone(npt.assert_allclose(af.k1, k1it))
            self.assertIsNone(npt.assert_allclose(af.k2, k2it))

    def testCamber(self) -> None:
        """Test the camber relations."""
        af_classic = Naca5DigitCamberReflexedClassic(camber_loc=2)
        af_enhanced = Naca5DigitCamberReflexedEnhanced(p=0.25, Cl_ideal=0.35)

        def compare_values(xi: np_type.NDArray,
                           af: Naca5DigitCamberReflexedBase) -> None:
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
        af = Naca5DigitCamberReflexedClassic(camber_loc=3)

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
        af = Naca5DigitCamberReflexedClassic(camber_loc=3)

        self.assertListEqual([0.2170], af.joints())


if __name__ == "__main__":
    unittest.main(verbosity=1)
