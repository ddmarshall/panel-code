#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 18 17:15:25 2022

@author: ddmarshall
"""

from os.path import abspath, dirname
import unittest

import numpy as np
import numpy.testing as npt

from pyPC.vortex_flow import PointVortex2D


class TestPointVortex2D(unittest.TestCase):
    """Class to test the 2D point vortex."""

    def testPotential(self) -> None:
        """Test the calculation of the potential."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        pv = PointVortex2D()
        pv.x0 = 0.1
        pv.y0 = 0.2
        pv.set_strength(0.5)

        # test some hand calculations
        phi = pv.potential(xpt[0], ypt[0])
        phi_ref = -0.0578243602
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

        phi = pv.potential(xpt, ypt)
        phi_ref = np.array([-0.0578243602, -0.0603497810])
        self.assertIsNone(npt.assert_allclose(phi, phi_ref))

    def testStreamFunction(self) -> None:
        """Test the calculation of the stream function."""
        xpt = np.array([1., 2.])
        ypt = np.array([1., 2.])

        pv = PointVortex2D()
        pv.x0 = 0.1
        pv.y0 = 0.2
        pv.set_strength(0.5)

        # test some hand calculations
        psi = pv.stream_function(xpt[0], ypt[0])
        psi_ref = 0.0147840442
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

        psi = pv.stream_function(xpt, ypt)
        psi_ref = np.array([0.0147840442, 0.0765634212])
        self.assertIsNone(npt.assert_allclose(psi, psi_ref))

    def testVelocity(self) -> None:
        """Test the calculation of the velocity vector."""
        xp = np.array([1., 2.])
        yp = np.array([1., 2.])

        pv = PointVortex2D()
        pv.x0 = 0.1
        pv.y0 = 0.2
        pv.set_strength(0.5)

        # test some hand calculations
        u, v = pv.velocity(xp[1], yp[1])
        u_ref = 0.0209108684
        v_ref = -0.0220725833
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

        u, v = pv.velocity(xp, yp)
        u_ref = np.array([0.0439048119, 0.0209108684])
        v_ref = np.array([-0.0493929134, -0.0220725833])
        self.assertIsNone(npt.assert_allclose(u, u_ref))
        self.assertIsNone(npt.assert_allclose(v, v_ref))

    def testReferenceImplementation(self) -> None:
        """Test the calculations against a reference implementation."""

        # Read the reference data
        directory = dirname(abspath(__file__))
        ref_filename = directory + "/data/point_vortex_2d.dat"
        with open(ref_filename, "r", encoding="utf8") as ref_file:
            # read element location
            line = ref_file.readline()
            data = line.split()
            self.assertTrue(len(data), 2)
            x0 = float(data[0])
            y0 = float(data[1])

            # read element strength
            line = ref_file.readline()
            data = line.split()
            self.assertTrue(len(data), 1)
            strength = float(data[0])

            # read in dataset size
            line = ref_file.readline()
            data = line.split()
            self.assertTrue(len(data), 2)
            nx = int(data[0])
            ny = int(data[1])

            xp = np.empty(shape=[nx, ny])
            yp = np.empty(shape=[nx, ny])
            u_ref = np.empty(shape=[nx, ny])
            v_ref = np.empty(shape=[nx, ny])
            phi_ref = np.empty(shape=[nx, ny])
            psi_ref = np.empty(shape=[nx, ny])
            # read in all of the values
            for i in range(nx):
                for j in range(ny):
                    line = ref_file.readline()
                    data = line.split()
                    self.assertTrue(len(data), 6)
                    xp[i, j] = float(data[0])
                    yp[i, j] = float(data[1])
                    u_ref[i, j] = float(data[2])
                    v_ref[i, j] = float(data[3])
                    phi_ref[i, j] = float(data[4])
                    psi_ref[i, j] = float(data[5])

        pv = PointVortex2D(x0=x0, y0=y0, strength=strength)
        phi = pv.potential(xp, yp)
        self.assertIsNone(npt.assert_allclose(phi, phi_ref, rtol=1e-5))
        psi = pv.stream_function(xp, yp)
        self.assertIsNone(npt.assert_allclose(psi, psi_ref, rtol=1e-5))
        u, v = pv.velocity(xp, yp)
        self.assertIsNone(npt.assert_allclose(u, u_ref, rtol=2e-5))
        self.assertIsNone(npt.assert_allclose(v, v_ref, rtol=1e-5))


if __name__ == "__main__":
    unittest.main(verbosity=1)
