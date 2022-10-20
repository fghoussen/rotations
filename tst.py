#!/usr/bin/python3

import unittest

import gimbalLock
import numpy as np

class TestGimbalLock(unittest.TestCase):

    def runTest(self, V, angles, rotOrder):
        print('')
        print('Rotation order %s' % rotOrder)
        print('Alpha %3d, Beta %3d, Gamma %3d' % angles)
        data = (V[0], V[1], V[2], np.linalg.norm(V))
        print('Initial vector:      V = (%.3f, %.3f, %.3f), ||V|| = %.6f' % data)
        eulerW = gimbalLock.applyEulerRotation(V, angles, rotOrder)
        quaternionW = gimbalLock.applyQuaternionRotation(V, angles, rotOrder)
        D = np.fabs(eulerW - quaternionW)
        data = (D[0], D[1], D[2], np.linalg.norm(D))
        print('Difference:          D = (%.3f, %.3f, %.3f), ||D|| = %.6f' % data)
        self.assertTrue(np.linalg.norm(D) < 1.e-6)

    def test1(self):
        # X -> ...
        V = np.array([[1.], [0.], [0.]])
        for rotOrder in ['ZYX', 'XYZ']:
            for coefV in [1., -1, 2.]:
                for coefA in [1., -1.]:
                    angles = (0., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 0., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 0., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 90., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)

    def test2(self):
        # Y -> ...
        V = np.array([[0.], [1.], [0.]])
        for rotOrder in ['ZYX', 'XYZ']:
            for coefV in [1., -1, 2.]:
                for coefA in [1., -1.]:
                    angles = (0., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 90., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 90., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)

    def test3(self):
        # Z -> ...
        V = np.array([[0.], [0.], [1.]])
        for rotOrder in ['ZYX', 'XYZ']:
            for coefV in [1., -1, 2.]:
                for coefA in [1., -1.]:
                    angles = (0., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 90., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 90., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (90., 90., 90.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)

    def test4(self):
        # V -> ...
        V = np.array([[1.], [1.], [1.]])
        for rotOrder in ['ZYX', 'XYZ']:
            for coefV in [1., -1, 2.]:
                for coefA in [1., -1.]:
                    angles = (0., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (45., 0., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 45., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 0., 45.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (45., 45., 0.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (0., 45., 45.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (45., 0., 45.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)
                    angles = (45., 45., 45.)
                    self.runTest(coefV*V, tuple(coefA*np.array(angles)), rotOrder)

if __name__ == '__main__':
    unittest.main(verbosity=2, failfast=True)
