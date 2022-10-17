#!/usr/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, TextBox, CheckButtons

# Global variables.

fig, axis = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
eAxis, quaternion = axis
vx, vy, vz, alpha, beta, gamma = 1., 1., 0., 0, 0, 0
chkBtn, sameView, sameLim = None, True, True
vectorChanged, eulerLim, quaternionLim = True, None, None

# Define transformations.

class eulerTransform:

    @staticmethod
    def getRx(alpha):
        # Return rotation over x axis.
        theta = np.deg2rad(alpha)
        Rx = np.array([[ 1.,              0.,               0.],
                       [ 0., math.cos(theta), -math.sin(theta)],
                       [ 0., math.sin(theta),  math.cos(theta)]])
        return Rx

    @staticmethod
    def getRy(beta):
        # Return rotation over y axis.
        theta = np.deg2rad(beta)
        Ry = np.array([[  math.cos(theta), 0., math.sin(theta)],
                       [               0., 1.,              0.],
                       [ -math.sin(theta), 0., math.cos(theta)]])
        return Ry

    @staticmethod
    def getRz(gamma):
        # Return rotation over z axis.
        theta = np.deg2rad(gamma)
        Rz = np.array([[  math.cos(theta), -math.sin(theta), 0.],
                       [  math.sin(theta),  math.cos(theta), 0.],
                       [               0.,               0., 1.]])
        return Rz

class quaternionTransform:

    def __init__(self, scalar, vector):
        # Initialization.
        assert isinstance(vector, np.ndarray) and vector.shape == (3, 1)
        self.scalar = float(scalar)
        self.vector = vector

    def normalize(self, rotation=False):
        # Normalize vector.
        if rotation: # Rotation normalization.
            norm = np.linalg.norm(self.vector)
            if norm > 0.:
                self.vector = self.vector/norm
            theta = np.deg2rad(self.scalar)
            self.scalar = math.cos(0.5*theta)
            self.vector = math.sin(0.5*theta)*self.vector
        else: # Standard normalization.
            norm2 = self.scalar**2 + float(np.dot(self.vector.T, self.vector))
            norm = math.sqrt(norm2)
            self.scalar = self.scalar/norm
            self.vector = self.vector/norm

    def conjugate(self):
        # Conjugate.
        self.vector = -self.vector

    def inverse(self):
        # Inverse.
        self.conjugate()
        norm2 = self.scalar**2 + float(np.dot(self.vector.T, self.vector))
        self.scalar = self.scalar/norm2
        self.vector = self.vector/norm2

    def product(self, q, right=True):
        # Product.
        scalar, vector = self.scalar, self.vector # Copy.
        self.scalar = scalar*q.scalar - float(np.dot(vector.T, q.vector))
        self.vector = scalar*q.vector + q.scalar*vector
        if right: # Product self*q (!= q*self).
            self.vector += np.cross(vector, q.vector, axis=0)
        else: # Product q*self (!= self*q).
            self.vector += np.cross(q.vector, vector, axis=0)

    def rotate(self, lsQ):
        # Compose rotations.
        if len(lsQ) == 0:
            return
        q = quaternionTransform(lsQ[0].scalar, lsQ[0].vector)
        q.normalize(rotation=True)
        for qNext in lsQ[1:]:
            qNext.normalize(rotation=True)
            q.product(qNext, right=False)

        # Rotate.
        qInv = quaternionTransform(q.scalar, q.vector)
        qInv.inverse()
        self.product(qInv, right=True)
        self.product(q, right=False)

# Update 3-D axis.

def initAxis(axis):
    # Show cartesian axes.
    axis.clear() # Reset plot.
    axis.quiver(-1.,  0.,  0., 2., 0., 0., color='gray', linestyle='dashed')
    axis.quiver( 0., -1.,  0., 0., 2., 0., color='gray', linestyle='dashed')
    axis.quiver( 0.,  0., -1., 0., 0., 2., color='gray', linestyle='dashed')
    # Vector before rotation
    axis.quiver(0., 0., 0., vx, vy, vz, color='orange', linestyle='dashed')

def setAxisLim(axis, W, axisLim):
    # Set axis limits.
    if W is not None and vectorChanged: # Fit limits to new data.
        norm, coef = np.linalg.norm(W), 1.1
        axis.set_xlim3d(-coef*norm, coef*norm)
        axis.set_ylim3d(-coef*norm, coef*norm)
        axis.set_zlim3d(-coef*norm, coef*norm)
    else: # Preserve previous setup.
        axis.set_xlim3d(axisLim[0])
        axis.set_ylim3d(axisLim[1])
        axis.set_zlim3d(axisLim[2])

# Apply rotation.

def applyEulerRotation(V, angles):
    # Rotate vector.
    thetaX, thetaY, thetaZ = angles
    trf = eulerTransform()
    Rz = trf.getRz(thetaZ)
    Ry = trf.getRy(thetaY)
    Rx = trf.getRx(thetaX)
    W = Rx@Ry@Rz@V

    # Print.
    data = (W[0], W[1], W[2], np.linalg.norm(W))
    print('Euler rotation:      W = (%.3f, %.3f, %.3f), ||W|| = %.6f' % data)

    # Plotting.
    global eulerLim
    eulerLim = (eAxis.get_xlim3d(), eAxis.get_ylim3d(), eAxis.get_zlim3d())
    initAxis(eAxis)
    eAxis.quiver(0., 0., 0., W[0], W[1], W[2], color='orange')
    setAxisLim(eAxis, W, eulerLim)
    eAxis.set_title('Euler rotation')

    return W

def applyQuaternionRotation(V, angles):
    # Rotate vector.
    thetaX, thetaY, thetaZ = angles
    zAxis = np.array([[0.], [0.], [1.]])
    Rz = quaternionTransform(thetaZ, zAxis) # Rotation z axis.
    yAxis = np.array([[0.], [1.], [0.]])
    Ry = quaternionTransform(thetaY, yAxis) # Rotation y axis.
    xAxis = np.array([[1.], [0.], [0.]])
    Rx = quaternionTransform(thetaX, xAxis) # Rotation x axis.
    P = quaternionTransform(0., V) # Pure quaternion.
    P.rotate([Rz, Ry, Rx])
    W = P.vector # Rotated vector.

    # Print.
    data = (W[0], W[1], W[2], np.linalg.norm(W))
    print('Quaternion rotation: W = (%.3f, %.3f, %.3f), ||W|| = %.6f' % data)

    # Plotting.
    global quaternionLim
    quaternionLim = (quaternion.get_xlim3d(), quaternion.get_ylim3d(), quaternion.get_zlim3d())
    initAxis(quaternion)
    quaternion.quiver(0., 0., 0., W[0], W[1], W[2], color='orange')
    setAxisLim(quaternion, W, quaternionLim)
    quaternion.set_title('Quaternion rotation')

    return W

def applyRotation():
    angles = (alpha, beta, gamma)
    print('Alpha %3d, Beta %3d, Gamma %3d' % angles)
    V = np.array([[vx], [vy], [vz]])
    data = (V[0], V[1], V[2], np.linalg.norm(V))
    print('Initial vector:      V = (%.3f, %.3f, %.3f), ||V|| = %.6f' % data)
    eulerW = applyEulerRotation(V, angles)
    quaternionW = applyQuaternionRotation(V, angles)
    D = np.fabs(eulerW - quaternionW)
    data = (D[0], D[1], D[2], np.linalg.norm(D))
    print('Difference:          D = (%.3f, %.3f, %.3f), ||D|| = %.6f' % data)
    print('')
    vectorChanged = False
    plt.draw() # Update plots.

# Callback on GUI widgets.

def applyVx(val):
    global vx, vectorChanged
    try:
        vx = float(val)
        vectorChanged = True
    except:
        return
    applyRotation()

def applyVy(val):
    global vy, vectorChanged
    try:
        vy = float(val)
        vectorChanged = True
    except:
        return
    applyRotation()

def applyVz(val):
    global vz, vectorChanged
    try:
        vz = float(val)
        vectorChanged = True
    except:
        return
    applyRotation()

def applyRx(val):
    global alpha
    alpha = val
    applyRotation()

def applyRy(val):
    global beta
    beta = val
    applyRotation()

def applyRz(val):
    global gamma
    gamma = val
    applyRotation()

def onChkBtnChange(label):
    if label == 'same view':
        global sameView
        sameView = chkBtn.get_status()[0]
    if label == 'same limits':
        global sameLim
        sameLim = chkBtn.get_status()[1]

def onMove(event):
    global eAxis, quaternion, eulerLim, quaternionLim
    if event.inaxes is not None:
        if sameView:
            if event.inaxes == eAxis:
                quaternion.view_init(eAxis.elev, eAxis.azim)
            if event.inaxes == quaternion:
                eAxis.view_init(quaternion.elev, quaternion.azim)
        if sameLim:
            if event.inaxes == eAxis:
                eulerLim = (eAxis.get_xlim3d(),
                            eAxis.get_ylim3d(),
                            eAxis.get_zlim3d())
                quaternionLim = eulerLim
                setAxisLim(quaternion, None, quaternionLim)
            if event.inaxes == quaternion:
                quaternionLim = (quaternion.get_xlim3d(),
                                 quaternion.get_ylim3d(),
                                 quaternion.get_zlim3d())
                eulerLim = quaternionLim
                setAxisLim(eAxis, None, eulerLim)
        plt.draw() # Update plots.

# Main function.

def main():
    # Make textbox to initialize vector to rotate.
    axisLabelVx = fig.add_axes([0.1, 0.09, 0.2, 0.03])
    textboxLabelVx = TextBox(axisLabelVx, 'Vx', textalignment='center')
    textboxLabelVx.set_val("%s" % vx)
    textboxLabelVx.on_text_change(applyVx)
    axisLabelVy = fig.add_axes([0.1, 0.06, 0.2, 0.03])
    textboxLabelVy = TextBox(axisLabelVy, 'Vy', textalignment='center')
    textboxLabelVy.set_val("%s" % vy)
    textboxLabelVy.on_text_change(applyVy)
    axisLabelVz = fig.add_axes([0.1, 0.03, 0.2, 0.03])
    textboxLabelVz = TextBox(axisLabelVz, 'Vz', textalignment='center')
    textboxLabelVz.set_val("%s" % vz)
    textboxLabelVz.on_text_change(applyVz)

    # Make horizontal sliders to control the angles.
    axisSliderRx = fig.add_axes([0.5, 0.09, 0.2, 0.05])
    sliderRx = Slider(ax=axisSliderRx, label='alpha (Rx) [°]',
                      valmin=-180, valmax=180, valinit=alpha, valfmt='%i', valstep=5)
    sliderRx.on_changed(applyRx)
    axisSliderRy = fig.add_axes([0.5, 0.06, 0.2, 0.05])
    sliderRy = Slider(ax=axisSliderRy, label='beta (Ry) [°]',
                      valmin=-180, valmax=180, valinit=beta, valfmt='%i', valstep=5)
    sliderRy.on_changed(applyRy)
    axisSliderRz = fig.add_axes([0.5, 0.03, 0.2, 0.05])
    sliderRz = Slider(ax=axisSliderRz, label='gamma (Rz) [°]',
                      valmin=-180, valmax=180, valinit=gamma, valfmt='%i', valstep=5)
    sliderRz.on_changed(applyRz)
    global chkBtn
    axisChkBtn = fig.add_axes([0.8, 0.03, 0.1, 0.1])
    chkBtn = CheckButtons(axisChkBtn,
                          ('same view', 'same limits'),
                          (sameView, sameLim))
    chkBtn.on_clicked(onChkBtnChange)
    fig.canvas.mpl_connect('motion_notify_event', onMove)

    # Draw plots.
    applyRotation() # Init plots.
    plt.show() # Show plots.

# Python script entry point.

if __name__ == "__main__":
    main()
