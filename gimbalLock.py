#!/usr/bin/python3

import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, TextBox

# Global variables.
fig, axis = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
euler, quaternion = axis
vx, vy, vz, alpha, beta, gamma = 1., 1., 0., 0, 0, 0
vectorChanged, eulerLim = True, None

# Update 3-D axis.
def initAxis(axis):
    # Show cartesian axes.
    axis.clear() # Reset plot.
    axis.quiver(-1.,  0.,  0., 2., 0., 0., color='gray', linestyle='dashed')
    axis.quiver( 0., -1.,  0., 0., 2., 0., color='gray', linestyle='dashed')
    axis.quiver( 0.,  0., -1., 0., 0., 2., color='gray', linestyle='dashed')
    # Vector before rotation
    axis.quiver(0., 0., 0., vx, vy, vz, color='b')
def setAxisLim(axis, W, axisLim):
    # Set axis limits.
    global vectorChanged
    if vectorChanged: # Fit limits to new data.
        normW, coef = np.linalg.norm(W), 1.1
        axis.set_xlim3d(-coef*normW, coef*normW)
        axis.set_ylim3d(-coef*normW, coef*normW)
        axis.set_zlim3d(-coef*normW, coef*normW)
        vectorChanged = False
    else: # Preserve previous setup.
        axis.set_xlim3d(axisLim[0])
        axis.set_ylim3d(axisLim[1])
        axis.set_zlim3d(axisLim[2])

# Apply rotation.
def applyEulerRotation():
    # Rotate vector.
    theta = np.deg2rad(alpha)
    Rx = np.array([[ 1.,              0.,               0.],
                   [ 0., math.cos(theta), -math.sin(theta)],
                   [ 0., math.sin(theta),  math.cos(theta)]])
    theta = np.deg2rad(beta)
    Ry = np.array([[  math.cos(theta), 0., math.sin(theta)],
                   [               0., 1.,              0.],
                   [ -math.sin(theta), 0., math.cos(theta)]])
    theta = np.deg2rad(gamma)
    Rz = np.array([[  math.cos(theta), -math.sin(theta), 0.],
                   [  math.sin(theta),  math.cos(theta), 0.],
                   [               0.,               0., 1.]])
    V = np.array([[vx], [vy], [vz]])
    W = Rx@Ry@Rz@V

    # Plotting.
    eulerLim = (euler.get_xlim3d(), euler.get_ylim3d(), euler.get_zlim3d())
    initAxis(euler)
    euler.quiver(0., 0., 0., W[0], W[1], W[2], color='g')
    setAxisLim(euler, W, eulerLim)
    euler.set_title('Euler rotation')
def applyQuaternionRotation():
    # Defining all 3 axes.
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z + beta)
    y = z * np.cos(25 * z + gamma)

    # Plotting.
    initAxis(quaternion)
    quaternion.plot3D(x, y, z, 'red')
    quaternion.set_title('Quaternion rotation')
def applyRotation():
    print('vx %7.3f, vy %7.3f, vz %7.3f' % (vx, vy, vz), end=', ')
    print('alpha %3d, beta %3d, gamma %3d' % (alpha, beta, gamma))
    applyEulerRotation()
    applyQuaternionRotation()
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

# Main function.
def main():
    # Make textbox to initialize vector to rotate.
    axisLabelVx = fig.add_axes([0.1, 0.09, 0.3, 0.03])
    textboxLabelVx = TextBox(axisLabelVx, 'Vx', textalignment='center')
    textboxLabelVx.on_text_change(applyVx)
    textboxLabelVx.set_val("%s" % vx)
    axisLabelVy = fig.add_axes([0.1, 0.06, 0.3, 0.03])
    textboxLabelVy = TextBox(axisLabelVy, 'Vy', textalignment='center')
    textboxLabelVy.on_text_change(applyVy)
    textboxLabelVy.set_val("%s" % vy)
    axisLabelVz = fig.add_axes([0.1, 0.03, 0.3, 0.03])
    textboxLabelVz = TextBox(axisLabelVz, 'Vz', textalignment='center')
    textboxLabelVz.on_text_change(applyVz)
    textboxLabelVz.set_val("%s" % vz)

    # Make horizontal sliders to control the angles.
    axisSliderRx = fig.add_axes([0.6, 0.09, 0.3, 0.05])
    sliderRx = Slider(ax=axisSliderRx, label='alpha (Rx) [°]',
                      valmin=-180, valmax=180, valinit=alpha, valfmt='%i', valstep=5)
    sliderRx.on_changed(applyRx)
    axisSliderRy = fig.add_axes([0.6, 0.06, 0.3, 0.05])
    sliderRy = Slider(ax=axisSliderRy, label='beta (Ry) [°]',
                      valmin=-180, valmax=180, valinit=beta, valfmt='%i', valstep=5)
    sliderRy.on_changed(applyRy)
    axisSliderRz = fig.add_axes([0.6, 0.03, 0.3, 0.05])
    sliderRz = Slider(ax=axisSliderRz, label='gamma (Rz) [°]',
                      valmin=-180, valmax=180, valinit=gamma, valfmt='%i', valstep=5)
    sliderRz.on_changed(applyRz)

    # Show plots.
    plt.show()

# Python script entry point.
if __name__ == "__main__":
    main()
