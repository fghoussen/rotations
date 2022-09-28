#!/usr/bin/python3

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.widgets import Slider, TextBox

# Create 3-D axis.
fig, axis = plt.subplots(1, 2, subplot_kw=dict(projection='3d'))
euler, quaternion = axis

# Apply rotation.
vx, vy, vz, alpha, beta, gamma = 1., 0., 0., 45, 45, 45
def applyEulerRotation():
    # Defining all 3 axes.
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z + alpha)
    y = z * np.cos(25 * z + beta)

    # Plotting.
    euler.clear() # Reset plot.
    euler.set_title('Euler rotation')
    euler.plot3D(x, y, z, 'green')

def applyQuaternionRotation():
    # Defining all 3 axes.
    z = np.linspace(0, 1, 100)
    x = z * np.sin(25 * z + beta)
    y = z * np.cos(25 * z + gamma)

    # Plotting.
    quaternion.clear() # Reset plot.
    quaternion.set_title('Quaternion rotation')
    quaternion.plot3D(x, y, z, 'red')

def applyRotation():
    print('vx %.3f, vy %.3f, vz %.3f' % (vx, vy, vz), end=', ')
    print('alpha %03d, beta %03d, gamma %03d' % (alpha, beta, gamma))
    applyEulerRotation()
    applyQuaternionRotation()
    plt.draw() # Update plots.

# Callback on GUI widgets.
def applyVx(val):
    global vx
    try:
        vx = float(val)
    except:
        return
    applyRotation()
def applyVy(val):
    global vy
    try:
        vy = float(val)
    except:
        return
    applyRotation()
def applyVz(val):
    global vz
    try:
        vz = float(val)
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
                      valmin=0, valmax=360, valinit=alpha, valfmt='%i', valstep=1)
    sliderRx.on_changed(applyRx)
    axisSliderRy = fig.add_axes([0.6, 0.06, 0.3, 0.05])
    sliderRy = Slider(ax=axisSliderRy, label='beta (Ry) [°]',
                      valmin=0, valmax=360, valinit=beta, valfmt='%i', valstep=1)
    sliderRy.on_changed(applyRy)
    axisSliderRz = fig.add_axes([0.6, 0.03, 0.3, 0.05])
    sliderRz = Slider(ax=axisSliderRz, label='gamma (Rz) [°]',
                      valmin=0, valmax=360, valinit=gamma, valfmt='%i', valstep=1)
    sliderRz.on_changed(applyRz)

    # Show plots.
    plt.show()

# Python script entry point.
if __name__ == "__main__":
    main()
