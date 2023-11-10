import numpy as np
import matplotlib.pyplot as plt
from BiotSavart import BiotSavart

circle = lambda t : [np.cos(t), np.sin(t), t * 0]
dCircle = lambda t : [-np.sin(t), np.cos(t), t * 0]
circlePoints = np.linspace(0,2*np.pi,1000)

s = np.array(circle(circlePoints))
ds = np.array(dCircle(circlePoints))

TOTAL_TIME = 10
TIME_STEPS = 100
timeSlope = TOTAL_TIME / TIME_STEPS
pts = np.zeros([3,TIME_STEPS])

# Initial condition
pts[:,0] = np.array([0,0,-1])
for step in range(1,TIME_STEPS):
    pts[:, step] = pts[:,step - 1] + timeSlope * (BiotSavart(s,ds,pts[:,step - 1]))

fig = plt.figure()
ax = plt.axes(projection='3d')

ax.plot(s[0,:],s[1,:],s[2,:])
ax.plot(pts[0,:],pts[1,:],pts[2,:])

plt.show()