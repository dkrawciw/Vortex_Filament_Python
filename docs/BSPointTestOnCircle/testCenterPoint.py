import numpy as np
import matplotlib.pyplot as plt
from TestUtils.circleTest import circleTest

# Calculate points on the circle to plot
circle = lambda t : [np.cos(t), np.sin(t), t * 0]
circlePoints = np.linspace(0,2*np.pi,1000)
s = np.array(circle(circlePoints))

# Set up some values as initial conditions
TOTAL_TIME = 10
TOTAL_STEPS = 100
IV = np.array( [0,0,0] )

# Simulate the point
pts = circleTest(TOTAL_TIME, TOTAL_STEPS, IV)

# Plot the points around the circle
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot(s[0,:],s[1,:],s[2,:])
ax.plot(pts[0,:],pts[1,:],pts[2,:])

plt.show()