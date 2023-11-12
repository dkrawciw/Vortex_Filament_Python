import numpy as np
from TestUtils.BiotSavart import BiotSavart

circle = lambda t : [np.cos(t), np.sin(t), t * 0]
dCircle = lambda t : [-np.sin(t), np.cos(t), t * 0]
circlePoints = np.linspace(0,2*np.pi,1000)

s = np.array(circle(circlePoints))
ds = np.array(dCircle(circlePoints))

def circleTest(totalTime, totalSteps, IV):
    timeSlope = totalTime / totalSteps
    pts = np.zeros([3, totalSteps])

    pts[:,0] = IV
    for step in range(1,totalSteps):
        pts[:, step] = pts[:,step - 1] + timeSlope * (BiotSavart(s,ds,pts[:,step - 1]))
    
    return pts