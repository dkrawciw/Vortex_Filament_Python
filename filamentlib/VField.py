import numpy as np

class VField():
# Define various methods of calculating the velocity field at a point

# BiotSavart is the general way of calculating the velocity field at a point
    @staticmethod
    def BiotSavart(curve, curveTangent, curvePoint):
        s = range(0,len(curve[0]))
        dv = []
        for c in s:
            num = np.cross( curveTangent[:,c], curvePoint - curve[:,c] )
            den = np.linalg.norm(curvePoint - curve[:,c]) ** 3
            dv.append( np.divide(num,den) )

        dv = np.array(dv)

        v = np.trapz([dv[:,0],dv[:,1],dv[:,2]], s)
        return v