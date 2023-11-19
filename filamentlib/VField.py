import numpy as np

class VField():
# Define various methods of calculating the velocity field at a point

#    BiotSavart is the general way of calculating the velocity field at a point
    @staticmethod
    def BiotSavartPoint(curve, curveTangent, curvePoint):
        s = range(0,len(curve[0]))
        dv = []
        for c in s:
            num = np.cross( curveTangent[:,c], curvePoint - curve[:,c] )
            den = np.linalg.norm(curvePoint - curve[:,c]) ** 3
            dv.append( np.divide(num,den) )

        dv = np.array(dv)

        v = np.trapz([dv[:,0],dv[:,1],dv[:,2]], s)
        return v
    
    # Having a function deal with a list of 3D points in a nxm matrix where m = 3 (3 columns)
    @staticmethod
    def BiotSavartPoints(curve, curveTangent, curvePoints):
        biotSavartPoints = []

        for i in range( len(curvePoints[0,:]) ):
            curvePoint = curvePoints[:,i]
            biotSavartPoints.append( VField.BiotSavartPoint( curve, curveTangent, curvePoint ) )
        
        return np.array( biotSavartPoints )