import numpy as np

class VField():
# Define various methods of calculating the velocity field at a point

    #BiotSavart is the general way of calculating the velocity field at some given points over a curve with given tangent points
    @staticmethod
    def BiotSavart( curve: np.array, curveTangent: np.array, fieldPoints: np.array ):
        pointFieldStrength = np.zeros( ( len( fieldPoints[0,:] ), len( fieldPoints[:,0] ) ) )

        s = range( len(curve[0,:]) )

        for i in s:
            pointDistances = fieldPoints.T.reshape(-1, fieldPoints.shape[0])
            pointDistances =  pointDistances - curve[:,i]

            pointNorms = np.linalg.norm(pointDistances, axis=1, keepdims=True)
            pointNormsCubed = np.power( pointNorms, 3)

            crossProduct = np.cross(curveTangent[: , i], pointDistances )
            pointFieldStrength += np.divide(crossProduct, pointNormsCubed)

        np.trapz( [pointFieldStrength[:,0], pointFieldStrength[:,1], pointFieldStrength[:,2]], s )
        return pointFieldStrength