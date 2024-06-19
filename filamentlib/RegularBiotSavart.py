import numpy as np

def RegularBiotSavart( curve: np.array, curveTangent: np.array, fieldPoints: np.array, eps=10**-8 ):
    pointFieldStrength = np.zeros( (len(curve[0,:]), len( fieldPoints[:,0] ), len( fieldPoints[0,:] ) ) )

    s = range( len(curve[0,:]) )

    for i in s:
        pointDistances =  fieldPoints - curve[:,i].reshape(-1,1)

        pointNorms = np.linalg.norm(pointDistances, axis=0)
        pointNormsCubed = np.power( pointNorms, 3)

        crossProduct = ( np.cross(curveTangent[:,i], pointDistances.T ) ).T
        pointFieldStrength[i,:,:] = crossProduct / (pointNormsCubed + eps)

    v = np.trapz( pointFieldStrength, s , axis=0)
    return v