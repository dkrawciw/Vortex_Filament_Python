import numpy as np

class VField():
# Define various methods of calculating the velocity field at a point

    #BiotSavart is the general way of calculating the velocity field at some given points over a curve with given tangent points
    @staticmethod
    def BiotSavart( curve: np.array, curveTangent: np.array, fieldPoints: np.array ):
        pointFieldStrength = np.zeros( (len(curve[0,:]), len( fieldPoints[:,0] ), len( fieldPoints[0,:] ) ) )

        s = range( len(curve[0,:]) )

        for i in s:
            pointDistances =  fieldPoints - curve[:,i].reshape(-1,1)

            pointNorms = np.linalg.norm(pointDistances, axis=0)
            pointNormsCubed = np.power( pointNorms, 3)

            crossProduct = ( np.cross(curveTangent[:,i], pointDistances.T ) ).T
            pointFieldStrength[i,:,:] = crossProduct / pointNormsCubed

        v = np.trapz( pointFieldStrength, s , axis=0)
        return v
    
    # This method adds an epsilon in the denominator where the divide by zero error occurs
    @staticmethod
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
    
    @staticmethod
    def KappaBinormal( curve: np.array ):
        curveTangent = np.gradient(curve)[1]
        curveNormal = np.gradient(curveTangent)[1]
        curveBinormal = np.cross(curveTangent,curveNormal, axisa=0, axisb=0, axisc=0)

        Kappa = np.divide( np.linalg.norm( curveBinormal ), np.power( np.linalg.norm(curveTangent),3 ) )

        KBApprox = np.multiply(Kappa, curveBinormal)

        return KBApprox