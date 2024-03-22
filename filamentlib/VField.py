import numpy as np

class VField():
    # Define various methods of calculating the velocity field at a point
    @staticmethod
    def closedFirstGradient(arr: np.array):
        grad = np.zeros((arr.shape[0],arr.shape[1]))
        for i in range( 1, arr.shape[1] - 1 ):
            num = arr[:,i+1] - arr[:,i-1]
            den = np.linalg.norm(arr[:,i+1] - arr[:,i-1])
            grad[:,i] = np.divide(num, den)

        grad[:,-1] = (arr[:,0] - arr[:,-2]) / (np.linalg.norm(arr[:,0] - arr[:,-2]))
        grad[:,0] = (arr[:,1] - arr[:,-1]) / (np.linalg.norm(arr[:,1] - arr[:,-1]))

        return grad
    
    @staticmethod
    def closedSecondGradient(arr: np.array):
        grad = np.zeros((arr.shape[0],arr.shape[1]))
        for i in range( 1, arr.shape[1] - 1 ):
            num = arr[:,i+1] - 2*arr[:,i] + arr[:,i-1]
            den = np.power(np.linalg.norm(arr[:,i+1] - arr[:,i-1]),2)
            grad[:,i] = np.divide(num, den)

        grad[:,-1] = arr[:,0] - 2*arr[:,-1] + arr[:,-2] / np.power(np.linalg.norm(arr[:,0] - arr[:,-2]),2)
        grad[:,0] = arr[:,1] - 2*arr[:,0] + arr[:,-1] / np.power(np.linalg.norm(arr[:,1] - arr[:,-1]),2)

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
    
    # Fast Approximation to BiotSavart
    @staticmethod
    def KappaBinormal( curve: np.array ):
        curveTangent = VField.closedFirstGradient(curve)
        curveNormal = VField.closedSecondGradient(curve)
        curveBinormal = np.cross(curveTangent,curveNormal, axisa=0, axisb=0, axisc=0)

        Kappa = np.divide( np.linalg.norm( curveBinormal ), np.power( np.linalg.norm(curveTangent),3 ) )

        KBApprox = np.multiply(Kappa, curveBinormal)

        return KBApprox