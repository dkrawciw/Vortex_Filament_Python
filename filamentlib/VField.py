import numpy as np

class VField():
    # Closed Derivative methods
    @staticmethod
    def closedFirstGradient(f: np.array, t: np.array = None) -> np.array:
        if t is None:
            t = np.linspace(0,1,f.shape[1],endpoint=False)

        fPlusOne = np.roll( f, -1, axis=1 )
        fMinusOne = np.roll( f, 1, axis=1 )
        
        # Define the time step for every value of t
        h = np.roll(t, -1) - t
        h[-1] = h[0]

        num = fPlusOne - fMinusOne
        den = 2 * h
        grad = np.divide(num,den)

        return grad
    
    @staticmethod
    def closedSecondGradient(f: np.array, t: np.array = None):
        if t is None:
            t = np.linspace(0,1,f.shape[1],endpoint=False)

        fPlusOne = np.roll( f, -1, axis=1 )
        fMinusOne = np.roll( f, 1, axis=1 )

        # Define the time step for every value of t
        h = np.roll(t, -1) - t
        h[-1] = h[0]

        num = fPlusOne - 2*f + fMinusOne
        den = np.power(h,2)
        grad = num/den

        return grad

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
    def KappaBinormal( curve: np.array, t: np.array = None ):
        if t is None:
            t = np.linspace(0,1,curve.shape[1])
        
        curveTangent = VField.closedFirstGradient(curve, t)
        curveNormal = VField.closedSecondGradient(curve, t)
        curveBinormal = np.cross(curveTangent,curveNormal, axisa=0, axisb=0, axisc=0)

        Kappa = np.divide( np.linalg.norm( curveBinormal, axis=0 ), np.power( np.linalg.norm(curveTangent, axis=0),3 ) )

        KBApprox = np.multiply(Kappa, curveBinormal)

        return KBApprox