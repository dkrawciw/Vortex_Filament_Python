import numpy as np
from scipy.integrate import solve_ivp

class VField():
    # Closed Derivative methods
    @staticmethod
    def __closedFirstGradient(f: np.array, t: np.array = None) -> np.array:
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
    def __closedSecondGradient(f: np.array, t: np.array = None):
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
    def KappaBinormal( curve: np.array, meshpoints: np.array ):
        # Finding the unit tangent vector
        dc = VField.__closedFirstGradient(curve, meshpoints)
        curveTangent = np.divide( dc, np.linalg.norm(dc, axis=0) )

        # Finding the unit normal vector
        d2c = VField.__closedSecondGradient(curve, meshpoints)
        curveNormal = np.divide( d2c, np.linalg.norm(d2c, axis=0) )

        # Using the unit tangent and unit normal vectors to find the binormal vector
        curveBinormal = np.cross(curveTangent,curveNormal, axisa=0, axisb=0, axisc=0)

        # Defining Kappa
        dTdt = VField.__closedFirstGradient(curveTangent,meshpoints)
        Kappa = np.divide( np.linalg.norm( dTdt,  axis=0 ), np.linalg.norm( dc, axis=0 ) )
        KBApprox = np.multiply(Kappa, curveBinormal)

        return KBApprox
    
    # Functions to update a curve over time
    # Euler's method on Kappa Binormal
    @staticmethod
    def EulerKappaBinormal( curve:np.array, meshpoints: np.array, deltaStep: float, maxSteps: int):
        curves = np.zeros( (maxSteps + 1,curve.shape[0],curve.shape[1]) )
        curves[0,:,:] = curve

        for i in range(1,maxSteps+1):
            curves[i,:,:] = curves[i-1, :, :] + deltaStep * VField.KappaBinormal(curves[i-1,:,:], meshpoints)
        return curves
    
    @staticmethod
    def __solve_ivp_KappaBinormal(curve:np.array, meshpoints:np.array) -> np.array:
        
        curve = curve.reshape( (3, meshpoints.shape[0]) )
        KBApprox = VField.KappaBinormal( curve, meshpoints )

        return KBApprox.flatten()

    @staticmethod
    def SolveKappaBinormal( curve:np.array, meshpoints: np.array, tspan: list, method: str = 'RK45' ) -> np.array:
        
        # Prepare the function and the initial condition to be put through scipy.integrate.solve_ivp
        y0 = curve.flatten()
        KBEqn = lambda t,y: VField.__solve_ivp_KappaBinormal( y, meshpoints )

        # solve the system
        soln = solve_ivp( KBEqn, tspan, y0, method=method )

        # Get the simulation curves to be in a standard format
        curves = np.zeros( (soln.y.shape[1], curve.shape[0], curve.shape[1]) )
        for i in range( soln.y.shape[1] ):
            curves[i,:,:] = soln.y[:,i].reshape( curve.shape )

        return curves