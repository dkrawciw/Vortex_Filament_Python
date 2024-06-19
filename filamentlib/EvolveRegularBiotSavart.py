import numpy as np
from scipy.integrate import solve_ivp

from ClosedDerivatives import closedFirstGradient
from RegularBiotSavart import RegularBiotSavart

# Method to flatten and unflatten inputted lists of curve points
# Definitely really inefficient but I haven't thought of a better way to use scipy's solve_ivp method without flattening the matrix
def solve_ivp_BiotSavart( curve:np.array, meshpoints:np.array, eps:float ) -> np.array:
        curve = curve.reshape( (3,meshpoints.shape[0]) )
        BSStep = RegularBiotSavart( curve, closedFirstGradient(curve, meshpoints), curve, eps )

        return BSStep.flatten()

# Method to 
def EvolveRegularBiotSavart( curve:np.array, meshpoints: np.array, eps:float, tspan: list, method: str = 'RK45' ) -> np.array:
        y0 = curve.flatten()
        BSEqn = lambda t,y: solve_ivp_BiotSavart( y, meshpoints, eps )

        soln = solve_ivp( BSEqn, tspan, y0, method )

        # Get the simulation curves to be in a standard format
        curves = np.zeros( (soln.y.shape[1], curve.shape[0], curve.shape[1]) )
        for i in range( soln.y.shape[1] ):
            curves[i,:,:] = soln.y[:,i].reshape( curve.shape )

        return curves