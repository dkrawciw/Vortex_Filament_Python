import numpy as np
from KappaBinormal import KappaBinormal
from scipy.integrate import solve_ivp

# Functions to update a curve over time
# Euler's method on Kappa Binormal
def EulerKappaBinormal( curve:np.array, meshpoints: np.array, deltaStep: float, maxSteps: int):
    curves = np.zeros( (maxSteps + 1,curve.shape[0],curve.shape[1]) )
    curves[0,:,:] = curve

    for i in range(1,maxSteps+1):
        curves[i,:,:] = curves[i-1, :, :] + deltaStep * KappaBinormal(curves[i-1,:,:], meshpoints)
    return curves

def solve_ivp_KappaBinormal(curve:np.array, meshpoints:np.array) -> np.array:
    curve = curve.reshape( (3, meshpoints.shape[0]) )
    KBApprox = KappaBinormal( curve, meshpoints )

    return KBApprox.flatten()

def EvolveKappaBinormal( curve:np.array, meshpoints: np.array, tspan: list, method: str = 'RK45' ) -> np.array: 
    # Prepare the function and the initial condition to be put through scipy.integrate.solve_ivp
    y0 = curve.flatten()
    KBEqn = lambda t,y: solve_ivp_KappaBinormal( y, meshpoints )

    # solve the system
    soln = solve_ivp( KBEqn, tspan, y0, method=method )

    # Get the simulation curves to be in a standard format
    curves = np.zeros( (soln.y.shape[1], curve.shape[0], curve.shape[1]) )
    for i in range( soln.y.shape[1] ):
        curves[i,:,:] = soln.y[:,i].reshape( curve.shape )

    return curves