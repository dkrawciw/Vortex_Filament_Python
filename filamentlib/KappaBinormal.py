import numpy as np
from filamentlib.ClosedDerivatives import closedFirstGradient, closedSecondGradient


def KappaBinormal( curve: np.array, meshpoints: np.array ):
    # Finding the unit tangent vector
    dc = closedFirstGradient(curve, meshpoints)
    curveTangent = np.divide( dc, np.linalg.norm(dc, axis=0) )

    # Finding the unit normal vector
    d2c = closedSecondGradient(curve, meshpoints)
    curveNormal = np.divide( d2c, np.linalg.norm(d2c, axis=0) )

    # Using the unit tangent and unit normal vectors to find the binormal vector
    curveBinormal = np.cross(curveTangent,curveNormal, axisa=0, axisb=0, axisc=0)

    # Defining Kappa
    dTdt = closedFirstGradient(curveTangent,meshpoints)
    Kappa = np.divide( np.linalg.norm( dTdt,  axis=0 ), np.linalg.norm( dc, axis=0 ) )
    KBApprox = np.multiply(Kappa, curveBinormal)

    return KBApprox