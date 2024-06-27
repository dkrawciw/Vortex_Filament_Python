# Vortex Filament Simulation

Creating a Vortex Filament Simulation using Python

## How to use the library

Functions:

- BiotSavart( curve, curveTangent, fieldPoints )
- RegularBiotSavart( curve, curveTangent, fieldPoints, *epsilon* )
- KappaBinormal( curve, meshpoints )
- EvolveKappaBinormal( curve, meshpoints, tspan, *method* )
- EvolveRegularBiotSavart( curve, meshpoints, eps, tspan, *method* )

### EvolveKappaBinormal

Example Code:

    from filamentlib import EvolveKappaBinormal

    # Creating the initial curve with 100 initial meshpoints from -pi to pi
    meshpoints = np.linspace(-np.pi,np.pi,100,endpoint=False)  
    f = lambda t: [ np.cos(t), np.sin(t), t*0 ]  
    curve = np.array( f(meshpoints) )

    # Integrator Settings
    tspan = [0,10]  
    method = "rk23"     # By default the method is rk23  

    evolvedCurve = EvolveKappaBinormal( curve, meshpoints, tspan, method )

