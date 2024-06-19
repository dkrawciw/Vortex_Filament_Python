import numpy as np

# Calculating the closed first derivative of a bunch of 3D points
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


# Calculating the closed second derivative of a bunch of 3D points
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