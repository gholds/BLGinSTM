import numpy as np
from scipy import optimize

def Newton2D(f1,f2,x0,tol = 0.001, max_iter=50):
    '''f1 and f2 are functions which take an array of length 2 as argument. x0 is a length 2 array.'''
    def F(x):
        return np.array([f1(x),f2(x)])
    
    # Jacobian is computed numerically
    def J(x):
        epsilon = np.array([tol,tol])
        f1p = optimize.approx_fprime(x,f1,epsilon)
        f2p = optimize.approx_fprime(x,f2,epsilon)
        return np.stack((f1p,f2p))
    
    def iterate(x):
        return x - np.asarray(np.dot(np.linalg.inv(J(x)),F(x)[:,np.newaxis]).T)[0]

    x1 = iterate(x0)
    
    iterations = 1
    while np.linalg.norm(x1-x0) > tol:
        x0 = x1
        x1 = iterate(x0)
        iterations +=1
        
    
    return x1