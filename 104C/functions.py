import numpy as np

def Normal_Eq(A, b):
    '''
    Finds the least squares solution by solving the Normal Equation
    Inputs:
    A - 2D array, matrix in Ax=b
    b - 1D array, output vector 
    Output:
    c - least squares solution
    '''
    c = np.linalg.solve(A.T @ A, A.T @ b)
    
    return c

def reduced_QR(A):
    '''
    Returns the reduced QR factorization of the matrix
    Input:
    A - 2D array, original matrix
    Outputs:
    Q, R - 2D arrays, corresponding QR factorization
    '''
    # create Q, R
    (m, n) = A.shape
    Q = np.zeros((m,n)) # size of the A matrix
    R = np.zeros((n,n)) # nxn upper triangular matrix 

    # full QR factorization
    for j in range(n):
        y = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], y)
            y = y - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]
    
    # using np.linalg.qr for more efficient computations
    #Q, R = np.linalg.qr(A)

    return Q, R


def QR(A):
    '''
    Finds the least squares solution using full QR factorization
    Inputs:
    A - 2D array, matrix in Ax=b
    b - 1D array, output vector
    Output:
    c - least squares solution
    '''
    # pad A with standard basis vectors
    (m, n) = A.shape
    A_ = np.eye(m)
    A_[:, :n] = A

    # create Q, R 
    Q = np.zeros((m, n))
    R = np.zeros((m, n))

    # full QR factorization
    '''for j in range(n):
        y = A[:, j].copy()
        for i in range(j):
            R[i, j] = np.dot(Q[:, i], y)
            y = y - R[i, j] * Q[:, i]
        R[j, j] = np.linalg.norm(y)
        Q[:, j] = y / R[j, j]'''
    
    # using np.linalg.qr for more efficient computations
    Q, R = np.linalg.qr(A_)

    # extract necessary information to solve for LSS
    '''R_ = R[:n, :n]
    d_ = (Q.T @ b)[:n]

    # solve
    c = np.zeros(n)
    c = np.linalg.solve(R_, d_)'''

    return Q, R

# helper function for Householder 
def HH_QR(A):
    '''
    Returns the QR decompoisiton using the Householder reflector
    Input:
    A - 2D array matrix to perform QR factorization
    Output:
    Q - 2D array, orthogonal matrix
    R - 2D array, upper triangular matrix
    '''
    (m, n) = A.shape
    R = A.copy()
    Q = np.eye(m)

    for i in range(n):
        k = m - i

        # create vectors
        x = R[i:, i]
        w = np.zeros_like(x)
        w[0] = -1 * np.sign(x[0]) * np.linalg.norm(x)
        v = w - x
        
        H = np.eye(m)
        H[i:, i:] = np.eye(k) - (2. * np.outer(v,v)/np.inner(v,v))

        R = H @ R
        Q = Q @ H

        '''H = np.eye(k) - (2 * np.outer(v,v)/np.inner(v,v))
        R[i:, i:] = H @ R[i:, i:]
        Q[:, i:] = Q[:, i:] @ H'''

    return Q, R

def Householder(Q, R, b):
    '''
    Using the QR factorization of the matrix frm HH_QR, returns the least squares solution
    Inputs:
    Q, R - 2D arrays , QR factorization of A in overdetermined system
    b - output vector in system
    Output:
    c - lease squares solution
    '''
    
    (m, n) = R.shape
    #Q, R = HH_QR(A)

    R_ = R[:n, :n]
    d_ = (Q.T @ b)[:n]

    c = np.linalg.solve(R_, d_)
    return c  