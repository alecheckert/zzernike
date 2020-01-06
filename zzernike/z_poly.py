'''
z_poly.py -- Zernike polynomial definitions
and related utilities.

INDEXING

The Zernike polynomials here are indexed using 
the OSA/ANSI standard indices. The polynomial
of coefficients (n, m) is indexed by a single
integer j such that

    j = (n*(n+2) + m) / 2

The first ten, which are primarily used in
this package, are

    j   m   n
    0   0   0

    1   -1  1
    2   +1  1

    3   -2  2
    4   0   2
    5   +2  2

    6   -3  3
    7   -1  3
    8   +1  3
    9   +3  3
    ...

FUNCTION CLASSES

The functions here are divided into two classes :

    1. Functions that derive each Zernike polynomial
        from scratch, given a set of Y- and X- 
        coordinates

    2. Functions that compute each Zernike polynomial
        from scratch, given a set of radial coordinates
        (R and theta)

    3. Functions that reuse calculations to calculate
        the first few Zernike polynomials quickly

The first kind are always written 'Z%d' % j where j
is the OSA/ANSI index for the polynomial. The second
kind are always written 'Zr%d' % j and the third 
kind are given special names.

ARGUMENTS

Throughout, *Y* and *X* refer to 2D ndarrays.

'''
# Numerical tools
import numpy as np 
import math

# Package utilites
from . import utils 

from time import time 

# Useful constants
sqrts = np.sqrt(np.arange(20))

#
# Functions that take Cartesian coordinates as arguments
#
def Zj(Y, X, j, scale=1.0, unit_disk_only=False):
    """General Zernike polynomial of OSA coefficient j"""

    # Get the corresponding Zernike coefficients
    m, n = utils.osa_to_z_index(j)
    m_abs = np.abs(m)

    # Rescale, if desired
    if scale != 1.0:
        Y_re = Y / scale 
        X_re = X / scale 
    else:
        Y_re = Y 
        X_re = X 

    # Radial part
    R2 = Y_re**2 + X_re**2
    R = np.sqrt(R2)
    radial_part = np.zeros(Y.shape, dtype = 'float64')
    c0 = int((n-m_abs) / 2)
    c1 = int((n+m_abs) / 2)
    for k in range(c0+1):
        R_p = np.power(R, n-2*k)
        radial_part += ((-1)**k) * np.power(R, n-2*k) * math.factorial(n-k) / \
            (math.factorial(k) * math.factorial(c1-k) * math.factorial(c0-k))

    # Angular part 
    theta = utils.angle(Y_re, X_re, R=R)
    if m == 0:
        angular_part = np.ones(Y.shape, dtype = 'float64')
    else:
        theta = utils.angle(Y_re, X_re, R=R)
        if m < 0:
            angular_part = np.sin(m_abs * theta)
        else:
            angular_part = np.cos(m_abs * theta)

    # Confine to the (rescaled) unit disk if desired
    if unit_disk_only:
        outside = R > 1.0
        radial_part[outside] = 0

    return radial_part * angular_part 

def Z0(Y, X):
    """Zernike polynomial of coefficient (0,0)"""
    return np.ones(Y.shape, dtype = 'float64')

def Z1(Y, X):
    """Zernike polynomial of coefficient (-1,1)"""
    return 2.0*Y

def Z2(Y, X):
    """Zernike polynomial of coefficient (+1,1)"""
    return 2.0*X

def Z3(Y, X):
    """Zernike polynomial of coefficient (-2,2)"""
    R2 = Y**2 + X**2
    theta = utils.angle(Y, X, R=np.sqrt(R2))
    return np.sqrt(6) * R2 * np.sin(2*theta)

def Z4(Y, X):
    """Zernike polynomial of coefficient (0,2)"""
    R2 = Y**2 + X**2
    return np.sqrt(3) * (2 * R2 - 1)

def Z5(Y, X):
    """Zernike polynomial of coefficient (+2,2)"""
    R2 = Y**2 + X**2
    theta = utils.angle(Y, X, R=np.sqrt(R2))
    return np.sqrt(6) * R2 * np.cos(2*theta)

def Z6(Y, X):
    """Zernike polynomial of coefficient (-3,3)"""
    R2 = Y**2 + X**2
    R = np.sqrt(R2)
    R3 = R**3
    theta = utils.angle(Y, X, R=R)
    return np.sqrt(8) * R3 * np.sin(3*theta)

def Z7(Y, X):
    """Zernike polynomial of coefficient (-1,3)"""
    R2 = Y**2 + X**2
    R = np.sqrt(R2)
    R3 = R**3
    theta = utils.angle(Y, X, R=R)
    return np.sqrt(8) * (3*R3-2*R) * np.sin(theta)

def Z8(Y, X):
    """Zernike polynomial of coefficient (+1,3)"""
    R2 = Y**2 + X**2
    R = np.sqrt(R2)
    R3 = R**3
    theta = utils.angle(Y, X, R=R)
    return np.sqrt(8) * (3*R3-2*R) * np.cos(theta)

def Z9(Y, X):
    """Zernike polynomial of coefficient (-3,3)"""
    R2 = Y**2 + X**2
    R = np.sqrt(R2)
    R3 = R**3
    theta = utils.angle(Y, X, R=R)
    return np.sqrt(8) * R3 * np.cos(3*theta)

# 
# Functions that reuse calculations for multiple
# Zernike polynomials
#

def Z_through_10(Y, X, scale=1.0, unit_disk_only=False, return_R=False):
    """
    Calculate the first ten Zernike polynomials.

    args
    ----
        Y, X :  2D ndarrays, the y- and x- coordinates
        scale :  float, rescaling of unit disk
        unit_disk_only :  bool, set coordinates outside
            the rescaled unit disk to 0
        return_R :  bool, also return the radius from
            the origin

    returns
    -------
        (
            3D ndarray of shape (10, size_y, size_x), the first ten
                Zernike polynomials;

            [if return_R] 2D ndarray of shape (size_y, size_x), the
                radius from the origin
        )

    """
    # Rescale field if necessary
    if scale != 1.0:
        Y_re = Y / scale 
        X_re = X / scale
    else:
        Y_re = Y 
        X_re = X 

    # Cartesian fields
    Y2 = Y_re ** 2
    X2 = X_re ** 2
    Y3 = Y_re ** 3
    X3 = X_re ** 3

    # Radial fields
    R2 = Y2 + X2 
    R = np.sqrt(R2)

    # Reusable factors
    X_R2 = X_re * R2 
    Y_R2 = Y_re * R2 


    # Out array
    result = np.empty((10, Y.shape[0], Y.shape[1]), dtype = 'float64')

    # Z[0,0]
    result[0,:,:] = 1.0

    # Z[-1,1]
    result[1,:,:] = 2 * Y_re

    # Z[+1,1]
    result[2,:,:] = 2 * X_re 

    # Z[-2,2]
    result[3,:,:] = 2 * sqrts[6] * Y_re * X_re 

    # Z[0,2]
    result[4,:,:] = sqrts[3] * (2 * R2 - 1)

    # Z[+2,2]
    result[5,:,:] = sqrts[6] * (2 * X2 - R2)

    # Z[-3,3]
    result[6,:,:] = sqrts[8] * (3 * Y_R2 - 4 * Y3)

    # Z[-1,3]
    result[7,:,:] = sqrts[8] * (3 * Y_R2 - 2 * Y_re)

    # Z[+1,3]
    result[8,:,:] = sqrts[8] * (3 * X_R2 - 2 * X_re)

    # Z[+3,3]
    result[9,:,:] = sqrts[8] * (4 * X3 - 3 * X_R2)

    # Only take values inside the unit disk, if desired
    if unit_disk_only:
        outside = (R > 1.0).nonzero()
        result[:, outside[0], outside[1]] = 0

    if return_R:
        return result, R 
    else:
        return result 

def Z_through_6(Y, X, scale=1.0, unit_disk_only=False, return_R=False):
    """
    Calculate the first six Zernike polynomials.

    args
    ----
        Y, X :  2D ndarrays, the y- and x- coordinates
        scale :  float, rescaling of unit disk
        unit_disk_only :  bool, set coordinates outside
            the rescaled unit disk to 0
        return_R :  bool, also return the radius from
            the origin

    returns
    -------
        (
            3D ndarray of shape (10, size_y, size_x), the first six
                Zernike polynomials;

            [if return_R] 2D ndarray of shape (size_y, size_x), the
                radius from the origin
        )

    """
    # Rescale field if necessary
    if scale != 1.0:
        Y_re = Y / scale 
        X_re = X / scale
    else:
        Y_re = Y 
        X_re = X 

    # Cartesian fields
    Y2 = Y_re ** 2
    X2 = X_re ** 2

    # Radial fields
    R2 = Y2 + X2 
    R = np.sqrt(R2)

    # Out array
    result = np.empty((6, Y.shape[0], Y.shape[1]), dtype = 'float64')

    # Z[0,0]
    result[0,:,:] = 1.0

    # Z[-1,1]
    result[1,:,:] = 2 * Y_re

    # Z[+1,1]
    result[2,:,:] = 2 * X_re 

    # Z[-2,2]
    result[3,:,:] = 2 * sqrts[6] * Y_re * X_re 

    # Z[0,2]
    result[4,:,:] = sqrts[3] * (2 * R2 - 1)

    # Z[+2,2]
    result[5,:,:] = sqrts[6] * (2 * X2 - R2)

    # Only take values inside the unit disk, if desired
    if unit_disk_only:
        outside = (R > 1.0).nonzero()
        result[:, outside[0], outside[1]] = 0

    if return_R:
        return result, R 
    else:
        return result 



















