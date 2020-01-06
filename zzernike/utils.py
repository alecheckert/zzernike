'''
utils.py

'''
import numpy as np 

def angle(Y, X, R=None):
    """ 
    Return the angle of a set of coordinates with respect
    to (0, 0).

    args
    ----
        y_field, x_field, r_field :  2D ndarrays

    returns
    -------
        2D ndarray, the angle of each point

    """
    result = np.empty(Y.shape, dtype = 'float64')
    x_non = X >= 0

    # Calculate the radius if not provided
    if (R is None):
        R = np.sqrt((Y**2) + (X**2))

    # Determine the angle of each point
    result[x_non] = np.arcsin(Y[x_non] / R[x_non])
    result[~x_non] = np.pi - np.arcsin(Y[~x_non] / R[~x_non])

    # Set bad values to 0 (usually when r == 0)
    result[np.isnan(result)] = 0

    return result 

def ring_mean(image):
    """
    Return the mean of the outer ring of pixels
    in an image, useful for estimating BG.

    args
    ----
        image :  2D ndarray

    returns
    -------
        float, mean of the outer ring of pixels

    """
    return np.array([
        image[0,:-1].mean(),
        image[:-1,-1].mean(),
        image[-1,1:].mean(),
        image[1:,0].mean()
    ]).mean()

def z_to_osa_index(m, n):
    """Convert a 2D Zernike polynomial index (m, n) 
    to a 1D OSA/ANSI standard index j"""
    return int((n * (n + 2) + m) / 2)

def osa_to_z_index(j):
    """Convert a 1D OSA/ANSI standard index j to
    a 2D Zernike polynomial index (m, n)"""
    n = 0
    j_copy = j 
    while j_copy > 0:
        n += 1
        j_copy -= (n+1)
    m = 2 * j - n * (n + 2) 
    return m, n 

#
# Localization utilities
#

def expand_window(image, N, M):
    '''
    Pad an image with zeros to force it into the
    shape (N, M), keeping the image centered in 
    the frame.

    args
        image       :   2D np.array with shape (N_in, M_in)
                        where both N_in < N and M_in < M

        N, M        :   floats, the desired image dimensions

    returns
        2D np.array of shape (N, M)

    '''
    N_in, M_in = image.shape
    out = np.zeros((N, M))
    nc = np.floor(N/2 - N_in/2).astype(int)
    mc = np.floor(M/2 - M_in/2).astype(int)
    out[nc:nc+N_in, mc:mc+M_in] = image
    return out

def local_max_2d(image):
    '''
    Determine local maxima in a 2D image.

    Returns 2D np.array of shape (image.shape), 
    a Boolean image of all local maxima
    in the image.

    '''
    N, M = image.shape
    ref = image[1:N-1, 1:M-1]
    pos_max_h = (image[0:N-2, 1:M-1] < ref) & (image[2:N, 1:M-1] < ref)
    pos_max_v = (image[1:N-1, 0:M-2] < ref) & (image[1:N-1, 2:M] < ref)
    pos_max_135 = (image[0:N-2, 0:M-2] < ref) & (image[2:N, 2:M] < ref)
    pos_max_45 = (image[2:N, 0:M-2] < ref) & (image[0:N-2, 2:M] < ref)
    peaks = np.zeros((N, M))
    peaks[1:N-1, 1:M-1] = pos_max_h & pos_max_v & pos_max_135 & pos_max_45
    return peaks.astype('bool')


def gaussian_model(sigma, window_size, offset_by_half = False):
    '''
    Generate a model Gaussian PSF in a square array
    by sampling the value of the Gaussian in the center 
    of each pixel. (**for detection**)

    args
        sigma       :   float, xy sigma
        window_size :   int, pixels per side

    returns
        2D np.array, the PSF model

    '''
    half_w = int(window_size) // 2
    ii, jj = np.mgrid[-half_w:half_w+1, -half_w:half_w+1]
    if offset_by_half:
        ii = ii - 0.5
        jj = jj - 0.5
    sig2 = sigma ** 2
    g = np.exp(-((ii**2) + (jj**2)) / (2 * sig2)) / sig2 
    return g 

def polynomial_derivative(C):
    """
    Given a coefficient matrix describing an n-dimensional
    polynomial of order m, return the coefficient matrix
    that gives its first derivative.

    Simple and not very interesting m = 2 case:

        C = [[c00, c01],
             [c10, c11]]

        [[x0(z)],  =   [[c00, c01],  *  [[1],
         [x1(z)]]       [c10, c11]]      [z]]

    Then

        [[dx0(z)/dz],  =  [[c01],  *  [[1,]]
         [dx1(z)/dx]]      [c11]]

    So the derivative matrix is [[c01], [c11]].

    args
    ----
        C :  2D ndarray of shape (n_dims, m_coefs)

    returns
    -------
        2D ndarray of shape (n_dims, m_coefs-1)

    """
    return C[:,1:]*np.arange(1, C.shape[1])

def polynomial_second_derivative(C):
    """
    As with above, but for the second derivative.

    args
    ----
        C :  2D ndarray of shape (n_dims, m_coefs)

    returns
    -------
        2D ndarray of shape (n_dims, m_coefs-1)

    """
    return C[:,2:] * (np.arange(2, C.shape[1]) * \
        np.arange(1, C.shape[1]-1))


def polynomial_model(z, c0, c1, c2, c3, c4, c5):
    """
    5th order polynomial model on vector argument.

    args
    ----
        z :  1D ndarray
        poly_coefs :  1D ndarray, polynomial
            coefficients

    returns
    -------
        1D ndarray, the result for each z

    """
    c = np.asarray([c0, c1, c2, c3, c4, c5])
    Z = np.power(np.asarray([z]).T, np.arange(6)).T 
    return c.dot(Z)

