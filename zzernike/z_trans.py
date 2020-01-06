'''
z_trans.py -- adaptations of the Zernike transform in L2

'''
# Numerical stuff
import numpy as np 

# Zernike polynomial definitions
from . import z_poly 

from time import time 

# Package utilities
from . import utils 

# Normalization factors for the first 10 Z transforms
z_trans_norm_10 = (2 * np.array([0.0, 1.0, 1.0, 2.0, 1.0, 
    2.0, 3.0, 3.0, 3.0, 3.0]) + 2) / np.pi
z_trans_norm_6 = z_trans_norm_10[:6] 

def fwd_10(image, center_yx, scale=4.0, unit_disk_only=True,
    center_int=False):
    """
    Forward Zernike transform for the first 10 coefficients.

    Profiling note: takes ~0.2 - 0.25 ms on image sizes 
    between 9 and 21 pixels square.

    args
    ----
        image :  2D ndarray
        center_yx :  (float, float), center of radial symmetry
        scale :  float, rescaling of unit disk
        unit_disk_only :  bool, only perform the transform
            on the unit disk (recommended)

    returns
        1D ndarray of shape (10,), the first 10 
            Zernike coefficients

    """
    # Define the coordinates
    Y, X = np.indices(image.shape)
    Y = Y.astype('float64') - center_yx[0]
    X = X.astype('float64') - center_yx[1]

    # Get the corresponding Zernike polynomials
    Z, R = z_poly.Z_through_10(Y, X, scale=scale,
        unit_disk_only=unit_disk_only, return_R=True)

    # Center the image intensities on the mean, if desired
    if center_int:
        if unit_disk_only:
            inside = R <= 1.0
            I_mean = image[inside].mean()
            _I = image - I_mean 
        else:
            _I = image - image.mean()
    else:
        _I = image 

    # Perform the transform
    result = np.empty(10, dtype = 'float64')
    result[0] = (Z[0,:,:] * image).sum() * z_trans_norm_10[0]
    result[1:] = (Z[1:,:,:] * _I).sum((1,2)) * z_trans_norm_10[1:]

    return result 

def fwd_6(image, center_yx, scale=4.0, unit_disk_only=True,
    center_int=False, subtract_bg=True):
    """
    Forward Zernike transform for the first 6 coefficients.

    args
    ----
        image :  2D ndarray
        center_yx :  (float, float), center of radial symmetry
        scale :  float, rescaling of unit disk
        unit_disk_only :  bool, only perform the transform
            on the unit disk (recommended)

    returns
        1D ndarray of shape (6,), the first 6 
            Zernike coefficients

    """
    if center_int and subtract_bg:
        raise RuntimeError("z_trans.fwd_6: cannot have both center_int and subtract_bg")

    # Define the coordinates
    Y, X = np.indices(image.shape)
    Y = Y.astype('float64') - center_yx[0]
    X = X.astype('float64') - center_yx[1]

    # Get the corresponding Zernike polynomials
    Z, R = z_poly.Z_through_6(Y, X, scale=scale,
        unit_disk_only=unit_disk_only, return_R=True)

    # Center the image intensities on the mean, if desired
    if center_int:
        if unit_disk_only:
            inside = R <= 1.0
            I_mean = image[inside].mean()
            _I = image - I_mean 
        else:
            _I = image - image.mean()
    elif subtract_bg:
        bg = utils.ring_mean(image)
        _I = image - bg
        _I[_I < 0] = 0
    else:
        _I = image 

    # Perform the transform
    # result = np.empty(6, dtype = 'float64')
    # result[0] = (Z[0,:,:] * image).sum() * z_trans_norm_6[0]
    # result[1:] = (Z[1:,:,:] * _I).sum((1,2)) * z_trans_norm_6[1:]

    result = (Z * _I).sum((1, 2)) * z_trans_norm_6

    return result 

def fwd_GS_10(image, center_yx, scale=4.0, unit_disk_only=True):
    """
    Obtain the coefficients for a sum of Zernike 
    polynomials that approximates the image about
    some target point, using the Gram-Schmidt
    process.

    Profiling note: takes ~0.3 - 0.4 ms for a (21, 21)
    image array.

    args
    ----
        image :  2D ndarray
        center_yx :  (int, int)
        scale :  float, size of unit disk to use
        unit_disk_only :  only try to approximate the 
            image inside the rescaled unit disk

    returns
    -------
        (
            1D ndarray, the first 10 coefficients,
            2D ndarray, residuals
        )

    """
    # Define coordinates
    Y, X = np.indices(image.shape)
    Y = Y.astype('float64') - center_yx[0]
    X = X.astype('float64') - center_yx[1]

    # Get corresponding Zernike polynomials
    Z, R = z_poly.Z_through_10(Y, X, scale=scale,
        unit_disk_only=unit_disk_only, return_R=True)

    I = image.copy().astype('float64')
    coefs = np.empty(10, dtype = 'float64')
    for z in range(10):
        coefs[z] = (I * Z[z,:,:] * R).sum()
        I -= coefs[z] * Z[z,:,:]

    return coefs, I

def inv_10(coefs, center_yx, shape, scale=4.0, unit_disk_only=True):

    assert len(coefs) == 10

    Y, X = np.indices(shape)
    Y -= center_yx[0]
    X -= center_yx[1]

    Z = z_poly.Z_through_10(Y, X, scale=scale, unit_disk_only=unit_disk_only)
    result = np.zeros(shape, dtype = 'float64')
    for j in range(10):
        result += coefs[j] * Z[j, :, :]

    return result 



