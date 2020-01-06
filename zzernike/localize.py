'''
localize.py -- localization utilities with computation of 
Zernike transforms for PSFs

'''
# Numerics
import numpy as np 

# File reader
from nd2reader import ND2Reader 

# I/O
import sys
import os

# Profiling
from time import time

# Dataframes
import pandas as pd 

# Progress bar
from tqdm import tqdm 

# Uniform kernel filter, for smoothing a PSF image
from scipy.ndimage import uniform_filter 

# Warnings filter for radial_symmetry method; the
# errors are divide-by-zero and are caught in a 
# subsequent step
import warnings

# Package utilities
from . import utils 

# Zernike transforms
from . import z_trans 

# Package visualization tools
from . import visualize 

def localize_biplane(
    nd2_path,
    detect_sigma = 1.0,
    detect_threshold = 20.0,
    detect_window_size = 9,
    fit_window_size = 15,
    scale = 5.0,
    unit_disk_only = True,
    center_int = False,
    subtract_bg = True,
    start_frame = None,
    stop_frame = None,
):
    '''
    Use the radial symmetry method to detect and localize
    spots in both channels of an ND2 file. Compute the
    first six Zernike coefficients for each localization.

    args
        nd2_path : str
        out_txt : str
        detect_sigma : float, kernel size for spot detection
        detect_threshold : float
        window_size : int
        start_frame, stop_frame : int

    returns
        (pandas.DataFrame, pandas.DataFrame), localizations
            in each channel

    '''
    # Make an image file reader
    reader = ND2Reader(nd2_path)
    n_frames = reader.metadata['total_images_per_channel']
    N = reader.metadata['height']
    M = reader.metadata['width']

    # Compute the kernels for spot detection
    g = utils.gaussian_model(detect_sigma, detect_window_size)
    gc = g - g.mean() 
    gaussian_kernel = utils.expand_window(gc, N, M)
    gc_rft = np.fft.rfft2(gaussian_kernel)

    # Compute some required normalization factors for spot detection
    n_pixels = detect_window_size ** 2
    Sgc2 = (gc ** 2).sum()
    half_w = fit_window_size // 2

    if start_frame is None:
        start_frame = 0
    if stop_frame is None:
        stop_frame = n_frames - 1

    # Do the same for both channels
    output_dfs = []
    for channel_idx in [0, 1]:

        print('Localizing %s in channel %d:' % (nd2_path, channel_idx))

        # Initialize output array
        locs = np.zeros((1000000, 13), dtype = 'float64')

        # Current localization index
        c_idx = 0

        for frame_idx in tqdm(range(start_frame, stop_frame + 1)):

            # Get the image corresponding to this frame
            image = reader.get_frame_2D(t = frame_idx, c = channel_idx).astype('float64')

            # Perform the convolutions required for the LL detection test
            A = uniform_filter(image, detect_window_size) * n_pixels
            B = uniform_filter(image**2, detect_window_size) * n_pixels
            im_rft = np.fft.rfft2(image)
            C = np.fft.ifftshift(np.fft.irfft2(im_rft * gc_rft))

            # Calculate the likelihood of a spot in each pixel,
            # and set bad values to 1.0 (i.e. no chance of detection)
            L = 1 - (C**2) / (Sgc2*(B - (A**2)/float(n_pixels)))
            L[:half_w,:] = 1.0
            L[:,:half_w] = 1.0
            L[-(1+half_w):,:] = 1.0
            L[:,-(1+half_w):] = 1.0
            L[L <= 0.0] = 0.001

            # Calculate log likelihood of the presence of a spot
            LL = -(n_pixels / 2) * np.log(L)

            # Find pixels that pass the detection threshold
            detections = LL > detect_threshold 

            # For each detection that consists of adjacent pixels,
            # take the local maximum
            peaks = utils.local_max_2d(LL) & detections

            # Find the coordinates of the detections
            detected_positions = np.asarray(np.nonzero(peaks)).T + 1
            
            # Copy the detection information to the result array
            n_detect = detected_positions.shape[0]
            locs[c_idx : c_idx+n_detect, 0] = frame_idx 
            locs[c_idx : c_idx+n_detect, 1:3] = detected_positions.copy()
            locs[c_idx : c_idx+n_detect, 3] = LL[np.nonzero(peaks)]
            
            # For each detection, run subpixel localization
            for d_idx in range(n_detect):
                yd, xd = locs[c_idx, 1:3].astype('uint16')
                psf_image = image[yd-half_w : yd+half_w+1, xd-half_w : xd+half_w+1]

                if psf_image.shape[0] != psf_image.shape[1]:
                    locs[c_idx, 6] = 1
                else:
                    y_rs, x_rs = radial_symmetry(psf_image)
                    locs[c_idx, 4:6] = np.array([y_rs, x_rs]) + locs[c_idx, 1:3] - half_w 

                    # Compute Zernike transform
                    locs[c_idx, 7:13] = z_trans.fwd_6(
                        psf_image,
                        (y_rs, x_rs),
                        scale = scale,
                        unit_disk_only = unit_disk_only,
                        center_int = center_int,
                        subtract_bg = subtract_bg,
                    ).copy()

                c_idx += 1

        locs = locs[:c_idx, :]

        # Format output and enforce some typing
        locs = pd.DataFrame(locs, columns = [
            'frame_idx', 'y_detect', 'x_detect', 'llr', 'y_pixels', 'x_pixels',
            'error_code', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5'
        ])
        locs['frame_idx'] = locs['frame_idx'].astype('int64')
        locs['y_detect'] = locs['y_detect'].astype('int64')
        locs['x_detect'] = locs['x_detect'].astype('int64')
        locs['error_code'] = locs['error_code'].astype('uint16')

        output_dfs.append(locs.copy())

    reader.close()

    return output_dfs

def radial_symmetry(psf_image):
    '''
    Use the radial symmetry method to estimate the center
    y and x coordinates of a PSF.

    This method was originally conceived by
    Parasarathy R Nature Methods 9, pgs 724–726 (2012).

    args
        image   :   2D np.array (ideally small and symmetric,
                    e.g. 9x9 or 13x13), the image of the PSF.
                    Larger frames relative to the size of the PSF
                    will reduce the accuracy of the estimation.

    returns
        np.array([y_estimate, x_estimate]), the estimated
            center of the PSF in pixels, relative to
            the corner of the image frame.

    '''
    # Get the size of the image frame and build
    # a set of pixel indices to match
    N, M = psf_image.shape
    N_half = N // 2
    M_half = M // 2
    ym, xm = np.mgrid[:N-1, :M-1]
    ym = ym - N_half + 0.5
    xm = xm - M_half + 0.5 
    
    # Calculate the diagonal gradients of intensities across each
    # corner of 4 pixels
    dI_du = psf_image[:N-1, 1:] - psf_image[1:, :M-1]
    dI_dv = psf_image[:N-1, :M-1] - psf_image[1:, 1:]
    
    # Smooth the image to reduce the effect of noise, at the cost
    # of a little resolution
    fdu = uniform_filter(dI_du, 3)
    fdv = uniform_filter(dI_dv, 3)
    
    dI2 = (fdu ** 2) + (fdv ** 2)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        m = -(fdv + fdu) / (fdu - fdv)
        
    # For pixel values that blow up, instead set them to a very
    # high float
    m[np.isinf(m)] = 9e9
    
    b = ym - m * xm

    sdI2 = dI2.sum()
    ycentroid = (dI2 * ym).sum() / sdI2
    xcentroid = (dI2 * xm).sum() / sdI2
    w = dI2 / np.sqrt((xm - xcentroid)**2 + (ym - ycentroid)**2)

    # Correct nan / inf values
    w[np.isnan(m)] = 0
    b[np.isnan(m)] = 0
    m[np.isnan(m)] = 0

    # Least-squares analytical solution to the point of 
    # maximum radial symmetry, given the slopes at each
    # edge of 4 pixels
    wm2p1 = w / ((m**2) + 1)
    sw = wm2p1.sum()
    smmw = ((m**2) * wm2p1).sum()
    smw = (m * wm2p1).sum()
    smbw = (m * b * wm2p1).sum()
    sbw = (b * wm2p1).sum()
    det = (smw ** 2) - (smmw * sw)
    xc = (smbw*sw - smw*sbw)/det
    yc = (smbw*smw - smmw*sbw)/det

    # Adjust coordinates so that they're relative to the
    # edge of the image frame
    yc = (yc + (N + 1) / 2.0) - 1
    xc = (xc + (M + 1) / 2.0) - 1

    # Add 0.5 pixel shift to get back to the original indexing.
    # This is not necessarily the desired behavior, so I've
    # commented out this.
    # fit_vector = np.array([yc, xc]) + 0.5
    fit_vector = np.array([yc, xc])

    return fit_vector

def distance_to_polynomial(
    point,
    C,
    guess_z,
    damp = 0.2,
    max_iter = 50,
    convergence_crit = 1.0e-3,
    plot = False,
):
    """
    Determine the Euclidean distance of an n-dimensional
    point from an n-dimensional, (m-1)^th order polynomial using
    Newton's method.

    Here, C is a matrix with the coefficients of the
    polynomial, so that

        point[0] = C[0,:] @ [1, z, z^2, z^3, ..., z^(m-1)]
        point[1] = C[1,:] @ [1, z, z^2, z^3, ..., z^(m-1)]
        ...

    Profiling note: With max_iter = 50, usually runs
    in <1 ms for a fifth-order polynomial, even with 
    computing all of the matrices each time.

    args
    ----
        point :  1D ndarray of shape (n,)
        C :  2D ndarray of shape (n, m), polynomial
            coefficients
        guess_z :  float, initial guess for z
        damp :  float, damping factor for stability
        max_iter :  int
        convergence_crit :  float
        plot :  bool, only for 2D

    returns
    -------
        float, the estimated z-position of the point

    """
    assert point.shape[0] == C.shape[0]

    n, m = C.shape 

    # Compute matrices for the first and second derivatives
    # of the polynomial
    C1 = utils.polynomial_derivative(C)
    C2 = utils.polynomial_second_derivative(C)

    # Initialize
    z_curr = guess_z 
    update = 1.0

    # Iterate until convergence or until max_iter is reached
    iter_idx = 0
    while (iter_idx < max_iter) and (np.abs(update) > convergence_crit):

        # Compute the vector of powers of z
        z_powers = np.power(z_curr, np.arange(m))

        # Compute the expected point indices and its
        # derivatives, given the current value of z
        x = C.dot(z_powers)
        dx_dz = C1.dot(z_powers[:-1])
        d2x_dz2 = C2.dot(z_powers[:-2])

        # Compute the first and second derivatives of the
        # squared distance D with respect to z
        dD_dz = -2 * ((point - x) * dx_dz).sum()
        d2D_dz2 = 2 * ((dx_dz**2) - (point - x) * d2x_dz2).sum()

        # Determine the update using Newton's method
        update = -dD_dz / d2D_dz2

        # Show for debugging, when necessary
        if plot:
            z_levels = np.arange(60)
            print(iter_idx, z_curr)
            visualize.plot_poly_with_point(C, z_levels, z_curr, point)

        # Update the current z position
        z_curr = z_curr + damp * update 
        iter_idx += 1

    return z_curr 

def match_locs(locs_0, locs_1, tol_radius=8.0, outlier_tol=8.0):
    """
    Match one set of localizations with another
    in a different channel.

    args
    ----
        locs_0, locs_1 :  pandas.DataFrame with localizations
            in each channel 
        tol_radius :  float, max radius in pixels
            to call the same spot

    returns
    -------
        2D ndarray of shape (n_matched_locs, 3) with columns

            result[:,0] -> frame index
            result[:,1] -> 

    """
    # Add localization index column
    locs_0['loc_idx'] = np.arange(len(locs_0), dtype = 'int64')
    locs_1['loc_idx'] = np.arange(len(locs_1), dtype = 'int64')

    # Convert to ndarray for speed
    positions_0 = np.asarray(locs_0[['frame_idx', 'loc_idx', 'y_pixels', 'x_pixels']])
    positions_1 = np.asarray(locs_1[['frame_idx', 'loc_idx', 'y_pixels', 'x_pixels']])

    # Get the size of the problem
    n_frames = int(max([positions_0[:,0].max(), positions_1[:,0].max()]))
    max_n_locs = int(max([len(locs_0), len(locs_1)]))

    # Initialize matches.
    # col 0 -> frame index
    # col 1 -> localization index in locs_0
    # col 2 -> localization index in locs_1
    matches = np.zeros((max_n_locs, 3), dtype = 'int64')

    # Current number of results
    c_idx = 0

    # Match localizations in each frame
    for frame_idx in tqdm(range(n_frames)):

        pos_0_in_frame = positions_0[positions_0[:,0] == frame_idx, :]
        pos_1_in_frame = positions_1[positions_1[:,0] == frame_idx, :]

        if (pos_0_in_frame.shape[0] == 0) or (pos_1_in_frame.shape[0] == 0):
            continue 

        distances = distance_matrix(pos_0_in_frame[:,2:], pos_1_in_frame[:,2:])
        in_radius = distances < tol_radius 

        indices_0, indices_1 = in_radius.nonzero()
        m = len(indices_0)
        matches[c_idx:c_idx+m, 0] = frame_idx 
        matches[c_idx:c_idx+m, 1] = pos_0_in_frame[indices_0, 1].copy()
        matches[c_idx:c_idx+m, 2] = pos_1_in_frame[indices_1, 1].copy()

        c_idx += m 

    # Truncate at the number of matched localizations
    matches = matches[:c_idx, :]

    # Make new dataframe with combined localizations
    new_index = np.arange(matches.shape[0], dtype = 'int64')
    locs_0_matched = locs_0.loc[matches[:,1]].set_index(new_index)
    locs_1_matched = locs_1.loc[matches[:,2]].set_index(new_index)

    # Add the shift to the first dataframe
    locs_0_matched['y_shift'] = locs_1_matched['y_pixels'] - locs_0_matched['y_pixels']
    locs_0_matched['x_shift'] = locs_1_matched['x_pixels'] - locs_0_matched['x_pixels']
    locs_0_matched['y_pixels_ch1'] = locs_1_matched['y_pixels']
    locs_0_matched['x_pixels_ch1'] = locs_1_matched['x_pixels']

    # Remove outliers
    y_shift_mean = locs_0_matched['y_shift'].mean()
    x_shift_mean = locs_0_matched['x_shift'].mean()
    locs_0_matched['y_shift_dev'] = locs_0_matched['y_shift'] - y_shift_mean
    locs_0_matched['x_shift_dev'] = locs_0_matched['x_shift'] - x_shift_mean

    locs_0_matched = locs_0_matched[np.abs(locs_0_matched['y_shift_dev']) < outlier_tol]
    locs_0_matched = locs_0_matched[np.abs(locs_0_matched['x_shift_dev']) < outlier_tol]

    return locs_0_matched 

def calculate_affine_field(matched_locs):
    """
    Given a set of matched localizations, find an affine 
    field matrix that maps the localizations in channel 
    0 to localizations in channel 1.

    If we call this matrix *AFM* and the offset *offset*,
    then

        pos_in_ch1 = (AFM + I).dot(pos_in_ch0) + offset

    args
    ----
        matched_locs :  pandas.DataFrame, perhaps output
            of match_locs()

    returns
    -------
        (
            2D ndarray of shape (2, 2), affine field matrix;
            1D ndarray of shape (2,), the offset vector
        )

    """
    # Extract affine field coefficients
    def affine_model(points, a, b, c):
        return a*points[:,0] + b*points[:,1] + c 

    popt_y, pcov = curve_fit(
        affine_model,
        np.asarray(locs_0_matched[['y_pixels', 'x_pixels']]),
        np.asarray(locs_0_matched['y_shift'])
    )
    popt_x, pcov = curve_fit(
        affine_model,
        np.asarray(locs_0_matched[['y_pixels', 'x_pixels']]),
        np.asarray(locs_0_matched['x_shift'])
    )   

    # Compare the actual observed y-/x- shifts with the affine
    # field model 
    if plot:
        locs_0_matched['y_shift_predicted'] = affine_model(
            np.asarray(locs_0_matched[['y_pixels', 'x_pixels']]),
            *popt_y,
        )
        fig, ax = plt.subplots(1, 2, figsize = (8, 4))
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = locs_0_matched,
            hue = 'y_shift',
            ax = ax[0],
            hue_norm = (locs_0_matched['y_shift_predicted'].min(), locs_0_matched['y_shift_predicted'].max()),
        )
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = locs_0_matched,
            hue = 'y_shift_predicted',
            ax = ax[1],
            hue_norm = (locs_0_matched['y_shift_predicted'].min(), locs_0_matched['y_shift_predicted'].max()),
        )
        ax[0].set_title('y shift, observed')
        ax[1].set_title('y shift, predicted')
        plt.show(); plt.close()


        locs_0_matched['x_shift_predicted'] = affine_model(
            np.asarray(locs_0_matched[['y_pixels', 'x_pixels']]),
            *popt_x,
        )
        fig, ax = plt.subplots(1, 2, figsize = (8, 4))
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = locs_0_matched,
            hue = 'x_shift',
            ax = ax[0],
            hue_norm = (locs_0_matched['x_shift_predicted'].min(), locs_0_matched['x_shift_predicted'].max()),
        )
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = locs_0_matched,
            hue = 'x_shift_predicted',
            ax = ax[1],
            hue_norm = (locs_0_matched['x_shift_predicted'].min(), locs_0_matched['x_shift_predicted'].max()),
        )
        ax[0].set_title('x shift, observed')
        ax[1].set_title('x shift, predicted')
        plt.show(); plt.close()

    # Make the output matrix and vector
    affine_field_matrix = np.array([
        [popt_y[0], popt_y[1]],
        [popt_x[0], popt_x[1]],
    ])
    offset = np.array([popt_y[2], popt_x[2]])

    return affine_field_matrix, offset 





