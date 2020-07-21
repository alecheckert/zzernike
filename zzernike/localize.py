'''
localize.py -- localization utilities with computation of 
Zernike transforms for PSFs

'''
# Numerics
import numpy as np 
from scipy.spatial import distance_matrix 

# LS fitting, for calculating the affine
# field between the channels
from scipy.optimize import curve_fit 

# Parallelization
import dask 

# File reader
from nd2reader import ND2Reader 

# I/O
import sys
import os

# Plotting
import matplotlib.pyplot as plt 
import seaborn as sns 

# Profiling
from time import time

# Dataframes
import pandas as pd 

# Progress bar
from tqdm import tqdm 

# Get a list of ND2 files
from glob import glob 

# Uniform kernel filter, for smoothing a PSF image
from scipy.ndimage import uniform_filter 

# Warnings filter for radial_symmetry method; the
# errors are divide-by-zero and are caught in a 
# subsequent step
import warnings

# Package utilities
from . import utils 

# Package I/O
from . import zio 

# Zernike transforms
from . import z_trans 

# Package visualization tools
from . import visualize 

def localize_biplane(
    nd2_path,
    channel_gains = [107.3153, 96.4723],
    channel_bgs = [470.7030, 212.1311],
    detect_sigma = 1.0,
    detect_threshold = 20.0,
    detect_window_size = 9,
    fit_window_size = 15,
    scale = 2.75,
    unit_disk_only = True,
    center_int = False,
    subtract_bg = False,
    start_frame = None,
    stop_frame = None,
    frames_to_do = None,
    progress_bar = True,
    n_coefs = 15,
    tol_radius = 8.0,
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

        n_coefs :  int, 6 or 10 or 15

    returns
        (pandas.DataFrame, pandas.DataFrame), localizations
            in each channel

    '''
    assert n_coefs in [6, 10, 15]

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

    if (frames_to_do is None):
        if start_frame is None:
            start_frame = 0
        if stop_frame is None:
            stop_frame = n_frames - 1
        frames_to_do = np.arange(start_frame, stop_frame + 1)

    # Run localization on each channel sequentially 
    output_dfs = []
    for channel_idx in [0, 1]:

        # Initialize output array
        if n_coefs == 6:
            locs = np.zeros((1000000, 14), dtype = 'float64')
        elif n_coefs == 10:
            locs = np.zeros((1000000, 18), dtype = 'float64')
        elif n_coefs == 15:
            locs = np.zeros((1000000, 23), dtype = 'float64')

        # Current localization index
        c_idx = 0

        # Make the progress bar with tqdm 
        if progress_bar:
            t = tqdm(frames_to_do, leave = False)
        else:
            t = tqdm(frames_to_do, disable = True, leave = False)

        for frame_idx in t:

            # Get the image corresponding to this frame
            image = reader.get_frame_2D(t = frame_idx, c = channel_idx).astype('float64')

            # Convert from grayvalues to photons
            image = (image - channel_bgs[channel_idx]) / channel_gains[channel_idx]
            image[image < 0] = 0

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
                    if n_coefs == 6:
                        locs[c_idx, 7:13] = z_trans.fwd_6(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                            center_int = center_int,
                            subtract_bg = subtract_bg,
                        ).copy()

                        # Compute mean radial distance from center
                        locs[c_idx, 13] = z_trans.mean_r(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                        )
                    elif n_coefs == 10:
                        locs[c_idx, 7:17] = z_trans.fwd_10(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                            center_int = center_int,
                            subtract_bg = subtract_bg,
                        ).copy()

                        # Compute mean radial distance from center
                        locs[c_idx, 17] = z_trans.mean_r(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                        )
                    elif n_coefs == 15:
                        locs[c_idx, 7:22] = z_trans.fwd_15(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                            center_int = center_int,
                            subtract_bg = subtract_bg,
                        ).copy()

                        # Compute mean radial distance from center
                        locs[c_idx, 22] = z_trans.mean_r(
                            psf_image,
                            (y_rs, x_rs),
                            scale = scale,
                            unit_disk_only = unit_disk_only,
                        )
                c_idx += 1

        locs = locs[:c_idx, :]

        # Format output and enforce some typing
        if n_coefs == 6:
            locs = pd.DataFrame(locs, columns = [
                'frame_idx', 'y_detect', 'x_detect', 'llr', 'y_pixels', 'x_pixels',
                'error_code', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'R0',
            ])
        elif n_coefs == 10:
            locs = pd.DataFrame(locs, columns = [
                'frame_idx', 'y_detect', 'x_detect', 'llr', 'y_pixels', 'x_pixels',
                'error_code', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7',
                'Z8', 'Z9', 'R0',
            ])  
        elif n_coefs == 15:
            locs = pd.DataFrame(locs, columns = [
                'frame_idx', 'y_detect', 'x_detect', 'llr', 'y_pixels', 'x_pixels',
                'error_code', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7',
                'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'R0',
            ])  
        locs['frame_idx'] = locs['frame_idx'].astype('int64')
        locs['y_detect'] = locs['y_detect'].astype('int64')
        locs['x_detect'] = locs['x_detect'].astype('int64')
        locs['error_code'] = locs['error_code'].astype('uint16')

        output_dfs.append(locs.copy())

    reader.close()

    return output_dfs

def localize_biplane_parallel_directory(
    nd2_directory,
    out_directory,
    dask_client,
    verbose = False,
    match = True,
    **kwargs,
):
    nd2_files = glob("%s/*.nd2" % nd2_directory)
    nd2_subpaths = [i.split('/')[-1] for i in nd2_files]
    if not os.path.isdir(out_directory):
        os.mkdir(out_directory)

    for nd2_idx, nd2_file in enumerate(nd2_files):
        loc_dfs = localize_biplane_parallel(
            nd2_file,
            dask_client,
            verbose = verbose,
            **kwargs,
        )
        out_file_0 = '%s/%s' % (
            out_directory,
            nd2_subpaths[nd2_idx].replace('.nd2', '_ch0_locs.txt')
        )
        out_file_1 = '%s/%s' % (
            out_directory,
            nd2_subpaths[nd2_idx].replace('.nd2', '_ch1_locs.txt')
        )

        loc_dfs[0].to_csv(out_file_0, sep = '\t', index = False)
        loc_dfs[1].to_csv(out_file_1, sep = '\t', index = False)

def localize_biplane_parallel(nd2_file, dask_client,
    verbose=False, **kwargs):
    """
    Localize all particles in both planes of a ND2 file
    using parallelization. Wrapper for localize_biplane().

    args
    ----
        nd2_file :  str
        dask_client :  dask.distributed.Client
        verbose :  bool
        **kwargs :  to localize_biplane()

    returns
    -------
        (
            pandas.DataFrame, localizations in channel 0;
            pandas.DataFrame, localizations in channel 1
        )

    """
    # Get the number of workers for this dask client
    n_workers = len(dask_client.scheduler_info()['workers'])

    kwargs['progress_bar'] = False
    kwargs['start_frame'] = None
    kwargs['stop_frame'] = None 

    reader = zio.BiplaneImageFileReader(nd2_file)
    N, M, n_frames = reader.get_shape()
    reader.close()

    # Divide the list of all frames into ranges that will be
    # given to each individual worker
    frames = np.arange(n_frames)
    frame_ranges = [frames[i::n_workers] for i in range(n_workers)]

    # Assign each frame range to a worker
    results = []
    for frame_range_idx, frame_range in enumerate(frame_ranges):
        kwargs['frames_to_do'] = frame_range 
        file_results = dask.delayed(localize_biplane)(nd2_file, **kwargs)
        results.append(file_results)

    t0 = time()
    out_tuples = dask_client.compute(results)
    out_tuples = [i.result() for i in out_tuples]
    t1 = time()
    if verbose:
        print('Finished with %s\nRun time: %.2f sec' % (nd2_file, t1 - t0))

    locs_0 = pd.concat(
        [out_tuples[i][0] for i in range(len(out_tuples))],
        ignore_index = True, sort = False,
    )
    locs_1 = pd.concat(
        [out_tuples[i][1] for i in range(len(out_tuples))],
        ignore_index = True, sort = False,
    )
    locs_0 = locs_0.sort_values(by = 'frame_idx')
    locs_1 = locs_1.sort_values(by = 'frame_idx')
    locs_0.index = np.arange(len(locs_0))
    locs_1.index = np.arange(len(locs_1))

    return locs_0, locs_1 

def localize_match(nd2_file, dask_client=None, tol_radius=8.0,
    cols = ['Z0', 'Z4', 'Z12'], verbose=False, lim=(-0.5, 1.5), **kwargs):
    """
    Localize spots in both channels of an ND2 file and also
    match localizations between channels, returning the 
    dataframe of all matched localizations.

    args
    ----
        nd2_file :  str
        dask_client :  dask.distributed.Client
        tol_radius :  float, maximum distance between
            matched localizations
        cols :  list of str
        lim :  (float, float), the boundaries on the data
        **kwargs :  to localize_biplane()

    returns
    -------
        pandas.DataFrame

    """
    # Localize spots in both channels
    if not (dask_client is None):
        locs_0, locs_1 = localize_biplane_parallel(nd2_file, dask_client,
            verbose=verbose, **kwargs)
    else:
        locs_0, locs_1 = localize_biplane(nd2_file, **kwargs)

    # Match localizations between channels 
    locs = match_locs(locs_0, locs_1, tol_radius=tol_radius,
        outlier_tol=tol_radius)

    # Calculate f-ratios for the desired parameters
    locs = f_ratio(locs, cols)
    fcols = ['f%s' % c for c in cols]

    # Remove crazy values
    locs = bound_data(locs, fcols, lim)

    return locs 


def localize_z_polynomial(matched_locs, C, guess_z, z_coefs = ['Z0', 'Z4'], **kwargs):
    """
    Use a polynomial fit to Zernike coefficients to 
    localize a particle in z. This is a wrapper for
    distance_to_polynomial() that runs on all of the 
    localizations in a dataframe.

    args
    ----
        matched_locs :  pandas.DataFrame, matched localizations
            between frames
        C :  2D ndarray of shape (n_coefs, 6), the polynomial
            coefficients corresponding to *z_coefs*
        guess_z :  float, the initial guess for the z-positions
        z_coefs :  list of int, the indices of the Zernike
            polynomials to use
        **kwargs :  to distance_to_polynomial()

    returns
    -------

    """
    # Work with a copy of the data
    df = matched_locs.copy() 

    n_locs = df.shape[0]
    n_coefs = len(z_coefs)
    cols_ch1 = ['%s_ch1' % i for i in z_coefs]
    cols_f = ['f%s' % i for i in z_coefs]

    # Compute the f-ratio for each coefficient
    for i in range(n_coefs):
        df[cols_f[i]] = df[z_coefs[i]] / \
            (df[[z_coefs[i], cols_ch1[i]]].sum(axis=1))

    # Get all of the point positions as an ndarray
    positions = np.asarray(df[cols_f])

    # Save the z-positions for each localization
    z_pos = np.zeros(n_locs, dtype = 'float64')

    # Fit each f-ratio to the model
    for loc_idx in tqdm(range(n_locs)):
        point = positions[loc_idx, :]
        z_pos[loc_idx] = distance_to_polynomial(
            point,
            C,
            guess_z,
            **kwargs
        )

    # Add the result as a column to the dataframe
    df['z_frames'] = z_pos 

    # Return the modified dataframe
    return df

def localize_z_eig(
    matched_locs,
    cols = ['fZ0', 'fZ4', 'fZ12'],
):
    
    cols_c = ['%s_c' % c for c in cols]

    # Center the data
    for c_idx, c in enumerate(cols):
        matched_locs[cols_c[c_idx]] = matched_locs[c] - matched_locs[c].mean()

    # Get the data as an ndarray
    data = np.asarray(matched_locs[cols_c])


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
        outlier_tol :  float, max radius in pixels for 
            second filter on spots

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
    for frame_idx in range(n_frames):

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

    # Add Zernike coefficients for ch1
    if 'Z14' in locs_0_matched.columns:
        n_coefs = 15
    elif 'Z9' in locs_0_matched.columns:
        n_coefs = 10
    else:
        n_coefs = 6
    for coef_idx in range(n_coefs):
        locs_0_matched['Z%d_ch1' % coef_idx] = locs_1_matched['Z%d' % coef_idx]

    # Add mean radial distance coefficients for ch1
    locs_0_matched['R0_ch1'] = locs_1_matched['R0']

    # Remove outliers
    y_shift_mean = locs_0_matched['y_shift'].mean()
    x_shift_mean = locs_0_matched['x_shift'].mean()
    locs_0_matched['y_shift_dev'] = locs_0_matched['y_shift'] - y_shift_mean
    locs_0_matched['x_shift_dev'] = locs_0_matched['x_shift'] - x_shift_mean

    locs_0_matched = locs_0_matched[np.abs(locs_0_matched['y_shift_dev']) < outlier_tol]
    locs_0_matched = locs_0_matched[np.abs(locs_0_matched['x_shift_dev']) < outlier_tol]

    return locs_0_matched 

def match_locs_by_affine_field(
    nd2_file,
    locs_0,
    afm, 
    offset,
    scale = 3.0,
    unit_disk_only = True,
    center_int = False,
    subtract_bg = False,
    window_size = 13,
    ch1_gain = 96.4723,
    ch1_bg = 212.1311,
):
    reader = ND2Reader(nd2_file)

    afmi = afm + np.identity(2)
    yx0 = np.asarray(locs_0[['y_pixels', 'x_pixels', 'frame_idx']])
    yx1 = afmi.dot(yx0[:,:2].T).T + offset
    yx1_int = yx1.astype('int64')

    hw = int(window_size) // 2
    n_locs = len(locs_0)
    unique_frames = np.unique(locs_0['frame_idx'])

    locs_1 = np.zeros((n_locs, 23), dtype = 'float64')

    c_idx = 0
    for frame_idx in tqdm(unique_frames):
        frame = reader.get_frame_2D(t=frame_idx, c=1).astype('float64')
        locs_in_frame = yx1_int[(yx0[:,2]==frame_idx), :]

        for l_idx in range(locs_in_frame.shape[0]):
            y0, x0 = locs_in_frame[l_idx, :]
            psf_image_ch1 = frame[
                y0-hw : y0+hw+1,
                x0-hw : x0+hw+1,
            ]

            # Convert to photons
            psf_image_ch1 = (psf_image_ch1 - ch1_bg) / ch1_gain 
            psf_image_ch1[psf_image_ch1 < 0] = 0

            # Save the results
            locs_1[c_idx, 0] = frame_idx
            locs_1[c_idx, 1:3] = locs_in_frame[l_idx, :]
            locs_1[c_idx, 3] = np.nan 
            locs_1[c_idx, 4:6] = radial_symmetry(psf_image_ch1)
            locs_1[c_idx, 7:22] = z_trans.fwd_15(
                psf_image_ch1,
                locs_1[c_idx, 4:6],
                scale = scale,
                unit_disk_only = unit_disk_only,
                center_int = center_int,
                subtract_bg = subtract_bg,
            ).copy()
            locs_1[c_idx, 22] = z_trans.mean_r(
                psf_image_ch1,
                locs_1[c_idx, 4:6],
                scale = scale,
                unit_disk_only = unit_disk_only,
            )
            c_idx += 1 

    reader.close()

    # Reformat as pandas.DataFrame
    locs_1 = pd.DataFrame(locs_1, columns = [
        'frame_idx', 'y_detect', 'x_detect', 'llr', 'y_pixels', 'x_pixels',
        'error_code', 'Z0', 'Z1', 'Z2', 'Z3', 'Z4', 'Z5', 'Z6', 'Z7',
        'Z8', 'Z9', 'Z10', 'Z11', 'Z12', 'Z13', 'Z14', 'R0',
    ])
    locs_1['frame_idx'] = locs_1['frame_idx'].astype('int64')
    locs_1['y_detect'] = locs_1['y_detect'].astype('int64')
    locs_1['x_detect'] = locs_1['x_detect'].astype('int64')
    locs_1['error_code'] = locs_1['error_code'].astype('int64')

    # Add the new columns to the original dataframe
    for c in ['y_pixels', 'x_pixels', 'error_code', 'Z0', 'Z1', 'Z2',
        'Z3', 'Z4', 'Z5', 'Z6', 'Z7', 'Z8', 'Z9', 'Z10', 'Z11', 'Z12',
        'Z13', 'Z14', 'R0']:
        locs_0['%s_ch1' % c] = locs_1[c]

    return locs_0


def f_ratio(matched_locs, col_name):
    """
    Convenience function to take the f-ratio of 
    quantity X in channels 0 and 1, defined as 

        f-ratio(X) = X(channel 0) / (X(channel 0) + X(channel 1))

    args
    ----
        matched_locs :  pandas.DataFrame
        col_name :  str or list of str

    returns
    -------
        pandas.DataFrame with new column

    """
    if isinstance(col_name, list):
        for c in col_name:
            matched_locs = f_ratio(matched_locs, c)
    else:
        col_name_ch1 = '%s_ch1' % col_name 
        assert col_name in matched_locs.columns 
        assert col_name_ch1 in matched_locs.columns

        matched_locs['f%s' % col_name] = matched_locs[col_name] / \
            matched_locs[[col_name, col_name_ch1]].sum(axis = 1)

    return matched_locs 

def bound_data(dataframe, columns, bounds):
    """
    Apply simple bounds to each of a set of 
    columns in a dataframe, returning only
    elements that lie within the bounds.

    args
    ----
        dataframe :  pandas.DataFrame
        columns :  str or list of str
        bounds :  (float, float)

    returns
    -------
        pandas.DataFrame

    """
    result = dataframe.copy()
    if isinstance(columns, str):
        result = result[result[columns] <= bounds[1]]
        result = result[result[columns] >= bounds[0]]
    elif isinstance(columns, list):
        for c in columns:
            result = result[result[c] <= bounds[1]]
            result = result[result[c] >= bounds[0]]
    return result 

def calculate_affine_field(matched_locs, plot=True):
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
        np.asarray(matched_locs[['y_pixels', 'x_pixels']]),
        np.asarray(matched_locs['y_shift'])
    )
    popt_x, pcov = curve_fit(
        affine_model,
        np.asarray(matched_locs[['y_pixels', 'x_pixels']]),
        np.asarray(matched_locs['x_shift'])
    )  

    # Make the output matrix and vector
    affine_field_matrix = np.array([
        [popt_y[0], popt_y[1]],
        [popt_x[0], popt_x[1]],
    ])
    offset = np.array([popt_y[2], popt_x[2]])

    # Compare the actual observed y-/x- shifts with the affine
    # field model 
    if plot:
        visualize.plot_affine_field(matched_locs, affine_field_matrix, offset)

    return affine_field_matrix, offset 


