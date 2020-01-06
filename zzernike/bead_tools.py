'''
bead_tools.py -- various utilities for localizing
and extracting information from calibration beads
'''
# Numerical tools
import numpy as np 
from scipy import ndimage as ndi 
from scipy.spatial import distance_matrix
from scipy.signal import correlate 

# LS fitting
from scipy.optimize import curve_fit 

# Radial symmetry method for particle localization
from pyspaz.localize import radial_symmetry

# Particle detection functions
from pyspaz.localize import _detect_dog_filter 

# Plotting
import matplotlib.pyplot as plt 

# Progress bar
from tqdm import tqdm 

# File readers
from . import zio 

# Package utilities
from . import utils 

# Decomposition into Zernike polynomials
from . import z_trans 

# Visualization utilities
from . import visualize 

def detect_beads_biplane(nd2_file, threshold = 5000, max_distance = 8.0):
    '''
    Use a simple DoG filter to find beads in both 
    channels of a z-stack.

    '''
    # Get an image file reader
    reader = zio.BiplaneImageFileReader(nd2_file)
    
    # Get maximum intensity projections of each channel 
    max_int_ch0 = reader.max_int_projection(c = 0)
    max_int_ch1 = reader.max_int_projection(c = 1)
    
    # Run detection
    detections_ch0 = _detect_dog_filter(max_int_ch0, threshold = threshold)[-1]
    detections_ch1 = _detect_dog_filter(max_int_ch1, threshold = threshold)[-1]
    
    # Take only spots that have a corresponding spot in the other channel
    distances = distance_matrix(detections_ch0, detections_ch1)
    matches_0, matches_1 = (distances < max_distance).nonzero()
    detections_ch0 = detections_ch0[matches_0, :]
    detections_ch1 = detections_ch1[matches_1, :]
    n_detect = detections_ch0.shape[0]
    
    reader.close()
    
    return detections_ch0, detections_ch1, max_int_ch0, max_int_ch1

def bead_center_zstack(nd2_file, detections, c = 0, window_size = 9):
    """
    Find the centers of beads in each frame of a z-stack
    using the radial symmetry method.

    args
    ----
        nd2_file :  str
        detections :  2D ndarray of shape (n_beads, 2), yx 
            coords of each bead
        c :  int, channel
        window_size :  int, size of window to use for fitting

    returns
    -------
        3D ndarray of shape (n_beads, n_planes, 2), the yx
            positions of each bead in each z-plane

    """
    hw = int(window_size) // 2
    n_beads = detections.shape[0]

    # Make image file reader 
    reader = zio.BiplaneImageFileReader(nd2_file)

    # Get the z-stack for the target channel 
    stack = reader.get_zstack(c=c)
    n_planes, N, M = stack.shape 

    # For each bead and each z-plane, run radial
    # symmetry in a window around the bead
    result = np.empty((n_beads, n_planes, 2), dtype = 'float64')
    for bead_idx in range(n_beads):
        y0, x0 = detections[bead_idx, :].astype('int64')
        for z_idx in range(n_planes):
            psf_image = stack[z_idx, y0-hw : y0+hw+1, x0-hw : x0+hw+1]
            result[bead_idx, z_idx, :] = radial_symmetry(psf_image) + np.array([y0, x0]) - hw 

    reader.close()

    return result 

def get_bead_images(nd2_file, centers, c=0, window_size = 13):
    """
    Get a subimage centered on the bead in each
    plane of a z-stack.

    args
    ----
        nd2_file :  str
        centers :  2D ndarray of shape (n_z_planes, 2), the
            yx coordinates of the bead in each frame
        c :  int, channel
        window_size :  int

    returns
    -------
        (
            3D ndarray of shape (n_z_planes, window_size, window_size);
            2D ndarray of shape (n_z_planes, 2), the centers of 
                radial symmetry in the context of the image
        )

    """
    hw = int(window_size) // 2
    centers_int = centers.astype('int64')
    centers_res = centers - centers_int + hw 

    reader = zio.BiplaneImageFileReader(nd2_file)
    stack = reader.get_zstack(c=c)
    n_planes, N, M = stack.shape 

    if len(centers.shape) == 3:
        n_beads = centers.shape[0]
        result = np.empty((n_beads, n_planes, window_size, window_size),
            dtype = stack.dtype)
        for bead_idx in range(n_beads):
            for plane_idx in range(n_planes):
                y0, x0 = centers_int[bead_idx, plane_idx, :]
                result[bead_idx, plane_idx, :, :] = stack[
                    plane_idx,
                    y0-hw : y0+hw+1,
                    x0-hw : x0+hw+1,
                ]

    elif len(centers.shape) == 2:
        result = np.empty((n_planes, window_size, window_size), dtype = stack.dtype)
        for plane_idx in range(n_planes):
            y0, x0 = centers_int[plane_idx, :]

            result[plane_idx, :, :] = stack[plane_idx, 
                y0-hw : y0+hw+1, x0-hw : x0+hw+1
            ]

    reader.close()
    return result, centers_res 

def zernike_coefs_zstack(zstack, scale=5.0, unit_disk_only=True, center_int=False):
    n_planes, w0, w1 = zstack.shape 
    zstack = zstack.astype('float64')

    coefs = np.zeros((n_planes, 10), dtype = 'float64')
    for z_idx in range(n_planes):

        # Use radial symmetry to get the bead center
        center = radial_symmetry(zstack[z_idx, :, :])

        # Use Gram-Schmidt to get the Zernike polynomial
        # coefficients
        coefs[z_idx, :] = z_trans.fwd_10(zstack[z_idx, :, :],
            center, scale=scale, unit_disk_only=unit_disk_only,
            center_int=center_int)

    return coefs 

def zernike_coefs_beads(
    nd2_file,
    threshold = 5000,
    max_distance = 8.0,
    window_size = 15,
    scale = 5.0,
    unit_disk_only = True,
    center_int = False,
    subtract_bg = True,
    plot_coefs = False,
    smooth_kernel = None,
):
    # Find beads in each channel 
    detections_0, detections_1, max_int_0, max_int_1 = detect_beads_biplane(
        nd2_file,
        threshold = threshold,
        max_distance = max_distance,
    )

    # Localize the beads to subpixel precision in each
    # frame of each channel 
    centers_ch0 = bead_center_zstack(
        nd2_file,
        detections_0, 
        c = 0,
        window_size = 9,
    )
    centers_ch1 = bead_center_zstack(
        nd2_file,
        detections_1,
        c = 1,
        window_size = 9,
    )

    # Get images of the beads in each channel
    bead_images_ch0, centers_re_ch0 = get_bead_images(
        nd2_file,
        centers_ch0,
        c = 0,
        window_size = window_size,
    )
    bead_images_ch1, centers_re_ch1 = get_bead_images(
        nd2_file,
        centers_ch1,
        c = 1,
        window_size = window_size,
    )
    n_beads = detections_0.shape[0]
    n_planes = bead_images_ch0.shape[1]

    # If desired, smooth the bead images
    if not (smooth_kernel is None):
        for bead_idx in range(n_beads):
            for plane_idx in range(n_planes):
                bead_images_ch0[bead_idx, plane_idx, :, :] = ndi.uniform_filter(
                    bead_images_ch0[bead_idx, plane_idx, :, :],
                    smooth_kernel
                )
                bead_images_ch1[bead_idx, plane_idx, :, :] = ndi.uniform_filter(
                    bead_images_ch1[bead_idx, plane_idx, :, :],
                    smooth_kernel
                )

    # Calculate the Zernike coefficients
    coefs = np.empty((n_beads, 2, n_planes, 6), dtype = 'float64')
    for bead_idx in range(n_beads):
        for plane_idx in range(n_planes):
            image_ch0 = bead_images_ch0[bead_idx, plane_idx, :, :]
            image_ch1 = bead_images_ch1[bead_idx, plane_idx, :, :]
            
            # Recalculate the center using the larger frame
            if (smooth_kernel is None):
                center_ch0 = radial_symmetry(image_ch0)
                center_ch1 = radial_symmetry(image_ch1)

            # Get the centers from the preexisting array
            else:   
                center_ch0 = centers_re_ch0[bead_idx, plane_idx, :]
                center_ch1 = centers_re_ch1[bead_idx, plane_idx, :]

            coefs[bead_idx, 0, plane_idx, :] = z_trans.fwd_6(
                image_ch0,
                center_ch0,
                scale=scale, 
                unit_disk_only=unit_disk_only,
                center_int=center_int,
                subtract_bg = subtract_bg,
            )
            coefs[bead_idx, 1, plane_idx, :] = z_trans.fwd_6(
                image_ch1,
                center_ch1,
                scale=scale, 
                unit_disk_only=unit_disk_only,
                center_int=center_int,
                subtract_bg = subtract_bg,
            )

    if plot_coefs:
        coefs_rescaled = coefs.copy()
        for bead_idx in range(n_beads):
            for z in range(6):
                coefs_rescaled[bead_idx, 0, :, z] = coefs_rescaled[bead_idx, 0, :, z] / \
                    np.abs(coefs_rescaled[bead_idx, 0, :, z]).max()
                coefs_rescaled[bead_idx, 1, :, z] = coefs_rescaled[bead_idx, 1, :, z] / \
                    np.abs(coefs_rescaled[bead_idx, 1, :, z]).max()

        fig, ax = plt.subplots(2, n_beads, figsize = (3*n_beads, 6))
        if len(ax.shape) == 1:
            ax = np.asarray([ax]).T
        z_levels = np.arange(n_planes)
        for bead_idx in range(n_beads):
            for coef_idx in range(6):
                ax[0, bead_idx].plot(z_levels, coefs[bead_idx, 0, :, coef_idx], label = coef_idx)
                ax[1, bead_idx].plot(z_levels, coefs[bead_idx, 1, :, coef_idx], label = coef_idx)
            ax[0, bead_idx].legend(frameon = False)
            ax[1, bead_idx].legend(frameon = False)
            ax[0, bead_idx].set_title('Bead %d, channel %d' % (bead_idx, 0))
            ax[1, bead_idx].set_title('Bead %d, channel %d' % (bead_idx, 1))
        plt.show(); plt.close()

    return bead_images_ch0, bead_images_ch1, centers_re_ch0, centers_re_ch1, coefs 

def find_and_align(
    *nd2_files,
    threshold = 5000,
    max_distance = 8.0,
    window_size = 15,
    scale = 5.0,
    unit_disk_only = True,
    center_int = False,
    subtract_bg = True,
    plot_coefs = False,
    align_kernel_size = 5,
    align_window_size = 61,
    return_window_size = 61,
    plot_errors = False,
):
    """
    Given a set of ND2 files, find all beads, 
    calculate Zernike coefficients, and align
    the signals.

    Robust combinations:
        scale = 5.0, center_int = False
        scale = 4.0, center_int = False
        scale = 5.0, center_int = True
        scale = 4.0, center_int = True 

    args
    ----
        *nd2_files :   str
        threshold :  float, detection stringency
        max_distance :  float, maximum tolerated distance
            between beads in channels 0 and 1
        window_size :  int, the fitting window size
        scale :  float, radius of the unit disk on which
            to compute Zernike coefficients in um
        unit_disk_only :  bool, only compute Zernike 
            coefficients on the rescaled unit disk (recommended)
        center_int :  bool, center the intensities before
            computing coefficients 1 onward
        plot_coefs :  bool
        align_kernel_size :  int, size of kernel to be used for
            smoothing signals before alignment
        align_window_size :  int, size of the window used for
            computing cross-correlation during alignment
        return_window_size :  int, size of the return window
            around the aligned signals
        plot_errors :  bool

    returns
    -------
        4D ndarray of shape (n_beads, 2, return_window_size, 6),
            the first six Zernike coefficients for each bead,
            channel, and z-plane

    """
    all_coefs = []

    # Find beads and calculate Zernike coefficients
    print("Finding beads and calculating Zernike coefficients...")
    for nd2_idx, nd2_file in tqdm(enumerate(nd2_files)):
        stack_ch0, stack_ch1, centers_ch0, centers_ch1, coefs = \
            zernike_coefs_beads(
                nd2_file,
                threshold = threshold,
                max_distance = max_distance,
                window_size = window_size,
                scale = scale,
                unit_disk_only = unit_disk_only,
                center_int = center_int,
                subtract_bg = subtract_bg,
                plot_coefs = plot_coefs,
                smooth_kernel = None,
            )
        all_coefs.append(coefs)

    # Align all of the signals
    print("Aligning signals...")
    n_signals = sum([all_coefs[i].shape[0] for i in range(len(all_coefs))])
    keep = np.ones(n_signals, dtype = 'bool')

    aligned_coefs = np.zeros((n_signals, 2, return_window_size, 6), dtype = 'float64')
    c_idx = 0

    ref_sig_ch0 = all_coefs[0][0,0,:,:]
    ref_sig_ch1 = all_coefs[0][0,1,:,:]
    hw = int(return_window_size) // 2

    for nd2_idx in range(len(nd2_files)):
        coefs = all_coefs[nd2_idx]
        n_beads = coefs.shape[0]
        for bead_idx in range(n_beads):

            # Try to do alignment. If it fails, skip
            # that bead.
            try:
                # Align channel 0
                _ref_aligned, sig_aligned, n0, n1 = align_coefs(
                    ref_sig_ch0,
                    coefs[bead_idx, 0, :, :],
                    kernel_size = align_kernel_size,
                    align_window_size = align_window_size,
                    return_size = return_window_size,
                    plot = False,
                )
                aligned_coefs[c_idx, 0, :, :] = sig_aligned[:,:6].copy()

                # Get the corresponding signal in channel 1
                aligned_coefs[c_idx, 1, :, :] = coefs[
                    bead_idx,
                    1,
                    n1-hw : n1+hw+1,
                    :6,
                ].copy()

                # Align channel 1
                # _ref_aligned, sig_aligned, n0, n1 = align_coefs(
                #     ref_sig_ch1,
                #     coefs[bead_idx, 1, :, :],
                #     kernel_size = align_kernel_size,
                #     align_window_size = align_window_size,
                #     return_size = return_window_size,
                #     plot = False,
                # )
                # aligned_coefs[c_idx, 1, :, :] = sig_aligned[:,:6].copy()
            except ValueError:  #Alignment failed, skip this bead
                if plot_errors:
                    print('Alignment failed for %d' % c_idx)
                    visualize.show_sig(*coefs[bead_idx, 0, :, :].T)
                    visualize.show_sig(*ref_sig_ch0.T)
                keep[c_idx] = False 

            c_idx += 1

    aligned_coefs = aligned_coefs[keep, :, :, :]

    return aligned_coefs


def align_coefs(coefs_0, coefs_1, plot=True, kernel_size=6,
    align_window_size=61, return_size=101):
    """
    Use the Zernike coefficients to align two
    z-stacks in the z-dimension.

    Untested when the signals are of different lengths.

    args
    ----
        coefs_0, coefs_1 :  2D ndarrays of shape (n_planes, n_coefs),
            the Zernike coefficients

        align_window :  int, the size of the window to return
            the aligned signals

        plot :  bool, show the signals before and after 
            alignment

    returns
    -------
        (
            2D ndarray, the aligned version of coefs_0;
            2D ndarray, the aligned version of coefs_1;
            int, the offset;
        )

    """
    # Use Zernike coefficients 0 and 4 to align the signals
    n00, n10 = find_offset(coefs_0, coefs_1, coef_idx=0, 
        kernel_size=kernel_size, align_window_size=align_window_size)
    n01, n11 = find_offset(coefs_0, coefs_1, coef_idx=4, 
        kernel_size=kernel_size, align_window_size=align_window_size)

    n0 = int((n00 + n01) / 2.0)
    n1 = int((n10 + n11) / 2.0)
    offset = n1 - n0 

    # Get aligned versions of the signals
    hw = int(return_size) // 2

    coefs_0_ali = coefs_0[n0-hw : n0+hw+1, :]
    coefs_1_ali = coefs_1[n1-hw : n1+hw+1, :]

    # Show alignment, if desired
    if plot:
        fig, ax = plt.subplots(2, 1, figsize = (6, 8))
        visualize.show_sig(coefs_0, coefs_1, ax=ax[0],
            legend=False)
        visualize.show_sig(coefs_0_ali, coefs_1_ali,
            ax=ax[1], legend=False)
        ax[0].set_title('Not aligned')
        ax[1].set_title('Aligned')
        ax[0].set_xlabel('z slice index')
        ax[1].set_xlabel('z slice index')
        plt.tight_layout(); plt.show(); plt.close()

    return coefs_0_ali, coefs_1_ali, n0, n1  

def find_offset(coefs_0, coefs_1, coef_idx=0, plot=False, kernel_size=5,
    align_window_size = 61):
    """
    Align two sets of Zernike coefficients by
    maximizing cross-correlation between a
    particular Zernike coefficient.

    args
    ----
        coefs_0, coefs_1 :  2D ndarrays of shape
            (n_planes, n_coefs), the Zernike 
            coefficients for each z-stack
        
        coef_idx :  int, the coefficient to use 
            for the correlation

        plot :  bool, show aligned signals

        kernel_size :  int, size of smoothing kernel

        align_window_size :  int, size of alignment
            window

    returns
    -------
        (int, int), the points in the two signals
            that are aligned

    """
    # Smooth the signals
    s0 = ndi.uniform_filter(coefs_0[:,coef_idx], kernel_size)
    s1 = ndi.uniform_filter(coefs_1[:,coef_idx], kernel_size)

    # Subtract BG and rescale
    s0 = s0 - s0.min()
    s1 = s1 - s1.min()
    s0 = s0 / s0.max()
    s1 = s1 / s1.max()
    s0 = s0 - s0.mean()
    s1 = s1 - s1.mean()

    # Take a small window around the maximum intensity point
    # in s1
    hw = int(align_window_size) // 2
    i1_max = np.argmax(np.abs(s1))
    s1_sub = s1[i1_max-hw : i1_max+hw+1]

    # Get the cross-correlation of the two signals
    corr = correlate(s0, s1_sub, mode = 'same')

    # Get the offset that maximizes cross-correlation
    m = np.argmax(corr)

    # Show alignment, if desired
    if plot:

        # Get a window around this point in each signal
        coefs_0_ali = coefs_0[m - 50 : m + 51, :]
        coefs_1_ali = coefs_1[i1_max - 50 : i1_max + 51, :]

        # Plot
        fig, ax = plt.subplots(2, 1, figsize = (6, 8))
        visualize.show_sig(coefs_0, coefs_1, ax=ax[0],
            legend=False)
        visualize.show_sig(coefs_0_ali, coefs_1_ali,
            ax=ax[1], legend=False)
        ax[0].set_title('Not aligned')
        ax[1].set_title('Aligned')
        ax[0].set_xlabel('z slice index')
        ax[1].set_xlabel('z slice index')
        plt.tight_layout(); plt.show(); plt.close()

    return m, i1_max


def mean_radial_distance_zstack(zstack, scale=5.0, unit_disk_only=True,
    center_int=False):
    """
    Calculate the mean distance of photons from the
    point of maximal radial symmetry. This is done
    at each plane of a z-stack.

    args
    ----
        zstack :  3D ndarray of shape (n_planes, w, w)
        scale :  float, unit disk size in pixels
        unit_disk_only :  float, only consider points
            inside the rescaled unit disk 

    returns
    -------
        1D ndarray of shape (n_planes,)

    """
    n_planes, w0, w1 = zstack.shape 
    zstack = zstack.astype('float64')

    result = np.zeros(n_planes, dtype = 'float64')
    for z_idx in range(n_planes):
        psf_image = zstack[z_idx, :, :]
        center = radial_symmetry(psf_image)
        Y, X = np.indices(psf_image.shape)
        Y = (Y.astype('float64') - center[0]) / scale 
        X = (X.astype('float64') - center[1]) / scale 
        R = np.sqrt(Y**2 + X**2)

        if center_int:
            _I = psf_image - psf_image.mean()
        else:
            _I = psf_image 

        if unit_disk_only:
            inside = R <= 1.0
            result[z_idx] = (R[inside] * _I[inside]).sum()
        else:
            result[z_idx] = (R * _I).sum()

    return result 

def fit_polynomial_model(coefs, plot=True):
    """
    Fit a fourth-order polynomial to Zernike coefficients
    as a function of z.

    args
    ----
        coefs :  3D ndarray of shape (n_samples, Z, n_coefs)

    """
    n_samples, n_planes, n_coefs = coefs.shape 

    # Make linear z-index
    z_levels = np.asarray(list(np.arange(n_planes)) * n_samples)

    # Define the merit function 
    n_poly_coefs = 6
    def polynomial_model(z, c0, c1, c2, c3, c4, c5):
        """
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
        Z = np.power(np.asarray([z]).T, np.arange(n_poly_coefs)).T 
        return c.dot(Z)

    # Coefficient matrix, for result
    C = np.zeros((n_coefs, n_poly_coefs), dtype = 'float64')

    # Fit each coefficient to a linear model 
    for coef_idx in range(n_coefs):
        response = coefs[:,:,coef_idx].flatten()
        poly_coefs, pcov = curve_fit(
            polynomial_model,
            z_levels,
            response,
        )
        C[coef_idx, :] = poly_coefs 

    # Show the result
    if plot:
        # Make 1D plot
        fig, ax = plt.subplots(1, n_coefs, figsize = (3 * n_coefs, 3))
        for coef_idx in range(n_coefs):
            fit = polynomial_model(z_levels[:n_planes], *C[coef_idx, :])
            ax[coef_idx].plot(
                z_levels,
                coefs[:,:,coef_idx].flatten(),
                marker = '.',
                markersize = 5,
                linestyle = '',
                color = 'r',
                label = 'Data',
            )
            ax[coef_idx].plot(
                z_levels[:n_planes],
                fit,
                linestyle = '--',
                color = 'k',
                label = 'Polynomial model',
            )
            ax[coef_idx].set_title('Coefficient %d' % coef_idx)
            ax[coef_idx].legend(frameon = False, loc = 'upper right')
        plt.show(); plt.close()

        # Make 2D plot
        if n_coefs == 2:
            visualize.plot_2d_poly_fit(coefs, C)

    return C 








