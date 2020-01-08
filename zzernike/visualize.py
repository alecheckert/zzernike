'''
visualize.py -- visualization utilities for zzernike

'''
# Numerical stuff
import numpy as np 

# Dataframes
import pandas as pd 

# matplotlib
import matplotlib.pyplot as plt 
import seaborn as sns 
from matplotlib import cm 

# 3d plotting
from mpl_toolkits.mplot3d import Axes3D 

# I/O
import os
import sys
import tifffile

# Progress bar
from tqdm import tqdm 

# Package utilities
from . import utils 
from . import zio 

# Interactive functions for Jupyter notebooks
import ipywidgets as widgets
from ipywidgets import interact, interactive, fixed, interact_manual

def wrapup(out_png):
    plt.tight_layout()
    plt.savefig(out_png, dpi = 600)
    plt.close()
    os.system('open %s' % out_png)

def imshow(*imgs, vmax=1.0):
    n = len(imgs)
    if n == 1:
        fig, ax = plt.subplots(figsize = (4, 4))
        ax.imshow(imgs[0], cmap='gray', vmax = imgs[0].max()*vmax)
    else:
        fig, ax = plt.subplots(1, n, figsize = (3*n, 3))
        for i in range(n):
            ax[i].imshow(imgs[i], cmap='gray', vmax=imgs[i].max()*vmax)

    plt.show(); plt.close()

def imshow_3d(*imgs):
    n = len(imgs)
    if n == 1:
        fig = plt.figure(figsize = (8, 4))
        Y, X = np.indices(imgs[0].shape)
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_wireframe(Y, X, imgs[0], color = 'k')
    else:
        fig = plt.figure(figsize = (3*n, 3))
        subplot_indices = [int('1%d%d' % (n, j+1)) for j in range(n)]
        for i in range(n):
            Y, X = np.indices(imgs[i].shape)
            ax = fig.add_subplot(subplots_indices[i], projection = '3d')
            ax.plot_wireframe(Y, X, imgs[i], color = 'k')
    plt.show(); plt.close()

def show_Z(Z_array):
    """
    Display a set of Zernike polynomials with 
    OSA/ANSI indexing.

    args
    ----
        Z_array :  3D ndarray with shape 
            (n_polynomials, y_size, x_size)

    """
    n_poly = Z_array.shape[0]
    n_ax = (n_poly // 4) + 1
    fig, ax = plt.subplots(n_ax, 4, figsize = (2 * n_ax, 2 * 4))
    for j in range(n_poly):
        ax[j//4, j%4].imshow(Z_array[j,:,:], cmap='gray')
        m, n = utils.osa_to_z_index(j)
        ax[j//4, j%4].set_title("Z[{}, {}]".format(m, n))

    # Set the rest of the axes to invisible
    for j in range(n_poly, n_ax*4):
        for _s in ['top', 'bottom', 'left', 'right']:
            ax[j//4, j%4].spines[_s].set_visible(False)
        ax[j//4, j%4].set_xticks([])
        ax[j//4, j%4].set_yticks([])

    plt.show(); plt.close()

def show_Z_3d(Z_array):
    """
    Display a set of Zernike polynomials with
    OSA/ANSI indexing in 3D.

    args
    ----
        Z_array :  3D ndarray with shape 
            (n_polynomials, y_size, x_size)

    """
    Y, X = np.indices(Z_array.shape[1:])
    n_poly = Z_array.shape[0]
    n_ax = (n_poly // 4) + 1

    fig_coords = [int('%d%d%d' % (n_ax, 4, j+1)) for j in range(n_poly)]
    figsize = (n_ax * 4, 6)

    fig = plt.figure(figsize = figsize)
    for j in range(n_poly):
        m, n = utils.osa_to_z_index(j)
        ax = fig.add_subplot(fig_coords[j], projection='3d')
        ax.plot_wireframe(Y, X, Z_array[j,:,:], color = 'k')
        ax.set_title('Z[{}, {}]'.format(m, n))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('Y')
        ax.set_ylabel('X')

    plt.show(); plt.close()

def imshow_zstacks(*stacks, vmax=1.0):
    n = len(stacks)
    if n == 1:
        tifffile.imshow(stacks[0], cmap='gray', vmax=stacks[0].max()*vmax)
    else:
        size_x = sum([stacks[i].shape[2] for i in range(n)])
        size_y = max([stacks[i].shape[1] for i in range(n)])
        size_z = max([stacks[i].shape[0] for i in range(n)])
        I_max = max([stacks[i].max() for i in range(n)])
        result = np.zeros((size_z, size_y, size_x), dtype = stacks[0].dtype)
        cx = 0
        for i in range(n):
            stack_z, stack_y, stack_x = stacks[i].shape
            result[:stack_z, :stack_y, cx:cx+stack_x] = stacks[i]
            cx += stack_x 
        tifffile.imshow(result, cmap='gray', vmax=I_max*vmax)

    plt.show(); plt.close()

def overlay_points(stack, centers, upsampling=2, vmax=1.0,
    crosshair_len=3,):
    """
    args
    ----
        stacks :  3D ndarray of shape (n_planes, w, w)
        centers :  2D ndarray of shape (n_planes, 2),
            yx coords
        vmax :  float

    returns
    -------
        None

    """
    orig_Z, orig_N, orig_M = stack.shape
    shape = (orig_Z, upsampling * orig_N, upsampling * orig_M)

    # Expand the image stack 
    upstack = np.zeros(shape, dtype = stack.dtype)
    for i in range(upsampling):
        for j in range(upsampling):
            upstack[:, i::upsampling, j::upsampling] = stack[:, :, :]

    # Overlay the points
    I = upstack.max()
    centers_int = (centers * upsampling).astype('int64')
    for z_idx in range(shape[0]):
        y, x = centers_int[z_idx, :]

        for c in range(-crosshair_len, crosshair_len + 1):
            upstack[z_idx, y+c, x] = I 
            upstack[z_idx, y, x+c] = I

    tifffile.imshow(upstack, cmap = 'gray', vmax = upstack.max() * vmax)
    plt.show(); plt.close() 
            
def plot_coefs(coefs_0, coefs_1):
    fig, ax = plt.subplots(2, 1, figsize = (6, 9))

    z0 = np.arange(coefs_0.shape[0])
    z1 = np.arange(coefs_1.shape[0])

    for v in range(coefs_0.shape[1]):
        ax[0].plot(z0, coefs_0[:,v], label = v)
    for v in range(coefs_1.shape[1]):
        ax[1].plot(z1, coefs_1[:,v], label = v)
    for j in range(2):
        ax[j].legend(frameon = False)
    plt.show(); plt.close()

def show_sig(*signals, ax=None, legend=True, y_min=None, y_max=None):
    """
    Simple utility function to plot a set of 1D 
    signals on the same axis.

    args
    ----
        signals :  1D ndarrays

    """
    if (ax is None):
        fig, ax = plt.subplots(figsize = (6, 4))
        finish = True 
    else:
        finish = False

    n = len(signals)
    for i in range(n):
        x = np.arange(len(signals[i]))
        ax.plot(x, signals[i], label = i)
    if legend: ax.legend(frameon = False)

    if not (y_min is None) and not (y_max is None):
        ax.set_ylim((y_min, y_max))
    elif not (y_min is None):
        ax.set_ylim((y_min, ax.get_ylim()[1]))
    elif not (y_max is None):
        ax.set_ylim((ax.get_ylim()[1], y_max))
    else:
        pass

    if finish:
        plt.show(); plt.close()

def scatter(coefs, ax=None, out_png=None, C=None, xlabel=None, ylabel=None):
    """
    args
    ----
        coefs :  3D ndarray of shape (n_signals, Z, 2)

    """
    assert len(coefs.shape) == 3
    assert coefs.shape[2] == 2

    n_signals, Z, n_coefs = coefs.shape
    fig, ax = plt.subplots(figsize = (4, 4))
    colors = sns.color_palette('viridis', Z).as_hex()

    for z_idx, color in enumerate(colors):
        ax.scatter(
            coefs[:,z_idx,0],
            coefs[:,z_idx,1],
            c = color,
            s = 10,
        )

    if not (C is None):
        z_levels = np.arange(coefs.shape[1])
        model_x = utils.polynomial_model(z_levels, *C[0,:])
        model_y = utils.polynomial_model(z_levels, *C[1,:])
        ax.plot(model_x, model_y, linestyle = '', marker = '.', color = 'k')

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xlim((0, 1))
    ax.set_ylim((0, 1))
    if not (out_png is None):
        wrapup(out_png)
    else:
        plt.show(); plt.close()

def scatter_flat(coefs, aspect_equal=True, ylim=None, xlim=None):
    """
    args
    ----
        coefs :  2D ndarray of shape (n_signals, 2), two
            columns of Zernike coefficients

    """
    if isinstance(coefs, pd.DataFrame):
        coefs = np.asarray(coefs)
    assert len(coefs.shape) == 2
    n_signals, n_coefs = coefs.shape
    fig, ax = plt.subplots(figsize = (4, 4))
    colors = sns.color_palette('viridis', 2).as_hex()
    ax.scatter(coefs[:,0], coefs[:,1], c = colors[0], s = 10)
    ax.set_xlabel('coef 0')
    ax.set_ylabel('coef 1')
    if aspect_equal: ax.set_aspect('equal')

    if not (ylim is None): ax.set_ylim(ylim)
    if not (xlim is None): ax.set_ylim(xlim)

    plt.show(); plt.close()

def plot_dist(array, bin_size):
    array_max = max(array)
    array_min = min(array)
    delta = array_max - array_min 
    n_bins = int(delta / bin_size) + 1
    bin_seq = [bin_size*i for i in range(n_bins)]
    bar_width = bin_size * 0.8

    histo, edges = np.histogram(array, bins = bin_seq)
    bin_centers = edges[:-1] + (edges[1] - edges[0])/2.0
    fig, ax = plt.subplots(figsize = (3, 3))
    ax.bar(bin_centers, histo, width = bar_width, color = 'r', edgecolor = 'k',)
    plt.show(); plt.close()

def plot_2d_poly_fit(coefs, C, ylim=None, xlim=None, aspect_equal=False):
    """
    args
    ----
        coefs :  3D ndarray of shape (n_samples, n_planes, 2)
        C :  2D ndarray of shape (2, m)

    """
    n_samples, n_planes, n_coefs = coefs.shape 
    n_coefs, m = C.shape 

    z_levels = np.arange(n_planes)
    fit_0 = utils.polynomial_model(z_levels, *C[0,:])
    fit_1 = utils.polynomial_model(z_levels, *C[1,:])

    fig, ax = plt.subplots(figsize = (4, 4))
    ax.plot(
        coefs[:,:,0].flatten(),
        coefs[:,:,1].flatten(),
        marker = '.',
        color = 'r',
        linestyle = '', 
        markersize = 5,
        label = 'Data',
    )
    ax.plot(
        fit_0, 
        fit_1,
        color = 'k',
        linestyle ='--',
        label = 'Polynomial model',
    )
    ax.set_xlabel('Coefficient 0')
    ax.set_ylabel('Coefficient 1')
    ax.legend(frameon = False, loc = 'upper left')
    if aspect_equal: ax.set_aspect('equal')
    if not (ylim is None): ax.set_ylim(ylim)
    if not (xlim is None): ax.set_ylim(xlim)
    plt.show(); plt.close()

def plot_2d_poly_fit_flat(coefs, C, z_levels, ylim=None, xlim=None,
    aspect_equal=True, model_type='points'):
    """
    args
    ----
        coefs :  2D ndarray of shape (n_samples, 2)
        C :  2D ndarray of shape (2, m)
        z_levels : 1D ndarray 

        model_type :  points or line

    """
    n_samples, n_coefs = coefs.shape 
    n_planes = z_levels.shape[0]
    n_coefs, m = C.shape 

    fit_0 = utils.polynomial_model(z_levels, *C[0,:])
    fit_1 = utils.polynomial_model(z_levels, *C[1,:])

    fig, ax = plt.subplots(figsize = (4, 4))
    ax.plot(
        coefs[:,0],
        coefs[:,1],
        marker = '.',
        color = 'r',
        linestyle = '', 
        markersize = 5,
        label = 'Data',
    )
    if model_type == 'line':
        ax.plot(
            fit_0, 
            fit_1,
            color = 'k',
            linestyle ='--',
            label = 'Polynomial model',
        )
    elif model_type == 'points':
        ax.plot(
            fit_0, 
            fit_1,
            color = 'k',
            linestyle ='',
            marker = '.',
            markersize = 5,
            label = 'Polynomial model',
        )
    ax.set_xlabel('Coefficient 0')
    ax.set_ylabel('Coefficient 1')
    ax.legend(frameon = False, loc = 'upper left')

    if not (ylim is None): ax.set_ylim(ylim)
    if not (xlim is None): ax.set_xlim(xlim)
    if aspect_equal: ax.set_aspect('equal')

    plt.show(); plt.close()

def plot_poly_with_point(C, z_levels, z_guess, point):
    poly_x = utils.polynomial_model(z_levels, *C[0,:])
    poly_y = utils.polynomial_model(z_levels, *C[1,:])
    proj_x = utils.polynomial_model(np.asarray([z_guess]), *C[0,:])
    proj_y = utils.polynomial_model(np.asarray([z_guess]), *C[1,:])

    fig, ax = plt.subplots(figsize = (4, 4))
    ax.plot(poly_x, poly_y, color = 'k', linestyle = '--')
    ax.plot([point[0], proj_x], [point[1], proj_y], linestyle = '--', marker = None,
        color = 'r')
    ax.plot([point[0]], [point[1]], linestyle = '', marker = '.', markersize = 20,
        color = 'r')
    ax.plot(proj_x, proj_y, linestyle = '', marker = '.',
        markersize = 10, color = 'k')

    ax.set_aspect('equal')

    plt.show(); plt.close()

#def plot_affine_field(matched_locs, field_coefs_y, field_coefs_x):
def plot_affine_field(matched_locs, affine_field_matrix, offset):
        field_coefs_y = np.array([affine_field_matrix[0,0], affine_field_matrix[0,1], offset[0]])
        field_coefs_x = np.array([affine_field_matrix[1,0], affine_field_matrix[1,1], offset[1]])

        matched_locs['y_shift_predicted'] = utils.affine_model(
            np.asarray(matched_locs[['y_pixels', 'x_pixels']]),
            *field_coefs_y,
        )
        fig, ax = plt.subplots(1, 2, figsize = (8, 4))
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = matched_locs,
            hue = 'y_shift',
            ax = ax[0],
            hue_norm = (matched_locs['y_shift_predicted'].min(), matched_locs['y_shift_predicted'].max()),
        )
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = matched_locs,
            hue = 'y_shift_predicted',
            ax = ax[1],
            hue_norm = (matched_locs['y_shift_predicted'].min(), matched_locs['y_shift_predicted'].max()),
        )
        ax[0].set_title('y shift, observed')
        ax[1].set_title('y shift, predicted')
        plt.show(); plt.close()


        matched_locs['x_shift_predicted'] = utils.affine_model(
            np.asarray(matched_locs[['y_pixels', 'x_pixels']]),
            *field_coefs_x,
        )
        fig, ax = plt.subplots(1, 2, figsize = (8, 4))
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = matched_locs,
            hue = 'x_shift',
            ax = ax[0],
            hue_norm = (matched_locs['x_shift_predicted'].min(), matched_locs['x_shift_predicted'].max()),
        )
        sns.scatterplot(
            x = 'x_pixels',
            y = 'y_pixels',
            data = matched_locs,
            hue = 'x_shift_predicted',
            ax = ax[1],
            hue_norm = (matched_locs['x_shift_predicted'].min(), matched_locs['x_shift_predicted'].max()),
        )
        ax[0].set_title('x shift, observed')
        ax[1].set_title('x shift, predicted')
        plt.show(); plt.close()

def show_eig_component(data, eig_index=0, lim=None, head=None):
    """
    Visualize the amount of each data point in the
    a particular eigenvector.

    args
    ----
        data :  2D ndarray of shape (n_data_points, 2)
            or shape (n_data_points, 3)
        eig_index :  int, the index of the corresponding
            eigenvector in order of decreasing eigenvalue
        lim :  (float, float), limits for plot axes
        head :  int, the nubmer of first values to show

    """
    assert len(data.shape) == 2
    data = np.asarray(data)

    data_c = data - data.mean(axis = 0)
    p, vec = utils.eig_component(data, eig_index=eig_index)
    data_proj = np.asarray([p]).T.dot(np.asarray([vec]))

    if not (head is None):
        data_c = data_c[:head, :]
        data_proj = data_proj[:head, :]

    if data.shape[1] == 2:
        fig, ax = plt.subplots(figsize = (4, 4))

        ax.scatter(
            data_c[:,0],
            data_c[:,1],
            s = 5,
            c = 'r',
        )
        ax.scatter(
            data_proj[:,0],
            data_proj[:,1],
            s = 5,
            c = 'k',
        )
        ax.set_xlabel('Parameter 0')
        ax.set_ylabel('Parameter 1')

        if not (lim is None):
            ax.set_xlim(lim)
            ax.set_ylim(lim)

        plt.show(); plt.close()

    elif data.shape[1] == 3:
        fig = plt.figure(figsize = (8, 4))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(
            data_c[:,0], data_c[:,1], data_c[:,2],
            s = 5, c = 'r',
        )
        ax.scatter(
            data_proj[:,0], data_proj[:,1], data_proj[:,2],
            s = 5, c = 'k',
        )

        ax.set_xlabel('Parameter 0')
        ax.set_ylabel('Parameter 1')
        ax.set_zlabel('Parameter 2')

        if not (lim is None):
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)

        plt.show(); plt.close()

def show_eig_projection(data, eig_index=0, lim=None, head=None,
    aspect_equal=False):
    """
    Project a set of data onto one of its eigenvectors.

    args
    ----
        data :  2D ndarray of shape (n_data_points, 2)
            or shape (n_data_points, 3)
        eig_index :  int, the index of the corresponding
            eigenvector in order of decreasing eigenvalue
        lim :  (float, float), limits for plot axes
        head :  int, the number of first values to show

    """
    assert len(data.shape) == 2
    data = np.asarray(data)

    # Diagonalize the covariance matrix and get
    # the inverse eigenvectors
    V, X = utils.principal_components(data)
    X_inv = np.linalg.inv(X)

    # Project the centered data onto the desired
    # eigenvector
    data_c = data - data.mean(axis = 0)
    vec = X[:, eig_index]
    data_proj = utils.project(data_c, vec)

    # Get the component of the data along the 
    # desired eigenvector
    eig_comps = utils.eig_component(data_proj, eig_index=0)

    # If desired, only show the first few observations
    if not (head is None):
        data_c = data_c[:head, :]
        data_proj = data_proj[:head, :]

    # 2D plot - just two parameters
    if data.shape[1] == 2:
        fig, ax = plt.subplots(figsize = (4, 4))

        ax.scatter(
            data_c[:,0],
            data_c[:,1],
            s = 5,
            c = 'r',
        )
        ax.scatter(
            data_proj[:,0],
            data_proj[:,1],
            s = 5,
            c = 'k',
        )
        ax.set_xlabel('Parameter 0')
        ax.set_ylabel('Parameter 1')

        if not (lim is None):
            ax.set_xlim(lim)
            ax.set_ylim(lim)
        if aspect_equal:
            ax.set_aspect('equal')

        plt.show(); plt.close()

    # 3D plot - three parameters
    elif data.shape[1] == 3:
        fig = plt.figure(figsize = (8, 4))
        ax = fig.add_subplot(111, projection = '3d')
        ax.scatter(
            data_c[:,0], data_c[:,1], data_c[:,2],
            s = 5, c = 'r',
        )
        ax.scatter(
            data_proj[:,0], data_proj[:,1], data_proj[:,2],
            s = 5, c = 'k',
        )

        ax.set_xlabel('Parameter 0')
        ax.set_ylabel('Parameter 1')
        ax.set_zlabel('Parameter 2')

        if not (lim is None):
            ax.set_xlim(lim)
            ax.set_ylim(lim)
            ax.set_zlim(lim)
        if aspect_equal:
            ax.set_aspect('equal')

        plt.show(); plt.close()

def overlay_trajs(
    nd2_file,
    trajs,
    start_frame,
    stop_frame,
    channel_idx = 0,
    out_tif = None,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    white_out_singlets = True,
): 
    n_frames_plot = stop_frame - start_frame + 1
    reader = zio.BiplaneImageFileReader(nd2_file)
    crosshair_len = 3 * upsampling_factor 

    N, M, n_frames = reader.get_shape()

    N_up = N * upsampling_factor 
    M_up = M * upsampling_factor
    image_min, image_max = reader.min_max(start_frame, stop_frame, c=channel_idx)
    vmin = image_min
    vmax = image_max * vmax_mod 

    trajs = trajs.assign(color_idx = (trajs['traj_idx'] * 173) % 256)
    n_locs = len(trajs)
    if channel_idx == 0:
        required_columns = ['frame_idx', 'traj_idx', 'y_pixels', 'x_pixels']
    elif channel_idx == 1:
        required_columns = ['frame_idx', 'traj_idx', 'y_pixels_ch1', 'x_pixels_ch1']
    if any([c not in trajs.columns for c in required_columns]):
        raise RuntimeError('overlay_trajs: dataframe must contain frame_idx, traj_idx, y_pixels, x_pixels')

    # Convert to ndarray -> faster indexing
    locs = np.asarray(trajs[required_columns])

    locs[:,:2] = locs[:,:2] + 0.5

    # Convert to upsampled pixels
    locs[:, 2:] = locs[:, 2:] * upsampling_factor
    locs = locs.astype('int64') 

    # Add a unique random index for each trajectory
    new_locs = np.zeros((locs.shape[0], 5), dtype = 'int64')
    new_locs[:,:4] = locs 
    new_locs[:,4] = (locs[:,1] * 173) % 256
    locs = new_locs 

    # If the length of a trajectory is 1, then make its color white
    if white_out_singlets:
        for traj_idx in range(locs[:,1].max()):
            if (locs[:,1] == traj_idx).sum() == 1:
                locs[(locs[:,1] == traj_idx), 4] = -1

    # Do the plotting
    colors = generate_rainbow_palette()

    result = np.zeros((n_frames_plot, N_up, M_up * 2 + upsampling_factor, 4), dtype = 'uint8')
    frame_exp = np.zeros((N_up, M_up), dtype = 'uint8')
    for frame_idx in tqdm(range(n_frames_plot)):
        frame = reader.get_frame(t = frame_idx + start_frame, c = channel_idx).astype('float64')
        frame_rescaled = ((frame / vmax) * 255)
        frame_rescaled[frame_rescaled > 255] = 255 
        frame_8bit = frame_rescaled.astype('uint8')

        for i in range(upsampling_factor):
            for j in range(upsampling_factor):
                frame_exp[i::upsampling_factor, j::upsampling_factor] = frame_8bit

        result[frame_idx, :, :M_up, 3] = frame_exp.copy()
        result[frame_idx, :, M_up + upsampling_factor:, 3] = frame_exp.copy()

        result[frame_idx, :, M_up:M_up+upsampling_factor, :] = 255

        for j in range(3):
            result[frame_idx, :, :M_up, j] = frame_exp.copy()
            result[frame_idx, :, M_up + upsampling_factor:, j] = frame_exp.copy()

        locs_in_frame = locs[(locs[:,0] == frame_idx + start_frame).astype('bool'), :]

        for loc_idx in range(locs_in_frame.shape[0]):

            # Get the color corresponding to this trajectory
            color_idx = locs_in_frame[loc_idx, 4]
            if color_idx == -1:
                color = np.array([255, 255, 255, 255]).astype('uint8')
            else:
                color = colors[color_idx, :]

            try:
                result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
            except (KeyError, ValueError, IndexError) as e2: #edge loc
                pass
            for j in range(1, crosshair_len + 1):
                try:
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] - j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2] + j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
                    result[frame_idx, locs_in_frame[loc_idx, 2] - j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color
                except (KeyError, ValueError, IndexError) as e3:  #edge loc 
                    continue 

    if out_tif == None:
        out_tif = 'default_overlay_trajs.tif'

    tifffile.imsave(out_tif, result)
    reader.close()

    if sys.platform == 'darwin':
        os.system('open %s -a Fiji' % out_tif)

def overlay_trajs_color_z(
    nd2_file,
    trajs,
    start_frame,
    stop_frame,
    z_col,
    channel_idx = 0,
    out_tif = None,
    vmax_mod = 0.7,
    upsampling_factor = 2,
):
    n_frames_plot = stop_frame - start_frame + 1
    reader = zio.BiplaneImageFileReader(nd2_file)
    crosshair_len = 3 * upsampling_factor 

    N, M, n_frames = reader.get_shape()

    N_up = N * upsampling_factor 
    M_up = M * upsampling_factor
    image_min, image_max = reader.min_max(start_frame, stop_frame, c=channel_idx)
    vmin = image_min
    vmax = image_max * vmax_mod 

    trajs = trajs.assign(color_idx = (trajs['traj_idx'] * 173) % 256)
    n_locs = len(trajs)
    if channel_idx == 0:
        required_columns = ['frame_idx', 'traj_idx', 'y_pixels', 'x_pixels', z_col]
    elif channel_idx == 1:
        required_columns = ['frame_idx', 'traj_idx', 'y_pixels_ch1', 'x_pixels_ch1', z_col]
    if any([c not in trajs.columns for c in required_columns]):
        raise RuntimeError('overlay_trajs: dataframe must contain frame_idx, traj_idx, y_pixels, x_pixels')

    # Convert to ndarray -> faster indexing
    locs = np.asarray(trajs[required_columns])

    locs[:,:2] = locs[:,:2] + 0.5

    # Convert to upsampled pixels
    locs[:, 2:] = locs[:, 2:] * upsampling_factor
    locs = locs.astype('int64') 

    # Rescale the z_values
    locs[:,4] = locs[:,4] - locs[:,4].min()
    locs[:,4] = locs[:,4] / locs[:,4].max()

    # Add a unique random index for each trajectory
    new_locs = np.zeros((locs.shape[0], 6), dtype = 'int64')
    new_locs[:,:5] = locs 
    new_locs[:,5] = locs[:,4] * 256
    locs = new_locs 

    # If the length of a trajectory is 1, then make its color white
    if white_out_singlets:
        for traj_idx in range(locs[:,1].max()):
            if (locs[:,1] == traj_idx).sum() == 1:
                locs[(locs[:,1] == traj_idx), 4] = -1

    # Do the plotting
    colors = generate_rainbow_palette()

    result = np.zeros((n_frames_plot, N_up, M_up * 2 + upsampling_factor, 4), dtype = 'uint8')
    frame_exp = np.zeros((N_up, M_up), dtype = 'uint8')
    for frame_idx in tqdm(range(n_frames_plot)):
        frame = reader.get_frame(t = frame_idx + start_frame, c = channel_idx).astype('float64')
        frame_rescaled = ((frame / vmax) * 255)
        frame_rescaled[frame_rescaled > 255] = 255 
        frame_8bit = frame_rescaled.astype('uint8')

        for i in range(upsampling_factor):
            for j in range(upsampling_factor):
                frame_exp[i::upsampling_factor, j::upsampling_factor] = frame_8bit

        result[frame_idx, :, :M_up, 3] = frame_exp.copy()
        result[frame_idx, :, M_up + upsampling_factor:, 3] = frame_exp.copy()

        result[frame_idx, :, M_up:M_up+upsampling_factor, :] = 255

        for j in range(3):
            result[frame_idx, :, :M_up, j] = frame_exp.copy()
            result[frame_idx, :, M_up + upsampling_factor:, j] = frame_exp.copy()

        locs_in_frame = locs[(locs[:,0] == frame_idx + start_frame).astype('bool'), :]

        for loc_idx in range(locs_in_frame.shape[0]):

            # Get the color corresponding to this trajectory
            color_idx = locs_in_frame[loc_idx, 5]
            if color_idx == -1:
                color = np.array([255, 255, 255, 255]).astype('uint8')
            else:
                color = colors[color_idx, :]

            try:
                result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
            except (KeyError, ValueError, IndexError) as e2: #edge loc
                pass
            for j in range(1, crosshair_len + 1):
                try:
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] + j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2], M_up + locs_in_frame[loc_idx, 3] - j + upsampling_factor, :] = color
                    result[frame_idx, locs_in_frame[loc_idx, 2] + j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color 
                    result[frame_idx, locs_in_frame[loc_idx, 2] - j, M_up + locs_in_frame[loc_idx, 3] + upsampling_factor, :] = color
                except (KeyError, ValueError, IndexError) as e3:  #edge loc 
                    continue 

    if out_tif == None:
        out_tif = 'default_overlay_trajs.tif'

    tifffile.imsave(out_tif, result)
    reader.close()

    if sys.platform == 'darwin':
        os.system('open %s -a Fiji' % out_tif)

def overlay_trajs_interactive(
    nd2_file,
    trajs,
    start_frame,
    stop_frame,
    vmax_mod = 1.0,
    upsampling_factor = 1,
    crosshair_len = 'dynamic',
    continuous_update = True,
    white_out_singlets = True,
):
    if crosshair_len == 'dynamic':
        crosshair_len = int(3 * upsampling_factor)

    if type(trajs) == type('') and 'Tracked.mat' in trajs:
        out_tif = '%soverlay.tif' % tracked_mat_file.replace('Tracked.mat', '')
        overlay_trajs_tracked_mat(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    elif type(trajs) == type('') and ('.trajs' in trajs or '.txt' in trajs):
        out_tif = '%s_overlay.tif' % os.path.splitext(trajs)[0]
        trajs, metadata = spazio.load_locs(trajs)
        print(trajs.columns)
        overlay_trajs_df(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    elif type(trajs) == type(pd.DataFrame([])):
        out_tif = 'default_overlay.tif'
        overlay_trajs_df(
            nd2_file,
            trajs,
            start_frame,
            stop_frame,
            out_tif = out_tif,
            vmax_mod = vmax_mod,
            upsampling_factor = upsampling_factor,
            crosshair_len = crosshair_len,
            white_out_singlets = white_out_singlets,
        )
    else:
        raise RuntimeError('overlay_trajs_interactive: trajs argument not understood')

    reader = tifffile.TiffFile(out_tif)
    n_frames = len(reader.pages)
    
    def update(frame_idx):
        fig, ax = plt.subplots(figsize = (14, 7))
        page = reader.pages[frame_idx].asarray()
        page[:,:,-1] = 255
        ax.imshow(
            page,
        )
        ax.set_xticks([])
        ax.set_yticks([])
        for spine_dir in ['top', 'bottom', 'left', 'bottom']:
            ax.spines[spine_dir].set_visible(False)
        plt.show(); plt.close()

    interact(update, frame_idx = widgets.IntSlider(
        min=0, max=n_frames, continuous_update=continuous_update))

def generate_rainbow_palette(n_colors = 256):
    '''
    Generate a rainbow color palette in RGBA format.
    '''
    result = np.zeros((n_colors, 4), dtype = 'uint8')
    for color_idx in range(n_colors):
        result[color_idx, :] = (np.asarray(cm.gist_rainbow(color_idx)) * \
            255).astype('uint8')
    return result 




