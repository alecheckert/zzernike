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

# 3d plotting
from mpl_toolkits.mplot3d import Axes3D 

# I/O
import os
import sys
import tifffile

# Package utilities
from . import utils 

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

def scatter(coefs, ax=None):
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
    ax.set_xlabel('signal 0')
    ax.set_ylabel('signal 1')
    plt.show(); plt.close()

def plot_2d_poly_fit(coefs, C):
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



