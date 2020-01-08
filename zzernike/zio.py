'''
zio.py - I/O for zzernike

'''
import os
import sys

# Numerical stuff
import numpy as np 

# Nikon ND2 file reader
from nd2reader import ND2Reader

# TIF(F) file reader
import tifffile

class BiplaneImageFileReader():
    def __init__(self, nd2_filename):
        self.file_name = nd2_filename
        self.reader = ND2Reader(nd2_filename)
        self.is_closed = False 

    def get_frame(self, z = 0, t = 0, c = 0):
        return self.reader.get_frame_2D(t = t, c = c, z = z).astype('float64')

    def get_shape(self):
        N = self.reader.metadata['height']
        M = self.reader.metadata['width']
        n_frames = self.reader.metadata['total_images_per_channel']
        return N, M, n_frames 

    def close(self):
        self.reader.close()
        self.is_closed = True 

    def get_zstack(self, c = 0, t = 0):
        N, M, n_frames = self.get_shape()
        result = np.zeros((n_frames, N, M), dtype = 'float64')
        for z_idx in range(n_frames):
            result[z_idx, :, :] = self.get_frame(z = z_idx, c = c, t = t)
        return result 

    def max_int_projection(self, c = 0, t = 0):
        zstack = self.get_zstack(c = c, t = t)
        return zstack.max(axis = 0)

    def get_time_series(self, start_t, stop_t, c = 0, z = 0):
        N, M, n_frames = self.get_shape()
        if stop_t >= n_frames:
            stop_t = n_frames - 1

        T = stop_t - start_t + 1
        result = np.zeros((T, N, M), dtype = 'float64')
        for t_idx in range(T):
            result[t_idx, :, :] = self.reader.get_frame_2D(
                t = t_idx + start_t, 
                c = c, 
                z = z,
            )
        return result 

    def get_zstack_subwindow(self, yx_coords, window_size, c = 0, t = 0):
        half_w = int(window_size) // 2
        y_field, x_field = np.mgrid[
            yx_coords[0]-half_w : yx_coords[0]+half_w+1,
            yx_coords[1]-half_w : yx_coords[1]+half_w+1,
        ]
        zstack = self.get_zstack(c = c, t = t)
        subwindow = zstack[:, y_field, x_field]
        return subwindow, y_field, x_field 

    def show_frame(self, frame_idx, vmax = 1.0):
        image_ch0 = self.get_frame(t = frame_idx, c = 0)
        image_ch1 = self.get_frame(t = frame_idx, c = 1)

        fig, ax = plt.subplots(1, 2, figsize = (8, 4))
        ax[0].imshow(image_ch0, cmap = 'gray', vmax = image_ch0.max() * vmax)
        ax[1].imshow(image_ch1, cmap = 'gray', vmax = image_ch1.max() * vmax)
        plt.show(); plt.close()

    def min_max(self, start_frame, stop_frame, c = 0):
        c_min, c_max = np.inf, 0 
        N, M, n_frames = self.get_shape()
        if n_frames - 1 < stop_frame:
            stop_frame = n_frames - 1
        for frame_idx in range(start_frame, stop_frame+1):
            frame = self.get_frame(t=frame_idx, c=c)
            frame_min = frame.min()
            frame_max = frame.max()
            if frame_min < c_min:
                c_min = frame_min 
            if frame_max > c_max:
                c_max = frame_max 
        return c_min, c_max 


