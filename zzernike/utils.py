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

def affine_model(points, a, b, c):
    return a*points[:,0] + b*points[:,1] + c 

def principal_components(data, max_only=False):
    """
    Get the principal components of a set of data values.

    args
    ----
        data :  2D ndarray of shape (n_data_points, n_pars),
            a set of parameters for each data point
        max_only :  bool, only return the principal
            component with maximum variance

    returns
    -------
        (
            1D ndarray of shape (n_pars,), the eigenvalues;
            2D ndarray of shape (n_pars, n_pars), the
                corresponding eigenvectors
        )

    """
    assert len(data.shape) == 2

    # Make sure we're working with ndarrays
    data = np.asarray(data)

    # Center the data about the mean along each axis
    data_means = data.mean(axis = 0)
    data_c = data - data_means 

    # Calculate covariance matrix
    C = data_c.T.dot(data_c)

    # Diagonalize the covariance matrix
    V, X = np.linalg.eig(C)

    # Sort by decreasing eigenvalue
    order = np.argsort(V)[::-1]
    V = V[order]
    X = X[:,order]

    return V, X 

def projection_matrix(col_vectors):
    """
    Make a projection matrix from a set of column
    vectors.

    args
    ----
        col_vectors :  2D ndarray of shape (n_pars, n_vec)
            (multiple vectors) or 1D ndarray of shape (n_pars,)
            (single vector)

    returns
    -------
        2D ndarray of shape (n_pars, n_pars), the projection
            matrix

    """
    if len(col_vectors.shape) == 1:
        col_vectors = np.asarray([col_vectors]).T 

    return col_vectors.dot(np.linalg.inv(col_vectors.T.dot(col_vectors))).dot(col_vectors.T)

def project(data, col_vectors):
    """
    Project a set of data onto a set of column vectors.

    args
    ----
        data :  2D ndarray of shape (n_data_points, n_pars)
        col_vectors :  2D ndarray of shape (n_pars, n_vec) 
            (multiple vectors) or 1D ndarray of shape (n_pars,)
            (single vectors)

    returns
    -------
        2D ndarray of shape (n_data_points, n_vec), the
            projection of the data onto the set of vectors

    """
    # Make the projection matrix 
    P = projection_matrix(col_vectors)

    # Perform the projection 
    return P.dot(data.T).T 

def first_eig(data):
    """
    Return the component of the data in the principal
    eigenvector.

    args
    ----
        data :  2D ndarray of shape (n_data_points, n_pars),
            a set of observations with n_pars each

    returns
    -------
        (
            1D ndarray of shape (n_data_points), the component
                of the unitary eigenvector for each data point;
            1D ndarray of shape (n_pars,), the principal
                eigenvector
        )

    """
    # Center the data
    data_c = data - data.mean(axis=0)

    # Calculate covariance matrix
    C = data_c.T.dot(data_c)

    # Diagonalize the covariance matrix
    V, X = np.linalg.eig(C)

    # Sort by the size of the eigenvalue
    order = np.argsort(V)[::-1]
    V = V[order]
    X = X[:,order]

    # Invert the eigenvector matrix and
    # take the linear combination of data
    # that returns the first eigenvector
    # (corresponding to largest eigenvalue)
    X_inv = np.linalg.inv(X)
    vec = X_inv[0,:]

    # Calculate the amount of that eigenvector
    # in each data point
    return vec.dot(data_c.T), X[:,0]

def eig_component(data, eig_index=0):
    """
    Return the component of data along a particular
    eigenvector.

    args
    ----
        data :  2D ndarray of shape (n_data_points,
            n_pars)
        eig_index :  int between 0 and n_pars-1, the
            index of the eigevector to use in 
            order of decreasing eigenvalue

    returns
    -------
        (
            1D ndarray of shape (n_data_points), the component
                in the eigenvector for each data point;
            1D ndarray of shape (n_pars,), the corresponding
                eigenvector
        )

    """
    # Center the data
    data_c = data - data.mean(axis=0)

    # Calculate covariance matrix
    C = data_c.T.dot(data_c)

    # Diagonalize the covariance matrix
    V, X = np.linalg.eig(C)

    # Sort by the size of the eigenvalue
    order = np.argsort(V)[::-1]
    V = V[order]
    X = X[:,order]

    # Invert the eigenvector matrix and
    # take the linear combination of data
    # that returns the first eigenvector
    # (corresponding to largest eigenvalue)
    X_inv = np.linalg.inv(X)
    vec = X_inv[eig_index,:]

    # Calculate the amount of that eigenvector
    # in each data point
    return vec.dot(data_c.T), X[:,eig_index]

# 
# Tracking utilities
# 
def connected_components(semigraph):
    '''
    Find independent subgraphs in a semigraph by a floodfill procedure.
    
    args
        semigraph : 2D binary np.array (only 0/1 values), representing
            a semigraph
    
    returns
        subgraphs : list of 2D np.array, the independent adjacency subgraphs;
        
        subgraph_y_indices : list of 1D np.array, the y-indices of each 
            independent adjacency subgraph;
            
        subgraph_x_indices : list of 1D np.array, the x-indices of each 
            independent adjacency subgraph;
        
        y_without_x : 1D np.array, the y-indices of y-nodes without any edges
            to x-nodes;
        
        x_without_y : 1D np.array, the x-indices of x-nodes without any edges
            to y-nodes.
            
    '''
    if semigraph.max() > 1:
        raise RuntimeError("connected_components only takes binary arrays")
        
    # The set of all y-nodes (corresponding to y-indices in the semigraph)
    y_indices = np.arange(semigraph.shape[0]).astype('uint16')
    
    # The set of all x-nodes (corresponding to x-indices in the semigraph)
    x_indices = np.arange(semigraph.shape[1]).astype('uint16')

    # Find y-nodes that don't connect to any x-node,
    # and vice versa
    where_y_without_x = (semigraph.sum(axis = 1) == 0)
    where_x_without_y = (semigraph.sum(axis = 0) == 0)
    y_without_x = y_indices[where_y_without_x]
    x_without_y = x_indices[where_x_without_y]
    
    # Consider the remaining nodes, which have at least one edge
    # to a node of the other class 
    semigraph = semigraph[~where_y_without_x, :]
    semigraph = semigraph[:, ~where_x_without_y]
    y_indices = y_indices[~where_y_without_x]
    x_indices = x_indices[~where_x_without_y]
    
    # For the remaining nodes, keep track of (1) the subgraphs
    # encoding connected components, (2) the set of original y-indices
    # corresponding to each subgraph, and (3) the set of original x-
    # indices corresponding to each subgraph
    subgraphs = []
    subgraph_y_indices = []
    subgraph_x_indices = []

    # Work by iteratively removing independent subgraphs from the 
    # graph. The list of nodes still remaining are kept in 
    # *unassigned_y* and *unassigned_x*
    unassigned_y, unassigned_x = (semigraph == 1).nonzero()
    
    # The current index is used to floodfill the graph with that
    # integer. It is incremented as we find more independent subgraphs. 
    current_idx = 2
    
    # While we still have unassigned nodes
    while len(unassigned_y) > 0:
        
        # Start the floodfill somewhere with an unassigned y-node
        semigraph[unassigned_y[0], unassigned_x[0]] = current_idx
    
        # Keep going until subsequent steps of the floodfill don't
        # pick up additional nodes
        prev_nodes = 0
        curr_nodes = 1
        while curr_nodes != prev_nodes:
            # Only floodfill along existing edges in the graph
            where_y, where_x = (semigraph == current_idx).nonzero()
            
            # Assign connected nodes to the same subgraph index
            semigraph[where_y, :] *= current_idx
            semigraph[semigraph > current_idx] = current_idx
            semigraph[:, where_x] *= current_idx
            semigraph[semigraph > current_idx] = current_idx
            
            # Correct for re-finding the same nodes and multiplying
            # them more than once (implemented in the line above)
            # semigraph[semigraph > current_idx] = current_idx
            
            # Update the node counts in this subgraph
            prev_nodes = curr_nodes
            curr_nodes = (semigraph == current_idx).sum()
        current_idx += 1 

        # Get the local indices of the y-nodes and x-nodes (in the context
        # of the remaining graph)
        where_y = np.unique(where_y)
        where_x = np.unique(where_x)

        # Use the local indices to pull this subgraph out of the 
        # main graph 
        subgraph = semigraph[where_y, :]
        subgraph = subgraph[:, where_x]

        # Save the subgraph
        if not (subgraph.shape[0] == 0 and subgraph.shape[0] == 0):
            subgraphs.append(subgraph)
        
            # Get the original y-nodes and x-nodes that were used in this
            # subgraph
            subgraph_y_indices.append(y_indices[where_y])
            subgraph_x_indices.append(x_indices[where_x])

        # Update the list of unassigned y- and x-nodes
        unassigned_y, unassigned_x = (semigraph == 1).nonzero()

    return subgraphs, subgraph_y_indices, subgraph_x_indices, y_without_x, x_without_y

def sq_radial_distance(vector, points):
    return ((vector - points) ** 2).sum(axis = 1)

def sq_radial_distance_array(points_0, points_1):
    '''
    args
        points_0    :   np.array of shape (N, 2), coordinates
        points_1    :   np.array of shape (M, 2), coordinates

    returns
        np.array of shape (N, M), the radial distances between
            each pair of points in the inputs

    '''
    array_points_0 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    array_points_1 = np.zeros((points_0.shape[0], points_1.shape[0], 2), dtype = 'float')
    for idx_0 in range(points_0.shape[0]):
        array_points_0[idx_0, :, :] = points_1 
    for idx_1 in range(points_1.shape[0]):
        array_points_1[:, idx_1, :] = points_0
    result = ((array_points_0 - array_points_1)**2).sum(axis = 2)
    return result 

