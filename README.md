# Isomap
Isomap implementation and application to swiss roll dataset

# Requirements
  - Python 3.9
  - NumPy
  - matplotlib.pyplot
  - scikit-learn

# How to run
Download the 'Isomap.py' file and run in terminal from that folder using:
'python Isomap.py'

# Expected output
'Top eigenvalues:        [670256.97893833  39627.5824787 ]'
'Residual variance:      0.0005'
'Straight line distance: 20.22'
'Isomap graph path:      91.90'
'True geodesic distance: 89.60'

# Using the class on your own dataset
Create a new file in the same directory as 'Isomap.py' and add the following:
'import numpy as np'
'from Isomap import Isomap'

Import your dataset for example:
'X = np.random.rand(500, 10)'

Use the isomap on your dataset:
'isomap = Isomap(n_neighbors = 10, n_components = 2)'
'Y = isomap.fit_transform(X)'

To test the isomap on new data X_new use 'isomap.transform()'
'Y_new = isomap.transform(X_new)'

This gives the low dimensional embedding of the data Y_new. You can plot it to visualize your data in lower dimension. Other useful information is stored in 'isomap.dist_matrix', the estimated geodesic distances, 'isomap.embedding' the raw low dimensional coordinates, or 'isomap.reconstruction_error()' to get the residual variance that shows how well the embedding preserved the geodesic distances.

# Methods and functions
'knn(distances)':             Builds nearest neighbor graph
'floyd_warshall(weighted)':   Geodesic distance estimates of shortest path between pairs
'MDS(X)':                     Runs MDS and returns the fitted object
'fit_transform(X)':           Builds neighborhood graph, computes geodesic distances and runs MDS and                                      returns the embedding
'transform(X_new)':           Embeds new points using the fitted model
'reconstruction_error()':     Returns the residual variance between the geodesic and embedded distances
'shortest_path(i, j)':        Returns the training point indices for the shortest path along the graph                                     between i and j
'true_geodesic':              Returns the exact geodesic distance between points on the swiss roll
