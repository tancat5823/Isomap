import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets

class Isomap:
    """
    Isometric Mapping (isomap) demensionality reduction method.
    params:
    n_neighbors(int):      Number of nearest neighbors used to build the neighborhood graph.
    n_components(int):     Dimensions of the embedding.
    """
    def __init__(self, n_neighbors = 10, n_components = 2):
        self.n_neighbors = n_neighbors
        self.n_components = n_components

    def knn(self, distances):
        """
        Generate a symmetric k nearest neighbor graph. Takes distances, a nxn matrix of the weights describing the neighborhood graph.
        """
        n = distances.shape[0]                                                 #num of data points
        k = self.n_neighbors                                                   #neighbors each point connects to
        neighbors = np.argsort(distances, axis = 1)[:, 1 : k + 1]              #get the k neighbors from distance matrix for each point
        weighted = np.full((n, n), np.inf)                                     #nxn matrix filled with infinity
        rows = np.repeat(np.arange(n), k)                                      #row indices that pair with neighbors
        weighted[rows, neighbors.ravel()] = distances[rows, neighbors.ravel()] #fills edges
        weighted = np.minimum(weighted, weighted.T)                            #if i is js neares neighbor bu j is not is nearest neighbor, then j to i is infinite while i to j is finite, min to get finite distance between both
        np.fill_diagonal(weighted, 0.0)                                        #The diagonal is the distance from a point to itself, which is 0
        return weighted

    def floyd_warshall(weighted):
        """
        find shortest path between pairs and generate nxn geodesic distance matrix
        """
        fw = weighted.copy()
        for k in range(fw.shape[0]):
            #search minimum paths for each pair dist[i][j] = min(dist[i][j], dist[i][k] + dist[k][j])
            fw = np.minimum(fw, fw[:, k, None] + fw[None, k, :])
        return fw

    def MDS(self, X):
        """
        Multi Dimensional Scaling
        Converts the geodesic distance matrix to get a lower dimensional space while largely preserving preserving the pairwise geodesic distances
        """
        X = np.asarray(X, dtype = float)
        n = X.shape[0]                                           # number of rows
        if self.n_neighbors >= n:                                # cant have more neighbors than points, raise error
            raise ValueError("n_neighbors must be smaller than number of samples")
        
        self.X_train = X
        #make nxn matrix of euclidian distances between each pair of training points
        distances = (X**2).sum(1)[:, None] + (X**2).sum(1)[None, :] - 2.0 * X @ X.T
        distances = np.sqrt(np.maximum(distances, 0.0))
        self.distances = distances
        weighted = self.knn(distances)                            # generate graph weight matrix from the distances
        weighted = self.floyd_warshall(weighted)                  # convert to shortes path on neighbor edges
        #If any infinite values remain, points may not be able to reach each other which would cause MDS to fail
        if np.isinf(weighted).any():
            raise ValueError("Neighborhood graph is not connected, increase n_neighbors.")
        self.dist_matrix = weighted

        K = -0.5 * weighted**2                     # first half of double centering, transform square distance to inner product
        self.K_col_mean = K.mean(axis = 0)         # mean of columns
        self.K_all_mean = K.mean()                 # matrix mean
        #second half of double centering
        K = K -self.K_col_mean[None, :] - self.K_col_mean[:, None] + self.K_all_mean

        eigenvalues, eigenvectors = np.linalg.eigh(K)               # eigendecomposition
        idx = np.argsort(eigenvalues)[::-1][: self.n_components]    # pick top n_components eigenvalues, represents the direction with most variance
        self.eigenvalues = np.maximum(eigenvalues[idx], 1e-12)      # floors eigenvalues, prevents negatives or zeros
        self.eigenvectors = eigenvectors[:, idx]

        self.embedding = self.eigenvectors * np.sqrt(self.eigenvalues) # nxd matrix of the lowered demension coordinates
        return self

    def transform(self, X_new):
        """
        Add new points to the embedding and generates their lower dimension coordinates
        """
        X_new = np.asarray(X_new, dtype = float)
        distances = self.distances
        #find each new points k nearest training points
        neighbors = np.argsort(distances, axis = 1)[:, : self.n_neighbors]
        #m new points, n training points
        m, n = distances.shape
        new_weights = np.empty((m,n))
        
        #estimate distance from new point to training points
        for i in range(m):
            distance_i = distances[i, neighbors[i]]
            new_weights = (distance_i[:, None] + self.dist_matrix[neighbors[i]]).min(axis = 0)
        
        #double centering
        new_K = -0.5 * new_weights**2
        new_K = (new_K - new_K.mean(axis = 1, keepdims = True) - self.K_col_mean[None, :] + self.K_all_mean)
        #project centered kernel rows onto eigenvectors, gives a mxd array of low dimensional coordinates
        return new_K @ self.eigenvectors / np.sqrt(self.eigenvalues)

    def reconstruction_error(self):
        """
        Residual variance
        Gives the linear correlation between the geodesic distances and the euclidian distances
        """
        #nxn matrix of distances between points in the embedded space
        embedded_distances = (self.embedding**2).sum(1)[:, None] + (self.embedding**2).sum(1)[None, :] - 2.0 * self.embedding @ self.embedding.T
        embedded_distances = np.sqrt(np.maximum(embedded_distances, 0.0))
        #extract the upper triangle so we only have one set of distances
        tri = np.triu_indices_from(embedded_distances, k=1)
        #Calculate pearson correlation
        r = np.corrcoef(self.dist_matrix[tri], embedded_distances[tri])[0, 1]
        #returns r^2 value
        return 1.0 - r**2



def main():
    n_samples = 1000
    X, t = datasets.make_swiss_roll(n_samples = n_samples, noise = 0.05, random_state = 0)

    rng = np.random.default_rng(0)
    perm = rng.permutation(n_samples)
    test_idx, train_idx = perm[:100], perm[100:]
    X_train, t_train = X[train_idx], t[train_idx]
    X_test, t_test = X[test_idx], t[test_idx]

    isomap = Isomap(n_neighbors = 10, n_components = 2)
    Y_train = isomap.fit_transform(X_train)
    Y_test = isomap.transform(X_test)

    print(f"Training points:     {X_train.shape[0]}")
    print(f"Top eigenvalues:     {isomap.eigenvalues}")
    print(f"Residual variance:   {isomap.reconstruction_error():.4f}")

    fig = plt.figure(figsize=(13, 4.5))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c = t_train, cmap="Spectral", s = 8)
    ax.set_title("Swiss Roll Isomap")

    ax = fig.add_subplot(1, 3, 2)
    ax.scatter(Y_train[:, 0], Y_train[:, 1], Y_train[:, 2], c = t_train, cmap="Spectral", s = 8)
    ax.set_title("Isomap Embedding")
    ax.set_aspect("equal", adjustable="datalim")

    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(Y_train[:, 0], Y_train[:, 1], c="lightgray", s = 6, label = "train")
    ax.scatter(Y_test[:, 0], Y_test[:, 1], c = t_test, cmap="Spectral", s = 18, edgecolor = "k", linewidths = 0.4, label = "held-out")
    ax.set_title("Out of Sample Points")
    ax.legend()
    ax.set_aspect("equal", adjustable = "datalim")

    plt.tight_layout()
    plt.savefig("Isomap_Swiss_Roll.png", dpi = 150)

if __name__ == "__main__":
    main()