# Kenneth Meyer
# CSE 382M HW1
# 1/23/23

"""
6. [3 pts] In your preferred programming languague, uniformly generate 100
points on the surface of a sphere in 3-dimensions and in 100-dimensions.
Create a histogram of all distances between the pairs of points in both
cases and discuss.
"""
# uniformly sample!

import numpy as np
import matplotlib.pyplot as plt

# use spherical gaussians as discussed in class

def sphere_generator(pts, dim):
    """
        Uniformly samples "pts" points on a sphere of dimension "dim".

        Parameters
        ----------
        pts : int
            number of points to randomly generate on sphere surface
        dim : int
            dimension of the sphere
        
        Returns
        -------
        data : the sampled points on the surface
    """

    data = np.random.normal(size=(pts,dim))
    #print(data.shape)
    
    # normalize the data
    for i in range(0,data.shape[0]):
        data[i,:] = data[i,:]/np.linalg.norm(data[i,:])

    print(data.shape)
    return data

# don't need to access the pdf; just need to compute the distance between the points.

# compute the distance between each point pair
def compute_distances(data : np.array) -> np.array:
    """
    Computes distance between all point pairs in the data object given
    
    """

    # generate output data
    n = data.shape[0]
    pairs = np.math.factorial(n)/(2 * np.math.factorial(n-2))
    distances = np.zeros(int(pairs))

    # loop through each pair to generate distances
    count = 0 # counter bc I'm lazy
    for i in range(0,n):
        for j in range(i+1,n):
            distances[count] = np.linalg.norm(data[i,:] - data[j,:])
            count = count + 1
    return distances

# generate results for 3d and 100d cases, 100 points each
data_3d = sphere_generator(100,3)
data_100d = sphere_generator(100,100)

# generate distances for each sample
distances_3d = compute_distances(data_3d)
distances_100d = compute_distances(data_100d)

# plot a histogram of the results - 3d sphere
counts, bins = np.histogram(distances_3d)
plt.stairs(counts, bins)
plt.savefig("hw1_sphere_3d.png")

# plot a histogram of the results - 100d sphere
counts, bins = np.histogram(distances_100d)
plt.stairs(counts, bins)
plt.legend(["3D Sphere", "100D Sphere"])
plt.savefig("hw1_sphere_100d.png")