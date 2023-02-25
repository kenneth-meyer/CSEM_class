# Kenneth Meyer
# 2/18/23
# CSE 382M coding questions

import numpy as np
from scipy.cluster.vq import kmeans2
import matplotlib.pyplot as plt

### filepaths ###


### functions ###
def construct_mat():
    '''
        Constructs matrix A and B for question 1

        PARAMETERS
        ----------
        mat : str, 'A' or 'B' is expected

        OUTPUT
        -----
        A : 150 x 150 np.array, 3x3 block matrix
    
    '''

    p = 0.7
    q = 0.3

    # not sure if this is the most efficient way to generate A
    A_diag = np.ones((50,50))*p
    A_offdiag = np.ones((50,50))*q
        
    A_1 = np.hstack((A_diag,A_offdiag,A_offdiag))
    A_2 = np.hstack((A_offdiag,A_diag,A_offdiag))
    A_3 = np.hstack((A_offdiag,A_offdiag,A_diag))
    A = np.vstack((A_1,A_2,A_3))
        
    B = np.random.rand(150,150)

    # might be a faster way than for loops, check this out
    for i in range(0,150):
        for j in range(0,150):
            if B[i,j] > A[i,j]:
                A[i,j] = 1
            else:
                A[i,j] = 0

    # definitely didn't need to generate A as a block matrix, could compare
    # B[i,j] to 0.3 or 0.7 depending on what i and j are equal to!
    return A

def random_permutation(A):
    '''
        Generates random permutation to apply to A in Q1
    '''

    # need to permute rows AND columns (does this work?...)
    #rng = np.random.default_rng()
    #A = rng.permutation(A,axis=0)
    #A = rng.permutation(A,axis=1)

    # need to use A random permutation
    #I = np.eye(150)
    idx = np.arange(150)
    np.random.shuffle(idx) # shuffles idx

    # might be a better way to do this, idk
    P = np.zeros((150,150))

    print(type(P))
    print(type(idx))
    for i in range(0,150):
        P[i,idx[i]] = 1

    # permute rows and columns
    A_rand = P @ A @ P.T

    return A_rand

def q1_a(k=3):
    '''
        Runs code for q1a
    '''

    A = construct_mat()
    A = random_permutation(A)
    clusters = kmeans(A,3)
    return clusters

def q1_b():
    '''
        Runs code for q1b
    '''
    
    A = construct_mat()
    A = random_permutation(A)

    ks = np.arange(1,11)
    sum_squares_err = np.zeros(ks.shape[0])

    for i in range(0,ks.shape[0]):
        sum_squares_err[i] = kmeans(A,ks[i])[-1] 
        # last value is sum of squares

    fig = plt.figure()
    ax = fig.add_subplot()
    ax.scatter(ks,sum_squares_err)
    plt.savefig("./figs/q1b.png")

def q1_c():
    '''
        RUns code for q1c
    '''
    phi = np.random.rand(10,150)


### code to run if the script is run ###
if __name__ == "__main__":
    print("running hw3 code")
    clusters = q1_a() # no idea how to check if they are the correct clusters
    # yes? I found the correct clusters?
    q1_b()
    # ^ also not sure what this means, error keeps going down...is 10 clusters best??

    # c and d ask "do you find the correct clusters"
    # ^ and I still don't know how to answer this lol