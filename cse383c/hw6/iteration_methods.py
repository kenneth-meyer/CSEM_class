# Kenneth Meyer
# 12/6/22
# rayleigh and power iteration methods

import numpy as np
## matplotlib imports ##
from matplotlib.widgets import LassoSelector
from matplotlib.path import Path

import matplotlib.style
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

import scipy.optimize as opt

import scipy

# matplotlib options
# consider changing figsize and aspect ratio until I find something that fits beamer slides well
mpl.rcParams['figure.figsize'] = [8.0,8.0]
mpl.rcParams['figure.dpi'] = 125
mpl.rcParams['savefig.dpi'] = 600
mpl.rcParams['font.size'] = 11
# Setup font labels for 3d plots.
labelfont = {'family': 'serif',
        'color':  'red',
        'weight': 'normal',
        'size': 12,
        }

from scipy.stats import ortho_group # for (random) unitary matrices

def power_it(A: np.ndarray,max_its: int) -> tuple:
    '''
        Parameters
        ----------
        A : n x n matrix

        Returns
        -------
        (eig,eigvec)
        eig : converged eigenvalue
        eigvec : converged eigenvector
    '''

    v_ = np.random.rand(A.shape[0])
    v0 = v_/np.linalg.norm(v_)
    
    # add one extra entry to account for 0th iteration/initial guess
    v_k = np.zeros((A.shape[0],max_its + 1))
    lambda_k = np.zeros((max_its+1))

    v_k[:,0] = v0 # initial guess for the eigenvector

    #error = []

    for i in range(1,max_its+1):
        w = A @ v_k[:,i-1]
        v_k[:,i] = w/np.linalg.norm(w)
        lambda_k[i] = v_k[:,i].T @ A @ v_k[:,i]

    # ignore the 0th iteration when returning things.
    return lambda_k[1:], v_k[:,1:]


def rayleigh_it(A: np.ndarray,max_its: int, power_its: int) -> tuple:
    '''
        Parameters
        ----------
        A : n x n matrix
        max_its : int, maximum iterations for rayleigh to run
        power_its : int, number of iterations to warm up v0 with using power iteration

        Returns
        -------
        (eig,eigvec)
        eig : converged eigenvalue
        eigvec : converged eigenvector
    '''

    # get warmed up and find some initial guess for lambda!

    power_eigval, power_eigvec = power_it(A,power_its)

    #v_ = np.random.rand(A.shape[0])
    v_ = power_eigvec[:,-1]
    v0 = v_/np.linalg.norm(v_)
    # add one extra entry to account for 0th iteration/initial guess
    v_k = np.zeros((A.shape[0],max_its + 1))
    v_k[:,0] = v0 # initial guess for the eigenvector

    lambda_k = np.zeros((max_its+1))
    lambda_0 = v0.T @ A @ v0 # rayleigh quotient for first iteration
    lambda_k[0] = lambda_0

    n = A.shape[0]

    #error = []

    # algorithm 27.3 - rayleigh quotient iteration
    for i in range(1,max_its+1):
        # solve (A - lambda_[k-1]*I)w = v[k_1]
        w = np.linalg.inv(A - lambda_k[i-1]*np.eye(n)) @ v_k[:,i-1]
        v_k[:,i] = w/np.linalg.norm(w)
        lambda_k[i] = v_k[:,i].T @ A @ v_k[:,i]

    # ignore the 0th iteration when returning things.
    return lambda_k[1:], v_k[:,1:]

def plot_error(eigval: np.ndarray, error: np.ndarray):
    '''
        Parameters
        ----------
        eigval : n x 1 array of eigenvalues with each iteration (n)
        eigval : n x m array of eigenvectors with each iteration (n)
        error : n x 2 array of eigenvalue and eigenvector relative errors, respectively.
        max_eigval : maximum eigenvalue for a given matrix A
        max_eigvec : eigenvector corresponding to max eigenvalue

        Returns
        -------
        (eig,eigvec)
        eig : converged eigenvalue
        eigvec : converged eigenvector
    '''
    # fails if the number of its < number of eignevalues right now lol
    its = np.arange(1,np.max(eigval.shape)+1)

    fig = plt.figure()
    ax = fig.add_subplot()
    
    # plotting eigenvector and eigenvalue error for #1
    if len(error.shape) > 1:
        ax.semilogy(its,error[:,0])
        ax.semilogy(its,error[:,1])
        ax.legend(["eigenvalue error","eigenvector error"])
    # only plotting eigenvalue error for QR
    else:
        ax.semilogy(its,error)
        ax.legend(["eigenvalue error"])
    ax.set_xlabel("iteration")
    ax.set_ylabel("error")

    return fig,ax

def compute_error(eigval, eigvec, max_eigval: np.float64,max_eigvec: np.ndarray):
    '''
        Parameters
        ----------
        eigval : eigenvalues
        eigvec : eigenvalues, (m,n) -> (eigvec,nth iteration)
    
        Returns
        -------
        error : (n,2), relative error of eigenvalue and 
                eigenvector, respectively, for each iteration.
    '''

    num_its = eigval.shape[0] # number of iterations it took to converge
    error = np.zeros((eigval.shape[0],2))

    for i in range(0,num_its):
        error[i,0] = np.abs(eigval[i] - max_eigval)/max_eigval
        error[i,1] = np.linalg.norm(np.abs(eigvec[:,i])/np.linalg.norm(eigvec[:,i]) - np.abs(max_eigvec))/np.linalg.norm(max_eigvec)
        
    return error


###### 3a : examine power iteration and rayleigh iteration #######

n = 10
Q = ortho_group.rvs(dim=n)
L = np.diag(np.arange(1,n+1)*(1/n))
A_ = Q @ L @ Q.T

A = scipy.linalg.hessenberg(A_)
#QQ = ortho_group.rvs(dim=n)
#print(QQ @ A @ QQ.T)
#np.savetxt("checking_matrix.txt",A)

max_eig = 1 # known for all matricies A
eigval_A,eigvec_A = np.linalg.eig(A)
# extract the index for the maximum eigenvalue
idx = np.argmax(eigval_A)

#for i in range(0,eigval_A.shape[0]):
#    if np.isclose(eigval_A[i],max_eig):
#        idx = i
#        print("max eigenvalue ")
#idx = np.where(np.isclose(eigval_A,max_eig)) # might not work - check this

max_eigvec = eigvec_A[:,idx]
ax = plt.subplot()
im = ax.imshow(A)
plt.colorbar(im)
plt.savefig('A_hessenberg')



## finally running things - power iteration
max_its_power = 300
eigval_power, eigvec_power = power_it(A,max_its_power)
error = compute_error(eigval_power,eigvec_power, max_eig, max_eigvec)
fig,ax = plot_error(eigval_power,error)
ax.set_title("Power Iteration")
plt.savefig("Power_Iteration_result")

# checking eigenvalues, raw
fig = plt.figure()
ax = fig.add_subplot()
ax.plot(np.arange(max_its_power),eigval_power)
ax.set_title("Power iteration, 3a")
plt.savefig("power iteration test")

## finally running things - power iteration
max_its_rayleigh = 100
eigval_rayleigh, eigvec_rayleigh = rayleigh_it(A,max_its_rayleigh,5)
error_rayleigh = compute_error(eigval_rayleigh, eigvec_rayleigh, max_eig, max_eigvec)
fig,ax = plot_error(eigval_rayleigh,error_rayleigh)
ax.set_title("Rayleigh quotient with 5 warmup power iterations")
plt.savefig("rayleigh_quotient_result")

#### 3b: implementation of 28.1, "Pure QR algorithm" #####
def pure_QR(A: np.ndarray, max_its=None):
    '''
        Parameters
        ----------
        A : n x n numpy array representating a matrix
        tol (optional) : tolerance for convergence (L2 norm)
        max_its (optional) : maximum iterations the algorithm will complete

        Returns
        -------
        eigval
        eigvec
    '''

    # can maybe add optionality to make the tolerance cheaper

    # will likely just plot the error and search for the value in the array

    A_ = A

    # storing the diagonals of A to check what the eigenvalues are
    eigvals = np.zeros((A.shape[0],max_its))
    # not bothering checking the eigenvectors (maybe add later)

    for k in range(0,max_its):
        Q_k,R_k = np.linalg.qr(A_) # Q_k @ R_k = A_(k-1)
        A = R_k @ Q_k              # A_k = R_k @ Q_k

        eigvals[:,k] = np.diag(A)
        A_ = A # save A for the next iteration


        # consider saving the matrices to inspect what they are
        # consider implementing an option to control the tolerance of the method

    return A,eigvals

def error_pure_QR(true_eigvals,pure_qr_eigvals):
    '''
        Parameters
        ----------
        eigvals : actual eigenvales of the matrix A
        pure_qr_eigvals : eigenvalues as determined by the matrix A at each step
    
        Returns
        -------
        error : n_its x 1 array of errors at each iteration of the method
    '''
    n_its = np.max(pure_qr_eigvals.shape)
    error = np.zeros(n_its)

    # sort ground truth eigenvalues if not sorted already
    true_eigvals = np.sort(true_eigvals)

    # compute error at each step
    for k in range(0,n_its):
        eig_k = pure_qr_eigvals[:,k]
        error[k] = np.linalg.norm(true_eigvals - np.sort(eig_k))

    return error

def shifted_qr(A,zero_tol,max_its):
    '''
        Modified shifted QR algorithm. Removes last row and column when last eigenvalue has been "found"

        Parameters
        ----------
        A : TRI-diagonal matrix
        zero_tol : tolerance used to determine when the entries on the off-diagonals are sufficiently small
        max_its : maximum number of iterations the solver will use.

        Returns
        -------
        eigenvalues or something
    '''

    A_k_1 = A
    #eigs = np.zeros((A.shape[0],max_its))
    #count = A.shape[0] # keep track of eigenvalues being found

    eigs = np.zeros(A.shape[0])
    n = A.shape[0] - 1

    for i in range(0,max_its):
        u_k = A_k_1[-1,-1] # pick the shift!
        Q_k, R_k = np.linalg.qr(A_k_1 - u_k*np.eye(A_k_1.shape[0]))
        A_k = R_k @ Q_k + u_k*np.eye(A_k_1.shape[0])

        # save eigenvalues, repeating those that were previously saved
        #eigs[:A_k.shape[0],i] = np.diag(A_k)
        #if count < A.shape[0]:
        #    eigs[count:,i] = eigs[count:,i-1]
        # ^ some entries will be zero, need to fix this...

        # check if the off diagonals above and below the last diagonal entry are close to zero
        if A_k[-1,-2] < zero_tol and A_k[-2,-1] < zero_tol:
            eigs[n] = A_k[-1,-1] # save eigenvalue
            n = n - 1
            A_k = A_k[:-1,:-1] # remove last row and column

        A_k_1 = A_k

        # checking for when the problem is done
        if np.max(A_k.shape) < 2:
            eigs[n] = A_k[-1,-1] # save eigenvalue
            print("Results for shifted QR:")
            print("solution reached")
            print("num_its: " + str(i))
            #print("last eigenvalue: " + str(eigs[n]))
            break

    return eigs

eig_A_ordered = np.sort(eigval_A)
A_pure_QR,eig_pure_QR = pure_QR(A,max_its=200)
err_pure_qr = error_pure_QR(eig_A_ordered,eig_pure_QR)
fig,ax = plot_error(eig_pure_QR,err_pure_qr)
ax.set_title("Pure QR, Alg. 28.1, Eigenvalue convergence")
plt.savefig("Pure QR Cumulative Eigenvalue Error")

# A is tridiagonal, there is nothing to be done for Q3
# some work for Q3:

eigs_shifted_qr = shifted_qr(A,3e-16,10000)
# error functions are the same for both QR methods
#err_shifted_qr = error_pure_QR(eig_A_ordered,eigs_shifted_qr)
#fig,ax = plot_error(eigs_shifted_qr,err_shifted_qr)
#ax.set_title("Shifted QR, Alg. 28.2, Eigenvalue convergence")
#plt.savefig("Shifted QR Cumulative Eigenvalue Error")
print("Actual eigenvalues:")
print(eig_A_ordered)
print("Eigenvalues from shifted QR:")
print(np.sort(eigs_shifted_qr))
print("Relative Error:")
print(np.linalg.norm(eig_A_ordered - np.sort(eigs_shifted_qr))/np.linalg.norm(eig_A_ordered))