# Kenneth Meyer
# 2/7/23
# plots and numerical results for quantum hw 2

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve

### QUESTION 1 ###
secular_eqn = lambda x: np.sqrt(100 - x**2)*np.tanh(0.2*np.sqrt(100 - x**2)) - x*(np.tan(0.2*x)*np.tan(x)+1)/(np.tan(0.2*x) - np.tan(x))

roots = fsolve(secular_eqn,0.1)
print(roots)

# plotting wavefunction and probability density of lowest-energy even-parity state
def plot_wavefunction(x):
    '''
        Plots wavefunction for a given energy state x
    '''

def plot_pdf(x):
    '''
        Plots pdf of wavefunction
    
    '''

# ^ think I'm doing this tomorrow, want to check plots as I do them