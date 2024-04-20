import numpy as np 
import pandas as pd

def compute_cost(y, z):
    n = len(y)
    J = 1/n * np.sum((y - z)**2)
    return J
