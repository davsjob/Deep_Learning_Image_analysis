import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def initialize_parameters(X_shape):
    """
    Initializes weights and biases
    In this implementation they are initialized as zeros.

    Args:
    X_shape -- tuple, shape of the input X (m, n)
    
    Returns:
    w, b -- tuple(np.ndarray, np.ndarray)
    """
    n = X_shape[1]
    m = 1
    
    w = np.zeros((n, m))
    b = np.zeros(m)
    return w, b

def model_forward(X, w, b):
    """
    Forward propagation of the linear regression model.
    
    Args:
    X -- input data, (np.ndarray, shape: (m, 1) ) 
    w -- weights, (np.ndarray, shape: (n, 1) ) 
    b -- bias, (np.ndarray, shape: scalar)
    
    Returns:
    Z -- predicted values of the model, shape: (m, 1)
    """
    Z = np.dot(X, w) + b
    return Z

def compute_cost(Z, Y):
    """
    Computes the cost J in through mean squared error.
    
    Args:
    Z -- predictions, (np.ndarray, shape: (m, 1) ) 
    Y -- true values, (np.ndarray, shape: (m, 1) ) 
    
    Returns:
    J --  mean squared error, (float)
    """
    
    J = np.mean((Z-Y)**2)
    return J

def model_backward(X, Z, y):
    """
    Computes the gradients of the cost function with respect to parameters w and b.
    
    Arguments:
    X -- input data, shape: (m, n)
    predictions -- predictions of the model, shape: (m, 1)
    y -- true values, shape: (m, 1)
    
    Returns:
    dw -- gradient of the cost with respect to w, shape: (n, 1)
    db -- gradient of the cost with respect to b, scalar
    """
    m = X.shape[0]
    dw = (-2/ m)*np.dot(X.T, (y - Z))
    db = (-2/ m)*np.sum(y - Z)
    return dw, db

def update_parameters(w, b, dw, db, gamma):
    """
    Updates the parameters w and b using gradient descent.
    
    Arguments:
    w -- weights, shape: (n, 1)
    b -- bias, scalar
    dw -- gradient of the cost with respect to w, shape: (n, 1)
    db -- gradient of the cost with respect to b, scalar
    gamma -- learning rate
    
    Returns:
    w, b -- updated weights and bias
    """
    w -= gamma * dw
    b -= gamma * db

    return w, b

def predict(X, w, b):
    """
    Predicts values using the trained model.
    
    Arguments:
    X -- input data, shape: (m, n)
    w -- weights, shape: (n, 1)
    b -- bias, scalar
    
    Returns:
    Z -- predictions of the model, shape: (m, 1)
    """
    Z = model_forward(X, w, b)
    return Z

def train_linear_model(X, y, learning_rate, num_iterations):
    """
    Trains a linear regression model using gradient descent.
    
    Arguments:
    X -- input data, shape: (m, n)
    y -- true values, shape: (m, 1)
    learning_rate -- learning rate for gradient descent
    num_iterations -- number of iterations for training
    
    Returns:
    w -- trained weights
    b -- trained bias
    costs -- list of costs over training
    """
    # Initialize parameters
    w, b = initialize_parameters(X.shape)

    # Initialize vector for costs
    costs = np.empty(num_iterations)

    # Train the model with gradient descent
    for i in range(num_iterations):
        # Forward propagation
        Z = predict(X, w, b)

        # Compute cost
        costs[i] = compute_cost(Z, y)

        # Backward propagation
        dw, db = model_backward(X, Z, y)

        # Update parameters
        w, b = update_parameters(w, b, dw, db, learning_rate)
        

    return w, b, costs

