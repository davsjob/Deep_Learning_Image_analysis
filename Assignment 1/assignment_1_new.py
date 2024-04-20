import numpy as np
import pandas as pd
from load_auto import load_auto

def load_hp():
    # import data
    Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

    # Extract relevant data features
    X_train = Auto[['horsepower']].values
    Y_train = Auto[['mpg']].values

    return X_train, Y_train

def initialize_parameters(input_shape):
    """
    Initializes parameters w and b.
    
    Arguments:
    input_shape -- tuple, shape of the input X (m, n)
    
    Returns:
    w -- initialized weights, shape: (n, 1)
    b -- initialized bias, scalar
    """
    n = input_shape[1]
    w = np.zeros((n, 1))
    b = 0
    return w, b

def model_forward(X, w, b):
    """
    Forward propagation to evaluate the linear regression model.
    
    Arguments:
    X -- input data, shape: (m, n)
    w -- weights, shape: (n, 1)
    b -- bias, scalar
    
    Returns:
    predictions -- predictions of the model, shape: (m, 1)
    """
    predictions = np.dot(X, w) + b
    return predictions

def compute_cost(predictions, y):
    """
    Computes the cost (mean squared error) between predictions and true values.
    
    Arguments:
    predictions -- predictions of the model, shape: (m, 1)
    y -- true values, shape: (m, 1)
    
    Returns:
    cost -- mean squared error
    """
    #m = y.shape[0]
    cost = np.mean((predictions-y)**2)#np.sum((predictions - y) ** 2) / (m)
    return cost

def model_backward(X, predictions, y):
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
    dw = np.dot(X.T, (predictions - y))*(2/ m)
    db = np.sum(predictions - y)*(2 / m)
    return dw, db

def update_parameters(w, b, dw, db, learning_rate):
    """
    Updates the parameters based on gradients and learning rate.
    
    Arguments:
    w -- weights, shape: (n, 1)
    b -- bias, scalar
    dw -- gradient of the cost with respect to w, shape: (n, 1)
    db -- gradient of the cost with respect to b, scalar
    learning_rate -- learning rate
    
    Returns:
    w -- updated weights
    b -- updated bias
    """
    w -= learning_rate * dw
    b -= learning_rate * db
    return w, b

def predict(X, w, b):
    """
    Predicts values using the trained model.
    
    Arguments:
    X -- input data, shape: (m, n)
    w -- weights, shape: (n, 1)
    b -- bias, scalar
    
    Returns:
    predictions -- predictions of the model, shape: (m, 1)
    """
    predictions = model_forward(X, w, b)
    return predictions

def train_linear_model(X, y, learning_rate, num_iterations):
    """
    Trains a linear regression model using gradient descent.
    
    Arguments:
    X -- input data, shape: (m, n)
    y -- true values, shape: (m, 1)
    learning_rate -- learning rate
    num_iterations -- number of iterations for training
    
    Returns:
    w -- trained weights
    b -- trained bias
    costs -- list of costs over training
    """
    w, b = initialize_parameters(X.shape)
    costs = []
    
    for i in range(num_iterations):
        predictions = model_forward(X, w, b)
        cost = compute_cost(predictions, y)
        dw, db = model_backward(X, predictions, y)
        w, b = update_parameters(w, b, dw, db, learning_rate)
        costs.append(cost)
    
    return w, b, costs

def main():
    # Load data
    X_all, Y = load_auto()
    X_all = (X_all - np.mean(X_all)) / np.std(X_all)
    
    X_hp, Y = load_hp()
    X_hp = (X_hp - np.mean(X_hp)) / np.std(X_hp)
    
    
    # Train model
    gamma = [0.1, 0.01, 0.001]
    predictions_all = []
    predictions_hp = []
    costs_all = []
    costs_hp = []
    num_iterations = 10**6
    for i in range(len(gamma)):
        w, b, cost = train_linear_model(X_all, Y, gamma[i], num_iterations)
        predictions_all.append(predict(X_all, w, b))
        print(f"Cost for gamma with all features= {gamma[i]}: ", cost[-1])
        costs_all.append(cost)
        
        w, b, cost = train_linear_model(X_hp, Y, gamma[i], num_iterations)
        predictions_hp.append(predict(X_hp, w, b))
        print(f"Cost for gamma with only hp= {gamma[i]}: ", cost[-1])
        costs_hp.append(cost)
    return predictions_all, predictions_hp, costs_all, costs_hp

def plot_costs(cost, gammas):
    """
    Plots the costs over training for different learning rates.
    
    Arguments:
    cost -- list of costs over training for all features
    """
    import matplotlib.pyplot as plt
    
    for i in range(len(gammas)):
        plt.plot(range(len(cost[i])), cost[i], label=(f"$\\alpha$ = 1e{gammas[i]}"))
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost over Training for Different Learning Rates')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    predictions_all, predictions_hp, costs_all, costs_hp = main()

    plot_costs(costs_all, [0.1, 0.01, 0.001])
    plot_costs(costs_hp, [0.1, 0.01, 0.001])
    