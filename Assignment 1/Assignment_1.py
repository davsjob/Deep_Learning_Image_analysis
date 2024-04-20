import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from load_auto import load_auto


def initialize_parameters(x):
    w = np.zeros((1,x.shape[1]))
    b = 0
    return w, b

def compute_cost(y, z):
    
    J = np.mean((y - z)**2)
    return J

def update_parameters(w, b, dJ_db, dJ_dw, gamma):
    w = w - gamma * dJ_dw
    b = b - gamma * dJ_db
    return w, b

def model_forward(x, w, b):
    z_i = np.dot(x,w.T) + b
    
    return z_i

def model_backward(z, y, x):
    n = len(y)
    dJ_dz = (2/n)*(z - y)
    dz_dw = x.T
    dJ_dw = np.dot(dz_dw, dJ_dz)
    dJ_db = np.sum(dJ_dz)
    return dJ_db, dJ_dw


def predict(X, w, b):
    return model_forward(X, w, b)


def train_linear_model(X, Y, gamma, num_iterations):
    w, b = initialize_parameters(X)
    cost = np.zeros(num_iterations)
    for i in range(num_iterations):
        z = model_forward(X, w, b)
        cost[i] = compute_cost(Y, z)
        dJ_db, dJ_dw = model_backward(z, Y, X)
        w, b = update_parameters(w, b, dJ_db, dJ_dw, gamma)
    return w, b, cost

def load_hp():

	# import data
	Auto = pd.read_csv('Auto.csv', na_values='?', dtype={'ID': str}).dropna().reset_index()

	# Extract relevant data features
	X_train = Auto[['horsepower']].values
	Y_train = Auto[['mpg']].values

	return X_train, Y_train

def main():
    # Load data
    X_all, Y = load_auto()
    X_all = (X_all - X_all.mean()) / X_all.std()
    
    X_hp, Y = load_hp()
    X_hp = (X_hp - X_hp.mean()) / X_hp.std()
    
    
    # Train model
    gamma = [1, 0.1, 0.01, 0.001, 0.0001]
    predictions_all = []
    predictions_hp = []
    costs_all = []
    costs_hp = []
    num_iterations = 10**6
    for i in range(len(gamma)):
        w, b, cost = train_linear_model(X_all, Y, gamma[i], num_iterations)
        predictions_all.append(predict(X_all, w, b))
        print("Cost for all features with gamma", gamma[i], ":", cost[-1])
        #print("Equation w gamma = ", gamma[i], ":", w, "b gamma = ", gamma[i], ":", b)
        costs_all.append(cost)

    for i in range(len(gamma)):
        w, b, cost = train_linear_model(X_hp, Y, gamma[i], num_iterations)
        predictions_hp.append(predict(X_hp, w, b))
        print("Cost for only hp with gamma", gamma[i], ":", cost[-1])
        #print("Equation w gamma = ", gamma[i], ":", w, "b gamma = ", gamma[i], ":", b)

        costs_hp.append(cost)
   

    # Plot costs for all features and horsepower only
    plt.figure()
    plt.subplot(1, 2, 1)
    for i in range(len(gamma)):
        plt.plot(range(num_iterations), costs_all[i])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend(['Gamma = 1' ,'Gamma = 0.1', 'Gamma = 0.01', 'Gamma = 0.001', 'Gamma = 0.0001'])


    plt.subplot(1, 2, 2)
    for i in range(len(gamma)):
        plt.plot(range(num_iterations), costs_hp[i])
    plt.xlabel('Iteration')
    plt.ylabel('Cost')
    plt.legend(['Gamma = 1', 'Gamma = 0.1', 'Gamma = 0.01', 'Gamma = 0.001', 'Gamma = 0.0001'])
    plt.show()

    

    
    
if __name__ == "__main__":
    main()