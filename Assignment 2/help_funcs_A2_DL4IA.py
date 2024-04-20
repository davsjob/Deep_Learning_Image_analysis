import numpy as np
import matplotlib.pyplot as plt

def initialize_parameters(layer_dims: list[tuple[int, int]]) -> dict[str, np.ndarray]:
    """
    Initializes the weights and biases for each layer in the model.

    Args:
    layer_dims -- list[tuple[int, int]], dimensions of each layer

    Returns:
    parameters -- dict[str, np.ndarray], initialized weights and biases
    """
    parameters = {}
    

    for i in range(1, len(layer_dims)):
        layer_input_dim = layer_dims[i-1][1]
        layer_output_dim = layer_dims[i][1]

        parameters[f'w{i}'] = np.random.randn(layer_input_dim, layer_output_dim) * 0.01
        parameters[f'b{i}'] = np.zeros((1, layer_output_dim))

    return parameters


def sigmoid(x):
    """
    Compute the sigmoid function for the given input.

    Parameters:
    x (float or numpy.ndarray): The input value or array.

    Returns:
    float or numpy.ndarray: The sigmoid value(s) of the input.

    """
    h = 1 / (1 + np.e**(-x))
    return h

def relu(x):
    """
    Compute the ReLU function for the given input.

    Parameters:
    x (float or numpy.ndarray): The input value or array.

    Returns:
    float or numpy.ndarray: The ReLU value(s) of the input.

    """
    return np.maximum(0, x)

def softmax(z):
    """
    Compute the softmax function for the given input.

    Parameters:
    z (numpy.ndarray): The input array.

    Returns:
    numpy.ndarray: The softmax values of the input.

    """
    return np.e**z / np.sum(np.e**z)




def compute_cost(z, y, num_classes):
    def one_hot_encode(y, num_classes):
        one_hot = np.zeros((len(y), num_classes))
        one_hot[np.arange(len(y)), y] = 1
        return one_hot

    def loss(z, y):
        z = z - np.max(z) # for numerical stability
        Li = np.log2(np.sum(np.e**(z))) - np.sum(z*y)
        return Li
    y = one_hot_encode(y, num_classes)
    l = loss(z, y)
    return np.mean(l)

def linear_forward(X, w, b):
    """
    Computes the linear part of the forward propagation.

    Args:
    X -- np.ndarray, shape (m, n), input data
    w -- np.ndarray, shape (n, m), weights
    b -- np.ndarray, shape (1, m), biases

    Returns:
    z -- np.ndarray, shape (m, m), the input of the activation function
    """
    z = np.dot(X, w) + b
    return z

def activation_forward(z, activation):
    """
    Computes the forward propagation for the given activation function.

    Args:
    z -- np.ndarray, shape (m, m), the input of the activation function
    activation -- str, the activation function to use

    Returns:
    a -- np.ndarray, shape (m, m), the output of the activation function
    """
    if activation == "sigmoid":
        a = sigmoid(z)
    elif activation == "relu":
        a = relu(z)
    else:
        raise ValueError("Activation function not supported.")
    return a

def model_forward(X: np.ndarray, parameters: dict[str, np.ndarray], activation: list[str]
                  ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Computes the forward propagation for the model.

    Args:
    X -- np.ndarray, shape (m, n), input data
    parameters -- dict, containing the weights and biases
    activation -- list[str], the activation functions to use

    Returns:
    a -- np.ndarray, shape (m, m), the output of the model
    caches -- list[tuple[np.ndarray, np.ndarray]], containing the caches for each layer
    """
    cache = []
    a = X
    for i in range(len(parameters) // 2):
        w = parameters["w" + str(i)]
        b = parameters["b" + str(i)]
        z = linear_forward(a, w, b)
        a = activation_forward(z, activation=activation[i])
        cache.append((a, z))
    return a, cache


def linear_backward(dz, cache):
        """
        Computes the backward propagation for the linear layer.

        Args:
        dz -- np.ndarray, shape (m, m), gradient of the cost with respect to the linear output
        cache -- tuple, containing the inputs (X, w, b) from the forward propagation

        Returns:
        dX -- np.ndarray, shape (m, n), gradient of the cost with respect to the input X
        dw -- np.ndarray, shape (n, m), gradient of the cost with respect to the weights w
        db -- np.ndarray, shape (1, m), gradient of the cost with respect to the biases b
        """
        X, w, b = cache
        m = X.shape[0]

        dX = np.dot(dz, w.T)
        dw = np.dot(X.T, dz)
        db = np.sum(dz, axis=0, keepdims=True)

        return dX, dw, db


def sigmoid_backward(da, z):
    """
    Computes the backward propagation for the sigmoid activation function.

    Args:
    da -- np.ndarray, shape (m, m), gradient of the cost with respect to the activation output
    z -- np.ndarray, shape (m, m), the input of the activation function

    Returns:
    dz -- np.ndarray, shape (m, m), gradient of the cost with respect to the linear output
    """
    h = sigmoid(z)
    dz = da * h * (1 - h)
    return dz

def relu_backward(da, z):
    """
    Computes the backward propagation for the ReLU activation function.

    Args:
    da -- np.ndarray, shape (m, m), gradient of the cost with respect to the activation output
    z -- np.ndarray, shape (m, m), the input of the activation function

    Returns:
    dz -- np.ndarray, shape (m, m), gradient of the cost with respect to the linear output
    """
    dz = np.where(z > 0, da, 0)
    return dz

def activation_backward(da, z, activation):
    """
    Computes the backward propagation for the given activation function.

    Args:
    da -- np.ndarray, shape (m, m), gradient of the cost with respect to the activation output
    z -- np.ndarray, shape (m, m), the input of the activation function
    activation -- str, the activation function used

    Returns:
    dz -- np.ndarray, shape (m, m), gradient of the cost with respect to the linear output
    """
    if activation == "sigmoid":
        dz = sigmoid_backward(da, z)
    elif activation == "relu":
        dz = relu_backward(da, z)
    else:
        raise ValueError("Activation function not supported.")
    return dz

def model_backward(a, y, w, b, caches, activation):
    """
    Computes the backward propagation for the model.

    Args:
    a -- np.ndarray, shape (m, m), output of the model
    y -- np.ndarray, shape (m, m), true labels
    caches -- list, containing the caches for each layer
    activation -- str, the activation function used

    Returns:
    grads -- dict, containing the gradients for each parameter
    """
    grads = {}
    m = a.shape[0]
    y = one_hot_encode(y, a.shape[1])
    da = - (np.divide(y, a) - np.divide(1 - y, 1 - a))

    for i in reversed(range(len(caches))):
        a, z = caches[i]
        if i == len(caches) - 1:
            dz = da
        else:
            dz = activation_backward(da, z, activation=activation[i])
        dX, dw, db = linear_backward(dz, cache=(a, w, b))
        grads["dX" + str(i)] = dX
        grads["dw" + str(i)] = dw
        grads["db" + str(i)] = db
        da = dX
    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Updates the parameters using the gradient descent algorithm.

    Args:
    parameters -- dict, containing the weights and biases
    grads -- dict, containing the gradients for each parameter
    learning_rate -- float, the learning rate

    Returns:
    parameters -- dict, containing the updated weights and biases
    """
    for i in range(len(parameters) // 2):
        parameters["w" + str(i)] -= learning_rate * grads["dw" + str(i)]
        parameters["b" + str(i)] -= learning_rate * grads["db" + str(i)]
    return parameters

def predict(X, y, parameters, activation):
    """
    Predicts the softmax outputs and computes the accuracy.

    Args:
    X -- np.ndarray, shape (m, n), input data
    parameters -- dict, containing the weights and biases
    activation -- str, the activation function used

    Returns:
    predictions -- np.ndarray, shape (m,), predicted class labels
    accuracy -- float, the accuracy of the predictions
    """
    m = X.shape[0]
    predictions = np.zeros(m)
    a, _ = model_forward(X, parameters, activation)
    softmax_outputs = softmax(a)
    predictions = np.argmax(softmax_outputs, axis=1)
    accuracy = np.mean(predictions == y) * 100
    return predictions, accuracy

def random_mini_batches(X, y, mini_batch_size):
    """
    Randomly partitions the training data into mini-batches.

    Args:
    X -- np.ndarray, shape (m, n), input data
    y -- np.ndarray, shape (m,), true labels
    mini_batch_size -- int, size of each mini-batch

    Returns:
    mini_batches -- list, containing mini-batches of input data and labels
    """
    m = X.shape[0]
    mini_batches = []
    
    # Shuffle the data
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]
    
    # Partition the data into mini-batches
    num_complete_minibatches = m // mini_batch_size
    for k in range(num_complete_minibatches):
        mini_batch_X = shuffled_X[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch_y = shuffled_y[k * mini_batch_size : (k + 1) * mini_batch_size]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    
    # Handle the case when the last mini-batch has fewer examples
    if m % mini_batch_size != 0:
        mini_batch_X = shuffled_X[num_complete_minibatches * mini_batch_size :]
        mini_batch_y = shuffled_y[num_complete_minibatches * mini_batch_size :]
        mini_batch = (mini_batch_X, mini_batch_y)
        mini_batches.append(mini_batch)
    
    return mini_batches

def train_model(x_train, y_train, model_architecture, iterations, learning_rate, batch_size, x_test=None, y_test=None):
    """
    Trains a model using the given data and hyperparameters.

    Args:
    x_train -- np.ndarray, shape (m, n), training data
    y_train -- np.ndarray, shape (m,), training labels
    model_architecture -- list[int], defining the model architecture
    iterations -- int, number of iterations
    learning_rate -- float, learning rate α
    batch_size -- int, number of training examples to use for each step
    x_test -- np.ndarray, shape (m, n), test data (optional)
    y_test -- np.ndarray, shape (m,), test labels (optional)

    Returns:
    train_costs -- list[float], training costs at each iteration
    train_accuracies -- list[float], training accuracies at each iteration
    test_costs -- list[float], test costs at each iteration (empty if x_test and y_test are not provided)
    test_accuracies -- list[float], test accuracies at each iteration (empty if x_test and y_test are not provided)
    """
    train_costs = []
    train_accuracies = []
    test_costs = []
    test_accuracies = []

    # Initialize parameters
    parameters = initialize_parameters((x_train.shape[0], model_architecture[-1]))

    # Iterate over the specified number of iterations
    for i in range(iterations):
        # Create mini-batches
        mini_batches = random_mini_batches(x_train, y_train, batch_size)

        # Iterate over each mini-batch
        for mini_batch in mini_batches:
            mini_batch_X, mini_batch_y = mini_batch

            # Forward propagation
            a, caches = model_forward(mini_batch_X, parameters, activation=["relu"] * (len(model_architecture) - 1))

            # Compute cost
            cost = compute_cost(loss(a, mini_batch_y))
            train_costs.append(cost)

            # Backward propagation
            grads = model_backward(a, mini_batch_y, parameters, caches, activation=["relu"] * (len(model_architecture) - 1))

            # Update parameters
            parameters = update_parameters(parameters, grads, learning_rate)

        # Print or plot the results live during training
        if (i + 1) % k == 0:
            train_predictions, train_accuracy = predict(x_train, y_train, parameters, activation=["relu"] * (len(model_architecture) - 1))
            train_accuracies.append(train_accuracy)

            if x_test is not None and y_test is not None:
                test_predictions, test_accuracy = predict(x_test, y_test, parameters, activation=["relu"] * (len(model_architecture) - 1))
                test_accuracies.append(test_accuracy)
                test_costs.append(compute_cost(loss(model_forward(x_test, parameters, activation=["relu"] * (len(model_architecture) - 1))[0], y_test)))

