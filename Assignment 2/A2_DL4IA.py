import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist
from time import time


class FeedForwardBackPropagationNetwork():
    def __init__(self, model: dict) -> None:
        """
        Intialise the FeedForwardBackPropagationNetwork with the model architecture
        :param model: A dictionary containing the model architecture    
        """
        self.model = model
        self.states = {} # Store the states of the network as key-value pairs
        self.n_images = 0
        self.costs_training = []
        self.costs_validation = []
        self.accuracies_training = []
        self.accuracies_validation = []
        self.num_classes = 10
        np.random.seed(0)

    def intialise_parameters(self, x_train: np.ndarray, y_train: np.ndarray,
                              mu: float, sigma: float) -> None:
        """
        Intialise the weights and biases for the network
        :param x_train: Training data
        :param y_train: Training labels
        :param mu: Mean for the normal distribution
        :param sigma: Standard deviation for the normal distribution
        """
        
        # Number of images, input dimension and number of classes
        self.n_images = x_train.shape[0]
        input_dim = x_train.shape[1]
        classes = y_train.shape[1]
        self.num_classes = classes
        # Initialise weights and biases for all layers
        for layer in self.model:
            self.states[layer] = {}
            layer_dim = self.model[layer]["nodes"]
            self.states[layer]["W"] = np.random.normal(mu, sigma, size = (layer_dim, input_dim))     
            self.states[layer]["b"] = np.zeros((layer_dim,1))
            self.states[layer]["activation"] = self.model[layer]["activation"]
            input_dim = layer_dim

        print("Architecture: \n")
        for layer in self.model:
            print(f"Layer: {layer}, Nodes: {self.model[layer]['nodes']}, Activation: {self.model[layer]['activation']}")
            print(f"Weight shape: {self.states[layer]['W'].shape},Bias shape: {self.states[layer]['b'].shape}")
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        """
        Sigmoid activation function
        :param x: Input to the activation function
        :return: Output of the activation function
        """

        return 1/(1 + np.exp(-x))
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """
        Relu activation function
        :param x: Input to the activation function
        :return: Output of the activation function
        """
        return np.maximum(0,x)
    
    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Softmax activation function
        :param x: Input to the activation function
        :return: Output of the activation function
        """
        exps = np.exp(x - np.max(x))
        return exps / np.sum(exps, axis=0)  
        
    def linear_forward(self, X: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Linear forward pass
        :param X: Input to the layer
        :param W: Weights of the layer
        :param b: Biases of the layer
        :return: Output of the linear forward pass
        """
        return np.dot(W, X) + b
    
    def compute_cost(self, y: np.ndarray) -> float:
        """
        Compute the cost function
        :param y: True labels
        :return: Cost of the network
        """
        def loss (z: np.ndarray, y: np.ndarray) -> float:
            """
            Compute the cross-entropy loss
            :param z: Prediction of the network
            :param y: True labels
            :return: Loss of the network
            """
            z_temp = z - np.max(z) # For numerical stability
            return np.log(np.sum(np.exp(z_temp), axis=1)) - np.sum(np.multiply(z_temp, y), axis=1)
        prediction = self.states["output"]["a"].T
        return np.mean(loss(prediction, y))
    
    
    def activation_forward(self, z: np.ndarray, activation: str) -> np.ndarray:
        """
        Activation forward pass
        :param z: Previous prediction as input to the layer
        :param activation: Activation function to be used
        :return: Output of the activation forward pass
        """
        if activation == "sigmoid":
            return self.sigmoid(z)
        elif activation == "relu":
            return self.relu(z)
        elif activation == "softmax":
            return self.softmax(z)
        else:
            raise ValueError("Invalid activation function")
    
    def model_forward(self, X: np.ndarray) -> None:
        """
        Forward pass through the network
        :param X: Input to the network
        :return: Output of the network layers
        """
        next_input = X.T # Transpose the input
        for layer in self.states.keys():
            self.states[layer]["x"] = next_input
            W = self.states[layer]["W"]
            b = self.states[layer]["b"]
            z = self.linear_forward(next_input, W, b)
            activation = self.states[layer]["activation"]
            a = self.activation_forward(z, activation)
            self.states[layer]["z"] = z
            self.states[layer]["a"] = a
            next_input = a
            if layer == "output":
                self.states["output"]["a"] = a
            
        return self.states["output"]["a"]
    
    def sigmoid_backward(self, da: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Sigmoid backward pass
        :param da: Gradient of the activation function
        :param z: Input to the activation function
        :return: Gradient of the sigmoid activation function
        """
        s = self.sigmoid(z)
        return da * s * (1 - s)

    def relu_backward(self, da: np.ndarray, z: np.ndarray) -> np.ndarray:
        """
        Relu backward pass
        :param da: Gradient of the activation function
        :param z: Input to the activation function
        :return: Gradient of the relu activation function
        """
        dz = np.array(da, copy=True)
        dz[z <= 0] = 0
        return dz
    
    def linear_backward(self, da: np.ndarray, z: np.ndarray, ) -> np.ndarray:
        """
        Linear backward pass
        :param da: Gradient of the activation function
        :param z: Input to the layer
        :return: Gradients of the linear backwards pass
        """
        dw_back = np.dot(da, z.T)
        db_back = np.sum(da, axis=1, keepdims=True)
        
        return dw_back, db_back
    
    def activation_backward(self, da: np.ndarray, z: np.ndarray, activation: str) -> np.ndarray:
        """
        Activation backward pass
        :param da: Gradient of the activation function
        :param z: Input to the activation function
        :param activation: Activation function to be used
        :return: Gradient of the activation function
        """
        if activation == "sigmoid" or activation == "softmax":
            return self.sigmoid_backward(da, z)
        elif activation == "relu":
            return self.relu_backward(da, z)
        else:
            raise ValueError("Invalid activation function")

    def model_backward(self, y: np.ndarray) -> None:
        """
        Backward pass through the network
        :param y: True labels
        """
        da = self.states["output"]["a"] - y.T # Derivative of the loss function
        # Iterate backwards through the layers
        # Compute the gradients for each layer
        for layer in reversed(list(self.states.keys())):
            activation = self.states[layer]["activation"]
            output = self.states[layer]["z"]
            input = self.states[layer]["x"]
            da = self.activation_backward(da, output, activation)
            dw, db = self.linear_backward(da, input)
            self.states[layer]["dW"] = dw
            self.states[layer]["db"] = db
            da = np.dot(self.states[layer]["W"].T, da)

    def update_parameters(self, learning_rate: float) -> None:
        """
        Update the weights and biases of the network
        :param learning_rate: Learning rate of the network
        """
        for layer in self.states.keys():
            self.states[layer]["W"] -= learning_rate*self.states[layer]["dW"]
            self.states[layer]["b"] -= learning_rate*self.states[layer]["db"]
    
    def predict(self, X: np.ndarray, encoded_labels: np.ndarray) -> list[np.ndarray, np.ndarray]:
        """
        Predict the labels for the input data
        :param X: Input data
        :return: Predicted labels
        """
        self.model_forward(X)
        prediction = np.argmax(self.states["output"]["a"], axis=0)
        classification = np.argmax(encoded_labels, axis=1)
        return [classification, np.mean(prediction == classification)]
    
    
    
    def random_mini_batches(self, x: np.ndarray, y:np.ndarray, batch_size: int) -> list:
        """
        Generate random mini-batches
        :param x: Input data
        :param y: True labels
        :param batch_size: Size of the mini-batches
        :return: Mini-batches of the input data and labels
        """
        m = x.shape[0] # Number of training examples
        permutation = list(np.random.permutation(m)) 
        shuffled_x = x[permutation, :]
        shuffled_y = y[permutation, :]
        num_batches = m // batch_size
        mini_batches = []
        for i in range(num_batches):
            mini_batch_x = shuffled_x[i*batch_size:(i+1)*batch_size, :]
            mini_batch_y = shuffled_y[i*batch_size:(i+1)*batch_size, :]
            mini_batches.append((mini_batch_x, mini_batch_y))
        if m % batch_size != 0:
            mini_batch_x = shuffled_x[num_batches*batch_size:, :]
            mini_batch_y = shuffled_y[num_batches*batch_size:, :]
            mini_batches.append((mini_batch_x, mini_batch_y))
        return mini_batches
    
    def train_model(self, x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, y_test: np.ndarray,
                    mu: float, sigma: float,  learning_rate: float, iterations: int, batch_size: int):
        """
        Train the network
        :param x_train: Training data
        :param y_train: Training labels
        :param x_test: Test data
        :param y_test: Test labels
        :param learning_rate: Learning rate of the network
        :param iterations: Number of epochs
        :param batch_size: Size of the mini-batches
        """
        print("Beginning training...")
        self.intialise_parameters(x_train, y_train, mu, sigma)
        for i in range(iterations):
            
            mini_batches = self.random_mini_batches(x_train, y_train, batch_size)
            epoch_cost = 0
            epoch_accuracy = 0

            for mini_batch in mini_batches:
                x_mini, y_mini = mini_batch
                self.model_forward(x_mini)
                cost = self.compute_cost(y_mini)
                epoch_cost += cost
                epoch_accuracy += self.predict(x_mini, y_mini)[1]
                self.model_backward(y_mini)
                self.update_parameters(learning_rate)
            self.costs_training.append(epoch_cost/len(mini_batches))
            self.accuracies_training.append(100*(epoch_accuracy/len(mini_batches)))
            self.accuracies_validation.append(100*self.predict(x_test, y_test)[1])
            self.model_forward(x_test)
            self.costs_validation.append(self.compute_cost(y_test))
            if i % 10 == 0:
                print(f"Iteration: {i}")
                print(f"Training cost: {self.costs_training[-1]}, Training accuracy: {self.accuracies_training[-1]}")
                print(f"Validation cost: {self.costs_validation[-1]}, Validation accuracy: {self.accuracies_validation[-1]}")

            
    def training_curve_plot(self, title, train_losses, test_losses, train_accuracy, test_accuracy):
        """ 
        convenience function for plotting train and test loss and accuracy
        """
        lg=13
        md=10
        sm=9
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        fig.suptitle(title, fontsize=lg)
        x = range(1, len(train_losses)+1)
        axs[0].plot(x, train_losses, label=f'Final train loss: {train_losses[-1]:.4f}')
        axs[0].plot(x, test_losses, label=f'Final test loss: {test_losses[-1]:.4f}')
        axs[0].set_title('Losses', fontsize=md)
        axs[0].set_xlabel('Iteration', fontsize=md)
        axs[0].set_ylabel('Loss', fontsize=md)
        axs[0].legend(fontsize=sm)
        axs[0].tick_params(axis='both', labelsize=sm)
        # Optionally use a logarithmic y-scale
        #axs[0].set_yscale('log')
        axs[0].grid(True, which="both", linestyle='--', linewidth=0.5)
        axs[1].plot(x, train_accuracy, label=f'Final train accuracy: {train_accuracy[-1]:.4f}%')
        axs[1].plot(x, test_accuracy, label=f'Final test accuracy: {test_accuracy[-1]:.4f}%')
        axs[1].set_title('Accuracy', fontsize=md)
        axs[1].set_xlabel('Iteration', fontsize=md)
        axs[1].set_ylabel('Accuracy (%)', fontsize=sm)
        axs[1].legend(fontsize=sm)
        axs[1].tick_params(axis='both', labelsize=sm)
        axs[1].grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.show()

    def plot_weights(self) -> None:
        """
        Plot the weights of the output layer as images
        """
        first_hidden_layer = list(self.states.keys())[0]
        weight_matrix = self.states[first_hidden_layer]["W"]
        figsize = [9.5, 5]
        nrows, ncols = 2, 5
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for counter, axi in enumerate(ax.flat):
            img = weight_matrix[counter, :].reshape((28,28))
            axi.imshow(img, cmap='gray')
            axi.set_title("Row: " +str(counter))
        plt.show()

if __name__ == "__main__":
    print("Loading dataset...")
    x_train, y_train, x_test, y_test = load_mnist()
    x_holdout = x_test[50000:]
    y_holdout = y_test[50000:]
    x_test = x_test[:50000]
    y_test = y_test[:50000]
    print("Loading complete")
    print("Data shape: \n")
    print(f"Training data: {x_train.shape}")
    print(f"Training labels: {y_train.shape}")
    print(f"Test data: {x_test.shape}")
    print(f"Test labels: {y_test.shape}")
    model = {
        "layer1": {"nodes": 256, "activation": "relu"},
        "output": {"nodes": 10, "activation": "softmax"}
    }
    network = FeedForwardBackPropagationNetwork(model)
    mu = 0
    sigma = 0.01
    learning_rate = 0.01
    iterations = 100
    batch_size = 64
    start = time()
    network.train_model(x_train, y_train, x_test, y_test, mu, sigma, learning_rate, iterations, batch_size)
    end = time() - start
    print(f"Training complete. Time taken: {end} seconds")
    network.training_curve_plot("Training and Validation Curves", network.costs_training, network.costs_validation, network.accuracies_training, network.accuracies_validation)
    network.plot_weights()
    
    