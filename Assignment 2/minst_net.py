import numpy as np
from load_mnist import load_mnist
import matplotlib.pyplot as plt
import math

class feedforward_backprop_neuralNetwork():

    def __init__(self, model):
        self.model = model
        self.model_state = {} # Dictionary where keys are the layers and values are dictionaries containing related variables
        self.number_images = 0
        self.costs_train = []
        self.costs_test = []
        self.accuracies_train = []
        self.accuracies_test = []

    def initialise_parameters(self, x_train, y_train):
        mu, sigma = 0, 0.01

        self.number_images = x_train.shape[0]
        input_dimension = x_train.shape[1]
        number_classes = y_train.shape[1]

        for layer in self.model.keys(): # Iterate through hidden layers and initialise their parameters
            self.model_state[layer] = {}
            output_dimension = model[layer]['nodes']
            weight_matrix = np.random.normal(mu, sigma, size=(output_dimension, input_dimension))
            offset_b = np.zeros((output_dimension, 1))
            self.model_state[layer]['weights'] = weight_matrix
            self.model_state[layer]['offset_b'] = offset_b
            input_dimension = output_dimension
        
        # Define output layer and initialise its parameters
        self.model_state['output_layer'] = {} 
        weight_matrix = np.random.normal(mu, sigma, size=(number_classes, input_dimension))
        offset_b = np.zeros((number_classes, 1))
        self.model_state['output_layer']['weights'] = weight_matrix
        self.model_state['output_layer']['offset_b'] = offset_b

        print('\nNetwork Architecture:\n')
        for layer in self.model_state.keys():
            print('Layer: ' + str(layer))
            print('Weights shape: ' + str(self.model_state[layer]['weights'].shape))
            print('Offset shape: ' + str(self.model_state[layer]['offset_b'].shape) + '\n')
    
    def linear_forward(self, x, weights, offset_b):
        return weights.dot(x) + offset_b

    def sigmoid(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def relu(self, Z):
        return np.maximum(0, Z)
    
    def softmax(self, Z):
        tmp = np.exp(Z - np.max(Z))
        return  tmp / tmp.sum(axis=0, keepdims=True)
    
    def activation_forward(self, Z, activation_model):
        if activation_model == 'sigmoid':
            return self.sigmoid(Z)
        elif activation_model == 'relu':
            return self.relu(Z)
        elif activation_model == 'softmax':
            return self.softmax(Z)
        else:
            raise Exception('Non-supported activation function')
    
    def model_forward(self, input_data):
        input_to_next_layer = input_data.T # The first layer receives the pixels of the images as input

        for layer in self.model_state.keys(): # Iterate through all layers while performing the forward pass
            self.model_state[layer]['linear_input'] = input_to_next_layer
            self.model_state[layer]['linear_output'] = self.linear_forward(input_to_next_layer, self.model_state[layer]['weights'], self.model_state[layer]['offset_b'])
            
            if layer != 'output_layer':
                self.model_state[layer]['activation_model'] = 'relu'
                print('Layer: ' + str(layer))
                print('Activation function: ' + self.model_state[layer]['activation_model'])
                print("Shape of linear output: " + str(self.model_state[layer]['linear_output'].shape) + '\n')
                self.model_state[layer]['activation_output'] = self.activation_forward(self.model_state[layer]['linear_output'], self.model_state[layer]['activation_model'])
                input_to_next_layer = self.model_state[layer]['activation_output']
            else:
                self.model_state[layer]['activation_model'] = 'softmax'
                self.model_state[layer]['activation_output'] = self.activation_forward(self.model_state[layer]['linear_output'], self.model_state[layer]['activation_model'])
        
        return self.model_state['output_layer']['activation_output']

    def sigmoid_backward(self, Z, upstream_gradient):
        sig = self.sigmoid(Z)
        return upstream_gradient * sig * (1 - sig)

    def relu_backward(self, Z, upstream_gradient):
        tmp = np.array(upstream_gradient, copy = True)
        tmp[Z <= 0] = 0
        return tmp
    
    def activation_backward(self, activation_output, upstream_gradient, activation_model):
        if activation_model == 'sigmoid':
            return self.sigmoid_backward(activation_output, upstream_gradient)
        elif activation_model == 'relu':
            return self.relu_backward(activation_output, upstream_gradient)
        elif activation_model == 'softmax':
            return self.sigmoid_backward(activation_output, upstream_gradient)
        else:
            raise Exception('Non-supported activation function')
    
    def linear_backward(self, linear_input, upstream_gradient):
        print("Upstream gradient shape: ", upstream_gradient.shape)
        print("Linear input shape: ", linear_input.shape)
        gradient_weights = np.dot(upstream_gradient, linear_input.T)
        gradient_b = np.sum(upstream_gradient, axis=1, keepdims=True)
        return gradient_weights, gradient_b
    
    def model_backward(self, true_labels):
        predictions = self.model_state['output_layer']['activation_output']
        upstream_gradient = predictions - true_labels.T

        for layer in reversed(list(self.model_state.keys())): # Iterate through all layers (reversed) while performing backprop
            print(self.model_state[layer].keys())
            activation_model = self.model_state[layer]['activation_model']
            linear_output = self.model_state[layer]['linear_output']
            linear_input = self.model_state[layer]['linear_input']
            upstream_gradient = self.activation_backward(linear_output, upstream_gradient, activation_model)
            print(upstream_gradient.shape)
            breakpoint()
            gradient_weights, gradient_b = self.linear_backward(linear_input, upstream_gradient)
            print(f"gradient shape ", gradient_weights.shape)
            breakpoint()
            self.model_state[layer]['gradient_weights'] = gradient_weights
            self.model_state[layer]['gradient_b'] = gradient_b
            upstream_gradient = self.model_state[layer]['weights'].T.dot(upstream_gradient)

    def update_parameters(self, learning_rate):
        for layer in self.model_state.keys():
            self.model_state[layer]['weights'] = self.model_state[layer]['weights'] - (learning_rate * self.model_state[layer]['gradient_weights'])
            self.model_state[layer]['offset_b'] = self.model_state[layer]['offset_b'] - (learning_rate * self.model_state[layer]['gradient_b'])

    def compute_loss(self, true_labels):
        predictions = self.model_state['output_layer']['activation_output'].T
        loss = np.log(np.sum(np.exp(predictions), axis=1)) - np.sum(np.multiply(true_labels, predictions), axis=1)
        return loss
    
    def predict(self, x, one_hot_encoded_label):
        self.model_forward(x)
        predictions = np.argmax(self.model_state['output_layer']['activation_output'], axis=0)
        classification = np.argmax(one_hot_encoded_label, axis=1)
        accuracy = (predictions == classification).mean()
        return accuracy * 100
    
    def random_mini_batches(self, x, y, mini_batch_size):
        mini_batches = []
        number_full_batches = math.floor(self.number_images / mini_batch_size)
        permutation = list(np.random.permutation(self.number_images))
        shuffled_images = x[permutation, :]
        shuffled_labels = y[permutation, :]

        for batch in range(0, number_full_batches):
            start_index = batch * mini_batch_size
            stop_index = start_index + mini_batch_size
            batch_images = shuffled_images[start_index : stop_index, :]
            batch_labels = shuffled_labels[start_index : stop_index, :]
            mini_batch = (batch_images, batch_labels)
            mini_batches.append(mini_batch)

        if self.number_images % mini_batch_size != 0:
            batch_images = shuffled_images[number_full_batches * mini_batch_size : self.number_images, :]
            batch_labels = shuffled_labels[number_full_batches * mini_batch_size : self.number_images, :]
            mini_batch = (batch_images, batch_labels)
            mini_batches.append(mini_batch)
        
        return mini_batches
    
    def plot_cost_iteration(self, iterations):
        plt.plot(np.arange(start=0, stop=iterations, step= 1), self.costs_train, label='Training data')
        plt.plot(np.arange(start=0, stop=iterations, step= 1), self.costs_test, label='Test data')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.title('Cost per Iteration')
        plt.legend()
        plt.show()
    
    def plot_accuracy_iteration(self, iterations):
        plt.plot(np.arange(start=1, stop=iterations+1, step= 1), self.accuracies_train, label='Training data')
        plt.plot(np.arange(start=1, stop=iterations+1, step= 1), self.accuracies_test, label='Test data')
        plt.xlabel('Iteration')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Iteration')
        plt.legend()
        plt.show()
    
    def plot_weights(self):
        weight_matrix = self.model_state['output_layer']['weights']
        figsize = [9.5, 5]
        nrows, ncols = 2, 5
        fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
        for counter, axi in enumerate(ax.flat):
            img = weight_matrix[counter, :].reshape((28,28))
            axi.imshow(img)
            axi.set_title("Row: " +str(counter))
        plt.show()

    def train_model(self, x_train, y_train, x_test, y_test, iterations, learning_rate, mini_batch_size):
        self.initialise_parameters(x_train, y_train)
        print('Training started ...\n')
        for epoch in range(0, iterations): # Iterate through the specified number of iterations (ephocs)
            ephoc_cost = 0
            ephoc_accuracy = 0
            mini_batches = self.random_mini_batches(x_train, y_train, mini_batch_size)
            for mini_batch in mini_batches: # Iterate through all mini-batches for this particular ephoc
                (mini_x_train, mini_y_train) = mini_batch
                self.model_forward(mini_x_train)
                loss = self.compute_loss(mini_y_train)
                ephoc_cost += np.mean(loss, axis=0)
                self.model_backward(mini_y_train)
                self.update_parameters(learning_rate)
                ephoc_accuracy += self.predict(mini_x_train, mini_y_train)
                print("All shapes")
                for layer in self.model_state.keys():
                    print('Layer: ' + str(layer))
                    print('Weights shape: ' + str(self.model_state[layer]['weights'].shape))
                    print('Offset shape: ' + str(self.model_state[layer]['offset_b'].shape))
                    print('Gradient Weights shape: ' + str(self.model_state[layer]['gradient_weights'].shape))
                    print('Gradient Offset shape: ' + str(self.model_state[layer]['gradient_b'].shape) + '\n')
                breakpoint()
            self.costs_train.append(ephoc_cost/len(mini_batches))
            self.accuracies_train.append(ephoc_accuracy/len(mini_batches))
            self.accuracies_test.append(self.predict(x_test, y_test))
            self.model_forward(x_test)
            self.costs_test.append(np.mean(self.compute_loss(y_test), axis=0))
            
            if epoch % 10 == 0:
                print("Train Cost: " + str(self.costs_train[epoch]))
            if epoch % 100 == 0 and epoch != 0:
                print("Train Accuracy: " + str(self.accuracies_train[epoch]) + '\n')
            elif epoch == (iterations - 1):
                print('Training completed!\n')
                print("Final Train Accuracy: " + str(self.accuracies_train[epoch]))
                print("Test Accuracy: " + str(self.accuracies_test[epoch]))
        
        self.plot_cost_iteration(iterations)
        self.plot_accuracy_iteration(iterations)
        if len(self.model_state) == 1: # Print rows of weight matrix as 28x28 images for 1-layer network
            self.plot_weights()

if __name__ == '__main__':
    print("Loading MNIST data ...")
    x_train, y_train, x_test, y_test = load_mnist()
    print("Data loaded!\n")
    print('X_train shape: ' + str(x_train.shape))
    print('Y_train shape: ' + str(y_train.shape))

    # Dictionary defining the number of nodes in each hidden layer. Use model = {} for 1-layer network.
    model = {'layer_1': {'nodes': 100}}
    neural_network = feedforward_backprop_neuralNetwork(model)
    neural_network.train_model(x_train, y_train, x_test, y_test, iterations=20, learning_rate=0.005, mini_batch_size=32)