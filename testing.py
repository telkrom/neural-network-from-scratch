inputs = [1, 2, 3, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
biases = [2, 3, 0.5] # 3 biases == 3 neurons

# Output of current layer
layer_outputs = []

# For each neuron
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0

    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight

    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)

# print(layer_outputs) #[4.8, 1.21, 2.385]

import numpy as np 

inputs = [1.0, 2.0, 3.0, 2.5]
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(weights, inputs) + biases
# print(outputs) # [4.8   1.21  2.385]

# Row Vector 
a = np.array([[1,2,3]])
# print(a.shape) # (1, 3)

# Column Vector
b = np.array([[2,3,4]]).T
# print(b.shape) # (3, 1)

abdot = np.dot(a, b)
badot = np.dot(b, a)

"""
print(abdot, abdot.shape)
print("----")
print(badot, badot.shape)
"""

# ***** Single Layer *****
# Samples of data
inputs = [
        [1.0, 2.0, 3.0, 2.5],
        [2.0, 5.0, -1.0, 2.0],
        [-1.5, 2.7, 3.3, -0.8]]
weights = [
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

outputs = np.dot(inputs, np.array(weights).T) + biases
# IMPORTANT!!
# outputs will become the inputs for the next layer of neurons

# ***** More Layers *****
inputs = [
    [1.0, 2.0, 3.0, 2.5],
    [2.0, 5.0, -1.0, 2.0],
    [-1.5, 2.7, 3.3, -0.8]]
weights = [
    [0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]
biases = [2.0, 3.0, 0.5]

weights2 = [
    [0.1, -0.14, 0.5],
    [-0.5, 0.12, -0.33],
    [-0.44, 0.73, -0.13]]
biases2 = [-1, 2, -0.5]

# Output of layer1 are the inputs for layer2
layer1_outputs = np.dot(inputs, np.array(weights).T) + biases
layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

# ReLU Activation function
inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]

# ReLU activation
class Activation_ReLU:
# Forward pass
    def forward(self, inputs):
    # Calculate output values from input
        self.output = np.maximum(0, inputs)


# Softmax Activation Function
import math

layer_outputs = inputs =  [2.0, 1.0, 0.1]
E = math.e
exp_values = []
for output in layer_outputs:
    exp_values.append(E ** output)
#print("Exponentiaded Values:")
#print(exp_values)

# Now Normalize Values
norm_base = sum(exp_values)
norm_values = []
for value in exp_values:
    norm_values.append(value / norm_base)
#print("Normalized Expornentiaded values:")
#print(norm_values)

#print("Sum of normalized values:", sum(norm_values))

print("**** WITH NUMPY ****")
exp_values = np.exp(inputs - np.max(inputs))
probabilities = exp_values / np.sum(exp_values)
print(probabilities)