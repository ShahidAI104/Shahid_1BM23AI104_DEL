#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt

def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def perceptron_output(x, w, b, activation_func):
    """
    x: input
    w: weight
    b: bias
    activation_func: activation function (linear, sigmoid, tanh)
    """
    z = w * x + b
    return activation_func(z)

x_vals = np.linspace(-20, 30, 400)

weight = 1.0
bias = 0.0

linear_outputs = perceptron_output(x_vals, weight, bias, linear_activation)
sigmoid_outputs = perceptron_output(x_vals, weight, bias, sigmoid_activation)
tanh_outputs = perceptron_output(x_vals, weight, bias, tanh_activation)

plt.figure(figsize=(10, 6))

plt.plot(x_vals, linear_outputs, label="Linear Activation", color='blue')
plt.plot(x_vals, sigmoid_outputs, label="Sigmoid Activation", color='green')
plt.plot(x_vals, tanh_outputs, label="Tanh Activation", color='red')

plt.title("Perceptron Output with Different Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output (y)")
plt.grid(True)
plt.legend()
plt.show()


# In[4]:


import numpy as np
import matplotlib.pyplot as plt

def linear(x):
    return x

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

class Perceptron:
    def __init__(self, activation_fn):
        self.activation_fn = activation_fn
        self.weights = np.random.randn(2)
        self.bias = np.random.randn()

    def forward(self, x):
        z = np.dot(x, self.weights) + self.bias
        return self.activation_fn(z)

X = np.linspace(-20, 30, 400)
inputs = np.array([[x, x] for x in X])

perceptron_linear = Perceptron(linear)
perceptron_sigmoid = Perceptron(sigmoid)
perceptron_tanh = Perceptron(tanh)

output_linear = np.array([perceptron_linear.forward(x) for x in inputs])
output_sigmoid = np.array([perceptron_sigmoid.forward(x) for x in inputs])
output_tanh = np.array([perceptron_tanh.forward(x) for x in inputs])

plt.figure(figsize=(12, 8))

plt.subplot(3, 1, 1)
plt.plot(X, output_linear, label="Linear Activation")
plt.title("Linear Activation Function (y = x)")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.grid(True)

plt.subplot(3, 1, 2)
plt.plot(X, output_sigmoid, label="Sigmoid Activation", color='orange')
plt.title("Sigmoid Activation Function")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.grid(True)

plt.subplot(3, 1, 3)
plt.plot(X, output_tanh, label="Tanh Activation", color='green')
plt.title("Tanh Activation Function")
plt.xlabel("Input x")
plt.ylabel("Output y")
plt.grid(True)

plt.tight_layout()
plt.show()


# In[ ]:




