import random
import math

class Node:
    def __init__(self, value, gradient=0.0):
        self.value = value # node value
        self.gradient = gradient # node gradient
        self.parent = [] # list of parent node, it contains tuple (parent_node, local_gradient)
        self.op = None # operation that produce this node
    
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value + other.value)
        result.parent = [(self, 1.0), (other, 1.0)] # gradient to its parents
        self.op = "add"
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value * other.value)
        result.parent = [(self, other.value), (other, self.value)] # local gradient
        result.op = "mul"
        return result

    def __sub__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value - other.value)
        result.parent = [(self, 1.0), (other, -1.0)]
        result.op = "sub"
        return result

    def exp(self):
        value = math.exp(self.value)
        result = Node(value)
        result.parent = [(self, value)] # d/dx e^x = e^x
        result.op = "exp"
        return result

    def linear(self):
        result = Node(self.value)
        result.parent = [(self, 1.0)]
        result.op = "linear"
        return result
    
    def relu(self):
        value = max(0, self.value)
        result = Node(value)
        result.parent = [(self, 1.0 if self.value > 0.0 else 0.0)]
        result.op = "relu"
        return result
    
    def sigmoid(self):
        value = 1 / (1 + math.exp(-self.value))
        result = Node(value)
        result.parent = [(self, value * (1 - value))]
        result.op = "sigmoid"
        return result
    
    def tanh(self):
        value = math.tanh(self.value)
        result = Node(value)
        result.parent = [(self, 1 - value * value)]
        result.op = "tanh"
        return result

    def leaky_relu(self, alpha=0.01):
        value = self.value if self.value > 0 else alpha * self.value
        result = Node(value)
        result.parent = [(self, 1.0 if self.value > 0 else alpha)]
        result.op = "leaky_relu"
        return result

    def elu(self, alpha=1.0):
        value = self.value if self.value > 0 else alpha * (math.exp(self.value) - 1)
        result = Node(value)
        result.parent = [(self, 1.0 if self.value > 0 else alpha * math.exp(self.value))]
        result.op = "elu"
        return result

    def backward(self, gradient=1.0):
        self.gradient += gradient
        for parent, local_gradient in self.parent:
            parent.backward(gradient * local_gradient)
