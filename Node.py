import random
import math

class Node:
    def __init__(self, value, gradient=0.0):
        self.value = value
        self.gradient = gradient
        self.parent = []
        self.op = None
    
    def __add__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value + other.value)
        result.parent = [(self, 1.0), (other, 1.0)]
        result.op = "add"
        return result
    
    def __mul__(self, other):
        other = other if isinstance(other, Node) else Node(other)
        result = Node(self.value * other.value)
        result.parent = [(self, other.value), (other, self.value)]
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
        result.parent = [(self, value)]
        result.op = "exp"
        return result

    def relu(self):
        value = max(0, self.value)
        result = Node(value)
        result.parent = [(self, 1.0 if self.value > 0 else 0.0)]
        result.op = "relu"
        return result
    
    def sigmoid(self):
        value = 1 / (1 + math.exp(-self.value))
        result = Node(value)
        result.parent = [(self, value * (1 - value))]
        result.op = "sigmoid"
        return result

    def backward(self):
        stack = [(self, 1.0)]
        visited = set()
        while stack:
            node, grad = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            node.gradient += grad
            for parent, local_grad in node.parent:
                stack.append((parent, grad * local_grad))