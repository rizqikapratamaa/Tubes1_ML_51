import numpy as np

class Variable:
    def __init__(self, value, requires_grad=True):
        self.value = np.array(value, dtype=float)
        self.grad = np.zeros_like(self.value) if requires_grad else None
        self.requires_grad = requires_grad
        self.children = []
        self.operation = None

    def backward(self, grad=None):
        if not self.requires_grad:
            return
        if grad is None:
            grad = np.ones_like(self.value)
        self.grad = self.grad + grad if self.grad is not None else grad
        for child, child_grad in self.children:
            if child.requires_grad:
                child.backward(child_grad(self.value, grad))

    def add(x, y):
        result = Variable(x.value + y.value)
        if x.requires_grad or y.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad),
                (y, lambda out, grad: grad)
            ]
        return result

    def matmul(x, y):
        result = Variable(np.dot(x.value, y.value))
        if x.requires_grad or y.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: np.dot(grad, y.value.T)),
                (y, lambda out, grad: np.dot(x.value.T, grad))
            ]
        return result

    def sigmoid(x):
        sig = 1 / (1 + np.exp(-x.value))
        result = Variable(sig)
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * sig * (1 - sig))
            ]
        return result

    def relu(x):
        result = Variable(np.maximum(0, x.value))
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * (x.value > 0).astype(float))
            ]
        return result

    def tanh(x):
        result = Variable(np.tanh(x.value))
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * (1 - np.tanh(x.value)**2))
            ]
        return result

    def leaky_relu(x):
        result = Variable(np.where(x.value > 0, x.value, 0.01 * x.value))
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * np.where(x.value > 0, 1, 0.01))
            ]
        return result

    def elu(x):
        result = Variable(np.where(x.value > 0, x.value, 1.0 * (np.exp(x.value) - 1)))
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * np.where(x.value > 0, 1, np.exp(x.value)))
            ]
        return result

    def softmax(x):
        exp_x = np.exp(x.value - np.max(x.value, axis=-1, keepdims=True))
        result = Variable(exp_x / np.sum(exp_x, axis=-1, keepdims=True))
        if x.requires_grad:
            result.requires_grad = True
            result.children = [
                (x, lambda out, grad: grad * result.value * (1 - result.value))  # Simplified softmax grad
            ]
        return result

    def mse_loss(output, target):
        diff = output.value - target
        result = Variable(np.mean(diff ** 2, axis=-1))
        if output.requires_grad:
            result.requires_grad = True
            result.children = [
                (output, lambda out, grad: 2 * (output.value - target) / output.value.shape[-1])
            ]
        return result

    def bce_loss(output, target):
        output_clipped = np.clip(output.value, 1e-15, 1 - 1e-15)
        result = Variable(-np.mean(target * np.log(output_clipped) + (1 - target) * np.log(1 - output_clipped), axis=-1))
        if output.requires_grad:
            result.requires_grad = True
            result.children = [
                (output, lambda out, grad: (output_clipped - target) / (output_clipped * (1 - output_clipped) * output.value.shape[0]))
            ]
        return result

    def cce_loss(output, target):
        result = Variable(-np.mean(target * np.log(output.value + 1e-15), axis=-1))
        if output.requires_grad:
            result.requires_grad = True
            result.children = [
                (output, lambda out, grad: (output.value - target) / output.value.shape[0])
            ]
        return result