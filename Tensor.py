import numpy as np

class Tensor:
    def __init__(self, data, _children=(), _op='', requires_grad=True):
        self.data = np.array(data, dtype=float)
        self.grad = np.zeros_like(self.data) if requires_grad else None
        self._backward = lambda: None
        self._children = set(_children)
        self._op = _op
        self.requires_grad = requires_grad

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data + other.data, (self, other), '+')
        
        def _backward():
            if self.requires_grad:
                if self.data.shape != out.grad.shape:
                    self.grad += np.sum(out.grad, axis=0)
                else:
                    self.grad += out.grad
            if other.requires_grad:
                if other.data.shape != out.grad.shape:
                    other.grad += np.sum(out.grad, axis=0)
                else:
                    other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(self.data * other.data, (self, other), '*')
        
        def _backward():
            if self.requires_grad:
                self.grad += other.data * out.grad
            if other.requires_grad:
                other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        out = Tensor(-self.data, (self,), 'neg')
        
        def _backward():
            if self.requires_grad:
                self.grad += -out.grad
        out._backward = _backward
        return out

    def dot(self, other):
        other = other if isinstance(other, Tensor) else Tensor(other)
        out = Tensor(np.dot(self.data, other.data), (self, other), 'dot')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.dot(out.grad, other.data.T)
            if other.requires_grad:
                other.grad += np.dot(self.data.T, out.grad)
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(0, self.data), (self,), 'relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += (self.data > 0).astype(float) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        sig = 1 / (1 + np.exp(-self.data))
        out = Tensor(sig, (self,), 'sigmoid')
        
        def _backward():
            if self.requires_grad:
                self.grad += sig * (1 - sig) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = np.tanh(self.data)
        out = Tensor(t, (self,), 'tanh')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 - t**2) * out.grad
        out._backward = _backward
        return out

    def leaky_relu(self):
        out = Tensor(np.where(self.data > 0, self.data, 0.01 * self.data), (self,), 'leaky_relu')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, 0.01) * out.grad
        out._backward = _backward
        return out

    def elu(self):
        out = Tensor(np.where(self.data > 0, self.data, 1.0 * (np.exp(self.data) - 1)), (self,), 'elu')
        
        def _backward():
            if self.requires_grad:
                self.grad += np.where(self.data > 0, 1, 1.0 * np.exp(self.data)) * out.grad
        out._backward = _backward
        return out

    def softmax(self):
        exp_x = np.exp(self.data - np.max(self.data, axis=-1, keepdims=True))
        out_data = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        out = Tensor(out_data, (self,), 'softmax')
        
        def _backward():
            if self.requires_grad:
                s = out.data
                self.grad += (s * (out.grad - np.sum(out.grad * s, axis=-1, keepdims=True)))
        out._backward = _backward
        return out

    def log(self):
        out = Tensor(np.log(self.data + 1e-15), (self,), 'log')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / (self.data + 1e-15)) * out.grad
        out._backward = _backward
        return out

    def mean(self):
        out = Tensor(np.mean(self.data), (self,), 'mean')
        
        def _backward():
            if self.requires_grad:
                self.grad += (1 / self.data.size) * out.grad * np.ones_like(self.data)
        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()

    def rms_norm(self, epsilon=1e-8):
        rms = np.sqrt(np.mean(self.data**2, axis=-1, keepdims=True) + epsilon)
        out_data = self.data / rms
        out = Tensor(out_data, (self,), 'rms_norm', requires_grad=self.requires_grad)

        def _backward():
            if self.requires_grad:
                n = self.data.shape[-1]
                grad_rms = -np.sum(self.data * out.grad, axis=-1, keepdims=True) / (rms**2 * n)
                grad_x = (out.grad / rms) + (self.data * grad_rms / (rms * n))
                self.grad += grad_x
        out._backward = _backward
        return out