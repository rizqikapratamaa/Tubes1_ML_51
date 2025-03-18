from Node import Node
import random
import math

class FFNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, hidden_activation = "sigmoid", output_activation = "sigmoid"):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()

        valid_activations = {'linear', 'relu', 'sigmoid', 'tanh'}
        if self.hidden_activation not in valid_activations:
            raise ValueError(f"Hidden activations should be one of: {valid_activations}")
        if self.output_activation not in valid_activations:
            raise ValueError(f"Output activations should be one of: {valid_activations}")
        
        self.weights_input_hidden = [[Node(random.uniform(-1, 1)) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[Node(random.uniform(-1, 1)) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [Node(random.uniform(-1, 1)) for _ in range(hidden_size)]
        self.bias_output = [Node(random.uniform(-1, 1)) for _ in range(output_size)]

    def apply_activation(self, node, activation_type, softmax_output = None):
        if activation_type == 'linear':
            return node.linear()
        elif activation_type == 'relu':
            return node.relu()
        elif activation_type == 'sigmoid':
            return node.sigmoid()
        elif activation_type == 'tanh':
            return node.tanh()
        # tambahin softmax di sini
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")

    def feedforward(self, inputs):
        inputs = [Node(x) for x in inputs]
        hidden_layer = [None] * self.hidden_size
        output_layer = [None] * self.output_size
        
        for i in range(self.hidden_size):
            hidden_sum = self.bias_hidden[i]
            for j in range(self.input_size):
                hidden_sum = hidden_sum + (inputs[j] * self.weights_input_hidden[j][i])
            hidden_layer[i] = self.apply_activation(hidden_sum, self.hidden_activation)
        

        for i in range(self.output_size):
            output_sum = self.bias_output[i]
            for j in range(self.hidden_size):
                output_sum = output_sum + (hidden_layer[j] * self.weights_hidden_output[j][i])
            output_layer[i] = output_sum
        
        if self.output_activation == 'softmax':
            # ntar ubah aja ini pake softmax
            output_layer = [self.apply_activation(o, self.output_activation) for o in output_layer]
        else:
            output_layer = [self.apply_activation(o, self.output_activation) for o in output_layer]
        
        return hidden_layer, output_layer
    
    def compute_loss(self, outputs, targets):
        loss = Node(0.0)
        for i in range(self.output_size):
            diff = outputs[i] + Node(-targets[i])
            loss = loss + (diff * diff)
        loss = loss * Node(1.0 / self.output_size)
        return loss 

    def train(self, training_data, target_data, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in zip(training_data, target_data):
                for row in self.weights_input_hidden:
                    for w in row:
                        w.gradient = 0.0
                for row in self.weights_hidden_output:
                    for w in row:
                        w.gradient = 0.0
                for b in self.bias_hidden:
                    b.gradient = 0.0
                for b in self.bias_output:
                    b.gradient = 0.0

                hidden_layer, outputs = self.feedforward(inputs)

                loss = self.compute_loss(outputs, target)
                total_loss += loss.value

                # print(f"Before backward: w[0][0].gradient = {self.weights_input_hidden[0][0].gradient}")
                loss.backward()
                # print(f"After backward: w[0][0].gradient = {self.weights_input_hidden[0][0].gradient}")

                # update weights and biases base on gradient

                for i in range(self.input_size):
                    for j in range(self.hidden_size):
                        w = self.weights_input_hidden[i][j]
                        w.value -= self.learning_rate * w.gradient
                for i in range(self.hidden_size):
                    for j in range(self.output_size):
                        w = self.weights_hidden_output[i][j]
                        w.value -= self.learning_rate * w.gradient
                for i in range(self.hidden_size):
                    self.bias_hidden[i].value -= self.learning_rate * self.bias_hidden[i].gradient
                for i in range(self.output_size):
                    self.bias_output[i].value -= self.learning_rate * self.bias_output[i].gradient
            
            # if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    def predict(self, inputs):
        _, output = self.feedforward(inputs)
        return [o.value for o in output]
