import random
import math

class FFNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        
        self.weights_input_hidden = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(input_size)]
        self.weights_hidden_output = [[random.uniform(-1, 1) for _ in range(output_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def feedforward(self, inputs):
        hidden_layer = [0] * self.hidden_size
        output_layer = [0] * self.output_size
        
        for i in range(self.hidden_size):
            hidden_layer[i] = sum(inputs[j] * self.weights_input_hidden[j][i] for j in range(self.input_size)) + self.bias_hidden[i]
            hidden_layer[i] = self.sigmoid(hidden_layer[i])
        
        for i in range(self.output_size):
            output_layer[i] = sum(hidden_layer[j] * self.weights_hidden_output[j][i] for j in range(self.hidden_size)) + self.bias_output[i]
            output_layer[i] = self.sigmoid(output_layer[i])
        
        return hidden_layer, output_layer

    def backpropagation(self, inputs, hidden_layer, outputs, target):
        output_errors = [target[i] - outputs[i] for i in range(self.output_size)]
        output_deltas = [output_errors[i] * self.sigmoid_derivative(outputs[i]) for i in range(self.output_size)]
        
        hidden_errors = [sum(output_deltas[j] * self.weights_hidden_output[i][j] for j in range(self.output_size)) for i in range(self.hidden_size)]
        hidden_deltas = [hidden_errors[i] * self.sigmoid_derivative(hidden_layer[i]) for i in range(self.hidden_size)]
        
        for i in range(self.hidden_size):
            for j in range(self.output_size):
                self.weights_hidden_output[i][j] += self.learning_rate * output_deltas[j] * hidden_layer[i]
        
        for i in range(self.output_size):
            self.bias_output[i] += self.learning_rate * output_deltas[i]
        
        for i in range(self.input_size):
            for j in range(self.hidden_size):
                self.weights_input_hidden[i][j] += self.learning_rate * hidden_deltas[j] * inputs[i]
        
        for i in range(self.hidden_size):
            self.bias_hidden[i] += self.learning_rate * hidden_deltas[i]

    def train(self, training_data, target_data, epochs):
        for epoch in range(epochs):
            total_loss = 0
            for inputs, target in zip(training_data, target_data):
                hidden_layer, outputs = self.feedforward(inputs)
                self.backpropagation(inputs, hidden_layer, outputs, target)
                total_loss += sum((target[i] - outputs[i]) ** 2 for i in range(self.output_size))
            
            # if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {total_loss}")

    def predict(self, inputs):
        _, output = self.feedforward(inputs)
        return output
