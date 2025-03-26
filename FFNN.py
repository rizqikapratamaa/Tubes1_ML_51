from Node import Node
from loss import *
import random
import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class FFNN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.5, hidden_activation="sigmoid", 
                 output_activation="sigmoid", loss_function='mse', zero_init=False, init_type="uniform", 
                 lower_bound=-1, upper_bound=1, mean=0, variance=1, seed=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.hidden_activation = hidden_activation.lower()
        self.output_activation = output_activation.lower()
        self.loss_function = loss_function.lower()

        valid_activations = {'linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu'}
        valid_losses = {'mse', 'cce', 'bce'}

        if self.hidden_activation not in valid_activations:
            raise ValueError(f"Hidden activations should be one of: {valid_activations}")
        if self.output_activation not in valid_activations:
            raise ValueError(f"Output activations should be one of: {valid_activations}")
        if self.loss_function not in valid_losses:
            raise ValueError(f"Loss function should be one of: {valid_losses}")
        
        # set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        
        if not zero_init:
            # inisialisasi weights dan bias berdasarkan tipe distribusi
            if init_type == "uniform":
                self.weights_input_hidden = [[Node(random.uniform(lower_bound, upper_bound)) for _ in range(hidden_size)] for _ in range(input_size)]
                self.weights_hidden_output = [[Node(random.uniform(lower_bound, upper_bound)) for _ in range(output_size)] for _ in range(hidden_size)]
                self.bias_hidden = [Node(random.uniform(lower_bound, upper_bound)) for _ in range(hidden_size)]
                self.bias_output = [Node(random.uniform(lower_bound, upper_bound)) for _ in range(output_size)]
            elif init_type == "normal":
                self.weights_input_hidden = [[Node(np.random.normal(mean, math.sqrt(variance))) for _ in range(hidden_size)] for _ in range(input_size)]
                self.weights_hidden_output = [[Node(np.random.normal(mean, math.sqrt(variance))) for _ in range(output_size)] for _ in range(hidden_size)]
                self.bias_hidden = [Node(np.random.normal(mean, math.sqrt(variance))) for _ in range(hidden_size)]
                self.bias_output = [Node(np.random.normal(mean, math.sqrt(variance))) for _ in range(output_size)]
            else:
                raise ValueError(f"Unknown initialization type: {init_type}. Use 'uniform' or 'normal'.")
        else:
            self.weights_input_hidden = [[Node(0.0) for _ in range(hidden_size)] for _ in range(input_size)]
            self.weights_hidden_output = [[Node(0.0) for _ in range(output_size)] for _ in range(hidden_size)]
            self.bias_hidden = [Node(0.0) for _ in range(hidden_size)]
            self.bias_output = [Node(0.0) for _ in range(output_size)]
        
        self.history = {"train_loss": [], "val_loss": []}
    
    def apply_activation(self, node, activation_type):
        if activation_type == 'linear':
            return node.linear()
        elif activation_type == 'relu':
            return node.relu()
        elif activation_type == 'sigmoid':
            return node.sigmoid()
        elif activation_type == 'tanh':
            return node.tanh()
        elif activation_type == 'leaky_relu':
            return node.leaky_relu()
        elif activation_type == 'elu':
            return node.elu()
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
    
    def apply_softmax(self, nodes):
        # apply softmax ke list of nodes
        max_val = max(n.value for n in nodes)
        exp_vals = [Node(math.exp(n.value - max_val)) for n in nodes]
        sum_exp = sum(n.value for n in exp_vals)
        
        return [Node(n.value / sum_exp) for n in exp_vals]
    
    def feedforward(self, inputs):
        inputs = [Node(x) for x in inputs]
        hidden_layer = [None] * self.hidden_size
        output_layer = [None] * self.output_size
        
        for i in range(self.hidden_size):
            hidden_sum = self.bias_hidden[i]
            for j in range(self.input_size):
                hidden_sum = hidden_sum + (inputs[j] * self.weights_input_hidden[j][i])
            if self.hidden_activation == 'softmax':
                # ambil semua hidden_sum values
                hidden_sums = [None] * self.hidden_size
                for k in range(self.hidden_size):
                    h_sum = self.bias_hidden[k]
                    for j in range(self.input_size):
                        h_sum = h_sum + (inputs[j] * self.weights_input_hidden[j][k])
                    hidden_sums[k] = h_sum
                
                # apply softmax ke semua hidden nodes at once
                hidden_activated = self.apply_softmax(hidden_sums)
                for k in range(self.hidden_size):
                    hidden_layer[k] = hidden_activated[k]
            else:
                # for non softmax
                hidden_layer[i] = self.apply_activation(hidden_sum, self.hidden_activation)
        
        for i in range(self.output_size):
            output_sum = self.bias_output[i]
            for j in range(self.hidden_size):
                output_sum = output_sum + (hidden_layer[j] * self.weights_hidden_output[j][i])
            output_layer[i] = output_sum
        
        if self.output_activation == 'softmax':
            output_layer = self.apply_softmax(output_layer)
        else:
            output_layer = [self.apply_activation(o, self.output_activation) for o in output_layer]
        
        return hidden_layer, output_layer
    
    def compute_loss(self, outputs, targets):
        if self.output_activation == 'softmax':
            return cce(outputs, targets, self.output_size)
        else:
            if self.loss_function == 'cce':
                return cce(outputs, targets, self.output_size)
            elif self.loss_function == 'bce':
                return bce(outputs, targets)
            else: # default mse
                loss = Node(0.0)
                for i in range(self.output_size):
                    diff = outputs[i] + Node(-targets[i])
                    loss = loss + (diff * diff)
                loss = loss * Node(1.0 / self.output_size)
                return loss 
    
    def train(self, training_data, target_data, validation_data, validation_target, epochs, batch_size = 1, verbose = 1):
        for epoch in range(epochs):
            total_loss = 0
            total_val_loss = 0
            
            num_batches = math.ceil(len(training_data) / batch_size)
            if verbose == 1:
                pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", ncols=50, bar_format='{l_bar}{bar}| {postfix}', leave=True)
            else:
                pbar = range(num_batches)
                
            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(training_data))
                batch_inputs = training_data[start_idx:end_idx]
                batch_targets = target_data[start_idx:end_idx]
                
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
                
                batch_loss = Node(0.0)
                for inputs, target in zip(batch_inputs, batch_targets):
                    _, outputs = self.feedforward(inputs)
                    loss = self.compute_loss(outputs, target)
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss * Node(1.0 / len(batch_inputs))
                total_loss += batch_loss.value
                
                batch_loss.backward()
                
                # Update weights
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
                
                if verbose == 1:
                    avg_train_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix()
            
            for inputs, target in zip(validation_data, validation_target):
                _, outputs = self.feedforward(inputs)
                loss = self.compute_loss(outputs, target)
                total_val_loss += loss.value
            
            avg_train_loss = total_loss / num_batches
            avg_val_loss = total_val_loss / len(validation_data)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            if verbose == 1:
                print(f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


    def predict(self, inputs):
        _, output = self.feedforward(inputs)
        return [o.value for o in output]
    
    # compare sama sklearn MLP
    def compare_lib(self, X_train, y_train, X_test, y_test):
        mlp = MLPClassifier(hidden_layer_sizes=(self.hidden_size,), activation=self.hidden_activation, solver='sgd', learning_rate_init=self.learning_rate, random_state=42)
        mlp.fit(X_train, [t.index(1) for t in y_train])
        
        y_pred_ffnn = [self.predict(x).index(max(self.predict(x))) for x in X_test]
        y_pred_mlp = mlp.predict(X_test)
        
        acc_ffnn = accuracy_score([t.index(1) for t in y_test], y_pred_ffnn)
        acc_mlp = accuracy_score([t.index(1) for t in y_test], y_pred_mlp)
        
        print(f"Accuracy FFNN: {acc_ffnn * 100:.2f}%")
        print(f"Accuracy MLP Sklearn: {acc_mlp * 100:.2f}%")
    
    def plot_training_history(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.history['train_loss'], label='Training Loss', marker='o', markersize=2)
        plt.plot(self.history['val_loss'], label='Validation Loss', marker='s', markersize=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()
    
    def get_layer_weights(self, layer_index):
        if layer_index == 0:
            # weights input-hidden
            return [w.value for row in self.weights_input_hidden for w in row]
        elif layer_index == 1:
            # weights hidden-output
            return [w.value for row in self.weights_hidden_output for w in row]
        elif layer_index == 2:
            # bias hidden
            return [b.value for b in self.bias_hidden]
        elif layer_index == 3:
            # bias output
            return [b.value for b in self.bias_output]
        else:
            raise ValueError("Layer index harus 0, 1, 2, atau 3")
    
    def get_layer_gradients(self, layer_index):
        if layer_index == 0:
            # weights input-hidden
            return [w.gradient for row in self.weights_input_hidden for w in row]
        elif layer_index == 1:
            # weights hidden-output
            return [w.gradient for row in self.weights_hidden_output for w in row]
        elif layer_index == 2:
            # bias hidden
            return [b.gradient for b in self.bias_hidden]
        elif layer_index == 3:
            # bias output
            return [b.gradient for b in self.bias_output]
        else:
            raise ValueError("Layer index harus 0, 1, 2, atau 3")
    
    def get_layer_name(self, layer_index):
        if layer_index == 0:
            return "Input-Hidden Weights"
        elif layer_index == 1:
            return "Hidden-Output Weights"
        elif layer_index == 2:
            return "Hidden Bias"
        elif layer_index == 3:
            return "Output Bias"
        else:
            return f"Unknown Layer {layer_index}"
    
    def plot_weight_distributions(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = [0, 1, 2, 3]  
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("Tidak ada layer yang ditentukan untuk ditampilkan.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]  
        
        for i, layer_idx in enumerate(layers_to_plot):
            weights = self.get_layer_weights(layer_idx)
            layer_name = self.get_layer_name(layer_idx)
            
            axes[i].hist(weights, bins=30, alpha=0.7)
            axes[i].set_title(f"Weight Distribution - {layer_name}")
            axes[i].set_xlabel("Weights")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_gradient_distributions(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = [0, 1, 2, 3]  
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("Tidak ada layer yang ditentukan untuk ditampilkan.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
        if n_plots == 1:
            axes = [axes]  
        
        for i, layer_idx in enumerate(layers_to_plot):
            gradients = self.get_layer_gradients(layer_idx)
            layer_name = self.get_layer_name(layer_idx)
            
            if all(g == 0 for g in gradients):
                axes[i].text(0.5, 0.5, "Semua gradien bernilai 0", 
                             horizontalalignment='center',
                             verticalalignment='center',
                             transform=axes[i].transAxes)
            else:
                axes[i].hist(gradients, bins=30, alpha=0.7)
            
            axes[i].set_title(f"Gradient Distribution - {layer_name}")
            axes[i].set_xlabel("Gradients")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
