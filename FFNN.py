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
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.5, hidden_activations="sigmoid", 
                 output_activation="sigmoid", loss_function='mse', zero_init=False, init_type="uniform", 
                 lower_bound=-1, upper_bound=1, mean=0, variance=1, seed=None):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = len(hidden_sizes)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function.lower()

        valid_activations = {'linear', 'relu', 'sigmoid', 'tanh', 'softmax', 'leaky_relu', 'elu'}
        valid_losses = {'mse', 'cce', 'bce'}

        if isinstance(hidden_activations, str):
            self.hidden_activations = [hidden_activations.lower()] * self.num_hidden_layers
        else:
            if len(hidden_activations) != self.num_hidden_layers:
                raise ValueError("Activation functions should be as much as the number of the hidden layers.")
            self.hidden_activations = [act.lower() for act in hidden_activations]
        
        self.output_activation = output_activation.lower()

        for act in self.hidden_activations:
            if act not in valid_activations:
                raise ValueError(f"Hidden activations should be one of: {valid_activations}")
            
        if self.output_activation not in valid_activations:
            raise ValueError(f"Output activations should be one of: {valid_activations}")
        if self.loss_function not in valid_losses:
            raise ValueError(f"Loss function should be one of: {valid_losses}")
        
        # set random seed
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.weights = []
        self.biases = []

        layer_sizes = [input_size] + hidden_sizes+ [output_size]


        
        for i in range(len(layer_sizes) - 1):
            if not zero_init:
                if init_type == "uniform":
                    w = [[Node(random.uniform(lower_bound, upper_bound)) 
                          for _ in range(layer_sizes[i+1])] 
                         for _ in range(layer_sizes[i])]
                    b = [Node(random.uniform(lower_bound, upper_bound)) 
                         for _ in range(layer_sizes[i+1])]
                elif init_type == "normal":
                    w = [[Node(np.random.normal(mean, math.sqrt(variance))) 
                          for _ in range(layer_sizes[i+1])] 
                         for _ in range(layer_sizes[i])]
                    b = [Node(np.random.normal(mean, math.sqrt(variance))) 
                         for _ in range(layer_sizes[i+1])]
                else:
                    raise ValueError("init_type should be 'uniform' or 'normal'")
            else:
                w = [[Node(0.0) for _ in range(layer_sizes[i+1])] 
                     for _ in range(layer_sizes[i])]
                b = [Node(0.0) for _ in range(layer_sizes[i+1])]

            self.weights.append(w)
            self.biases.append(b)

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
        layer_outputs = [inputs]

        for layer_idx in range(self.num_hidden_layers):
            current_input = layer_outputs[-1]
            next_size = self.hidden_sizes[layer_idx]
            next_layer = [None] * next_size

            for i in range(next_size):
                layer_sum = self.biases[layer_idx][i]
                for j in range(len(current_input)):
                    layer_sum = layer_sum + (current_input[j] * self.weights[layer_idx][j][i])
                
                if self.hidden_activations[layer_idx] == 'softmax':
                    layer_sums = [None] * next_size
                    for k in range(next_size):
                        s = self.biases[layer_idx][k]
                        for j in range(len(current_input)):
                            s = s + (current_input[j] * self.weights[layer_idx][j][k])
                        layer_sums[k] = s
                    activated = self.apply_softmax(layer_sums)
                    for k in range(next_size):
                        next_layer[k] = activated[k]
                    break
                else:
                    next_layer[i] = self.apply_activation(layer_sum, self.hidden_activations[layer_idx])

            layer_outputs.append(next_layer)

        output_layer = [None] * self.output_size
        for i in range(self.output_size):
            output_sum = self.biases[-1][i]
            for j in range(self.hidden_sizes[-1]):
                output_sum = output_sum + (layer_outputs[-1][j] * self.weights[-1][j][i])
            output_layer[i] = output_sum

        if self.output_activation == 'softmax':
            output_layer = self.apply_softmax(output_layer)
        else:
            output_layer = [self.apply_activation(o, self.output_activation) for o in output_layer]

        layer_outputs.append(output_layer)
        return layer_outputs
    
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
    
    def train(self, training_data, target_data, validation_data, validation_target, epochs, batch_size=1, verbose=1):
        for epoch in range(epochs):
            total_loss = 0
            total_val_loss = 0
            num_batches = math.ceil(len(training_data) / batch_size)

            if verbose == 1:
                pbar = tqdm(range(num_batches), 
                        desc=f"Epoch {epoch+1}/{epochs}", 
                        ncols=50, 
                        bar_format='{l_bar}{bar}| {postfix}')
            else:
                pbar = range(num_batches)

            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(training_data))
                batch_inputs = training_data[start_idx:end_idx]
                batch_targets = target_data[start_idx:end_idx]

                for layer_weights in self.weights:
                    for row in layer_weights:
                        for w in row:
                            w.gradient = 0.0
                for layer_biases in self.biases:
                    for b in layer_biases:
                        b.gradient = 0.0

                batch_loss = Node(0.0)
                for inputs, target in zip(batch_inputs, batch_targets):
                    layer_outputs = self.feedforward(inputs)
                    loss = self.compute_loss(layer_outputs[-1], target)
                    batch_loss = batch_loss + loss
                batch_loss = batch_loss * Node(1.0 / len(batch_inputs))
                total_loss += batch_loss.value

                batch_loss.backward()

                # Update bobot
                for layer_idx in range(len(self.weights)):
                    for i in range(len(self.weights[layer_idx])):
                        for j in range(len(self.weights[layer_idx][i])):
                            w = self.weights[layer_idx][i][j]
                            w.value -= self.learning_rate * w.gradient
                    for i in range(len(self.biases[layer_idx])):
                        b = self.biases[layer_idx][i]
                        b.value -= self.learning_rate * b.gradient

                if verbose == 1:
                    avg_train_loss = total_loss / (batch_idx + 1)
                    pbar.set_postfix_str()

            for inputs, target in zip(validation_data, validation_target):
                layer_outputs = self.feedforward(inputs)
                loss = self.compute_loss(layer_outputs[-1], target)
                total_val_loss += loss.value

            avg_train_loss = total_loss / num_batches
            avg_val_loss = total_val_loss / len(validation_data)
            self.history["train_loss"].append(avg_train_loss)
            self.history["val_loss"].append(avg_val_loss)

            if verbose == 1:
                print(f"Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f}")


    def predict(self, inputs):
        layer_outputs = self.feedforward(inputs)
        return [o.value for o in layer_outputs[-1]]
    
    # compare sama sklearn MLP
    def compare_lib(self, X_train, y_train, X_test, y_test):
        mlp_activation = self.hidden_activations[0] if self.hidden_activations[0] in {'relu', 'sigmoid', 'tanh'} else 'relu'
        mlp = MLPClassifier(hidden_layer_sizes=tuple(self.hidden_sizes), 
                            activation=mlp_activation, 
                            solver='sgd', 
                            learning_rate_init=self.learning_rate, 
                            random_state=42)
        
        y_train_mlp = [t.index(1) for t in y_train]
        mlp.fit(X_train, y_train_mlp)

        y_pred_ffnn = [self.predict(x).index(max(self.predict(x))) for x in X_test]
        y_pred_mlp = mlp.predict(X_test)

        y_test_true = [t.index(1) for t in y_test]
        acc_ffnn = accuracy_score(y_test_true, y_pred_ffnn)
        acc_mlp = accuracy_score(y_test_true, y_pred_mlp)

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
        """
        Mengembalikan bobot untuk layer tertentu.
        - layer_index 0 sampai num_hidden_layers-1: weights antar hidden layer
        - layer_index num_hidden_layers: weights hidden terakhir ke output
        - layer_index num_hidden_layers+1 dan seterusnya: bias untuk setiap layer
        """
        total_weight_layers = self.num_hidden_layers + 1  # Jumlah matriks bobot
        if layer_index < total_weight_layers:
            return [w.value for row in self.weights[layer_index] for w in row]
        elif layer_index < total_weight_layers + self.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [b.value for b in self.biases[bias_idx]]
        else:
            raise ValueError(f"Layer index harus antara 0 dan {total_weight_layers + self.num_hidden_layers}")

    def get_layer_gradients(self, layer_index):
        """
        Mengembalikan gradien untuk layer tertentu.
        - layer_index 0 sampai num_hidden_layers-1: gradien weights antar hidden layer
        - layer_index num_hidden_layers: gradien weights hidden terakhir ke output
        - layer_index num_hidden_layers+1 dan seterusnya: gradien bias untuk setiap layer
        """
        total_weight_layers = self.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            return [w.gradient for row in self.weights[layer_index] for w in row]
        elif layer_index < total_weight_layers + self.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [b.gradient for b in self.biases[bias_idx]]
        else:
            raise ValueError(f"Layer index harus antara 0 dan {total_weight_layers + self.num_hidden_layers}")

    def get_layer_name(self, layer_index):
        """
        Mengembalikan nama layer berdasarkan indeks.
        """
        total_weight_layers = self.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            if layer_index == 0:
                return "Input-Hidden1 Weights"
            elif layer_index < self.num_hidden_layers:
                return f"Hidden{layer_index}-Hidden{layer_index+1} Weights"
            else:
                return f"Hidden{self.num_hidden_layers}-Output Weights"
        elif layer_index < total_weight_layers + self.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            if bias_idx < self.num_hidden_layers:
                return f"Hidden{bias_idx+1} Bias"
            else:
                return "Output Bias"
        else:
            return f"Unknown Layer {layer_index}"

    def plot_weight_distributions(self, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = list(range(self.num_hidden_layers + 1 + self.num_hidden_layers + 1))  # Semua weights dan biases
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("Tidak ada layer yang ditentukan untuk ditampilkan.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
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
            layers_to_plot = list(range(self.num_hidden_layers + 1 + self.num_hidden_layers + 1))
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("Tidak ada layer yang ditentukan untuk ditampilkan.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
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
