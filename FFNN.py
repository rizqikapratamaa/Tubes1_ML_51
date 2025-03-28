import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm

class FFNN:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.5, hidden_activations=["sigmoid"], 
                 output_activation="sigmoid", loss_function='mse', zero_init=False, init_type="uniform", 
                 lower_bound=-1, upper_bound=1, mean=0, variance=1, seed=None, reg_type='none', reg_lambda = 0.01,
                 rms_norm=False, rms_epsilon=1e-8):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.num_hidden_layers = len(hidden_sizes)
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.loss_function = loss_function.lower()
        self.reg_type = reg_type.lower()
        self.reg_lambda = reg_lambda
        self.rms_norm = rms_norm
        self.rms_epsilon = rms_epsilon
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation

        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []
        self.weights_grad = []
        self.biases_grad = []
        self.rms_weights_cache = []
        self.rms_biases_cache = []

        layer_sizes = [input_size] + hidden_sizes + [output_size]

        for i in range(len(layer_sizes) - 1):
            if not zero_init:
                if init_type == "uniform":
                    w = np.random.uniform(lower_bound, upper_bound, (layer_sizes[i], layer_sizes[i+1]))
                    b = np.random.uniform(lower_bound, upper_bound, layer_sizes[i+1])
                elif init_type == "normal":
                    w = np.random.normal(mean, np.sqrt(variance), (layer_sizes[i], layer_sizes[i+1]))
                    b = np.random.normal(mean, np.sqrt(variance), layer_sizes[i+1])
                elif init_type == "xavier":
                    limit = np.sqrt(6 / (layer_sizes[i] + layer_sizes[i+1]))
                    w = np.random.uniform(-limit, limit, (layer_sizes[i], layer_sizes[i+1]))
                    b = np.zeros(layer_sizes[i+1])
                elif init_type == "he":
                    w = np.random.normal(0, np.sqrt(2 / layer_sizes[i]), (layer_sizes[i], layer_sizes[i+1]))
                    b = np.zeros(layer_sizes[i+1])
                else:
                    raise ValueError("init_type should be 'uniform', 'normal', 'xavier', or 'he'")
            else:
                w = np.zeros((layer_sizes[i], layer_sizes[i+1]))
                b = np.zeros(layer_sizes[i+1])

            self.weights.append(w)
            self.biases.append(b)
            self.weights_grad.append(np.zeros_like(w))
            self.biases_grad.append(np.zeros_like(b))

            if rms_norm:
                self.rms_weights_cache.append(np.zeros_like(w))
                self.rms_biases_cache.append(np.zeros_like(b))

            self.history = {"train_loss": [], "val_loss": []}
    
    def apply_activation(self, x, activation_type):
        if activation_type == 'linear':
            return x
        elif activation_type == 'relu':
            return np.maximum(0, x)
        elif activation_type == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif activation_type == 'tanh':
            return np.tanh(x)
        elif activation_type == 'leaky_relu':
            return np.where(x > 0, x, 0.01 * x)
        elif activation_type == 'elu':
            return np.where(x > 0, x, 1.0 * (np.exp(x) - 1))
        elif activation_type == 'softmax':
            exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return exp_x / np.sum(exp_x, axis=-1, keepdims=True)    
    
    def feedforward(self, inputs):
        layer_outputs = [inputs]

        for layer_idx in range(self.num_hidden_layers):
            z = np.dot(layer_outputs[-1], self.weights[layer_idx]) + self.biases[layer_idx]
            a = self.apply_activation(z, self.hidden_activations[layer_idx])
            layer_outputs.append(a)

        z = np.dot(layer_outputs[-1], self.weights[-1]) + self.biases[-1]
        output = self.apply_activation(z, self.output_activation)
        layer_outputs.append(output)

        return layer_outputs

    
    def compute_loss(self, outputs, targets):
        if self.loss_function == 'mse':
            diff = outputs - targets
            return np.mean(diff ** 2, axis=-1)
        elif self.loss_function == 'cce':
            return -np.mean(targets * np.log(outputs + 1e-15), axis=-1)
        elif self.loss_function == 'bce':
            outputs = np.clip(outputs, 1e-15, 1 - 1e-15)
            return -np.mean(targets * np.log(outputs) + (1 - targets) * np.log(1 - outputs), axis=-1)
        
    def compute_loss_grad(self, outputs, targets):
        if self.loss_function == 'mse':
            return 2 * (outputs - targets) / outputs.shape[-1]
        elif self.loss_function == 'cce':
            return (outputs - targets) / outputs.shape[0]
        elif self.loss_function == 'bce':
            outputs = np.clip(outputs, 1e-15, 1 - 1e-15)
            return (outputs - targets) / (outputs * (1 - outputs) * outputs.shape[0])
            
    def compute_regularization_loss(self):
        reg_loss = 0.0
        if self.reg_type == 'l1':
            reg_loss = self.reg_lambda * sum(np.sum(np.abs(w)) for w in self.weights)
        elif self.reg_type == 'l2':
            reg_loss = (self.reg_lambda / 2.0) * sum(np.sum(w ** 2) for w in self.weights)
        return reg_loss

    
    def train(self, training_data, target_data, validation_data, validation_target, epochs, batch_size=32, verbose=1):
        training_data = np.array(training_data)
        target_data = np.array(target_data)
        validation_data = np.array(validation_data)
        validation_target = np.array(validation_target)

        for epoch in range(epochs):
            total_loss = 0
            total_val_loss = 0
            num_batches = math.ceil(len(training_data) / batch_size)

            if verbose == 1:
                pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{epochs}", ncols=100)

            for w_grad, b_grad in zip(self.weights_grad, self.biases_grad):
                w_grad.fill(0)
                b_grad.fill(0)

            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(training_data))
                batch_inputs = training_data[start_idx:end_idx]
                batch_targets = target_data[start_idx:end_idx]

                layer_outputs = self.feedforward(batch_inputs)
                outputs = layer_outputs[-1]

                batch_loss = np.mean(self.compute_loss(outputs, batch_targets))
                if self.reg_type != 'none':
                    reg_loss = self.compute_regularization_loss()
                    batch_loss += reg_loss
                total_loss += batch_loss

                delta = self.compute_loss_grad(outputs, batch_targets)
                for layer_idx in range(self.num_hidden_layers, -1, -1):
                    a_prev = layer_outputs[layer_idx]
                    w = self.weights[layer_idx]
                    delta_next = delta

                    self.weights_grad[layer_idx] += np.dot(a_prev.T, delta)
                    self.biases_grad[layer_idx] += np.sum(delta, axis=0)

                    if layer_idx > 0:
                        delta = np.dot(delta, w.T)
                        if self.hidden_activations[layer_idx-1] == 'sigmoid':
                            delta *= layer_outputs[layer_idx] * (1 - layer_outputs[layer_idx])
                        elif self.hidden_activations[layer_idx-1] == 'relu':
                            delta *= (layer_outputs[layer_idx] > 0).astype(float)

                for layer_idx in range(len(self.weights)):
                    if self.reg_type == 'l1':
                        self.weights_grad[layer_idx] += self.reg_lambda * np.sign(self.weights[layer_idx])
                    elif self.reg_type == 'l2':
                        self.weights_grad[layer_idx] += self.reg_lambda * self.weights[layer_idx]

                    if self.rms_norm:
                        self.rms_weights_cache[layer_idx] = 0.9 * self.rms_weights_cache[layer_idx] + 0.1 * (self.weights_grad[layer_idx] ** 2)
                        adjusted_lr = self.learning_rate / (np.sqrt(self.rms_weights_cache[layer_idx] + self.rms_epsilon))
                        self.weights[layer_idx] -= adjusted_lr * self.weights_grad[layer_idx]

                        self.rms_biases_cache[layer_idx] = 0.9 * self.rms_biases_cache[layer_idx] + 0.1 * (self.biases_grad[layer_idx] ** 2)
                        adjusted_lr = self.learning_rate / (np.sqrt(self.rms_biases_cache[layer_idx] + self.rms_epsilon))
                        self.biases[layer_idx] -= adjusted_lr * self.biases_grad[layer_idx]
                    else:
                        self.weights[layer_idx] -= self.learning_rate * self.weights_grad[layer_idx]
                        self.biases[layer_idx] -= self.learning_rate * self.biases_grad[layer_idx]

                if verbose == 1:
                    pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

            # Validasi
            val_outputs = self.feedforward(validation_data)[-1]
            total_val_loss = np.mean(self.compute_loss(val_outputs, validation_target))
            if self.reg_type != 'none':
                total_val_loss += self.compute_regularization_loss()

            self.history["train_loss"].append(total_loss / num_batches)
            self.history["val_loss"].append(total_val_loss)

            if verbose == 1:
                print(f"Train Loss: {total_loss / num_batches:.4f} - Val Loss: {total_val_loss:.4f}")


    def predict(self, inputs):
        layer_outputs = self.feedforward(inputs)
        return layer_outputs[-1].tolist()
    
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
        total_weight_layers = self.num_hidden_layers + 1  # Jumlah matriks bobot
        if layer_index < total_weight_layers:
            return [w.value for row in self.weights[layer_index] for w in row]
        elif layer_index < total_weight_layers + self.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [b.value for b in self.biases[bias_idx]]
        else:
            raise ValueError(f"Layer index harus antara 0 dan {total_weight_layers + self.num_hidden_layers}")

    def get_layer_gradients(self, layer_index):
        total_weight_layers = self.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            return [w.gradient for row in self.weights[layer_index] for w in row]
        elif layer_index < total_weight_layers + self.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [b.gradient for b in self.biases[bias_idx]]
        else:
            raise ValueError(f"Layer index harus antara 0 dan {total_weight_layers + self.num_hidden_layers}")

    def get_layer_name(self, layer_index):
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

    def visualize_network(self):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_title("Feedforward Neural Network Structure")

        layer_sizes = [self.input_size] + self.hidden_sizes + [self.output_size]
        num_layers = len(layer_sizes)

        layer_spacing = 1.0
        max_neurons = max(layer_sizes)
        neuron_spacing = 1.0 / (max_neurons + 1)

        neuron_positions = {}
        for layer_idx, size in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            for neuron_idx in range(size):
                y = (max_neurons / 2 - size / 2 + neuron_idx + 0.5) * neuron_spacing
                neuron_positions[(layer_idx, neuron_idx)] = (x, y)

        for (layer_idx, neuron_idx), (x, y) in neuron_positions.items():
            if layer_idx == 0:
                label = f"I{neuron_idx}"
            elif layer_idx == num_layers - 1:
                label = f"O{neuron_idx}"
                bias = self.biases[-1][neuron_idx].value
                ax.text(x, y + 0.05, f"b={bias:.2f}", fontsize=8, ha='center')
            else:
                label = f"H{layer_idx-1}_{neuron_idx}"
                bias = self.biases[layer_idx-1][neuron_idx].value
                ax.text(x, y + 0.05, f"b={bias:.2f}", fontsize=8, ha='center')
            ax.scatter(x, y, s=100, label=label if neuron_idx == 0 else None)
            ax.text(x, y, label, fontsize=8, ha='center', va='center', color='white')

        for layer_idx in range(num_layers - 1):
            for i in range(layer_sizes[layer_idx]):
                for j in range(layer_sizes[layer_idx + 1]):
                    x1, y1 = neuron_positions[(layer_idx, i)]
                    x2, y2 = neuron_positions[(layer_idx + 1, j)]
                    w = self.weights[layer_idx][i][j].value
                    g = self.weights[layer_idx][i][j].gradient
                    ax.plot([x1, x2], [y1, y2], 'k-', alpha=0.2)
                    mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                    ax.text(mid_x, mid_y, f"w={w:.2f}\ng={g:.2f}", fontsize=6, alpha=0.7)

        for layer_idx, size in enumerate(layer_sizes):
            x = layer_idx * layer_spacing
            y = max_neurons * neuron_spacing + 0.1
            if layer_idx == 0:
                ax.text(x, y, "Input Layer", ha='center')
            elif layer_idx == num_layers - 1:
                ax.text(x, y, f"Output Layer\n{self.output_activation}", ha='center')
            else:
                ax.text(x, y, f"Hidden Layer {layer_idx}\n{self.hidden_activations[layer_idx-1]}", ha='center')

        ax.set_xlim(-0.5, (num_layers - 1) * layer_spacing + 0.5)
        ax.set_ylim(-0.1, max_neurons * neuron_spacing + 0.3)
        ax.set_aspect('equal')
        ax.axis('off')
        plt.show()
