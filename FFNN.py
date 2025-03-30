import math, os
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from tqdm import tqdm
from Tensor import Tensor

class FFNN:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.5, hidden_activations=["sigmoid"], 
                 output_activation="sigmoid", loss_function='mse', zero_init=False, init_type="uniform", 
                 lower_bound=-1, upper_bound=1, mean=0, variance=1, seed=None, reg_type='none', reg_lambda = 0.01,
                 rms_norm=False, rms_epsilon=1e-8, rms_prop=False):
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
        self.rms_prop = rms_prop
        self.hidden_activations = hidden_activations
        self.output_activation = output_activation

        if seed is not None:
            np.random.seed(seed)

        self.weights = []
        self.biases = []
        self.weights_grad = []
        self.biases_grad = []

        if self.rms_prop:
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

            self.weights.append(Tensor(w))
            self.biases.append(Tensor(b))
            self.weights_grad.append(np.zeros_like(w))
            self.biases_grad.append(np.zeros_like(b))

            if rms_prop:
                self.rms_weights_cache.append(np.zeros_like(w))
                self.rms_biases_cache.append(np.zeros_like(b))

        self.history = {"train_loss": [], "val_loss": []}
    
    def apply_activation(self, x, activation_type):
        if activation_type == 'linear':
            return x
        elif activation_type == 'relu':
            return x.relu()
        elif activation_type == 'sigmoid':
            return x.sigmoid()
        elif activation_type == 'tanh':
            return x.tanh()
        elif activation_type == 'leaky_relu':
            return x.leaky_relu()
        elif activation_type == 'elu':
            return x.elu()
        elif activation_type == 'softmax':
            return x.softmax() 
    
    def feedforward(self, inputs):
        inputs = Tensor(inputs, requires_grad=False)
        layer_outputs = [inputs]

        for layer_idx in range(self.num_hidden_layers):
            z = layer_outputs[-1].dot(self.weights[layer_idx]) + self.biases[layer_idx]
            a = self.apply_activation(z, self.hidden_activations[layer_idx])
            if self.rms_norm:
                a = a.rms_norm(epsilon=self.rms_epsilon)
            layer_outputs.append(a)

        z = layer_outputs[-1].dot(self.weights[-1]) + self.biases[-1]
        output = self.apply_activation(z, self.output_activation)
        layer_outputs.append(output)

        return layer_outputs

    
    def compute_loss(self, outputs, targets):
        targets = Tensor(targets, requires_grad=False)
        if self.loss_function == 'mse':
            diff = outputs - targets
            return (diff * diff).data.mean(axis=-1)
        elif self.loss_function == 'cce':
            return (-(targets * np.log(outputs.data + 1e-15))).data.mean(axis=-1)
        elif self.loss_function == 'bce':
            outputs_data = np.clip(outputs.data, 1e-15, 1 - 1e-15)
            return (-(targets * np.log(outputs_data) + (1 - targets) * np.log(1 - outputs_data))).data.mean(axis=-1)
        
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
            reg_loss = self.reg_lambda * sum(np.sum(np.abs(w.data)) for w in self.weights)
        elif self.reg_type == 'l2':
            reg_loss = (self.reg_lambda / 2.0) * sum(np.sum(w.data ** 2) for w in self.weights)
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

            for batch_idx in pbar:
                start_idx = batch_idx * batch_size
                end_idx = min((batch_idx + 1) * batch_size, len(training_data))
                batch_inputs = training_data[start_idx:end_idx]
                batch_targets = target_data[start_idx:end_idx]

                for w, b in zip(self.weights, self.biases):
                    w.grad = np.zeros_like(w.data)
                    b.grad = np.zeros_like(b.data)

                layer_outputs = self.feedforward(batch_inputs)
                outputs = layer_outputs[-1]

                if self.loss_function == 'mse':
                    diff = outputs - Tensor(batch_targets, requires_grad=False)
                    loss = (diff * diff).mean()
                elif self.loss_function == 'cce':
                    targets = Tensor(batch_targets, requires_grad=False)
                    loss = (-(targets * outputs.log())).mean()
                elif self.loss_function == 'bce':
                    targets = Tensor(batch_targets, requires_grad=False)
                    outputs_clipped = Tensor(np.clip(outputs.data, 1e-15, 1 - 1e-15))
                    loss = (-(targets * outputs_clipped.log() + (Tensor(1) - targets) * (Tensor(1) - outputs_clipped).log())).mean()

                batch_loss = loss.data
                if self.reg_type != 'none':
                    reg_loss = self.compute_regularization_loss()
                    batch_loss += reg_loss
                total_loss += batch_loss

                loss.backward()

                for layer_idx in range(len(self.weights)):
                    w_grad = self.weights[layer_idx].grad
                    b_grad = self.biases[layer_idx].grad

                    if self.reg_type == 'l1':
                        w_grad += self.reg_lambda * np.sign(self.weights[layer_idx].data)
                    elif self.reg_type == 'l2':
                        w_grad += self.reg_lambda * self.weights[layer_idx].data

                    if self.rms_prop:
                        self.rms_weights_cache[layer_idx] = 0.9 * self.rms_weights_cache[layer_idx] + 0.1 * (w_grad ** 2)
                        adjusted_lr = self.learning_rate / (np.sqrt(self.rms_weights_cache[layer_idx] + self.rms_epsilon))
                        self.weights[layer_idx].data -= adjusted_lr * w_grad

                        self.rms_biases_cache[layer_idx] = 0.9 * self.rms_biases_cache[layer_idx] + 0.1 * (b_grad ** 2)
                        adjusted_lr = self.learning_rate / (np.sqrt(self.rms_biases_cache[layer_idx] + self.rms_epsilon))
                        self.biases[layer_idx].data -= adjusted_lr * b_grad
                    else:
                        self.weights[layer_idx].data -= self.learning_rate * w_grad
                        self.biases[layer_idx].data -= self.learning_rate * b_grad

                if verbose == 1:
                    pbar.set_postfix({'loss': total_loss / (batch_idx + 1)})

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
        return layer_outputs[-1].data.tolist()
    
    # compare sama sklearn MLP
    def compare_lib(self, X_train, y_train, X_test, y_test):
        mlp_activation = self.hidden_activations[0] if self.hidden_activations[0] in {'relu', 'sigmoid', 'tanh'} else 'relu'
        mlp = MLPClassifier(hidden_layer_sizes=tuple(self.hidden_sizes), 
                            activation=mlp_activation, 
                            solver='sgd', 
                            learning_rate_init=self.learning_rate, 
                            random_state=42,
                            max_iter=200)
        
        y_train_mlp = np.argmax(y_train, axis=1)
        mlp.fit(X_train, y_train_mlp)

        y_pred_ffnn = [np.argmax(self.predict(x)) for x in X_test]
        
        y_pred_mlp = mlp.predict(X_test)

        y_test_true = np.argmax(y_test, axis=1)
        
        acc_ffnn = accuracy_score(y_test_true, y_pred_ffnn)
        acc_mlp = accuracy_score(y_test_true, y_pred_mlp)

        print(f"Accuracy FFNN: {acc_ffnn * 100:.2f}%")
        print(f"Accuracy MLP Sklearn: {acc_mlp * 100:.2f}%")

    def save_model(self, filename="ffnn_model.npz"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        model_params = {
            'input_size': self.input_size,
            'hidden_sizes': self.hidden_sizes,
            'output_size': self.output_size,
            'learning_rate': self.learning_rate,
            'hidden_activations': self.hidden_activations,
            'output_activation': self.output_activation,
            'loss_function': self.loss_function,
            'reg_type': self.reg_type,
            'reg_lambda': self.reg_lambda,
            'rms_norm': self.rms_norm,
            'rms_epsilon': self.rms_epsilon,
            'rms_prop': self.rms_prop
        }
        
        for i in range(len(self.weights)):
            model_params[f'weights_{i}'] = self.weights[i].data
            model_params[f'biases_{i}'] = self.biases[i].data
        
        if self.rms_prop:
            for i in range(len(self.rms_weights_cache)):
                model_params[f'rms_weights_cache_{i}'] = self.rms_weights_cache[i]
                model_params[f'rms_biases_cache_{i}'] = self.rms_biases_cache[i]
        
        model_params['history'] = self.history
        
        np.savez(filepath, **model_params)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filename="ffnn_model.npz"):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        filepath = os.path.join(current_dir, filename)
        
        data = np.load(filepath, allow_pickle=True)
        
        input_size = int(data['input_size'])
        hidden_sizes = data['hidden_sizes'].tolist()
        output_size = int(data['output_size'])
        learning_rate = float(data['learning_rate'])
        hidden_activations = data['hidden_activations'].tolist()
        output_activation = str(data['output_activation'])
        loss_function = str(data['loss_function'])
        reg_type = str(data['reg_type'])
        reg_lambda = float(data['reg_lambda'])
        rms_norm = bool(data['rms_norm'])
        rms_epsilon = float(data['rms_epsilon'])
        rms_prop = float(data['rms_prop'])
        
        model = cls(
            input_size=input_size,
            hidden_sizes=hidden_sizes,
            output_size=output_size,
            learning_rate=learning_rate,
            hidden_activations=hidden_activations,
            output_activation=output_activation,
            loss_function=loss_function,
            reg_type=reg_type,
            reg_lambda=reg_lambda,
            rms_norm=rms_norm,
            rms_epsilon=rms_epsilon,
            rms_prop = rms_prop
        )
        
        num_layers = len(hidden_sizes) + 1
        for i in range(num_layers):
            model.weights[i] = Tensor(data[f'weights_{i}'])
            model.biases[i] = Tensor(data[f'biases_{i}'])
            
            if rms_prop:
                model.rms_weights_cache[i] = data[f'rms_weights_cache_{i}']
                model.rms_biases_cache[i] = data[f'rms_biases_cache_{i}']
        
        model.history = data['history'].item()
        
        print(f"Model loaded from {filepath}")
        return model