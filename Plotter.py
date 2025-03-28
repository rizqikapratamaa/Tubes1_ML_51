import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display

class Plotter:
    def __init__(self):
        pass
    def visualize_network_light(self, ffnn, max_neurons_per_layer=5):
        G = nx.DiGraph()
        
        layer_sizes = [ffnn.input_size] + ffnn.hidden_sizes + [ffnn.output_size]
        num_layers = len(layer_sizes)
        
        display_sizes = [min(size, max_neurons_per_layer) for size in layer_sizes]
        
        node_positions = {}
        for layer_idx, size in enumerate(layer_sizes):
            display_size = display_sizes[layer_idx]
            for neuron_idx in range(display_size):
                node_id = f"L{layer_idx}N{neuron_idx}"
                G.add_node(node_id)
                node_positions[node_id] = (layer_idx * 2, -neuron_idx * 0.5)
            
            if size > max_neurons_per_layer:
                ellipsis_id = f"L{layer_idx}N..."
                G.add_node(ellipsis_id)
                node_positions[ellipsis_id] = (layer_idx * 2, -(display_size + 1) * 0.5)
        
        for layer_idx in range(num_layers - 1):
            size_curr = display_sizes[layer_idx]
            size_next = display_sizes[layer_idx + 1]
            for i in range(size_curr):
                for j in range(size_next):
                    G.add_edge(f"L{layer_idx}N{i}", f"L{layer_idx + 1}N{j}")
        
        plt.figure(figsize=(12, 6))
        
        nx.draw_networkx(
            G,
            pos=node_positions,
            node_size=500,
            node_color='lightblue',
            font_size=8,
            arrows=False,
            with_labels=True,
            edge_color='gray',
            alpha=0.5,
            width=0.5
        )
        
        # Menambahkan label layer di atas
        for layer_idx, size in enumerate(layer_sizes):
            if layer_idx == 0:
                label = f"Input\n({size} neurons)"
            elif layer_idx == len(layer_sizes) - 1:
                label = f"Output\n({size} neurons)\n{ffnn.output_activation}"
            else:
                label = f"Hidden {layer_idx}\n({size} neurons)\n{ffnn.hidden_activations[layer_idx-1]}"
            plt.text(layer_idx * 2, 1, label, 
                    horizontalalignment='center', 
                    verticalalignment='bottom',
                    fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8, edgecolor='none'))
        
        plt.title("Lightweight Neural Network Visualization (Limited Neurons)")
        plt.axis('off')
        plt.tight_layout()
        
        display(plt.gcf())
        plt.close()
    
    def plot_training_history(self, ffnn):
        if not ffnn.history['train_loss']:
            print("No training history available.")
            return
            
        plt.figure(figsize=(8, 5))
        plt.plot(ffnn.history['train_loss'], label='Training Loss', marker='o', markersize=2)
        plt.plot(ffnn.history['val_loss'], label='Validation Loss', marker='s', markersize=2)
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid()
        plt.show()
    
    def get_layer_weights(self, ffnn, layer_index):
        total_weight_layers = ffnn.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            return [w for row in ffnn.weights[layer_index].data for w in row]
        elif layer_index < total_weight_layers + ffnn.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [b for b in ffnn.biases[bias_idx].data]
        else:
            raise ValueError(f"Layer index must be between 0 and {total_weight_layers + ffnn.num_hidden_layers}")

    def get_layer_gradients(self, ffnn, layer_index):
        total_weight_layers = ffnn.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            return [g for row in ffnn.weights[layer_index].grad for g in row]
        elif layer_index < total_weight_layers + ffnn.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            return [g for g in ffnn.biases[bias_idx].grad]
        else:
            raise ValueError(f"Layer index must be between 0 and {total_weight_layers + ffnn.num_hidden_layers}")

    def get_layer_name(self, ffnn, layer_index):
        total_weight_layers = ffnn.num_hidden_layers + 1
        if layer_index < total_weight_layers:
            if layer_index == 0:
                return "Input-Hidden1 Weights"
            elif layer_index < ffnn.num_hidden_layers:
                return f"Hidden{layer_index}-Hidden{layer_index+1} Weights"
            else:
                return f"Hidden{ffnn.num_hidden_layers}-Output Weights"
        elif layer_index < total_weight_layers + ffnn.num_hidden_layers + 1:
            bias_idx = layer_index - total_weight_layers
            if bias_idx < ffnn.num_hidden_layers:
                return f"Hidden{bias_idx+1} Bias"
            else:
                return "Output Bias"
        else:
            return f"Unknown Layer {layer_index}"

    def plot_weight_distributions(self, ffnn, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = list(range(ffnn.num_hidden_layers + 1 + ffnn.num_hidden_layers + 1))
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("No layers specified for plotting.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for i, layer_idx in enumerate(layers_to_plot):
            weights = self.get_layer_weights(ffnn, layer_idx)
            layer_name = self.get_layer_name(ffnn, layer_idx)
            
            axes[i].hist(weights, bins=30, alpha=0.7)
            axes[i].set_title(f"Weight Distribution - {layer_name}")
            axes[i].set_xlabel("Weights")
            axes[i].set_ylabel("Frequency")
            axes[i].grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()

    def plot_gradient_distributions(self, ffnn, layers_to_plot=None):
        if layers_to_plot is None:
            layers_to_plot = list(range(ffnn.num_hidden_layers + 1 + ffnn.num_hidden_layers + 1))
        
        n_plots = len(layers_to_plot)
        if n_plots == 0:
            print("No layers specified for plotting.")
            return
        
        fig, axes = plt.subplots(1, n_plots, figsize=(5 * n_plots, 5))
        if n_plots == 1:
            axes = [axes]

        for i, layer_idx in enumerate(layers_to_plot):
            gradients = self.get_layer_gradients(ffnn, layer_idx)
            layer_name = self.get_layer_name(ffnn, layer_idx)
            
            if all(g == 0 for g in gradients):
                axes[i].text(0.5, 0.5, "All gradients are 0", 
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