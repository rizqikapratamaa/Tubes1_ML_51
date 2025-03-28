from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from IPython.display import display
from FFNN import FFNN

def visualize_network_light(ffnn):
    # Membuat graf directed
    G = nx.DiGraph()
    
    # Menentukan ukuran layer
    layer_sizes = [ffnn.input_size] + ffnn.hidden_sizes + [ffnn.output_size]
    num_layers = len(layer_sizes)
    
    # Menambahkan node untuk setiap neuron
    node_positions = {}
    for layer_idx, size in enumerate(layer_sizes):
        for neuron_idx in range(size):
            node_id = f"L{layer_idx}N{neuron_idx}"
            G.add_node(node_id)
            # Posisi sederhana: layer_idx sebagai x, neuron_idx sebagai y
            node_positions[node_id] = (layer_idx, -neuron_idx)
    
    # Menambahkan edge antar layer (hanya koneksi tanpa bobot untuk efisiensi)
    for layer_idx in range(num_layers - 1):
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx + 1]):
                G.add_edge(f"L{layer_idx}N{i}", f"L{layer_idx + 1}N{j}")
    
    # Setup visualisasi
    plt.figure(figsize=(10, 6))
    
    # Menggambar graf
    nx.draw_networkx(
        G,
        pos=node_positions,
        node_size=300,
        node_color='skyblue',
        font_size=8,
        arrows=False,  # Menghilangkan panah untuk performa
        with_labels=True,
        edge_color='gray',
        alpha=0.6
    )
    
    # Menambahkan label layer
    for layer_idx, size in enumerate(layer_sizes):
        if layer_idx == 0:
            label = "Input"
        elif layer_idx == len(layer_sizes) - 1:
            label = f"Output\n{ffnn.output_activation}"
        else:
            label = f"Hidden {layer_idx}\n{ffnn.hidden_activations[layer_idx-1]}"
        plt.text(layer_idx, 1, label, 
                horizontalalignment='center', 
                verticalalignment='bottom',
                fontsize=10)
    
    plt.title("Lightweight Neural Network Visualization")
    plt.axis('off')
    plt.tight_layout()
    
    # Tampilkan di Jupyter Notebook
    display(plt.gcf())
    plt.close()

def preprocess_mnist(num_samples=20000):
    mnist = fetch_openml('mnist_784', version=1, as_frame=False)
    X = mnist.data
    y = mnist.target.astype(np.int32)
    
    np.random.seed(42)
    indices = np.random.choice(len(X), num_samples, replace=False)
    X = X[indices]
    y = y[indices]
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y_one_hot = np.zeros((len(y), 10))
    y_one_hot[np.arange(len(y)), y] = 1
    
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def preprocess_iris():
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target
    
    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)
    
    y_one_hot = [[1 if i == label else 0 for i in range(3)] for label in y]
    
    # 70% training - 15% validation - 15% testing
    X_train, X_temp, y_train, y_temp = train_test_split(X, y_one_hot, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    return X_train, X_val, X_test, y_train, y_val, y_test

if __name__ == "__main__":
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_mnist(num_samples=10000)
    
    ffnn = FFNN(
        input_size=784,
        hidden_sizes=[256, 128],
        output_size=10,
        learning_rate=0.001,
        hidden_activations=['relu', 'relu'],
        output_activation='softmax',
        loss_function='cce',
        reg_type='l2',
        reg_lambda=0.001,
        rms_norm=True,
        init_type='he'
    )
    ffnn.train(X_train, y_train, X_val, y_val, epochs=30, batch_size=32, verbose=1)
    
    correct = 0
    for inputs, target in zip(X_test, y_test):
        prediction = ffnn.predict(inputs)
        predicted_class = np.argmax(prediction)
        actual_class = np.argmax(target)
        if predicted_class == actual_class:
            correct += 1
    
    print(f"Akurasi: {correct / len(y_test) * 100:.2f}%")

    ffnn.plot_training_history()
    ffnn.plot_weight_distributions()
    ffnn.plot_gradient_distributions()
    visualize_network_light(ffnn)