from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from FFNN import FFNN

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
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_iris()
    
    ffnn = FFNN(
        input_size=4,
        hidden_sizes=[10, 5],
        output_size=3,
        learning_rate=0.1,
        hidden_activations=['relu', 'relu'],
        output_activation='sigmoid', 
        loss_function='mse',                 
    )
    ffnn.train(X_train, y_train, X_val, y_val, epochs=20, batch_size=1, verbose=1)
    
    correct = 0
    for inputs, target in zip(X_test, y_test):
        prediction = ffnn.predict(inputs)
        predicted_class = prediction.index(max(prediction))
        actual_class = target.index(max(target))
        if predicted_class == actual_class:
            correct += 1
    
    # print(f"Akurasi: {correct / len(y_test) * 100:.2f}%")

    ffnn.compare_lib(X_train, y_train, X_test, y_test)
    ffnn.plot_training_history()
    ffnn.plot_weight_distributions()
    ffnn.plot_gradient_distributions()
