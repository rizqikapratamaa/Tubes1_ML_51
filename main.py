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
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess_iris()
    
    ffnn = FFNN(input_size=4, hidden_size=5, output_size=3, learning_rate=0.5, hidden_activation='relu', output_activation='sigmoid')
    ffnn.train(X_train, y_train, epochs=20)
    
    correct = 0
    for inputs, target in zip(X_test, y_test):
        prediction = ffnn.predict(inputs)
        predicted_class = prediction.index(max(prediction))
        actual_class = target.index(max(target))
        if predicted_class == actual_class:
            correct += 1
    
    print(f"Akurasi: {correct / len(y_test) * 100:.2f}%")
