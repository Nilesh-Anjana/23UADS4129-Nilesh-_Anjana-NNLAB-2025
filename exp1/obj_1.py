import numpy as np

class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=1000):
        self.learning_rate = learning_rate  # Step size for weight updates
        self.epochs = epochs  # Number of times the model will iterate over the dataset
        self.weights = None  # To store weights
        self.bias = None  # To store bias
    
    def activation_function(self, x):
        return 1 if x >= 0 else 0  # Step function for binary classification
    
    def fit(self, X, y):
        num_features = X.shape[1]  # Number of input features
        self.weights = np.zeros(num_features)  # Initialize weights as zeros
        self.bias = 0  # Initialize bias as zero
        
        for _ in range(self.epochs):  # Iterate over dataset multiple times
            for i in range(len(X)):
                linear_output = np.dot(X[i], self.weights) + self.bias  # Weighted sum
                y_predicted = self.activation_function(linear_output)  # Apply step function
                
                update = self.learning_rate * (y[i] - y_predicted)  # Calculate update amount
                self.weights += update * X[i]  # Update weights
                self.bias += update  # Update bias
    
    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias  # Weighted sum
        return np.array([self.activation_function(x) for x in linear_output])  # Apply step function

# Example usage:
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Input data (AND gate example)
y = np.array([0, 0, 0, 1])  # Labels for AND gate

perceptron = Perceptron(learning_rate=0.1, epochs=10)
perceptron.fit(X, y)

predictions = perceptron.predict(X)
print("Predictions:", predictions)
