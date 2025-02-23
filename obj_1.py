import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100):
        """
        Initialize the perceptron with random weights and bias.
        :param input_size: Number of input features
        :param learning_rate: Learning rate for weight updates
        :param epochs: Number of training iterations
        """
        self.weights = np.random.randn(input_size)
        self.bias = np.random.randn()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def activation(self, x):
        """Step activation function (threshold at 0)"""
        return 1 if x >= 0 else 0
    
    def predict(self, inputs):
        """Compute weighted sum and apply activation function."""
        linear_output = np.dot(inputs, self.weights) + self.bias
        return self.activation(linear_output)
    
    def train(self, X, y):
        """
        Train the perceptron using the given dataset.
        :param X: Input features (numpy array)
        :param y: Target labels (numpy array)
        """
        for epoch in range(self.epochs):
            total_error = 0
            for inputs, target in zip(X, y):
                prediction = self.predict(inputs)
                error = target - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error
                total_error += abs(error)
            
            # Stop early if no errors
            if total_error == 0:
                break
    
    def evaluate(self, X, y):
        """Evaluate the perceptron's accuracy on a dataset."""
        correct_predictions = sum(self.predict(x) == target for x, target in zip(X, y))
        accuracy = correct_predictions / len(y)
        return accuracy

# Define NAND and XOR truth tables
nand_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
nand_outputs = np.array([1, 1, 1, 0])  # NAND logic

xor_inputs = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
xor_outputs = np.array([0, 1, 1, 0])  # XOR logic (not linearly separable)

# Train perceptron for NAND
nand_perceptron = Perceptron(input_size=2)
nand_perceptron.train(nand_inputs, nand_outputs)
nand_accuracy = nand_perceptron.evaluate(nand_inputs, nand_outputs)
print(f"NAND Perceptron Accuracy: {nand_accuracy * 100:.2f}%")

# Train perceptron for XOR (expected to fail)
xor_perceptron = Perceptron(input_size=2)
xor_perceptron.train(xor_inputs, xor_outputs)
xor_accuracy = xor_perceptron.evaluate(xor_inputs, xor_outputs)
print(f"XOR Perceptron Accuracy: {xor_accuracy * 100:.2f}% (Expected to be low)")
