import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Disable eager execution for TensorFlow 2.x compatibility mode
tf.compat.v1.disable_eager_execution()

# Load MNIST dataset using TensorFlow built-in data loader
mnist_path = tf.keras.utils.get_file('mnist.npz', 'https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz')
data = np.load(mnist_path)
x_train, y_train, x_test, y_test = data['x_train'], data['y_train'], data['x_test'], data['y_test']

data.close()

# Normalize pixel values (0-255 to 0-1)
x_train, x_test = x_train / 255.0, x_test / 255.0

# Flatten images (28x28 -> 784)
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1, 784)

# Convert labels to one-hot encoding
y_train = np.eye(10)[y_train]
y_test = np.eye(10)[y_test]

# Visualize some sample images
fig, axes = plt.subplots(1, 5, figsize=(10, 2))
for i in range(5):
    axes[i].imshow(x_train[i].reshape(28, 28), cmap='gray')
    axes[i].axis('off')
plt.show()

# Define network parameters
n_input = 784   # 28x28 pixels (flattened image)
n_hidden1 = 128 # First hidden layer with 128 neurons
n_hidden2 = 64  # Second hidden layer with 64 neurons
n_output = 10   # Output layer with 10 classes (digits 0-9)
learning_rate = 0.01
epochs = 10
batch_size = 100

# Define placeholders for input features (X) and labels (Y)
X = tf.compat.v1.placeholder(tf.float32, [None, n_input])
Y = tf.compat.v1.placeholder(tf.float32, [None, n_output])

# Initialize weights and biases for each layer
weights = {
    'h1': tf.Variable(tf.random.normal([n_input, n_hidden1], stddev=0.1)),
    'h2': tf.Variable(tf.random.normal([n_hidden1, n_hidden2], stddev=0.1)),
    'out': tf.Variable(tf.random.normal([n_hidden2, n_output], stddev=0.1))
}
biases = {
    'b1': tf.Variable(tf.zeros([n_hidden1])),
    'b2': tf.Variable(tf.zeros([n_hidden2])),
    'out': tf.Variable(tf.zeros([n_output]))
}

# Define the neural network model
def neural_net(x):
    # First hidden layer with ReLU activation
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, weights['h1']), biases['b1']))
    
    # Second hidden layer with ReLU activation
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
    
    # Output layer (logits)
    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return out_layer

# Define forward propagation
logits = neural_net(X)  # Compute logits
prediction = tf.nn.softmax(logits)  # Apply softmax to get class probabilities

# Define loss function (cross-entropy for classification)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))

# Define backpropagation optimizer
optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Define accuracy computation
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # Compare predicted and actual labels
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the model
with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())  # Initialize all variables
    
    loss_values = []  # Store loss values for visualization
    acc_values = []  # Store accuracy values for visualization
    
    for epoch in range(epochs):
        avg_loss = 0
        total_batch = int(x_train.shape[0] / batch_size)
        
        for i in range(total_batch):
            batch_x = x_train[i * batch_size: (i + 1) * batch_size]
            batch_y = y_train[i * batch_size: (i + 1) * batch_size]
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += batch_loss / total_batch  # Compute average loss per epoch
        
        # Compute training and test accuracy
        train_acc = sess.run(accuracy, feed_dict={X: x_train, Y: y_train})
        test_acc = sess.run(accuracy, feed_dict={X: x_test, Y: y_test})
        
        loss_values.append(avg_loss)
        acc_values.append(test_acc)
        
        # Print epoch-wise results
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    print("Training complete!")
    
    # Plot loss and accuracy trends
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), loss_values, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Trend')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), acc_values, label='Test Accuracy', color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Trend')
    plt.legend()
    
    plt.show()