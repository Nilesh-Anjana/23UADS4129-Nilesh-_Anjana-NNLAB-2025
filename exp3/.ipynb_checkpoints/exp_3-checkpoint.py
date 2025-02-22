import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Load MNIST dataset
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Define network parameters
n_input = 784   # 28x28 pixels (flattened image)
n_hidden1 = 128 # First hidden layer with 128 neurons
n_hidden2 = 64  # Second hidden layer with 64 neurons
n_output = 10   # Output layer with 10 classes (digits 0-9)
learning_rate = 0.01
epochs = 10
batch_size = 100

# Define placeholders for input features (X) and labels (Y)
X = tf.placeholder(tf.float32, [None, n_input])
Y = tf.placeholder(tf.float32, [None, n_output])

# Initialize weights and biases for each layer
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden1])),
    'h2': tf.Variable(tf.random_normal([n_hidden1, n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_hidden2, n_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden1])),
    'b2': tf.Variable(tf.random_normal([n_hidden2])),
    'out': tf.Variable(tf.random_normal([n_output]))
}

# Define the neural network model
def neural_net(x):
    # First hidden layer with ReLU activation
    layer1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer1 = tf.nn.relu(layer1)
    
    # Second hidden layer with ReLU activation
    layer2 = tf.add(tf.matmul(layer1, weights['h2']), biases['b2'])
    layer2 = tf.nn.relu(layer2)
    
    # Output layer (logits)
    out_layer = tf.add(tf.matmul(layer2, weights['out']), biases['out'])
    return out_layer

# Define forward propagation
logits = neural_net(X)  # Compute logits
prediction = tf.nn.softmax(logits)  # Apply softmax to get class probabilities

# Define loss function (cross-entropy for classification)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=Y))

# Define backpropagation optimizer
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

# Define accuracy computation
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))  # Compare predicted and actual labels
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())  # Initialize all variables
    
    for epoch in range(epochs):
        avg_loss = 0
        total_batch = int(mnist.train.num_examples / batch_size)
        
        for i in range(total_batch):
            batch_x, batch_y = mnist.train.next_batch(batch_size)  # Fetch mini-batch
            _, batch_loss = sess.run([optimizer, loss], feed_dict={X: batch_x, Y: batch_y})
            avg_loss += batch_loss / total_batch  # Compute average loss per epoch
        
        # Compute training and test accuracy
        train_acc = sess.run(accuracy, feed_dict={X: mnist.train.images, Y: mnist.train.labels})
        test_acc = sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels})
        
        # Print epoch-wise results
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Train Accuracy: {train_acc:.4f}, Test Accuracy: {test_acc:.4f}")
    
    print("Training complete!")
