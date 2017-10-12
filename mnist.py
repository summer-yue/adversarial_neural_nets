# coding: utf-8

# In[1]:

# Basic imports for libraries and the mnist data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+labels)
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
input_dimension = 784
output_dimension = 10
learning_rate = 0.001
batch_size = 100
l1 = 200
l2 = 300
l3 = 10

def visualize_original_data():
    # Visualize the original data
    index = 1
    print(mnist.train.images.shape)
    print(mnist.train.labels.shape)

    plt.imshow(mnist.train.images[index, :].reshape(28, 28))
    plt.show()
    print ("y = " + str(mnist.train.labels[index, :]))

def build_network(x, l1_units, l2_units, l3_units):
    """Build the MNIST model with 2 hidden layers and one linear layer.
    params: 
        x: input placeholder
    Returns: 
        Output tensor with the computed logits
    """
    
    # Hidden 1
    with tf.variable_scope("layer1"):
        w1 = tf.get_variable("w",[input_dimension, l1_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b", [l1_units], initializer = tf.zeros_initializer())
        z1 = tf.matmul(x, w1) + b1
        y1 = tf.nn.relu(z1)
    
    # Hidden 2
    with tf.variable_scope("layer2"):
        w2 = tf.get_variable("w",[l1_units, l2_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b", [l2_units], initializer = tf.zeros_initializer())
        z2 = tf.matmul(y1, w2) + b2
        y2 = tf.nn.relu(z2)
   
    # softmax
    with tf.variable_scope("layer3"):
        w3 = tf.get_variable("w",[l2_units, l3_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b", [l3_units], initializer = tf.zeros_initializer())
        z3 = tf.matmul(y2, w3) + b3
        y_ = tf.nn.softmax(z3)
  
    return z3, y_ # logits and probilities

def calc_loss(z3, y):
    """Calculates the loss from the logits and the labels.
    params:
        z3: logits for output tensor from build_network
        y: labeled data indicating the numeric value of the picture, [None, 10]
    """
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=z3, labels=y)
    cross_entropy = tf.reduce_mean(cross_entropy)
    return cross_entropy

def training(loss, learning_rate):
    """Sets up the training Ops.
    params:
        loss: cross entropy loss
        learning_rate
    """
    beta1=0.9
    beta2=0.999
    epsilon=1e-08
 
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon)
    train_op = optimizer.minimize(loss)
    return train_op

def evaluation(y, y_):
    """Evaluate the quality of the logits at predicting the label.
    Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
    return accuracy

# Preparing for mini-batching training data
def generate_shuffled_train_data():
    """ Shuffle the images and labels in the training set in unison
    Return:
        shuffled train images and train labels
    """
    assert len(mnist.train.images) == len(mnist.train.labels)
    p = np.random.permutation(len(mnist.train.labels))
    shuffled_train_images = mnist.train.images[p]
    shuffled_train_labels = mnist.train.labels[p]
    return shuffled_train_images, shuffled_train_labels


def generate_mini_batches(batch_size, train_images, train_labels):
    """ Yield mini batches in tuples from the original dataset with a specified batch size
    Params: 
        batch_size: number of training data in a sample
        train_images: all of train images [None, 784] after shuffling
        train_labels: all of train labels [None, 10] after shuffling
    Return:
        A generator yielding each mini batch([batch_num, 784], [batch_num, 10])
    Notes:
        the last data not divisible by mini-batch is thrown away
    """
    train_image_num = len(train_labels)
    for i in range(int(train_image_num / batch_size)):
        start_slice_index = i * batch_size
        end_slice_index = (i + 1) * batch_size
        yield (train_images[start_slice_index:end_slice_index],
               train_labels[start_slice_index:end_slice_index])



def search_hyperparameters():
    # Grid search for hyperparameters
    # Optimal batch size is:100 
    # Optimal number of neurons for layer 1 is:200 (25, 50, 100, 200, 300)
    # Optimal number of neurons for layer 2 is:300 (25, 50, 100, 200, 300)
    # Optimal number of neurons for layer 3 is:10
    # Optimal epoch number is:8 (1 - 30)
    # Best validation accuracy:0.9858
    # Optimal train accuracy:0.99

    batch_size_options = [100] #  is the optimal batch size after hyperparameter tuning
    learning_rate_options = [0.001] #  is the optimal learnig rate after hyperparameter tuning
    l_options = [100, 200, 300]

    optimal_batch_size = batch_size_options[0]
    optimal_learning_rate = learning_rate_options[0]
    optimal_l1 = l_options[0]
    optimal_l2 = l_options[0]
    optimal_l3 = l_options[0]
    optimal_epoch_number = 1

    max_validation_accuracy = 0
    optimal_train_accuracy = 0
        
    for learning_rate in learning_rate_options:
        for l1 in l_options: #Optimal l1 = 200
            for l2 in l_options: # Optimal l2 = 300
                l3 = 10

                # Resetting the default graph before building a new graph
                tf.reset_default_graph()
                # Train MNIST classifer and evaluate the accuracy
                with tf.Session() as sess:

                    # Building the graph
                    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
                    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
                    z3, y_ = build_network(x, l1, l2, l3)
                    loss = calc_loss(z3, y)
                    accuracy = evaluation(y, y_)
                    train_op = training(loss, learning_rate)
                    sess.run(tf.global_variables_initializer())

                    for batch_size in batch_size_options:
                        # Training the classifier with 10 epochs with shuffled training data in mini batches
                        for epoch in range(30): # This is the optimal number of epochs after tuning
                            shuffled_train_images, shuffled_train_labels = generate_shuffled_train_data()
                            for (batch_train_images, batch_train_labels) in\
                                generate_mini_batches(batch_size, shuffled_train_images, shuffled_train_labels):
                                    sess.run(train_op, feed_dict={x: batch_train_images, y: batch_train_labels}) 
                                    train_accuracy = sess.run(accuracy, feed_dict={x: batch_train_images, y: batch_train_labels})
                                    
                            # Update the optimal hyperparameters if the validation accuracy is higher than the old maximum validation accuracy
                            validation_accuracy = sess.run(accuracy, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})  
                            print(str(epoch) + "th epoch, batch size: " + str(batch_size) + ", learning rate=" + str(learning_rate)\
                                + ",l1=" + str(l1) +  ",l2=" + str(l2) + ",l3=" + str(l3))
                            print("Training accuracy:" + str(train_accuracy) + " Validation_accuracy:" + str(validation_accuracy))
                            if (validation_accuracy > max_validation_accuracy):
                                optimal_batch_size = batch_size    
                                optimal_learning_rate = learning_rate
                                optimal_l1 = l1
                                optimal_l2 = l2
                                optimal_l3 = l3
                                optimal_epoch_number = epoch
                                max_validation_accuracy = validation_accuracy
                                optimal_train_accuracy = train_accuracy
                    

        print("Optimal learning rate is:" + str(optimal_learning_rate))
        print("Optimal batch size is:" + str(optimal_batch_size))
        print("Optimal number of neurons for layer 1 is:" + str(optimal_l1))
        print("Optimal number of neurons for layer 2 is:" + str(optimal_l2))
        print("Optimal number of neurons for layer 3 is:" + str(optimal_l3))
        print("Optimal epoch number is:" + str(optimal_epoch_number))
        print("Best validation accuracy:" + str(max_validation_accuracy))
        print("Optimal train accuracy:" + str(optimal_train_accuracy))


def create_mnist_model():
    """ Construct and train the neural net and save the model in /tmp/original_mnist_model-8
    """
    # Building the graph
    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
    z3, y_ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)
    train_op = training(loss, learning_rate)
    saver = tf.train.Saver()
    # Train MNIST classifer and evaluate the accuracy
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # mini-batch training for 8 epochs
        for epoch in range(8): # This is the optimal number of epochs after tuning
            shuffled_train_images, shuffled_train_labels = generate_shuffled_train_data()
            for (batch_train_images, batch_train_labels) in\
                generate_mini_batches(batch_size, shuffled_train_images, shuffled_train_labels):
                    sess.run(train_op, feed_dict={x: batch_train_images, y: batch_train_labels}) 

        # Save the variables and model to disk.
        save_path = saver.save(sess, './tmp/original_mnist_model', global_step=8)
        print("Model saved in file: %s" % save_path)
        
def calculate_test_accuracy():
    """ Restore the saved model and calculate its test accuracy
    """
    tf.reset_default_graph()

    # Building the graph
    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
    z3, y_ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)
    train_op = training(loss, learning_rate)

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver()
    saver.restore(sess, "./tmp/original_mnist_model-8")
    print("Model restored.")

    # Check the values of the variables
    test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
    print("Test accuracy is:" + str(test_accuracy))

create_mnist_model()
calculate_test_accuracy()
