# This is a basic 3-layer classifier we built for the MNIST dataset.
# The performance of this classifier is used as a base line reference.

# Learning rate = 0.001, Epoch num: 4, train_loss: 0.027191724635529536, validation_loss: 0.06550877619758981
# Train accuracy is:0.991618181818 #Test accuracy is:0.9794
# Basic imports for libraries and the mnist data
# models are stored in models/original/original_epoch-15

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+l/Users/summer/Documents/Research/adversarial_neural_nets/mnist.pyabels)
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
input_dimension = 784
output_dimension = 10
batch_size = 100
l1 = 200
l2 = 300
l3 = 10

def build_network(x, l1_units, l2_units, l3_units):
    """Build the MNIST model with 2 hidden layers and one linear layer.
    params: 
        x: input placeholder
    Returns: 
        Output tensor with the computed logits
    """
    
    # Hidden 1
    with tf.variable_scope("layer1"):
        w1 = tf.get_variable("w",[input_dimension, l1_units], dtype = tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b", [l1_units], dtype = tf.float64, initializer = tf.zeros_initializer())
        z1 = tf.matmul(x, w1) + b1
        y1 = tf.nn.relu(z1)
    
    # Hidden 2
    with tf.variable_scope("layer2"):
        w2 = tf.get_variable("w",[l1_units, l2_units], dtype = tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b", [l2_units], dtype = tf.float64, initializer = tf.zeros_initializer())
        z2 = tf.matmul(y1, w2) + b2
        y2 = tf.nn.relu(z2)
   
    # softmax
    with tf.variable_scope("layer3"):
        w3 = tf.get_variable("w",[l2_units, l3_units], dtype = tf.float64, initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b", [l3_units], dtype = tf.float64, initializer = tf.zeros_initializer())
        z3 = tf.matmul(y2, w3) + b3
        y_ = tf.nn.softmax(z3)
  
    params = {}
    params["w1"] = w1
    params["w2"] = w2
    params["w3"] = w3
    params["b1"] = b1
    params["b2"] = b2
    params["b3"] = b3

    return z3, y_, params # logits, probilities and weights

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
 
    optimizer = tf.train.AdamOptimizer(learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08,)
    train_op = optimizer.minimize(loss)
    return train_op

def evaluation(y, y_):
    """Evaluate the quality of the logits at predicting the label.
    Returns:
    A scalar int64 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
    """
    correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float64), name="accuracy")
    return accuracy

# Preparing for mini-batching training data
def generate_shuffled_train_data():
    """ Shuffle the images and labels in the training set in unison
    Return:
        shuffled train images and train labels
    """
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

def create_mnist_model(epoch_num_start, epoch_num_end, stride, round_num):
    """ Construct and train the neural net and save the models in /models/overfitted-(round_num)/original_epoch-(epoch_number)
    Args:
        epoch_num_start: start recording models from this epoch
        epoch_num_end: stop training at this epoch number
        stride: number of epochs between two adjacentmodels
        round: number of experiments used to average results
    """
    # Building the graph
    x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
    z3, y_, _ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)
    learning_rate = tf.placeholder(tf.float64, shape=[])
    train_op = training(loss, learning_rate)
    saver = tf.train.Saver(max_to_keep=500)

    # Train MNIST classifer and evaluate the accuracy
    with tf.Session() as sess:
        # mini-batch training
        for r in range(round_num):
            sess.run(tf.global_variables_initializer())
            print("on round number:" + str(r))
            for epoch_num in range(epoch_num_start, epoch_num_end):
                shuffled_train_images, shuffled_train_labels = generate_shuffled_train_data()

                for (batch_train_images, batch_train_labels) in\
                    generate_mini_batches(batch_size, shuffled_train_images, shuffled_train_labels):
                        sess.run(train_op,
                                 feed_dict={x: batch_train_images, y: batch_train_labels, learning_rate: 0.001}) 
                print("on epoch number:" + str(epoch_num))

                #Save and report the model every "stride" number of epochs
                if (epoch_num - epoch_num_start) % stride == 0:
                    loss_on_valid_set = sess.run(loss, feed_dict={x: mnist.validation.images, y: mnist.validation.labels, learning_rate: 0.001})
                    loss_on_train_set = sess.run(loss, feed_dict={x: mnist.train.images, y: mnist.train.labels, learning_rate: 0.001})
                    print("For epoch {0}, the validation loss is {1}".format(epoch_num, loss_on_valid_set))
                    # Save the variables and model to disk.
                    save_path = saver.save(sess, './models/overfitted-' + str(r) + '/original_epoch', global_step=epoch_num)
                    print("Model saved in file: %s" % save_path)
                
def calculate_test_accuracy(epoch_num_start, epoch_num_end, stride, round_num):
    """ Restore the saved models and calculate their test accuracy
    Args:
        epoch_num_start: start recording models from this epoch
        epoch_num_end: stop training at this epoch number
        stride: number of epochs between two adjacentmodels
        round: number of experiments used to average results
    """
    tf.reset_default_graph()
    # Building the graph
    x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
    z3, y_, _ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)
    learning_rate = tf.placeholder(tf.float64, shape=[])
    train_op = training(loss, learning_rate)

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())

    saver = tf.train.Saver(max_to_keep=500)

    for epoch_num in range(epoch_num_start, epoch_num_end, stride):
        saver.restore(sess, "./models/overfitted-" + str(round_num) + "/original_epoch-" + str(epoch_num))
        print("Model " + str(round_num) + "." + str(epoch_num) + " restored.")

        # Check the values of the variables
        print("For epoch:" + str(epoch_num))
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Train accuracy is:" + str(train_accuracy))
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy is:" + str(test_accuracy))

# This function was used for epoch number tuning
def demonstrate_valid_train_loss(epoch_num_start, epoch_num_end, stride, round_num):
    """Demonstrate the change in validation loss and train loss as more epochs are trained
    Args:
        epoch_num_start: start recording models from this epoch
        epcoh_num_end: stop training at this epoch number
        stride: number of epochs between two adjacentmodels
        round: number of experiments used to average results
    """
    # Building the graph
    x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
    z3, y_, _ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)

    epoch_nums = []
    valid_losses = []
    train_losses = []

    for epoch_num in range(epoch_num_start, epoch_num_end, stride):

        sess=tf.Session()   
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=500)
        saver.restore(sess, "./models/overfitted-" + str(round_num) + "/original_epoch-" + str(epoch_num))

        train_loss = sess.run(loss, feed_dict={x: mnist.train.images, y:mnist.train.labels})
        valid_loss = sess.run(loss, feed_dict={x: mnist.validation.images, y:mnist.validation.labels})

        epoch_nums.append(epoch_num)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print("Epoch num: {0}, train_loss: {1}, validation_loss: {2}".format(epoch_num, train_loss, valid_loss))

    plt.plot(epoch_nums, train_losses)
    plt.plot(epoch_nums, valid_losses)
    plt.title('Losses as the neural net is trained with more epochs')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Loss')
    plt.show()

#create_mnist_model(1, 200, 20, 5)
#demonstrate_valid_train_loss(1, 200, 20, 0) # See results saved in averaged_results/train_valid_loss
calculate_test_accuracy(1, 200, 20, 4)