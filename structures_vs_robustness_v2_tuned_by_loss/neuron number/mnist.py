# In structures_vs_robustness, we illustrated that an increase 
# in epoch number results in a higher resistance to adversarial attacks.
# We are curious if increasing the number of neurons per layer - leading to overfitting potentially
# may also have the same effect.

# Basic imports for libraries and the mnist data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

tf.set_random_seed(0)

# Download images and labels into mnist.test (10K images+labels) and mnist.train (60K images+l/Users/summer/Documents/Research/adversarial_neural_nets/mnist.pyabels)
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)
input_dimension = 784
output_dimension = 10
learning_rate = 0.001
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
    with tf.variable_scope("layer1_" + str(l1_units)):
        w1 = tf.get_variable("w",[input_dimension, l1_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b1 = tf.get_variable("b", [l1_units], initializer = tf.zeros_initializer())
        z1 = tf.matmul(x, w1) + b1
        y1 = tf.nn.relu(z1)
    
    # Hidden 2
    with tf.variable_scope("layer2_" + str(l1_units)):
        w2 = tf.get_variable("w",[l1_units, l2_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b2 = tf.get_variable("b", [l2_units], initializer = tf.zeros_initializer())
        z2 = tf.matmul(y1, w2) + b2
        y2 = tf.nn.relu(z2)
   
    # softmax
    with tf.variable_scope("layer3_" + str(l1_units)):
        w3 = tf.get_variable("w",[l2_units, l3_units], initializer = tf.contrib.layers.xavier_initializer(seed = 1))
        b3 = tf.get_variable("b", [l3_units], initializer = tf.zeros_initializer())
        z3 = tf.matmul(y2, w3) + b3
        y_ = tf.nn.softmax(z3)
  
    params = {}
    params["w1"] = w1
    params["w2"] = w2
    params["w3"] = w3
    params["b1"] = b1
    params["b2"] = b2
    params["b3"] = b3

    return z3, y_, params # logits and probilities

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

def create_mnist_model(neuron_num_start, neuron_num_end, jump_num):
    """ Construct and train the neural net and save the model in /tmp/mnist_model_neurons-8
    """

    # Train MNIST classifer and evaluate the accuracy
    with tf.Session() as sess:
        # mini-batch training 
        for neuron_number in range(neuron_num_start, neuron_num_end, jump_num):

            # Building the graph
            l1 = neuron_number
            l2 = neuron_number
            l3 = 10
            x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
            y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
            z3, y_, nn_params = build_network(x, l1, l2, l3)
            loss = calc_loss(z3, y)
            accuracy = evaluation(y, y_)
            train_op = training(loss, learning_rate)
            saver = tf.train.Saver(max_to_keep=100)
            epoch_num = 20

            sess.run(tf.global_variables_initializer())
            for epoch in range(epoch_num):
                shuffled_train_images, shuffled_train_labels = generate_shuffled_train_data()
                for (batch_train_images, batch_train_labels) in\
                    generate_mini_batches(batch_size, shuffled_train_images, shuffled_train_labels):
                        sess.run(train_op, feed_dict={x: batch_train_images, y: batch_train_labels}) 
            loss_on_valid_set = sess.run(loss, feed_dict={x: mnist.validation.images, y: mnist.validation.labels})
            print("For neuron number per layer {0}, the total loss is {1}".format(neuron_number, loss_on_valid_set))
            # Save the variables and model to disk.
            save_path = saver.save(sess, './tmp/mnist_model_neurons', global_step=neuron_number)
            print("Model saved in file: %s" % save_path)
            
def calculate_test_accuracy(neuron_num_start, neuron_num_end, jump_num):
    """ Restore the saved model and calculate its test accuracy
    """
    tf.reset_default_graph()

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())

    for neuron_num in range(neuron_num_start, neuron_num_end, jump_num):
        # Building the graph
        x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
        y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
        z3, y_, nn_params = build_network(x, neuron_num, neuron_num, l3)
        loss = calc_loss(z3, y)
        accuracy = evaluation(y, y_)
        train_op = training(loss, learning_rate)

        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, "./tmp/mnist_model_neurons-" + str(neuron_num))
        # print("Model restored.")

        # Check the values of the variables
        print("For number of neurons per layer:" + str(neuron_num))
        train_accuracy = sess.run(accuracy, feed_dict={x: mnist.train.images, y: mnist.train.labels})
        print("Train accuracy is:" + str(train_accuracy))
        test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Test accuracy is:" + str(test_accuracy))

def demonstrate_valid_train_loss(neuron_num_start, neuron_num_end, jump_num):
    """Demonstrate the change in validation loss and train loss as more epochs are trained
    """
    neuron_nums = []
    valid_losses = []
    train_losses = []

    for neuron_num in range(neuron_num_start, neuron_num_end, jump_num):

        # Building the graph
        x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
        y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
        z3, y_, _ = build_network(x,neuron_num, neuron_num, l3)
        loss = calc_loss(z3, y)

        sess=tf.Session()   
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        print("trying to restore:" +  "./tmp/mnist_model_neurons-" + str(neuron_num))
        saver.restore(sess, "./tmp/mnist_model_neurons-" + str(neuron_num))

        train_loss = sess.run(loss, feed_dict={x: mnist.train.images, y:mnist.train.labels})
        valid_loss = sess.run(loss, feed_dict={x: mnist.validation.images, y:mnist.validation.labels})

        neuron_nums.append(neuron_num)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        print("Neuron num: {0}, train_loss: {1}, validation_loss: {2}".format(neuron_num, train_loss, valid_loss))

    plt.plot(neuron_nums, train_losses)
    plt.plot(neuron_nums, valid_losses)
    plt.title('Losses as the neural net is trained with more neurons per layer')
    plt.legend(['Train Loss', 'Validation Loss'], loc='upper left')
    plt.xlabel('Neuron Number')
    plt.ylabel('Loss')
    plt.show()

def get_average_weights_from_neuron_num(neuron_num, params, layer_num):
    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./tmp/mnist_model_neurons-" + str(neuron_num))
    params_nn = sess.run(params, feed_dict={})

    #Return the spectral norm for one of the weight matrices
    return params_nn["w"+str(layer_num)].max()

def demonstrate_neuron_number_vs_weights(neuron_num_start, neuron_num_end, jump_num, layer_num):
    """ Demonstrate the neural net's average weights vs epoch number
    Args:
        layer_num: the layer we want to show analysis for
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    neuron_nums = []
    weights_average = []
    for neuron_num in range(neuron_num_start, neuron_num_end, jump_num):

        # Building the graph
        x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
        y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
        z3, y_, params_nn = build_network(x,neuron_num, neuron_num, l3)
        loss = calc_loss(z3, y)

        neuron_nums.append(neuron_num)
        weights_for_neuron_num = get_average_weights_from_neuron_num(neuron_num, params_nn, layer_num)
        weights_average.append(weights_for_neuron_num)

        print("Neuron num: {0}, Weights on average for neuron number: {1}".format(neuron_num, weights_for_neuron_num))
    plt.plot(neuron_nums, weights_average)
    plt.title('MNIST Classifier Weights VS Neuron Numbers')
    plt.xlabel('Neuron Number')
    plt.ylabel('Average Weights')
    plt.show()

#create_mnist_model(50, 550, 50)
#demonstrate_valid_train_loss(50, 500, 50)
#calculate_test_accuracy(50, 500, 50)
demonstrate_neuron_number_vs_weights(50, 500, 50, 1)
