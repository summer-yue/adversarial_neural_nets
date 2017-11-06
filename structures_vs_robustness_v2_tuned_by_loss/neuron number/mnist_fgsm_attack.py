import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import build_network, calc_loss, evaluation
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Importing MNIST datasets
mnist = mnist_data.read_data_sets('MNIST_data', one_hot=True)

# Tuned hyperparameters from mnist.py
input_dimension = 784
output_dimension = 10
learning_rate = 0.001
batch_size = 100
l1 = 200
l2 = 300
l3 = 10

tf.reset_default_graph()

def accuracy_after_fgsm_attack(images, labels, neuron_num, loss, x, y, accuracy):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """
    #apply pertubation to images
    epsilon = 0.1
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./tmp/mnist_model_neurons-" + str(neuron_num))

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    return perturbed_accuracy

def demonstrate_attack_error_rate(neuron_num_start, neuron_num_end, jump_num):
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    neuron_nums = []
    perturbed_accuracies = []
    for neuron_num in range(neuron_num_start, neuron_num_end, jump_num):

        # Building the graph
        x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
        y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
        z3, y_, _ = build_network(x, neuron_num, neuron_num, l3)
        loss = calc_loss(z3, y)
        accuracy = evaluation(y, y_)

        neuron_nums.append(neuron_num)
        perturbed_accuracy = accuracy_after_fgsm_attack(sample_images, sample_labels, neuron_num, loss, x, y, accuracy)
        perturbed_accuracies.append(perturbed_accuracy)
        print("Neuron num: {0}, perturbed accuracy: {1}".format(neuron_num, perturbed_accuracy))
    plt.plot(neuron_nums, perturbed_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Neuron Numbers')
    plt.xlabel('Neuron Number')
    plt.ylabel('Accuracy')
    plt.show()

demonstrate_attack_error_rate(50, 550, 50)