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

def accuracy_after_fgsm_attack(images, labels, l1, l2):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """
    tf.reset_default_graph()
    epsilon = 0.1

    # Building the graph
    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
    z3, y_ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)

    #apply pertubation to images
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, './tmp/mnist_model_neuron_num-' + str(l1*1000+l2))

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    return perturbed_accuracy

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result by showing
    How the value of epsilon changes the error rate on the test set
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    total_neuron_numbers = []
    perturbed_accuracies = []
    l1_units = l1+60
    # for l1_units in range(l1,l1+90,30):
    for l2_units in range (l2, l2+90, 30):
        total_neuron_numbers.append(l1_units + l2_units + l3)
        perturbed_accuracy = accuracy_after_fgsm_attack(sample_images, sample_labels, l1_units, l2_units)
        perturbed_accuracies.append(perturbed_accuracy)
    plt.plot(total_neuron_numbers, perturbed_accuracies)
    plt.title('FGSM Attack on Classifier')
    plt.xlabel('Total Neuron Number')
    plt.ylabel('Accuracy')
    plt.show()

# demonstrate_attack_result()
demonstrate_attack_error_rate()