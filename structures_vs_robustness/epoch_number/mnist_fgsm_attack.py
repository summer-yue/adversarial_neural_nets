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

# Building the graph
x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
z3, y_ = build_network(x, l1, l2, l3)
loss = calc_loss(z3, y)
accuracy = evaluation(y, y_)

def accuracy_after_fgsm_attack(images, labels, epoch_num):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """

    #apply pertubation to images
    epsilon = 0.01
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, "./tmp/mnist_model_epochs-" + str(epoch_num))

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    return perturbed_accuracy

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    perturbed_accuracies = []
    for epoch_num in range(5,15):
        epoch_nums.append(epoch_num)
        perturbed_accuracy = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        perturbed_accuracies.append(perturbed_accuracy)
    plt.plot(epoch_nums, perturbed_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

demonstrate_attack_error_rate()