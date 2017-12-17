# Running the fast gradient sign method attack on the original classifier (epoch-4)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from base_classifier import build_network, calc_loss, evaluation
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
epsilon = 0.05

tf.reset_default_graph()

# Building the graph
x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
z3, y_, _ = build_network(x, l1, l2, l3)
loss = calc_loss(z3, y)
accuracy = evaluation(y, y_)

def accuracy_after_fgsm_attack(images, labels, epoch_num):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """
    #apply pertubation to images
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./models/original/original_epoch-" + str(epoch_num))

    non_zero_elements_num_in_pertubation = np.count_nonzero(sess.run(pertubation, feed_dict={x: images, y: labels}))
    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
   
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})   
    normal_accuracy = sess.run(accuracy, feed_dict={x: images, y: labels})
    return perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    perturbed_accuracies = []
    normal_accuracies = []
    zero_elements_num_in_pertubation_list = []
    for epoch_num in range(1, 200, 20):
        epoch_nums.append(epoch_num)

        perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        zero_elements_num_in_pertubation_list.append(10000*784 - non_zero_elements_num_in_pertubation)
        perturbed_accuracies.append(perturbed_accuracy)
    
        normal_accuracies.append(normal_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))

    plt.plot(epoch_nums, perturbed_accuracies)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.legend(['Pertubed Accuracy', 'Normal Accuracy'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

def demonstrate_zeros_in_perturbation():
    """ Demonstrate FGSM attack's perturbation matrix contains how many 0s.
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    perturbed_accuracies = []
    normal_accuracies = []
    zero_elements_num_in_pertubation_list = []
    for epoch_num in range(1, 200, 20):
        epoch_nums.append(epoch_num)

        perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        zero_elements_num_in_pertubation_list.append(10000*784 - non_zero_elements_num_in_pertubation)
        perturbed_accuracies.append(perturbed_accuracy)
        normal_accuracies.append(normal_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))

    plt.plot(epoch_nums, zero_elements_num_in_pertubation_list)
    plt.title('FGSM Attack Perturbation Zero Element Number As Overfitting')
    plt.xlabel('Epoch Number')
    plt.ylabel('Number of zero values in Perturbation matrix')
    plt.show()

demonstrate_attack_error_rate()
#demonstrate_zeros_in_perturbation()