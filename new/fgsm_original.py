# Running the fast gradient sign method attack on the original classifier (epoch-4)
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from base_classifier import build_network, calc_loss, evaluation
from tensorflow.examples.tutorials.mnist import input_data as mnist_data

# Average perturbed accuracy is going up. 
# Next: Try overfitting more so normal accuracy doesn't go up.
# Epoch num: 1, perturbed accuracy: 0.3143, normal accuracy: 0.9541 
# Epoch num: 21, perturbed accuracy: 0.2626, normal accuracy: 0.9768 
# Epoch num: 41, perturbed accuracy: 0.354, normal accuracy: 0.9789 
# Epoch num: 61, perturbed accuracy: 0.4321, normal accuracy: 0.9816 
# Epoch num: 81, perturbed accuracy: 0.4955, normal accuracy: 0.9818 
# Epoch num: 101, perturbed accuracy: 0.5159, normal accuracy: 0.9829 
# Epoch num: 121, perturbed accuracy: 0.5196, normal accuracy: 0.983 
# Epoch num: 141, perturbed accuracy: 0.5262, normal accuracy: 0.9836 
# Epoch num: 161, perturbed accuracy: 0.5267, normal accuracy: 0.9844 
# Epoch num: 181, perturbed accuracy: 0.5273, normal accuracy: 0.9844 

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

def accuracy_after_fgsm_attack(images, labels, epoch_num, round_number):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """
    #apply pertubation to images
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=1000)
    saver.restore(sess, "./models/overfitted-" + str(round_number) + "/original_epoch-" + str(epoch_num))

    non_zero_elements_num_in_pertubation = np.count_nonzero(sess.run(pertubation, feed_dict={x: images, y: labels}))
    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
   
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})   
    normal_accuracy = sess.run(accuracy, feed_dict={x: images, y: labels})
    return perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation

def demonstrate_attack_error_rate(epoch_num_start, epoch_num_end, stride, total_round_num):
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    We take the results from the average of "total_round_num" number of experiemnts saved in /models/overfitted-
    Args:
        epoch_num_start: start recording models from this epoch
        epcoh_num_end: stop training at this epoch number
        stride: number of epochs between two adjacentmodels
        total_round_num: number of experiments used to average results
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels 

    epoch_nums = []
    perturbed_accuracies = []
    normal_accuracies = []

    total_perturbed_accuracy = 0
    total_normal_accuracy = 0

    for epoch_num in range(epoch_num_start, epoch_num_end, stride):
        epoch_nums.append(epoch_num)
        for round_number in range(total_round_num):
            perturbed_accuracy, normal_accuracy, _ = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num, round_number)
            total_perturbed_accuracy += perturbed_accuracy
            total_normal_accuracy += normal_accuracy
        perturbed_accuracies.append(total_perturbed_accuracy * 1.0 / total_round_num)
        normal_accuracies.append(total_normal_accuracy * 1.0 / total_round_num)

        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))

    plt.plot(epoch_nums, perturbed_accuracies)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.legend(['Pertubed Accuracy', 'Normal Accuracy'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

def demonstrate_zeros_in_perturbation(epoch_num_start, epoch_num_end, stride, total_round_num):
    """ Demonstrate FGSM attack's perturbation matrix contains how many 0s.
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    zero_elements_num_in_pertubation_list = []

    for epoch_num in range(epoch_num_start, epoch_num_end, stride):

        tf.reset_default_graph()

        # Building the graph
        x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
        y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
        z3, y_, _ = build_network(x, l1, l2, l3)
        loss = calc_loss(z3, y)
        accuracy = evaluation(y, y_)

        zero_elements_num_in_pertubation = 0
        epoch_nums.append(epoch_num)
        for round_number in range(total_round_num):

            _, _, non_zero_elements_num_in_pertubation = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num, round_number)
            zero_elements_num_in_pertubation += 10000*784 - non_zero_elements_num_in_pertubation
        zero_elements_num_in_pertubation = zero_elements_num_in_pertubation * 1.0 / total_round_num
        zero_elements_num_in_pertubation_list.append(zero_elements_num_in_pertubation)
       
        print("Epoch num: {0}, average number of 0s in the gradient {1} ".format(epoch_num, zero_elements_num_in_pertubation))

    plt.plot(epoch_nums, zero_elements_num_in_pertubation_list)
    plt.title('FGSM Attack Perturbation Zero Element Number As Overfitting')
    plt.xlabel('Epoch Number')
    plt.ylabel('Number of zero values in Perturbation matrix')
    plt.show()

demonstrate_attack_error_rate(101, 200, 20, 5)
#demonstrate_zeros_in_perturbation()