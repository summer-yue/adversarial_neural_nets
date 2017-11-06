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
z3, y_, params = build_network(x, l1, l2, l3)
loss = calc_loss(z3, y)
accuracy = evaluation(y, y_)

def accuracy_after_fgsm_attack(images, labels, epoch_num):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """

    #apply pertubation to images
    epsilon = 0.05
    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess, "./tmp/mnist_model_epochs-" + str(epoch_num))

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    return perturbed_accuracy

def get_average_weights_from_epoch(epoch_num):
    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=50)
    saver.restore(sess, "./tmp/mnist_model_epochs-" + str(epoch_num))
    params_nn = sess.run(params, feed_dict={})
    w1 = params_nn["w1"]
    w2 = params_nn["w2"]
    w3 = params_nn["w3"]
    # print(np.mean(w1))
    # print(np.mean(w2))
    # print(np.mean(w3))
    return np.mean(w1) * np.mean(w2) * np.mean(w3)

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    perturbed_accuracies = []
    for epoch_num in range(0, 45):
        epoch_nums.append(epoch_num)
        perturbed_accuracy = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        perturbed_accuracies.append(perturbed_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}".format(epoch_num, perturbed_accuracy))
    plt.plot(epoch_nums, perturbed_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

def demonstrate_epoch_vs_weights():
    """ Demonstrate the neural net's average weights vs epoch number
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    weights_average = []
    for epoch_num in range(0, 45):
        epoch_nums.append(epoch_num)
        weights_for_epoch = get_average_weights_from_epoch(epoch_num)
        weights_average.append(weights_for_epoch)

        print("Epoch num: {0}, Weights on average for epoch: {1}".format(epoch_num, weights_for_epoch))
    plt.plot(epoch_nums, weights_average)
    plt.title('MNIST Classifier Weights VS Epoch')
    plt.xlabel('Epoch Number')
    plt.ylabel('Average Weights')
    plt.show()

#demonstrate_epoch_vs_weights()
demonstrate_attack_error_rate()

