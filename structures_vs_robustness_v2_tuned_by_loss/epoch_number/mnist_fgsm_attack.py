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
epsilon = 0.2

tf.reset_default_graph()

# Building the graph
x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
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
    saver = tf.train.Saver(max_to_keep=10)
    saver.restore(sess, "./tmp2/mnist_model_epochs-" + str(epoch_num))

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    normal_accuracy = sess.run(accuracy, feed_dict={x: images, y: labels})
    return perturbed_accuracy, normal_accuracy

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.train.images
    sample_labels = mnist.train.labels

    epoch_nums = []
    perturbed_accuracies = []
    normal_accuracies = []
    for epoch_num in range(0, 300, 25):
        epoch_nums.append(epoch_num)
        perturbed_accuracy, normal_accuracy = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        perturbed_accuracies.append(perturbed_accuracy)
        normal_accuracies.append(normal_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))
    
    with open("perturbed_and_normal_accuracies"+str(epsilon) + ".txt", "w") as text_file:
        text_file.write(str(epoch_nums))
        text_file.write(str(normal_accuracies))
        text_file.write(str(perturbed_accuracies))

    plt.plot(epoch_nums, perturbed_accuracies)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.legend(['Pertubed Accuracy', 'Normal Accuracy'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

def graph_global_view():
    epoch_nums = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    perturbed_accuracies_01 = [0.92023635, 0.95392728, 0.96780002, 0.97327274, 0.99209088, 0.99816364, 0.99787271, 0.98989093, 0.9964, 0.98103637, 0.9848727, 0.98145455]
    perturbed_accuracies_05 = [0.3126182, 0.44250908, 0.54401821, 0.4479818, 0.48654544, 0.47703636, 0.46900001, 0.47299999, 0.48983636, 0.48398182, 0.50870907, 0.49821818]
    perturbed_accuracies_10 = [0.032109089, 0.15150909, 0.1686909, 0.15456364, 0.17196363, 0.21296364, 0.22258182, 0.18930909, 0.2658, 0.25341818, 0.31563637, 0.36585453]
    perturbed_accuracies_15 = [0.0035636364, 0.068509094, 0.065272726, 0.092454545, 0.10812727, 0.17423636, 0.19859999, 0.14816363, 0.24007273, 0.22579999, 0.29923636, 0.35349092]
    perturbed_accuracies_20 = [0.00058181817, 0.040381819, 0.03089091, 0.074836366, 0.089199997, 0.1648, 0.19347273, 0.13829091, 0.23418182, 0.22045454, 0.2955091, 0.35001817]
    normal_accuracies = [0.9683091, 0.99672729, 0.9975273, 0.99929088, 1.0, 1.0, 1.0, 1.0, 1.0, 0.99958181, 0.99969089, 0.99930906]
    plt.plot(epoch_nums, perturbed_accuracies_01)
    plt.plot(epoch_nums, perturbed_accuracies_05)
    plt.plot(epoch_nums, perturbed_accuracies_10)
    plt.plot(epoch_nums, perturbed_accuracies_15)
    plt.plot(epoch_nums, perturbed_accuracies_20)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.legend(['Pertubed Accuracy With Epsilon 0.01', 'Epsilon 0.05',
        'Epsilon 0.1', 'Epsilon 0.15',
        'Epsilon 0.2', 'Normal Accuracy'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

#demonstrate_attack_error_rate()
graph_global_view()