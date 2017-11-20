import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist import build_network, build_network_32, calc_loss, evaluation
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

def accuracy_after_fgsm_attack(images, labels):
    """ Perform fast gradient sign method attack on a list of (image, label) pair to lead mnist classifier to misclassify
    return: new accuracy after perturbation
    """
    #apply pertubation to images
    # Building the graph
    x = tf.placeholder(tf.float64, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float64, [None, output_dimension], name="labels")
    z3, y_, _ = build_network(x, l1, l2, l3)
    loss = calc_loss(z3, y)

    pertubation = tf.sign(tf.gradients(loss, x))
    perturbed_op = tf.squeeze(epsilon * pertubation) + images
  
    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./tmp_adam2_high_precision/mnist_model_epochs-26")

    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})

    return perturbed_images, perturbed_accuracy

def create_perturbed_images():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    perturbed_images, _ = accuracy_after_fgsm_attack(sample_images, sample_labels)
    np.savetxt('perturbed_images_under_float64_epoch26_epsilon_' + str(epsilon) + '.txt', perturbed_images, delimiter = ',') 

def demonstrate_attack_error_rate():
    perturbed_images = np.loadtxt(open('perturbed_images_under_float64_epoch26_epsilon_' + str(epsilon) + '.txt',"rb"),delimiter=",",skiprows=0)
    sample_labels = mnist.test.labels
    sample_images = mnist.test.images

    # Building the graph
    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
    z3, y_, _ = build_network_32(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./tmp_adam1/mnist_model_epochs-26")

    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: sample_labels})
    #normal_accuracy = sess.run(accuracy, feed_dict={x: sample_images, y: sample_labels})
    print("epsilon:" + str(epsilon))
    print("perturbed accuracy:" + str(perturbed_accuracy))
    #print("Normal accuracy:" + str(normal_accuracy))

def demonstrate_transfer_attack_vs_regular_attack():
    sample_labels = mnist.test.labels
    sample_images = mnist.test.images
    epsilons = [0.02, 0.05, 0.1, 0.15, 0.2]
    transfer_perturbed_accuracies = [0.962, 0.8853, 0.5813, 0.3507, 0.2258]
    regular_perturbed_accuracies = []

    x = tf.placeholder(tf.float32, [None, input_dimension], name="input")
    y = tf.placeholder(tf.float32, [None, output_dimension], name="labels")
    z3, y_, _ = build_network_32(x, l1, l2, l3)
    loss = calc_loss(z3, y)
    accuracy = evaluation(y, y_)

    for epsilon in epsilons:
        pertubation = tf.sign(tf.gradients(loss, x))
        perturbed_op = tf.squeeze(epsilon * pertubation) + sample_images
      
        sess=tf.Session()   
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=100)
        saver.restore(sess, "./tmp_adam1/mnist_model_epochs-26")

        perturbed_images = sess.run(perturbed_op, feed_dict={x: sample_images, y:sample_labels})
        regular_perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: sample_labels})

        regular_perturbed_accuracies.append(regular_perturbed_accuracy)

    plt.plot(epsilons, transfer_perturbed_accuracies)
    plt.plot(epsilons, regular_perturbed_accuracies)
    plt.title('FGSM Epoch 26 Regular Perturbed Accuracy VS Transferred Perturbed Accuracy from Accurate Model')
    plt.legend(['Transfer Pertubed Accuracy', 'Regular Perturbed Accuracy'], loc='upper left')
    plt.xlabel('Epsilon')
    plt.ylabel('Accuracy')
    plt.show()

#create_perturbed_images()
#demonstrate_attack_error_rate()
demonstrate_transfer_attack_vs_regular_attack()
