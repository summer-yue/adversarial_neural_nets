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

    # tmp_condition = tf.equal(pertubation, tf.zeros(tf.shape(pertubation)))
    # rand_pos_neg = tf.where(
    #     tf.random_uniform(
    #         tf.shape(pertubation),
    #         minval=0,
    #         maxval=1,
    #         dtype=tf.float64,
    #     ) > tf.ones(tf.shape(pertubation)) / 2,
    #     tf.ones(tf.shape(pertubation)),
    #     -1 * tf.ones(tf.shape(pertubation))
    # )
    # rand_pertubation = tf.where(tmp_condition,
    #     tf.ones(tf.shape(pertubation)),
    #     pertubation)
    # rand_perturbed_op = tf.squeeze(epsilon * rand_pertubation) + images

    sess=tf.Session()   
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=100)
    saver.restore(sess, "./tmp_adam3_float64/mnist_model_epochs-" + str(epoch_num))

    non_zero_elements_num_in_pertubation = np.count_nonzero(sess.run(pertubation, feed_dict={x: images, y: labels}))
    perturbed_images = sess.run(perturbed_op, feed_dict={x: images, y:labels})
    # rand_perturbed_images = sess.run(rand_perturbed_op, feed_dict={x: images, y:labels})
    
    perturbed_accuracy = sess.run(accuracy, feed_dict={x: perturbed_images, y: labels})
    # rand_perturbed_accuracy = sess.run(accuracy, feed_dict={x: rand_perturbed_images, y: labels})
    normal_accuracy = sess.run(accuracy, feed_dict={x: images, y: labels})
    # return perturbed_accuracy, rand_perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation
    return perturbed_accuracy, 0, normal_accuracy, non_zero_elements_num_in_pertubation

def demonstrate_attack_error_rate():
    """ Demonstrate the FGSM attack result error rate vs epoch number by showing
    How the value of epoch number changes the error rate on the test set
    When epsilon is fixed to 0.1
    """
    sample_images = mnist.test.images
    sample_labels = mnist.test.labels

    epoch_nums = []
    perturbed_accuracies = []
    rand_perturbed_accuracies = []
    normal_accuracies = []
    zero_elements_num_in_pertubation_list = []
    for epoch_num in range(1, 300, 25):
        epoch_nums.append(epoch_num)

        perturbed_accuracy, rand_perturbed_accuracy, normal_accuracy, non_zero_elements_num_in_pertubation = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        zero_elements_num_in_pertubation_list.append(10000*784 - non_zero_elements_num_in_pertubation)
        perturbed_accuracies.append(perturbed_accuracy)
        rand_perturbed_accuracies.append(rand_perturbed_accuracy)
        normal_accuracies.append(normal_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))
    
    # with open("final_64/perturbed_and_normal_accuracies"+str(epsilon) + ".txt", "w") as text_file:
    #     text_file.write(str(epoch_nums))
    #     text_file.write(str(normal_accuracies))
    #     text_file.write(str(perturbed_accuracies))

    plt.plot(epoch_nums, perturbed_accuracies)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch with Epsilon = ' + str(epsilon))
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
    for epoch_num in range(1, 300, 25):
        epoch_nums.append(epoch_num)

        perturbed_accuracy, _, normal_accuracy, non_zero_elements_num_in_pertubation = accuracy_after_fgsm_attack(sample_images, sample_labels, epoch_num)
        zero_elements_num_in_pertubation_list.append(10000*784 - non_zero_elements_num_in_pertubation)
        perturbed_accuracies.append(perturbed_accuracy)
        normal_accuracies.append(normal_accuracy)
        print("Epoch num: {0}, perturbed accuracy: {1}, normal accuracy: {2} ".format(epoch_num, perturbed_accuracy, normal_accuracy))

    plt.plot(epoch_nums, zero_elements_num_in_pertubation_list)
    plt.title('FGSM Attack Perturbation Zero Element Number As Overfitting Epsilon = ' + str(epsilon))
    plt.xlabel('Epoch Number')
    plt.ylabel('Number of zero values in Perturbation matrix')
    plt.show()

def graph_global_view():
    epoch_nums = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    perturbed_accuracies_01 = [0.7802, 0.72770000000000001, 0.70069999999999999, 0.65129999999999999, 0.64129999999999998, 0.59650000000000003, 0.56189999999999996, 0.52510000000000001, 0.52329999999999999, 0.53220000000000001, 0.48980000000000001, 0.49380000000000002]
    perturbed_accuracies_02 = [0.60929999999999995, 0.43190000000000001, 0.4103, 0.32979999999999998, 0.3281, 0.30170000000000002, 0.2873, 0.26400000000000001, 0.26429999999999998, 0.25330000000000003, 0.24079999999999999, 0.25169999999999998]
    perturbed_accuracies_05 = [0.18340000000000001, 0.14460000000000001, 0.21809999999999999, 0.18240000000000001, 0.21279999999999999, 0.21049999999999999, 0.20880000000000001, 0.21829999999999999, 0.20710000000000001, 0.21099999999999999, 0.2097, 0.20979999999999999]
  
    plt.plot(epoch_nums, perturbed_accuracies_01)
    plt.plot(epoch_nums, perturbed_accuracies_02)
    plt.plot(epoch_nums, perturbed_accuracies_05)

    plt.title('FGSM Attack on MNIST Classifier With Various Epoch')
    plt.legend(['Pertubed Accuracy With Epsilon 0.01', 'Epsilon 0.02',
        'Epsilon 0.05'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

def graph_global_view_sgd():
    epoch_nums = [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
    perturbed_accuracies_01 = [0.85909998, 0.93529999, 0.93309999, 0.9321, 0.93279999, 0.9332, 0.93349999, 0.93370003, 0.93339998, 0.9339, 0.9339, 0.93419999]
    perturbed_accuracies_05 = [0.33070001, 0.21160001, 0.2154, 0.2651, 0.2969, 0.32139999, 0.33450001, 0.34509999, 0.3545, 0.36269999, 0.36719999, 0.37290001]
    perturbed_accuracies_10 = [0.0228, 0.022299999, 0.060199998, 0.07, 0.074199997, 0.0757, 0.077100001, 0.077200003, 0.075999998, 0.077500001, 0.077600002, 0.077100001]
    perturbed_accuracies_20 = [0.0, 0.0026, 0.0052999998, 0.0046999999, 0.0049000001, 0.0048000002, 0.0043000001, 0.0038999999, 0.0044999998, 0.0046000001, 0.0048000002, 0.0043000001]
    normal_accuracies = [0.91689998, 0.97839999, 0.97939998, 0.98079997, 0.98030001, 0.9806, 0.98079997, 0.98079997, 0.98100001, 0.98100001, 0.98070002, 0.98070002]
    plt.plot(epoch_nums, perturbed_accuracies_01)
    plt.plot(epoch_nums, perturbed_accuracies_05)
    plt.plot(epoch_nums, perturbed_accuracies_10)
    plt.plot(epoch_nums, perturbed_accuracies_20)
    plt.plot(epoch_nums, normal_accuracies)
    plt.title('FGSM Attack on MNIST Classifier With Various Epoch with SGD Optimizer')
    plt.legend(['Pertubed Accuracy With Epsilon 0.01', 'Epsilon 0.05',
        'Epsilon 0.1',
        'Epsilon 0.2', 'Normal Accuracy'], loc='upper left')
    plt.xlabel('Epoch Number')
    plt.ylabel('Accuracy')
    plt.show()

#demonstrate_attack_result()
demonstrate_attack_error_rate()
#demonstrate_zeros_in_perturbation()
#graph_global_view()
#graph_global_view_sgd()