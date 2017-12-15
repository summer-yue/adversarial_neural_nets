# Resisting Adversarial Attacks Against Neural Networks

In this project, we explored the effect of overfitting on a simple MNIST classifier's robustness against the Fast Gradient Sign Method (FGSM) attack.

Major Results and Interesting Observations:
1. Adversarial examples for a normal classifier and test sets (that had a low accuracy) for an overfitted classifier have similar behaviors. In overfitting, the classifier memorizes the training set to some extent, therefore generating inaccurate results on the test set data. In adversarial examples, the classifier memorizes the "normal" training data to some extent, therefore generating inaccurate results on adversarial examples.

2. The difference between overfitting and adversarial examples: by adding random noise to the test set, an overfitted classifier performs significantly worse. But if we add random noise to the normal examples in the test set (instead of performing a special gradient based attack), the performance stays good.

3. General regularization techniques, such as L2 regularziation, which are normally used to reduce the effect of overfitting, can increase a classifier's robustness to FGSM attack as well. 

4. An overfitted neural net, (a classifier trained a lot longer than needed), is more prone to adversarial attacks.

5. The FGSM attack results as a robustness measurement can be misleading. For example, in our very overfitted classifier, there are many 0 elements in our gradient, causing FGSM to stop attacking, leading to a false impression of a "more robust classifier". This has the same effects as gradient masking.

## Step 1: Set up a baseline classifier

We developed a simple 3 layer vanilla classifier for the MNIST data set

## Step 2: reproduce the fast gradient sign attack algorithm

We reproduced a fast gradient sign attack algorithm that preprocesses images from test data, with the goal of causing the classifier to misclassify

## Step 3: Figure out a robustness measurement
We look at the accuracy for perturbed examples vs epsilon used in the FGSM attack to analyze the robustness of the neural network.

## Step 4: Analyze how robustness changes as we overfit the classifier or use regularization techniques.
1. Using different regularization techniques - the neural net becomes MORE resistant to adversarial attacks
2. Increase the epoch number used in training to overfit the classifier - the neural net becomes LESS resistant to adversarial attacks

## Step 5: Develop a theory on what are the causes on why DNNs susceptible to adversarial examples
Please see compelete report [here](report.pdf)

## Authors
* **Summer Yue** 
* **Sangdon Park** 
