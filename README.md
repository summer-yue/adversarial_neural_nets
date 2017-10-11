# Resisting Adversarial Attacks Against Neural Networks

Neural networks have become pervasive in various domains. However, in industries such as finance, healthcare and autonomous vehicles, insecure algorithms prone to adversarial attacks can pose extreme security risks. In this proposed research project, we explore how difference in DNNs' architectures impacts their resistance to adversarial attacks from the fast gradient sign as well as other gradient based methods.

The questions we ask include, but not limited to:

1. Is a deeper neural net more prone to adversarial attack than a shallow neural net?
2. Is a neural net with fewer neurons per layer more resistant to adversarial attacks?
3. How would different regularization algorithms impact the resistance to attacks?
4. Would classifiers with low dimension inputs naturally resist adversarial examples?
5. What kind of activation functions would resist attacks better?
6. Would training on more examples lead to better resistance to attacks?

## Step 1: Set up a baseline classifier (DONE)

Develop a simple 3 layer vanilla classifier for the mnist data set

## Step 2: Develop the fast gradient sign attack algorithm

Develop a fast gradient sign attack algorithm that preprocesses images from test data, with the goal of causing the classifier to misclassify

## Step 3: Figure out a robustness measurement
Look into a better measurement for a classifier's robustness, a simple one would be the distortion needed to be applied to each pixel in the fast gradient sign method that would lead to a decrease test accuracy. This can be represented in a line graph, where the x axis is distortion and y axis is test accuracy.

## Step 4: Analyze what happens to robustness when the architecture of the DNN changes.
1. Adding layers into the original classifier
2. Increasing the number neurons to each layer
3. Using different regularization techniques
4. Use dimensionality reduction on inputs before training and testing
5. Using linear vs nonlinear activation functions

We want to observe how robustness measurements and accuracy would change under these scenarios.

## Step 5: Develop a theory on what are the causes on why DNNs susceptible to adversarial examples:
EXPLAINING AND HARNESSING ADVERSARIAL EXAMPLES by Goodfellow's group argued that the existance of adversarial examples is due to the linear properties of the neural net. That is shown by adversarial examples on relatively linear networks with high dimensional inputs.

I suspect in addition to the input dimensions, larger weights on each layer, number of neurons on each layer and an increase in the number of layers would lead to higher distortions of output, therefore making the network less resistant to adversarial attacks. Regularization may enable the neural net to be more robust to attacks.

## Authors
* **Summer Yue** 
