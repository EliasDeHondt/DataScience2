![logo](https://eliasdh.com/assets/media/images/logo-github.png)
# 💙🤍Theory🤍💙

## 📘Table of Contents

1. [📘Table of Contents](#📘table-of-contents)
2. [🖖Introduction](#🖖introduction)
3. [📚Theory](#📚theory)
    1. [🧑‍💻Discriminant Analysis](#🧑‍💻discriminant-analysis)
    2. [📊Evaluation Metrics](#📊evaluation-metrics)
    3. [🤖Neural Networks](#🤖neural-networks)
       1. [🤓Extensive explanation (Out of scope)](#🤓extensive-explanation-out-of-scope)
          1. [Key Concepts in Neural Networks](#key-concepts-in-neural-networks)
          2. [Important Terms and Techniques](#important-terms-and-techniques)
          3. [Hyperparameters and Model Performance](#hyperparameters-and-model-performance)
    4. [🔭Meta-Heuristics](#🔭meta-heuristics)
4. [🔗Links](#🔗links)


---

## 🖖Introduction

This document contains the theory (P4) of the different topics that are covered in the course. The theory is divided into different sections, each section contains a brief explanation of the topic and the most important concepts that are related to it.

## 📚Theory

### 🧑‍💻Discriminant Analysis

> **NOTE:** See the following directory for more information about the topic: [Discriminant Analysis W19P4](/W19P4)

Discriminant analysis is a statistical method used to analyze differences between groups and to predict to which group a new observation belongs.

There are two main goals of discriminant analysis:
- **Determining differences between groups (descriptive discriminant analysis):**
  - This goal focuses on identifying the differences between two or more mutually exclusive groups. For example, you could look at the differences between groups such as men and women, or between smokers and non-smokers.
  - The group into which an observation falls is represented by the dependent variable. For example, the dependent variable could be `gender` with the groups `male` and `female`.
  - The method looks for patterns in the values of the independent variables (the characteristics or properties you measure) to describe the differences between the groups.
- **Predicting group membership of new observations (predictive discriminant analysis):**
  - This goal focuses on predicting the group to which a new observation belongs. For example, you could use discriminant analysis to predict whether a new customer will buy a product or not.
  - The method uses the values of the independent variables to predict the group membership of the dependent variable.
  - The method uses the differences between the groups identified in the descriptive discriminant analysis to make these predictions.

Summarizing:
- Descriptive discriminant analysis describes the differences between existing groups.
- Predictive discriminant analysis predicts the group to which a new observation will belong.

### 📊Evaluation Metrics

> **NOTE:** See the following directory for more information about the topic: [Evaluation Metrics W20P4](/W20P4)

Evaluation metrics are used to measure the quality of the statistical or machine learning model. Evaluating machine learning models or algorithms is essential for any project. There are several evaluation metrics available to assess the performance of a model. Some of the most common evaluation metrics are:

- **Accuracy:**
  - Interpretation: Accuracy is the most intuitive performance measure, and it is simply a ratio of correctly predicted observation to the total observations. One may think that, if we have high accuracy then our model is best. Yes, accuracy is a great measure but only when you have symmetric datasets where values of false positive and false negatives are almost same. Therefore, you have to look at other parameters to evaluate the performance of your model.
  - Formula: `Accuracy = (TP + TN) / (TP + TN + FP + FN)`

- **Precision:**
    - Interpretation: Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate.
    - Formula: `Precision = TP / (TP + FP)`

- **Recall (Sensitivity):**
    - Interpretation: Recall is the ratio of correctly predicted positive observations to the all observations in actual class. The question recall answers is: Of all the passengers that truly survived, how many did we label? It is also called Sensitivity.
    - Formula: `Recall = TP / (TP + FN)`

- **F measure (F1 score):**
    - Interpretation: F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. It is a good way to balance the precision and recall. F1 Score is the harmonic mean of Precision and Recall and gives a better measure of the incorrectly classified cases than the Accuracy Metric.
    - Formula: `Fm = (b^2 + 1) * Precision * Recall / (b^2 * Precision + Recall)`

- **True Positive Rate (TPR):**
    - Interpretation: True Positive Rate is synonymous with Recall.
    - Formula: `TPR = TP / (TP + FN)`

- **False Positive Rate (FPR):**
  - Interpretation: The false positive rate is the proportion of all actual negative cases that got classified as positive. It is a measure of the model's precision for negative predictions.
  - Formula: `FPR = FP / (FP + TN)`

- **Threshold:**
  - Interpretation: The threshold is the value that determines the classification of the data. If the probability of the class is higher than the threshold, the data is classified as that class. If the probability is lower than the threshold, the data is classified as the other class.
  - Example: `Threshold = 0.5`

- **Receiver Operator Characteristic Curve (ROC):**
  - Interpretation: The ROC curve is a graphical representation of the true positive rate against the false positive rate. It shows the trade-off between sensitivity and specificity.

![ROC Space](/P4%20Repetition/Images/ROC-Space.png)

### 🤖Neural Networks

> **NOTE:** See the following directory for more information about the topic: [Neural Networks W22P4](/W22P4)

A neural network is a series of algorithms that endeavors to recognize underlying relationships in a set of data through a process that mimics the way the human brain operates. Neural networks can adapt to changing input; so the network generates the best possible result without needing to redesign the output criteria.

- **Input Layer:**
  - The input layer is the first layer of the neural network. It is responsible for receiving the input data and passing it to the hidden layers.

- **Hidden Layer:**
  - The hidden layers are the layers between the input and output layers. They are responsible for processing the input data and passing it to the output layer.

- **Output Layer:**
  - The output layer is the last layer of the neural network. It is responsible for generating the output data based on the input data.


#### 🤓Extensive explanation (Out of scope)

A neural network is a series of algorithms designed to recognize underlying relationships in data by mimicking the human brain's operations. Neural networks can adapt to changing inputs, enabling them to generate the best possible results without redesigning the output criteria.

##### Key Concepts in Neural Networks

- **Artificial Neurons**:
  - Units or nodes in a neural network, inspired by biological neurons.
  - Connections between neurons are like synapses in a brain, transmitting signals.

- **Signal Processing**:
  - Signals at neuron connections are real numbers.
  - Outputs are computed by non-linear functions of the sum of inputs.
  - Connections, called 'edges', have weights that adjust during learning.
  - Neurons may have thresholds, sending signals only if the aggregate signal exceeds the threshold.

- **Layers**:
  - Neurons are typically grouped into layers (input, hidden, and output layers).
  - Each layer performs different transformations on inputs.

##### Important Terms and Techniques

- **Activation Function**:
  - Introduces non-linearity into the neural network, determining neuron outputs.

- **Backpropagation**:
  - A supervised learning algorithm for training neural networks.
  - Calculates gradients of the loss function with respect to network weights and updates weights accordingly.

- **Gradient Descent**:
  - An optimization algorithm that minimizes the loss function by iteratively updating weights in the negative gradient direction.

- **Loss Function**:
  - Measures the difference between predicted and actual outputs.
  - Used to train the network by minimizing prediction errors.

- **Optimization Algorithm**:
  - Minimizes the loss function by updating network weights based on gradients.

- **Regularization**:
  - Prevents overfitting by adding a penalty term to the loss function that discourages large weights.

- **Dropout**:
  - A regularization technique that randomly sets a fraction of weights to zero during training to prevent overfitting.

- **Batch Normalization**:
  - Normalizes inputs to each layer, improving training efficiency by maintaining zero mean and unit variance.

##### Hyperparameters and Model Performance

- **Hyperparameters**:
  - Set before training; include learning rate, batch size, and number of hidden layers.

- **Overfitting**:
  - When a model performs well on training data but poorly on test data, learning noise instead of the underlying pattern.

- **Underfitting**:
  - When a model performs poorly on both training and test data, being too simple to capture data patterns.

- **Vanishing Gradient**:
  - Gradients become very small as they propagate back through the network, often due to small weight initialization.

- **Exploding Gradient**:
  - Gradients become excessively large as they propagate back through the network, often due to large weight initialization.

### 🔭Meta-Heuristics

> **NOTE:** See the following directory for more information about the topic: [Meta-heuristics W23P4-W24P4](/W23P4-W24P4)

Meta-heuristics are high-level strategies that guide the search process to find near-optimal solutions. 
They are used to solve optimization problems that are difficult to solve using exact methods. 
Meta-heuristics are general problem-solving techniques that can be applied to a wide range of problems.

There are two types:
- Simulated Annealing
  - Simulated annealing is a probabilistic optimization algorithm that is used to find the global optimum of a function.
- Genetic Algorithms
  - Genetic algorithms are a type of optimization algorithm that is inspired by the process of natural selection.

Summarizing:
- Meta-heuristics does not provide the best solution, but good enough. An algorithm gives the best solution, and probably the only one.

## 🔗Links
- 👯 Web hosting company [EliasDH.com](https://eliasdh.com).
- 📫 How to reach us elias.dehondt@outlook.com