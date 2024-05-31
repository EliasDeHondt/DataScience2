![logo](https://eliasdh.com/assets/media/images/logo-github.png)
# 💙🤍Small explainer🤍💙

## 📘Table of Contents

1. [📘Table of Contents](#📘table-of-contents)
2. [🖖Introduction](#🖖introduction)
3. [📊Dropout Layer](#📊dropout-layer)
    1. [🐍Python Example:](#🐍python-example)
4. [📊Batch Normalization Layer](#📊batch-normalization-layer)
    1. [🐍Python Example:](#🐍python-example)
5. [📊Activation Layer](#📊activation-layer)
    1. [🐍Python Example:](#🐍python-example)
6. [📊LeakyReLU Layer](#📊leakyrelu-layer)
    1. [🐍Python Example:](#🐍python-example)
7. [📊Gaussian Noise Layer](#📊gaussian-noise-layer)
    1. [🐍Python Example:](#🐍python-example)
8. [📊Gaussian Dropout Layer](#📊gaussian-dropout-layer)
    1. [🐍Python Example:](#🐍python-example)
9. [📊ELU (Exponential Linear Unit) Layer](#📊elu-exponential-linear-unit-layer)
    1. [🐍Python Example:](#🐍python-example)
10. [📊Example of Incorporating These Layers](#📊example-of-incorporating-these-layers)
    1. [🐍Python Example:](#🐍python-example)
11. [🔗Links](#🔗links)

---

## 🖖Introduction

This document provides a brief overview of some common probability distributions and how to generate random numbers from them using Python.

## 📊Dropout Layer

Dropout layers help prevent overfitting by randomly setting a fraction of input units to 0 at each update during training time.

### 🐍Python Example:

```python
from keras.layers import Dropout

x_iris_model = Dropout(0.5)(x_iris_model)
```

## 📊Batch Normalization Layer

Batch normalization layers normalize the output of the previous layers to speed up training and provide some regularization.

### 🐍Python Example:

```python
from keras.layers import BatchNormalization

x_iris_model = BatchNormalization()(x_iris_model)
```

## 📊Activation Layer

Instead of specifying the activation function within the Dense layer, you can use a separate Activation layer for more complex activation functions.

### 🐍Python Example:

```python
from keras.layers import Activation

x_iris_model = Activation('relu')(x_iris_model)
```

## 📊LeakyReLU Layer

LeakyReLU is an alternative to standard ReLU activation that allows a small gradient when the unit is not active, which can help with the "dying ReLU" problem.

### 🐍Python Example:

```python
from keras.layers import LeakyReLU

x_iris_model = LeakyReLU(alpha=0.1)(x_iris_model)
```

## 📊Gaussian Noise Layer

This layer adds Gaussian noise to the input, which can help with regularization.

### 🐍Python Example:

```python
from keras.layers import GaussianNoise

x_iris_model = GaussianNoise(0.1)(x_iris_model)
```

## 📊Gaussian Dropout Layer

This layer applies multiplicative 1-centered Gaussian noise.

### 🐍Python Example:

```python
from keras.layers import GaussianDropout

x_iris_model = GaussianDropout(0.5)(x_iris_model)
```

## 📊ELU (Exponential Linear Unit) Layer

TELU is an alternative activation function that can help with the vanishing gradient problem.

### 🐍Python Example:

```python
from keras.layers import ELU

x_iris_model = ELU(alpha=1.0)(x_iris_model)
```

## 📊Example of Incorporating These Layers

Here's an example of how you might integrate some of these layers into your model:

### 🐍Python Example:

```python
from keras.models import Model
from keras.layers import Input, Dense, Dropout, BatchNormalization, Activation, LeakyReLU, GaussianNoise, ELU
from keras.optimizers import Adam

inputs_iris = Input(shape=(4,))
x_iris_model = Dense(8)(inputs_iris)
x_iris_model = BatchNormalization()(x_iris_model)
x_iris_model = LeakyReLU(alpha=0.1)(x_iris_model)
x_iris_model = Dense(16)(x_iris_model)
x_iris_model = ELU(alpha=1.0)(x_iris_model)
x_iris_model = Dropout(0.5)(x_iris_model)
x_iris_model = Dense(32)(x_iris_model)
x_iris_model = GaussianNoise(0.1)(x_iris_model)
x_iris_model = Activation('sigmoid')(x_iris_model)
x_iris_model = Dense(16)(x_iris_model)
x_iris_model = Activation('sigmoid')(x_iris_model)
x_iris_model = Dense(8)(x_iris_model)
x_iris_model = Activation('sigmoid')(x_iris_model)
outputs_iris = Dense(3, activation='softmax')(x_iris_model)
model_iris = Model(inputs_iris, outputs_iris, name='Iris_NN')

model_iris.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Assuming x_iris_learning_normalize and y_iris_learning_replace are already defined
history_iris = model_iris.fit(
    x_iris_learning_normalize,
    y_iris_learning_replace,
    epochs=1000,
    callbacks=[PlotLossesKeras()],
    verbose=False
)
```

## 🔗Links
- 👯 Web hosting company [EliasDH.com](https://eliasdh.com).
- 📫 How to reach us elias.dehondt@outlook.com.