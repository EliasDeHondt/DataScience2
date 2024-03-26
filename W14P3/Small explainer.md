![logo](https://eliasdh.com/assets/media/images/logo-github.png)
# 💙🤍Small explainer🤍💙

## 📘Table of Contents

1. [📘Table of Contents](#📘table-of-contents)
2. [🖖Introduction](#🖖introduction)
3. [📊Binomiale Distribution](#📊binomiale-distribution)
    1. [🐍Python Example](#🐍python-example)
4. [📊Normale Distribution](#📊normale-distribution)
    1. [🐍Python Example](#🐍python-example)
5. [📊Poisson Distribution](#📊poisson-distribution)
    1. [🐍Python Example](#🐍python-example)
6. [🔗Links](#🔗links)

---

## 🖖Introduction

In this document, we will explain the following probability distributions: **binomial**, **normal**, and **Poisson**. We will provide a brief description of each distribution and a Python example using the `scipy.stats` module.

## 📊Binomiale Distribution

- The **binomial distribution** models the probability of a specific number of successes in a fixed number of repeated independent experiments with two possible outcomes.
> Translation in Dutch: De **binomiale verdeling** modelleert de kans op een bepaald aantal successen in een vast aantal herhaalde onafhankelijke experimenten met twee mogelijke uitkomsten.

### 🐍Python Example:

```python
from scipy.stats import binom as binomial # Binomial distribution

n = 5  # Number of trials
k = 2 # Probability of exactly k successes
p = 0.3  # Probability of success in each trial

binomial_pmf = binomial.pmf(k, n, p)
print(f"Probability of {k} successes: {binomial_pmf:.4f}")

```

## 📊Normale Distribution

- The **normal distribution** is a continuous probability distribution commonly used to model natural phenomena, where most observations cluster around the mean according to the bell-shaped curve.
> Translation in Dutch: De **normale verdeling** is een continue kansverdeling die vaak wordt gebruikt voor het modelleren van natuurlijke verschijnselen, waarbij de meeste waarnemingen rond het gemiddelde liggen volgens de klokvormige curve.

### 🐍Python Example:

```python
from scipy.stats import norm as normal # Normal distribution

x = 60 # Probability of a value less than x
mean = 50  # Mean
std_dev = 10  # Standard deviation

normal_cdf = normal.cdf(x, mean, std_dev)
print(f"Probability of a value less than {x}: {normal_cdf:.4f}")
```

## 📊Poisson Distribution

- The **Poisson distribution** models the probability of a specific number of events occurring in a fixed time interval, given the average frequency of events.
> Translation in Dutch: De **Poisson verdeling** modelleert de kans op een bepaald aantal gebeurtenissen dat plaatsvindt in een vast tijdsinterval, gegeven de gemiddelde frequentie van gebeurtenissen.

### 🐍Python Example:

```python
from scipy.stats import poisson as poisson # Poisson distribution

k = 2 # Probability of exactly k events
lambda_ = 3  # Average frequency of events

poisson_pmf = poisson.pmf(k, lambda_)
print(f"Probability of {k} events: {poisson_pmf:.4f}")
```

## 🔗Links
- 👯 Web hosting company [EliasDH.com](https://eliasdh.com).
- 📫 How to reach us eliasdehondt@outlook.com.