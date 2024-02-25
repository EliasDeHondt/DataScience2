# Small explainer

## Binomiale Distribution

- The **binomial distribution** models the probability of a specific number of successes in a fixed number of repeated independent experiments with two possible outcomes.
> Translation in Dutch: De **binomiale verdeling** modelleert de kans op een bepaald aantal successen in een vast aantal herhaalde onafhankelijke experimenten met twee mogelijke uitkomsten.

### Python Example:

```python
from scipy.stats import binom as binomial # Binomial distribution

n = 5  # Number of trials
p = 0.3  # Probability of success in each trial

# Probability of exactly k successes
k = 2
binomial_pmf = binomial.pmf(k, n, p)
print(f"Probability of {k} successes: {binomial_pmf:.4f}")

```

## Normale Distribution

- The **normal distribution** is a continuous probability distribution commonly used to model natural phenomena, where most observations cluster around the mean according to the bell-shaped curve.
> Translation in Dutch: De **normale verdeling** is een continue kansverdeling die vaak wordt gebruikt voor het modelleren van natuurlijke verschijnselen, waarbij de meeste waarnemingen rond het gemiddelde liggen volgens de klokvormige curve.

### Python Example:

```python
from scipy.stats import norm as normal # Normal distribution

mean = 50  # Mean
std_dev = 10  # Standard deviation

# Probability of a value less than 60
x = 60
normal_cdf = normal.cdf(x, mean, std_dev)
print(f"Probability of a value less than {x}: {normal_cdf:.4f}")
```

## Poisson Distribution

- The **Poisson distribution** models the probability of a specific number of events occurring in a fixed time interval, given the average frequency of events.
> Translation in Dutch: De **Poisson verdeling** modelleert de kans op een bepaald aantal gebeurtenissen dat plaatsvindt in een vast tijdsinterval, gegeven de gemiddelde frequentie van gebeurtenissen.

### Python Example:

```python
from scipy.stats import poisson as poisson # Poisson distribution

lambda_ = 3  # Average frequency of events

# Probability of exactly 2 events
k = 2
poisson_pmf = poisson.pmf(k, lambda_)
print(f"Probability of {k} events: {poisson_pmf:.4f}")
```