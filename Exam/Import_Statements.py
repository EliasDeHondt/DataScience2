############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 01/03/2024        #
############################

import math                                                   # Mathematical functions
import pandas as pd                                           # Data manipulation
import numpy as np                                            # Scientific computing
import matplotlib.pyplot as plt                               # Data visualization
from scipy.stats import binom as binomial                     # Binomial distribution
from scipy.stats import norm as normal                        # Normal distribution
from scipy.stats import poisson as poisson                    # Poisson distribution
from scipy.stats import t as student                          # Student distribution
from scipy.stats import chi2                                  # Chi-squared distribution
from scipy.stats import ttest_1samp                           # One-sample t-test
from scipy.stats import chisquare                             # Chi-squared test
from scipy.special import comb                                # Combinations
from mlxtend.frequent_patterns import apriori                 # Apriori algorithm
from mlxtend.frequent_patterns import fpgrowth                # FP-growth algorithm
from mlxtend.frequent_patterns import association_rules       # Association rules
from mlxtend.preprocessing import TransactionEncoder          # Transaction encoder


# Binomial distribution
# binomial.pmf(x,n,p) # x = number of successes | n = number of trials | p = probability of success
# binomial.cdf(x,n,p) # x = number of successes | n = number of trials | p = probability of success
# binomial.std(n,p) # n = number of trials | p = probability of success
# binomial.mean(n,p) # n = number of trials | p = probability of success

# Poisson distribution
# poisson.pmf(x,λ) # x = number of events | λ = average number of events
# poisson.cdf(x,λ) # x = number of events | λ = average number of events
# poisson.std(λ) # λ = average number of events
# poisson.mean(λ) # λ = average number of events

# Normal distribution
# normal.cdf(x,loc,scale) # x = value | loc = average | scale = standard deviation
# normal.ppf(x,loc,scale) # x = cumulative probability | loc = average | scale = standard deviation
# normal.std(loc,scale) # loc = average | scale = standard deviation
# normal.mean(loc,scale) # loc = average | scale = standard deviation
# normal.ppf(0.025,loc,scale) # loc = average | scale = standard deviation