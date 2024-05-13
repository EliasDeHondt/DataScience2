############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 01/03/2024        #
############################

import math  # type: ignore                                                           # Mathematical functions
import pandas as pd  # type: ignore                                                   # Data manipulation
import numpy as np  # type: ignore                                                    # Scientific computing
import matplotlib.pyplot as plt  # type: ignore                                       # Data visualization
from scipy.stats import binom as binomial  # type: ignore                             # Binomial distribution
from scipy.stats import norm as normal  # type: ignore                                # Normal distribution
from scipy.stats import poisson as poisson  # type: ignore                            # Poisson distribution
from scipy.stats import t as student  # type: ignore                                  # Student distribution
from scipy.stats import chi2  # type: ignore                                          # Chi-squared distribution
from scipy.stats import ttest_1samp  # type: ignore                                   # One-sample t-test
from scipy.stats import chisquare  # type: ignore                                     # Chi-squared test
from scipy.special import comb  # type: ignore                                        # Combinations
from mlxtend.frequent_patterns import apriori  # type: ignore                         # Apriori algorithm
from mlxtend.frequent_patterns import fpgrowth  # type: ignore                        # FP-growth algorithm
from mlxtend.frequent_patterns import association_rules  # type: ignore               # Association rules
from mlxtend.preprocessing import TransactionEncoder  # type: ignore                  # Transaction encoder
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # type: ignore  # Discriminant Analysis
from tensorflow import keras  # type: ignore                                          # Deep Learning library
from tensorflow.keras import Model  # type: ignore                                    # Model class
from tensorflow.keras.layers import Input, Dense, BatchNormalization  # type: ignore  # Layers
from tensorflow.keras.utils import to_categorical  # type: ignore                     # One-hot encoding
from tensorflow. keras.optimizers import Adam  # type: ignore                         # Optimizer
from livelossplot import PlotLossesKeras  # type: ignore                              # Live plot
from keras.src.optimizers import RMSprop  # type: ignore                              # Optimizer
from sklearn.model_selection import train_test_split  # type: ignore                  # Train-test split
import warnings  # type: ignore                                                       # Disable warnings
from Resources.Functions import *  # type: ignore                                     # Custom functions
warnings.filterwarnings("ignore")

# Probability Mass Function (PMF)
# Cumulative Distribution Function (CDF)
# The Percentile Point Function (PPF)

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
# normal.cdf(x,loc,scale) # x = value | loc = average number | scale = standard deviation
# normal.ppf(y,loc,scale) # y = cumulative probability | loc = average number | scale = standard deviation
# normal.std(loc,scale) # loc = average number | scale = standard deviation
# normal.mean(loc,scale) # loc = average number | scale = standard deviation