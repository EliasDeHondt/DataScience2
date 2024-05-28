############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 01/03/2024        #
############################

def rule_filter(row, min_len, max_len):
    """
    Filters a row based on the combined length of its 'antecedents' and 'consequents'.

    Parameters:
    - row (dict): A dictionary representing a row, containing 'antecedents' and 'consequents'.
    - min_len (int): The minimum length for the combined 'antecedents' and 'consequents'.
    - max_len (int): The maximum length for the combined 'antecedents' and 'consequents'.

    Returns:
    - bool: True if the length of 'antecedents' + 'consequents' is within the specified range, otherwise False.

    Usage:
    filtered_row = rule_filter(row, 2, 5)
    """
    length = len(row['antecedents']) + len(row['consequents'])
    return min_len <= length <= max_len


def get_item_list(string):
    """
    Converts a string representation of a list (where items are separated by semicolons and enclosed in square brackets)
    into an actual Python list.

    Parameters:
    - string (str): A string representing a list, e.g., '[item1;item2;item3]'.

    Returns:
    - list: A list of items extracted from the input string.

    Usage:
    items = get_item_list("[item1;item2;item3]")
    """
    items = string[1:-1]
    return items.split(';')


def no_outliers(data):
    """
    Removes outliers from a dataset based on the Interquartile Range (IQR) method.
    This function calculates the first (Q1) and third quartiles (Q3), determines the interquartile range (IQR),
    and filters out any data points that lie beyond 1.5 times the IQR from Q1 or Q3.

    Parameters:
    - data (pd.Series): A pandas Series containing numerical data from which to remove outliers.

    Returns:
    - pd.Series: The input data with outliers removed.

    Usage:
    clean_data = no_outliers(data_series)
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    i = Q3 - Q1
    low = Q1 - 1.5 * i
    high = Q3 + 1.5 * i
    outliers = data[(data < low) | (data > high)]

    print("Low: ", low)
    print("High:", high)
    print("Len: ", len(data))
    print("Outliers:", outliers.values, "\n")
    return data[(data >= low) & (data <= high)]


def plot_confidence_interval(population_size, sample_mean, sample_standard_deviation, degrees_freedom, plot_factor):
    """
    Plots a confidence interval for a given sample mean and standard deviation, assuming a t-distribution.
    This function visualizes the interval on a graph with the sample mean,
    lower and upper bounds, and the t-distribution curve.

    Parameters:
    - population_size (int): The size of the population/sample.
    - sample_mean (float): The mean of the sample.
    - sample_standard_deviation (float): The standard deviation of the sample.
    - degrees_freedom (int): Degrees of freedom, typically the sample size minus one.
    - plot_factor (float): The factor used to scale the margin of error.

    Returns:
    - None: This function plots a graph directly.

    Usage:
    plot_confidence_interval(100, 50, 10, 99, 1.96)
    """
    from matplotlib import pyplot as plt
    import numpy as np
    from scipy.stats import t as student

    margin_of_error = plot_factor * sample_standard_deviation / np.sqrt(population_size)
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    # Plotting the confidence interval
    plt.figure(figsize=(10, 6))
    x_axis = np.linspace(sample_mean - 3 * sample_standard_deviation, sample_mean + 3 * sample_standard_deviation, 1000)
    y_axis = student.pdf(x_axis, degrees_freedom, loc=sample_mean,
                         scale=sample_standard_deviation / np.sqrt(population_size))

    plt.plot(x_axis, y_axis, label='t-distribution')
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound, color='blue', linestyle='--', label='Upper Bound')
    plt.axvline(sample_mean, color='green', linestyle='-', label='Sample Mean')

    # Mark the confidence interval
    plt.fill_betweenx(y_axis, lower_bound, upper_bound, where=(x_axis >= lower_bound) & (x_axis <= upper_bound),
                      color='orange', label='Confidence Interval')

    plt.title('Confidence Interval Plot')
    plt.xlabel('Sample Mean')
    plt.ylabel('Probability Density Function')
    plt.legend()
    plt.grid(True)
    plt.show()


def LDA_coefficients(x, lda):
    """
    Computes the Linear Discriminant Analysis (LDA) coefficients for each class.
    This function transforms the input data using LDA and calculates the coefficients for the discriminant functions.

    Parameters:
    - X (pd.DataFrame): Input features for LDA.
    - lda (object): Trained LDA model.

    Returns:
    - pd.DataFrame: A dataframe containing the LDA coefficients for each class.

    Usage:
    coefficients = LDA_coefficients(X, lda_model)
    """
    import numpy as np
    import pandas as pd

    nb_col = x.shape[1]
    matrix = np.zeros((nb_col + 1, nb_col), dtype=int)
    Z = pd.DataFrame(data=matrix, columns=x.columns)
    for j in range(0, nb_col):
        Z.iloc[j, j] = 1
    LD = lda.transform(Z)
    # nb_funct = LD.shape[1]
    resultaat = pd.DataFrame()
    index = ['const']
    for j in range(0, LD.shape[0] - 1):
        index = np.append(index, 'C' + str(j + 1))
    for i in range(0, LD.shape[1]):
        coef = [LD[-1][i]]
        for j in range(0, LD.shape[0] - 1):
            coef = np.append(coef, LD[j][i] - LD[-1][i])
        result = pd.Series(coef)
        result.index = index
        column_name = 'LD' + str(i + 1)
        resultaat[column_name] = result
    return resultaat


def trueFalsef(confusion_matrix, columnnb=0):
    """
    Calculates and prints the True Positive (TP), True Negative (TN), False Positive (FP),
    and False Negative (FN) rates from a confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.
    - columnnb (int): Index of the class for which to compute the metrics (default is 0).

    Returns:
    - None: This function prints the metrics directly.

    Usage:
    trueFalsef(confusion_matrix, columnnb=0)
    """
    import numpy as np

    TP = confusion_matrix.values[columnnb][columnnb]
    print('TP', TP)
    TN = np.diag(confusion_matrix).sum() - TP
    print('TN:', TN)
    FP = confusion_matrix.values[:, columnnb].sum() - TP
    print('FP:', FP)
    FN = confusion_matrix.values[columnnb, :].sum() - TP
    print('FN:', FN)
    return


def accuracyf(matrix):
    """
    Calculates the overall accuracy from a confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.

    Returns:
    - float: The accuracy of the classification.

    Usage:
    accuracy = accuracyf(matrix)
    """
    import numpy as np

    return np.diag(matrix).sum() / matrix.sum().sum()


def precisionf(matrix):
    """
    Calculates the precision for each class from a confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.

    Returns:
    - list: A list of precision values for each class.

    Usage:
    precision = precisionf(matrix)
    """
    results = []
    n = matrix.shape[1]
    for i in range(0, n):
        TP = matrix.values[i][i]
        results = results + [TP / matrix.values[:, i].sum()]
    return results


def recallf(matrix):
    """
    Calculates the recall for each class from a confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.

    Returns:
    - list: A list of recall values for each class.

    Usage:
    recall = recallf(matrix)
    """
    results = []
    n = matrix.shape[0]
    for i in range(0, n):
        TP = matrix.values[i][i]
        results = results + [TP / matrix.values[i, :].sum()]
    return results


def f_measuref(matrix, beta):
    """
    Calculates the F-measure (F1 score) for each class from a confusion matrix using a specified beta value.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.
    - beta (float): The beta value to weigh precision and recall (default is 1, which gives the F1 score).

    Returns:
    - list: A list of F-measure values for each class.

    Usage:
    f_measure = f_measuref(matrix, beta=1)
    """
    precisionarray = precisionf(matrix)
    recallarray = recallf(matrix)
    fmeasure = []
    n = len(precisionarray)
    for i in range(0, n):
        p = precisionarray[i]
        r = recallarray[i]
        fmeasure = fmeasure + [((beta * beta + 1) * p * r) / (beta * beta * p + r)]
    return fmeasure


def overview_metrieken(matrix, beta):
    """
    Provides an overview of classification metrics (precision, recall, F-measure) for each class in a confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.
    - beta (float): The beta value to weigh precision and recall for the F-measure (default is 1).

    Returns:
    - list: A list containing a dataframe with precision, recall, and F-measure for each class.

    Usage:
    metrics_overview = overview_metrieken(matrix, beta=1)
    """
    import numpy as np
    import pandas as pd

    overview_1 = np.transpose(precisionf(matrix))
    overview_2 = np.transpose(recallf(matrix))
    overview_3 = np.transpose(f_measuref(matrix, beta))
    overview_table = pd.DataFrame(data=np.array([overview_1, overview_2, overview_3]), columns=matrix.index)
    overview_table.index = ['precision', 'recall', 'fx']
    return [overview_table]


def positiveratesf(matrix):
    """
    Calculates and prints the True Positive Rate (TPR) and False Positive Rate (FPR) for a
    binary classification confusion matrix.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for binary classification.

    Returns:
    - None: This function prints the TPR and FPR directly.

    Usage:
    positiveratesf(matrix)
    """
    if (matrix.shape[0] == 2) & (matrix.shape[1] == 2):
        TPR = matrix.values[0][0] / matrix.values[0, :].sum()
        print('TPR', TPR)
        FPR = matrix.values[1][0] / matrix.values[1, :].sum()
        print('FPR', FPR)
    return


def plot_rocf(y_true, y_score, title='ROC Curve', **kwargs):
    """
    Plots the Receiver Operating Characteristic (ROC) curve and calculates the Area Under the Curve (AUC) for a set of
    true labels and predicted scores. It also highlights the optimal threshold on the ROC curve.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_score (array-like): Target scores, probability estimates of the positive class.
    - title (str): Title of the ROC curve plot (default is 'ROC Curve').
    - **kwargs (dict): Additional keyword arguments for customizing the plot.

    Returns:
    - None: This function plots the ROC curve directly.

    Usage:
    plot_rocf(y_true, y_score, title='My ROC Curve', pos_label=1, figsize=(8, 8))
    """
    import numpy as np
    import matplotlib.pyplot as plt

    from sklearn.metrics import roc_curve, roc_auc_score

    if 'pos_label' in kwargs:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=kwargs.get('pos_label'))
        auc = roc_auc_score(y_true, y_score)
    else:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        auc = roc_auc_score(y_true, y_score)

    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    figsize = kwargs.get('figsize', (7, 7))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(linestyle='--')

    ax.plot(fpr, tpr, color='darkorange', label='AUC: {}'.format(auc))
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange', edgecolor='black')

    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               label='optimal cutoff {:.2f} on ({:.2f},{:.2f})'.format(optimal_threshold, fpr[optimal_idx],
                                                                       tpr[optimal_idx]), color='red')
    ax.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], linestyle='--', color='red')
    ax.plot([0, fpr[optimal_idx]], [tpr[optimal_idx], tpr[optimal_idx]], linestyle='--', color='red')

    ax.legend(loc='lower right')
    plt.show()


def evaluate_classifier(matrix, beta=1, threshold=0.9):
    """
    Evaluates a classifier based on its confusion matrix and specified threshold for various metrics.
    This function checks if the classifier meets the threshold criteria for accuracy, precision, recall, and F1-score.

    Parameters:
    - confusion_matrix (pd.DataFrame): Confusion matrix for the classification.
    - beta (float): The beta value to weigh precision and recall for the F-measure (default is 1).
    - threshold (float): The threshold value to evaluate the metrics (default is 0.9).

    Returns:
    - None: This function prints the evaluation result directly.

    Usage:
    evaluate_classifier(confusion_matrix, beta=1, threshold=0.9)
    """
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")  # Ignoring future dependency warning.

    # Calculate TP, TN for each class
    TP = np.diag(matrix).sum()
    TN = np.sum(np.diag(matrix)) - TP

    # Calculate accuracy
    accuracy = (TP + TN) / matrix.sum().sum()

    # Calculate precision
    n = matrix.shape[1]
    precision = [np.diag(matrix)[i] / np.sum(matrix.iloc[i, :]) if np.sum(
        matrix.iloc[i, :]) > 0 else 0 for i in range(0, n)]

    # Calculate recall
    n = matrix.shape[0]
    recall = [np.diag(matrix)[i] / np.sum(matrix.iloc[:, i]) if np.sum(
        matrix.iloc[:, i]) > 0 else 0 for i in range(0, n)]

    # Calculate F1-score
    f1_score = [((beta ** 2 + 1) * p * r) / ((beta ** 2 * p) + r) if (p + r) > 0 else 0 for p, r in
                zip(precision, recall)]

    # Evaluate classifier (threshold)
    if accuracy >= threshold and all(prec >= threshold for prec in precision) and all(
            rec >= threshold for rec in recall) and all(f1 >= threshold for f1 in f1_score):
        print(f"This is a good classifier with a threshold of {threshold * 100}%")
    else:
        print(f"This is a bad classifier with a threshold of {threshold * 100}%")

    warnings.filterwarnings("default")  # Ignoring future dependency warning.


def categorize_variables(df):
    """
    Categorize columns in a DataFrame into potential dependent (categorical)
    and independent (numerical) variables for Discriminant Analysis.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the data to be analyzed.

    Returns:
    tuple: A tuple containing two lists:
        - dependent_vars (list): List of column names suitable as dependent variables (categorical).
        - independent_vars (list): List of column names suitable as independent variables (numerical).
    """
    import warnings
    import pandas as pd
    warnings.filterwarnings("ignore")  # Ignoring future dependency warning.

    independentVars = []
    dependentVars = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            independentVars.append(col)
        elif pd.api.types.is_categorical_dtype(df[col]) or pd.api.types.is_object_dtype(
                df[col]) or pd.api.types.is_bool_dtype(df[col]):
            dependentVars.append(col)

    warnings.filterwarnings("default")  # Ignoring future dependency warning.
    return independentVars, dependentVars


def find_best_threshold(y_true, y_score, beta=1):
    """
    Finds the optimal threshold for classification by maximizing the F1-score based on the precision-recall curve.

    Parameters:
    - y_true (array-like): True binary labels.
    - y_score (array-like): Target scores, probability estimates of the positive class.
    - beta (float): The beta value to weigh precision and recall for the F-measure (default is 1).

    Returns:
    - float: The optimal threshold that maximizes the F1-score.

    Usage:
    best_threshold = find_best_threshold(y_true, y_score, beta=1)
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_score = [(beta ** 2 + 1) * p * r / ((beta ** 2 * p) + r) if (p != 0 and r != 0) else 0 for p, r in
                zip(precision, recall)]
    optimal_idx = f1_score.index(max(f1_score))
    return thresholds[optimal_idx]


def obj_func(solution, weights):
    """
    Objective function for the Traveling Salesman Problem (TSP) that evaluates the quality of a solution.
    This function calculates the total distance traveled based on the given solution.

    Parameters:
    - solution (list): A list representing the order of cities to visit.
    - weights (list): A list of weights representing the distances between cities.

    Returns:
    - float: The total distance traveled based on the solution.

    Usage:
    distance = obj_func(solution, weights)
    """
    import math
    import numpy as np

    n = int(math.sqrt(len(solution)))
    leaveOK = np.array([sum(solution[i::n]) for i in range(n)])
    arriveOK = np.array([sum(solution[i * n:(i + 1) * n]) for i in range(n)])
    notStayingOK = sum(solution[0::n + 1])
    city, loop_length = 0, 0
    while loop_length < n + 1 and (loop_length := loop_length + 1):
        city = next((i for i in range(n) if solution[city * n + i]), 0)
        if not solution[city * n + city]: break
    return np.sum(solution * weights) if notStayingOK == 0 and all(arriveOK) and all(leaveOK) and loop_length == n \
        else 1000 * n + np.sum(solution * weights)
