############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 01/03/2024        #
############################

def rule_filter(row, min_len, max_len):
    """
    Filterfunctie voor regelgebaseerde classificatieregels.

    Parameters:
        row (pandas.Series): Een rij van de regelgebaseerde classificatieregel.
        min_len (int): Minimale lengte van de gecombineerde antecedenten en consequenten.
        max_len (int): Maximale lengte van de gecombineerde antecedenten en consequenten.

    Returns:
        bool: True als de lengte binnen het opgegeven bereik valt, anders False.
    """
    length = len(row['antecedents']) + len(row['consequents'])
    return min_len <= length <= max_len


def get_item_list(string):
    """
    Converteert een door puntkomma gescheiden string naar een lijst van items.

    Parameters:
        string (str): Door puntkomma gescheiden string.

    Returns:
        list: Lijst van items.
    """
    items = string [1:-1]
    return items.split(';')


def no_outliers(data):
    """
    Verwijder outliers uit een dataset.

    Parameters:
        data (pandas.Series): Dataset.

    Returns:
        pandas.Series: Dataset zonder outliers.
    """
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    I = Q3 - Q1
    low = Q1 - 1.5 * I
    high = Q3 + 1.5 * I
    outliers = data[(data < low) | (data > high)]

    print("Low: ", low)
    print("High:", high)
    print("Len: ", len(data))
    print("Outliers:", outliers.values, "\n")
    return data[(data >= low) & (data <= high)]


def plot_confidence_interval(population_size, sample_mean, sample_standard_deviation, degrees_freedom, plot_factor):
    """
    Plot een confidence interval met behulp van de t-verdeling.

    Parameters:
        population_size (int): Grootte van de populatie.
        sample_mean (float): Gemiddelde van de steekproef.
        sample_standard_deviation (float): Standaardafwijking van de steekproef.
        degrees_freedom (int): Vrijheidsgraden van de t-verdeling.
        plot_factor (float): Factor voor de grootte van de plot.

    Returns:
        None
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
    y_axis = student.pdf(x_axis, degrees_freedom, loc=sample_mean, scale=sample_standard_deviation / np.sqrt(population_size))

    plt.plot(x_axis, y_axis, label='t-distribution')
    plt.axvline(lower_bound, color='red', linestyle='--', label='Lower Bound')
    plt.axvline(upper_bound, color='blue', linestyle='--', label='Upper Bound')
    plt.axvline(sample_mean, color='green', linestyle='-', label='Sample Mean')

    # Mark the confidence interval
    plt.fill_betweenx(y_axis, lower_bound, upper_bound, where=(x_axis >= lower_bound) & (x_axis <= upper_bound), color='orange', label='Confidence Interval')

    plt.title('Confidence Interval Plot')
    plt.xlabel('Sample Mean')
    plt.ylabel('Probability Density Function')
    plt.legend()
    plt.grid(True)
    plt.show()


def LDA_coefficients(X, lda):
    """
    Bereken de coëfficiënten voor lineaire discriminantanalyse (LDA).

    Parameters:
        X (pandas.DataFrame): Kenmerkenset.
        lda (sklearn.discriminant_analysis.LinearDiscriminantAnalysis): Getraind LDA-model.

    Returns:
        pandas.DataFrame: DataFrame met de LDA-coëfficiënten.
    """
    import numpy as np
    import pandas as pd

    nb_col = X.shape[1]
    matrix= np.zeros((nb_col+1,nb_col), dtype=int)
    Z=pd.DataFrame(data=matrix,columns=X.columns)
    for j in range(0,nb_col):
        Z.iloc[j,j] = 1
    LD = lda.transform(Z)
    nb_funct= LD.shape[1]
    resultaat = pd.DataFrame()
    index = ['const']
    for j in range(0,LD.shape[0]-1):
        index = np.append(index,'C'+str(j+1))
    for i in range(0,LD.shape[1]):
        coef = [LD[-1][i]]
        for j in range(0,LD.shape[0]-1):
            coef = np.append(coef,LD[j][i]-LD[-1][i])
        result = pd.Series(coef)
        result.index = index
        column_name = 'LD' + str(i+1)
        resultaat[column_name] = result
    return resultaat


def trueFalsef(confusion_matrix, columnnb=0):
    """
    Bereken True Positives (TP), True Negatives (TN), False Positives (FP) en False Negatives (FN) voor een gegeven
    confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.
        columnnb (int): Index van de kolom waarvoor de berekening moet worden uitgevoerd.

    Returns:
        None
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


def accuracyf(confusion_matrix):
    """
    Bereken de nauwkeurigheid (accuracy) van een classifier op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.

    Returns:
        float: Nauwkeurigheid van de classifier.
    """
    import numpy as np

    return np.diag(confusion_matrix).sum() / confusion_matrix.sum().sum()


def precisionf(confusion_matrix):
    """
    Bereken de precisie (precision) voor elke klasse op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.

    Returns:
        list: Precisie voor elke klasse.
    """
    precision = []
    n = confusion_matrix.shape[1]
    for i in range(0, n):
        TP = confusion_matrix.values[i][i]
        precision = precision + [TP / confusion_matrix.values[:, i].sum()]
    return precision


def recallf(confusion_matrix):
    """
    Bereken de recall (recall) voor elke klasse op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.

    Returns:
        list: Recall voor elke klasse.
    """
    recall = []
    n = confusion_matrix.shape[0]
    for i in range(0, n):
        TP = confusion_matrix.values[i][i]
        recall = recall + [TP / confusion_matrix.values[i, :].sum()]
    return recall


def f_measuref(confusion_matrix, beta):
    """
    Bereken de F-maat (F-measure) voor elke klasse op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.
        beta (float): Beta-waarde voor de F-maat.

    Returns:
        list: F-maat voor elke klasse.
    """
    precisionarray = precisionf(confusion_matrix)
    recallarray = recallf(confusion_matrix)
    fmeasure = []
    n = len(precisionarray)
    for i in range(0, n):
        p = precisionarray[i]
        r = recallarray[i]
        fmeasure = fmeasure + [((beta * beta + 1) * p * r) / (beta * beta * p + r)]
    return fmeasure


def overview_metrieken(confusion_matrix, beta):
    """
    Genereer een overzichtstabel van precisie, recall en F-maat voor elke klasse op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.
        beta (float): Beta-waarde voor de F-maat.

    Returns:
        list: Lijst met overzichtstabellen voor precisie, recall en F-maat.
    """
    import numpy as np
    import pandas as pd

    overview_1 = np.transpose(precisionf(confusion_matrix))
    overview_2 = np.transpose(recallf(confusion_matrix))
    overview_3 = np.transpose(f_measuref(confusion_matrix, beta))
    overview_table = pd.DataFrame(data=np.array([overview_1, overview_2, overview_3]), columns=confusion_matrix.index)
    overview_table.index = ['precision', 'recall', 'fx']
    return [overview_table]


def positiveratesf(confusion_matrix):
    """
    Bereken de positieve en negatieve tarieven (rates) voor een tweeklassen classifier op basis van de confusion matrix.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.

    Returns:
        None
    """
    if (confusion_matrix.shape[0] == 2) & (confusion_matrix.shape[1] == 2):
        TPR = confusion_matrix.values[0][0] / confusion_matrix.values[0, :].sum()
        print('TPR', TPR)
        FPR = confusion_matrix.values[1][0] / confusion_matrix.values[1, :].sum()
        print('FPR', FPR)
    return


def plot_rocf(y_true, y_score, title='ROC Curve', **kwargs):
    """
    Plot de Receiver Operating Characteristic (ROC) curve voor een classifier.

    Parameters:
        y_true (array-like): Ware labels.
        y_score (array-like): Scores die zijn toegewezen aan de labels.
        title (str): Titel van de plot.
        **kwargs: Extra argumenten voor de plot.

    Returns:
        None
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


def evaluate_classifier(confusion_matrix, beta=1, threshold=0.9):
    """
    Evalueer een classifier op basis van de gegeven confusion matrix en de opgegeven threshold.

    Parameters:
        confusion_matrix (pandas.DataFrame): Confusion matrix.
        beta (float): Beta-waarde voor de F-maat.
        threshold (float): Drempelwaarde voor evaluatie.

    Returns:
        None
    """
    import warnings
    import numpy as np
    warnings.filterwarnings("ignore")  # Ignoring future dependency warning.

    # Calculate TP, TN for each class
    TP = np.diag(confusion_matrix).sum()
    TN = np.sum(np.diag(confusion_matrix)) - TP

    # Calculate accuracy
    accuracy = (TP + TN) / confusion_matrix.sum().sum()

    # Calculate precision
    n = confusion_matrix.shape[1]
    precision = [np.diag(confusion_matrix)[i] / np.sum(confusion_matrix.iloc[i, :]) if np.sum(
        confusion_matrix.iloc[i, :]) > 0 else 0 for i in range(0, n)]

    # Calculate recall
    n = confusion_matrix.shape[0]
    recall = [np.diag(confusion_matrix)[i] / np.sum(confusion_matrix.iloc[:, i]) if np.sum(
        confusion_matrix.iloc[:, i]) > 0 else 0 for i in range(0, n)]

    # Calculate F1-score
    f1_score = [((beta ** 2 + 1) * p * r) / ((beta ** 2 * p) + r) if (p + r) > 0 else 0 for p, r in
                zip(precision, recall)]

    # Evaluate classifier (threshold)
    if accuracy >= threshold and all(prec >= threshold for prec in precision) and all(
            rec >= threshold for rec in recall) and all(f1 >= threshold for f1 in f1_score):
        print(f"This is a good classifier with a threshold of {threshold}")
    else:
        print(f"This is a bad classifier with a threshold of {threshold}")

    warnings.filterwarnings("default") # Ignoring future dependency warning.

def find_best_threshold(y_true, y_score, beta=1):
    """
    Zoek de beste drempelwaarde voor een classifier op basis van de gegeven labels en scores.

    Parameters:
        y_true (array-like): Ware labels.
        y_score (array-like): Scores die zijn toegewezen aan de labels.
        beta (float): Beta-waarde voor de F-maat.

    Returns:
        float: Beste drempelwaarde.
    """
    from sklearn.metrics import precision_recall_curve

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1_score = [(beta ** 2 + 1) * p * r / ((beta ** 2 * p) + r) if (p != 0 and r != 0) else 0 for p, r in zip(precision, recall)]
    optimal_idx = f1_score.index(max(f1_score))
    return thresholds[optimal_idx]