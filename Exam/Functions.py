############################
# @author Elias De Hondt   #
# @see https://eliasdh.com #
# @since 01/03/2024        #
############################

def rule_filter(row, min_len, max_len):
    length = len(row['antecedents']) + len(row['consequents'])
    return min_len <= length <= max_len

def get_item_list (string):
    items = string [1:-1]
    return items.split(';')

def plot_confidence_interval(population_size, sample_mean, sample_standard_deviation, degrees_freedom, plot_factor):
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

def LDA_coefficients(X,lda):
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

def trueFalse (confusion_matrix, columnnb=0):
    TP = confusion_matrix.iloc[columnnb][columnnb]
    print('TP', TP)
    TN = np.diag(confusion_matrix).sum() - TP
    print('TN:', TN)
    FP = confusion_matrix.iloc[:, columnnb].sum() - TP
    print('FP:', FP)
    FN = confusion_matrix.iloc[columnnb,:].sum() - TP
    print('FN:', FN)
    return

def accuracy(confusion_matrix):
    return np.diag(confusion_matrix).sum()/confusion_matrix.sum().sum()

def precision(confusion_matrix):
    precision = []
    n = confusion_matrix.shape[1]
    for i in range(0,n):
        TP = confusion_matrix.iloc[i][i]
        precision = precision + [TP/confusion_matrix.iloc[:, i].sum()]
    return precision

def recall(confusion_matrix):
    recall = []
    n = confusion_matrix.shape[0]
    for i in range(0,n):
        TP = confusion_matrix.iloc[i][i]
        recall = recall + [TP/confusion_matrix.iloc[i, :].sum()]
    return recall

def f_measure(confusion_matrix, beta):
    precisionarray = precision(confusion_matrix)
    recallarray = recall(confusion_matrix)
    fmeasure=[]
    n = len(precisionarray)
    for i in range(0,n):
        p = precisionarray[i]
        r = recallarray [i]
        fmeasure = fmeasure + [((beta*beta+1)*p*r)/(beta*beta*p+r)]
    return fmeasure

def overview_metrieken(confusion_matrix, beta):
    overview_1 = np.transpose(precision (confusion_matrix))
    overview_2 = np.transpose(recall(confusion_matrix))
    overview_3 = np.transpose(f_measure(confusion_matrix,beta))
    overview_table=pd.DataFrame (data=np.array([overview_1, overview_2, overview_3]), columns=confusion_matrix.index)
    overview_table.index = ['precision', 'recall', 'fx']
    return[overview_table]

def positiverates(confusion_matrix):
    if (confusion_matrix.shape[0] == 2) & (confusion_matrix.shape[1] == 2):
        TPR = confusion_matrix.iloc[0][0]/confusion_matrix.iloc[0, :].sum()
        print('TPR', TPR)
        FPR = confusion_matrix.iloc[1][0]/confusion_matrix.iloc[1, :].sum()
        print('FPR', FPR)
    return


def plot_roc(y_true, y_score, title='ROC Curve', **kwargs):
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, roc_auc_score

    if 'pos_label' in kwargs:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score, pos_label=kwargs.get('pos_label'))
        auc = roc_auc_score(y_true, y_score)
    else:
        fpr, tpr, thresholds = roc_curve(y_true=y_true, y_score=y_score)
        auc = roc_auc_score(y_true, y_score)

    # Bereken de optimale cut-off met de Youden index methode
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]

    figsize = kwargs.get('figsize', (7, 7))
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.grid(linestyle='--')

    # plot de ROC curve
    ax.plot(fpr, tpr, color='darkorange', label='AUC: {}'.format(auc))
    ax.set_title(title)
    ax.set_xlabel('False Positive Rate (FPR)')
    ax.set_ylabel('True Positive Rate (TPR)')
    ax.fill_between(fpr, tpr, alpha=0.3, color='darkorange', edgecolor='black')

    # plot de classifier
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    # plot de optimale cut-off
    ax.scatter(fpr[optimal_idx], tpr[optimal_idx],
               label='optimal cutoff {:.2f} on ({:.2f},{:.2f})'.format(optimal_threshold, fpr[optimal_idx],
                                                                       tpr[optimal_idx]), color='red')
    ax.plot([fpr[optimal_idx], fpr[optimal_idx]], [0, tpr[optimal_idx]], linestyle='--', color='red')
    ax.plot([0, fpr[optimal_idx]], [tpr[optimal_idx], tpr[optimal_idx]], linestyle='--', color='red')

    ax.legend(loc='lower right')
    plt.show()