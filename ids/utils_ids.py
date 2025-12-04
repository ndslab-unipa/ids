import numpy as np
import pandas as pd
        

def binarize(y, train=True) -> list:
    """
    Binarizes labels for binary classification:
    - BENIGN → 0
    - Any attack (any label not BENIGN or Unknown) → 1

    Parameters
    ----------
    y : pandas.DataFrame
        DataFrame containing a 'Label' column.
    train : bool, optional (default=True)
        If True, converts labels to integers.

    Returns
    -------
    pandas.DataFrame
        A binarized version of the input DataFrame (0/1 labels).
    """
    # make a copy of y to prevent labels loss
    y_bin = y.copy(deep=True)

    # where labels is not BENIGN then 1 (Attack)
    y_bin['Label'] = np.where((y_bin['Label'] != 'BENIGN')&(y_bin['Label'] != 'Unknown'), 1, y_bin['Label'])
    #y_bin['Label'] = np.where(y_bin['Label'] != 'BENIGN', 1, y_bin['Label'])

    # where labels is BENING then 0 (Not Attack)
    y_bin['Label'] = np.where(y_bin['Label'] == 'BENIGN', 0, y_bin['Label'])

    # convert datatype to integer
    if train:
        y_bin = y_bin.astype('int')

    return y_bin
    
    
def ocsvm_binarize(y) -> list:
    """
    Binarizes labels for One-Class SVM training:
    - BENIGN → 1   (normal)
    - Attack → -1  (anomaly)

    Parameters
    ----------
    y : pandas.DataFrame
        DataFrame with a 'Label' column.

    Returns
    -------
    pandas.DataFrame
        A binarized DataFrame with labels +1 (normal) and -1 (anomaly).
    """
    # make a copy of y to prevent labels loss
    y_bin = y.copy(deep=True)

    # where labels is not BENIGN then 1 (Attack)
    y_bin['Label'] = np.where(y_bin['Label'] != 'BENIGN', -1, y_bin['Label'])

    # where labels is BENING then 0 (Not Attack)
    y_bin['Label'] = np.where(y_bin['Label'] == 'BENIGN', 1, y_bin['Label'])

    # convert datatype to integer
    y_bin = y_bin.astype('int')

    return y_bin

def entropy(proba, log_proba) -> float:
    """
    Computes entropy for each sample using the formula:
        H = - Σ p(x) * log(p(x))

    Parameters
    ----------
    proba : array-like
        Predicted probabilities.
    log_proba : array-like
        Log-probabilities corresponding to `proba`.

    Returns
    -------
    list
        A list containing the entropy value for each sample.
    """
    entropy = [0 for _ in range(len(proba))]
    for i in range(len(proba)):
        for j in range(len(proba[i])):
            if log_proba[i][j] != -np.inf:
                entropy[i] += proba[i][j] * log_proba[i][j]
        entropy[i] = entropy[i] * -1 
            
    return entropy

def entropy_single(proba, log_proba) -> float:
    """
    Computes entropy for a single probability distribution.

    Parameters
    ----------
    proba : array-like
        Probabilities.
    log_proba : array-like
        Log-probabilities.

    Returns
    -------
    float
        The entropy value.
    """
    entropy = 0
    for i in range(len(proba)):
        if log_proba[i] != -np.inf:
            entropy += proba[i] * log_proba[i]
    entropy = entropy * -1 
            
    return entropy

def max_proba(max, proba):
    """
    Multiplies each element in `proba` by the corresponding value in `max`.

    Parameters
    ----------
    max : array-like
        Maximum values to apply.
    proba : array-like
        Probabilities to modify.

    Returns
    -------
    array-like
        The modified probability array.
    """
    for i in range(len(max)):
        try:
            proba[i] = max[i] * proba[i]
        except Exception:
            proba[i] = 0 * max[i] 

    return proba

def weighted_proba(proba, classes, weight):
    """
    Applies class-specific weights to predicted probabilities.

    Parameters
    ----------
    proba : ndarray
        Probability matrix of shape (n_samples, n_classes).
    classes : list or array
        Class labels in order of prediction.
    weight : dict
        Mapping {class_label: weight}.

    Returns
    -------
    ndarray
        Weighted probability matrix.
    """
    result = np.zeros((len(proba), len(proba[0])))
    for i in range(len(proba)):
        for j in range(len(proba[i])):
            result[i][j] = weight[classes[j]] * proba[i][j]
            
    return result