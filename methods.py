"""The methods used to run the Pharmacogenomic ML pipeline.

It includes each of the methods necessary for preprocessing, normalization,
feature selection, domain adaptation and drug resistance prediction

"""

import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn.feature_selection import SelectKBest, SelectPercentile, f_regression, mutual_info_regression, SelectFromModel, VarianceThreshold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn import linear_model
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import QuantileTransformer
from functools import reduce
from jive.AJIVE import AJIVE
from jive.PCA import PCA


def combine(ge, dr, drug, metric='AUC_IC50'):
    """Combines drug resistance and gene expression data

    This method puts gene expression and drug resistance data together into one dataframe,,
    drug resistance is incorporated using the defined metric

    Args:
        ge (`pandas.DataFrame`): gene expression data
        dr (`pandas.DataFrame`): drug resistance data
        drug (str): drug name
        metric (str): determines the drug resistance measure that will be used.

    Returns:
        A pandas dataframe containing both gene expression and drug resistance measurements
        for the given drug.
    """
    return ge.join(dr[dr['Drug_name'] == drug][['CCL', metric]].set_index('CCL'), how='right')


def pre(data, p = 0.1, t = 4):
    """Applies pre-processing to data

    Performs pre-processing on the gene expression data and returns a dataframe containing the
    selected genes. This pre-processing is used to find unexpressed genes that a
    microarray could detect as background noise.

    Args:
        data (`pandas.DataFrame`): contains the data for which pre-processing will be made, rows should be
            cancer cell lines and columns should be genes.
        t (float): determines the threshold below which genes are considered to be unexpressed.
        p (float): is in the range ]0,1] and determines what is the minimum percentage of the CCLs that
            needs to be expressed. If the actual percentage is smaller then that specific gene will be dropped.

    Returns:
        A Pandas dataframe with only the selected genes included.
    """
    #We look for elements below the threshold
    data = data.fillna(0)
    under = (data.to_numpy()>t).T.astype(np.int8)

    #If p is bigger than one we consider it the number of CCLs in which a gene
    #needs to be express to be selected. Otherwise it's the percentage of the total.
    if p < 1:
        n = [np.count_nonzero(i) > p * data.shape[0] for i in under]
    else:
        n = [np.count_nonzero(i) > p for i in under]

    names = {data.keys()[k]:v for k, v in enumerate(n)}
    index = [k for k,v in names.items() if v]

    return data[index]

def norm(model, ge):
    """Applies normalization to data

    Normalizes the given data using the given model then returns the normalized data.

    Args:
        model(`sklearn.base.TransformerMixin`): a normalization method on which fit_transform can be called
        ge(`pandas Dataframe` or `numpy array`): the data that should be normalized

    Returns:
        A pandas DataFrame or numpy aarray containing the transformed data.

    """
    if model == QuantileTransformer:
        model = model(output_distribution='normal')
    else:
        model = model()
    return model.fit_transform(ge)

def group_ajive(data,joint):
    """Performs Angle-based Joint and Individual Variation (AJIVE), a type of Domain Adaptation

    This method performs domain adaptation by learning the joint space projections from the train
    set and then projecting the test set onto the joint space. For this, the methods `ajive` and
    `ajive_predict` are used respectively.


    Args:
        data (`pandas DataFrame`): the data for which domain adaptation is being performed
        joint (int): determines the rank of the joint space of the domains
    """
    result = {}
    train, model = ajive(data['train'],joint)
    result['train'] = train
    result['test'] = ajive_predict(model, data['test'])
    return result

def ajive(data, joint):
    #Only overlapping cancer cell lines will be used
    blocks = {j:data.loc[j] for j in data.index.levels[0]}
    ccls = {k:v.index for k,v in blocks.items()}
    overlap = reduce(np.intersect1d, ccls.values())
    blocks = {k:v.loc[overlap] for k, v in blocks.items()}

    #Calculates the number of singular values that should be used for the PCA
    #projections of the blocks
    init = {k:svals(v, joint) for k,v in blocks.items()}

    model = AJIVE(init, joint_rank=joint)
    model.fit(blocks)
    result = ajive_predict(model, data)

    return result, model

def ajive_predict(model, data):
    mod = {}
    for i in model.blocks.keys():
        for j, ele in enumerate(model.blocks[i].joint.predict_scores(data.loc[i])):
            mod[(i, data.loc[i].index[j])] = list(ele)
    result = pd.DataFrame(mod.values(), index=mod.keys())

    return result


def svals(data, joint):
    init = 0
    last = 0
    previous_step = 0

    for i in PCA().fit(data).svals_:
        if last-i < 0.2*previous_step:
            init += 1
        previous_step = last-i
        last = i
    return max(init, joint)


def fs(model, X_train, X_test, y, n=0, tuning=None):
    """Selects features using the given model

    This method selects the best features based on a given model.

    Args:
        model (estimator object): this is the method that will
            be used to determine the best features.
        X_train (`pandas DataFrame` or `numpy array`): train data on which
            the best features will be selected.
        X_test (`pandas DataFrame` or `numpy array`): test data, this will only be
            transformed by selecting the specified features.
        y (`pandas DataFrame` or `numpy array`): labels of the train set
        n (float): determines what percentage of the features will be selected
        tuning (`classes.tuning`): defines a hyper parameter search space over which the feature selection
            method can be optimized.

    Returns:
        Two DataFrames, each containing the train and test set with only the selected features
        and a dictionary containing the selected features and their respective weights.

    """
    # This is used for model-based feature selection
    if model in [DecisionTreeRegressor, RandomForestRegressor, linear_model.ElasticNet, linear_model.Lasso]:
        model = model()
        model.fit(X_train, y)

        if tuning != None:
            r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring, iid=False)
            r.fit(X_train, y)
            model = r.best_estimator_
        n = n*X_train.shape[1]


        fs = SelectFromModel(estimator = model, max_features = int(n), threshold=-np.inf)
        X_train = fs.fit_transform(X_train, y)

    # This is used for variance threshold selection
    elif model == VarianceThreshold:
        fs = VarianceThreshold(n)
        X_train = fs.fit_transform(X_train)

    # This is used for selectkbest and selectpercentile
    else:
        n = int(n*X_train.shape[1])
        fs = SelectKBest(model, k = n)


        X_train = fs.fit_transform(X_train, y)

    var = fs.get_support()
    index = [i for i, x in enumerate(var) if x]
    X_test = np.apply_along_axis(lambda x: x[index], 1, X_test)
    print('After fs '+str(X_test.shape) +' ' +str(X_train.shape))
    return X_train, X_test, var

def jump(domain, n, data):
    result = []
    for i, ele in enumerate(data):
        result.append(ele)
        for j in range(0, domain):
            result.append(0)
        result.append(ele)
        for j in range(0, n-domain-1):
            result.append(0)
    return result

def feda(domains):
    """Performs Frustratingly Easy Domain Adaptation, a type of Domain Adaptation.

    This method performs domain adaptation on the drug data by using Frustratingly Easy
    Domain Adaptation.

    Args:
        domains(`pandas DataFrame`): data on which FEDA will be performed, it should
            contain the domain on the 0 level of the keys

    Returns:
        A DataFrame with the transformed data

    """

    n = len(domains)

    samples = 0
    for i in domains:
        samples += i.shape[0]

    features = domains[0].shape[1]*(n+1)

    new = np.zeros(features)
    for i, data in enumerate(domains):
        for j in data:
            new = np.vstack([new, jump(i, n, j)])

    return new[1:]


def drp(model, X, y, tuning=None):
    """Trains model on drug data.

    Trains the model specified on `model` on the given data by using the specified method.
    It then stores the model on the drug's `model` variable.

    Args:
        model (estimator object): model that will be trained on the data
        X (`pandas DataFrame` or `numpy array`): features of the samples
            on which the model is trained
        y (`pandas DataFrame` or `numpy array`): labels of the samples on which
            the model is trained
        tuning (`classes.tuning`): hyper parameter search space over which
            the model will be optimized. If not specified default values will
            be used.
    Returns:
        The fitted model
    """

    if tuning != None:
        r = RandomizedSearchCV(model, tuning.space, n_iter = tuning.iterations, n_jobs=tuning.jobs, cv= tuning.cv, scoring = tuning.scoring, iid=False)
        r.fit(X, y)
        drp = r.best_estimator_
    else:
        drp = model.fit(X, y)
    return drp
