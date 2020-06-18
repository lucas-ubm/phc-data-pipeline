"""A method to test a specific pipeline configuration
"""
from classes import drug
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error
from sklearn.feature_selection import f_regression, mutual_info_regression, VarianceThreshold
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MaxAbsScaler
from sklearn.preprocessing import QuantileTransformer
import config as c
import pandas as pd
#import xgboost as xgb
#import lightgbm as lgb

models = [RandomForestRegressor, SVR, KNeighborsRegressor, RandomForestRegressor, DecisionTreeRegressor, VarianceThreshold, f_regression, mutual_info_regression, linear_model.ElasticNet, linear_model.Lasso] #xgb.XGBRegressor, lgb.LGBMRegressor
models = {k.__name__:k for k in models}

norms = [StandardScaler, MinMaxScaler, MaxAbsScaler, QuantileTransformer]
norms = {k.__name__:k for k in norms}

def run(ge, fs, feda, model, drugs=1000, n=0, ajive = 0,fs_tuning=None, norm='', tuning=None, p = 0.01, t = 4, metric='AUC_IC50', test=None):
    """Runs a number of drugs through a specific configuration of the pipeline

    This method trains the models for a number of drugs and a specific configuration.

    Args:
        ge (dict): a dictionary where keys are the name of the datasets and value is
            True if the dataset will be used and False otherwise
        fs (str): name of the feature selection method being used
        feda (bool): boolean specifying if FEDA will be used
        model (str): name of the model that will be used for drug resistance prediction
        drugs (int): maximum number of drugs that should be selected
        n (float): parameter used for :func:`~classes.drug.fs`, percentage of features selected
        ajive (int): rank of the joint space of the domains used for :func:`~classes.drug.ajive`
        fs_tuning (`classes.tuning`): tuning space used for :func:`~classes.drug.fs`
        norm (str): determines the normalization method that will be used if empty no
            normalization will be performed.
        tuning (`classes.tuning`): tuning space used for :func:`~classes.drug.drp`
        p (float): parameter used for :func:`~classes.drug.pre`
        t (float): parameter used for :func:`~classes.drug.pre`
        metric (str): parameter used for :func:`~classes.drug.combine`
        test (float): parameter used for :func:`~classes.drug.split`

    Returns:
        Two dictionaries, both containing the name of the drug as key with one having the
        trained models as values and the other one having the scores obtained.

    """

    expression_data = {}
    drug_data = {}

    # Select used datasets
    for k,v in ge.items():
        if v:
            expression_data[k] = pd.read_csv(c.dir+k+'_cell_ge.csv').fillna(0).set_index('CCL')
            drug_data[k] = pd.read_csv(c.dir+k+'_poz_dr.csv').fillna(0)


    # Select drugs, first creating a set with all drug names
    names = set([item for sublist in [list(j['Drug_name'].unique()) for j in drug_data.values()] for item in sublist])

    for i in drug_data.keys():
        names = names & set(drug_data[i]['Drug_name'].unique())
    names = list(names)
    #Select either all of the intersection or the number of drugs asked for
    names = names[:min(len(names), drugs)]

    # Select feature selection and model
    fs = models[fs]
    model = models[model]()

    # Select normalization
    if norm in norms:
        norm = norms[norm]

    drugs = {}
    for i in names:
        ele = drug(i, expression_data, drug_data)
        if not norm=='':
            ele.norm(norm)
        ele.pre(p = p, t = t)
        ele.combine(metric = metric)
        ele.split(test = test)
        if not fs=='':
            ele.fs(fs, n=n, tuning=fs_tuning)

        #You should not use two different types of DA
        if feda and ajive == 0:
            ele.feda()
        if ajive > 0 and not feda:
            ele.ajive(ajive)

        ele.train(model, tuning=tuning)
        ele.metrics([r2_score, mean_absolute_error, mean_squared_error, median_absolute_error])
        drugs[i] = ele

    scores = {k: v.scores for k,v in drugs.items()}

    return drugs, scores
