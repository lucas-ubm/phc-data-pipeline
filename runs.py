from classes import drug
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import median_absolute_error


def run(name, ge, dr, fs, feda, model, n=0, fs_tuning=None, tuning=None, p = 0.01, t = 4, metric='AUC_IC50', test=None):
    drugs = {}
    for i in name:
        ele = drug(i, ge, dr)
        ele.pre(p = p, t = t)
        ele.combine(metric = metric)
        ele.split(test = test)
        ele.fs(fs, n=n, tuning=fs_tuning)
        if feda:
            ele.feda()
        ele.train(model, tuning=tuning)
        ele.metrics([r2_score, mean_absolute_error, mean_squared_error, median_absolute_error])
        drugs[i] = ele
        
    return drugs
        
            
        
    