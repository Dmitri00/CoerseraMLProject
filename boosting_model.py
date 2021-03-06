
import numpy as np
import pandas as pd
import sklearn
import time
import datetime
import json
from sklearn.model_selection import KFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier


### read data
## data consists of 108 columns. 0 column - match_id,
## [1,101] - input data, such as heroes, gold, xp, ..
## [102,107] - output (winner, duration, barracks and tower status)

ft = pd.read_csv('features.csv',header=0, index_col='match_id')
ft = ft.fillna(0)
x = ft.copy().drop(['duration', 'radiant_win','tower_status_radiant', 
    'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']).as_matrix()
y = ft.copy()['radiant_win'].as_matrix()

### containers for metrics
## estimators_space - values of optimized hyper-parameter -
## number of trees in gradient boosting
metrics = []
time = []
scores = []
estimators_space = np.linspace(5,65,12)

## cross-validation

folder = KFold(shuffle=True, random_state=72365, n_splits=5)
roc_auc = sklearn.metrics.roc_auc_score
scorer = sklearn.metrics.make_scorer(roc_auc)

for n_est in estimators_space:

    clf = GradientBoostingClassifier(n_estimators=n_est, random_state=7896, max_depth=3)

    start_time = datetime.datetime.now()
    score = cross_val_score(clf, x, y, cv=folder, scoring=scorer)
    time.append((datetime.datetime.now() - start_time).seconds)
    scores.append(np.mean(score))
    

### save data to json
metrics.append(estimators_space.tolist())
metrics.append(scores)
metrics.append(time)

json.dump(metrics, open('gradient_boosting_data', 'w'))

