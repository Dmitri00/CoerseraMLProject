
import numpy as np
import pandas as pd
import sklearn
import json
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


### read data
## data consists of 108 columns. 0 column - match_id,
## [1,101] - input data, such as heroes, gold, xp, ..
## [102,107] - output (winner, duration, barracks and tower status)

ft = pd.read_csv('features.csv',header=0, index_col='match_id')
ft = ft.fillna(0)
x = ft.copy().drop(['duration', 'radiant_win','tower_status_radiant', 
    'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1).as_matrix()
x = StandardScaler().fit_transform(x)
y = ft.copy()['radiant_win'].as_matrix()

### containers for metrics
## c_space - values of optimized L2 coefficient in LogisticRegression
metrics = []

grid = {
'C': np.logspace(start=-5, stop=1, num=9)
}

## cross-validation

folder = KFold(shuffle=True, random_state=72365, n_splits=5)
clf = LogisticRegression(penalty='l2', random_state=34532)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=folder)

gs.fit(x,y)

### save data to json
metrics.append(gs.best_score_)
metrics.append(gs.best_params_)
print(metrics)

json.dump(metrics, open('logit_data', 'w'))

