import numpy as np
import pandas as pd
import sklearn
import time
import datetime
import json
import re
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


### read data
## data consists of 108 columns. 0 column - match_id,
## [1,101] - input data, such as heroes, gold, xp, ..
## [102,107] - output (winner, duration, barracks and tower status)

ft = pd.read_csv('features.csv',header=0, index_col='match_id')
ft = ft.fillna(0)
y = ft.copy()['radiant_win'].as_matrix()
ft_header=list(ft)

### delete categorial features r/d#_hero, lobby_type
i=0
cutted_header = ft_header
while i < len(cutted_header):
    h = cutted_header[i]
    if re.match('[rd][1-5]_hero', h):
        del cutted_header[i]
    elif h == 'lobby_type':
        del cutted_header[i]
    else:
        i+=1
for i in ['duration', 'radiant_win','tower_status_radiant', 'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire']:
    cutted_header.remove(i)
x = ft.copy()[cutted_header]
x = StandardScaler().fit_transform(x)
print(list(x))
### lets make new features of heroes from the old one
### create new feature for every heroe, where his feature = 1, when he is picked by radiant
### -1 - by dire, 0 - he isn`t in the game

### Calculate, how many heroes are in the game
### We can calculate the number of unique hero's id at any column,
### because there are 100 hundred thousands of rows.
### All ids assigned from 1 incrementally. Thats why max id = # of heroes


N = sorted(pd.unique(ft['d2_hero']))[-1]

## make new features by described scheme
x_pick = np.zeros((ft.shape[0], N))
for i, match_id in enumerate(ft.index):
    for p in range(5):
        x_pick[i, ft.loc[match_id, "r%d_hero"% (p+1)]-1] = 1
        x_pick[i, ft.loc[match_id, "r%d_hero"% (p+1)]-1] = -1
x = np.c_[x,x_pick]


### containers for metrics
## c_space - values of optimized L2 coefficient in LogisticRegression
metrics = []
grid = {
'C': np.logspace(start=-5, stop=3, num=9)
}

## cross-validation

folder = KFold(shuffle=True, random_state=72365, n_splits=5)
clf = LogisticRegression(penalty='l2', random_state=7896)
gs = GridSearchCV(clf, grid, scoring='roc_auc', cv=folder)
gs.fit(x,y)
    
### save data to json
metrics.append(gs.best_score_)
metrics.append(gs.best_params_)
print(metrics)
json.dump(metrics, open('logit_gs_data', 'w'))





