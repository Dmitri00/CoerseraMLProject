import numpy as np
import pandas as pd
import sklearn
import time
import datetime
import json
import re
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import LogisticRegression


### read data
## data consists of 108 columns. 0 column - match_id,
## [1,101] - input data, such as heroes, gold, xp, ..
## [102,107] - output (winner, duration, barracks and tower status)

ft = pd.read_csv('features.csv',header=0, index_col='match_id')
ft = ft.fillna(0)
x = ft.copy().drop(['duration', 'radiant_win','tower_status_radiant', 
    'tower_status_dire', 'barracks_status_radiant', 'barracks_status_dire'], axis=1).as_matrix()
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
time = []
scores = []
c_space = np.logspace(start=-5, stop=3, num=15, base=2).tolist()

## cross-validation

folder = KFold(len(ft), shuffle=True, random_state=72365, n_folds=5)
roc_auc = sklearn.metrics.roc_auc_score
scorer = sklearn.metrics.make_scorer(roc_auc)

for c in c_space:

    clf = LogisticRegression(C=c, random_state=7896)

    start_time = datetime.datetime.now()
    score = cross_val_score(clf, x, y, cv=folder, scoring=scorer)
    time.append((datetime.datetime.now() - start_time).seconds)
    scores.append(np.mean(score))
    

max_score_index = np.argmax(scores)
max_c = c_space(max_score_index)

### save data to json
metrics.append(c_space)
metrics.append(scores)
metrics.append(time)
metrics.append([max_c,'best c'])

json.dump(metrics, open('logit_new_features_data', 'w'))





