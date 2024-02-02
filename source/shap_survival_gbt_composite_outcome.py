'''
Calculates SHAP values over 10-fold cross-validation for a 
Gradient Boosting Survival Analysis model.
'''
import pickle
import shap
import sys
import yaml
import pandas as pd
import numpy as np
from types import SimpleNamespace

from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from model_utilities import *
from sklearn.model_selection import StratifiedKFold


shap.initjs()


def get_tree(model, idx, gradient=False):
    
    tree_dicts = []
    for tree in model.estimators_:
        tree_tmp = tree[0].tree_ if gradient else tree.tree_
    
        left = tree_tmp.children_left  
        right = tree_tmp.children_right
        default = right.copy()
        features = tree_tmp.feature
        thresholds = tree_tmp.threshold
        
        if gradient:
            values = tree_tmp.value.reshape(tree_tmp.value.shape[0], 1)
        else:
            values = tree_tmp.value[:,idx,0].reshape(tree_tmp.value.shape[0], 1)

        node_sample_weight = tree_tmp.weighted_n_node_samples
        
        tree_dict = {
            'children_left': left,
            'children_right': right,
            'children_default': default,
            'features': features,
            'thresholds': thresholds,
            'values': values * model.learning_rate if gradient else values,
            'node_sample_weight': node_sample_weight}
        
        tree_dicts.append(tree_dict)
        
    return {'trees': tree_dicts}


# In[3]:
with open('H:\\code\shallow_model_baseline_survival.yml') as fin:
    config = yaml.full_load(fin)
    config = SimpleNamespace(**config)


# In[4]:
data_meta = pickle.load(open('H:\\code\\'+config.data.replace('/', '\\'), 'rb'))
data = data_meta['data']
code_vocab = pickle.load(open('H:\\code\\'+config.types.replace('/', '\\'), 'rb'))

# In[5]:
X, feature_names = get_time_invariant_features(data, code_vocab, config,
                                                   return_feature_names=True)

# In[6]:
y_var = 'castle_new_cardiac_arrest'
# if using fixed follow-up window
duration = []

if not config.max_followup_months: 
    max_followup = max([patient.outcomes.get('end_of_followup_date') for patient in data])

for patient in data:
    if patient.outcomes.get(y_var, None):
        duration.append(patient.outcomes[y_var+'_days'])
    #check for censoring
    elif patient.outcomes['outcome_death']:
        duration.append(patient.outcomes['outcome_death_days'])
    # use individual censoring date if follow-up is fixed window
    elif config.max_followup_months:
        duration.append(patient.outcomes['end_of_followup_days'])
    # if using all follow-up, use max date
    else: 
        duration.append((max_followup - patient.index_date).days)

event = [bool(patient.outcomes.get(y_var, 0)) for patient in data]
event = np.array(event, dtype=np.double)
duration = np.array(duration, dtype=np.double)
if np.isnan(duration).any():
    print('NAN in duration array, stopping.')
    sys.exit()
elif np.isnan(event).any():
    print('NAN in event array, stopping.')
    sys.exit()


# In[7]:
df = pd.DataFrame(X)
df['duration'] = duration
df['event'] = event

df_temp = df

X_temp = df_temp.drop(columns=['duration', 'event']).values
duration = df_temp.duration.values
event = df_temp.event.values
y_train, _ = duration_event_tuple(event, duration, event, duration)

# In[8]:
model_name = 'gbt'
main_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

# ## CV Shap
list_shap_values = []
list_test_sets = []
# load hyperparameters during hyperparameter search
params_set = pickle.load(open(f'hyper_parameters/{model_name}_castle_new_cardiac_arrest_09_3yearback.pk', 'rb'))

# calculate SHAP values over 10-fold CV
for i, (train,test) in enumerate(main_split.split(X_temp, event)):
    x_train, t_train, e_train = X_temp[train], duration[train], event[train]
    x_test, t_test, e_test = X_temp[test], duration[test], event[test]
    y_train, y_test = duration_event_tuple(e_train, t_train, e_test, t_test)
    model = GradientBoostingSurvivalAnalysis(**params_set[i])
    model.fit(x_train, y_train)
    shap_gbt = get_tree(model, 0, gradient=True)

    # explain the model's predictions using SHAP values
    explainer = shap.TreeExplainer(shap_gbt, x_train, model_output='raw')
    _shap_values = explainer.shap_values(x_test)
    list_shap_values.append(_shap_values)
    list_test_sets.append(test)

# concatenate shap values from folds
shap_test = np.concatenate(list_shap_values, axis=0)
test_set = np.concatenate(list_test_sets, axis=0)
X_test = X_temp[test_set]

# In[14]:
pickle.dump({'shap':shap_test, 'X':X_test, 'names': feature_names}, open('castle_gbt_shap_sep22.pk', 'wb'))