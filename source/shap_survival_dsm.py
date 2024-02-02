'''
Calculates SHAP values over 10-fold cross-validation for a 
Deep Survival Machines model.
'''
import pickle
import shap
import sys
import yaml
import pandas as pd
import numpy as np
from types import SimpleNamespace

from auton_survival.models.dsm import DeepSurvivalMachines

from model_utilities import *
from competing_risks_models import *

from sklearn.model_selection import StratifiedKFold


shap.initjs()


with open('H:\\code\shallow_model_baseline_survival.yml') as fin:
    config = yaml.full_load(fin)
    config = SimpleNamespace(**config)

data_meta = pickle.load(open('H:\\code\\'+config.data.replace('/', '\\'), 'rb'))
data = data_meta['data']
code_vocab = pickle.load(open('H:\\code\\'+config.types.replace('/', '\\'), 'rb'))

X, feature_names = get_time_invariant_features(data, code_vocab, config,
                                                   return_feature_names=True)

# In[6]:
# if using fixed follow-up window
duration = []
event = []
if not config.max_followup_months: 
    max_followup = max([patient.outcomes.get('end_of_followup_date') for patient in data])

for patient in data:
    event_i = []
    duration_i = []
    for risk in config.competing_risks:
        # composite risks
        if type(risk) == list:
            d, e = list(zip(*[get_individual_risk_event_duration(patient, r) for r in risk]))
            event_i.append(any(e))
            duration_i.append(min(d))
        else:
            # single risks
            d, e = get_individual_risk_event_duration(patient, risk)
            event_i.append(e)
            duration_i.append(d)

    if sum(event_i) > 0:
        first_risk = np.argmin(duration_i)
        duration.append(duration_i[first_risk])
        event.append(first_risk+1)
    else:
        event.append(0)
        if config.max_followup_months:
            duration.append(patient.outcomes['end_of_followup_days'])
        # if using all follow-up, use max date
        else: 
            duration.append((max_followup - patient.index_date).days)
    
assert len(set(event)) == len(config.competing_risks)+1, 'error in competing risk processing'

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


times = np.quantile(duration[event>0], config.horizons).tolist()    
main_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

# ## CV Shap
list_shap_values = []
list_test_sets = []
params_set = pickle.load(open('hyper_parameters/competing_dsm_bleeding_09_back3_3risks.pk', 'rb'))

# calculate SHAP values over 10-fold CV
for i, (train, test) in enumerate(main_split.split(X_temp, event)):
    train_data, val_tt = get_validation(X_temp[train], duration[train], event[train])
    x_train, t_train, e_train = train_data
    x_test, t_test, e_test = X_temp[test], duration[test], event[test]

    params = SimpleNamespace(**params_set[i])
    if params.layers == 'single':
        params.layers = [params.nodes]
    elif params.layers == 'double':
        params.layers = [params.nodes, params.nodes]
    else:
        params.layers = None
    model = DeepSurvivalMachines(k=params.k, layers=params.layers, 
                             distribution=params.distribution, 
                             discount=params.discount)

    model.fit(x_train, t_train, e_train, learning_rate=params.learning_rate, 
          batch_size=params.batch_size, iters=config.epochs, val_data=val_tt)
    
    # wrap predict survival function for SHAP
    def f(t):
        out_survival = model.predict_survival(t, times[0], risk=1)
        return 1-out_survival#.flatten().T
    
    # explain the model's predictions using SHAP values
    explainer = shap.Explainer(f, x_train, feature_names=feature_names)
    shap_obj = explainer(x_test)
    _shap_values = shap_obj.values
    list_shap_values.append(_shap_values)
    list_test_sets.append(test)

# concatenate shap values from folds
shap_test = np.concatenate(list_shap_values, axis=0)
test_set = np.concatenate(list_test_sets, axis=0)
X_test = X_temp[test_set]

# In[12]:
pickle.dump({'shap':shap_test, 'X':X_test, 'names': feature_names}, open('major_bleeding_dsm_shap_sep22.pk', 'wb'))