"""
Used to generate the results of the composite outcome (Table 2 results).
"""
import gc
import pickle
import optuna
import yaml
import pandas as pd
import numpy as np

from tqdm import tqdm
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold

from model_utilities import *
from models import *

from datetime import datetime

def main():
    start_time = datetime.now()

    with open('shallow_model_baseline_survival.yml') as fin:
          config = yaml.full_load(fin)
          config = SimpleNamespace(**config)

    data_meta = pickle.load(open(config.data, 'rb'), encoding='bytes')
    data = data_meta['data']
    code_vocab = pickle.load(open(config.types, 'rb'))

    X, feature_names = get_time_invariant_features(data, code_vocab, config,
                                    return_feature_names=True)
    
    for y_var in tqdm(config.outcomes):
        print('^'*20)
        print(y_var)
        print('^'*20)
        
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
            return
        elif np.isnan(event).any():
            print('NAN in event array, stopping.')
            return
        
        df = pd.DataFrame(X, columns=feature_names)
        df['duration'] = duration
        df['event'] = event
        
        df_temp = df
        
        X_temp = df_temp.drop(columns=['duration', 'event']).values
        duration = df_temp.duration.values
        event = df_temp.event.values

        # split for hyperparameter tuning 
        hyper_split = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
        # split for model training and testing
        main_split = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)

        if not config.logging:
            optuna.logging.set_verbosity(optuna.logging.CRITICAL)
        
        # iterate over each model specified in the `config.models` list and 
        # perform model training and evaluation.
        for model_name in config.models:
            print('#####################')
            print(model_name)
            if model_name in ['coxnet', 'gbt', 'rsf', 'svm', 'component_gbt']:
                result = model_sksurv(X_temp, event, duration, main_split, hyper_split, 
                             config, model_name)
            
            elif model_name in ['deephit', 'coxcc', 'coxtime']:
                result = model_coxmlp(df_temp, main_split, hyper_split, config, model_name) 
            
            elif model_name in ['dsm', 'dcm', 'deepsurv']:
                result = model_dsm(X_temp, event, duration, main_split, hyper_split, 
                          config, model_name)

            pickle.dump(result['data'], open(f'results/folds/{model_name}_{y_var}_09_3yearback_redo3.pk', 'wb'))
            pickle.dump(result['params'], open(f'hyper_parameters/{model_name}_{y_var}_09_3yearback_redo3.pk', 'wb'))
            
            del result
            gc.collect()
            print(flush=True)

    print('Running time: ', datetime.now() - start_time)
    

if __name__ == '__main__':
    main()