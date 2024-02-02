"""
Used to generate the results for major bleeding events (Table 3 results).
"""
import optuna
import pickle
import yaml
import pandas as pd
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import RepeatedStratifiedKFold

import model_utilities as util
from competing_risks_models import *


def main():

    with open('shallow_model_baseline_survival.yml') as fin:
          config = yaml.full_load(fin)
          config = SimpleNamespace(**config)

    data_meta = pickle.load(open(config.data, 'rb'), encoding='bytes')
    data = data_meta['data']
    code_vocab = pickle.load(open(config.types, 'rb'))

    X, feature_names = util.get_time_invariant_features(data, code_vocab, config, 
                                    return_feature_names=True)   

    event, duration = util.get_competing_risk(data, config)
    
    competing_risks = len(config.competing_risks)    
    df = pd.DataFrame(X, columns=feature_names)
    df['duration'] = duration
    df['event'] = event
    
    df_temp = df
    
    X_temp = df_temp.drop(columns=['duration', 'event']).values
    duration = df_temp.duration.values
    event = np.copy(df_temp.event.values)
    
    n_splits = 10
    n_repeats = 3
    hyper_split = StratifiedKFold(n_splits=2, shuffle=True, random_state=123)
    main_split = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=123)
    
    if not config.logging:
        optuna.logging.set_verbosity(optuna.logging.CRITICAL)
    
    # iterate over the models specified in the `config.models` list. For each
    # model, evaluate the model's performance and print results.
    for model_name in config.models:
        print(model_name)
        if model_name == 'deephit':
            result = model_deephit_competing(df_temp, main_split, hyper_split, config, 
                                    competing_risks)
        elif model_name == 'dsm':       
            result = model_dsm_competing(X_temp, event, duration, main_split, hyper_split,
              config, len(config.competing_risks))
            result = result['params']
        else:
            result = cause_specific(df_temp, event, duration, main_split, hyper_split, 
                           config, competing_risks, model_name)

        fname = f'competing_{model_name}_09_3yearback_{n_splits}x{n_repeats}_2.pk'
        pickle.dump(result, open(f'hyper_parameters/may_{fname}', 'wb'))
        del result
        gc.collect()


if __name__ == '__main__':
    main()
