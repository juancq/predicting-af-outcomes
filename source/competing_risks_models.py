import gc
import optuna
import numpy as np
import pandas as pd
from collections import defaultdict

from pycox.models import DeepHit
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

from sklearn_pandas import DataFrameMapper

import torch
import torchtuples as tt

from model_utilities import *
from models import *


class CauseSpecificNet(torch.nn.Module):
    def __init__(self, in_features, num_nodes_shared, num_nodes_indiv,
                 num_risks, out_features, batch_norm=True, dropout=None):
        super().__init__()
        self.shared_net = tt.practical.MLPVanilla(
            in_features, num_nodes_shared[:-1], num_nodes_shared[-1],
            batch_norm, dropout,
            )
        self.risk_nets = torch.nn.ModuleList()
        for _ in range(num_risks):
            net = tt.practical.MLPVanilla(
                num_nodes_shared[-1], num_nodes_indiv, out_features,
                batch_norm, dropout,
                )
            self.risk_nets.append(net)
    
    def forward(self, input):
        out = self.shared_net(input)
        out = [net(out) for net in self.risk_nets]
        out = torch.stack(out, dim=1)
        return out
    

class LabTransform(LabTransDiscreteTime):
    def transform(self, durations, events):
        durations, is_event = super().transform(durations, events > 0)
        events[is_event == 0] = 0
        return durations, events.astype('int64')


def model_deephit_competing(df, main_split, hyper_split, config, num_risks):
    horizons = config.horizons
    times = np.quantile(df.duration[df.event>0], horizons).tolist()
    
    train_scores, test_scores = list(zip(*[get_score_list(times) for _ in range(num_risks)]))
    
    output_bias = False
    callbacks = []
    #callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    verbose = config.verbose
    epochs = config.epochs
    batch_norm= True
    
    cols = df.drop(columns=['duration', 'event']).columns.tolist()
    leave = [(col, None) for col in cols]
    x_mapper = DataFrameMapper(leave, drop_cols=['duration', 'event'])
    
    params_set = []
    
    get_target = lambda d: (d.duration.values, d.event.values)
    
    for train, test in main_split.split(df, df.event):
        
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')

        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: trial_deephit_competing(trial, df_train,
                                                             x_mapper,
                                               times, hyper_split, config,
                                               num_risks), 
                       n_trials=config.n_trials)
        
        # get best parameters
        print('Best params: ', study.best_trial.params)
        params_set.append(study.best_trial.params)
        params = SimpleNamespace(**study.best_trial.params)
        nodes_shared = [params.nodes] * params.layers
        #num_durations = 24
        labtrans = LabTransform(params.num_durations)
        dropout = params.dropout
        #dropout = 0
        
        # data transformations
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        
        y_train = get_target(df_train)
        y_train = labtrans.fit_transform(*y_train)
        durations_test, events_test = get_target(df_test)
        
        out_features = len(labtrans.cuts)

        net = CauseSpecificNet(x_train.shape[1], nodes_shared, 
                                 params.num_nodes_ind, num_risks,
                                 out_features, batch_norm, 
                                 dropout)
        
        model = DeepHit(net, tt.optim.Adam, alpha=params.alpha, sigma=params.sigma,
                        duration_index=labtrans.cuts)
        model.name_ = 'deephit'
        model.fit(x_train, y_train, params.batch_size, epochs, 
                  callbacks, verbose)#, val_data=val)

        e_train = df_train.event.values
        t_train = df_train.duration.values
        e_test = df_test.event.values
        t_test = df_test.duration.values
        
        cif_train = model.predict_cif(x_train)
        cif_test = model.predict_cif(x_test)

        for j in range(num_risks):
            df_surv_train = pd.DataFrame(1-cif_train[j], model.duration_index)
            out_survival_train = get_survival_at_times(df_surv_train, times)
            out_risk_train = 1 - out_survival_train
                    
            df_surv_test = pd.DataFrame(1-cif_test[j], model.duration_index)
            out_survival_test = get_survival_at_times(df_surv_test, times)
            out_risk_test = 1 - out_survival_test

            evaluate(e_train, e_train, t_train, t_train, out_risk_train,
                     out_survival_train, train_scores[j])
            evaluate(e_train, e_test, t_train, t_test, out_risk_test,
                     out_survival_test, test_scores[j])
        
        del model
    
    gc.collect()
    for j in range(num_risks):
        print(f'Competing Risk {j+1}')
        report(horizons, train=train_scores[j], test=test_scores[j])

    return params_set


def trial_deephit_competing(trial, df, x_mapper, times, split, config, num_risks):
    num_nodes_ind = trial.suggest_categorical('num_nodes_ind', config.optuna['nodes'])
    nodes = trial.suggest_categorical('nodes', config.optuna['nodes'])
    layers = trial.suggest_int('layers', 1, 2)
    dropout = trial.suggest_categorical('dropout', config.optuna['dht_dropout'])
    num_durations = trial.suggest_categorical('num_durations', config.optuna['dht_durations'])
    # output bias - True or False
    alpha = trial.suggest_categorical('alpha', config.optuna['dht_alpha'])
    sigma = trial.suggest_categorical('sigma', config.optuna['dht_sigma'])
    learning_rate = trial.suggest_categorical('learning_rate', config.optuna['learning_rate'])
    batch_size = trial.suggest_categorical('batch_size', config.optuna['batch_size'])
    epochs = config.optuna['epochs']
    verbose = config.verbose
    callbacks = []
    #callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    output_bias = False
    batch_norm = True
    nodes_shared = [nodes] * layers
        
    #num_durations = 24
    labtrans = LabTransform(num_durations)
    get_target = lambda d: (d.duration.values, d.event.values)
    
    result = []
    # y = (duration array, event array)
    for train, test in split.split(df, df.event):
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        
        #df_val = df_train.groupby('event', group_keys=False).apply(lambda x: x.sample(frac=0.15))
        #df_train = df_train.drop(df_val.index)
        
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        #x_val = x_mapper.transform(df_val).astype('float32')  
        x_test = x_mapper.transform(df_test).astype('float32')
            
        y_train = get_target(df_train)
        y_train = labtrans.fit_transform(*y_train)
        #y_val = get_target(df_val)
        #y_val = labtrans.transform(*y_val)
        #val = tt.tuplefy(x_val, y_val)
        out_features = len(labtrans.cuts)
        
        net = CauseSpecificNet(x_train.shape[1], nodes_shared, 
                                 num_nodes_ind, num_risks,
                                 out_features, batch_norm, 
                                 dropout)
    
        #optimizer = tt.optim.AdamWR(lr=learning_rate, 
        #                            decoupled_weight_decay=0.01,
        #                            cycle_eta_multiplier=0.8)
        model = DeepHit(net, tt.optim.Adam, alpha=alpha, sigma=sigma,
                        duration_index=labtrans.cuts)
    
        model.fit(x_train, y_train, batch_size, epochs, 
                  callbacks, verbose)#, val_data=val)#,
        
        cif = model.predict_cif(x_test)
        cif1 = pd.DataFrame(cif[0], model.duration_index)
        out_cif = get_survival_at_times(cif1, times)
        
        et_train, et_test = duration_event_tuple(df_train.event.values, 
                                                 df_train.duration.values,
                             df_test.event.values, df_test.duration.values)

        trial_score = trial_metric(et_train, et_test, out_cif, times)
        
        result.append(trial_score)
        
        del model
    
    return np.mean(result)


def trial_dsm_competing(trial, X, event, duration, times, split, config,
              competing_risks):
    k = trial.suggest_int('k', 2, 6)
    distribution = trial.suggest_categorical('distribution', ['Weibull', 'LogNormal'])
    discount = trial.suggest_categorical('discount', [0.5, 0.75, 1])
    nodes = trial.suggest_categorical('nodes', config.optuna['nodes'])
    layers = trial.suggest_categorical('layers', ['single', 'double', 'none'])
    batch_size = trial.suggest_categorical('batch_size', config.optuna['batch_size'])
    learning_rate = trial.suggest_categorical('learning_rate', config.optuna['learning_rate'])
   
    
    if layers == 'single':
        layers = [nodes]
    elif layers == 'double':
        layers = [nodes, nodes]
    else:
        layers = None
    
    result = []
    for train, test in split.split(X, event):
        train_data, val_tt = get_validation(X[train], duration[train], event[train])
        x_train, t_train, e_train = train_data
        x_test, t_test, e_test = X[test], duration[test], event[test]
        x_test = x_test.astype(np.double)
        
        et_train = np.array(list(zip(e_train, t_train)), 
                           dtype=[('e', bool), ('t', float)])
        et_test = np.array(list(zip(e_test, t_test)), 
                           dtype=[('e', bool), ('t', float)])
        
        model = DeepSurvivalMachines(k=k, layers=layers, distribution=distribution, 
                                     discount=discount)
            
        model.fit(x_train, t_train, e_train, learning_rate=learning_rate, 
                  batch_size=batch_size, iters=config.optuna['epochs'],
                  val_data=val_tt)

        # if optimizing over all risks
        trial_score = []
        for j in range(competing_risks):
            out_risk = 1 - model.predict_survival(x_test.astype(np.double), 
                                                  times, risk=j+1)      
            trial_score.append(trial_metric(et_train, et_test, out_risk, times))
        result.extend(trial_score)
        
        # just look at first risk, i.e. major bleeding
        #out_risk = 1 - model.predict_survival(x_test.astype(np.double), 
        #                                      times, risk=1)      
        result.append(trial_metric(et_train, et_test, out_risk, times))

        del model
    
    return np.mean(result)


def model_dsm_competing(X, event, duration, main_split, hyper_split, config, 
                        competing_risks):   
    horizons = config.horizons
    times = np.quantile(duration[event>0], horizons).tolist()
    train_scores = []
    test_scores = []
    params_set = []

    save = defaultdict(list)
    
    for risk in range(competing_risks):
        _train_scores, _test_scores = get_score_list(times)
        train_scores.append(_train_scores)
        test_scores.append(_test_scores)

    for train, test in main_split.split(X, event):
        train_data, val_tt = get_validation(X[train], duration[train], event[train])
        x_train, t_train, e_train = train_data
        x_test, t_test, e_test = X[test], duration[test], event[test]
        x_test = x_test.astype(np.double)
        
        study = optuna.create_study(direction='maximize')
        study_lambda = lambda trial: trial_dsm_competing(trial, x_train, e_train, t_train, 
                                               times, hyper_split, 
                                               config, competing_risks)
        study.optimize(study_lambda, n_trials=config.n_trials)
        params_set.append(study.best_trial.params)
        print('Best params: ', study.best_trial.params)
        params = SimpleNamespace(**study.best_trial.params)
        if params.layers == 'single':
            params.layers = [params.nodes]
        elif params.layers == 'double':
            params.layers = [params.nodes, params.nodes]
        else:
            params.layers = None
        
        model = DeepSurvivalMachines(k=params.k, layers=params.layers, 
                                     distribution=params.distribution, 
                                     discount=params.discount)
        model.name_ = 'dsm'
        
        model.fit(x_train, t_train, e_train, learning_rate=params.learning_rate, 
                  batch_size=params.batch_size, iters=config.epochs,
                  val_data=val_tt)

        for j in range(competing_risks):
            
            evaluate(model, (e_train, t_train), 
                 (x_train, e_train, t_train), 
                 train_scores[j], competing=j+1)
        
            evaluate(model, (e_train, t_train), 
                 (x_test, e_test, t_test), 
                 test_scores[j], competing=j+1)
        
        out_survival, _ = generate_predictions(model, x_test, times, competing_risk=1)
        save['predictions'].append(out_survival)
        save['event'].append(e_test)
        save['duration'].append(t_test)
        save['pred_at_event'].append(generate_individual_predictions(model, 
                                                                     x_test, t_test, e_test, competing_risk=1))
        
    test_scores[0].d_calibration = d_calibration(np.concatenate(save['event']), 
                                          np.concatenate(save['pred_at_event']))
    save['test_score'] = test_scores
    save['train_score'] = train_scores
    
    for j in range(competing_risks):
        print(f'Competing Risk {j+1}')
        report(horizons, train=train_scores[j], test=test_scores[j])
    
    return dict(params=params_set, data=save)


def cause_specific(df_temp, event, duration, main_split, hyper_split,
                   config, competing_risks, model_name):
    
    X_temp = df_temp.drop(columns=['duration', 'event']).values
    
    params = []
    print('#####################')
    # for every competing risk, do the following
    for risk in range(competing_risks): 
        print(f'Competing Risk {risk+1}')
        print(f'{config.competing_risks[risk]}')
        event_i = convert_to_cause_specific(event, risk+1)
        df_temp.loc[:,'event'] = event_i
        
        print('#####################')
        print(model_name)
        if model_name in ['coxnet', 'gbt', 'rsf']:
            result = model_sksurv(X_temp, event_i, duration, main_split, hyper_split, 
                         config, model_name)
        
        elif model_name in ['coxcc', 'coxtime']:
            result = model_coxmlp(df_temp, main_split, hyper_split, config, model_name) 
        
        elif model_name in ['dcm', 'dsm_single', 'deepsurv']:
            if model_name == 'dsm_single':
                _model_name = 'dsm'
            else:
                _model_name = model_name
            result = model_dsm(X_temp, event_i, duration, main_split, hyper_split, 
                      config, _model_name)
        
        else:
            print('unsupported model_name!!')

        params.append((f'risk_{risk}', result['params']))
        # returning here because we only care about first risk
        #return result
    return params