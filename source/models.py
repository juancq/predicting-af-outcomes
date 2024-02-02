import gc
import optuna
import numpy as np
import pandas as pd

from collections import defaultdict

from sklearn.preprocessing import StandardScaler

from sksurv.ensemble import RandomSurvivalForest
from sksurv.ensemble import GradientBoostingSurvivalAnalysis
from sksurv.ensemble import ComponentwiseGradientBoostingSurvivalAnalysis
from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis

from auton_survival.models.dsm import DeepSurvivalMachines
from auton_survival.models.cph import DeepCoxPH
from auton_survival.models.dcm import DeepCoxMixtures

import torchtuples as tt
from pycox.models import CoxCC, CoxPH
from pycox.models import CoxTime
from pycox.models import DeepHitSingle
from pycox.models.cox_time import MLPVanillaCoxTime
from pycox.evaluation import EvalSurv
from sklearn_pandas import DataFrameMapper

from model_utilities import *


def model_sksurv(X, event, duration, main_split, hyper_split, config, model_name):
    # static
    time_sets = get_evaluation_times(duration[event==1], config.horizons)
    times = time_sets['time_horizons']
    time_10_90 = time_sets['time_10_90']
    save = defaultdict(list)
    
    params_set = []
    train_scores, test_scores = get_score_list(times)

    for train, test in main_split.split(X, event):
        
        x_train, t_train, e_train = X[train], duration[train], event[train]
        x_test, t_test, e_test = X[test], duration[test], event[test]
        
        y_train, y_test = duration_event_tuple(e_train, t_train, e_test, t_test)
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: 
            trial_sksurv(trial, x_train, e_train, t_train, 
                             times, hyper_split, config, 
                             model_name), 
            n_trials=config.n_trials
        )
            
        params = study.best_trial.params
        params_set.append(params)
        if model_name == 'rsf':
            model = RandomSurvivalForest(**params)
        elif model_name == 'gbt':
            model = GradientBoostingSurvivalAnalysis(**params)
        elif model_name == 'component_gbt':
            model = ComponentwiseGradientBoostingSurvivalAnalysis(**params)
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            model = CoxnetSurvivalAnalysis(l1_ratio=params['l1_ratio']/10, 
                                           alphas=[params['alpha']],
                                           fit_baseline_model=True,
                                           tol=1e-6)
            #,
            #                               max_iter=10000,
            #                               tol=1e-6)
            
        
        model.name_ = model_name
        model.fit(x_train, y_train)
                   
        evaluate(model, (e_train, t_train), 
                 (x_train, e_train, t_train), 
                 train_scores)
        
        evaluate(model, (e_train, t_train), 
                 (x_test, e_test, t_test), 
                 test_scores)
        
        save['estimators'].append(model)
        out_survival, _ = generate_predictions(model, x_test, times)
        save['predictions'].append(out_survival)
        save['event'].append(e_test)
        save['duration'].append(t_test)
        save['pred_at_event'].append(generate_individual_predictions(model, x_test, t_test, e_test))
        
    test_scores.d_calibration = d_calibration(np.concatenate(save['event']), 
                                          np.concatenate(save['pred_at_event']))
    save['test_score'] = test_scores
    save['train_score'] = train_scores

    report(config.horizons, train=train_scores, test=test_scores)
    return dict(params=params_set, data=save)
       

def trial_sksurv(trial, X, event, duration, times, split, config, model_name):
    params = {}
    if model_name == 'rsf':
        params['n_estimators'] = trial.suggest_categorical('n_estimators', [10, 50, 100, 200])
        params['max_depth'] = trial.suggest_categorical('max_depth', [3, 5, 7])#10, None])
        params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', 100, 200])
        params['min_samples_leaf'] = trial.suggest_categorical('min_samples_leaf', [3, 10, 150, 200])
        #mtry ['sqrt', 50, 75, 'all']
        #min_node_split [150, 200, 250]
    elif model_name == 'gbt':
        params['n_estimators'] = trial.suggest_categorical('n_estimators', [10, 50, 100, 200])
        params['learning_rate'] = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 0.5])
        params['max_features'] = trial.suggest_categorical('max_features', ['sqrt', 'log2', 100, 200])
        params['min_samples_leaf'] = trial.suggest_categorical('min_samples_leaf', [1, 3, 10, 150, 200])
        params['max_depth'] = trial.suggest_categorical('max_depth', [3, 5, 7])#10])
    elif model_name == 'component_gbt':
        #loss = trial.suggest_categorical('loss', ['coxph', 'squared', 'ipcwls'])
        params['n_estimators'] = trial.suggest_categorical('n_estimators', [10, 50, 100, 200])
        params['learning_rate'] = trial.suggest_categorical('learning_rate', [0.001, 0.01, 0.1, 0.5])
        params['subsample'] = trial.suggest_uniform('subsample', 0, 1)
    elif model_name == 'svm':
        params['rank_ratio'] = trial.suggest_float('rank_ratio', 0, 1, step=0.1)
        params['alpha'] = trial.suggest_uniform('alpha', 0.0001, 0.05)
    else:
        params['alphas'] = [trial.suggest_uniform('alpha', 0.0001, 0.05)]
        l1_ratio = trial.suggest_int('l1_ratio', 2, 10)
        params['l1_ratio'] = l1_ratio / 10.
        params['fit_baseline_model']=True
        params['max_iter']=50000
        params['tol']=1e-6
    
    result = []
    for train, test in split.split(X, event):
        
        x_train, t_train, e_train = X[train], duration[train], event[train]
        x_test, t_test, e_test = X[test], duration[test], event[test]
        
        y_train, y_test = duration_event_tuple(e_train, t_train, e_test, t_test)
        
        if model_name == 'rsf':
            model = RandomSurvivalForest(**params)
        elif model_name == 'gbt':
            model = GradientBoostingSurvivalAnalysis(**params)
        elif model_name == 'component_gbt':
            model = ComponentwiseGradientBoostingSurvivalAnalysis(**params)
        else:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.transform(x_test)
            model = CoxnetSurvivalAnalysis(**params)
        
        model.name_ = model_name
        try:
            model.fit(x_train, y_train)
        except Exception as e:
            #raise(e)
            result.append(0)
            continue

        out_survival, out_risk = generate_predictions(model,  
                                                      x_test, times)
        
        trial_score = trial_metric(y_train, y_test, out_risk, times)
        result.append(trial_score)
    
    return np.mean(result)


def trial_dsm(trial, X, event, duration, times, split, config, model_name):
    if model_name in ['dsm', 'dcm']:    
        k = trial.suggest_int('k', 2, 6)
    if model_name == 'dsm':
        distribution = trial.suggest_categorical('distribution', ['Weibull', 'LogNormal'])
        discount = trial.suggest_categorical('discount', [0.5, 0.75, 1])
    nodes = trial.suggest_categorical('nodes', config.optuna['nodes'])
    layers = trial.suggest_categorical('layers', ['single', 'double', 'none'])
    batch_size = trial.suggest_categorical('batch_size', config.optuna['batch_size'])
    learning_rate = trial.suggest_categorical('learning_rate', config.optuna['learning_rate'])
    #iters = trial.suggest_int('iters', 1, 51, 10)

    if layers == 'single':
        layers = [nodes]
    elif layers == 'double':
        layers = [nodes, nodes]
    else:
        layers = None
    
    result = []
    for train, test in split.split(X, event):
        x_test, t_test, e_test = X[test], duration[test], event[test]        
        train_data, val_tt = get_validation(X[train], duration[train], event[train])
        x_train, t_train, e_train = train_data
        
        if model_name == 'dsm':
            model = DeepSurvivalMachines(k=k, layers=layers, distribution=distribution, 
                                         discount=discount)
        elif model_name == 'dcm':
            model = DeepCoxMixtures(k=k, layers=layers)
        else:
            model = DeepCoxPH(layers=layers)
        
        model.name_ = model_name
        model.fit(x_train, t_train, e_train, learning_rate=learning_rate, 
                  batch_size=batch_size, iters=config.optuna['epochs'],
                  val_data=val_tt)#config.optuna['epochs'])
        
        x_test = x_test.astype(np.double)
        out_risk = 1 - model.predict_survival(x_test, times)
        
        et_train, et_test = duration_event_tuple(e_train, t_train, e_test, t_test)
        
        trial_score = trial_metric(et_train, et_test, out_risk, times)
        result.append(trial_score)
        
        del model
    
    if np.isnan(result).any():
        return 0
    return np.mean(result)


def model_dsm(X, event, duration, main_split, hyper_split, config, model_name):   
    # static
    time_sets = get_evaluation_times(duration[event==1], config.horizons)
    times = time_sets['time_horizons']
    time_10_90 = time_sets['time_10_90']
    save = defaultdict(list)
    
    train_scores, test_scores = get_score_list(times)
    params_set = []

    for train, test in main_split.split(X, event):
        x_test, t_test, e_test = X[test], duration[test], event[test]
        x_test = x_test.astype(np.double)
        
        train_data, val_tt = get_validation(X[train], duration[train], event[train])
        x_train, t_train, e_train = train_data
        
        study = optuna.create_study(direction='maximize')
        study_lambda = lambda trial: trial_dsm(trial, x_train, e_train, t_train, 
                                               times, hyper_split, 
                                               config, model_name)
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
        
        if model_name == 'dsm':
            model = DeepSurvivalMachines(k=params.k, layers=params.layers, 
                                         distribution=params.distribution, 
                                         discount=params.discount)
        elif model_name == 'dcm':
            model = DeepCoxMixtures(k=params.k, layers=params.layers)
        else:
            model = DeepCoxPH(layers=params.layers)
        
        model.name_ = model_name
        model.fit(x_train, t_train, e_train, learning_rate=params.learning_rate, 
                  batch_size=params.batch_size, iters=config.optuna['epochs'],
                  val_data=val_tt)#=config.epochs)#,
                  #patience=10)

        evaluate(model, (e_train, t_train), 
                 (x_train, e_train, t_train), 
                 train_scores)
        
        evaluate(model, (e_train, t_train), 
                 (x_test, e_test, t_test), 
                 test_scores)
        
        save['estimators'].append(model)
        out_survival, _ = generate_predictions(model, x_test, times)
        save['predictions'].append(out_survival)
        save['event'].append(e_test)
        save['duration'].append(t_test)
        save['pred_at_event'].append(generate_individual_predictions(model, x_test, t_test, e_test))

    test_scores.d_calibration = d_calibration(np.concatenate(save['event']), 
                                          np.concatenate(save['pred_at_event']))
        
    save['test_score'] = test_scores
    save['train_score'] = train_scores
    report(config.horizons, train=train_scores, test=test_scores)
    return dict(params=params_set, data=save)
    

def trial_coxcc(trial, df, x_mapper, times, split, config, 
                model_name=None):
    nodes = trial.suggest_categorical('nodes', config.optuna['nodes'])
    layers = trial.suggest_categorical('layers', [1, 2])
    #batch_norm = trial.suggest_categorical('batch_norm', [True, False])
    batch_norm = True
    dropout = trial.suggest_categorical('dropout', config.optuna['dropout'])
    # output bias - True or False
    batch_size = trial.suggest_categorical('batch_size', config.optuna['batch_size'])
    #batch_size = 128
    learning_rate = trial.suggest_categorical('learning_rate', config.optuna['learning_rate'])
    
    if model_name == 'deephit':
        num_durations = trial.suggest_categorical('num_durations', config.optuna['dht_durations'])
        alpha = trial.suggest_categorical('alpha', config.optuna['dht_alpha'])
        sigma = trial.suggest_categorical('sigma', config.optuna['dht_sigma'])
    elif model_name == 'coxcc':
        shrink = trial.suggest_categorical('shrink', config.optuna['shrink'])
    
    epochs = config.optuna['epochs']
    out_features = 1
    verbose = False
    if config.early_stopping:
        callbacks = [tt.callbacks.EarlyStopping(patience=20)]#patience=3)]
    else:
        callbacks = []
    
    if config.load_best:
        callbacks.append(tt.callbacks.BestWeights())
        
    output_bias = False
    nodes = [nodes] * layers
    if 'coxtime' == model_name:
        labtrans = CoxTime.label_transform()
    elif 'deephit' == model_name:
        labtrans = DeepHitSingle.label_transform(num_durations)
    
    result = []
    # y = (duration array, event array)
    for train, test in split.split(df, df.event):
       
        df_train = df.iloc[train]
        df_test = df.iloc[test]  
        if config.early_stopping:
            df_1 = df_train[df_train.event==1].sample(frac=0.15)
            df_0 = df_train[df_train.event==0].sample(frac=0.15)
            df_val = pd.concat([df_0, df_1])
            #df_val = df_train.sample(frac=0.15)
            df_train = df_train.drop(df_val.index)
            
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.transform(df_test).astype('float32')
        y_train = (df_train.duration.values, df_train.event.values)
        
        if config.early_stopping:
            x_val = x_mapper.transform(df_val).astype('float32')
            y_val = (df_val.duration.values, df_val.event.values)
            val = tt.tuplefy(x_val, y_val)
        
        if 'coxtime' ==  model_name:
            net = MLPVanillaCoxTime(x_train.shape[1], nodes,
                                          batch_norm, dropout)
            model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
            y_train = labtrans.fit_transform(*y_train)
        elif 'deephit' == model_name:
            y_train = labtrans.fit_transform(*y_train)
            net = tt.practical.MLPVanilla(x_train.shape[1], nodes,
                                          labtrans.out_features, batch_norm, dropout)
            model = DeepHitSingle(net, tt.optim.Adam, alpha=alpha, sigma=sigma,
                                  duration_index=labtrans.cuts)
            
            if config.early_stopping:
                y_val = labtrans.transform(*y_val)
                val = tt.tuplefy(x_val, y_val)
        else: 
            net = tt.practical.MLPVanilla(x_train.shape[1], nodes,
                                          out_features, batch_norm,
                                          dropout, output_bias=output_bias)
            model = CoxCC(net, optimizer=tt.optim.Adam, shrink=shrink)
        
        model.name_ = model_name     

        model.optimizer.set_lr(learning_rate)
        
        if config.early_stopping:
            model.fit(x_train, y_train, batch_size, epochs, 
                  callbacks, verbose, val_data=val)
        else:
            model.fit(x_train, y_train, batch_size, epochs, 
                  callbacks, verbose)#,
        
        if not('deephit' == model_name):
            _ = model.compute_baseline_hazards()
        
        out_survival, out_risk = generate_predictions(model,
                                                      x_test, times)
        
        et_train, et_test = duration_event_tuple(df_train.event.values, 
                                            df_train.duration.values,
                        df_test.event.values, df_test.duration.values)

        #_, mean_auc = cumulative_dynamic_auc(et_train, et_test, out_risk,times)
        trial_score = trial_metric(et_train, et_test, out_risk, times)
        result.append(trial_score)
        
        del model
    
    return np.mean(result)


def model_coxmlp(df, main_split, hyper_split, config, model_name):
    # static
    time_sets = get_evaluation_times(df.duration[df.event==1], config.horizons)
    times = time_sets['time_horizons']
    time_10_90 = time_sets['time_10_90']
    
    save = defaultdict(list)
    
    batch_norm = True
    out_features = 1
    output_bias = False
    if config.early_stopping:
        callbacks = [tt.callbacks.EarlyStopping(patience=20)]
    else:
        callbacks = []
    
    if config.load_best:
        callbacks.append(tt.callbacks.BestWeights())
        
    verbose = False
    epochs = config.epochs
    
    cols = df.drop(columns=['duration', 'event']).columns.tolist()
    if config.feature_type == 'count':
        std = [([col], StandardScaler()) for col in cols]
        leave = []
    else:
        print(cols)
        cols.remove('age')
        leave = [(col, None) for col in cols]
        std = [([col], StandardScaler()) for col in ['age']]
    col_ops = leave + std
    x_mapper = DataFrameMapper(col_ops, drop_cols=['duration', 'event'])
    if 'coxtime' == model_name:
        labtrans = CoxTime.label_transform()
    
    train_scores, test_scores = get_score_list(times)
    params_set = []
    
    get_target = lambda d: (d.duration.values, d.event.values)
    
    for i, (train, test) in enumerate(main_split.split(df, df.event)):
        
        df_train = df.iloc[train]
        df_test = df.iloc[test]
        
        study = optuna.create_study(direction='maximize')
        study.optimize(lambda trial: trial_coxcc(trial, df_train,
                                                 x_mapper,
                                               times, hyper_split, config,
                                               model_name), 
                       n_trials=config.n_trials)

        params_set.append(study.best_trial.params)
        print('Best params: ', study.best_trial.params)
        params = SimpleNamespace(**study.best_trial.params)
        nodes = [params.nodes] * params.layers
        #params.batch_size = 128
        
        if config.early_stopping:
            df_1 = df_train[df_train.event==1].sample(frac=0.15)
            df_0 = df_train[df_train.event==0].sample(frac=0.15)
            df_val = pd.concat([df_0, df_1])
            #df_val = df_train.sample(frac=0.15)
            df_train = df_train.drop(df_val.index)
            
        x_train = x_mapper.fit_transform(df_train).astype('float32')
        x_test = x_mapper.fit_transform(df_test).astype('float32')
        y_train = get_target(df_train)
        durations_test, events_test = get_target(df_test)
        
        if config.early_stopping:
            x_val = x_mapper.fit_transform(df_val).astype('float32')
            y_val = (df_val.duration.values, df_val.event.values)
            val = tt.tuplefy(x_val, y_val)
        
        if 'coxtime' ==  model_name:
            net = MLPVanillaCoxTime(x_train.shape[1], nodes,
                                          batch_norm, params.dropout)
            model = CoxTime(net, tt.optim.Adam, labtrans=labtrans)
            y_train = labtrans.fit_transform(*y_train)
        elif 'deephit' == model_name:
            labtrans = DeepHitSingle.label_transform(params.num_durations)
            y_train = labtrans.fit_transform(*y_train)
            net = tt.practical.MLPVanilla(x_train.shape[1], nodes,
                                          labtrans.out_features, 
                                          batch_norm, params.dropout)
            model = DeepHitSingle(net, tt.optim.Adam, alpha=params.alpha, 
                                  sigma=params.sigma,
                                  duration_index=labtrans.cuts)
            
            if config.early_stopping:
                y_val = labtrans.transform(*y_val)
                val = tt.tuplefy(x_val, y_val)
        else: 
            net = tt.practical.MLPVanilla(x_train.shape[1], nodes,
                                     out_features, batch_norm,
                                      params.dropout, output_bias=output_bias)
            model = CoxCC(net, optimizer=tt.optim.Adam, shrink=params.shrink)
        
        model.name_ = model_name
        model.optimizer.set_lr(params.learning_rate)
        if config.early_stopping:
            print(val[0].shape)
            model.fit(x_train, y_train, params.batch_size, epochs, 
                  callbacks, verbose, val_data=val)
        else:
            model.fit(x_train, y_train, params.batch_size, epochs, 
                  callbacks, verbose)

        e_train = df_train.event.values
        t_train = df_train.duration.values
        e_test = df_test.event.values
        t_test = df_test.duration.values
        if not('deephit' == model_name):
            _ = model.compute_baseline_hazards()
        
        evaluate(model, (e_train, t_train), 
                 (x_train, e_train, t_train), 
                 train_scores)
        
        evaluate(model, (e_train, t_train), 
                 (x_test, e_test, t_test), 
                 test_scores)
        
        save['estimators'].append(model)
        out_survival, _ = generate_predictions(model, x_test, times)
        save['predictions'].append(out_survival)
        save['event'].append(e_test)
        save['duration'].append(t_test)
        save['pred_at_event'].append(generate_individual_predictions(model, x_test, t_test, e_test))
        
    test_scores.d_calibration = d_calibration(np.concatenate(save['event']), 
                                          np.concatenate(save['pred_at_event']))
    save['test_score'] = test_scores
    save['train_score'] = train_scores

    report(config.horizons, train=train_scores, test=test_scores)
    return dict(params=params_set, data=save)
    
