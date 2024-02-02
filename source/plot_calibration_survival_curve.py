import pickle
import yaml
import pandas as pd
import numpy as np
from types import SimpleNamespace

from lifelines import CRCSplineFitter
from lifelines.fitters.coxph_fitter import CoxPHFitter
import model_utilities as util

import matplotlib.pyplot as plt


def survival_curve_modified(df, duration, event, t0, ax=None, label=''):
    def ccl(p):
        return np.log(-np.log(1-p))
    
    if ax is None:
        ax = plt.gca()
        
    T = 'duration'
    E = 'event'
    
    event = (event > 0).astype(int)
        
    predictions_at_t0 = np.clip(df, 1e-10, 1-1e-10)
    
    prediction_df = pd.DataFrame({f'ccl_at_{t0}': ccl(predictions_at_t0),
                                  T:duration, E:event})
    
    n_knots = 3
    regressors = {'beta_': [f'ccl_at_{t0}'], 'gamma0_': '1', 
                  'gamma1_': '1', 'gamma2_': '1'}
    
    crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.000001)
    crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)
    
    x = np.linspace(np.quantile(predictions_at_t0, 0.01), 
                    np.quantile(predictions_at_t0, 0.99), 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({f'ccl_at_{t0}': ccl(x)}),
                                                        times=[t0]).T.squeeze()
    ax.set_title(label)
    color = 'tab:red'
    ax.plot(x, y, label='smoothed calibration curve', color=color)
    ax.set_xlabel(f'Predicted probability of t <= {t0}')
    ax.set_ylabel(f'Observed probability of t <= {t0}', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax.plot(x, x, c='k', ls='--')
    ax.legend()
    
    color = 'tab:blue'
    twin_ax = ax.twinx()
    twin_ax.set_ylabel('Count of predicted prob', color=color)
    twin_ax.tick_params(axis='y', labelcolor=color)
    twin_ax.hist(predictions_at_t0, alpha=0.3, bins='sqrt', color=color)
    
    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()-predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    print(f'ICI = {ICI:.3f}')
    print(f'E50 = {E50:.3f}')
    
    return ax

    
def survival_curve(df, duration, event, t0, ax=None, label='', model='crc'):
    def ccl(p):
        return np.log(-np.log(1-p))
    
    if ax is None:
        ax = plt.gca()
        
    T = 'duration'
    E = 'event'
    
    predictions_at_t0 = np.clip(df, np.quantile(df, 0.01), np.quantile(df, 0.99))
    #predictions_at_t0 = np.clip(df, 1e-10, 1-1e-10)

    prediction_df = pd.DataFrame({f'ccl_at_{t0}': ccl(predictions_at_t0),
                                  T:duration, E:event})
    
    n_knots = 3
    regressors = {'beta_': [f'ccl_at_{t0}']} | {f'gamma{i}_':'1' for i in range(n_knots)} 
    
    if model == 'crc':
        crc = CRCSplineFitter(n_baseline_knots=n_knots, penalizer=0.00001)
        crc.fit_right_censoring(prediction_df, T, E, regressors=regressors)#,
                            #robust=True)
    elif model == 'cox':
        crc = CoxPHFitter(baseline_estimation_method='spline', n_baseline_knots=n_knots,
                          penalizer=0.00001)
        crc.fit_right_censoring(prediction_df, T, E, step_size=0.05)
    
    #x = np.linspace(np.clip(predictions_at_t0.min()-0.01, 0, 1), 
    #                np.clip(predictions_at_t0.max()+0.01, 0, 1), 100)

    #lower_bound = np.clip(predictions_at_t0.min()-0.01, 0, 1)
    #upper_bound = np.clip(predictions_at_t0.max()+0.01, 0, 1)    
    lower_bound = np.quantile(predictions_at_t0, 0.01)
    upper_bound = np.quantile(predictions_at_t0, 0.99)
    x = np.linspace(lower_bound, upper_bound, 100)
    y = 1 - crc.predict_survival_function(pd.DataFrame({f'ccl_at_{t0}': ccl(x)}),
                                                        times=[t0]).T.squeeze()
    ax.set_title(label)
    color = 'tab:red'
    ax.plot(x, y, label='smoothed calibration curve', color=color)
    ax.set_xlabel(f'Predicted probability of t <= {t0}')
    ax.set_ylabel(f'Observed probability of t <= {t0}', color=color)
    ax.tick_params(axis='y', labelcolor=color)
    
    ax.plot(x, x, c='k', ls='--')
    ax.legend()
    
    
    predictions_at_t0_truncated = np.clip(predictions_at_t0, 
                                          lower_bound, 
                                          upper_bound)
    
    color = 'tab:blue'
    twin_ax = ax.twinx()
    twin_ax.set_ylabel('Count of predicted prob', color=color)
    twin_ax.tick_params(axis='y', labelcolor=color)
    bins = 10 #'sqrt'
    twin_ax.hist(predictions_at_t0_truncated, alpha=0.3, bins='sqrt', color=color)
    
    ax.set_title(label)
    
    deltas = ((1 - crc.predict_survival_function(prediction_df, times=[t0])).T.squeeze()-predictions_at_t0).abs()
    ICI = deltas.mean()
    E50 = np.percentile(deltas, 50)
    print(f'ICI = {ICI:.3f}')
    print(f'E50 = {E50:.3f}')
    
    return ax

def bin_calibration_plot(df, duration, event, t0, ax=None, label=''):
    num_people = len(df)
    
    assert num_people == len(duration) and len(duration) == len(event)
    
    df = pd.DataFrame({'risk': df, 't': duration, 'event':event})
    
    df['risk'] = df['risk'] * 100
    
    df['quantile'] = pd.qcut(df['risk'], q=10, labels=list(range(10)))
    df = df.groupby('quantile').agg({'risk': 'mean', 'event': 'sum'})
    
    df['event_perc'] = df['event'] / num_people * 100
    df = df.reset_index()
    
    ax.scatter(df['event_perc'], df['risk'], c='cornflowerblue')
    
    ylim = max([df['event_perc'].max(), df['risk'].max()]) + 1
    
    ax.set_xlim(0, ylim)
    ax.set_ylim(0, ylim)
    ax.plot([0, ylim], [0, ylim], color='black', linewidth=1)
    
    ax.set_xlabel('Observed events [%]')
    ax.set_ylabel('Mean predicted year risk [%]')
    ax.set_title(label)
    
    return ax
    

def main():

    with open('shallow_model_baseline_survival.yml') as fin:
          config = yaml.full_load(fin)
          config = SimpleNamespace(**config)

    data_meta = pickle.load(open(config.data, 'rb'), encoding='bytes')
    data = data_meta['data']
    code_vocab = pickle.load(open(config.types, 'rb'))

    X, feature_names = util.get_time_invariant_features(data, code_vocab, config,
                                                   return_feature_names=True)
    y_var = 'castle_new_cardiac_arrest'
    models = ['deepsurv']#, 'dsm']

    df = pd.DataFrame(X, columns=feature_names)
    
    for model_name in models:
        if y_var == 'castle_new_cardiac_arrest':
            event, duration = util.get_single_risk(y_var, data, config)
        else:
            event, duration = util.get_competing_risk(data, config)
        
        times = np.quantile(duration[event>0], config.horizons).tolist()
        
        if y_var == 'major_bleeding':
            predictions = pickle.load(open('calibration/dsm_predictions_bleeding.pk', 'rb'))
        else:
            # load trained estimators
            folds = pickle.load(open(f'results/folds/{model_name}_castle_new_cardiac_arrest_09_3yearback_redo3.pk', 'rb'))

        survival_set = np.concatenate(folds['predictions'])
        t_set = np.concatenate(folds['duration'])
        e_set = np.concatenate(folds['event'])

        quants = [25, 50, 75]
    
        path = 'results/folds/calibration'
        for i, t0 in enumerate(times):
            t0 = f'{t0:.0f}'
            for fitter in ['crc', 'cox']:
                fig, ax = plt.subplots()
                try:
                    survival_curve(1-survival_set[:,i], t_set, e_set,t0, ax, 
                                   label=f'T {quants[i]} Percentile')
                    plt.savefig(f'{path}/{y_var}_{model_name}_folds_{t0}_final_09_b3_{fitter}_redo3.png',
                                 bbox_inches='tight', dpi=400)
                except:
                    print(f'Could not generate test set survival curve for {model_name} at {t0}')
            
            continue
            fig, ax = plt.subplots()
            bin_calibration_plot(1-survival_set[:,i], t_set, e_set, t0, ax, 
                           label=f'T {quants[i]} Percentile Test {model_name}')
            plt.savefig(f'{path}/bin_plot_{y_var}_{model_name}_folds_{t0}_final_09_b3_test.png',
                         bbox_inches='tight', dpi=400)
            

            
if __name__ == '__main__':
    main()