import optuna
import pickle
import sys

import itertools
import pandas as pd
import numpy as np
from collections import Counter
from types import SimpleNamespace

import statsmodels.stats.api as sms

from sklearn.preprocessing import MultiLabelBinarizer
from sksurv.metrics import concordance_index_ipcw
from sksurv.metrics import integrated_brier_score

from calibration import calibration_curve, d_calibration, xcal
  

def get_evaluation_times(time, horizons):
    """
    The function "get_evaluation_times" calculates the time horizons and the time range between the 10th
    and 90th percentiles of a given time array.
    
    :param time: The `time` parameter is a list or array of time values. It represents the time taken
    for some process or event to occur
    :param horizons: The `horizons` parameter is a list of quantiles that you want to calculate for the
    `time` data
    :return: a dictionary with two keys: 'time_horizons' and 'time_10_90'. The value associated with the
    'time_horizons' key is a list of quantiles of the input time array based on the specified horizons.
    The value associated with the 'time_10_90' key is an array of values ranging from the 10th
    percentile to the
    """
    time_horizons = np.quantile(time, horizons).tolist()
    lower, upper = np.quantile(time, [.1, .9])
    time_10_90 = np.arange(lower, upper+1)
    print(lower, upper)
    
    return {'time_horizons':time_horizons, 'time_10_90': time_10_90}
    

def print_eval(scores, horizons):
    """
    The function `print_eval` prints evaluation metrics and confidence intervals for a given set of
    scores and horizons.
    
    :param x: The parameter `x` in the `conf_int` function represents the data for which the confidence
    interval is calculated. It is used to calculate the lower and upper bounds of the confidence
    interval
    :param lowerl: The parameter `lowerl` is used to specify the lower limit for the confidence
    interval. It is an optional parameter and its default value is `None`. If a value is provided for
    `lowerl`, it will be used as the lower limit for the confidence interval calculation. If no value
    is provided
    :param upperl: The parameter `upperl` is an optional argument that specifies the upper limit for
    the confidence interval. If `upperl` is provided, the upper limit of the confidence interval will
    be the minimum value between the calculated upper limit and `upperl`. If `upperl` is not provided,
    the
    :return: The function `print_eval` does not return any value. It only prints the evaluation metrics
    and their confidence intervals.
    """

    def conf_int(x, lowerl=None, upperl=None):
        lower, upper = sms.DescrStatsW(x).tconfint_mean()
        if lowerl is not None:
            lower = max(lowerl, lower)
        if upperl is not None:
            upper = min(upperl, upper)
        
        return lower, upper

    for i, horizon in enumerate(horizons):
        print('---------')
        print(f'For {horizon} quantile,')
        print(f'TD concordance index: {np.mean(scores.cis[i]):.2f} ({np.std(scores.cis[i]):.2f})')    
        lower, upper = conf_int(scores.cis[i], 0, 1)
        print(f'TD concordance index: ({lower:.2f}, {upper:.2f})')        
        #print(f'AUC: {np.mean(scores.roc_auc[i]):.2f} ({np.std(scores.roc_auc[i]):.2f})')
        print(f'ECE: {np.mean(scores.ece[i]):.3f} ({np.std(scores.ece[i]):.3f})')
        lower, upper = conf_int(scores.ece[i], 0, 1)
        print(f'ECE: ({lower:.3f}, {upper:.3f})')   
    
    print(f'D-cal value: {np.mean(scores.dcal_value):.3f} ({np.std(scores.dcal_value):.3f})')
    lower, upper = conf_int(scores.dcal_value, lowerl=0)
    print(f'D-cal value: ({lower:.3f}, {upper:.3f})')  
    print(f'Integrated brier score: {np.mean(scores.brs):.3f} ({np.std(scores.brs):.3f})')
    lower, upper = conf_int(scores.brs, 0, 1)
    print(f'Integrated brier score: ({lower:.3f}, {upper:.3f})')
    
    print(f'D-cal: {sum(scores.dcal)} / {len(scores.dcal)}') 
    
    if hasattr(scores, 'd_calibration'):
        print('D-calibration p-value: {:.3f}'.format(scores.d_calibration['p_value']))
        print(scores.d_calibration)

        
def report(horizons, train=None, test=None):
    if train:
        print('---------')
        print('Train scores:')
        print_eval(train, horizons)
    if test:
        print('---------')
        print('Test scores:')
        print_eval(test, horizons)
        

def get_score_list(times):
    """
    The function `get_score_list` returns two objects, `train_scores` and `test_scores`, which are
    initialized with empty lists for various score metrics.
    
    :param times: The "times" parameter is a list that represents the different time points or
    iterations for which scores are being calculated
    :return: two objects: `train_scores` and `test_scores`.
    """
    score_list = lambda: [[] for _ in range(len(times))]
    scores_ = lambda: SimpleNamespace(times=times, 
                                      cis=score_list(),
                                      roc_auc=score_list(), 
                                      brs=[],
                                      ece=score_list(),
                                      dcal_value=[],
                                      dcal=[],
                                      xcal=[])    
    train_scores = scores_()
    test_scores = scores_()
    
    return train_scores, test_scores


def generate_predictions(model, x_test, times, competing_risk=False):
    """
    The function `generate_predictions` takes a machine learning model, test data, and time points as
    input, and returns the predicted survival probabilities and risks for each time point.
    
    :param model: The model is the machine learning model that has been trained to make predictions on
    survival data. It could be any model such as Random Survival Forest (RSF), Cox proportional hazards
    model, Gradient Boosting Trees (GBT), DeepHit, etc
    :param x_test: The input data for which you want to generate predictions. It should be a numpy array
    or pandas DataFrame with the same number of features as the training data
    :param times: The `times` parameter is a list of time points at which you want to generate
    predictions for survival probabilities. It represents the specific time points for which you want to
    obtain survival probabilities from the model
    :param competing_risk: The `competing_risk` parameter is a boolean flag that indicates whether the
    model should consider competing risks when generating predictions. Competing risks refer to the
    presence of multiple possible events that can occur, and the model needs to account for the
    possibility of each event when estimating survival probabilities. If `comp, defaults to False
    (optional)
    :return: The function `generate_predictions` returns two variables: `out_survival` and `out_risk`.
    """
    if model.name_ == 'rsf':
        out_survival = model.predict_survival_function(x_test, return_array=True)
        df_surv = pd.DataFrame(out_survival, columns=model.event_times_)
        out_survival = get_survival_at_times(df_surv.T, times)
        
    elif model.name_ in ['coxnet', 'gbt', 'component_gbt', 'cox']:
        step_func = model.predict_survival_function(x_test)
        out_survival = np.array([fn(times) for fn in step_func]) 
        
    elif model.name_ in ['deephit', 'coxcc', 'coxtime']:
        df_surv = model.predict_surv_df(x_test)
        #ev = EvalSurv(df_surv, durations_test, events_test)
        #l_cindex.append(ev.concordance_td())
        out_survival = get_survival_at_times(df_surv, times)
    elif model.name_ == 'dsm' and competing_risk:
        out_survival = model.predict_survival(x_test, times, risk=competing_risk)
    elif model.name_ in ['dsm', 'dcm', 'deepsurv']:
        out_survival = model.predict_survival(x_test, times)
    else:
        print(f'Error when generating predictions for {model.name_}', file=sys.stderr)

    out_risk = 1 - out_survival
    return out_survival, out_risk


def generate_individual_predictions(model, x_test, t_test, e_test, competing_risk=False):
    """
    The function `generate_individual_predictions` takes a machine learning model, test data, and event
    times as input, and returns individual survival predictions based on the model.
    
    :param model: The model is the machine learning model that has been trained to make predictions. It
    could be any model such as CoxNet, GBT, Component GBT, RSF, Cox, DeepHit, CoxCC, CoxTime, DSM, DCM,
    or DeepSurv
    :param x_test: The input features for the test data. It is a numpy array of shape (n_samples,
    n_features), where n_samples is the number of samples in the test data and n_features is the number
    of features
    :param t_test: t_test is an array containing the observed event times for the test data
    :param e_test: The parameter `e_test` represents the event indicator for each test sample. It is a
    binary variable that indicates whether an event (e.g., death, failure) has occurred for each sample.
    If `e_test[i]` is 1, it means an event has occurred for the i-th
    :param competing_risk: The parameter `competing_risk` is a boolean flag that indicates whether the
    model should consider competing risks when generating predictions. If `competing_risk` is set to
    `True`, the model will take into account the presence of competing risks in the data. If it is set
    to `False, defaults to False (optional)
    :return: an array of individual survival predictions.
    """
          
    if model.name_ in ['coxnet', 'gbt', 'component_gbt', 'rsf', 'cox']:
        step_func = model.predict_survival_function(x_test)
        #print('in generation', max(t_test), min(t_test))
        if hasattr(model, 'event_times_'):
            t_test = t_test.copy()
            max_t = max(model.event_times_)
            min_t = min(model.event_times_)
            t_test[t_test > max_t] = max_t
            t_test[t_test < min_t] = min_t
        out_survival = np.array([fn(t) for fn,t in zip(step_func, t_test)]) 
        
    elif model.name_ in ['deephit', 'coxcc', 'coxtime']:
        df_surv = []
        for x_i,t in zip(x_test.astype(np.double), t_test):
            preds = model.predict_surv_df(np.array([x_i]).astype('float32'))
            df_surv.append(get_survival_at_times(preds, [t]))        
        out_survival = np.array(df_surv).flatten()

    elif model.name_ == 'dsm' and competing_risk:
         out_survival = np.array([model.predict_survival(np.array([x_i]), [t], risk=competing_risk) for x_i,t in zip(x_test.astype(np.double), t_test)])
    elif model.name_ in ['dsm', 'dcm', 'deepsurv']:
        out_survival = np.array([model.predict_survival(np.array([x_i]), [t]) for x_i,t in zip(x_test.astype(np.double), t_test)])
    else:
        print(f'Error when generating predictions for {model.name_}', file=sys.stderr)
    
    return out_survival.flatten()


def duration_event_tuple(e_train, t_train, e_test, t_test):
    """
    The function duration_event_tuple takes in two lists of events and durations, and returns them as
    numpy arrays with structured data types.
    
    :param e_train: A list of boolean values representing whether an event occurred or not during
    training
    :param t_train: The parameter `t_train` represents the duration of events in the training dataset
    :param e_test: The parameter `e_test` represents a list of boolean values indicating whether an
    event occurred or not during a specific time interval
    :param t_test: The parameter `t_test` represents the time duration of events in the test dataset
    :return: two numpy arrays, `et_train` and `et_test`, which are created by combining the `e_train`
    and `t_train` arrays into a structured array with fields 'e' and 't', and combining the `e_test` and
    `t_test` arrays into another structured array with the same fields.
    """
    et_train = np.array(list(zip(e_train, t_train)), 
                       dtype=[('e', bool), ('t', float)])
    et_test = np.array(list(zip(e_test, t_test)), 
                       dtype=[('e', bool), ('t', float)])
    return et_train, et_test


def evaluate(model, y_train, test, scores, recurrent=False, competing=False):
    """
    The function evaluates the performance of a survival model by calculating various metrics such as
    concordance index, integrated Brier score, and calibration curve.
    
    :param model: The model is the machine learning model that has been trained to make predictions. It
    could be any model such as a neural network, random forest, or support vector machine
    :param y_train: The target variable for the training data. It consists of two arrays: e_train (event
    indicator) and t_train (time-to-event)
    :param test: The `test` parameter is a tuple containing the test data. It consists of three arrays:
    :param scores: The `scores` parameter is an object that stores the evaluation metrics for the model.
    It has the following attributes:
    :param recurrent: The `recurrent` parameter is a boolean flag that indicates whether the data is
    recurrent or not. If `recurrent=True`, it means that the data has a recurrent structure, where each
    instance can have multiple events and event times. If `recurrent=False`, it means that the data is
    not, defaults to False (optional)
    :param competing: The parameter `competing` is a boolean flag that indicates whether the model is
    trained to handle competing risks. If `competing=True`, it means that the model is trained to handle
    multiple types of events that can occur simultaneously, such as death and disease recurrence. If
    `competing=False`, it, defaults to False (optional)
    """

    times = scores.times
    
    e_train, t_train = y_train
    x_test, e_test, t_test = test
    
    lower, upper = np.quantile(t_test[e_test > 0], [.1, .9])
    time_10_90 = np.arange(lower, upper).tolist()
    
    et_train = np.array(list(zip(e_train, t_train)), 
                           dtype=[('e', bool), ('t', float)])
    et_test = np.array(list(zip(e_test, t_test)), 
                           dtype=[('e', bool), ('t', float)])
    
    out_survival, out_risk = generate_predictions(model, x_test, times, competing)
        
    if recurrent: 
        et_train = np.array([(e_train[i][j],t_train[i][j]) for i in range(len(e_train))\
                     for j in range(len(e_train[i]))],
                    dtype=[('e', bool), ('t', float)])
            
        et_test = np.array([(e_test[i][j],t_test[i][j]) for i in range(len(e_test))\
                 for j in range(len(e_test[i]))],
                dtype=[('e', bool), ('t', float)])
            
        # this is the indexing for recurrent data
        #out_risk = np.concatenate([model.predict_risk(x_test, [i]) for i in times], axis=1)
        #out_survival = np.concatenate([model.predict_survival(x_test, [i]) for i in times], axis=1)

    for i in range(len(times)):
        try:
            cis_i = concordance_index_ipcw(et_train, et_test, out_risk[:,i],
                                           times[i])[0]
            scores.cis[i].append(cis_i)
        except Exception as e:
            print(out_risk[:,i])
            print('error concordance index ipcw')
            print(e)
            pass
    
    if time_10_90 is not None:
        out_survival_10_90, _ = generate_predictions(model, x_test, time_10_90)
        ibs = integrated_brier_score(et_train, et_test, out_survival_10_90, 
                                       time_10_90)
        scores.brs.append(ibs)
    
    for i in range(len(times)):
        a, b, ece_i = calibration_curve(out_survival[:,i], e_test, t_test, 
                                        None, None, times[i], n_bins=10)
        scores.ece[i].append(ece_i)
     
    d_out_survival = generate_individual_predictions(model, x_test, t_test, e_test, 
                                                     competing)
    result = d_calibration(e_test, d_out_survival)
    
    scores.dcal_value.append(result['d_calibration'])
    scores.dcal.append(int(result['p_value']>=0.05))
    
    result = xcal(d_out_survival, e_test, nbins=10)
    scores.xcal.append(result)


def trial_metric_single(et_train, et_test, out_risk, times):
    '''
    Calculate concordance index of first event horizon. 
    '''
    try:
        cis = concordance_index_ipcw(et_train, et_test, out_risk[:,0],
                               times[0])[0]
        return cis
    except Exception as e:
        print('Too few events in trial')
        print('-->>', e)
    return 0


def trial_metric(et_train, et_test, out_risk, times):
    '''
    Calculate concordance index of all event horizons and return mean. 
    '''
    scores = []
    for i in range(len(times)):
        try:
            cis_i = concordance_index_ipcw(et_train, et_test, out_risk[:,i],
                                   times[i])[0]
            scores.append(cis_i)
        except Exception as e:
            print('Too few events in trial')
            print(e)
    return np.mean(scores)


def trial_metric_integrated_brier(et_train, et_test, out_risk, times):
    try:
        score = integrated_brier_score(et_train, et_test, out_risk, times)
    except Exception as e:
        print('Too few events in trial')
        print(e)
    return score


def get_survival_at_times(df_surv, times):
    index_surv = df_surv.index.values
    assert pd.Series(index_surv).is_monotonic_increasing, 'need monotonic increasing'
    
    nans = np.full(df_surv.shape[1], np.nan)
    not_in_index = list(set(times) - set(index_surv))
    for idx in not_in_index:
        df_surv.loc[idx] = nans
    return df_surv.sort_index(axis=0).interpolate().interpolate(method='bfill').T[times].values
    

def get_time_invariant_features(data, code_vocab, config, 
                                return_feature_names=False,
                                codes_only=False):
    patient_codes = []
    patient_features = []
    inv_code_vocab = {v:k for k,v in code_vocab.items()}
  
    code_counter = Counter()
    #exclude_codes = set(['I48', 'I48.0', 'I48.9', 'I48.1', 'I48.2', 
    #                   'I48.3', 'I48.4', '38290', '38290-01', '38287-02'])
    
    count_codes = []
    
    for patient in data:
        codes = list(set(itertools.chain(*patient.visit_codes)))
        patient_codes.append(codes)
        age_sex = patient.data.iloc[-1][['age_recode', 'sex']].astype(float).tolist()
        patient_features.append(age_sex)
        code_counter.update(codes)
        
        if config.feature_type == 'count':
             count_codes.append(list(itertools.chain(*patient.visit_codes)))
           
    if config.code_threshold:
        # find codes that less than X people have
        to_drop = set([_code for _code,_count in code_counter.items() if _count < config.code_threshold])
    
        # drop codes that have a per person count below threshold
        new_patient_codes = [[i for i in codeset if i not in to_drop] for codeset in patient_codes]
        unique_codes = set(itertools.chain(*new_patient_codes))
        
        code_vocab = {key:value for key,value in code_vocab.items() if value in unique_codes}
        inv_code_vocab = {v:k for k,v in code_vocab.items()}
        patient_codes = new_patient_codes
        
        if config.feature_type == 'count':
            count_codes = [[i for i in codeset if i not in to_drop]\
                           for codeset in count_codes]
    
    if config.feature_type == 'count':
        df_count = pd.DataFrame([Counter(x) for x in count_codes]).fillna(0)
        X_codes = df_count.values.astype(np.uint8)
    else:
        mlb = MultiLabelBinarizer()
        mlb.fit([code_vocab.values()])
        X_codes = mlb.transform(patient_codes).astype(np.uint8)
    
    print(f'Dataset shape: {X_codes.shape}')
    
    X = np.hstack((X_codes, np.array(patient_features))) 
    
    feature_names = [inv_code_vocab[k] for k in sorted(inv_code_vocab.keys())]
    feature_names.extend(['age', 'sex'])
    
    if return_feature_names:
        return X, feature_names
    else:
        return X


def get_individual_risk_event_duration(patient, risk):
    event = bool(patient.outcomes.get(risk, 0))
    if event:
        duration = patient.outcomes[risk+'_days']
    else:
        duration = 10000000

    return duration, event
    

def convert_to_cause_specific(event, risk=1):
    mask = event == risk
    if not mask.any():
        print(f'Competing risk {risk} not found')
        return
    
    event_ = np.copy(event)
    event_[mask] = 1
    # censor all other competing events
    event_[~mask] = 0
    
    return event_
        

def get_validation(x, duration, event, frac=0.15, num_risks=1):    
    df = pd.DataFrame(x)
    df['duration'] = duration
    df['event'] = event

    df_1 = [df[df.event==(i+1)].sample(frac=frac) for i in range(num_risks)]
    df_0 = df[df.event==0].sample(frac=frac)
    df_val = pd.concat([df_0] + df_1)
    df_train = df.drop(df_val.index)
    
    x_train = df_train.drop(columns=['duration', 'event']).values
    t_train, e_train = df_train.duration.values, df_train.event.values
    train =  (x_train, t_train, e_train)
    
    x_val = df_val.drop(columns=['duration', 'event']).values
    t_val, e_val = df_val.duration.values, df_val.event.values
    val = (x_val, t_val, e_val)
    
    return train, val


def save_params(params, fname, model_name):
    pickle.dump(params, 
                open(f'hyper_parameters/{model_name}_{fname}.pk', 'wb'))
    

def get_optuna_study(config):
    study = optuna.create_study(direction='maximize')
    
    
def get_competing_risk(data, config, assingle=None):
    duration = []
    event = []
    
    if not config.max_followup_months: 
        max_followup = max([patient.outcomes.get('end_of_followup_date') for patient in data])
    
    death = 0
    for patient in data:
        event_i = []
        duration_i = []
        death_i = 0
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
        death += death_i
    
    assert len(set(event)) == len(config.competing_risks)+1, 'error in competing risk processing'

    event = np.array(event, dtype=np.double)
    duration = np.array(duration, dtype=np.double)    
    
    if np.isnan(duration).any():
        print('NAN in duration array, stopping.')
        return
    elif np.isnan(event).any():
        print('NAN in event array, stopping.')
        return
    
    censor_stats(event, duration)
    
    print((event==1).sum())
    
    if assingle is not None:
        if assingle > 0:
            event = convert_to_cause_specific(event, assingle)
        else:
            print('Invalid risk value')

    return event, duration


def get_single_risk(y_var, data, config):
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
    
    censor_stats(event, duration)
    
    return event, duration


def censor_stats(event, duration):
    censored = np.array(duration)[np.array(event)==0]
    print('censored times', censored, 'max censor time', censored.max())
    print('censored before end of study', (censored < censored.max()).sum())

    if np.isnan(duration).any():
        raise Exception('NAN in duration array, stopping.')
    if np.isnan(event).any():
        raise Exception('NAN in duration array, stopping.')