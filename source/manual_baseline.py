import pickle
import re
import yaml
import pandas as pd
import numpy as np
from types import SimpleNamespace
from sklearn.model_selection import RepeatedStratifiedKFold
from collections import OrderedDict
from sksurv.metrics import integrated_brier_score, concordance_index_ipcw
from calibration import calibration_curve, d_calibration, xcal
import model_utilities as util


def chadsvasc(patient):
    score = 0
    if patient['age'] >= 65 and patient['age'] < 75:
        score += 1
    elif patient['age'] >= 75:
        score += 2
    
    if patient['sex'] == 2:
        score += 1
    
    if patient['congestive_hf']:
        score += 1
    
    if patient['hypertension']:
        score += 1
        
    if patient['stroke_tia']:
        score += 1

    if patient['mi'] or patient['coronary_artery'] or patient['peripheral'] or patient['atherosclerosis']:
        score += 1
            
    if patient['diabetes']:
        score += 1

    return score


def hasbled(patient):
    score = 0
    if patient['age'] > 65:
        score += 1
    
    if patient['hypertension']:
        score += 1
        
    if patient['stroke_tia']:
        score += 1

    if patient['kidney_disease']:
        score += 1
    
    if patient['liver_disease']:
        score += 1

    if patient['major_bleeding']:
        score += 1
    
    if patient['alcoholism']:
        score += 1
            
    if patient['antiplatelet'] or patient['nsaid']:
        score += 1
    
    return score


def build_re_patterns():
    """
    The function builds regular expression patterns for different medical feature sets.
    :return: a dictionary containing regular expressions for various medical conditions and 
    their corresponding codes.
    """
    feature_sets = [
        ['diabetes', 'E1[0-4]'],
        ['hypertension', 'I1[0-5]'],
        ['stroke_tia', 'I63,G45.9,I69.3'],
        ['congestive_hf', 'I50'],
        ['mi', 'I21'],
        ['coronary_artery', 'I25.0-.19'],
        ['peripheral', 'I79.[28],I70.2,I73.[18]'],
        ['atherosclerosis', 'I70.0'],
        # has-bled
        ['alcoholism', 'E24.4,E52,G31.2,G62.1,G72.1,I42.6,K70,K86.0,O35.4,T51,Z71.4,Z72.1'],
        ['kidney_disease', 'N0[0-8],N1[4-6],N18.[1-5],N19'],
        ['liver_disease', 'B18,K70.[0-39],K71.[3-57],K7[34],K76.[02-489],Z94.4,I85.[09],I86.4,I98.2,K70.4,K71.1,K72.[19],K76.[5-7]'],
        ['nsaid', 'M01A'],
        ['antiplatelet', 'B01AC'],
        ['major_bleeding', 'I85.0,K22.[16],K25.[0246],K26.[0246],K27.[0246],'+
         'K28.[046],K29.[0-9]1,K31.82,K55.22,K57.[0-9]1,K57.[0-9]3,K62.5,K92.[012],'+
         'S06.3[45678],S06.[456],'+
         'I31.2,K66.1,M25.0,R04.[12],R31,R58']
        ]
    
    feature_extraction_re = {}
    for name,codes in feature_sets:
        split_codes = [c.strip().replace('.','\.') for c in codes.split(',')]
        regex = '|'.join([f'(?:{c})' for c in split_codes])
        regex = re.compile(regex)
        feature_extraction_re[name] = regex

    return feature_extraction_re


def risk_function_chadsvasc(score):
    if score == 0:
        return 0.2
    elif score == 1:
        return 0.6 
    elif score == 2:
        return 2.2 
    elif score == 3:
        return 3.2 
    elif score == 4:
        return 4.8
    elif score == 5:
        return 7.2 
    elif score == 6:
        return 9.7 
    elif score == 7:
        return 11.2 
    elif score == 8:
        return 10.8 
    elif score == 9:
        return 12.23


def hasbled_risk(score):
    if score == 0:
        return 0.5
    elif score == 1:
        return 2.1 
    elif score == 2:
        return 3.6 
    elif score == 3:
        return 5.5 
    elif score == 4:
        return 7.8
    elif score == 5:
        return 9.0
    elif score >= 6:
        return 27.0 


def get_monthly_risk_chadvasc2(score):
    return 1-(1-risk_function_chadsvasc(score)/100)**(1/12)

def get_monthly_risk_hasbled(score):
    return 1 - (1-hasbled_risk(score)/100)**(1/12)


def main():
    """
    Calculates discrimination and calibration performance metrics of HAS-BLED and CHAD2VASC2.
    This is baseline metric.
    """
    with open('shallow_model_baseline_survival.yml') as fin:
          config = yaml.full_load(fin)
          config = SimpleNamespace(**config)

    data_meta = pickle.load(open(config.data, 'rb'), encoding='bytes')
    data = data_meta['data']
    code_vocab = pickle.load(open(config.types, 'rb'))

    X, feature_names = util.get_time_invariant_features(data, code_vocab, config, 
                                    return_feature_names=True)
    
    regex = build_re_patterns()
    table_columns = OrderedDict()
    
    # checks if a feature is present in the data
    for name,cond_regex in regex.items():
        table_columns[name] = [i for i in feature_names if bool(cond_regex.search(i))]

    event, duration = util.get_competing_risk(data, config)
    
    competing_risks = len(config.competing_risks)    
    df = pd.DataFrame(X, columns=feature_names)

    for col_name, cols in table_columns.items():
        df[col_name] = df[cols].any(axis=1)
    
    #------------
    # use chadvasc2 score as single feature with age and sex
    chad = df.apply(chadsvasc, axis=1)
    df['chad'] = chad
    
    hasbled_score = df.apply(hasbled, axis=1)
    df['hasbled'] = hasbled_score
    df = df[['hasbled', 'chad']].copy()
    #------------
    
    df['duration'] = duration
    df['event'] = event
    
    X = df.drop(columns=['duration', 'event']).values
    event = np.copy(df.event.values)
    
    main_split = RepeatedStratifiedKFold(n_splits=10, n_repeats=1, random_state=123)
    
    # calculate cause specific predictive performance of hasbled and chadvasc2
    for risk in range(competing_risks): 
        print(f'Competing Risk {risk+1}')
        print(f'{config.competing_risks[risk]}')
        event_i = util.convert_to_cause_specific(event, risk+1)
    
        time_sets = util.get_evaluation_times(duration[event_i==1], config.horizons)
        times = time_sets['time_horizons']

        train_scores, test_scores = util.get_score_list(times)
    
        all_ind_predictions = []
        all_test = []
        # calculate metrics using cross-validation
        for train, test in main_split.split(X, event_i):
            
            t_train, e_train = duration[train], event_i[train]
            x_test, t_test, e_test = X[test], duration[test], event_i[test]
            
            y_train, y_test = util.duration_event_tuple(e_train, t_train, e_test, t_test)
           
            lower, upper = np.quantile(t_test[e_test > 0], [.1, .9])
            _time_10_90 = np.arange(lower, upper).tolist()
           
            if risk == 0:
                risk_func = get_monthly_risk_hasbled
            else:
                risk_func = get_monthly_risk_chadvasc2
                
            monthly_risk = np.array(list(map(lambda t: risk_func(t), x_test[:,risk])))
            ind_risk_at_event = np.array([risk_func(x_i) * t/12 for x_i,t in zip(x_test.astype(np.double)[:,risk], t_test)])
            risk_at_quantiles = np.array([monthly_risk*t/12 for t in times])
            
            all_ind_predictions.extend(ind_risk_at_event.tolist())
            all_test.extend(e_test.tolist())
            
            # calculate metrics at the event horizons
            for i in range(len(times)):
                cis_i = concordance_index_ipcw(y_train, y_test, risk_at_quantiles[i,:],
                                           times[i])[0]
                test_scores.cis[i].append(cis_i)
                
                _, _, ece_i = calibration_curve(1-risk_at_quantiles[i,:], e_test, t_test, 
                                        None, None, times[i], n_bins=10)
                test_scores.ece[i].append(ece_i)
            
            # integrated brier score calculation
            risk_at_10_90 = np.array([monthly_risk*t/12 for t in _time_10_90])

            ibs = integrated_brier_score(y_train, y_test, (1-risk_at_10_90).T, 
                                       _time_10_90)
            test_scores.brs.append(ibs)
            

            # d-calibration calculation
            result = d_calibration(e_test, 1-ind_risk_at_event)
    
            test_scores.dcal_value.append(result['d_calibration'])
            test_scores.dcal.append(int(result['p_value']>=0.05))
            
            result = xcal(ind_risk_at_event, e_test, nbins=10)
            test_scores.xcal.append(result)    

        util.report(config.horizons, train=train_scores, test=test_scores)
        result = d_calibration(all_test, 1-np.array(all_ind_predictions))
    
        print('aggregated d-calibration')
        print(result)
        print(int(result['p_value']>=0.05))
        

if __name__ == '__main__':
    main()
