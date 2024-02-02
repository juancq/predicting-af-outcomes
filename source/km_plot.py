import pickle
import yaml
import pandas as pd
from types import SimpleNamespace

import model_utilities as util
from lifelines import KaplanMeierFitter

import matplotlib.pyplot as plt


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
    
    if y_var == 'castle_new_cardiac_arrest':
        event, duration = util.get_single_risk(y_var, data, config)
        title
    else:
        event, duration = util.get_competing_risk(data, config, assingle=1)

    df = pd.DataFrame(X)
    df['duration'] = duration/ 365.2425 * 12 
    df['observed'] = event
   
    ax = plt.subplot(111)
    km = KaplanMeierFitter()
    km.fit(df['duration'], event_observed=df['observed'], label='')
    km.plot_cumulative_density(ax=ax, legend=False, at_risk_counts=True)
    ax.set_xlim((0,36))
    ax.grid(visible=True, which='major', color='silver', linewidth=1.0, alpha=0.3)
    ax.set_xlabel('Months')
   
    plt.savefig(f'calibration/km_plot_cumulative_{y_var}.png',
                     bbox_inches='tight', dpi=400)

    plt.clf()
    ax = plt.subplot(111)
    km.plot_survival_function(ax=ax, legend=False, at_risk_counts=True)
    ax.set_xlim((0,36))
    ax.grid(visible=True, which='major', color='silver', linewidth=1.0, alpha=0.3)
    ax.set_xlabel('Months')
   
    plt.savefig(f'calibration/km_plot_survival_{y_var}.png',
                     bbox_inches='tight', dpi=400)    


if __name__ == '__main__':
    main()
