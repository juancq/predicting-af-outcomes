import pickle
import shap
import matplotlib.pyplot as plt



shap.initjs()


# plot shap values of GBT from pickle file
d = pickle.load(open('castle_gbt_shap_sep22.pk', 'rb'))

shap_test = d['shap']
X_test = d['X']
read_names = d['names']

# rename features for readability
rename = {'C03CA01':'furosemide', 
           'I10': 'essential (primary) hypertension',
          'Z92.1':'long term use of anticoagulants',
            'C01BC04':'flecainide',
          'R03BB04': 'tiotropium bromide',
          'I50.1': 'left ventricular failure',
          'A03FA01': 'metoclopramide',
          'J01FA06': 'roxithromycin',
          'I50.0': 'congestive HF',
          'Y92.22': 'place of occurrence, health service area',
          'C01BD01': 'amiodarone',
          'C10AA03': 'pravastin',
          'M04AA01': 'allopurinol',
          'C03DA01': 'spironolactone',
          'Z95.0': 'presence of cardiac device',
          'I47.2': 'ventricular tachycardia',
          'I25.11': 'atherosclerotic HD',
            'C01AA05': 'digoxin',
         'B01AC06': 'acetylsalicylic acid',
          'J01DB01': 'cefalexin',
          'E87.7': 'fluid overload',
          'E83.4': 'disorders of magnesium metabolism',
          'J18.9': 'pneumonia, unspecified',
          'N18.3': 'chronic kidney disease, stage 3 or higher',
          'A10BA02': 'metformin', 
          'B01AA03': 'warfarin',
          'C07AB07': 'bisoprolol',
            'J01CR02': 'amoxillin and beta-lactamase inhibitor', 
            'I48.0': 'paroxysmal AF',
           'U82.3': 'chronic hypertension',
          'Z72.0': 'tobacco use',
          'Z86.43': 'tobacco use history',
           'C01BD': 'class III antiarrhythmics',
            'S01AA01': 'chloramphenicol',
            'C10AA05': 'atorvastatin',
          'H02AB07': 'prednisone',
          'J01CA04': 'amoxicillin',
          'A02BC02': 'pantoprazole',
            'I48.9': 'AF and atrial flutter, unspecified',
         }

read_names_new = read_names.copy()
for old,new in rename.items():
    read_names_new[read_names.index(old)]=new

# generate SHAP plot for GBT predictions of composite outcome
shap.summary_plot(shap_test, X_test, feature_names=read_names_new, show=False, 
                  max_display=20)
plt.gcf().axes[-1].set_aspect(80)
plt.gcf().axes[-1].set_box_aspect(80)

plt.savefig('shap_plots/shap_summary_plot_gbt_castle_readable.png', 
            bbox_inches='tight', dpi=400)


# ## Major Bleeding

d = pickle.load(open('major_bleeding_dsm_shap_sep22.pk', 'rb'))
shap_test = d['shap']
X_test = d['X']
read_names = d['names']

# major bleeding names
rename = {'C01BC04':'flecainide', 
          'J01CR02': 'amoxillin and beta-lactamase inhibitor', 
          'U82.3': 'chronic hypertension',
         'C01BD': 'class III antiarrhythmics',
         'B01AA03': 'warfarin',
         'Z72.0': 'tobacco use',
         'Y92.22': 'place of occurrence, health service area',
         'Z86.43': 'tobacco use history',
         'C10AA05': 'atorvastatin',
         'S01AA01': 'chloramphenicol',
         'I10': 'essential (primary) hypertension',
         'B01AF01': 'rivaroxaban',
          'Z12.1': 'screening neoplasm of intestinal tract',
          'B01AB05': 'enoxaparin',
          'I25.11': 'atherosclerotic HD',
          'D07AC01': 'betamethasone',
          'I48.0': 'paroxysmal AF',
          'B01AC04': 'clopidogrel',
          'N02AJ06': 'codeine and paracetamol'
         }

read_names_new = read_names.copy()
for old,new in rename.items():
    read_names_new[read_names.index(old)]=new

plt.clf()
# generate SHAP plot for DSM predictions of major bleeding events
shap.summary_plot(shap_test, X_test, feature_names=read_names, show=False, max_display=20)
plt.gcf().axes[-1].set_aspect(80)
plt.gcf().axes[-1].set_box_aspect(80)
plt.savefig('shap_plots/shap_summary_plot_dsm_major_bleeding.png', bbox_inches='tight', dpi=400)