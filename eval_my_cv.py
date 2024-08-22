import os
import json
import numpy as np
import matplotlib.pyplot as plt
import re

folder = "./dynamic_GLMiHMM_crossvals/infos/"

cvlls_0_03 = []
subjects_0_03 = []

cvlls_0_035 = []
subjects_0_035 = []

cvlls_0_025 = []
subjects_0_025 = []

cvlls_0_03_var_0_04 = []
subjects_0_03_var_0_04 = []

cvlls_0_03_var_0_02 = []
subjects_0_03_var_0_02 = []

pattern = r"(.*)_(\d+)_cvll_(-?\d+_.*)_(0\.02|0\.03|0\.04)_.*\.json"

for file in os.listdir(folder):
    if not file.endswith('.json'):
        continue
    infos = json.load(open(folder + file, 'r'))
    match = re.search(pattern, file)
    subject = match.group(1)

    if '/summarised_sessions/0_3/' in infos['file_name'] and infos['fit_variance'] == 0.03:
        if subject in subjects_0_03 and np.mean(infos['cross_val_preds'][-1000:]) in cvlls_0_03:
            print(file)
            os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
        cvlls_0_03.append(np.mean(infos['cross_val_preds'][-1000:]))
        subjects_0_03.append(subject)
    elif '/summarised_sessions/0_35/' in infos['file_name'] and infos['fit_variance'] == 0.03:
        if subject in subjects_0_035 and np.mean(infos['cross_val_preds'][-1000:]) in cvlls_0_035:
            print(file)
            os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
        cvlls_0_035.append(np.mean(infos['cross_val_preds'][-1000:]))
        subjects_0_035.append(subject)
    elif '/summarised_sessions/0_25/' in infos['file_name'] and infos['fit_variance'] == 0.03:
        if subject in subjects_0_025 and np.mean(infos['cross_val_preds'][-1000:]) in cvlls_0_025:
            print(file)
            os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
        cvlls_0_025.append(np.mean(infos['cross_val_preds'][-1000:]))
        subjects_0_025.append(subject)
    elif '/summarised_sessions/0_3/' in infos['file_name'] and infos['fit_variance'] == 0.04:
        if subject in subjects_0_03_var_0_04 and np.mean(infos['cross_val_preds'][-1000:]) in cvlls_0_03_var_0_04:
            print(file)
            os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
        cvlls_0_03_var_0_04.append(np.mean(infos['cross_val_preds'][-1000:]))
        subjects_0_03_var_0_04.append(subject)
    elif '/summarised_sessions/0_3/' in infos['file_name'] and infos['fit_variance'] == 0.02:
        if subject in subjects_0_03_var_0_02 and np.mean(infos['cross_val_preds'][-1000:]) in cvlls_0_03_var_0_02:
            print(file)
            os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
        cvlls_0_03_var_0_02.append(np.mean(infos['cross_val_preds'][-1000:]))
        subjects_0_03_var_0_02.append(subject)
    else:
        print("fits no schema: " + infos['file_name'])

# using the 5 cvlls dictionaries, plot boxplots for each of them
plt.figure()
plt.boxplot([np.exp(cvlls_0_03), np.exp(cvlls_0_035), np.exp(cvlls_0_025), np.exp(cvlls_0_03_var_0_04), np.exp(cvlls_0_03_var_0_02)], labels=['0.03', '0.035', '0.025', '0.03 var 4', '0.03 var 2'])
plt.xticks([1, 2, 3, 4, 5], ['0.03', '0.035', '0.025', '0.03 var 4', '0.03 var 2'])
plt.show()

quit()
plt.hist(np.exp(cvlls_0_03), bins=np.linspace(0.5, 1), alpha=0.3, label='0.03')
plt.hist(np.exp(cvlls_0_035), bins=np.linspace(0.5, 1), alpha=0.3, label='0.035')
plt.hist(np.exp(cvlls_0_025), bins=np.linspace(0.5, 1), alpha=0.3, label='0.025')
plt.hist(np.exp(cvlls_0_03_var_0_04), bins=np.linspace(0.5, 1), alpha=0.3, label='0.03 var 4')
plt.hist(np.exp(cvlls_0_03_var_0_02), bins=np.linspace(0.5, 1), alpha=0.3, label='0.03 var 2')

plt.axvline(np.mean(np.exp(cvlls_0_03)), color='blue', label='0.03 mean')
plt.axvline(np.mean(np.exp(cvlls_0_035)), color='red', label='0.035 mean')
plt.axvline(np.mean(np.exp(cvlls_0_025)), color='blue', label='0.025 mean')
plt.axvline(np.mean(np.exp(cvlls_0_03_var_0_04)), color='red', label='0.03 var4 mean')
plt.axvline(np.mean(np.exp(cvlls_0_03_var_0_02)), color='blue', label='0.03 var2 mean')

plt.legend()

plt.show()