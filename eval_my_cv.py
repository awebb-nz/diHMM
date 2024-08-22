import os
import numpy as np
import matplotlib.pyplot as plt
import json

folder = "./dynamic_GLMiHMM_crossvals/infos/"

# Define conditions
conditions = [
    {'path': '/summarised_sessions/0_3/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_35/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_25/', 'variance': 0.03},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.04},
    {'path': '/summarised_sessions/0_3/', 'variance': 0.02}
]

# Initialize dictionaries to store results
cvlls = {i: [] for i in range(len(conditions))}
subjects = {i: [] for i in range(len(conditions))}

# Process files
for file in os.listdir(folder):
    infos = json.load(open(folder + file, 'r'))
    subject = infos['subject']
    for i, condition in enumerate(conditions):
        if condition['path'] in infos['file_name'] and infos['fit_variance'] == condition['variance']:
            if subject in subjects[i] and np.mean(infos['cross_val_preds'][-1000:]) in cvlls[i]:
                print(file)
                os.rename(folder + file, "./dynamic_GLMiHMM_crossvals/infos_error/" + file)
            else:
                cvlls[i].append(np.mean(infos['cross_val_preds'][-1000:]))
                subjects[i].append(subject)
                break
    else:
        print("fits no schema: " + infos['file_name'])

# Plot boxplots
plt.figure()
labels = [f"{cond['variance']} var {cond['path'].split('/')[-2]}" for cond in conditions]
plt.boxplot([np.exp(cvlls[i]) for i in range(len(conditions))], labels=labels)
plt.xticks(range(1, len(conditions) + 1), labels)
plt.show()

quit()

# Plot histograms
for i, label in enumerate(labels):
    plt.hist(np.exp(cvlls[i]), bins=np.linspace(0.5, 1), alpha=0.3, label=label)

for i, label in enumerate(labels):
    plt.axvline(np.mean(np.exp(cvlls[i])), label=f'{label} mean')

plt.legend()
plt.show()