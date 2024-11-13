"""
    Script for collection info about the MCMC chains of all subjects.
"""
import os
import json
import re
import numpy as np

prebias_subinfo = {}
bias_subinfo = {}

fitvar = '0.04'

file_prefix = ['./', '/usr/src/app/'][0]
data_folder = "dynamic_GLMiHMM_crossvals"

verificator = lambda x: '/summarised_sessions/0_25/' in x['file_name'] and x['fit_variance'] == 0.04 and ('dur' not in x or x['dur'] == 'yes') and np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and x['n_states'] == 15 and x['gamma_a_0'] == 0.01 and x['gamma_b_0'] == 100 and x['alpha_a_0'] == 0.01 and x['alpha_b_0'] == 100

# look through all files
for filename in os.listdir(file_prefix + data_folder + "/"):
    if not filename.endswith('.p'):
        continue

    # extract information about the chain
    regexp = re.compile(r'((\w|-)+)_fittype_(\w+)_var_{}_(\d+)_(\d+)_(\d+)'.format(fitvar))
    result = regexp.search(filename)
    if result is None:
        continue
    subject = result.group(1)
    fit_type = result.group(3)
    seed = result.group(4)
    fit_num = result.group(5)
    chain_num = result.group(6)

    # KS014_0.04_102_522.json
    infos = json.load(open(file_prefix + data_folder + "/infos/" + "{}_{}_{}_{}.json".format(subject, fitvar, seed, fit_num), 'r'))
    if not verificator(infos):
        continue

    # add info to correct dictionary
    if fit_type == 'prebias':
        local_dict = prebias_subinfo
    elif fit_type == 'bias':
        local_dict = bias_subinfo
    if subject not in local_dict:
        local_dict[subject] = {"seeds": [], "fit_nums": [], "chain_num": 0}
    if int(chain_num) == 0:  # if this is the first file of that chain, save some info
        local_dict[subject]["seeds"].append(seed)
        local_dict[subject]["fit_nums"].append(fit_num)
    else:
        local_dict[subject]["chain_num"] = max(local_dict[subject]["chain_num"], int(chain_num))


# old code for weeding out old fits (used seeds below 400)
for s in prebias_subinfo.keys():
    if len(prebias_subinfo[s]["fit_nums"]) == 48 or len(prebias_subinfo[s]["fit_nums"]) == 32:
        new_fit_nums = []
        new_seeds = []
        for fit_num, seed in zip(prebias_subinfo[s]["fit_nums"], prebias_subinfo[s]["seeds"]):
            if int(seed) >= 400:
                new_fit_nums.append(fit_num)
                new_seeds.append(seed)
        prebias_subinfo[s]["fit_nums"] = new_fit_nums
        prebias_subinfo[s]["seeds"] = new_seeds

# save results
json.dump(prebias_subinfo, open(file_prefix + "/canonical_infos_fitvar_{}.json".format(fitvar), 'w'))
json.dump(bias_subinfo, open(file_prefix + "/canonical_infos_bias_fitvar_{}.json".format(fitvar), 'w'))
