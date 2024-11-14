"""
    Script for collection info about the MCMC chains of all subjects.
"""
import argparse
import os
import json
import re
import numpy as np


def index_mice(file_prefix, data_folder, fitvar):
    prebias_subinfo = {}
    bias_subinfo = {}
    verificator = lambda x: '/summarised_sessions/0_25/' in x['file_name'] and \
                            x['fit_variance'] == 0.04 and \
                            ('dur' not in x or x['dur'] == 'yes') and \
                            np.array_equal(x['dur_params']['r_support'], np.arange(5, 705)) and \
                            x['n_states'] == 15 and \
                            x['gamma_a_0'] == 0.01 and \
                            x['gamma_b_0'] == 100 and \
                            x['alpha_a_0'] == 0.01 and \
                            x['alpha_b_0'] == 100

    print(file_prefix, data_folder, fitvar)

    # look through all files
    target_directory = os.path.join(file_prefix, data_folder)
    for filename in os.listdir(target_directory):
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
        infos_filename = os.path.join(file_prefix, data_folder, "infos", f"{subject}_{fitvar}_{seed}_{fit_num}.json")
        infos = json.load(open(infos_filename))
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

if __name__ == "__main__":
    fitvar = '0.04'
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--cluster", action="store_true")
    args = parser.parse_args()

    if args.cluster:
        file_prefix =  '/usr/src/app'
    else:
        file_prefix = '.'  # cluster or local prefix
    data_folder = "dynamic_GLMiHMM_crossvals"

    index_mice(file_prefix, data_folder, fitvar)
