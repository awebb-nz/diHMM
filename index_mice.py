"""
    Script for collection info about the MCMC chains of all subjects.
"""
import os
import json
import re

prebias_subinfo = {}
bias_subinfo = {}

fitvar = '0.03'

# look through all files
for filename in os.listdir("./dynamic_GLMiHMM_crossvals/"):
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
json.dump(prebias_subinfo, open("canonical_infos_fitvar_{}.json".format(fitvar), 'w'))
json.dump(bias_subinfo, open("canonical_infos_bias_fitvar_{}.json".format(fitvar), 'w'))
