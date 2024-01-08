"""
    Script for getting data from fits into a state to be analysed

    Script for taking a list of subects and extracting statistics from the chains
    which can be used to assess which chains have converged to the same regions

"""
import os
os.environ["OMP_NUM_THREADS"] = "16" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "16" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "16" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "16" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "16" # export NUMEXPR_NUM_THREADS=6
import numpy as np
import pyhsmm
import pickle
import json
from dyn_glm_chain_analysis import MCMC_result
from mcmc_chain_analysis import state_size_helper, state_num_helper, find_good_chains_unsplit_greedy, gamma_func, alpha_func
import index_mice  # executes function for creating dict of available fits
from dyn_glm_chain_analysis import MCMC_result_list
import sys


fit_type = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][2]

subjects = ['fip_37', 'fip_32', 'GLM_Sim_11_sub', 'GLM_Sim_11', 'GLM_Sim_11_trick', 'fip_14', 'fip_30', 'fip_36', 'fip_27', 'GLM_Sim_14', 'fip_34', 'fip_28', 'fip_13', 'GLM_Sim_16', 'fip_15', 'GLM_Sim_15', 'fip_38', 'fip_26', 'fip_35', 'fip_33', 'fip_31', 'GLM_Sim_07', 'GLM_Sim_13', 'fip_29', 'fip_16', 'GLM_Sim_12']
subjects = [subjects[int(sys.argv[1])]]

thinning = 25

fit_variance = 0.03
func1 = state_num_helper(0.2)
func2 = state_num_helper(0.1)
func3 = state_size_helper()
func4 = state_size_helper(1)
dur = 'yes'

loading_info = json.load(open("canonical_infos_fitvar_{}.json".format(fit_variance), 'r'))

def create_mode_indices(test, subject, fit_type):
    dim = 3

    print('Doing PCA')
    ev, eig, projection_matrix, dimreduc = test.state_pca(subject, pca_type='dists', dim=dim)#, save_add="_var_{}".format(fit_variance).replace('.', '_'))
    xy = np.vstack([dimreduc[i] for i in range(dim)])
    from scipy.stats import gaussian_kde
    z = gaussian_kde(xy)(xy)
    pickle.dump((xy, z), open("multi_chain_saves/xyz_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'wb'))

for subject in subjects:
    assert len(loading_info[subject]["seeds"]) == 16
    assert len(loading_info[subject]["fit_nums"]) == 16
    r_hats = {}
    results = []
    summary_info = {"thinning": thinning, "contains": [], "seeds": [], "fit_nums": []}
    m = len(loading_info[subject]["fit_nums"])
    print(subject)
    n_runs = -1
    counter = -1
    n = (loading_info[subject]['chain_num']) * 4000 // thinning
    chains1 = np.zeros((m, n))
    chains2 = np.zeros((m, n))
    chains3 = np.zeros((m, n))
    chains4 = np.zeros((m, n))
    for j, (seed, fit_num) in enumerate(zip(loading_info[subject]['seeds'], loading_info[subject]['fit_nums'])):
        counter += 1
        print(seed)
        info_dict = pickle.load(open("./session_data/{}_info_dict.p".format(subject), "rb"))
        samples = []

        mini_counter = 1 # start at 1, discard first 4000 as burnin
        while True:
            try:
                file = "./dynamic_GLMiHMM_crossvals/{}_fittype_{}_var_{}_{}_{}{}.p".format(subject, fit_type, fit_variance, seed, fit_num, '_{}'.format(mini_counter))
                samples += pickle.load(open(file, "rb"))
                mini_counter += 1
            except Exception:
                break

        if n_runs == -1:
            n_runs = mini_counter
        else:
            if n_runs != mini_counter:
                print("Problem")
                print(n_runs, mini_counter)
                quit()


        save_id = "{}_fittype_{}_var_{}_{}_{}.p".format(subject, fit_type, fit_variance, seed, fit_num).replace('.', '_')

        print("loaded seed {}".format(seed))

        result = MCMC_result(samples[::thinning],
                             infos=info_dict, data=samples[0].datas,
                             sessions=fit_type, fit_variance=fit_variance,
                             dur=dur, save_id=save_id)
        results.append(result)

        res = func1(result)
        chains1[counter] = res
        res = func2(result)
        chains2[counter] = res
        res = func3(result)
        chains3[counter] = res
        res = func4(result)
        chains4[counter] = res

    pickle.dump(chains1, open("multi_chain_saves/{}_state_num_0_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains2, open("multi_chain_saves/{}_state_num_1_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains3, open("multi_chain_saves/{}_largest_state_0_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))
    pickle.dump(chains4, open("multi_chain_saves/{}_largest_state_1_fittype_{}_var_{}_{}_{}_state_num.p".format(subject, fit_type, fit_variance, seed, fit_num), 'wb'))

    # R^hat tests
    # test = MCMC_result_list([fake_result(100) for i in range(8)])
    # test.r_hat_and_ess(return_ascending, False)
    # test.r_hat_and_ess(return_ascending_shuffled, False)

    print()
    print("Checking R^hat, finding best subset of chains")

    sol, final_r_hat = find_good_chains_unsplit_greedy(chains1, chains2, chains3, chains4, reduce_to=chains1.shape[0] // 2)
    r_hats[subject] = final_r_hat
    loading_info[subject]['ignore'] = sol

    print(r_hats[subject])
    json.dump(loading_info, open("multi_chain_saves/canonical_infos_{}_{}_var_{}.json".format(subject, fit_type, fit_variance), 'w'))
    json.dump(r_hats, open("multi_chain_saves/canonical_info_r_hats_{}_{}_var_{}.json".format(subject, fit_type, fit_variance), 'w'))

    if r_hats[subject] >= 1.05:
        print("Skipping canonical result due to R^hat")
        continue
    else:
        print("Making canonical result")

    # subset data
    summary_info['contains'] = [i for i in range(m) if i not in sol]
    summary_info['seeds'] = [loading_info[subject]['seeds'][i] for i in summary_info['contains']]
    summary_info['fit_nums'] = [loading_info[subject]['fit_nums'] for i in summary_info['contains']]
    results = [results[i] for i in summary_info['contains']]

    test = MCMC_result_list(results, summary_info)
    pickle.dump(test, open("multi_chain_saves/canonical_result_{}_{}_var_{}.p".format(subject, fit_type, fit_variance), 'wb'))

    test.r_hat_and_ess(state_num_helper(0.2), False)
    test.r_hat_and_ess(state_num_helper(0.1), False)
    test.r_hat_and_ess(state_num_helper(0.05), False)
    test.r_hat_and_ess(state_size_helper(), False)
    test.r_hat_and_ess(state_size_helper(1), False)
    test.r_hat_and_ess(gamma_func, True)
    test.r_hat_and_ess(alpha_func, True)

    print('Computing sub result')
    create_mode_indices(test, subject, fit_type)
