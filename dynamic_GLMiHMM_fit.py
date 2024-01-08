"""Start a (series) of iHMM fit(s)."""
import os
os.environ["OMP_NUM_THREADS"] = "1" # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1" # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import pyhsmm
import pyhsmm.basic.distributions as distributions
import copy
import warnings
import pickle
import time
from scipy.special import digamma
import os.path
import numpy as np
from itertools import product
import json
import sys


def eleven2nine(x):
    """Map from 11 possible contrasts to 9, for the non-training phases.

    1 and 9 can't appear there, make other elements consider this.

    E.g.:
    [2, 0, 4, 8, 10] -> [1, 0, 3, 7, 8]
    """
    assert 1 not in x and 9 not in x
    x[x > 9] -= 1
    x[x > 1] -= 1
    return x


def eval_cross_val(models, data, unmasked_data, n_all_states):
    """Eval cross_validation performance on held-out datapoints of an instantiated model"""
    lls = np.zeros(len(models))
    cross_val_n = 0
    for sess_time, (d, full_data) in enumerate(zip(data, unmasked_data)):
        held_out = np.isnan(d[:, -1])
        cross_val_n += held_out.sum()
        d[:, -1][held_out] = full_data[:, -1][held_out]
        for i, m in enumerate(models):
            for s in range(n_all_states):
                mask = np.logical_and(held_out, m.stateseqs[sess_time] == s)
                if mask.sum() > 0:
                    ll = m.obs_distns[s].log_likelihood(d[mask], sess_time)
                    lls[i] += np.sum(ll)
    lls /= cross_val_n
    ll_mean = np.mean(lls[-1000:])
    return lls, ll_mean


# following Nick Roys contrasts: following tanh transformation of the contrasts x has a
# free parameter p which we set as p= 5 throughout the paper: xp = tanh (px)/ tanh (p).
contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}
cont_mapping = np.vectorize(num_to_contrast.get)

data_folder = 'session_data'

# test subjects:
subjects = ['KS014']
num_subjects = len(subjects)
subjects = [a for a in subjects for i in range(3)]
# seeds = [505, 506, 507, 505, 506, 508, 509, 506, 507, 508, 505, 506, 508, 509, 505, 506, 507, 508, 509, 506, 507, 508, 505, 506, 507, 508, 505, 506, 507, 508, 505, 506, 507, 506, 507, 508, 506, 507, 508, 506, 507, 508]
seeds = list(range(506, 509))
seeds = seeds * num_subjects
# lst = list(range(10))
# cv_nums = [a for a in lst for i in range(8)]
cv_nums = [0] * 3 * num_subjects

seeds = [seeds[int(sys.argv[1])]]
cv_nums = [cv_nums[int(sys.argv[1])]]
subjects = [subjects[int(sys.argv[1])]]

print(cv_nums)
print(subjects)

for loop_count_i, (s, cv_num, seed) in enumerate(zip(subjects, cv_nums, seeds)):
    params = {}  # save parameters in a dictionary to save later
    params['subject'] = s
    params['cross_val_num'] = cv_num
    params['fit_variance'] = 0.03
    params['jumplimit'] = 1
    all_regressors = ['contR', 'contL', 'cont', 'prevA', 'weighted_prevA', 'WSLS', 'bias']
    params['regressors'] = [all_regressors[i] for i in [0, 1, 3, 6]]

    # default (non-iteration) settings:
    params['fit_type'] = ['prebias', 'bias', 'all', 'prebias_plus', 'zoe_style'][0]
    if 'prevA' in params['regressors'] or 'weighted_prevA' in params['regressors']:
        params['exp_decay'], params['exp_length'] = 0.3, 5
        params['exp_filter'] = np.exp(- params['exp_decay'] * np.arange(params['exp_length']))
        params['exp_filter'] /= params['exp_filter'].sum()
        print(params['exp_filter'])
    params['dur'] = 'yes'
    params['obs_dur'] = ['glm', 'cat'][0]

    # more obscure params:
    params['gamma'] = None  # 0.005
    params['alpha'] = None  # 1
    if params['gamma'] is not None:
        print("_______________________")
        print("Warning, gamma is fixed")
        print("_______________________")
    params['gamma_a_0'] = 0.001
    params['gamma_b_0'] = 1000
    params['init_var'] = 8
    params['init_mean'] = np.zeros(len(params['regressors']))

    r_support = np.cumsum(np.arange(5, 100, 5))
    r_support = np.arange(5, 705, 4)
    params['dur_params'] = dict(r_support=r_support,
                                r_probs=np.ones(len(r_support))/len(r_support), alpha_0=1, beta_0=1)
    params['alpha_a_0'] = 0.1
    params['alpha_b_0'] = 10
    params['init_state_concentration'] = 3

    # Parameter needed if one uses a Categorical distribution
    params['conditioned_on'] = 'nothing'

    params['cross_val'] = False
    params['cross_val_fold'] = 10
    params['CROSS_VAL_SEED'] = 4  # Do not change this, it's 4

    params['seed'] = seed

    params['n_states'] = 15
    params['n_samples'] = 60000 if params['obs_dur'] == 'glm' else 12000
    if params['cross_val']:
        params['n_samples'] = 12000
    if s.startswith("GLM_Sim"):
        print("reduced sample size")
        params['n_samples'] = 48000

    print(params['n_samples'])

    # now actual fit:

    # find a unique identifier to save this fit
    while True:
        folder = "./dynamic_GLMiHMM_crossvals/"
        rand_id = np.random.randint(1000)
        if params['cross_val']:
            id = "{}_crossval_{}_{}_var_{}_{}_{}".format(params['subject'], params['cross_val_num'], params['fit_type'],
                                                         params['fit_variance'], params['seed'], rand_id)
        else:
            id = "{}_fittype_{}_var_{}_{}_{}".format(params['subject'], params['fit_type'],
                                                     params['fit_variance'], params['seed'], rand_id)
        if not os.path.isfile(folder + id + '_0.p'):
            break
    # create placeholder dataset for rand_id purposes
    pickle.dump(params, open(folder + id + '_0.p', 'wb'))
    if params['obs_dur'] == 'glm':
        print(params['regressors'])
    else:
        print('using categoricals')
    print(id)
    params['file_name'] = folder + id
    np.random.seed(params['seed'])

    info_dict = pickle.load(open("./{}/{}_info_dict.p".format(data_folder, params['subject']), "rb"))
    # Determine session numbers
    if params['fit_type'] == 'prebias':
        till_session = info_dict['bias_start']
    elif params['fit_type'] == 'bias' or params['fit_type'] == 'zoe_style' or params['fit_type'] == 'all':
        till_session = info_dict['n_sessions']
    elif params['fit_type'] == 'prebias_plus':
        till_session = min(info_dict['bias_start'] + 6, info_dict['n_sessions'])  # 6 here will actually turn into 7 later

    from_session = info_dict['bias_start'] if params['fit_type'] in ['bias', 'zoe_style'] else 0

    models = []

    if params['obs_dur'] == 'glm':
        n_inputs = len(params['regressors'])
        T = till_session - from_session + (params['fit_type'] != 'prebias')
        obs_hypparams = {'n_inputs': n_inputs, 'T': T, 'jumplimit': params['jumplimit'], 'prior_mean': params['init_mean'],
                         'P_0': params['init_var'] * np.eye(n_inputs), 'Q': params['fit_variance'] * np.tile(np.eye(n_inputs), (T, 1, 1))}
        obs_distns = [distributions.Dynamic_GLM(**obs_hypparams) for state in range(params['n_states'])]
    else:
        n_inputs = 9 if params['fit_type'] == 'bias' else 11
        obs_hypparams = {'n_inputs': n_inputs * (1 + (params['conditioned_on'] != 'nothing')), 'n_outputs': 2, 'T': till_session - from_session + (params['fit_type'] != 'prebias'),
                         'jumplimit': params['jumplimit'], 'sigmasq_states': params['fit_variance']}
        obs_distns = [Dynamic_Input_Categorical(**obs_hypparams) for state in range(params['n_states'])]

    dur_distns = [distributions.NegativeBinomialIntegerR2Duration(**params['dur_params']) for state in range(params['n_states'])]

    # Select correct model
    if params['dur'] == 'yes':
        if params['gamma'] is None:
            posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                    # https://math.stackexchange.com/questions/449234/vague-gamma-prior
                    alpha_a_0=params['alpha_a_0'], alpha_b_0=params['alpha_b_0'],  # gamma steers state number
                    gamma_a_0=params['gamma_a_0'], gamma_b_0=params['gamma_b_0'],
                    init_state_concentration=params['init_state_concentration'],
                    obs_distns=obs_distns,
                    dur_distns=dur_distns,
                    var_prior=params['fit_variance'])
        else:
            posteriormodel = pyhsmm.models.WeakLimitHDPHSMM(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    init_state_concentration=params['init_state_concentration'],
                    obs_distns=obs_distns,
                    dur_distns=dur_distns,
                    var_prior=params['fit_variance'])
    else:
        if params['gamma'] is None:
            posteriormodel = pyhsmm.models.WeakLimitHDPHMM(
                    alpha_a_0=params['alpha_a_0'], alpha_b_0=params['alpha_b_0'],
                    gamma_a_0=params['gamma_a_0'], gamma_b_0=params['gamma_b_0'],
                    init_state_concentration=params['init_state_concentration'],
                    obs_distns=obs_distns,
                    var_prior=params['fit_variance'])
        else:
            posteriormodel = pyhsmm.models.WeakLimitHDPHMM(
                    alpha=params['alpha'],
                    gamma=params['gamma'],
                    init_state_concentration=params['init_state_concentration'],
                    obs_distns=obs_distns,
                    var_prior=params['fit_variance'])

    print(from_session, till_session + (params['fit_type'] != 'prebias'))

    if params['cross_val']:
        rng = np.random.RandomState(params['CROSS_VAL_SEED'])

    data_save = []
    for j in range(from_session, till_session + (params['fit_type'] != 'prebias')):
        try:
            data = pickle.load(open("./{}/{}_fit_info_{}.p".format(data_folder, params['subject'], j), "rb"))
        except FileNotFoundError:
            continue
        if data.shape[0] == 0:
            print("meh, skipped session")
            continue

        if params['obs_dur'] == 'glm':
            for i in range(data.shape[0]):
                data[i, 0] = num_to_contrast[data[i, 0]]
            mask = data[:, 1] != 1
            mask[0] = False
            if params['fit_type'] == 'zoe_style':
                mask[90:] = False
            mega_data = np.empty((np.sum(mask), n_inputs + 1))

            for i, reg in enumerate(params['regressors']):
                # positive numbers are contrast on the right
                if reg == 'contR':
                    mega_data[:, i] = np.maximum(data[mask, 0], 0)
                elif reg == 'contL':
                    mega_data[:, i] = np.abs(np.minimum(data[mask, 0], 0))
                elif reg == 'cont':
                    mega_data[:, i] = data[mask, 0]
                elif reg == 'prevA':
                    new_prev_ans = data[:, 1].copy()
                    new_prev_ans -= 1
                    new_prev_ans = np.convolve(np.append(0, new_prev_ans), params['exp_filter'])[:-(params['exp_filter'].shape[0])]
                    mega_data[:, i] = new_prev_ans[mask]
                elif reg == 'weighted_prevA':
                    prev_ans = data[:, 1].copy()
                    prev_ans -= 1
                    # weigh the tendency by how clear the previous contrast was
                    weighted_prev_ans = data[:, 0] + prev_ans
                    weighted_prev_ans = np.convolve(np.append(0, weighted_prev_ans), params['exp_filter'])[:-(params['exp_filter'].shape[0])]
                    mega_data[:, i] = weighted_prev_ans[mask]
                elif reg == 'WSLS':
                    side_info = pickle.load(open("./{}/{}_side_info_{}.p".format(data_folder, params['subject'], j), "rb"))
                    prev_reward = side_info[:, 1]
                    prev_reward[1:] = prev_reward[:-1]
                    prev_ans = data[:, 1].copy()
                    prev_ans[1:] = prev_ans[:-1] - 1
                    mega_data[:, i] = prev_ans[mask]
                    mega_data[prev_reward[mask] == 0, i] *= -1
                elif reg == 'bias':
                    # bias is now always active
                    mega_data[:, i] = 1

                mega_data[:, -1] = data[mask, 1] / 2
        elif params['obs_dur'] == 'cat':
            mask = data[:, 1] != 1
            mask[0] = False
            data = data[:, [0, 1]]
            data[:, 1] = data[:, 1] / 2
            mega_data = data[mask]

        data_save.append(mega_data.copy())

        if params['cross_val']:
            test_sets = np.tile(np.arange(params['cross_val_fold']), mega_data.shape[0] // params['cross_val_fold'] + 1)[:mega_data.shape[0]]
            rng.shuffle(test_sets)
            mega_data[:, -1][test_sets == params['cross_val_num']] = None

        posteriormodel.add_data(mega_data)

    if not os.path.isfile('./{}/data_save_{}.p'.format(data_folder, params['subject'])):
        pickle.dump(data_save, open('./{}/data_save_{}.p'.format(data_folder, params['subject']), 'wb'))

    time_save = time.time()
    likes = np.zeros(params['n_samples'])
    with warnings.catch_warnings():  # ignore the scipy warning
        warnings.simplefilter("ignore")
        for j in range(params['n_samples']):

            if j % 400 == 0 or j == 3:
                print(j)

            posteriormodel.resample_model()

            likes[j] = posteriormodel.log_likelihood()
            model_save = copy.deepcopy(posteriormodel)
            if j != params['n_samples'] - 1 and j != 0 and j % 2000 != 1:
                # To save on memory:
                model_save.delete_data()
                model_save.delete_obs_data()
                model_save.delete_dur_data()
            models.append(model_save)

            # save unfinished results
            if j % 400 == 0 and j > 0:
                if params['n_samples'] <= 4000:
                    pickle.dump(models, open(folder + id + '.p', 'wb'))
                else:
                    pickle.dump(models, open(folder + id + '_{}.p'.format(j // 4001), 'wb'))
                    if j % 4000 == 0:
                        models = []
    print(time.time() - time_save)

    if params['cross_val']:
        lls, lls_mean = eval_cross_val(models, posteriormodel.datas, data_save, n_all_states=params['n_states'])
        params['cross_val_preds'] = lls
        params['cross_val_preds'] = params['cross_val_preds'].tolist()

    print(id)
    if 'exp_filter' in params:
        params['exp_filter'] = params['exp_filter'].tolist()
    params['dur_params']['r_support'] = params['dur_params']['r_support'].tolist()
    params['dur_params']['r_probs'] = params['dur_params']['r_probs'].tolist()
    params['ll'] = likes.tolist()
    params['init_mean'] = params['init_mean'].tolist()
    if params['cross_val']:
        json.dump(params, open(folder + "infos/" + '{}_{}_cvll_{}_{}_{}_{}_{}.json'.format(params['subject'], params['cross_val_num'], str(np.round(lls_mean, 3)).replace('.', '_'),
                                                                                               params['fit_type'], params['fit_variance'], params['seed'], rand_id), 'w'))
    else:
        json.dump(params, open(folder + "infos/" + '{}_{}_{}_{}_{}.json'.format(params['subject'], params['fit_type'],
                                                                                    params['fit_variance'], params['seed'], rand_id), 'w'))
    pickle.dump(models, open(folder + id + '_{}.p'.format(j // 4001), 'wb'))
