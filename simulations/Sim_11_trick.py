"""
Generate data from a simulated mouse-GLM.

This mouse uses 1 states throughout, which goes from poor to good, but not on session bounds, but continuously on every trial
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


subject = 'KS014'
new_name = 'GLM_Sim_11_trick'
seed = 212

print(new_name)
np.random.seed(seed)

info_dict = pickle.load(open("../session_data/{}_info_dict.p".format(subject), "rb"))
assert info_dict['subject'] == subject
info_dict['subject'] = new_name
pickle.dump(info_dict, open("../session_data/{}_info_dict.p".format(new_name), "wb"))

till_session = info_dict['bias_start']
from_session = 0

contrasts_L = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])
contrasts_R = np.array([1., 0.987, 0.848, 0.555, 0.302, 0, 0, 0, 0, 0, 0])[::-1]

def weights_to_pmf(weights, with_bias=1):
    if weights.shape[0] == 3 or weights.shape[0] == 5:
        psi = weights[0] * contrasts_L + weights[1] * contrasts_R + with_bias * weights[-1]
        return 1 / (1 + np.exp(-psi))
    elif weights.shape[0] == 11:
        return weights[:, 0]
    else:
        print('new weight shape')
        quit()


start = np.array([-0.1, 0.2, -1.])
end = np.array([-3.7, 3.5, -0.7])
GLM_weights = np.tile(start, (till_session + 1, 1))

neg_bin_params = [(30, 0.15)]
states = [(0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,), (0,)]

contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}

state_posterior = np.zeros((till_session - from_session + 1, 1))

total_trial_count = 0
for k, j in enumerate(range(from_session, till_session + 1)):
    data = pickle.load(open("../session_data/{}_fit_info_{}.p".format(subject, j), "rb"))
    total_trial_count += data.shape[0]

internal_trial_count = 0
for k, j in enumerate(range(from_session, till_session + 1)):

    GLM_weights[k] = start + (internal_trial_count / total_trial_count) * (end - start)

    plt.plot(weights_to_pmf(GLM_weights[k]))
    if j == till_session - 1:
        plt.ylim(bottom=0, top=1)
        plt.show()

    data = pickle.load(open("../session_data/{}_fit_info_{}.p".format(subject, j), "rb"))
    side_info = pickle.load(open("../session_data/{}_side_info_{}.p".format(subject, j), "rb"))

    contrasts = np.vectorize(num_to_contrast.get)(data[:, 0])

    predictors = np.zeros(3)
    state_plot = np.zeros(1)
    count = 0
    curr_state = states[j][0]
    curr_dur = np.random.negative_binomial(*neg_bin_params[curr_state]) + 1

    prev_choice = 2 * int(np.random.rand() > 0.5)
    data[0, 1] = prev_choice
    side_info[0, 1] = (prev_choice == 2 and contrasts[0] < 0) or ((prev_choice == 0 and contrasts[0] > 0))
    if contrasts[0] == 0:
        side_info[0, 1] = 0.5

    state_counter = 0

    for i, c in enumerate(contrasts[1:]):
        internal_trial_count += 1
        predictors[0] = max(c, 0)
        predictors[1] = abs(min(c, 0))
        predictors[2] = 1
        data[i+1, 1] = 2 * (np.random.rand() < 1 / (1 + np.exp(- np.sum(start + ((internal_trial_count / total_trial_count) * (end - start)) * predictors))))
        state_plot[curr_state] += 1
        side_info[i + 1, 1] = (data[i+1, 1] == 2 and c < 0) or ((data[i+1, 1] == 0 and c > 0))
        if c == 0:
            side_info[i + 1, 1] = 0.5
        curr_dur -= 1
        if curr_dur == 0:
            state_counter += 1
            curr_state = states[j][state_counter % len(states[j])]
            curr_dur = np.random.negative_binomial(*neg_bin_params[curr_state]) + 1

    state_posterior[k] = state_plot / len(data[:, 0])
    pickle.dump(data, open("../session_data/{}_fit_info_{}.p".format(new_name, j), "wb"))
    pickle.dump(side_info, open("../session_data/{}_side_info_{}.p".format(new_name, j), "wb"))

plt.figure(figsize=(16, 9))
s = 0
plt.fill_between(range(till_session + 1 - from_session), s - state_posterior[:, s] / 2, s + state_posterior[:, s] / 2)

truth = {'state_posterior': state_posterior, 'weights': GLM_weights, 'state_map': {0: 0}}
pickle.dump(truth, open("truth_{}.p".format(new_name), "wb"))

plt.savefig('states_11_trick')
plt.show()
