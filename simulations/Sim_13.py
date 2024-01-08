"""
Generate data from a simulated mouse-GLM.

This mouse uses 5 states throughout
States alternate during the session, duration is negative binomial
I paid attention that if a state filled out a session, it's dur dist reflected that
"""
import numpy as np
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from scipy.stats import nbinom


subject = 'CSH_ZAD_022'
new_name = 'GLM_Sim_13'
seed = 14

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


GLM_weights = [np.array([-2.5, 2.3, -1.3]),
               np.array([-2.4, 2.1, 1.1]),
               np.array([-0.5, 0.4, -2.5]),
               np.array([0.1, 0.9, -0.6]),
               np.array([0.1, -0.2, 1.])]

for gw in GLM_weights:
    plt.plot(weights_to_pmf(gw))
plt.show()
neg_bin_params = [(180, 0.2), (105, 0.132), (5, 0.2), (30, 0.5), (30, 0.28)]
# [5, 15, 30, 50, 75, 105, 140, 180, 225, 275, 330, 390, 455, 525, 600, 680, 765, 855, 950]
GLM_weights = list(reversed(GLM_weights))
states = [(0,), (1,), (1,), (1, 2,), (1,), (1,), (3, 4,), (3, 4,), (4, 3, 2, 4, 3, 4, 3,), (3, 4,), (3, 4, 2,),
          (3, 4, 3, 2,), (3, 4,), (3, 4, 3, 4, 2,), (3, 4, 3, 2,), (3, 4,), (3, 4,), (3, 4, 3, 2,), (3, 4,)]

contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}

state_posterior = np.zeros((till_session - from_session + 1, len(GLM_weights)))
state_counter_array = np.zeros((till_session - from_session + 1, len(GLM_weights)))
for k, j in enumerate(range(from_session, till_session + 1)):
    for i, w in enumerate(GLM_weights):
        if i in states[j]:
            # print(j, i)
            w += np.random.normal(np.zeros(3), 0.03 * np.ones(3))
    # plt.plot(weights_to_pmf(GLM_weights[-3]))
    if j == till_session - 1:
        plt.ylim(bottom=0, top=1)
        plt.show()

    data = pickle.load(open("../session_data/{}_fit_info_{}.p".format(subject, j), "rb"))
    side_info = pickle.load(open("../session_data/{}_side_info_{}.p".format(subject, j), "rb"))

    if len(states[j]) <= 1:
        print(states[j])
        print(1 - nbinom.cdf(data.shape[0], *neg_bin_params[states[j][0]]))
    print(data.shape)
    print()

    contrasts = np.vectorize(num_to_contrast.get)(data[:, 0])

    predictors = np.zeros(3)
    state_plot = np.zeros(len(GLM_weights))
    count = 0
    curr_state = states[j][0]
    curr_dur = np.random.negative_binomial(*neg_bin_params[curr_state]) + 1

    prev_choice = 2 * int(np.random.rand() > 0.5)
    data[0, 1] = prev_choice
    side_info[0, 1] = (data[0, 1] == 2 and contrasts[0] < 0) or ((data[0, 1] == 0 and contrasts[0] > 0))
    if contrasts[0] == 0:
        side_info[0, 1] = 0.5

    state_counter = 0

    for i, c in enumerate(contrasts[1:]):
        predictors[0] = max(c, 0)
        predictors[1] = abs(min(c, 0))
        predictors[2] = 1
        data[i+1, 1] = 2 * (np.random.rand() < 1 / (1 + np.exp(- np.sum(GLM_weights[curr_state] * predictors))))
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
    state_counter_array[k] = state_plot
    pickle.dump(data, open("../session_data/{}_fit_info_{}.p".format(new_name, j), "wb"))
    pickle.dump(side_info, open("../session_data/{}_side_info_{}.p".format(new_name, j), "wb"))

plt.figure(figsize=(16, 9))
for s in range(len(GLM_weights)):
    plt.fill_between(range(till_session + 1 - from_session), s - state_posterior[:, s] / 2, s + state_posterior[:, s] / 2)

truth = {'state_posterior': state_posterior, 'weights': GLM_weights, 'state_map': {0: 0, 1: 1, 2: 4, 3: 3, 4: 2}}
pickle.dump(truth, open("truth_{}.p".format(new_name), "wb"))
plt.savefig('states_13')
plt.show()
