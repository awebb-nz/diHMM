"""
Generate data from a simulated mouse-GLM.

This mouse has a lot of sessions, that's the main test here
Also, we use all other tricks in the book:
states alternate during the session, duration is negative binomial
now with new perseveration, and a bit of noise
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
from scipy.stats import nbinom


subject = 'DY_008'
new_name = 'GLM_Sim_18'
seed = 19

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
    psi = weights[0] * contrasts_R + weights[1] * contrasts_L + with_bias * weights[-1]
    return 1 / (1 + np.exp(psi))  # we somehow got the answers twisted, so we drop the minus here to get the opposite response probability for plotting


GLM_weights = [np.array([-4.1, 5.3, 0., -2.5]),
               np.array([-4.5, 4.3, 0., -0.7]),
               np.array([-0.3, 3.2, 0.5, -0.8]),
               np.array([-2.3, 1.5, 0.3, -0.2]),
               np.array([-1.9, 0.3, 1.5, 0.2]),
               np.array([-0.3, 0.5, -0.8, 2.5]),
               np.array([-1, 1.2, 2.1, 1]),
               np.array([0.2, 0.3, 2, -0.3]),
               np.array([-0.1, 0.2, 0., -1.])]
GLM_weights = list(reversed(GLM_weights))
neg_bin_params = [(190, 0.2), (75, 0.15), (105, 0.2), (50, 0.18), (120, 0.42), (100, 0.12), (150, 0.43), (150, 0.17), (460, 0.56)]
for i, gw in enumerate(GLM_weights):
    plt.plot(weights_to_pmf(gw), label=i)
plt.ylim(0, 1)
plt.legend()
plt.show()
states = [(0,), (0,), (1, 0), (1,), (1,), (0,), (2,), (1, 0), (1, 0), (3, 1), (3, 1), (3, 0), (2,), (1, 3),
          (2,), (3,), (4,0), (4, 5), (4,0), (4,0), (4,0), (2,), (0, 1), (4, 5), (4, 5), (4, 5), (6, 2), (6, 2), (6, 2), (6, 7), (6, 7),
          (6, 3, 6), (6, 7), (6, 7), (7,), (7,), (7,), (0,), (8, 5), (8, 5), (6, 7), (8, 5), (8,), (8, 5), (0,), (7,), (7,), (7,),
          (8,), (7,), (1, 0), (7,), (8, 7), (8,), (4, 5), (8, 7), (6, 7), (8, 7)]

contrast_to_num = {-1.: 0, -0.987: 1, -0.848: 2, -0.555: 3, -0.302: 4, 0.: 5, 0.302: 6, 0.555: 7, 0.848: 8, 0.987: 9, 1.: 10}
num_to_contrast = {v: k for k, v in contrast_to_num.items()}

state_posterior = np.zeros((till_session - from_session, len(GLM_weights)))

exp_decay, exp_length = 0.3, 5
exp_filter = np.exp(- exp_decay * np.arange(exp_length))
exp_filter /= exp_filter.sum()
exp_filter = np.flip(exp_filter)  # because we don't convolve, we need to flip manually

for k, j in enumerate(range(from_session, till_session)):
    for i, w in enumerate(GLM_weights):
        if i in states[j]:
            w += np.random.normal(np.zeros(4), 0.03 * np.ones(4))
    data = pickle.load(open("../session_data/{}_fit_info_{}.p".format(subject, j), "rb"))
    side_info = pickle.load(open("../session_data/{}_side_info_{}.p".format(subject, j), "rb"))

    print(data.shape)
    if len(states[j]) <= 1:
        if 1 - nbinom.cdf(data.shape[0], *neg_bin_params[states[j][0]]) < 0.1:
            print()
            print(states[j])
            print(j)
            print(1 - nbinom.cdf(data.shape[0], *neg_bin_params[states[j][0]]))
            print()

    contrasts = np.vectorize(num_to_contrast.get)(data[:, 0])

    predictors = np.zeros(4)
    previous_answers = np.zeros(5)
    state_plot = np.zeros(len(GLM_weights))
    count = 0
    curr_state = states[j][0]
    curr_dur = np.random.negative_binomial(*neg_bin_params[curr_state]) + 1

    previous_answers[-1] = 2 * int(np.random.rand() > 0.5) - 1
    data[0, 1] = previous_answers[-1] + 1
    side_info[0, 1] = (data[0, 1] == 2 and contrasts[0] < 0) or ((data[0, 1] == 0 and contrasts[0] > 0))
    if contrasts[0] == 0:
        side_info[0, 1] = 0.5

    state_counter = 0

    for i, c in enumerate(contrasts[1:]):
        predictors[0] = max(c, 0)
        predictors[1] = abs(min(c, 0))
        predictors[2] = np.sum(previous_answers * exp_filter)
        predictors[3] = 1
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

        previous_answers[:-1] = previous_answers[1:]
        previous_answers[-1] = data[i+1, 1] - 1

    state_posterior[k] = state_plot / len(data[:, 0])
    pickle.dump(data, open("../session_data/{}_fit_info_{}.p".format(new_name, j), "wb"))
    pickle.dump(side_info, open("../session_data/{}_side_info_{}.p".format(new_name, j), "wb"))

plt.figure(figsize=(16, 9))
for s in range(len(GLM_weights)):
    plt.fill_between(range(till_session - from_session), s - state_posterior[:, s] / 2, s + state_posterior[:, s] / 2)

plt.savefig('states_18')
plt.show()

truth = {'state_posterior': state_posterior, 'weights': GLM_weights, 'state_map': dict(zip(list(range(9)), list(range(9)))), 'durs': neg_bin_params}
pickle.dump(truth, open("truth_{}.p".format(new_name), "wb"))
