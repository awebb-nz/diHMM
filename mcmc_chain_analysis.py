"""
    Functions to extract statistics from a set of chains
    Functions to compute R^hat from a set of statistic vectors
"""
import numpy as np
from scipy.stats import rankdata, norm
import pickle


def state_size_helper(n=0, mode_specific=False):
    """Returns a function that returns the # of trials associated to the nth largest state in a sample
       can be further specified to only look at specific samples, those of a mode"""
    if not mode_specific:
        def nth_largest_state_func(x):
            return np.partition(x.assign_counts, -1 - n, axis=1)[:, -1 - n]
    else:
        def nth_largest_state_func(x, ind):
            return np.partition(x.assign_counts[ind], -1 - n, axis=1)[:, -1 - n]
    return nth_largest_state_func


def state_num_helper(t, mode_specific=False):
    """Returns a function that returns the # of states which have more trials than a percentage threshold t in a sample
       can be further specified to only look at specific samples, those of a mode"""
    if not mode_specific:
        def state_num_func(x): return ((x.assign_counts / x.n_datapoints) > t).sum(1)
    else:
        def state_num_func(x, ind): return ((x.assign_counts[ind] / x.n_datapoints) > t).sum(1)
    return state_num_func


def gamma_func(x): return x.trans_distn.gamma


def alpha_func(x): return x.trans_distn.alpha


def ll_func(x): return x.sample_lls[-x.n_samples:]


def r_hat_array_comp(chains):
    """Computes R^hat on an array of features, following Gelman p. 284f
       Return R^hat itself, and var^hat^plus, which is needed for the effective sample size calculation"""
    m, n = chains.shape  # number of chains, length of chains
    psi_dot_j = np.mean(chains, axis=1)
    psi_dot_dot = np.mean(psi_dot_j)
    B = n / (m - 1) * np.sum((psi_dot_j - psi_dot_dot) ** 2)
    s_j_squared = np.sum((chains - psi_dot_j[:, None]) ** 2, axis=1) / (n - 1)
    W = np.mean(s_j_squared)
    var_hat_plus = (n - 1) / n * W + B / n
    if W == 0:  # sometimes a feature has 0 variance
        # print("all the same value")
        return 1, 0
    r_hat = np.sqrt(var_hat_plus / W)
    return r_hat, var_hat_plus


def eval_amortized_r_hat(chains, psi_dot_j, s_j_squared, m, n):
    """Unused version in which some things were computed ahead of function to save time."""
    psi_dot_dot = np.mean(psi_dot_j, axis=1)
    B = n / (m - 1) * np.sum((psi_dot_j - psi_dot_dot[:, None]) ** 2, axis=1)
    W = np.mean(s_j_squared, axis=1)
    var_hat_plus = (n - 1) / n * W + B / n
    r_hat = np.sqrt(var_hat_plus / W)
    return max(r_hat)


def r_hat_array_comp_mult(chains):
    """Compute R^hat of multiple features at once."""
    _, m, n = chains.shape
    psi_dot_j = np.mean(chains, axis=2)
    psi_dot_dot = np.mean(psi_dot_j, axis=1)
    B = n / (m - 1) * np.sum((psi_dot_j - psi_dot_dot[:, None]) ** 2, axis=1)
    s_j_squared = np.sum((chains - psi_dot_j[:, :, None]) ** 2, axis=2) / (n - 1)
    W = np.mean(s_j_squared, axis=1)
    var_hat_plus = (n - 1) / n * W + B / n
    r_hat = np.sqrt(var_hat_plus / W)
    return r_hat, var_hat_plus


def rank_inv_normal_transform(chains):
    """Gelman paper Rank-normalization, folding, and localization: An improved R_hat for assessing convergence of MCMC
       ranking with average rank for ties"""
    folded_chains = np.abs(chains - np.median(chains))
    ranked = rankdata(chains).reshape(chains.shape)
    folded_ranked = rankdata(folded_chains).reshape(folded_chains.shape)
    # inverse normal with fractional offset
    rank_normalised = norm.ppf((ranked - 3/8) / (chains.size + 1/4))
    folded_rank_normalised = norm.ppf((folded_ranked - 3/8) / (folded_chains.size + 1/4))
    return rank_normalised, folded_rank_normalised, ranked, folded_ranked


def eval_r_hat(chains):
    """Compute entire set of R^hat's for list of feature arrays, and return maximum across features.
       Computes all R^hat versions, as opposed to eval_simple_r_hat"""
    r_hats = []
    for chain in chains:
        rank_normalised, folded_rank_normalised, _, _ = rank_inv_normal_transform(chain)
        r_hats.append(comp_multi_r_hat(chain, rank_normalised, folded_rank_normalised))

    return max(r_hats)


def eval_simple_r_hat(chains):
    """Compute just simple R^hat's for list of feature arrays, and return maximum across features.
       Computes only the simple type of R^hat, no folding or rank normalising, making it much faster"""
    r_hats, _ = r_hat_array_comp_mult(chains)
    return max(r_hats)


def comp_multi_r_hat(chains, rank_normalised, folded_rank_normalised):
    """Compute full set of R^hat's, given the appropriately transformed chains."""
    lame_r_hat, _ = r_hat_array_comp(chains)
    rank_normalised_r_hat, _ = r_hat_array_comp(rank_normalised)
    folded_rank_normalised_r_hat, _ = r_hat_array_comp(folded_rank_normalised)
    return max(lame_r_hat, rank_normalised_r_hat, folded_rank_normalised_r_hat)


def sample_statistics(mode_indices, subject, period='prebias'):
    # prints out r_hats and sample sizes for given sample
    test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}.p".format(subject, period), 'rb'))
    test.r_hat_and_ess(state_size_helper(1), False)
    test.r_hat_and_ess(state_size_helper(1, mode_specific=True), False, mode_indices=mode_indices)
    print()
    test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}.p".format(subject, period), 'rb'))
    test.r_hat_and_ess(state_size_helper(), False)
    test.r_hat_and_ess(state_size_helper(mode_specific=True), False, mode_indices=mode_indices)
    print()
    test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}.p".format(subject, period), 'rb'))
    test.r_hat_and_ess(state_num_helper(0.05), False)
    test.r_hat_and_ess(state_num_helper(0.05, mode_specific=True), False, mode_indices=mode_indices)
    print()
    test = pickle.load(open("multi_chain_saves/canonical_result_{}_{}.p".format(subject, period), 'rb'))
    test.r_hat_and_ess(state_num_helper(0.02), False)
    test.r_hat_and_ess(state_num_helper(0.02, mode_specific=True), False, mode_indices=mode_indices)
    print()


def find_good_chains_unsplit_greedy(chains1, chains2, chains3, chains4, reduce_to=8, simple=False):
    delete_n = - reduce_to + chains1.shape[0]
    mins = np.zeros(delete_n + 1)
    n_chains = chains1.shape[0]
    chains = np.stack([chains1, chains2, chains3, chains4])

    r_hat = eval_r_hat([chains1, chains2, chains3, chains4])
    print("Without removals: {}".format(r_hat))
    mins[0] = r_hat

    r_hats = []
    solutions = []

    to_del = []
    for i in range(delete_n):
        r_hat_min = 50
        sol = 0
        for x in range(n_chains):
            if x in to_del:
                continue
            if not simple:
                r_hat = eval_r_hat([np.delete(chains1, to_del + [x], axis=0), np.delete(chains2, to_del + [x], axis=0), np.delete(chains3, to_del + [x], axis=0), np.delete(chains4, to_del + [x], axis=0)])
            else:
                r_hat = eval_simple_r_hat(np.delete(chains, to_del + [x], axis=1))
            if r_hat < r_hat_min:
                sol = x
            r_hat_min = min(r_hat, r_hat_min)
        to_del.append(sol)

        print("Minimum is {} (removed {})".format(r_hat_min, i + 1))
        print("Removed: {}".format(to_del))
        mins[i + 1] = r_hat_min

        r_hats.append(r_hat_min)
        solutions.append(to_del.copy())

    if simple:
        r_hat_local = eval_r_hat([np.delete(chains1, to_del, axis=0), np.delete(chains2, to_del, axis=0), np.delete(chains3, to_del, axis=0), np.delete(chains4, to_del, axis=0)])
        print("Minimum over everything is {} (removed {})".format(r_hat_local, i + 1))

    best = np.argmin(r_hats)

    return solutions[best], r_hats[best]