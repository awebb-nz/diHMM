"""
    Code for figure 8 and variations, as well as some side info.
    Also: Cool function for offsetting points in a scatter.
"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr
import seaborn as sns

fontsize = 38
ticksize = 20

if __name__ == "__main__":
    # [total # of regressions, # of sessions, # of regressions during type 1, type 2, type 3]
    regressions = np.array(pickle.load(open("regressions.p", 'rb')))
    regression_diffs = np.array(pickle.load(open("regression_diffs.p", 'rb')))

    assert (regressions[:, 0] == np.sum(regressions[:, 2:], 1)).all()  # total # of regressions must be sum of # of regressions per type

    print(pearsonr(regressions[:, 0], regressions[:, 1]))
    print("Percentage of mice with regressions across population: {}".format(100 * np.sum(regressions[:, 0] != 0) / regressions.shape[0]))
    # (0.7776378443719886, 2.4699096615011557e-25)
    # Percentage of mice with regressions across population: 86.5546218487395

    offset = 0.25
    plt.figure(figsize=(16 * 0.9, 9 * 0.9))
    # which x values exist
    for x in np.unique(regressions[:, 0]):
        # which and how many ys are associated with this x
        tempa, tempb = np.unique(regressions[regressions[:, 0] == x, 1], return_counts=True)
        for y, count in zip(tempa, tempb):
            # plot them
            plt.scatter(x + np.linspace(-count + 1, count - 1, count) / 2 * offset, [y] * count, color='grey', linewidths=2)
    plt.ylabel("# of sessions", size=fontsize)
    plt.xlabel("# of regressed sessions", size=fontsize)
    plt.xticks(size=ticksize)
    plt.yticks(size=ticksize)
    sns.despine()

    plt.tight_layout()
    plt.savefig("./summary_figures/Regression vs session length")
    plt.show()

    plt.figure(figsize=(16 * 0.9, 9 * 0.9))
    non_regressed_sessions = regressions[:, 1] - regressions[:, 0]
    # which x values exist
    for x in np.unique(regressions[:, 0]):
        # which and how many ys are associated with this x
        tempa, tempb = np.unique(non_regressed_sessions[regressions[:, 0] == x], return_counts=True)
        for y, count in zip(tempa, tempb):
            # plot them
            plt.scatter(x + np.linspace(-count + 1, count - 1, count) / 2 * offset, [y] * count, color='grey', linewidths=2)
    plt.ylabel("# of non-regressed sessions", size=fontsize)
    plt.xlabel("# of regressed sessions", size=fontsize)
    plt.xticks(size=ticksize)
    plt.yticks(size=ticksize)
    sns.despine()

    plt.tight_layout()
    plt.savefig("./summary_figures/Regression vs non-regressions")
    plt.show()

    # histogram of regressions per type
    plt.bar([0, 1, 2], np.sum(regressions[:, 2:], 0), color='grey')
    plt.ylabel("# of regressed sessions", size=fontsize)
    plt.gca().set_xticks([0, 1, 2], ['Type 1', 'Type 2', 'Type 3'], size=fontsize)

    sns.despine()
    plt.tight_layout()
    plt.savefig("./summary_figures/# of regressions per type")
    plt.show()

    # histogram of number of mice with regressions in the different types
    num_of_mice = [*np.sum(regressions[:, 2:] >= 1, 0), np.sum(regressions[:, 0] >= 1)]
    plt.bar([0, 1, 2, 3], num_of_mice, color='grey')
    plt.ylabel("# of mice with regressions", size=fontsize)
    plt.gca().set_xticks([0, 1, 2, 3], ['Type 1', 'Type 2', 'Type 3', 'Any'], size=fontsize)

    sns.despine()
    plt.tight_layout()
    plt.savefig("./summary_figures/# of mice with regressions per type")
    plt.show()

    # histogram of regression diffs
    plt.hist([item for row in regression_diffs for item in row], color='grey', bins=20)
    plt.ylabel("# regressions", size=fontsize)
    plt.xlabel("Reward rate diff", size=fontsize)

    sns.despine()
    plt.tight_layout()
    plt.savefig("./summary_figures/Regression diffs")
    plt.show()


# figure out whether mice with long training use their states for longer
all_state_percentages = pickle.load(open("multi_chain_saves/all_state_percentages.p", 'rb'))

total_lengths = []
mean_state_appearances = []
for asp in all_state_percentages:
    total_lengths.append(asp.shape[1])
    mean_state_appearances.append(np.mean(np.sum(asp > 0.05, 1)))

plt.scatter(total_lengths, mean_state_appearances)
plt.xlabel("# of sessions")
plt.ylabel("Mean state sessions")
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig("./summary_figures/training_time_versus_mean_state_sessions.png", dpi=300)
plt.show()