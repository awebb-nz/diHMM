# Dissecting the Complexities of Learning With Infinite Hidden Markov Models

## Abstract

Learning to exploit the contingencies of a complex experiment is not an easy task for animals. Individuals learn in an idiosyncratic manner, revising their approaches multiple times as they are shaped, or shape themselves, and potentially end up with different strategies. Their long-run learning curves are therefore a tantalizing target for the sort of individualized quantitative characterizations that sophisticated modelling can provide. However, any such model requires a flexible and extensible structure which can capture radically new behaviours as well as slow changes in existing ones. To this end, we suggest a dynamic input-output infinite hidden semi-Markov model, whose latent states are associated with specific components of behaviour. This model includes an infinite number of potential states and so has the capacity to describe substantially new behaviours by unearthing extra states; while dynamics in the model allow it to capture more modest adaptations to existing behaviours. We individually fit the model to data collected from more than 100 mice as they learned a contrast detection task over tens of sessions and around fifteen thousand trials each. Despite large individual differences, we found that most animals progressed through three major stages of learning, the transitions between which were marked by distinct additions to task understanding. We furthermore showed that marked changes in behaviour are much more likely to occur at the very beginning of sessions, i.e. after a period of rest, and that response biases in earlier stages are not predictive of biases later on in this task.

## Installation

This code relies on the installation of two packages, which are custom extensions of existing packages, as described there: \
https://github.com/SebastianBruijns/sab_pybasicbayes \
https://github.com/SebastianBruijns/sab_pyhsmm

### Installing with `conda`/`mamba`
```sh
mamba env create -f environment.yml
mamba activate hdp_env
pip install -r requirements_specific.txt
```

### Installing with `pyenv` and `pip`
```sh
pyenv local 3.7
python -m venv .env_hdp
source .env_hdp/bin/activate # on linux/mac
.env_hdp/Scripts/activate # on windows
pip install -r requirements_general.txt
pip install -r requirements_specific.txt
```

The data for this analysis is downloaded with the script (note that you will need to get the correct password from https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html): \
``behavioral_state_data_easier.py``

## Running

``dynamic_GLMiHMM_fit.py`` then fits the diHMM model to the specified subjects

The following 3 scripts process the MCMC-chains. They are split, because we usually run 1 and 3 on a cluster, as they are computationally intense, but 2, where one selects the samples to analyse, is run locally to view and interact with the results.\
``raw_fit_processing_part1.py``\
``raw_fit_processing_part2.py``\
``raw_fit_processing_part3.py``


``dyn_glm_chain_analysis.py`` goes through all the processed results, plotting overviews (figure 2 and 3), and collecting summaries of the data to process further. Also plots figure 5 and 7

``analysis_pmf.py`` plots figure 4\
``analysis_pmf_weights.py`` plots figure 6, 12, A16, and A17\
``analysis_regression.py`` plots figure 8
