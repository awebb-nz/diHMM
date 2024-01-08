This code relies on the installation of two packages, as described here:

The data for this analysis is downloaded with the script (note that you will need to get the correct password from https://int-brain-lab.github.io/iblenv/notebooks_external/data_download.html):
``behavioral_state_data_easier.py``

``dynamic_GLMiHMM_fit.py`` then fits the diHMM model to the specified subjects

The following 3 scripts process the MCMC-chains. They are split, because we usually run 1 and 3 on a cluster, as they are computationally intense, but 2, where one selects the samples to analyse, is run locally to view and interact with the results.
``raw_fit_processing_part1.py``
``raw_fit_processing_part2.py``
``raw_fit_processing_part3.py``

``dyn_glm_chain_analysis.py`` goes through all the processed results, plotting overviews (figure 2 and 3), and collecting summaries of the data to process further. Also plots figure 5 and 7

``analysis_pmf.py`` plots figure 4
``analysis_pmf_weights.py`` plots figure 6, 12, A16, and A17
``analysis_regression.py`` plots figure 8
