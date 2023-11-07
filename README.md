# AnDA_weight_idealAMOC


# DESCRIPTION
'AnDA_weight_idealAMOC' repository is associated to the following paper "Le Bras, P., Sévellec F., Tandeo P., Ruiz J., Ailliot P. (under review in NPG journal) "Selecting and weighting dynamical models using data-driven approaches".
The purpose of this study is to select and weight an ensemble of perturbed versions of an idealised dynamical model (describing a simplified version of the Atlantic Meridonal Overturning Circulation) using data-driven data assimilation.

The code for getting the figures in the paper is available in the ipython notebook "paper_figures_weighting_idealised_AMOC_model.ipynb". 


The scripts 'git_idealized_AMOC_model.py' and 'git_multimodel_idealized_AMOC_AnDA.py' generate the simulations data at the origin of the figures.

The script 'git_CME_scores.py' gathers the different scores used in the notebook to calculate the model weights.

The scripts 'git_AnDA_analog_forecasting.py' and 'git_AnDA_data_assimilation.py' allow to perform the analog data assimilation to each candidate model. The code is adapted from https://github.com/ptandeo/AnDA/tree/master.

The script 'git_AnDA_stat_functions.py' gathers useful statistical functions in other scripts.

# CONTACT
For further informations, please contact **Pierre Le Bras** (pierre.lebras@univ-brest.fr).

# Copyright
© 2023 Pierre Le Bras (pierre.lebras@univ-brest.fr)

This program is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License version 3 as published by the Free Software Foundation. See LICENCE file.
