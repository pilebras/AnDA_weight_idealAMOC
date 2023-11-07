#!/usr/bin/env python

""" multimodel_idealized_AMOC_AnDA.py: script (1) generating the synthetic data used in AnDA (i.e., catalogs, true states and observations) for a range of 11~perturbed model versions.
                                              (2) performing AnDA for the 11~model versions by alternately varying the model used to generate the pseudo-observations (i.e., Leave-One-
                                                  Out experiment). 
                                              Outputs: catalogs; true states (xts); observations (yos); forecasts states (mean xf and covariances pf) from AnDA runs. """

__author__ = "Pierre Le Bras adapted from Pierre Tandeo's code in https://github.com/ptandeo/AnDA/tree/master"
__version__ = "1.1"
__date__ = "2023-10-24"
__maintainer__ = "Pierre Le Bras"
__email__ = "pierre.lebras@univ-brest.fr"

from idealized_AMOC_model import AnDA_generate_data
from AnDA_analog_forecasting import AnDA_analog_forecasting
from AnDA_data_assimilation import AnDA_data_assimilation
import itertools
import warnings
import os

warnings.filterwarnings("ignore")


# we fix a random seed
seed10 = np.random.seed(10)

################################################################################
############ CREATE PERTURBED VERSIONS OF THE IDEALIZED AMOC MODEL #############
################################################################################


############## MODEL PARAMETERS ###############

### all possible combinations of parameters 
fact_param=0.5 # the factor referring to the degree of similarity between the parameterizations (closer from 1 is fact_param, more similar they are)

### REFERENCES VALUES FOR THE PARAMETERS OF THE IDEALIZED AMOC MODEL (from Sévellec and Fedorov, 2014)
## perturbed parameters
ref_lambdaa = 0.01
ref_epsilon = 0.35
ref_Omega0 = -0.025
ref_K = 0.0001

# fixed parameters
ref_F0 = 1
ref_beta = 0.0007
ref_S0 = 35
ref_h = 1000 

# all the possible combinations of *fact_param, /fact_param, fact_param
list_lambdaa = [ref_lambdaa, ref_lambdaa*fact_param] 
list_epsilon = [ref_epsilon, ref_epsilon*fact_param]
list_Omega0 = [ref_Omega0, ref_Omega0*fact_param, ref_Omega0/fact_param]
list_K = [ref_K, ref_K*fact_param, ref_K/fact_param]

list_parameters = [list_lambdaa, list_epsilon, list_Omega0, list_K]

param_combin = list(itertools.product(*list_parameters))
param_combin = param_combin[0:len(param_combin)-9] #-9 to delete identical parameterizations

## subset of 11 parameterizations selected for the study
list_param_exp = [0,3,5,7,9,11,13,15,16,17,19]



############## PARAMETERS FOR DATA GENERATION ##############

## Parameters for data generation
dt_catalog_obs = 20
nb_states_catalogs = 400000 
catalogs_size = dt_catalog_obs*nb_states_catalogs  #learning sample

nb_obs = 400
observed_size = dt_catalog_obs*nb_obs 
err_obs = [0.00001] # no significant noise here because we add the real noise a posteriori after the scaling normalization

idx_random_state=np.random.randint(0, round(catalogs_size/dt_catalog_obs)) # the observed period (for xts and yos) start at the same time step for all the models

## storage for synthetic data for each perturbed model
allcatalogs = [] # catalogs
allxts = [] # true states (to derive the noisy-observations)
        
    
    
############## DATA GENERATION FOR THE 11~MODELS ##############

for i in list_param_exp :
    params = param_combin[i]
    
    # definition of the model's parameters
    class model_phy:
        model = 'ideal_AMOC'

        class parameters:
            ## perturbed parameters
            lambdaa = params[0]
            epsilon = params[1]
            Omega0 = params[2]
            K = params[3]
            
            ## fixed parameters
            F0 = ref_F0
            beta = ref_beta
            S0 = ref_S0
            h = ref_h 
    
    ################## first run for the spin up => only the catalog is registered here
    class GD_AnDA_spin_up:
        x0 = None # no perturbation on the initial state
        dt_integration = 1 
        dt_states = dt_catalog_obs 
        dt_obs = dt_catalog_obs
        var_obs = np.array([0]) # only omega is observed
        nb_loop_test = 1 # no matter here since the true states will be registered in the next data generation: GD_DA)
        sigma2_obs = err_obs # error on omega observations

        nb_loop_train = catalogs_size # size of the catalog
        sigma2_catalog = 0.0 # catalog as the truth 
        
    catalog_i, xt_i_none, yo_i_none = AnDA_generate_data(model_phy, GD_AnDA_spin_up, seed=seed10)
    allcatalogs.append(catalog_i)
    
    # to generate the true states and derived observations, the 3D initial condition is randomly drawn in the spin-up catalog
    random_state = [catalog_i.analogs[idx_random_state,0],
                    catalog_i.analogs[idx_random_state,1],
                    catalog_i.analogs[idx_random_state,2]]
    
    ################# to get the observed period : true states (xts) & observations (yos)
    class GD_AnDA:
        x0 = random_state
        dt_integration = 1 
        dt_states = dt_catalog_obs 
        dt_obs = dt_catalog_obs
        var_obs = np.array([0])
        nb_loop_test = observed_size 
        sigma2_obs = err_obs

        nb_loop_train = 1 # no matter here since the catalogs have already been registered in the previous data generation: GD_DA_spin_up
        sigma2_catalog = 0.0 # catalog as the truth 
        
    catalog_i_none, xt_i, yo_i = AnDA_generate_data(model_phy, GD_AnDA, seed=seed10)
    allxts.append(xt_i) 
   
## number of parameterizations
nb_param = len(allcatalogs)    
        
    
############## NORMALIZING THE DATA FOR THE 11~MODELS ##############

## normalization by scaling the catalogs and the true states
def normalization_std(serie, std):
    res = serie/std
    return res

## observations parameters
seed10
nb_obs = allobs[0].values[:,0].shape[0]
err_obs_norm = 0.5 #fix the observations error on normalized data
noise_obs = np.random.normal(0, err_obs_norm, nb_obs) 

## storage of the normalized data (catalogs; true states; noisy-observations)
allcatalogs_normstd = []
allxts_normstd = []
allobs_normstd = []


for i in range(0, nb_param):
    
    ## NORMALIZING THE CATALOGS
    class catalog_i:
        analogs = []
        successors = []
        
    ## compute the standard deviation of each variable of the catalog-simulation   
    std_om_i = np.std(allcatalogs[i].analogs[:,0])
    std_sbt_i = np.std(allcatalogs[i].analogs[:,1])
    std_sns_i = np.std(allcatalogs[i].analogs[:,2])

    catalog_i.analogs = np.column_stack((normalization_std(allcatalogs[i].analogs[:,0], std_om_i),
                                         normalization_std(allcatalogs[i].analogs[:,1], std_sbt_i),
                                         normalization_std(allcatalogs[i].analogs[:,2], std_sns_i)))
    
    catalog_i.successors = np.column_stack((normalization_std(allcatalogs[i].successors[:,0], std_om_i),
                                            normalization_std(allcatalogs[i].successors[:,1], std_sbt_i),
                                            normalization_std(allcatalogs[i].successors[:,2], std_sns_i))) 
    
    allcatalogs_normstd.append(catalog_i)   
    
    ## NORMALIZING THE TRUE STATES
    class xt_i:
        values = []
        time = []
    
    xt_i.values = np.column_stack((normalization_std(allxts[i].values[:,0], std_om_i),
                                   normalization_std(allxts[i].values[:,1], std_sbt_i),
                                   normalization_std(allxts[i].values[:,2], std_sns_i)))
    xt_i.time = allxts[i].time
    allxts_normstd.append(xt_i)
    
    ## CREATING NOISY-OBSERVATIONS USING THE NORMALIZED TRUE STATES    
    class yo_i:
        values = []
        time = []
    
    yo_i.values = np.column_stack((normalization_std(allxts[i].values[:,0], std_om_i) + noise_obs, # here additive noise to the first variable of interest (i.e., omega)
                                   normalization_std(allobs[i].values[:,1], std_sbt_i),
                                   normalization_std(allobs[i].values[:,2], std_sns_i)))
    yo_i.time = allobs[i].time
    allobs_normstd.append(yo_i)
    
    
##############  EXPORTATION OF THE SYNTHETIC DATA FOR THE 11~MODELS ############## 

## path = '/home/'

## CATALOGS (for the three variables)
export_catalogs_omega = np.zeros((nb_param, allcatalogs_normstd[0].analogs[:,0].shape[0]))
export_catalogs_sbt = np.zeros((nb_param, allcatalogs_normstd[0].analogs[:,1].shape[0]))
export_catalogs_sns = np.zeros((nb_param, allcatalogs_normstd[0].analogs[:,2].shape[0]))
for i in range(0,nb_param):
    export_catalogs_omega[i,:]=allcatalogs_normstd[i].analogs[:,0]
    export_catalogs_sbt[i,:]=allcatalogs_normstd[i].analogs[:,1]
    export_catalogs_sns[i,:]=allcatalogs_normstd[i].analogs[:,2]
# savetxt(path+'/normstd_catalogs_omega.csv', export_catalogs_omega, delimiter=';')
# savetxt(path+'/normstd_catalogs_sbt.csv', export_catalogs_sbt, delimiter=';')
# savetxt(path+'/normstd_catalogs_sns.csv', export_catalogs_sns, delimiter=';')

## XTS (only for omega)
xts_tosave = np.zeros((nb_param, allxts_normstd[0].values[:,0].shape[0]))
for i in range(0,nb_param):
    xts_tosave[i,:]=allxts_normstd[i].values[:,0]
# savetxt(path+'/normstd_xts.csv', xts_tosave, delimiter=';')

## OBS (only for omega)
obs_tosave = np.zeros((nb_param, allobs_normstd[0].values[:,0].shape[0]))
for i in range(0,nb_param):
    obs_tosave[i,:]=allobs_normstd[i].values[:,0]
# savetxt(pâth+'/normstd_obs.csv', obs_tosave, delimiter=';')
    
    
    
    
#################################################################################
## AnDA ON THE 11~MODELS USING ALTERNATELY EACH OF THEM AS PSEUDO-OBSERVATIONS ##
#################################################################################

## FRAMEWORK FOR ANDA

## PATH to store the results
#fd = os.open(path, os.O_RDONLY)
#os.mkdir('AnDA_results_11perturbed_idealized_AMOC_model', dir_fd=fd)
#path_results = '/home/AnDA_results_11perturbed_idealized_AMOC_model'
# fd = os.open(path_results, os.O_RDONLY) 

## list of strings referring to the resulting vectors which are stored: forecasts states (mean and variance)
list_anda_outputs = ['XFSomega_anda', 'PFSomega_anda']


## AnDA PARAMETERS
nb_knn_analogs = 10000 # number of analogs used to fit the LLR to estimate the forecasts states (mean and covariance) 
nb_members_EnKF = 200 # number of members in the EnKF
inflation_fact_cov_forecast = 1 # inflation factor of the forecasts covariance matrix (nothing here)
err_obs_norm = [0.5] ## obs error

## dictionnary to store the overall results
AnDA_all_results = {}

for j_obs in range(nb_param): # for each model used as pseudo-observations    
    folder_obs_name = "NO_param_"+str(j_obs)+"_as_obs" ## folder creation for each Leave-one-out experiment
    os.mkdir(folder_obs_name, dir_fd=fd)
    
    ### AnDA results initializations
    xf_anda_omega = np.zeros((nb_param, nb_obs))
    # xf_anda_sbt = np.zeros((nb_param, nb_obs))
    # xf_anda_sns = np.zeros((nb_param, nb_obs))
    Pf_anda_omega = np.zeros((nb_param, nb_obs))
    # Pf_anda_sbt = np.zeros((nb_param, nb_obs))
    # Pf_anda_sns = np.zeros((nb_param, nb_obs))
    
    k=-1
    for i in range (nb_param): 
        k=k+1
        # parameters of the analog forecasting method
        class AF_MOC_1D:
            k = nb_knn_analogs
            neighborhood = np.ones([allxts_normstd[i].values.shape[1], allxts_normstd[i].values.shape[1]]) # global analogs (only ones in the matrix)
            catalog = allcatalogs_normstd[i] 
            regression = 'local_linear' 
            sampling = 'gaussian' 

        # parameters of the filtering method
        class AnDA_MOC_1D:
            method = 'AnEnKF'
            N = nb_members_EnKF  
            xb = allxts_normstd[i].values[0,:]; B = np.eye(allxts_normstd[i].values.shape[1]) # initialization of the state
            H = np.eye(allxts_normstd[i].values.shape[1])
            R = err_obs_norm*np.eye(allxts_normstd[i].values.shape[1])
            nb_var_obs = GD_AnDA.var_obs.shape[0]
            factor_infl = inflation_fact_cov_forecast
            @staticmethod
            def m(x):
                return AnDA_analog_forecasting(x, AF_MOC_1D)
    
        results_AnDA = AnDA_data_assimilation(allobs_normstd[j_obs], AnDA_MOC_1D, seed=seed10)
    
        ## storage 
        xf_anda_omega[k,:] = results_AnDA.x_fmean_DA[:,0]
        # xf_anda_sbt[k,:] = results_AnDA.x_fmean_DA[:,1]
        # xf_anda_sns[k,:] = results_AnDA.x_fmean_DA[:,2]
        Pf_anda_omega[k,:] = results_AnDA.P_f_DA[:,0,0] 
        # Pf_anda_sbt[k,:] = results_AnDA.P_f_DA[:,1,1]
        # Pf_anda_sns[k,:] = results_AnDA.P_f_DA[:,2,2]
        
        print("AnDA parameterization number",i,"for param",j_obs,"as pseudo-observations is correctly stored")
    
    AnDA_all_results_crossvalid[j_obs]=[xf_anda_omega, Pf_anda_omega]  
    
    ####### WITHIN THE LOOP: EXPORTATION OF AnDA RESULTS FOR EACH LEAVE-ONE-OUT EXPERIMENT
    # res_AnDA_for_obs_j = AnDA_all_results_crossvalid[j_obs]
    # for elem in range(len(res_AnDA_for_obs_j)):
    #    savetxt(path_results+"/"+folder_obs_name+"/"+list_anda_outputs[elem]+".csv", res_AnDA_for_obs_j[elem], delimiter=';')      
        