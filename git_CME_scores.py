#!/usr/bin/env python

""" CME_scores.py: script defining the functions for computing the model weights for the three benchmark scores (i.e., the model democracy, the climatological comparison, the best single model) and the three CME-based scores (CME-ClimWIP, CME best punctual, CME best persistent) used in the study.
A common argument 'include_obs' denotes if the model used to generate the pseudo-observation is included in the weights calculation (i.e., if we are setting up in the perfect or imperfect case). Two additional functions are useful to compute the overlap score with the true distributions """

__author__ = "Pierre Le Bras"
__version__ = "1.1"
__date__ = "2023-10-24"
__maintainer__ = "Pierre Le Bras"
__email__ = "pierre.lebras@univ-brest.fr"

from git_AnDA_stat_functions_OK import AnDA_normalize_weights
import numpy as np


##################################
#######  BENCHMARK SCORES ########
##################################


## MODEL DEMOCRACY SCORE 
def vect_weights_model_democracy(nb_param, include_obs=True):
    """ Compute the model weights for model democracy score """    
    if include_obs==False: 
        nb_param=nb_param-1
    vect_weights_md = np.repeat(1/(nb_param),nb_param)
    return vect_weights_md


## CLIMATOLOGICAL COMPARISON SCORE
def pourcentcommun_with_histogram_intersec(ts_mod, ts_obs, nb_bins, range_low, range_up):
    """ Preliminary function used to get the weights related to the climatological comparison score. Compute the percentage of distribution intersection between two normalized distributions (args ts_mod abd ts_obs) both assessed on 'nb_bins'"""
   # put on same equal-width bins the two distributions
    counts_mod, bins_mod = np.histogram(ts_mod, bins=nb_bins, range=(range_low, range_up), density=True)
    counts_obs, bins_obs = np.histogram(ts_obs, bins=nb_bins, range=(range_low, range_up), density=True)
    commun_counts = 0
    for i in range(0,len(counts_mod)):
        if counts_mod[i]<=counts_obs[i]:
            commun_counts = commun_counts+counts_mod[i]
        elif counts_mod[i]>counts_obs[i]:
            commun_counts = commun_counts+counts_obs[i]
    pourcent_commun = commun_counts/sum(counts_mod)
    return commun_counts, pourcent_commun

def vect_weights_distributions_comparison(mat_catalogs, vect_obs, nb_bins, ind_obs, include_obs=True):
    """ Compute the model weights for the climatological comparison score.
    args: mat_catalogs refers to the matrix of all the candidate model catalogs ; vect_obs: the 1-D observations vector ; ind_obs: the index within 'mat_catalogs' of the model used as pseudo-observations   """
    if include_obs==False:
        mat_catalogs=np.delete(mat_catalogs, ind_obs, axis=0)
    range_low = np.minimum(np.min(mat_catalogs), np.min(vect_obs))
    range_up = np.maximum(np.max(mat_catalogs), np.max(vect_obs))
    list_pourcent_commun_mod = []
    for i in range(mat_catalogs.shape[0]):
        commun_counts, pourcent_commun = pourcentcommun_with_histogram_intersec(mat_catalogs[i,:], vect_obs, nb_bins, range_low, range_up)
        list_pourcent_commun_mod.append(commun_counts)
    vect_weights_dc = AnDA_normalize_weights(list_pourcent_commun_mod)
    return vect_weights_dc


## BEST SINGLE MODEL SCORE
def best_model_weight(mat_catalogs, vect_obs, nb_bins, ind_obs, include_obs=True):
    """ Compute the model weights for best single model score based on the climatological comparison score. """ 
    if include_obs==False:
        mat_catalogs=np.delete(mat_catalogs, ind_obs, axis=0)
    vect_weights_dc = vect_weights_distributions_comparison(mat_catalogs, vect_obs, nb_bins, ind_obs, include_obs=True)
    vect_weights_bm = [0]*mat_catalogs.shape[0]
    index_bm = np.argmax(vect_weights_dc) 
    vect_weights_bm[index_bm]=1
    return vect_weights_bm


##################################
#######   CME-BASED SCORES #######
##################################

def online_CME_metric_1D(mean_target, cov_target, mean_test, cov_test, dim_target=1):
    """ Preliminary function used to get the CME weights. Compute the 1D CME between two mean states vectors (args 'mean_target' and 'mean_test') and their related covariances (args 'cov_target' and 'cov_test'). Particularly useful to calculate the model-model CME between two forecast states within the CME-ClimWIP score """
    SIGMA = cov_target+cov_test
    SIGMA_INV = 1/SIGMA
    innov_ll2 = (mean_target-mean_test)**2
    loglik = -0.5*(innov_ll2*SIGMA_INV)-0.5*(dim_target*np.log(2*np.pi)+np.log(SIGMA))
    return loglik


## CME-ClimWIP SCORE
def vect_weights_cme_climwip(mat_forecasts, mat_covforecasts, vect_obs, err_obs_vect, sig_perf, sig_indep, ind_obs, dim_target=1, include_obs=True, indep_score=True):
    """ Compute the model weights for CME-ClimWIP score.
    args: 'mat_forecasts' is the matrix of the 1D forecasts for the variable of interest (drawn from AnDA) for all the candidate models (per raw) ; 'mat_covforecasts' is the associated covariances matrix ; 'vect_obs' is the 1D vector of observations for the variable of interest with 'err_obs_vect' its associated vector of errors ; 'sig_perf' and 'sig_indep' are the two shape parameters for the score (calculated outside the function) ; 'ind_obs' is the index of the model used to generate the pseudo-observations (in 'mat_forecasts") ;  'indep_score=True' includes the co-dependency part at denominator in the score while specifying 'nothing' assesses the score only taking into account the performance part at numerator """
    if include_obs==False:
        mat_forecasts=np.delete(mat_forecasts, ind_obs, axis=0)
        mat_covforecasts=np.delete(mat_covforecasts, ind_obs, axis=0)
      
    ## numerator: performance score
    list_num_mod = []
    for i in range(mat_forecasts.shape[0]):
        cme_mod = online_CME_metric_1D(vect_obs, err_obs_vect, mat_forecasts[i,:], mat_covforecasts[i,:], dim_target=dim_target)
        num_mod = np.exp(np.nanmean(cme_mod)/sig_perf**2) 
        list_num_mod.append(num_mod)
    scores_performance = np.array(list_num_mod)
    
   ## denominator: co-dependency/similarity score
    if indep_score==True:
        list_denom_mod = []
        for i in range(mat_forecasts.shape[0]):
            ind_all_except_i = np.delete(np.arange(mat_forecasts.shape[0]), i)
            denom_sum_modi_j=0
            for j in ind_all_except_i:
                cme_ij = online_CME_metric_1D(mat_forecasts[i,:], mat_covforecasts[i,:], mat_forecasts[j,:], mat_covforecasts[j,:], dim_target=1)
                denom_modi_j = np.exp(np.nanmean(cme_ij)/sig_indep**2)
                denom_sum_modi_j = denom_sum_modi_j + denom_modi_j
            list_denom_mod.append(1+denom_sum_modi_j)
        scores_similarity = 1/np.array(list_denom_mod)
        score_fin_mod =  scores_performance * scores_similarity
        
    else:
        score_fin_mod = list_num_mod
        scores_similarity = np.repeat(1, mat_forecasts.shape[0])
        
    vect_weights_fin = AnDA_normalize_weights(score_fin_mod)
    return vect_weights_fin #, score_fin_mod, scores_performance, scores_similarity


## CME BEST PUNCTUAL MODEL SCORE
def best_local_model_CME_01(vect_obs, vect_err_obs, mat_forecasts, mat_covforecasts, ind_obs, dim_target=1, include_obs=True):
    """ Compute the model weight for the best punctual score.
    args: same as previously"""
    if include_obs==False:
        mat_forecasts=np.delete(mat_forecasts, ind_obs, axis=0)
        mat_covforecasts=np.delete(mat_covforecasts, ind_obs, axis=0)
        
    mat_CME = np.zeros((mat_forecasts.shape[0], mat_forecasts.shape[1]))
    for i in range(mat_forecasts.shape[0]):
        mat_CME[i,:] = online_CME_metric_1D(vect_obs, vect_err_obs, mat_forecasts[i,:], mat_covforecasts[i,:], dim_target=dim_target)
    mat01_bestmodel_per_time = np.zeros((mat_forecasts.shape[0], vect_obs.shape[0]))
    index_max = np.argmax(mat_CME, axis=0) 
    
    for i in range(index_max.shape[0]): 
        mat01_bestmodel_per_time[index_max[i],i]=1
    
    count_max_mod = np.sum(mat01_bestmodel_per_time, axis=1) 
    vect_weights_bestlocal = AnDA_normalize_weights(count_max_mod)
    
    return vect_weights_bestlocal, mat01_bestmodel_per_time



## CME BEST PERSISTENT MODEL SCORE
def best_local_consecutive_cumul_CME(vect_obs, vect_err_obs, mat_forecasts, mat_covforecasts, ind_obs, fact_reward=1, dim_target=1, include_obs=True):
    """ Compute the model weight for the best persistent model score.
    args: same as previously ; 'fact_reward' denotes the multiplicative factor used to overweight the consecutives best model states (e.g., 'fact_reward=1' gives local scores of 1-2-3-4 to the four consecutive states where the same model is the best at these times , 'fact_reward=2' gives 1-2-4-8, etc. """
    weights_best_local, mat_input = best_local_model_CME_01(vect_obs, vect_err_obs, mat_forecasts, mat_covforecasts, ind_obs, dim_target=dim_target, include_obs=include_obs)
    mat_output = np.zeros((mat_input.shape[0], mat_input.shape[1]))
    for i in range(mat_input.shape[0]):
        count=0
        for j in range(mat_input.shape[1]):
            if (mat_input[i,j]==1 and fact_reward==1):
                count=count+1
                mat_output[i,j]=count
            elif (mat_input[i,j]==1 and fact_reward>1):
                if count==0:
                    count=count+1
                    mat_output[i,j]=count
                elif count>0:
                    count=fact_reward*count
                    mat_output[i,j]=count
            elif mat_input[i,j]==0:
                count=0
               
    count_cumul_max_mod = np.sum(mat_output, axis=1)
    vect_weights_best_consecutive_cumul_local = AnDA_normalize_weights(count_cumul_max_mod)
    
    return vect_weights_best_consecutive_cumul_local #, mat_output


##########################################
####### RECONSTRUCTION PERFORMANCE #######
##########################################

## BUILDING THE WEIGHTED PDF
def weighted_PDF_by_bins(tab, nb_bins, weights, ind_obs, include_obs=False):   
    """ Construct a weighted PDF using the model weights for a specific score.
    args: 'tab' gathers simulations of same size for each candidate model (per raw) to be considered as PDFs ; 'nb_bins' is the number of bins used to construct the PDFs ; 
    'weights' is the vector of model weights ; 'ind_obs' is the index of the true model """
    bound_inf = np.amin(tab)
    bound_sup = np.amax(tab)
    if include_obs==False:
        tab=np.delete(tab, ind_obs, axis=0)
    
    counts1, bins1 = np.histogram(tab[0,:], bins=nb_bins, range=(bound_inf, bound_sup))
    weight_0_for_bins = np.repeat(weights[0], nb_bins)
    weighted_counts, bins2 =np.histogram(bins1[:-1], bins1, weights=counts1*weight_0_for_bins)
    
    for mod in range(1,tab.shape[0]):
        counts1_mod, bins1 = np.histogram(tab[mod,:], bins=nb_bins, range=(bound_inf, bound_sup))
        weight_mod_for_bins = np.repeat(weights[mod], nb_bins)
        weighted_count_mod, bins2 =np.histogram(bins1[:-1], bins1, weights=counts1_mod*weight_mod_for_bins)
        weighted_counts = np.vstack((weighted_counts, weighted_count_mod))  
    
    val_for_final_weighted_pdf = np.sum(weighted_counts,axis=0)
    
    return bins1, weighted_counts, val_for_final_weighted_pdf   



## CALCULATING THE OVERLAP SCORE BETWEEN TWO PDFS
def perf_metric_coverage_btw_two_pdfs(counts_x, counts_y, nb_decim):
    """ Compute the overlap score between two distributions.
    args: 'counts_x' and 'counts_y' are the vectors of same-binned counts associated with the two distributions"""
    commun_counts = 0
    for i in range(0,len(counts_x)):
        if counts_x[i]<=counts_y[i]:
            commun_counts = commun_counts+counts_x[i]
        elif counts_x[i]>counts_y[i]:
            commun_counts = commun_counts+counts_y[i]
    pourcent_commun = commun_counts/sum(counts_x)
    return commun_counts, np.round(pourcent_commun*100, nb_decim)