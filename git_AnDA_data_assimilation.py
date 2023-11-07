#!/usr/bin/env python

""" AnDA_data_assimilation.py: Apply stochastic and sequential data assimilation technics using model forecasting or analog forecasting. """

__author__ = "Pierre Le Bras adapted from Pierre Tandeo's code in https://github.com/ptandeo/AnDA/tree/master"
__version__ = "1.1"
__date__ = "2023-10-24"
__maintainer__ = "Pierre Le Bras"
__email__ = "pierre.lebras@univ-brest.fr"

import numpy as np
from scipy.stats import multivariate_normal
from AnDA_stat_functions import resampleMultinomial, inv_using_SVD
from tqdm import tqdm
import random



def AnDA_data_assimilation(yo, DA, seed):
    """ 
    Apply stochastic and sequential data assimilation technics using 
    model forecasting or analog forecasting. 
    """
    
    seed

    # dimensions
    n = len(DA.xb) # number of variables in the system
    T = yo.values.shape[0] # the time steps of the assimilation period
    p = yo.values.shape[1] # 
    nb_var_obs = DA.nb_var_obs #number of observed variables (<= n)

    # check dimensions
    if p!=DA.R.shape[0]:
        print("Error: bad R matrix dimensions")
        quit()

        
    # initialization
    class x_hat:
        time = yo.time 
        part = np.zeros([T,DA.N,n])
        
        ## analysis outputs
        weights = np.zeros([T,DA.N])
        values = np.zeros([T,n]) 
        P_hat = np.zeros([T,n,n])
  
        ## forecasts outputs
        x_fmean_DA = np.zeros([T,n]) # forecasts mean states
        P_f_DA = np.zeros([T,n,n]) # forecasts covariances matrices 
        loglik_DA = np.zeros([T]) # Contextual Model Evidence 
        
    # EnKF and EnKS methods
    if (DA.method =='AnEnKF' or DA.method =='AnEnKS'):
        
        ## AnDA
        m_xa_part = np.zeros([T,DA.N,n])
        xf_DA_part = np.zeros([T,DA.N,n])
        Pf_DA = np.zeros([T,n,n])
        
        for k in tqdm(range(0,T)):
            x_hat.weights[k,:] = 1.0/DA.N # same weight for each member in the EnKF/EnKS case 
                     
            ### UPDATE STEP (to compute forecasts)  
            if k==0: # initialization at k=0
                xf_DA = np.random.multivariate_normal(DA.xb, DA.B, DA.N)             
               
            else:
                xf_DA, m_xa_part_tmp = DA.m(x_hat.part[k-1,:,:]) 
                m_xa_part[k,:,:] = m_xa_part_tmp    
                
            # storage
            xf_DA_part[k,:,:] = xf_DA #all the EnKF.S members
            x_hat.x_fmean_DA[k,:] = np.sum(xf_DA_part[k,:,:]*x_hat.weights[k,:,np.newaxis],0) #mean state
            Ef_DA = np.dot(xf_DA.T,np.eye(DA.N)-np.ones([DA.N,DA.N])/DA.N) 
            Pf_DA[k,:,:] = DA.factor_infl * np.dot(Ef_DA, Ef_DA.T)/(DA.N-1) 
            x_hat.P_f_DA[k,:,:] = Pf_DA[k,:,:]   # covariances of the mean state
            
           ### ANALYSIS STEP         
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]    
            
            if (len(i_var_obs)>0): #if there are available obs at time k            
                eps_DA = np.random.multivariate_normal(np.zeros(len(i_var_obs)),DA.R[np.ix_(i_var_obs,i_var_obs)],DA.N)
                yf_DA = np.dot(DA.H[i_var_obs,:],xf_DA.T).T
                SIGMA_DA = np.dot(np.dot(DA.H[i_var_obs,:],Pf_DA[k,:,:]),DA.H[i_var_obs,:].T)+DA.R[np.ix_(i_var_obs,i_var_obs)]
                SIGMA_INV_DA = np.linalg.inv(SIGMA_DA)
                K = np.dot(np.dot(Pf_DA[k,:,:],DA.H[i_var_obs,:].T),SIGMA_INV_DA) 
                y_perturb_DA = yo.values[k,i_var_obs][np.newaxis]+eps_DA 
                d = y_perturb_DA - yf_DA 
                x_hat.part[k,:,:] = xf_DA + np.dot(d,K.T) # analysis states members
                xa = x_hat.part[k,:,:] 
                
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0) # analysis mean state
                Ea = np.dot(xa.T,np.eye(DA.N)-np.ones([DA.N,DA.N])/DA.N)
                x_hat.P_hat[k,:,:] = np.dot(Ea,Ea.T)/(DA.N-1) #
                
                
                ### CONTEXTUAL MODEL EVIDENCE
                innov_ll_DA = np.mean(yo.values[k,i_var_obs][np.newaxis]-yf_DA,0) 
                loglik_DA = -0.5*(np.dot(np.dot(innov_ll_DA.T, SIGMA_INV_DA),innov_ll_DA))-0.5*(nb_var_obs*np.log(2*np.pi)+np.log(np.linalg.det(SIGMA_DA)))   
                
                
            else: # #if there are not available obs at time k 
                x_hat.part[k,:,:] = xf_DA #
                x_hat.P_hat[k,:,:] = Pf_DA[k,:,:] #
                
            x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
            x_hat.loglik_DA[k] = loglik_DA
            
        # end AnEnKF
        
        # EnKS method (code not adapted here with the CME)
        if (DA.method == 'AnEnKS'):
            for k in tqdm(range(T-1,-1,-1)):           
                if k==T-1:
                    x_hat.part[k,:,:] = x_hat.part[T-1,:,:]
                else:
                    m_xa_part_tmp = m_xa_part[k+1,:,:]
                    tej, m_xa_tmp = DA.m(np.mean(x_hat.part[k,:,:],0)[np.newaxis])
                    tmp_1 =(x_hat.part[k,:,:]-np.repeat(np.mean(x_hat.part[k,:,:],0)[np.newaxis],DA.N,0)).T
                    tmp_2 = m_xa_part_tmp - m_xa_tmp
                    Ks = 1.0/(DA.N-1)*np.dot(np.dot(tmp_1,tmp_2),inv_using_SVD(Pf_DA[k+1,:,:],0.9999))                    
                    x_hat.part[k,:,:] = x_hat.part[k,:,:]+np.dot(x_hat.part[k+1,:,:]-xf_part[k+1,:,:],Ks.T)
                
                # mean state at time step k
                x_hat.values[k,:] = np.sum(x_hat.part[k,:,:]*x_hat.weights[k,:,np.newaxis],0)
                xa = x_hat.part[k,:,:]
                
                # error variance at time step k
                Ea = np.dot(xa.T,np.eye(DA.N)-np.ones([DA.N,DA.N])/DA.N) #
                x_hat.P_hat[k,:,:] = np.dot(Ea,Ea.T)/(DA.N-1) #
                
        # end AnEnKS  
    
    
    # particle filter method 
    elif (DA.method =='AnPF'):
        # special case for k=1
        k=0
        k_count = 0
        m_xa_traj = []
        weights_tmp = np.zeros(DA.N)
        xf = np.random.multivariate_normal(DA.xb, DA.B, DA.N)
        i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
        if (len(i_var_obs)>0): # si on a des obs
            # weights
            for i_N in range(0,DA.N):
                weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)])
            # normalization
            weights_tmp = weights_tmp/np.sum(weights_tmp)
            # resampling
            indic = resampleMultinomial(weights_tmp)
            x_hat.part[k,:,:] = xf[indic,:]         
            weights_tmp_indic = weights_tmp[indic]/sum(weights_tmp[indic])
            x_hat.values[k,:] = sum(xf[indic,:]*weights_tmp_indic[np.newaxis].T,0)
            # find number of iterations before new observation
            k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0])
        else: # si pas d'obs
            # weights
            weights_tmp = 1.0/N
            # resampling
            indic = resampleMultinomial(weights_tmp)
        x_hat.weights[k,:] = weights_tmp_indic
        
        for k in tqdm(range(1,T)):
            # update step (compute forecasts) and add small Gaussian noise
            xf, tej = DA.m(x_hat.part[k-1,:,:]) +np.random.multivariate_normal(np.zeros(xf.shape[1]),DA.B/100.0,xf.shape[0])        
            if (k_count<len(m_xa_traj)):
                m_xa_traj[k_count] = xf
            else:
                m_xa_traj.append(xf)
            k_count = k_count+1
            # analysis step (correct forecasts with observations)
            i_var_obs = np.where(~np.isnan(yo.values[k,:]))[0]
            if len(i_var_obs)>0:
                # weights
                for i_N in range(0,DA.N):
                    #weights_tmp[i_N] = multivariate_normal.pdf(yo.values[k,i_var_obs].T,np.dot(DA.H[i_var_obs,:],xf[i_N,:].T),DA.R[np.ix_(i_var_obs,i_var_obs)]) # likelihood
                    weights_tmp[i_N] = multivariate_normal.pdf(np.dot(DA.H[i_var_obs,:],xf[i_N,:].T), yo.values[k,i_var_obs].T, DA.R[np.ix_(i_var_obs,i_var_obs)]) # likelihood
                # normalization
                weights_tmp = weights_tmp/np.sum(weights_tmp)
                # resampling
                indic = resampleMultinomial(weights_tmp)            
                # stock results
                x_hat.part[k-k_count_end:k+1,:,:] = np.asarray(m_xa_traj)[:,indic,:]
                weights_tmp_indic = weights_tmp[indic]/np.sum(weights_tmp[indic])            
                x_hat.values[k-k_count_end:k+1,:] = np.sum(np.asarray(m_xa_traj)[:,indic,:]*np.tile(weights_tmp_indic[np.newaxis].T,(k_count_end+1,1,n)),1)
                k_count = 0
                # find number of iterations  before new observation
                try:
                    k_count_end = np.min(np.where(np.sum(1*~np.isnan(yo.values[k+1:,:]),1)>=1)[0])
                except ValueError:
                    pass
            else:
                # stock results
                x_hat.part[k,:,:] = xf
                x_hat.values[k,:] = np.sum(xf*weights_tmp_indic[np.newaxis].T,0)
            # stock weights => to built the pdf of the reconstructed state xa
            x_hat.weights[k,:] = weights_tmp_indic   
        # end AnPF
        
   # error
    else :
        print("Error: choose DA.method between 'AnEnKF', 'AnEnKS', 'AnPF' ")
        quit()
    return x_hat
