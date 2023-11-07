#!/usr/bin/env python

""" idealized_AMOC_model.py: script defining the three equations of the idealized AMOC model and generating the synthetic data used in AnDA (Synthetic data: catalogs, true states and noisy-observations). """

__author__ = "Pierre Le Bras adapted from Pierre Tandeo's code in https://github.com/ptandeo/AnDA/tree/master"
__version__ = "1.1"
__date__ = "2023-10-24"
__maintainer__ = "Pierre Le Bras"
__email__ = "pierre.lebras@univ-brest.fr"

import numpy as np
from scipy.integrate import odeint  
import random

def AnDA_ideal_AMOC(S,t,lambdaa,epsilon,beta,Omega0,K,F0,S0,h):
    """ Simplified AMOC model from Sévellec and Fedorov, 2013. """
    omega = -lambdaa*S[0]-epsilon*beta*S[2];
    S_BT = (Omega0+S[0])*S[2]-K*S[1]+(F0*S0)/h
    S_NS = -(Omega0+S[0])*S[1]-K*S[2]
    dS  = np.array([omega,S_BT,S_NS]);
    return dS



def AnDA_generate_data(model_phy, GD, seed):
    """ Generate the true state, noisy observations and catalog of numerical simulations for a dynamical model (here the 3D-AMOC idealised model). """
    
    seed

    # initialization
    class xt:
        values = [];
        time = [];
    class yo:
        values = [];
        time = [];
    class catalog:
        analogs = [];
        successors = [];
        source = [];
    
    # test on parameters
    if GD.dt_states>GD.dt_obs:
        print('Error: GD.dt_obs must be bigger than GD.dt_states');
    if (np.mod(GD.dt_obs,GD.dt_states)!=0):
        print('Error: GD.dt_obs must be a multiple of GD.dt_states');

    # use this to generate the same data for different simulations
    np.random.seed(1);
    
    if (model_phy.model == 'ideal_AMOC'):
           
        # 5 time steps (to be in the attractor space)       
        x0 = np.array([0,0,0]);
        S = odeint(AnDA_ideal_AMOC,x0,np.arange(0,5+0.000001,GD.dt_integration),
                   args=(model_phy.parameters.lambdaa, model_phy.parameters.epsilon, model_phy.parameters.beta, model_phy.parameters.Omega0,
                         model_phy.parameters.K,model_phy.parameters.F0, model_phy.parameters.S0,model_phy.parameters.h));  
        x00 = S[S.shape[0]-1,:];
        
        ## if the initial state is not specified in the model set up (i.e., GD.x0), then x00 is used instead       
        if GD.x0==None:
            S = odeint(AnDA_ideal_AMOC, x00, np.arange(0.01, GD.nb_loop_test+0.000001, GD.dt_integration),
                       args=(model_phy.parameters.lambdaa, model_phy.parameters.epsilon, model_phy.parameters.beta,model_phy.parameters.Omega0,
                             model_phy.parameters.K, model_phy.parameters.F0, model_phy.parameters.S0,model_phy.parameters.h));
        else:
            S = odeint(AnDA_ideal_AMOC, GD.x0, np.arange(0.01, GD.nb_loop_test+0.000001, GD.dt_integration),
                       args=(model_phy.parameters.lambdaa, model_phy.parameters.epsilon, model_phy.parameters.beta,model_phy.parameters.Omega0,
                             model_phy.parameters.K, model_phy.parameters.F0, model_phy.parameters.S0,model_phy.parameters.h));
            
        T_test = S.shape[0];      
        t_xt = np.arange(0,T_test,GD.dt_states);       
        xt.time = t_xt*GD.dt_integration;
        xt.values = S[t_xt,:];
        
        # generate  partial/noisy observations (yo) from the pre-generated true state S 
        eps = np.random.multivariate_normal(np.zeros(3),GD.sigma2_obs*np.eye(3,3),T_test);
        yo_tmp = S[t_xt,:] + eps[t_xt,:]; 
        t_yo = np.arange(0,T_test,GD.dt_obs); 
        i_t_obs = np.where((np.in1d(t_xt,t_yo))==True)[0];
        yo.values = xt.values*np.nan;
        yo.values[np.ix_(i_t_obs,GD.var_obs)] = yo_tmp[np.ix_(i_t_obs,GD.var_obs)];
        yo.time = xt.time;
       
        #generate catalog
        S =  odeint(AnDA_ideal_AMOC,S[S.shape[0]-1,:],np.arange(0.01,GD.nb_loop_train+0.000001,GD.dt_integration), 
                    args=(model_phy.parameters.lambdaa, model_phy.parameters.epsilon, model_phy.parameters.beta, model_phy.parameters.Omega0,
                          model_phy.parameters.K, model_phy.parameters.F0, model_phy.parameters.S0, model_phy.parameters.h));
        T_train = S.shape[0];
        eta = np.random.multivariate_normal(np.zeros(3),GD.sigma2_catalog*np.eye(3,3),T_train);
        catalog_tmp = S+eta;
        catalog.analogs = catalog_tmp[0:-GD.dt_states:GD.dt_states,:];
        catalog.successors = catalog_tmp[GD.dt_states::GD.dt_states,:]
        catalog.source = model_phy.parameters;
    
    # reinitialize random generator number
    np.random.seed()

    return catalog, xt, yo; 