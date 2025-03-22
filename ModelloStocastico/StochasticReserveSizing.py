# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 16:10:44 2023

@author: user
"""


import pypsa
import os
os.environ["OMP_NUM_THREADS"] = '1'
from sklearn.cluster import KMeans
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt
from math import e
import time
import pandas as pd
from itertools import product
from parfor import parfor
from multiprocessing import Process
from multiprocessing import Pool
import time
import sys, traceback
import cProfile
import pstats
import io
import json
from itertools import chain
import gzip
import gc
import copy
import re

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        return super(NumpyEncoder, self).default(obj)
    
    
###############################################
#Combination of assets to Study
file_path = 'CombToStudySobol.xlsx'
comb_to_study = pd.read_excel(file_path, index_col=None)
parallel_setting = True
year_MC = 192
##########################################################
#Network Path
network_name = 'Network'
current_directory = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.join('..','CreatorFileInput')
output_file_path = os.path.join(target_directory, network_name+ '_scheme.csv')
os.makedirs(target_directory, exist_ok=True)
df=pd.read_csv(output_file_path)

############################################################################
#Name path for saving results
simulation_name = 'SobolUpDown'
folder_simulation = network_name + simulation_name + '_' + str(year_MC) +'yr'

########################################################################
"""
n_comb_to_run is the number of Sobol sequence combinations you wish to run
in the Monte Carlo simulation for the stochastic reserve estimation 
"""
n_comb_to_run = 10

# Percorsi dei file
risultati_combination_TOT_file_path = os.path.join(folder_simulation, f'combinations_dfSA_parallel{parallel_setting}.csv')
risultati_reserve_output_TOT_file_path = os.path.join(folder_simulation, f'f_reserveSA_parallel{parallel_setting}.csv')
risultati_reserve_output_TOT_UP_file_path = os.path.join(folder_simulation, f'f_reserveSA_parallel{parallel_setting}_UP.csv')
risultati_reserve_output_TOT_DW_file_path = os.path.join(folder_simulation, f'f_reserveSA_parallel{parallel_setting}_DW.csv')

# Controlla se i file esistono
if os.path.exists(risultati_combination_TOT_file_path) and os.path.exists(risultati_reserve_output_TOT_file_path):
    combinations_df_TOT = pd.read_csv(risultati_combination_TOT_file_path)
    f_reserve_TOT = pd.read_csv(risultati_reserve_output_TOT_file_path)
    n0_comb_to_run = len(f_reserve_TOT)
else:
    n0_comb_to_run = 0
nend_comb_to_run = n0_comb_to_run + n_comb_to_run -1

comb_to_study = copy.deepcopy(comb_to_study.loc[n0_comb_to_run:nend_comb_to_run,:]).reset_index(drop=True)
############################################################################
risultati_file_path = os.path.join(folder_simulation)

"""
ActualRound is the number of iterations you are over the Sobol sequence data set.
"""
ActualRound = 1

risultati_round_file_path = os.path.join(risultati_file_path, 'SimulationRound_'+str(ActualRound))
os.makedirs(risultati_round_file_path, exist_ok=True)

##############################################################################

def generate_saw_sequence(tMC, DDM):
    saw_length = 15
    repetitions = tMC // saw_length
    remainder = tMC % saw_length
    
    # Creazione della sequenza sega per una singola ripetizione
    single_sequence = np.linspace(-1, 1, saw_length)
    
    # Ripetizione della sequenza sega
    saw_sequence = np.tile(single_sequence, repetitions)
    
    # Aggiunta della parte rimanente se necessario
    if remainder > 0:
        saw_sequence = np.concatenate((saw_sequence, np.linspace(-1, 1, remainder)))
    
    ramp_value = DDM/8
    return ramp_value * saw_sequence

# Esempio di utilizzo
tMC_max = 192 * 365 * 24 * 60

DDM= 1
ramp_sequence = generate_saw_sequence(tMC_max, DDM)


####################################################################
P_ref=1


def tripping(yr,network):
    
    #Montecarlo Time horizon
    tMC=yr*365*24*60
    #Trip imbalance
    
    #element that can trip
    elem_trip=network['prob_trip']>0
    #number of elements that can trip
    n_elem_trip=(elem_trip*network['n_mod']).sum()
    yr_limit=int(year_MC/16)
    tMC_par=int(yr_limit*365*24*60)
    
    repeated_trip=np.repeat(network['prob_trip'][elem_trip].values, network['n_mod'][elem_trip].values)
    trip_val_matrix=np.transpose(np.tile(repeated_trip, (tMC_par,1)))
    trip_index = {i: [] for i in range(n_elem_trip)}#+1)} ##CONTROLLA LA FUNZIONE SE LAVORA BENE #c'era questo +1 che secondoo me non serve!!!!
    for yr_indx in range(int(tMC/tMC_par)):
        #print(str(yr_indx)+' su '+ str(int(tMC/tMC_par)))
        tMC_past=yr_indx*tMC_par
        trip_prob=np.random.rand(n_elem_trip,tMC_par)
        trip_binary=trip_prob<trip_val_matrix
        #trip_binary=trip_binary.ravel()
        trip_index_current=np.where(trip_binary>0)
        
        for idx0, idx1 in zip(trip_index_current[0], trip_index_current[1]):
            trip_index[idx0].append(idx1+tMC_past)
    
    return trip_index

def reserve_dimensioning(yr,network,quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df_selected, ramp_val, tripping_event):
    
    """
    Calculate reserve requirements (FCR, aFRR, mFRR, RR) based on specified
    quantiles using a stochastic approach following the ENTSO-E methodology.

    Parameters
    ----------
    yr : int
        The number of years for which to perform the Monte Carlo simulation.
    network : pypsa.Network
        The PyPSA network object containing information about the power system.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).

    Returns
    -------
    tuple
        A tuple containing the calculated reserves: (FCR, aFRR, mFRR, RR).
    """
    
    gc.collect()
    
    #Montecarlo Time horizon
    tMC=yr*365*24*60
    #Trip imbalance
   
    #element that can trip
    elem_trip=network['prob_trip']>0
    #number of elements that can trip
    n_elem_trip=(elem_trip*network['n_mod']).sum()
    
    trip_imb_t=np.zeros((tMC))
    
    if n_elem_trip>0:
        #matrix with uniform distribution
        
        #If the probability is lower than the threshold associated with the
        #individual generator (specific trip probability),
        #the generator is considered to have tripped (=1).
        
        p_nom=network['P_nom-avlb'][elem_trip*(network['n_mod']>0)]
        n_mod=network['n_mod'][elem_trip*(network['n_mod']>0)]
        
        trip_element_status=[]
        elem_trip_online=network['n_mod'][elem_trip].values
        elem_trip_offline= network['n_mod_max'][elem_trip].values-network['n_mod'][elem_trip].values
        for a, b in zip(elem_trip_online, elem_trip_offline):
            trip_element_status.extend([1] * a)
            trip_element_status.extend([0] * b)
        
        online_elem_index=np.where(np.array(trip_element_status) == 1)[0]
        trip_index=[tripping_event[i] for i in online_elem_index]
        #t_trip is the time a generator remains off after being tripped.
        t_trip=30
        g_start=0
        #####################CASO ISOLA ATTENZIONARE PERCHè CONSIDERA LA P_EROGATED TRANNE CHE PER P=0 DOVE METTE 
        #CONTROLLA LA FUNZIONE SE LAVORA BENE spero di aver risolto sotto
        #p_nom.loc[p_nom==0]=100
        for g in range(len(p_nom)):
            g_end=g_start+n_mod.iloc[g]
            trip_index_list=list(chain.from_iterable(trip_index[g_start:g_end]))
            if len(trip_index_list)>0:
                shift_vec = np.tile(list(range(0,t_trip)), len(trip_index_list))
                all_idxs_tripped=np.repeat(trip_index_list, t_trip) + shift_vec
                
                vector_counts = np.bincount(all_idxs_tripped, minlength=tMC)[0:tMC]
                #trip_imb_t[all_idxs_tripped[all_idxs_tripped<tMC]] += p_nom.iloc[g]
                trip_imb_t += vector_counts * p_nom.iloc[g]
            g_start=g_end
    
    quartile_trip=np.percentile(trip_imb_t, 99.7)
    EA_trip=trip_imb_t.sum()/yr/60
    
    #opzione 2 bis
    i_proccess=10
    RAM_avlb=30*1024**3/i_proccess
    n_elem_lim = int(RAM_avlb/15/tMC)  # definiscilo tramite RAM
    #n_elem_lim=10
    t_frr = 15
    n_quart = int(tMC / t_frr)
    
    FCR_up = np.array([])
    aFRR_up = np.array([])
    mFRR_up = np.array([])
    RR_up = np.array([])
    FCR_dw = np.array([])
    aFRR_dw = np.array([])
    mFRR_dw = np.array([])
    RR_dw = np.array([])

    for i in range(0, len(sigma_df_selected), n_elem_lim):
        # Estrai il blocco corrente di sigma_df_selected utilizzando .loc
        end_i=min(i + n_elem_lim, len(sigma_df_selected))
        current_block = sigma_df_selected.iloc[i:end_i]
        current_ramp_val=ramp_val.iloc[i:end_i].values
        #CONTROLLA LA FUNZIONE SE LAVORA BENE
        #spero di star conteggiando bene la riserva
        #ramp_matrix = np.outer(current_ramp_val, ramp_sequence[:tMC])
        #ramp_matrix=error_5min_norm[0:tMC].reshape(1, -1) * current_ramp_val.reshape(-1, 1)
        ramp_matrix = ramp_sequence[:tMC].reshape(1, -1) * current_ramp_val.reshape(-1, 1)
        
        df_t = np.random.normal(np.zeros((len(current_block), 1)), current_block.values.reshape((len(current_block), 1)), (len(current_block), tMC))
        tot_imb_t = df_t + trip_imb_t + ramp_matrix #ramp_sequence[0:tMC]
        tot_imb_norm_t = df_t + ramp_matrix #ramp_sequence[0:tMC]
        
        #####################################################################
        
        FCR_block_up = np.percentile(tot_imb_t, quantile_fcr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        FCR_block_dw = np.percentile(tot_imb_t, (1-quantile_fcr_perc) * 100 * np.ones(len(current_block)),axis=1)[0]
    
        mean_imb_t = tot_imb_t.reshape(-1, t_frr).sum(1) / t_frr
        mean_imb_t = mean_imb_t.reshape(len(current_block), int(n_quart))
        tot_imb_frr_t = tot_imb_t - np.repeat(mean_imb_t, t_frr, axis=1)
        #tot_imb_frr_t = copy.deepcopy(mean_imb_t)
    
        aFRR_block_up = np.percentile(tot_imb_frr_t, quantile_afrr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        aFRR_block_dw = np.percentile(tot_imb_frr_t, (1-quantile_afrr_perc) * 100 * np.ones(len(current_block)),axis=1)[0]
        
        mFRR_block_up = -aFRR_block_up + np.percentile(tot_imb_frr_t, quantile_mfrr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        mFRR_block_dw = -aFRR_block_dw + np.percentile(tot_imb_frr_t, (1-quantile_mfrr_perc) * 100 * np.ones(len(current_block)),axis=1)[0]
        
        RR_block_up = -aFRR_block_up - mFRR_block_up + np.percentile(tot_imb_frr_t, quantile_rr_perc * 100 * np.ones(len(current_block)),axis=1)[0]
        RR_block_dw = -aFRR_block_dw - mFRR_block_dw + np.percentile(tot_imb_frr_t, (1-quantile_rr_perc) * 100 * np.ones(len(current_block)),axis=1)[0]
    
        # Aggiungi i risultati del blocco corrente ai vettori accumulati
        FCR_up = np.concatenate((FCR_up, FCR_block_up))
        FCR_dw = np.concatenate((FCR_dw, FCR_block_dw))
        aFRR_up = np.concatenate((aFRR_up, aFRR_block_up))
        aFRR_dw = np.concatenate((aFRR_dw, aFRR_block_dw))
        mFRR_up = np.concatenate((mFRR_up, mFRR_block_up))
        mFRR_dw = np.concatenate((mFRR_dw, mFRR_block_dw))
        RR_up = np.concatenate((RR_up, RR_block_up))
        RR_dw = np.concatenate((RR_dw, RR_block_dw))

   
    return (FCR_up,aFRR_up,mFRR_up,RR_up,FCR_dw,aFRR_dw,mFRR_dw,RR_dw, quartile_trip*np.ones(len(FCR_up)), EA_trip*np.ones(len(FCR_up)))


def reserve_function_builder(network, yr, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, cmb_select):
    """
    Build and evaluate reserve allocation strategies based on different generator statuses and available/demanded power levels.

    This function generates and evaluates reserve allocation strategies for the given network and specified parameters.
    The strategies are evaluated at various points within the specified domain defined by generator statuses and power levels.

    Parameters
    ----------
    network : DataFrame
        Network data containing information about generators and their characteristics.
    yr : int
        Number of years for Monte Carlo simulation.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).

    Returns
    -------
    f_reserve : DataFrame
        Dataframe containing reserve allocation strategies for each combination of generator statuses and available power levels.

    """
    
    elem_committable_name=network.Name[network['prob_trip']>0]
    elem_p_var_name=network.Name[network['resol_explor_P_avlb']>0]
    elem_p_var=network['resol_explor_P_avlb']>0
    
    combinations_df = copy.deepcopy(comb_to_study)  

    ######################################################################################à
    
    mod_columns = [col for col in comb_to_study.columns if '-mod' in col]
    range_tripping = {col: comb_to_study[col].unique() for col in mod_columns}      
    
    f_reserve= pd.DataFrame(np.zeros((len(combinations_df),10)), columns=['f_FCR_UP', 'f_aFRR_UP', 'f_mFRR_UP', 'f_RR_UP','f_FCR_DW', 'f_aFRR_DW', 'f_mFRR_DW', 'f_RR_DW','quartile_trip', 'EA_trip'])
    
    n_mod_p_var_commit=combinations_df[(elem_committable_name[elem_committable_name.isin(elem_p_var_name)]+'-mod').to_list()]
    sigma_df=sigma_df_eval(n_mod_p_var_commit, network, combinations_df.loc[:,list(elem_p_var_name)])
    
    trip_comb_indx = find_new_combination_index(combinations_df, len(range_tripping))
    
    tripping_events_file_path = os.path.join(folder_simulation , network_name + simulation_name + '_TE'+str(year_MC)+'yr.json')
    #upload=0  vs compute=1
    opt_trip=1
    if opt_trip==1:
        tripping_event=tripping(yr,network)
        with open(tripping_events_file_path, 'w') as json_file:
            json.dump(tripping_event, json_file, cls=NumpyEncoder)
    else:
        with open(tripping_events_file_path, 'r') as json_file:
            tripping_event = json.load(json_file)
            tripping_event = {int(key): value for key, value in tripping_event.items()}
    
    network.loc[:,'n_mod_max'] = copy.deepcopy(network.loc[:,'n_mod'])
    
    print('tripping matrix done')
    
    parallel = parallel_setting
    if parallel==True:
        if __name__ == '__main__':
            with Pool(processes=8) as pool:
                args_list = [(idx, trip_comb_indx, tripping_event, combinations_df, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df.iloc[idx:idx_end], combinations_df.Ramp.iloc[idx:idx_end]) for idx, idx_end in zip(trip_comb_indx, trip_comb_indx[1:] + [len(combinations_df)])]
                results = pool.map(evaluate_reserve_single_case, args_list)
            
            indx_end=0    
            for i, result in zip(range(len(combinations_df)), results):
                indx_start=indx_end
                indx_end=indx_end+len(result)
                f_reserve.iloc[indx_start:indx_end] = result
                
    else: 
        for i in range(len(trip_comb_indx)):
            #if i==len(trip_comb_indx)-1:
            #    print('ciao')
            #i=3
            idx=trip_comb_indx[i]
            if i==len(trip_comb_indx)-1:
                idx_end=len(combinations_df)
            else: 
                idx_end=trip_comb_indx[i+1]
            #print(i, ' ', len(sigma_df[idx:idx_end]))
            f_reserve.iloc[idx:idx_end,:]=evaluate_reserve(idx, trip_comb_indx, tripping_event, combinations_df, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df.iloc[idx:idx_end], combinations_df.Ramp.iloc[idx:idx_end])
    
    return f_reserve, combinations_df


def evaluate_reserve_single_case(args):
    i, trip_comb_indx, tripping_event, combinations_df, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df_selected, ramp_val = args
    return evaluate_reserve(i, trip_comb_indx, tripping_event, combinations_df, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df_selected, ramp_val)


def sigma_df_eval(combinations_n_mod, network, p_var):
    
    
    #Delta f
    
    #element which cause frequency imbalance
    elem_df=df['std_dev']>0
    n_elem_df=(elem_df*network['n_mod']).sum()
    
    elem_n_mod_fix=set(network.Name[elem_df]) - set(combinations_n_mod.columns.str.replace('-mod', ''))
    
    n_mod_fix=network.loc[network.Name.isin(elem_n_mod_fix), 'n_mod']
    n_mod_fix_matrix=np.tile(n_mod_fix.values, (len(combinations_n_mod), 1))
    
    n_mod = copy.deepcopy(combinations_n_mod)
    n_mod.columns=list(combinations_n_mod.columns.str.replace('-mod', ''))
    n_mod[list(elem_n_mod_fix)]=n_mod_fix_matrix
    
    std_dev_matrix_df=np.tile(network['std_dev'][elem_df].values, (len(combinations_n_mod), 1))
    
    intersection_indx=p_var.columns.intersection(n_mod.columns)
    #CONTROLLA LA FUNZIONE SE LAVORA BENE
    #1) Load: il carico elettrico è moltipliato per solo p_ref??? indipendentemente dal valore del load?
    #std_dev_matrix_tot=std_dev_matrix_df**2*p_var.loc[:,intersection_indx]*n_mod*P_ref  --> equazione prec utilizzata che ora silenzio avendo cambiato modo 
    std_dev_matrix_tot=std_dev_matrix_df**2*p_var.loc[:,intersection_indx]*P_ref
    std_dev_vect=np.sqrt(((std_dev_matrix_tot)).sum(1))
    
    return std_dev_vect

def evaluate_reserve_wrapper(args):
    """
    Wrapper function to call evaluate_reserve with the provided arguments.
    
    Parameters
    ----------
    args : tuple
        A tuple containing the arguments needed for evaluate_reserve function.
    
    Returns
    -------
    tuple
        The result of the evaluate_reserve function.
    """
    return evaluate_reserve(*args)

def evaluate_reserve(i, trip_comb_indx, tripping_event, combinations_df, yr, network, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df_selected, ramp_val):
    """
    Evaluate reserve allocation strategies for a specific combination of parameters
    that affect its sizing.
    
    Parameters
    ----------
    i : int
        Index representing the combination.
    combinations_df : DataFrame
        DataFrame containing combinations of generator statuses and available power levels.
    yr : int
        Number of years for Monte Carlo simulation.
    network : DataFrame
        Network data containing information about generators and their characteristics.
    quantile_fcr_perc : float
        The desired quantile for FCR (e.g., 0.997 for 99.7th percentile).
    quantile_afrr_perc : float
        The desired quantile for aFRR (e.g., 0.95 for 95th percentile).
    quantile_mfrr_perc : float
        The desired quantile for mFRR (e.g., 0.99 for 99th percentile).
    quantile_rr_perc : float
        The desired quantile for RR (e.g., 0.997 for 99.7th percentile).
    
    Returns
    -------
    tuple
        A tuple containing the reserve allocation strategies for the given combination.
    """
    elem_committable_name = network.Name[network['prob_trip'] > 0]
    elem_p_var_name = network.Name[network['resol_explor_P_avlb'] > 0]
    elem_committable = network['prob_trip'] > 0
    elem_p_var = network['resol_explor_P_avlb'] > 0

    network_actual = copy.deepcopy(network)
    network_actual.loc[elem_committable, 'n_mod'] = np.array(combinations_df[elem_committable_name+'-mod'].iloc[i], dtype=int)
    
    tMC=yr*365*24*60
    
    result_fcr_up, result_afrr_up, result_mfrr_up, result_rr_up, result_fcr_dw, result_afrr_dw, result_mfrr_dw, result_rr_dw, quartile_trip, EA_trip = reserve_dimensioning(yr, network_actual, quantile_fcr_perc, quantile_afrr_perc, quantile_mfrr_perc, quantile_rr_perc, sigma_df_selected, ramp_val, tripping_event)
    
    f_reserve_current = pd.DataFrame([result_fcr_up, result_afrr_up, result_mfrr_up, result_rr_up, result_fcr_dw, result_afrr_dw, result_mfrr_dw, result_rr_dw, quartile_trip, EA_trip]).T
    f_reserve_current.columns = ['f_FCR_UP', 'f_aFRR_UP', 'f_mFRR_UP', 'f_RR_UP','f_FCR_DW', 'f_aFRR_DW', 'f_mFRR_DW', 'f_RR_DW', 'quartile_trip', 'EA_trip']
    
    return f_reserve_current

def find_new_combination_index(dataframe, len_combination):
    first_four_columns = dataframe.iloc[:, :len_combination]
    indices = first_four_columns.apply(lambda x: x != x.shift(1)).any(axis=1)
    change_indices = indices.loc[indices].index.tolist()
    return change_indices


combination_selected = []
start = time.time()
(f_reserve, combinations_df)=reserve_function_builder(df, year_MC, 0.997, 0.95, 0.99, 0.997, combination_selected)
end = time.time()
time=end - start 

##########################################################
#Save results in Round Folder

risultati_combination_round_file_path = os.path.join(risultati_round_file_path , f'combinations_dfSA_parallel'+str(parallel_setting)+'.csv')
risultati_reserve_output_round_file_path = os.path.join(risultati_round_file_path , f'f_reserveSA_parallel'+str(parallel_setting)+'.csv')

combinations_df.to_csv(risultati_combination_round_file_path, index=False)
f_reserve.to_csv(risultati_reserve_output_round_file_path , index=False)

##########################################################
#Save risultati generali

if ActualRound >1:
#if os.path.exists(risultati_combination_TOT_file_path) and os.path.exists(risultati_reserve_output_TOT_file_path):
    combinations_df_TOT = pd.concat([combinations_df_TOT, combinations_df], ignore_index=True)
    f_reserve_TOT = pd.concat([f_reserve_TOT, f_reserve], ignore_index=True)
else:
    combinations_df_TOT =  copy.deepcopy(combinations_df)
    f_reserve_TOT =  copy.deepcopy(f_reserve)

combinations_df_TOT.to_csv(risultati_combination_TOT_file_path, index=False)
f_reserve_TOT.to_csv(risultati_reserve_output_TOT_file_path , index=False)

upward_col = [col for col in f_reserve_TOT.columns if "_DW" not in col]
f_reserve_TOT_UP = copy.deepcopy(f_reserve_TOT.loc[:,upward_col])
f_reserve_TOT_UP.columns = [col.replace("_UP", "") for col in upward_col]
f_reserve_TOT_UP.to_csv(risultati_reserve_output_TOT_UP_file_path , index=False)

downward_col = [col for col in f_reserve_TOT.columns if "_UP" not in col]
f_reserve_TOT_DW = copy.deepcopy(f_reserve_TOT.loc[:,downward_col])
f_reserve_TOT_DW.columns = [col.replace("_DW", "") for col in downward_col]
f_reserve_TOT_DW.to_csv(risultati_reserve_output_TOT_DW_file_path , index=False)





