# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 13:28:52 2024

@author: Enrico
"""

import numpy as np
from scipy.stats import qmc
import pandas as pd
import os

################################################################################
#Improt network
network_name = 'Network'
network=pd.read_csv(network_name)
ramp =  4 #MW
low_ramp = -ramp#-ramp
################################################################################
# trova elementi trip, vRES e colonne

elem_trip_name=network.loc[network.carrier!= 'Load', 'Name']
n_elem_trip_name = len(elem_trip_name)
elem_p_var_name=network.loc[network['resol_explor_P_avlb']>0, 'Name']
n_elem_p_var_name = len(elem_p_var_name)

mod_columns = elem_trip_name + '-mod'
p_var_columns = elem_p_var_name
columns = list(mod_columns) + list(p_var_columns)

columns.append('Ramp')

################################################################################
#calcolo numero max combinazioni
n_com_tot = network.loc[:, 'n_mod'].prod()
n_com_tot *= (network.loc[p_var_columns.index, 'n_mod'] * network.loc[p_var_columns.index, 'P_nom-avlb']).prod()
n_com_tot *= ramp
max_excel_dim = 1048576
com_to_explore = min(max_excel_dim, n_com_tot)
################################################################################

# Inizializzazione del generatore Sobol
sobol = qmc.Sobol(d=len(columns), scramble=False, seed=0)  # scramble=False per mantenere la deterministica

# Generazione esattamente di max_excel_dim-1 punti
samples = sobol.random_base2(m=int(np.log2(com_to_explore)))[0:max_excel_dim-1, :]

# Creazione di una matrice per salvare i risultati finali
result = np.zeros_like(samples)

################################################################################

# Mappare i tripping
for element in list(elem_trip_name):
    low, high = 0, network.loc[network.Name == element, 'n_mod'].iloc[0]
    i = columns.index(element+'-mod')
    result[:, i] = np.floor(samples[:, i] * (high - low + 1) + low)

# Mappare vRES
for element in list(elem_p_var_name):
    low, high = 0, network.loc[network.Name == element, 'n_mod'].iloc[0] * network.loc[network.Name == element, 'P_nom-avlb'].iloc[0]
    i = columns.index(element)
    result[:, i] = samples[:, i] * (high - low) + low

# Mappare Ramp
low, high = low_ramp, ramp
i = columns.index("Ramp")
result[:, i] = samples[:, i] * (high - low) + low

CombToStudySobol = pd.DataFrame(result, columns=columns)

################################################################################
#Salva Dataframe

current_directory = os.path.dirname(os.path.abspath(__file__))
target_directory = os.path.join(current_directory, '..', 'ModelloStocastico')
output_file_path = os.path.join(target_directory, 'CombToStudySobol.xlsx')
os.makedirs(target_directory, exist_ok=True)

# Salva il dataframe CombToStudy in un file Excel senza index ma con i nomi delle colonne
CombToStudySobol.to_excel(output_file_path, index=False)