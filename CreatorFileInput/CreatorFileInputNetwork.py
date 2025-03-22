# -*- coding: utf-8 -*-
"""
Created on Sat Mar 22 19:25:26 2025

@author: Enrico
"""


import pandas as pd
import os
import numpy as np

def af(discount_rate, lifetime, capex, f_opex):
    ann_factor=((1+discount_rate)**lifetime-1)/(discount_rate*(1+discount_rate)** lifetime)
    fix_cost=capex/ann_factor+f_opex
    
    return  fix_cost



columns = [
    'Name', 'Type_of_component', 'P_nom-avlb', 'prob_trip', 'std_dev',
    'Committable', 'n_mod', 'resol_explor_P_avlb', 'marginal_cost',
    'capital_cost', 'carrier', 'n_min', 'n_comb', 'Bus0', 'norm stand-by cost',
    'ctrb:FCR', 'ctrb:aFRR', 'ctrb:mFRR', 'ctrb:RR', 'efficiency'
]

Network = pd.DataFrame(columns=columns)


###############################################################################
#Input data

Network.loc[0, columns] = ['Generator-Diesel-cluster1', 'Generator', 4, 0.000014, 0, True, 2, 0, 300,
                            af(discount_rate=0.04, lifetime=30, capex=1.02 *10**6, f_opex=0), 'Diesel',
                            0, 3, 2, 65, 1, 1,1,1, np.nan] 


Network.loc[1, columns] = ['Generator-Solar PV-cluster1', 'Generator', .5, 0.000011, 0.08, False, 20, 1/3, 0,
                            af(discount_rate=0.04, lifetime=25, capex=0.905*10**6, f_opex=17*10**3), 'Solar PV',
                            0, 3, 20, np.nan, np.nan, np.nan,np.nan,np.nan, np.nan] 


Network.loc[2, columns] = ['Store_Li-ion', 'Store', .5, 0.000009, 0, False, 40, 0, 00,
                            af(discount_rate=0.04, lifetime=15, capex=.3*10**6, f_opex=6*10**3), 'Li-ion',
                            0, 3, 40, np.nan, 1, 1,1,1, 0.999875] 

Network.loc[3, columns] = ['Load', 'Load', 4, 0, .1, False, 1, .25, 0,
                            0, 'Load',
                            0, 1, 1, 65, np.nan, np.nan,np.nan,np.nan, np.nan] 


############################################################################
#Save Data
Network.to_csv('Network', index=False)
output_file_path = os.path.join('Network_scheme.csv')

# Salva il dataframe CombToStudy in un file Excel senza index ma con i nomi delle colonne
Network.to_csv(output_file_path, index=False)