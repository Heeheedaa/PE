# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:12:25 2024

@author: Yulin
"""


# Function to calculate leaf Nitrogen content; equation 6 at DOI: 10.1111/geb.13680
def calculate_N(row):
    return  (0.009 * (row['LMA']) + 0.007 * row['Vcmax25']  + 0.021 * row['LCC'] ),  (0.009 * (row['LMA']) + 0.007 * row['Vcmax25']  + 0.021 * row['LCC'] ) * (1 / row['LMA'] * 1000)
    

# Apply the custom function to calculate 'N' for each row
merged_df['N_area'], merged_df['N_mass'] = zip(*merged_df.apply(calculate_N, axis=1))


# Function to calculate leaf Phosphorous content; doi: 10.1093/nsr/nwx142 
def calculate_P(row):
    # Herb species
    if row['Landcover'] in ['CRO', 'GRA']:        
        return (row['N_mass'] / 17.8)** (1 / 0.659) / (1 / row['LMA'] * 1000)
        
    # Conifer woody species
    elif row['Landcover'] in ['ENF',  'DNF']:        
        return (row['N_mass'] / 19.95)** (1 / 0.610 ) / (1 / row['LMA'] * 1000) 
    
    # Deciduous broad-leaf species
    elif row['Landcover'] in ['DBF']:        
        return (row['N_mass'] / 19.95)** (1 / 0.712 ) / (1 / row['LMA'] * 1000)
    
    # Evergreen broad-leaf species
    elif row['Landcover'] in ['EBF']:        
        return (row['N_mass'] / 19.95)** (1 / 0.731 ) / (1 / row['LMA'] * 1000) 
    
    else:        
        return (row['N_mass'] / 18)** (1 / 0.678 ) / (1 / row['LMA'] * 1000) 
 

merged_df['P_area'] = merged_df.apply(calculate_P, axis=1)