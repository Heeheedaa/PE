# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:05:44 2024

@author: Yulin
"""

# LMA deducted from eco-optimality (Dong et al., 2022; DOI: 10.1111/1365-2745.13967)
# coefficients derived from Want et al., 2023; DOI: 10.1126/sciadv.add5667
# Define the function to calculate LMA based on Landcover
def calculate_LMA(row):
    # evergreen species
    if row['Landcover'] in ['ENF', 'EBF']:
        return np.exp(0.25 * np.log(row['f']) + 0.5 * np.log(row['Iabs_gs_mean']) - 0.013 * row['TA_F_gs_mean']
                      - 0.27 * row['aridity_gs_mean'] + 3.78)

    # deciduous species
    elif row['Landcover'] in ['CRO', 'DBF', 'DNF', 'GRA', 'SAV', 'WET']:
        return np.exp(np.log(row['f']) + np.log(row['Iabs_gs_mean']) - 0.052 * row['TA_F_gs_mean']
                      - 0.96 * row['aridity_gs_mean'] + 3.55)

    else:
        return 0.5 * (
            np.exp(0.25 * np.log(row['f']) + 0.5 * np.log(row['Iabs_gs_mean'])
                   - 0.01 * row['TA_F_gs_mean'] - 0.27 * row['aridity_gs_mean'] + 3.78) +
            np.exp(np.log(row['f']) + np.log(row['Iabs_gs_mean']) - 0.05 * row['TA_F_gs_mean']
                   - 0.96 * row['aridity_gs_mean'] + 3.55))


# Add the new column "LMA" using the custom function
merged_df['LMA'] = merged_df.apply(calculate_LMA, axis=1)