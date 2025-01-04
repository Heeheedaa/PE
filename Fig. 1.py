# -*- coding: utf-8 -*-
"""
Created on Mon May 27 14:55:13 2024

@author: Yulin
"""

import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import numpy as np
import glob
import pandas as pd
import scipy.io
import h5py
import shap
from scipy.interpolate import Akima1DInterpolator
from scipy.signal import savgol_filter

site = pd.read_csv('C:/Users/Yulin/Desktop/Data to Zenodo/FLUXNET2015v1-4.csv', sep=',', header=None)

# aridity calculated from BESS equals to ET/PET
aridity = scipy.io.loadmat('C:/Users/Yulin/Desktop/Data to Zenodo/BESSv2.ETtoPET.Daily.FLUXNET2015.mat')['data']
aridity = pd.DataFrame(aridity)
# assign site name to df
aridity.columns = site.iloc[1:207, 0]


# Function to filter out February 29 in leap years
def filter_leap_year(date):
    return (date.month != 2) or (date.day != 29)

# Create a date range from '2001-01-01' to '2016-12-31' while skipping February 29 in leap years
start_date = pd.to_datetime('2000-12-31')
end_date = pd.to_datetime('2016-12-31')
date_range = pd.date_range(start=start_date, end=end_date, freq='D', inclusive='right')
filtered_date_range = date_range[date_range.to_series().apply(filter_leap_year)]

aridity['TIMESTAMP'] = filtered_date_range


# Clumping index, interpolate from 8-day to daily

with h5py.File('C:/Users/Yulin/Desktop/Data to Zenodo/Clumping.8day.FLUXNET2015.mat', 'r') as file:
    Clumping = file['data'][:].T
    
Clumping = pd.DataFrame(Clumping)
Clumping.columns = site.iloc[1:207, 0]

# scaling factor
Clumping *= 0.001

# Create a date range from '2001-01-01' to '2017-12-31' while skipping February 29 in leap years
start_date = pd.to_datetime('2000-12-31')
end_date = pd.to_datetime('2018-02-28')
date_range = pd.date_range(start=start_date, end=end_date, freq='8D', inclusive='right')
filtered_date_range = date_range[date_range.to_series().apply(filter_leap_year)]


# Assign time information to rows
Clumping['TIMESTAMP'] = filtered_date_range

# Create a new DataFrame for the daily interpolated data
daily_clumping = pd.DataFrame(columns=Clumping.columns)

# Group the data by year and interpolate each column separately
def interpolate_group(group):
    year = group['TIMESTAMP'].dt.year.iloc[0]
    group = group.sort_values(by='TIMESTAMP')
    
    # Initialize a dictionary to store interpolated values for each column
    interpolated_data = {'TIMESTAMP': pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')}
    
    # Iterate over sites
    for i in range(1, 207):  
        column_name = Clumping.columns[i]  
        
        interp_func = Akima1DInterpolator(group['TIMESTAMP'].dt.dayofyear, group[column_name])
        interpolated_values = interp_func(interpolated_data['TIMESTAMP'].dayofyear)
        interpolated_data[column_name] = interpolated_values
    
    # Create a new DataFrame for the interpolated data and return it
    interpolated_df = pd.DataFrame(interpolated_data)
    
    return interpolated_df

# Apply the interpolation function to each group and concatenate the results
daily_clumping = Clumping.groupby(Clumping['TIMESTAMP'].dt.year).apply(interpolate_group).reset_index(drop=True)



# Load the data
FPAR_path = r'C:/Users/Yulin/Desktop/Data to Zenodo/filtered_lai_fpar_data.xlsx'
FPAR = pd.read_excel(FPAR_path, sheet_name='data')

# Convert date column to datetime
FPAR['TIMESTAMP'] = pd.to_datetime(FPAR['TIMESTAMP'], format='%Y-%m-%d')

# Create a new DataFrame for the daily interpolated data
FPAR_daily_interp_data = pd.DataFrame(columns=FPAR.columns)

# Group the data by 'site_name' and year, and perform interpolation within each group 
def interpolate_FPAR(group):
    site_name = group['site_name'].iloc[0]
    year = group['TIMESTAMP'].dt.year.iloc[0]
    group = group.sort_values(by='TIMESTAMP')
    
    # Check if arrays have at least 50%  elements
    if len(group['TIMESTAMP'].dt.dayofyear) < 22 or len(group['FPAR']) < 22:
        print("Arrays do not contain sufficient elements. Skipping interpolation.")
    else:
        
        window_length = min(11, len(group['FPAR']) - 1)
        polyorder = min(2, window_length - 1)
        smoothed_FPAR = savgol_filter(group['FPAR'], window_length=window_length, polyorder=polyorder)
        interp_func = Akima1DInterpolator(group['TIMESTAMP'].dt.dayofyear, smoothed_FPAR)
                
        # Create a new DataFrame with daily timestamps for the given year
        daily_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        
        # Interpolate data for each daily timestamp
        interpolated_values = interp_func(daily_dates.dayofyear)
        
        # Create a new DataFrame for the interpolated data and return it
        interpolated_data = pd.DataFrame({
            'TIMESTAMP': daily_dates,
            'site_name': site_name,
            'FPAR': interpolated_values
        })
        return interpolated_data

# Apply the interpolation function to each group and concatenate the results
FPAR_daily_interp_data = FPAR.groupby(['site_name', FPAR['TIMESTAMP'].dt.year]).apply(interpolate_FPAR).reset_index(drop=True)




# Preparing Flunxnet2015 dataset
filelist = glob.glob('C:\\Users\\Yulin\\Desktop\\Data to Zenodo\\**FULLSET_DD_**.csv')

combined_df = pd.DataFrame()

for file_path in filelist:
    # Read the text file as a DataFrame using pandas
    df = pd.read_csv(file_path)
    
    columns_to_select = ['TIMESTAMP', 'TA_F', 'TA_F_DAY', 'SW_IN_F', 'SW_IN_POT', 'VPD_F', 'PA_F', 'CO2_F_MDS', 'GPP_NT_VUT_REF', 'NEE_VUT_REF_NIGHT_QC']

    if 'SWC_F_MDS_1' in df.columns:
        columns_to_select.append('SWC_F_MDS_1')
    
    df = df[columns_to_select]

    df['site_name'] = file_path.split('\\')[5][4:10]
    df['TIMESTAMP'] = pd.to_datetime(df['TIMESTAMP'], format='%Y%m%d')

    # Extract the year from the 'TIMESTAMP' column
    df['Year'] = df['TIMESTAMP'].dt.year

    # Filter rows where 'TA_F' is greater than 10
    mask_gs = df['TA_F'] > 10

    # Group by 'Year' and calculate the length for each group
    f = df[mask_gs].groupby('Year').size()/365

    # Merge the calculated 'f' values with the original DataFrame
    df_gs = df[mask_gs].merge(f.rename('f'), on='Year', how='left')

    # Append the data to the combined DataFrame
    combined_df = pd.concat([combined_df, df_gs], ignore_index=True)

    combined_df['f'] = combined_df['f'].replace(366/365, 1)
    
# Merge BESS data with Flunxet2015
# Step 1: Reshape the 'aridity' DataFrame using 'melt'
aridity_melted = pd.melt(aridity, id_vars=['TIMESTAMP'], var_name='site_name', value_name='aridity')
FPAR_melted = pd.melt(FPAR, id_vars=['TIMESTAMP'], var_name='site_name', value_name='FPAR_value')
Clumping_melted = pd.melt(daily_clumping, id_vars=['TIMESTAMP'],var_name='site_name', value_name='daily_clumping')
Clumping_melted['TIMESTAMP'] = pd.to_datetime(Clumping_melted['TIMESTAMP'])

# Step 2: Perform the left merge using 'site_name' and 'Date'
merged_df = pd.merge(combined_df, aridity_melted, on=['site_name', 'TIMESTAMP'], how='left')
merged_df = pd.merge(merged_df, FPAR_daily_interp_data, on=['site_name', 'TIMESTAMP'], how='left')
merged_df = pd.merge(merged_df, Clumping_melted, on=['site_name', 'TIMESTAMP'], how='left')

# Assign column names to 'site' DataFrame
site.columns = ['site_name', 'Landcover', 'Latitude','Longitude', 'Timezone', 'Continent', 'Climate']

# Merge 'merged_df' with 'site' on the common column 'site_name'
merged_df = pd.merge(merged_df, site[['site_name', 'Landcover']], on='site_name', how='left')

# Prepare data in different time-step 
merged_df['aridity_mean'] = merged_df.groupby(['Year', 'site_name'])['aridity'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['aridity_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['aridity'].transform('mean')



merged_df['TA_F_mean'] = merged_df.groupby(['Year', 'site_name'])['TA_F'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['TA_F_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['TA_F'].transform('mean')

merged_df['PA_F_mean'] = merged_df.groupby(['Year', 'site_name'])['PA_F'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['PA_F_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['PA_F'].transform('mean')

merged_df['CO2_mean'] = merged_df.groupby(['Year', 'site_name'])['CO2_F_MDS'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['CO2_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['CO2_F_MDS'].transform('mean')

merged_df['SW_IN_F_mean'] = merged_df.groupby(['Year', 'site_name'])['SW_IN_F'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['FPAR_mean'] = merged_df.groupby(['Year', 'site_name'])['FPAR'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['FPAR_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['FPAR'].transform('mean')


merged_df['VPD_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['VPD_F'].transform('mean')
merged_df['VPD_mean'] = merged_df.groupby(['Year', 'site_name'])['VPD_F'].transform(lambda x: x.expanding(min_periods=1).mean())

# PPFD (mol photons m−2 day−1) was estimated as a constant fraction of downwelling, shortwave radiation (SW_IN_F, W m−2) using a conversion factor of 2.04 μmol J−1 (Meek et al., 1984)
merged_df['PPFD'] = merged_df['SW_IN_F'] * 2.04 / 1000000 * 86400
merged_df['PPFD_mean'] = merged_df.groupby(['Year', 'site_name'])['PPFD'].transform(lambda x: x.expanding(min_periods=1).mean())


merged_df['Iabs'] = merged_df['SW_IN_F'] * 2.04 / 1000000 * 86400 * merged_df['FPAR']
merged_df['Iabs_mean'] = merged_df.groupby(['Year', 'site_name'])['Iabs'].transform(lambda x: x.expanding(min_periods=1).mean())
merged_df['Iabs_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['Iabs'].transform('mean')

merged_df['CI'] = 1-(merged_df['SW_IN_F'] / merged_df['SW_IN_POT'])

merged_df['Clumping_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['daily_clumping'].transform('mean')


merged_df = merged_df[merged_df['Iabs_mean'] > 0]


# LMA deducted from eco-optimality (Dong et al., 2022; DOI: 10.1111/1365-2745.13967)
# coefficients derived from Want et al., 2023; DOI: 10.1126/sciadv.add5667
# Define the function to calculate LMA based on Landcover
def calculate_LMA(row):
    # evergreen species
    if row['Landcover'] in ['ENF', 'EBF']:
        return np.exp(0.25 * np.log(row['f']) + 0.5 * np.log(row['Iabs_gs_mean']) - 0.013 * row['TA_F_gs_mean']
                      - 0.27 * row['aridity_gs_mean'] + 3.78)

    # deciduous species
    elif row['Landcover'] in ['CRO', 'DBF', 'DNF', 'GRA', 'SAV']:
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

                 
# Define the function to calculate LL (leaf longevity) based on plant functional types
def calculate_LL(row):
    # mean  temperature
    Ta = row['TA_F_gs_mean'] + 273.16  # [K]
    # mean pressure
    Ps = row['PA_F_gs_mean'] * 1000 # [Pa]
    # mean VPD
    VPD = row['VPD_gs_mean'] * 100 # [Pa]    
    # mean CO2
    Ca = row['CO2_gs_mean']  # [umol/mol]
    
    # gas constant
    R = 8.314e-3  # [kJ K-1 mol-1]
    #  mean O2 partial pressure
    O = Ps * 0.21  # [Pa]
    # Michaelis–Menten constant for CO2 (Bernacchi et al. 2001)
    KC = np.exp(38.05 - 79.43 / (R * Ta)) * 1e-6 * Ps  # [Pa]
    # Michaelis–Menten constant for O2 (Bernacchi et al. 2001)
    KO = np.exp(20.30 - 36.38 / (R * Ta)) * 1000 * 1e-6 * Ps  # [Pa]
    # Michaelis–Menten coefficient for Rubisco-limited photosynthesis (Bernacchi et al. 2001)
    K = KC * (1 + O / KO)  # [Pa]
    # maximum quantum yield of photosystem II (Jiang et al., 2020)
    Phi_max = 0.352 + 0.022 * (Ta - 273.16) - 0.00034 * (Ta - 273.16)**2
    # intrinsic quantum yield
    Phi = Phi_max * row['FPAR_gs_mean'] *  0.5 * 12 / 4
    
    # CO2 compensation point (Bernacchi et al. 2001)
    GammaS = np.exp(19.02 - 37.83 / (R * Ta)) * 1e-6 * Ps  # [Pa]
    # viscosity of water relative to its value at 25 degrees (Wang et al. 2017)
    etaS = np.exp(-580 / (-138 + Ta)**2 * (Ta - 298))  # [-]
    # belta, sensitivity of chi to VPD (Wang et al. 2017)
    ksi = np.sqrt(240 * (K + GammaS) / (1.6 * etaS * 2.4))
    # ratio of intercellular CO2 concentration to ambient CO2 concentration (Wang et al. 2017)
    chi = (ksi + GammaS * np.sqrt(VPD) / Ca) / (ksi + np.sqrt(VPD))  # [-]
    # intercellular CO2 concentration
    Ci = Ca * chi * 1e-6 * Ps  # [Pa]

    # CO2 limitation term (Wang et al. 2017)
    m = (Ci - GammaS) / (Ci + 2 * GammaS)
    # electron transport-limited term
    mc = (Ci - GammaS) / (Ci + K)

    # modified Arrhenius function for Vcmax (Bernacchi et al. 2001)
    Arrhenius = np.exp(72 * (Ta - 298.15) / (R * Ta * 298.15)) * (1 + np.exp(((668 - 1.07*Ta)/1000 * 298.15 - 200) / (R * 298.15))) / (
            1 + np.exp(((668 - 1.07*Ta)/1000 * Ta - 200) / (R * Ta)))

    XT_ev = ((Arrhenius * mc) / (Phi * m))**0.5
    #XT_de = ((Arrhenius * mc) / (Phi * m))

    # evergreen species woody
    if row['Landcover'] in ['ENF', 'EBF']: #eq 24
        return row['LMA'] * XT_ev * (2 * 768 * 13.23 / row['f'] /(row['Iabs_gs_mean'] * 30))**0.5
    
    # deciduous species 
    elif row['Landcover'] in ['CRO','GRA','DBF', 'DNF']:

          return 365 * row['f'] 
          
    else:
        return (row['LMA'] * XT_ev * (2 * 768 * 13.23 / row['f'] /(row['Iabs_gs_mean'] * 30))**0.5                
            + 365 * row['f'])/2


merged_df['LL'] = merged_df.apply(calculate_LL, axis=1)

merged_df = merged_df[merged_df['LL'] < 2000]

# Define the function to calculate Vcmax
def calculate_Vcmax25(row):
    # mean  temperature
    Ta = row['TA_F_mean'] + 273.16  # [K]
    # mean pressure
    Ps = row['PA_F_mean'] * 1000 # [Pa]
    # mean VPD
    VPD = row['VPD_mean'] * 100 # [Pa]    
    # mean CO2
    Ca = row['CO2_mean']  # [umol/mol]
    # gas constant
    R = 8.314e-3  # [kJ K-1 mol-1]
    #  mean O2 partial pressure
    O = Ps * 0.21  # [Pa]
    
    # Michaelis–Menten constant for CO2 (Bernacchi et al. 2001)
    KC = np.exp(38.05 - 79.43 / (R * Ta)) * 1e-6 * Ps  # [Pa]
    # Michaelis–Menten constant for O2 (Bernacchi et al. 2001)
    KO = np.exp(20.30 - 36.38 / (R * Ta)) * 1000 * 1e-6 * Ps  # [Pa]
    # Michaelis–Menten coefficient for Rubisco-limited photosynthesis (Bernacchi et al. 2001)
    K = KC * (1 + O / KO)  # [Pa]
    # maximum quantum yield of photosystem II (Jiang et al., 2020)
    Phi_max = 0.352 + 0.022 * (Ta - 273.16) - 0.00034 * (Ta - 273.16)**2
    # intrinsic quantum yield
    Phi = Phi_max  * row['FPAR_mean'] *  0.5 * 12 / 4
    
    
    # CO2 compensation point (Bernacchi et al. 2001)
    GammaS = np.exp(19.02 - 37.83 / (R * Ta)) * 1e-6 * Ps  # [Pa]
    # viscosity of water relative to its value at 25 degrees (Wang et al. 2017)
    etaS = np.exp(-580 / (-138 + Ta)**2 * (Ta - 298))  # [-]
    # belta, sensitivity of chi to VPD (Wang et al. 2017)
    ksi = np.sqrt(240 * (K + GammaS) / (1.6 * etaS * 2.4))
    # ratio of intercellular CO2 concentration to ambient CO2 concentration (Wang et al. 2017)
    chi = (ksi + GammaS * np.sqrt(VPD) / Ca) / (ksi + np.sqrt(VPD))  # [-]
    # intercellular CO2 concentration
    Ci = Ca * chi * 1e-6 * Ps  # [Pa]

    # CO2 limitation term (Wang et al. 2017)
    m = (Ci - GammaS) / (Ci + 2 * GammaS)
    # electron transport-limited term
    mc = (Ci - GammaS) / (Ci + K)
    
    
    c = np.sqrt(1-(4 * 0.103 /m)**(2./3))
    c = np.real(c)
    m_ = m * c


    # modified Arrhenius function for Vcmax (Bernacchi et al. 2001)
    Arrhenius = np.exp(72 * (Ta - 298.15) / (R * Ta * 298.15)) * (1 + np.exp(((668 - 1.07*Ta)/1000 * 298.15 - 200) / (R * 298.15))) / (
            1 + np.exp(((668 - 1.07*Ta)/1000 * Ta - 200) / (R * Ta)))
    
    # compute Vcmax,25 # [umol m-2 s-1]
    return  Phi * row['PPFD_mean']  * (Ci + K) / (Ci + 2 * GammaS) * np.sqrt( 1- ( 4 * 0.103 *  (Ci + 2 * GammaS) / (Ci - GammaS))**(2./3)) /(60*60*24*1e-6*12) /Arrhenius

merged_df['Vcmax25'] = merged_df.apply(calculate_Vcmax25, axis=1)

merged_df = merged_df[merged_df['Vcmax25'] > 0]
merged_df = merged_df[merged_df['Vcmax25'] < 150]

merged_df['Vcmax25_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['Vcmax25'].transform('mean')



# remove rows where LMA equals to zero
merged_df = merged_df[(merged_df['LMA'] != 0) & (merged_df['LMA'] >= 15)]


# Load the data
LCC_path = r'C:/Users/Yulin/Desktop/Data to Zenodo/LCC 2000-2020 site-nine pixels.xlsx'
LCC = pd.read_excel(LCC_path)

# Convert date column to datetime
LCC['TIMESTAMP'] = pd.to_datetime(LCC['TIMESTAMP'], format='A%Y%j')

# Create a new DataFrame for the daily interpolated data
daily_interp_data = pd.DataFrame(columns=LCC.columns)

# Group the data by 'site_name' and year, and perform interpolation within each group 
def interpolate_group(group):
    site_name = group['site_name'].iloc[0]
    year = group['TIMESTAMP'].dt.year.iloc[0]
    group = group.sort_values(by='TIMESTAMP')
    
    # Create a cubic spline interpolation function
    interp_func = Akima1DInterpolator(group['TIMESTAMP'].dt.dayofyear, group['LCC'])
    
    # Create a new DataFrame with daily timestamps for the given year
    daily_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
    
    # Interpolate data for each daily timestamp
    interpolated_values = interp_func(daily_dates.dayofyear)
    
    # Create a new DataFrame for the interpolated data and return it
    interpolated_data = pd.DataFrame({
        'TIMESTAMP': daily_dates,
        'site_name': site_name,
        'LCC': interpolated_values
    })
    return interpolated_data

# Apply the interpolation function to each group and concatenate the results
daily_interp_data = LCC.groupby(['site_name', LCC['TIMESTAMP'].dt.year]).apply(interpolate_group).reset_index(drop=True)

# Join the two DataFrames on the 'date' and 'sitename' columns
merged_df = pd.merge(daily_interp_data, merged_df, on=['TIMESTAMP', 'site_name'])

merged_df['LCC_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['LCC'].transform('mean')



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


# Load the data
LAI_path = r'E:\Postdoc-LUE\LAI & fPAR\filtered_lai_fpar_data.xlsx'
LAI = pd.read_excel(LAI_path ,sheet_name='data')

# Convert date column to datetime
LAI['TIMESTAMP'] = pd.to_datetime(LAI['TIMESTAMP'], format='%Y-%m-%d')

# Define the filtering function
def filter_data(group):
    lai_threshold = 1
    time_threshold = 5
    
    group['LAI_change'] = group['LAI'].diff()
    group['time_diff'] = group['TIMESTAMP'].diff().dt.days
    
    filtered_group = group[(abs(group['LAI_change'].fillna(0)) < lai_threshold) & (group['time_diff'] < time_threshold)]
    
    return filtered_group

# Apply the filtering function to each group
LAI = LAI.groupby(['site_name', LAI['TIMESTAMP'].dt.year]).apply(filter_data).reset_index(drop=True)



# Create a new DataFrame for the daily interpolated data
LAI_daily_interp_data = pd.DataFrame(columns=LAI.columns)

# Group the data by 'site_name' and year, and perform interpolation within each group 
def interpolate_LAI(group):
    site_name = group['site_name'].iloc[0]
    year = group['TIMESTAMP'].dt.year.iloc[0]
    group = group.sort_values(by='TIMESTAMP')
    
    # Check if arrays have at least two elements
    if len(group['TIMESTAMP'].dt.dayofyear) < 2 or len(group['LAI']) < 2:
        print("Arrays do not contain sufficient elements. Skipping interpolation")
    else:
             
        window_length = min(4, len(group['LAI']) - 1)
        polyorder = min(2, window_length - 1)
        smoothed_LAI = savgol_filter(group['LAI'], window_length=window_length, polyorder=polyorder)
        
        interp_func = Akima1DInterpolator(group['TIMESTAMP'].dt.dayofyear, smoothed_LAI)
        
        # Create a new DataFrame with daily timestamps for the given year
        daily_dates = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        
        # Interpolate data for each daily timestamp
        interpolated_values = interp_func(daily_dates.dayofyear)
        
        # Create a new DataFrame for the interpolated data and return it
        interpolated_data = pd.DataFrame({
            'TIMESTAMP': daily_dates,
            'site_name': site_name,
            'LAI': interpolated_values
        })
        return interpolated_data

# Apply the interpolation function to each group and concatenate the results
LAI_daily_interp_data = LAI.groupby(['site_name', LAI['TIMESTAMP'].dt.year]).apply(interpolate_LAI).reset_index(drop=True)

merged_df = pd.merge( LAI_daily_interp_data,merged_df, on=['site_name', 'TIMESTAMP'])

merged_df['LAI_gs_mean'] = merged_df.groupby(['Year', 'site_name'])['LAI'].transform('mean')


merged_df = merged_df[merged_df['LAI'] > 0]
merged_df = merged_df[merged_df['LAI'] < 8]




#-----------------------------------------------------------------------------#
# Data quality control before analysis
merged_df = merged_df[merged_df['GPP_NT_VUT_REF'] > 0]

merged_df['LUE'] = merged_df['GPP_NT_VUT_REF']/merged_df['Iabs']/12


# constrain LUE whithin reasonable range
merged_df = merged_df[(merged_df['LUE'] < 0.12) & (merged_df['LUE'] > 0.001)]


merged_df = merged_df[merged_df['NEE_VUT_REF_NIGHT_QC'] > 0.8]

# Remove rows with any NaN values
merged_df = merged_df.dropna()

merged_df = merged_df[merged_df['SWC_F_MDS_1'] > 0]

merged_df = merged_df[merged_df['CO2_F_MDS'] > 0]

merged_df = merged_df[merged_df['CI'] > 0]

merged_df = merged_df[merged_df['VPD_F'] > 0]

merged_df = merged_df[merged_df['daily_clumping'] > 0]
merged_df = merged_df[merged_df['daily_clumping'] < 1]

# Remove wetland
merged_df = merged_df[merged_df['Landcover'] != 'WET']

# Remove C4 sites
c4_sites = ['AU-ASM','AU-Ade','AU-Dap','AU-DaS','AU-Dry','AU-Emr','AU-Stp','AU-How','AU-RDF','AU-TTE','CN-Cng','SD-Dem','SN-Dhr','US-KS2','US-Ne1','US-Ne2','CN-Qia','CN-Din','US-Whs','ZM-Mon']

merged_df = merged_df[~merged_df['site_name'].isin(c4_sites)]
#---------------------------------------------------------------------------------------------------#


#XGBoost model climate only
# Import necessary libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap  # Assuming SHAP is installed

# Define dependent variable and independent variables
y = merged_df['LUE']  # Dependent variable
X = merged_df[['SWC_F_MDS_1',  'CO2_F_MDS',  'aridity', 'TA_F', 'VPD_F', 'CI']]

# The new column names
new_columns = ['SWC', 'CO$_2$',  'ESI', 'T$_a$', 'VPD', 'CI$_{cloud}$']
# Rename the columns in X
X.columns = new_columns
feature_names = X.columns

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
mse_list = []
rmse_list = []
r2_list = []

# Iterate through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Build the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, max_depth=6, learning_rate=0.2,random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance using SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance for this fold
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Append results to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate and print the average performance metrics across all folds
average_mse = np.mean(mse_list)
average_rmse = np.mean(rmse_list)
average_r2 = np.mean(r2_list)

print(f"XGBoost Average RMSE: {average_rmse}")
print(f"XGBoost Average R-squared: {average_r2}")



fig, axs = plt.subplots(3, 3)
fig.set_size_inches(18, 16)


#heat map plot
plt.sca(axs[2, 0])

X_test.reset_index(drop=True, inplace=True)

X_test_sampled = X_test.sample(n=2000, random_state=42)

shap_values_sampled = explainer(X_test_sampled)
shap_values_sampled.values = shap_values_sampled.values.round(3) 

shap.plots.heatmap(shap_values_sampled, plot_width=16, cmap='PRGn', show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('SHAP value', labelpad=-25, fontsize=18)

 
plt.title('Model prediction', fontsize=20)
plt.xlabel("Instances", fontsize=18)

plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=18)


plt.sca(axs[1, 0])

# Assuming shap_values is a list of arrays for each output
shap.summary_plot(shap_values, X_test, feature_names=new_columns, plot_size=[16.5,13.5],  show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('Feature value',labelpad=-15, fontsize=18)   
plt.title('Feature importance', fontsize=20)
plt.xlabel("SHAP value", fontsize=18)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=18)


plt.sca(axs[0, 0])

# Create a scatter plot with custom size and alpha in gray
plt.scatter(
    y=y_test,
    x=y_pred,
    s=0.5,       # Adjust 's' for size
    color='gray',  # Color of the scatter points
    label='Data Points'  # Label for the legend
)

sns.kdeplot(y=y_test, x=y_pred, cmap="Blues", ax=axs[0, 0], fill=True, levels=5, alpha=0.85)

# Add a solid 1:1 line with a label for the legend
plt.plot([0, 0.12], [0, 0.12], 'k--', label='1:1 Line')

# Add a regression line between y_test and y_pred
sns.regplot(x=y_pred, y=y_test, scatter=False, 
            color='lightgray', label='Regression Line')
# Calculate and annotate the slope of the regression line
slope = np.polyfit(y_pred, y_test, 1)[0]
plt.annotate(f'slope = {slope:.2f}', xy=(0.001, 0.08), fontsize=14)

# Set axis labels and plot title
plt.xlabel("Predicted PE ($mol$ $C$ $mol$ $^{−1}$ $photons$)", fontsize=16)
plt.ylabel("Observed PE ($mol$ $C$ $mol$ $^{−1}$ $photons$)", fontsize=15)

# Add R2, RMSE, and bias annotations
plt.annotate(f'R$^2$ = {average_r2:.2f}', xy=(0.001, 0.12), fontsize=14)
plt.annotate(f'RMSE = {rmse:.4f} ({(rmse/np.mean(y_test))*100:.2f}%)', xy=(0.001, 0.11), fontsize=14)
# Calculate bias
bias = np.mean(y_pred - y_test)
plt.annotate(f'Bias = {bias:.5f}', xy=(0.001, 0.1), fontsize=14)
plt.annotate(f'n = {len(y_test)}', xy=(0.001, 0.09), fontsize=14)


# Manually add a legend for the KDE plot with a description
legend_elements = [
    plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='1:1 Line'),
    plt.Line2D([0], [0], color='lightgray', lw=2, label='Regression Line')
]
plt.legend(handles=legend_elements, loc='lower right')

# Set tick intervals
tick_interval = 0.02
x_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
y_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlim(-0.005,0.125)
plt.ylim(-0.005,0.125)
plt.title('Climatic factors', fontsize=20)
axs[0, 0].spines['top'].set_visible(False)
axs[0, 0].spines['right'].set_visible(False)  
   
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=16)



#--------------------------------------------------------------------------------------------------------------#

#XGBoost model dynamic leaf traits and canopy structure
# Import necessary libraries
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import shap  # Assuming SHAP is installed

# Define dependent variable and independent variables
y = merged_df['LUE']  # Dependent variable
X = merged_df[['LCC', 'daily_clumping', 'LMA',   'LL',  'Vcmax25', 'LAI' ]]

# The new column names
new_columns = [ 'Chl',  'CI$_{clump}$', 'LMA',   'LL',  'V$_{cmax25}$', 'LAI']
# Rename the columns in X
X.columns = new_columns
feature_names = X.columns
# Standardize the data including dependent variable

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
mse_list = []
rmse_list = []
r2_list = []

# Iterate through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Build the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, max_depth=6, learning_rate=0.2,random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance using SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance for this fold
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Append results to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate and print the average performance metrics across all folds
average_mse = np.mean(mse_list)
average_rmse = np.mean(rmse_list)
average_r2 = np.mean(r2_list)

print(f"XGBoost Average RMSE: {average_rmse}")
print(f"XGBoost Average R-squared: {average_r2}")

#heat map plot
plt.sca(axs[2, 1])

X_test.reset_index(drop=True, inplace=True)

X_test_sampled = X_test.sample(n=2000, random_state=42)

shap_values_sampled = explainer(X_test_sampled)
shap_values_sampled.values = shap_values_sampled.values.round(3)

shap.plots.heatmap(shap_values_sampled, plot_width=16, cmap='PRGn', show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('SHAP value', labelpad=-25, fontsize=18)

 

plt.title('Model prediction', fontsize=20)
plt.xlabel("Instances", fontsize=18)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=18)


plt.sca(axs[1, 1])

# Assuming shap_values is a list of arrays for each output
shap.summary_plot(shap_values, X_test, feature_names=new_columns, plot_size=[16.5,13.5],color_bar=True, show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('Feature value',labelpad=-15, fontsize=18)    
plt.title('Feature importance', fontsize=20)
plt.xlabel("SHAP value", fontsize=18)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=18)

plt.sca(axs[0, 1])

# Create a scatter plot with custom size and alpha in gray
plt.scatter(
    y=y_test,
    x=y_pred,
    s=0.5,       # Adjust 's' for size
    color='gray',  # Color of the scatter points
    label='Data Points'  # Label for the legend
)

sns.kdeplot(y=y_test, x=y_pred, cmap="Purples", ax=axs[0, 1], fill=True, levels=5, alpha=0.85)

# Add a solid 1:1 line with a label for the legend
plt.plot([0, 0.12], [0, 0.12], 'k--', label='1:1 Line')

# Add a regression line between y_test and y_pred
sns.regplot(x=y_pred, y=y_test, scatter=False, 
            color='lightgray', label='Regression Line')
# Calculate and annotate the slope of the regression line
slope = np.polyfit(y_pred, y_test, 1)[0]
plt.annotate(f'slope = {slope:.2f}', xy=(0.001, 0.08), fontsize=14)

# Set axis labels and plot title
plt.xlabel("Predicted PE ($mol$ $C$ $mol$ $^{−1}$ $photons$)", fontsize=16)
#plt.ylabel("Observed PE (mol C mol$^{−1}$ photons)", fontsize=14)
plt.ylabel("")

# Add R2, RMSE, and bias annotations
plt.annotate(f'R$^2$ = {average_r2:.2f}', xy=(0.001, 0.12), fontsize=14)
plt.annotate(f'RMSE = {rmse:.4f} ({(rmse/np.mean(y_test))*100:.2f}%)', xy=(0.001, 0.11), fontsize=14)
# Calculate bias
bias = np.mean(y_pred - y_test)
plt.annotate(f'Bias = {bias:.5f}', xy=(0.001, 0.1), fontsize=14)
plt.annotate(f'n = {len(y_test)}', xy=(0.001, 0.09), fontsize=14)


# Manually add a legend for the KDE plot with a description
legend_elements = [
    plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='1:1 Line'),
    plt.Line2D([0], [0], color='lightgray', lw=2, label='Regression Line')
]
plt.legend(handles=legend_elements, loc='lower right')

# Set tick intervals
tick_interval = 0.02
x_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
y_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlim(-0.005,0.125)
plt.ylim(-0.005,0.125)
plt.title('Plant traits', fontsize=20)
axs[0,1].spines['top'].set_visible(False)
axs[0,1].spines['right'].set_visible(False)  
   
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=16)



#------------------------------------------------------------------------------------------------------------------#
#XGBoost model including all factors

# Define dependent variable and independent variables
y = merged_df['LUE']  # Dependent variable

X = merged_df[['SWC_F_MDS_1', 'LCC', 'CO2_F_MDS', 'daily_clumping', 'LMA',   'LL',  'Vcmax25', 'LAI', 'aridity', 'TA_F', 'VPD_F', 'CI']]

# The new column names
new_columns = ['SWC', 'Chl', 'CO$_2$', 'CI$_{clump}$', 'LMA',   'LL',  'V$_{cmax25}$', 'LAI',  'ESI','T$_a$', 'VPD', 'CI$_{cloud}$',]


# Rename the columns in X
X.columns = new_columns
feature_names = X.columns

# Initialize K-Fold cross-validator
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store evaluation results for each fold
mse_list = []
rmse_list = []
r2_list = []

# Iterate through each fold
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Build the XGBoost model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=2000, max_depth=6, learning_rate=0.2,random_state=42)

    # Train the model
    model.fit(X_train, y_train)

    # Get feature importance using SHAP values
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(X_test)

    # Make predictions on the test data
    y_pred = model.predict(X_test)
    
    # Evaluate the model's performance for this fold
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    # Append results to lists
    mse_list.append(mse)
    rmse_list.append(rmse)
    r2_list.append(r2)

# Calculate and print the average performance metrics across all folds
average_mse = np.mean(mse_list)
average_rmse = np.mean(rmse_list)
average_r2 = np.mean(r2_list)

print(f"XGBoost Average RMSE: {average_rmse}")
print(f"XGBoost Average R-squared: {average_r2}")

#heat map plot
plt.sca(axs[2, 2])

X_test.reset_index(drop=True, inplace=True)

X_test_sampled = X_test.sample(n=2000, random_state=42)

shap_values_sampled = explainer(X_test_sampled)
shap_values_sampled.values = shap_values_sampled.values.round(3)

shap.plots.heatmap(shap_values_sampled, plot_width=16, cmap='PRGn', show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('SHAP value', labelpad=-25, fontsize=18)

 
plt.title('Model prediction', fontsize=20)
plt.xlabel("Instances", fontsize=18)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=16)


plt.sca(axs[1, 2])

# Assuming shap_values is a list of arrays for each output
shap.summary_plot(shap_values, X_test, feature_names=new_columns, plot_size=[16.5,13.5],  show=False)

cbar = plt.gcf().axes[-1]  
cbar.set_ylabel('Feature value',labelpad=-15, fontsize=18)    
plt.title('Feature importance', fontsize=20)
plt.xlabel("SHAP value", fontsize=18)
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=18)

plt.sca(axs[0, 2])

# Create a scatter plot with custom size and alpha in gray
plt.scatter(
    y=y_test,
    x=y_pred,
    s=0.5,       # Adjust 's' for size
    color='gray',  # Color of the scatter points
    label='Data Points'  # Label for the legend
)

sns.kdeplot(y=y_test, x=y_pred, cmap="Greens", ax=axs[0, 2], fill=True, levels=5, alpha=0.85)

# Add a solid 1:1 line with a label for the legend
plt.plot([0, 0.12], [0, 0.12], 'k--', label='1:1 Line')

# Add a regression line between y_test and y_pred
sns.regplot(x=y_pred, y=y_test, scatter=False, 
            color='lightgray', label='Regression Line')
# Calculate and annotate the slope of the regression line
slope = np.polyfit(y_pred, y_test, 1)[0]
plt.annotate(f'slope = {slope:.2f}', xy=(0.001, 0.08), fontsize=14)

# Set axis labels and plot title
plt.xlabel("Predicted PE ($mol$ $C$ $mol$ $^{−1}$ $photons$)", fontsize=16)

#plt.ylabel("Observed PE ($mol$ $C$ $mol$ $^{−1}$ $photons$)", fontsize=14)
plt.ylabel("")



# Add R2, RMSE, and bias annotations
plt.annotate(f'R$^2$ = {average_r2:.2f}', xy=(0.001, 0.12), fontsize=14)
plt.annotate(f'RMSE = {rmse:.5f} ({(rmse/np.mean(y_test))*100:.2f}%)', xy=(0.001, 0.11), fontsize=14)
# Calculate bias
bias = np.mean(y_pred - y_test)
plt.annotate(f'Bias = {bias:.5f}', xy=(0.001, 0.1), fontsize=14)
plt.annotate(f'n = {len(y_test)}', xy=(0.001, 0.09), fontsize=14)

# Manually add a legend for the KDE plot with a description
legend_elements = [
    plt.Line2D([0], [0], color='black', lw=2, linestyle='--', label='1:1 Line'),
    plt.Line2D([0], [0], color='lightgray', lw=2, label='Regression Line')
]
plt.legend(handles=legend_elements, loc='lower right')

# Set tick intervals
tick_interval = 0.02
x_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
y_ticks = np.arange(-0.000, 0.12 + tick_interval, tick_interval)
plt.xticks(x_ticks)
plt.yticks(y_ticks)

plt.xlim(-0.005,0.125)
plt.ylim(-0.005,0.125)
plt.title('All factors', fontsize=20)
axs[0,2].spines['top'].set_visible(False)
axs[0,2].spines['right'].set_visible(False)  
   
plt.tick_params(axis='both', which='both', length=6, width=2, labelsize=16)

#----------------------------------------------------------------------------------------------------------------#

annotations = ['a', 'b', 'c', 'd', 'e', 'f','g','h','i']

for i, ax in enumerate(axs.flatten()):
    ax.text(-0.17, 1.1, annotations[i], transform=ax.transAxes, fontsize=25, va='top', ha='left',fontweight='bold')
    
plt.tight_layout()

plt.savefig('Fig.1.jpg', dpi=600)   

