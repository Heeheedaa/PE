# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:07:06 2024

@author: Yulin
"""

                      
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