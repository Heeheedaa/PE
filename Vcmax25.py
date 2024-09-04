# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 15:10:12 2024

@author: Yulin
"""

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
    A = Phi * row['PPFD_mean'] * row['FPAR_mean'] * m_    # [gC m-2 d-1] 

    # modified Arrhenius function for Vcmax (Bernacchi et al. 2001)
    Arrhenius = np.exp(72 * (Ta - 298.15) / (R * Ta * 298.15)) * (1 + np.exp(((668 - 1.07*Ta)/1000 * 298.15 - 200) / (R * 298.15))) / (
            1 + np.exp(((668 - 1.07*Ta)/1000 * Ta - 200) / (R * Ta)))
    
    # compute Vcmax,25 # [umol m-2 s-1]
    return  Phi * row['PPFD_mean']  * (Ci + K) / (Ci + 2 * GammaS) * np.sqrt( 1- ( 4 * 0.103 *  (Ci + 2 * GammaS) / (Ci - GammaS))**(2./3)) /(60*60*24*1e-6*12) /Arrhenius
    #return  A / ((Ci - GammaS)/(Ci + K)) / (60*60*24*1e-6*12) /Arrhenius # [umol m-2 s-1]

merged_df['Vcmax25'] = merged_df.apply(calculate_Vcmax25, axis=1)