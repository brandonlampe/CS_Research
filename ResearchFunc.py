"""
    Module to be loaded for research on the consolidation of crushed salt
	Content includes:
		(1) minimeter_import: function for importing data recording using labview at the UNM lab
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import CoolProp as CP
from CoolProp.CoolProp import PropsSI
np.set_printoptions(precision=6, threshold=1000, suppress=False, linewidth=80)

def minimeter_import(fname, delimit=',', skip=1, date_format='%m/%d/%Y %I:%M:%S.%f %p'):
    """
        Function for loading labview data from '.csv' file
    """
    datestr2num = lambda x: datetime.strptime(x, date_format)
    data_inp = np.loadtxt(fname, delimiter=delimit, skiprows=skip, usecols=(1, 2, 3, 4, 5))
    time_str = np.loadtxt(fname, dtype=str, delimiter=delimit, skiprows=skip, usecols=[0]) 
    time_inp = np.zeros((len(data_inp[:, 0]), 2)).astype(datetime)
    start = datestr2num(time_str[0])
    for i in xrange(time_inp.shape[0]):
        time_inp[i, 0] = datestr2num(time_str[i])
        delta = time_inp[i, 0] - start
        time_inp[i, 1] = delta.seconds
    data = np.concatenate((time_inp, data_inp), axis=1)
    return data

def plot_mmdat(data, splice=0, idx_min=0, idx_max=0, 
               time_idx=1, pdiff_idx=2, p_idx=3, T_conf_idx=4, T_us_idx=5, T_ds_idx=6):
    """
    Function for plotting data obtained via LabView from the Minimeter in UNM's geotech lab
    """
    if splice != 0:
        time = data[idx_min:idx_max, time_idx]
        pdiff_psi = data[idx_min:idx_max, pdiff_idx]
        p_psi = data[idx_min:idx_max, p_idx]
        T_conf = data[idx_min:idx_max, T_conf_idx]
        T_us = data[idx_min:idx_max, T_us_idx]
        T_ds = data[idx_min:idx_max, T_ds_idx]
    else:
        time = data[:, time_idx]
        pdiff_psi = data[:, pdiff_idx]
        p_psi = data[:, p_idx]
        T_conf = data[:, T_conf_idx]
        T_us = data[:, T_us_idx]
        T_ds = data[:, T_ds_idx]

    fig_all, ax = plt.subplots(2, sharex=True, figsize=(12, 8))
    ax[0].plot(time, pdiff_psi, "o-")
    ax[1].plot(time, T_conf, "o-")
    ax[1].plot(time, T_us, "o-")
    ax[1].plot(time, T_ds, "o-")
    
    lbl_temp = np.array(['Confing Gas', 'Upstream Gas', 'Downstream Gas']).astype(str)
    ax[1].legend(lbl_temp, frameon=1, framealpha = 1, loc=0)

    ax[1].set_xlabel('Index', fontsize = 12)
    ax[0].set_ylabel('Upstream Pressure (psig)', fontsize = 12)
    ax[1].set_ylabel('Temperature (C)', fontsize = 12)
    ax[0].grid(True, which = 'major')
    ax[1].grid(True, which = 'major')

def visc_n2(T,P):
    """
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    mu = PropsSI('viscosity', 'T', T, 'P', P, 'Nitrogen') # Pa-s
    return mu

def z_n2(T,P):
    """
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    # Density of Air at standard atmosphere in kg/m^3
    z = PropsSI('Z','T',T,'P',P,'Nitrogen')
    return z

def rho_n2(T,P):
    """
    returns density of nitrogen (kg/m3)
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    M = 28.01348 #molecular weight of nitrogen (kg/kg-mole)
    R = 8314.0 #gas const (m3-Pa)/(K-kmol)
    z = z_n2(T,P)
    rho = M*P/(z*R*T) # kg/m3
#     PropsSI('D', 'T', 298.15, 'P', 101325, 'Nitrogen')
    return rho