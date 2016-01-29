"""
Script written to solve for the transport properties of crushed salt.
    Properties are obtained by solving the inverse problem of diffusional
    flow through crushed salt (1D Heat Eqn).
"""
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('/Users/Lampe/PyScripts/CS_Research')
import ResearchFunc as rf


# USER INPUT VALUES
LNTH = 10.0  # [CM] AXIAL LENGTH OF SAMPLE BEING TESTED
DIA = 9.0  # [CM] AVERAGE DIAMETER OF SAMPLE BEING TESTED
TSRT = 100  # [SEC] TIME OF INITIAL MEASUREMENT
TEND = 300  # [SEC] TIME OF FINAL MEASUREMENT
P_US = 8.2*(101325/14.6959)  # [PA] UPSTREAM PRESSURE
P_DS = 1.0*(101325/14.6959)  # [PA] DOWNSTREAM PRESSURE

# CRITICAL VALUES - USED TO NONDIMENSIONALIZE PARAMETERS
PERM_CRIT = 1E-18  # [METER2] PERMEABILITY
VISC_CRIT = 1.983E-5  # [PA-S] VISCOSITY OF AIR AT AMBIENT CONDITIONS
PRES_CRIT = P_US#*(101325/14.6959)  # [PA] MAX DIFFERENTIAL PRESSURE
CGAS_CRIT = 1/PRES_CRIT  # [1/PA] CRITICAL COMPERSSIBILITY
TIME_CRIT = TEND - TSRT  # [SEC] CRITICAL TIME (SHOULD BE TEST DURATION)
LNTH_CRIT = LNTH  # [CM] NORMALIZED OVER THE SAMPLE LENGTH

# MATERIAL PROPERTIES
PERM = 1E-16  # [METER2]
VISC = 1.7798E-5  # [PA-S] AVERAGE VISCOSITY OF N2
CGAS = 1.0/142517.8  # [1/PA] GAS COMPRESSIBILITY - SHOULD UPDATE IN FUTURE
CRCK = 1.0/40E9  # [1/PA] CRUSHED SALT BULK COMPRESSIBILITY [BROOME, 2014]
PORO = 0.03  # [DIMLESS] ROCK POROSITY
AREA = np.pi*DIA**2/4.0  # [CM2] X-SECT AREA OF SAMPLE, ORTH. TO FLOW

# DIMENSIONLESS PARAMETERS
PERM_BAR = PERM/PERM_CRIT
VISC_BAR = VISC/VISC_CRIT
CGAS_BAR = CGAS/CGAS_CRIT
CRCK_BAR = CGAS/CGAS_CRIT
LNTH_BAR = LNTH/LNTH_CRIT
COND_BAR = PERM_BAR / VISC_BAR  # HYDRAULIC CONDUCTIVITY
STOR_BAR = PORO*CGAS_BAR + CRCK_BAR  # SPECIFIC STORAGE
AREA_BAR = AREA/LNTH_CRIT**2

# PROBLEM DISCRETIZATION
# SPATIAL
NEL = 20
NNODE = NEL + 1
NODE_ARR = np.linspace(0, LNTH_CRIT, NNODE)
H_EL_BAR = np.float(NODE_ARR[-1] - NODE_ARR[0])/NEL

# TEMPORAL
TEND_BAR = (TEND - TSRT)/TIME_CRIT
TDEL_BAR = 0.1
TIME_ARR = np.linspace(0, TEND_BAR, TEND_BAR/TDEL_BAR + 1)

# print((TEND - TSRT)/TIME_CRIT)
print(TIME_ARR)

# ACTUAL DISC
H_EL = LNTH/NEL
TDEL = TIME_CRIT*TDEL_BAR

# BOUNDARY CONDTIONS
BCTYPE = [1, 1]  # 1 => essential, 0 => flux
BCUS_BAR = P_US/PRES_CRIT  # essential BCT at x=0, UPSTREAM
BCDS_BAR = P_DS/PRES_CRIT  # essential BCT at x=1, DOWNSTREAM

# INITIAL CONDITIONS
ICOND_BAR = np.zeros(NNODE)

# FORCING FUNCTION
FORC = lambda x: 0*x  # the forcing function F(x)

# INTEGRATION TYPE
ITYPE = 1  # fully implicit (DETERMINES THE INTEGRATION TYPE)

SOLN = rf.HeatEqn_1D(NODE_ARR, TIME_ARR, ICOND_BAR, STOR_BAR, COND_BAR, FORC,
                     BCTYPE, BCUS_BAR, BCDS_BAR, AREA_BAR, ITYPE)
out_i = SOLN.solve()

# PLOT RESULTS
FIG_I, AX1 = plt.subplots(figsize=(12, 8))

LBL = [None]*len(TIME_ARR)
for i in xrange(len(TIME_ARR)):
    LBL[i] = 'time = ' + str(TIME_ARR[i])

# when plotting from arrays, columns from each are plotted against eachother
AX1.plot(NODE_ARR, out_i.T, 'o-', lw=1)
AX1.legend(LBL, frameon=1, framealpha=1, loc=0)

AX1.set_xlabel('Length', fontsize=12)
AX1.set_ylabel('Pressure', fontsize=12)
AX1.set_title(''r'One-Dimensional Transient Analysis, $\alpha = 1 \Rightarrow$ Implicit' , fontsize = 16)
AX1.grid(b=True, which='major')
AX1.grid(b=True, which='major')
# plt.show()
