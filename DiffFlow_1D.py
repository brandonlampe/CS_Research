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
LNTH = 10.0E-2  # [M] AXIAL LENGTH OF SAMPLE BEING TESTED
DIA = 11.283790E-2  # [M] AVERAGE DIAMETER OF SAMPLE BEING TESTED
TSRT = 0  # [SEC] TIME OF INITIAL MEASUREMENT
TEND = 1.0 * 3600  # [SEC] TIME OF FINAL MEASUREMENT

# DEFINE BOUNDARY CONDITIONS
# FLUX:  [meter/sec]
# PRESSURE: [Pa]
BC_US = 20.0 * (101325/14.6959)  # UPSTREAM BCT (PRESSURE/FLUX)
BC_DS = 1E-6  # 0.0 * (101325/14.6959)  # DOWNSTREAM BCT (PRESSURE/FLUX)
P_INIT = 0.0 * (101325/14.6959)  # [PA] INITIAL CONDITIONS
BCTYPE = [1, 0]  # 1=>essential (CONST), 2=>essential (FUNC), 0=>flux

# MATERIAL PROPERTIES
PERM = 1E-17  # [METER2]
VISC = 1.0E-5  # 1.7798E-5  # [PA-S] AVERAGE VISCOSITY OF N2
CGAS = 1.0/142517.8  # [1/PA] GAS COMPRESSIBILITY - SHOULD UPDATE IN FUTURE
CRCK = 1.0/40E9  # [1/PA] CRUSHED SALT BULK COMPRESSIBILITY [BROOME, 2014]
PORO = 0.03  # [DIMLESS] ROCK POROSITY
AREA = np.pi*DIA**2/4.0  # [CM2] X-SECT AREA OF SAMPLE, ORTH. TO FLOW
STOR = PORO*CGAS + CRCK  # [1/PA] SPECIFIC STORAGE
COND = PERM/VISC  # [L2/PA-S] HYDRAULIC CONDUCTIVITY

# REFERENCE VALUES - USED TO NONDIMENSIONALIZE PARAMETERS
PERM_REF = 1E-18  # [METER2] REFERENCE PERMEABILITY
VISC_REF = 1.983E-5  # [PA-S] VISCOSITY OF AIR AT AMBIENT CONDITIONS
PRES_REF = BC_US  # [PA] MAX DIFFERENTIAL PRESSURE
COND_REF = PERM_REF/VISC_REF  # [L2/PA-S] HYDRAULIC CONDUCTIVITY
LNTH_REF = LNTH  # [M] SAMPLE LENGTH
TIME_REF = (LNTH_REF**2 * STOR)/COND_REF
FORC_REF = COND_REF*PRES_REF/LNTH_REF**2
FLUX_REF = (PRES_REF*COND_REF)/LNTH_REF

# CONSTANT DIMENSIONLESS PARAMETERS
LNTH_BAR = LNTH/LNTH_REF
COND_BAR = COND/COND_REF  # HYDRAULIC CONDUCTIVITY
AREA_BAR = AREA/LNTH_REF**2


# DISCRETIZATION OF INDEPENDENT VARIABLES
NEL = 20  # NUMBER OF SPATIAL INCREMENTS
TIME_INC = 200  # NUMBER OF TEMPORAL INCREMENTS

# SETUP SPATIAL NODES
NNODE = NEL + 1
NODE_ARR = np.linspace(0, LNTH, NNODE)
EL_SIZE = np.float(NODE_ARR[-1] - NODE_ARR[0])/NEL  # SIZE OF ELEMENT

# SETUP TEMPORAL NODES
TIME_DUR = TEND - TSRT  # DURATION OF ANALYSIS
TIME_ARR = np.linspace(0, TIME_DUR, TIME_INC + 1)
S_SIZE = np.float(TIME_ARR[-1] - TIME_ARR[0])/TIME_INC  # SIZE OF TIME STEP

# DIMENSIONLESS FORMS OF INDEPENDENT VARIABLES
NODE_BAR = NODE_ARR/LNTH_REF
TIME_BAR = TIME_ARR/TIME_REF

# BOUNDARY CONDTIONS (DIMENSIONLESS)
if BCTYPE[0] != 0:  # UPSTREAM BCT
    # ESSENTIAL BC
    BCUS_BAR = BC_US/PRES_REF  # essential BCT at x=0, UPSTREAM
else:
    # FLUX BC
    BCUS_BAR = BC_US*(LNTH_REF/(PRES_REF*COND_REF))

if BCTYPE[1] != 0:  # DOWNSTREAM BCT
    # ESSENTIAL BC
    BCDS_BAR = BC_DS/PRES_REF  # essential BCT at x=0, UPSTREAM
else:
    # FLUX BC
    BCDS_BAR = BC_DS*(LNTH_REF/(PRES_REF*COND_REF))
    print(BCDS_BAR)
# initial contions at all nodes
ICOND_BAR_VAL = P_INIT/PRES_REF

# INITIAL CONDITIONS (DIMENSIONLESS)
ICOND_BAR = np.ones(NNODE)*ICOND_BAR_VAL

# FORCING FUNCTION
def forc_func(x):
    """
    NEED TO BE AWARE OF DIMENSIONS OF FORCING FUNCTION
    """
    return x*0  # CHECK DIMENSIONS! forcing function F(x)

FORC_BAR = forc_func(NODE_BAR)/FORC_REF  # DIMENSIONLESS FORM

# INTEGRATION TYPE
ITYPE = 1  # fully implicit (DETERMINES THE INTEGRATION TYPE)

# DEFINE THE CLASS
SOLN = rf.GasFlow_1D(NODE_BAR, TIME_BAR, ICOND_BAR, 1.0, COND_BAR, forc_func,
                     BCTYPE, BCUS_BAR, BCDS_BAR, AREA_BAR, ITYPE)
PRES_BAR = SOLN.solve()

PLT_INC = TIME_INC/10.0
print(TIME_ARR[0::PLT_INC])  #[0::PLT_INC]

LBL = [None]*len(TIME_ARR[0::PLT_INC])
for i in xrange(len(TIME_ARR[0::PLT_INC])):
    LBL[i] = 'time = {:.3g} hr'.format(TIME_ARR[i*PLT_INC]/3600)  # + str(TIME_ARR[i])

"""
PLOTTING BELOW
"""
# CONVERT FROM DIMENSIONLESS VALUES
PRES_PA = PRES_BAR * PRES_REF
PRES_KPA = PRES_BAR * PRES_REF / 1000.0
PRES_PSI = PRES_KPA * (14.6959/101.325)
NODE_M = NODE_BAR * LNTH_REF
NODE_IN = NODE_M / 0.3048


# FIG_I, AX1 = plt.subplots(figsize=(12, 8))
# # when plotting from arrays, columns from each are plotted against eachother
# AX1.plot(NODE_BAR, PRES_BAR[0::PLT_INC,:].T, 'o-', lw=1)
# AX1.legend(LBL, frameon=1, framealpha=1, loc=0)
# AX1.set_xlabel('Dimensionless Length', fontsize=12)
# AX1.set_ylabel('Dimensionless Pressure', fontsize=12)
# AX1.set_title(''r'One-Dimensional Transient Analysis', fontsize=16)
# AX1.grid(b=True, which='major')
# AX1.grid(b=True, which='major')


FIG_II, AX2 = plt.subplots(figsize=(12, 8))
# when plotting from arrays, columns from each are plotted against eachother
AX2.plot(NODE_IN, PRES_PSI[0::PLT_INC,:].T, 'o-', lw=1)
AX2.legend(LBL, frameon=1, framealpha=1, loc=0)
AX2.set_xlabel('Length [inch]', fontsize=12)
AX2.set_ylabel('Pressure [psi]', fontsize=12)
AX2.set_title(''r'Transient Analysis', fontsize=16)
AX2.grid(b=True, which='major')
AX2.grid(b=True, which='major')


# FIG_II, AX3 = plt.subplots(figsize=(12, 8))
# # when plotting from arrays, columns from each are plotted against eachother
# AX3.plot(NODE_M, PRES_KPA[0::PLT_INC,:].T, 'o-', lw=1)
# AX3.legend(LBL, frameon=1, framealpha=1, loc=0)
# AX3.set_xlabel('Length [meter]', fontsize=12)
# AX3.set_ylabel('Pressure [kPa]', fontsize=12)
# AX3.set_title(''r'Transient Analysis', fontsize=16)
# AX3.grid(b=True, which='major')
# AX3.grid(b=True, which='major')

# SLOPE = (PRES_PA[:, -1] - PRES_PA[:, -2]) / EL_SIZE


# print(NODE_M)
# print(PRES_KPA[:, -2:])
# print(SLOPE)
plt.show()
#
