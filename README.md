# Crushed Salt Research Directory
This directory contains Python scripts for the analysis of crushed salt.

**Author: Brandon Lampe**

### ResearchFunc.py
    A group of special purpose functions used in the analysis of crushed salt, the functions included are:

        - GasFlow1D: solves transient gas (copmressible) flow in porous media
            - in 1D (Finite Diff)
            - handles Dirichlet and Neumann type BCTs
        - HeatEqn_1D: solves transient heat equation (incompressible)
            - in 1D (Finite Diff)
            - only Dirichlet type BCTs work
        - minimeter_import: function for importing data recording using
            labview at the UNM lab
        - plot_mmdat: plots data obtained form MiniMeter
        - visc_n2: uses CoolProp to calc viscosity of N2
        - z_n2: uses CoolProp to calc compressiblity factor of N2
        - rho_n2: uses CoolProp to calc density of N2
        - runavg: calculates the running avg of values in a vector

### DiffFlow_1D.py
    A script used as a wrapper to GasFlow1D to slve 1D compressible flow through porous media.  The script solves for $P$ in the following equation:

    \begin(equation}
        D \frac{\partial^2}{\partial x^2}\left( P^2 \right) + Q(x) = \frac{\partial \left( P^2 \right)}{\partial t}
    \end{equation}

        - formats input data (dimensionless form)
        - calls functions ResearchFunc.py
        - plots in primary variable $(P)$ versus indpendent variables $(x, t)$ in multiple units (SI, US, or Dimensionless)
