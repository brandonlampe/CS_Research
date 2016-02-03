"""
    Module to be loaded for research on the consolidation of crushed salt
    Content includes:
        (1) HeatEqn_1D: solves transient heat equation in 1D (Finite Diff)
        (2) minimeter_import: function for importing data recording using
            labview at the UNM lab
        (3) plot_mmdat: plots data obtained form MiniMeter
        (4) visc_n2: uses CoolProp to calc viscosity of N2
        (5) z_n2: uses CoolProp to calc compressiblity factor of N2
        (6) rho_n2: uses CoolProp to calc density of N2
        (7) runavg: calculates the running avg of values in a vector
"""

import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
# import CoolProp as CP
from CoolProp.CoolProp import PropsSI
from scipy.sparse import diags
import scipy.linalg
np.set_printoptions(precision=6, threshold=1000, suppress=False, linewidth=80)



class GasFlow_1D():
    """
    One-dimension gas flow through porous media
    """
    def __init__(self, nodes, times, IC, C=lambda x: 1, k=lambda x: 1,
                 f=lambda x: 0,
                 BC_type=[1, 0], BC_0=1, BC_L=0, area=1, alpha=0.5):
        """
            Initiates an object from the argument list to perform FD solution
                 of the steday-state 1D heat equation.
            Primary variable: T
            Independent variables: x & t

            **Governing Equation:
            C \frac{dT}{dt} - \frac{d}{dx}\left(k^A \frac{dT}{dx}\right) = f(x)

            * **Problem Domain**  $ x0\le x \le xL \quad & t0\le t \le tf$

            Input Arguments:
            (required) nodes: 1D array of nodal locations
            (required) times: 1D array of
            (required) IC: intial value of primary variable, these values may
                 be overwritten at boundarys
            (optional) C : capacitance, may be a function of x (default = 1)
            (optional) k : coeficient function of x (default = 1)
            (optional) f : forcing function of x (default = 0)
            (optional) BC_type: tuple for defining boundary condition types ->
                 [BC @ x = 0, BC @ x = L]
                Dirichlete (essential)
                    => 1, place "1" for constant BC
                    => 2, place "2" for BC that is a function of time
                    temperature prescribed (default = 0)
                Neumann (natural) => 0, place "0" for this type of BC
                    flux prescribed : Q = nkA (dT/dx)
                    where n = outward normal, A = cross-sectional area,
                        Q = flux, k = thermal cond.
                e.g., essential BC at x=0 and neumann at x=L -> BC_type = [1,0]
            (optional) BC_0: value of prescribed temperature or flux at x = 0
            (optional) BC_L: value of prescribed temperature or flux at x = L
            (optional) area: cross-sectional area of domain, orthogonal to
                 direction of heat flow (default = 1)
            (optional) alpha: defines the numerical integraction method
             (general trapezoidal rule) (default = 0.5)
                alpha = 0 => explicit integration
                alpha = 0.5 => Implicit integration (highest rate of
                     convergence) (default)
                alpha = 1 => fully implicit (most stable for dynamic problems)

            Output:
            T_sol: 2D array of the numerical approximation at defined nodes and
                 points in time
                - each row represents the spatial solution at a point in time,
                     T_sol[8,:] = solution at all nodes during the 8th
                     time step
                - each column reprsents temporal solution at a node,
                     T_sol[:,3] = solution at the third node for all times
            Example Input:
                example = OneDim_Trans_HeatEqn_FD(nodes,t_arr,IC, C, k, F,
                     bc_type, T_0, T_L, area, alpha)
                The solution will be stored in the Class "example" and the
                     solution is obtained by calling the "solve"
                     method: example.solve()
        """

        self.nodes = np.array(nodes, dtype=np.double)  # spatial discretization
        self.t = np.array(times, dtype=np.double)  # temporal discretization
        self.k = np.double(k)  # thermal conductivity
        self.f = f  # forcing function
        self.IC = np.array(IC, dtype=np.double)**2  # initial conditions
        self.alpha = alpha  # defines integration method

        # defines type of BCT:
        self.BC_type = BC_type  # 1=> Dirichlete,\ !=1 => Flux (Neumann)
        self.BC_0 = BC_0  # magnitude of BCT at x = 0
        self.BC_L = BC_L  # magnitude of BCT at x = L

        self.area = np.double(area)  # x-sectional area perpendicular to x
        self.h = (max(nodes) - min(nodes)) /\
                 (np.double(len(nodes)) - 1)  # element size
        self.node_cnt = len(nodes)
        self.C_func = C

    def time_step(self):
        if len(self.t) > 1:
            self.s = self.t[1] - self.t[0]  # time step size
        else:
            self.s = 1
        return self.s

    # def get_kA(self):
    #     """
    #         returns hydraulic conductivity multiplied by area (kA)
    #         k = perm/visc
    #     """
    #     self.kA = self.k * self.area
    #     return self.kA

    def assemble_C(self):
        """
            builds diagonal capacitance matrix [C]
            - the nodes for x=0 and x=L will be modified for BCTs
        """
        # capacitance array
        self.C = self.C_func * diags([1], [0], shape=(self.node_cnt,
                                                      self.node_cnt)).toarray()
        return self.C

    def assemble_K(self):
        """
            builds stiffness matrix [K]
            - the nodes for x=0 and x=L will be modified for BCTs
            - interior nodes will not be modified further
        """
        self.K = diags([-1, 2, -1], [-1, 0, 1],
                       shape=(self.node_cnt,
                              self.node_cnt)).toarray()  # interior nodes
        # accounts for thermal cond & element spacing
        self.K = (self.k*self.area)/np.double(self.h)**2 * self.K
        return self.K

    def apply_bc_A(self):
        """
            account for boundary conditions at x = 0 & x = L
            - additionally, the stiffness matrix [K] is modified to maintain
                 symmetry => positive definiteness
        """
        # apply BC at x = 0
        if self.BC_type[0] != 0:
            # essential BC
            self.A[0, 0:2] = np.array([1, 0])  # modifies first equation
            self.A[1, 0] = 0  # modification to mainatin symmetry
        elif self.BC_type[0] == 0:
            # natural BC, Flux (Q*)
            # n = -1.0  # unit outward normal
            # dT = self.BC_0 / (self.kA * n)

            # modifies [K] first equation
            self.A[0, 0:2] = self.k/np.double(self.h)**2 * np.array([1, -1])

        # apply BC at x = L
        if self.BC_type[1] != 0:
            # essential BC
            self.A[-1][-2:] = np.array([0, 1])  # modifies last equation
            self.A[-2][-1] = 0  # modification to mainatin symmetry
        elif self.BC_type[1] == 0:
            # natural BC, flux (Q*)
            # n = 1.0  # unit outward normal
            # dT = self.BC_L / (self.kA * n)

            # modifies last equation
            self.A[-1][-2:] = (self.k / (self.h)**2) * np.array([-1, 1])
        return self.A

    def apply_IC(self):
        """
            account for initial conditions at (boundaries) x = 0 & x = L
        """
        # force initial condition to satisfy BC
        # boundary condtion at x = 0
        if self.BC_type[0] == 1:
            # essential BC - constant
            self.IC[0] = self.BC_0**2
        elif self.BC_type[0] == 2:
            # essential BC - time dependent
            self.IC[0] = self.BC_0(self.t[0])**2

        # boundary condtion at x = L
        if self.BC_type[1] == 1:
            # essential BC - constant
            self.IC[-1] = self.BC_L**2
        elif self.BC_type[1] == 2:
            # essential BC - time dependent
            self.IC[-1] = self.BC_L(self.t[0])**2
        return self.IC

    def apply_bc_b(self, P_old, t=0):
        """
            account for boundary conditions in the {b} vector at x = 0 & x = L
            - additionally, modifications made to maintain symmetry of [A] =>
                positive definiteness
        """
        # boundary condtion at x = 0
        if self.BC_type[0] == 1:
            # essential BC - constant
            self.b[0] = self.BC_0**2  # modifies first equation
            # modification to mainatin symmetry
            self.b[1] = self.b[1] - self.A_old[0] * self.BC_0**2
        elif self.BC_type[0] == 2:
            # essential BC - time dependent
            self.b[0] = self.BC_0(t)**2  # modifies first equation
            # modification to mainatin symmetry
            self.b[1] = self.b[1] - self.A_old[0] * self.BC_0(t)**2
        elif self.BC_type[0] == 0:
            # natural BC, Flux (Q*)
            n = -1.0  # unit outward normal
            term_a = self.BC_L / (self.k * n)
            term_b = (term_a * self.h + np.sqrt(P_old[0]))**2 - P_old[0]
            # modifies {F} last equation
            self.b[0] = (self.k/self.h**2) * term_b

        # boundary condition at x = L
        if self.BC_type[1] == 1:
            # essential BC - constant
            self.b[-1] = self.BC_L**2  # modifies last equation
            # modification to mainatin symmetry
            self.b[-2] = self.b[-2] - self.A_old[1] * self.BC_L**2
        elif self.BC_type[1] == 2:
            # essential BC - time dependent
            self.b[-1] = self.BC_L(t)**2  # modifies last equation
            # modification to mainatin symmetry
            self.b[-2] = self.b[-2] - self.A_old[1] * self.BC_L(t)**2
        elif self.BC_type[1] == 0:
            # natural BC, flux (Q*)
            n = 1.0  # unit outward normal
            term_a = self.BC_L / (self.k * n)
            term_b = (term_a * self.h + np.sqrt(P_old[-2]))**2 - P_old[-2]
            # modifies {F} last equation
            self.b[-1] = (self.k/self.h**2) * term_b
        return self.b

    def solve(self):
        """
            Main function where the transient problem is solve
        """
        self.time_step()  # call in step size (uniform)
        self.assemble_C()  # call in capacitance matrix
        self.assemble_K()  # call in stiffness matrix
        # self.get_kA()
        T_sol = np.zeros((len(self.t), len(self.nodes)))  # build soln array

#         ipdb.set_trace()
        self.A = self.C + self.alpha * self.s * self.K  # build [A]
        # components to enforce symm., use in "apply_bc_b"
        self.A_old = np.array([self.A[1, 0], self.A[-2][-1]])
        self.B = self.C - (1 - self.alpha) * self.s * self.K  # build [B]

        # forcing function is assumed constant in time
        self.F = self.alpha * self.s * self.f(self.nodes) + \
            (1 - self.alpha) * self.s * self.f(self.nodes)
        # make modifications to [A] and intial conditions for bdry conditions
        self.apply_bc_A()
        self.apply_IC()  # enforce initial conditions to match BCTs

        T_old = self.IC  # current time
        T_sol[0, :] = T_old  # assign first row from initial conditions
        # decompose [A], Aq -> orthogonal, Ar -> upper triangular
        Aq, Ar = scipy.linalg.qr(self.A)

        for i in xrange(len(self.t[1:])):
            # solve: [A]{T_new} = {b} for all times
            t = self.t[i+1]
            self.b = np.dot(self.B, T_old) + self.f(self.nodes)
            self.apply_bc_b(T_old, t)  # apply boundary conditions to {b}
            b_hat = np.transpose(Aq).dot(self.b)  # [Q]^inv {b} = b_hat
            # performs back substitution
            T_new = scipy.linalg.solve_triangular(Ar, b_hat)
            T_sol[i+1, :] = T_new
            T_old = T_new
        return np.sqrt(T_sol)



class HeatEqn_1D():
    """
    One-dimension heat equation
    """
    def __init__(self, nodes, times, IC, C=lambda x: 1, k=lambda x: 1,
                 f=lambda x: 0,
                 BC_type=[1, 0], BC_0=1, BC_L=0, area=1, alpha=0.5):
        """
            Initiates an object from the argument list to perform FD solution
                 of the steday-state 1D heat equation.
            Primary variable: T
            Independent variables: x & t

            **Governing Equation:
            C \frac{dT}{dt} - \frac{d}{dx}\left(k^A \frac{dT}{dx}\right) = f(x)

            * **Problem Domain**  $ x0\le x \le xL \quad & t0\le t \le tf$

            Input Arguments:
            (required) nodes: 1D array of nodal locations
            (required) times: 1D array of
            (required) IC: intial value of primary variable, these values may
                 be overwritten at boundarys
            (optional) C : capacitance, may be a function of x (default = 1)
            (optional) k : coeficient function of x (default = 1)
            (optional) f : forcing function of x (default = 0)
            (optional) BC_type: tuple for defining boundary condition types ->
                 [BC @ x = 0, BC @ x = L]
                Dirichlete (essential)
                    => 1, place "1" for constant BC
                    => 2, place "2" for BC that is a function of time
                    temperature prescribed (default = 0)
                Neumann (natural) => 0, place "0" for this type of BC
                    flux prescribed : Q = nkA (dT/dx)
                    where n = outward normal, A = cross-sectional area,
                        Q = flux, k = thermal cond.
                e.g., essential BC at x=0 and neumann at x=L -> BC_type = [1,0]
            (optional) BC_0: value of prescribed temperature or flux at x = 0
            (optional) BC_L: value of prescribed temperature or flux at x = L
            (optional) area: cross-sectional area of domain, orthogonal to
                 direction of heat flow (default = 1)
            (optional) alpha: defines the numerical integraction method
             (general trapezoidal rule) (default = 0.5)
                alpha = 0 => explicit integration
                alpha = 0.5 => Implicit integration (highest rate of
                     convergence) (default)
                alpha = 1 => fully implicit (most stable for dynamic problems)

            Output:
            T_sol: 2D array of the numerical approximation at defined nodes and
                 points in time
                - each row represents the spatial solution at a point in time,
                     T_sol[8,:] = solution at all nodes during the 8th
                     time step
                - each column reprsents temporal solution at a node,
                     T_sol[:,3] = solution at the third node for all times
            Example Input:
                example = OneDim_Trans_HeatEqn_FD(nodes,t_arr,IC, C, k, F,
                     bc_type, T_0, T_L, area, alpha)
                The solution will be stored in the Class "example" and the
                     solution is obtained by calling the "solve"
                     method: example.solve()
        """

        self.nodes = np.array(nodes, dtype=np.double)  # spatial discretization
        self.t = np.array(times, dtype=np.double)  # temporal discretization
        self.k = k  # thermal conductivity
        self.f = f  # forcing function
        self.IC = np.array(IC, dtype=np.double)  # initial conditions
        self.alpha = alpha  # defines integration method

        # defines type of BCT:
        self.BC_type = BC_type  # 1=> Dirichlete,\ !=1 => Flux (Neumann)
        self.BC_0 = BC_0  # magnitude of BCT at x = 0
        self.BC_L = BC_L  # magnitude of BCT at x = L

        self.area = area  # x-sectional area perpendicular to x
        self.h = (max(nodes) - min(nodes)) /\
                 (np.double(len(nodes)) - 1)  # element size
        self.node_cnt = len(nodes)
        self.C_func = C

    def time_step(self):
        if len(self.t) > 1:
            self.s = self.t[1] - self.t[0]  # time step size
        else:
            self.s = 1
        return self.s

    def get_kA(self):
        """
            returns thermal conductivity multiplied by area (kA)
        """
        self.kA = self.k * self.area
        return self.kA

    def assemble_C(self):
        """
            builds diagonal capacitance matrix [C]
            - the nodes for x=0 and x=L will be modified for BCTs
        """
        # capacitance array
        self.C = self.C_func * diags([1], [0], shape=(self.node_cnt,
                                                      self.node_cnt)).toarray()
        return self.C

    def assemble_K(self):
        """
            builds stiffness matrix [K]
            - the nodes for x=0 and x=L will be modified for BCTs
            - interior nodes will not be modified further
        """
        self.K = diags([-1, 2, -1], [-1, 0, 1],
                       shape=(self.node_cnt,
                              self.node_cnt)).toarray()  # interior nodes
        # accounts for thermal cond & element spacing
        self.K = (self.k*self.area)/np.double(self.h)**2 * self.K
        return self.K

    def apply_bc_A(self):
        """
            account for boundary conditions at x = 0 & x = L
            - additionally, the stiffness matrix [K] is modified to maintain
                 symmetry => positive definiteness
        """
        # apply BC at x = 0
        if self.BC_type[0] != 0:
            # essential BC
            self.A[0, 0:2] = np.array([1, 0])  # modifies first equation
            self.A[1, 0] = 0  # modification to mainatin symmetry
        elif self.BC_type[0] == 0:
            # natural BC, Flux (Q*)
            n = -1.0  # unit outward normal
            dT = self.BC_0 / (self.kA * n)
            # modifies [K] first equation
            self.A[0, 0:2] = self.kA/np.double(self.h)**2 * np.array([1, -1])

        # apply BC at x = L
        if self.BC_type[1] != 0:
            # essential BC
            self.A[-1][-2:] = np.array([0, 1])  # modifies last equation
            self.A[-2][-1] = 0  # modification to mainatin symmetry
        elif self.BC_type[1] == 0:
            # natural BC, flux (Q*)
            n = 1.0  # unit outward normal
            dT = self.BC_L / (self.kA * n)
            self.A[-1][-2:] = self.kA/np.double(self.h)**2 *\
                np.array([-1, 1])  # modifies last equation
        return self.A

    def apply_IC(self):
        """
            account for initial conditions at (boundaries) x = 0 & x = L
        """
        # force initial condition to satisfy BC
        # boundary condtion at x = 0
        if self.BC_type[0] == 1:
            # essential BC - constant
            self.IC[0] = self.BC_0
        elif self.BC_type[0] == 2:
            # essential BC - time dependent
            self.IC[0] = self.BC_0(self.t[0])

        # boundary condtion at x = L
        if self.BC_type[1] == 1:
            # essential BC - constant
            self.IC[-1] = self.BC_L
        elif self.BC_type[1] == 2:
            # essential BC - time dependent
            self.IC[-1] = self.BC_L(self.t[0])
        return self.IC

    def apply_bc_b(self, t=0):
        """
            account for boundary conditions in the {b} vector at x = 0 & x = L
            - additionally, modifications made to maintain symmetry of [A] =>
                positive definiteness
        """
        # boundary condtion at x = 0
        if self.BC_type[0] == 1:
            # essential BC - constant
            self.b[0] = self.BC_0  # modifies first equation
            # modification to mainatin symmetry
            self.b[1] = self.b[1] - self.A_old[0] * self.BC_0
        elif self.BC_type[0] == 2:
            # essential BC - time dependent
            self.b[0] = self.BC_0(t)  # modifies first equation
            # modification to mainatin symmetry
            self.b[1] = self.b[1] - self.A_old[0] * self.BC_0(t)
        elif self.BC_type[0] == 0:
            # natural BC, Flux (Q*)
            n = -1.0  # unit outward normal
            dT = self.BC_0 / (self.kA * n)
            # modifies {F} first equation
            self.b[0] = self.kA/np.double(self.h)**2 *\
                (-self.kA/np.double(self.h)) * dT

        # boundary condition at x = L
        if self.BC_type[1] == 1:
            # essential BC - constant
            self.b[-1] = self.BC_L  # modifies last equation
            # modification to mainatin symmetry
            self.b[-2] = self.b[-2] - self.A_old[1] * self.BC_L
        elif self.BC_type[1] == 2:
            # essential BC - time dependent
            self.b[-1] = self.BC_L(t)  # modifies last equation
            # modification to mainatin symmetry
            self.b[-2] = self.b[-2] - self.A_old[1] * self.BC_L(t)
        elif self.BC_type[1] == 0:
            # natural BC, flux (Q*)
            n = 1.0  # unit outward normal
            dT = self.BC_L / (self.kA * n)
            self.b[-1] = self.kA/np.double(self.h)**2 *\
                self.kA/np.double(self.h) * dT  # modifies last equation
        return self.b

    def solve(self):
        """
            Main function where the transient problem is solve
        """
        self.time_step()  # call in step size (uniform)
        self.assemble_C()  # call in capacitance matrix
        self.assemble_K()  # call in stiffness matrix
        self.get_kA()
        T_sol = np.zeros((len(self.t), len(self.nodes)))  # build soln array

#         ipdb.set_trace()
        self.A = self.C + self.alpha * self.s * self.K  # build [A]
        # components to enforce symm., use in "apply_bc_b"
        self.A_old = np.array([self.A[1, 0], self.A[-2][-1]])
        self.B = self.C - (1 - self.alpha) * self.s * self.K  # build [B]

        # forcing function is assumed constant in time
        self.F = self.alpha * self.s * self.f(self.nodes) + \
            (1 - self.alpha) * self.s * self.f(self.nodes)
        # make modifications to [A] and intial conditions for bdry conditions
        self.apply_bc_A()
        self.apply_IC()  # enforce initial conditions to match BCTs

        T_old = self.IC  # current time
        T_sol[0, :] = T_old  # assign first row from initial conditions
        # decompose [A], Aq -> orthogonal, Ar -> upper triangular
        Aq, Ar = scipy.linalg.qr(self.A)

        for i in xrange(len(self.t[1:])):
            # solve: [A]{T_new} = {b} for all times
            t = self.t[i+1]
            self.b = np.dot(self.B, T_old) + self.f(self.nodes)
            self.apply_bc_b(t)  # apply boundary conditions to {b}
            b_hat = np.transpose(Aq).dot(self.b)  # [Q]^inv {b} = b_hat
            # performs back substitution
            T_new = scipy.linalg.solve_triangular(Ar, b_hat)
            T_sol[i+1, :] = T_new
            T_old = T_new
        return T_sol


def minimeter_import(fname, delimit=',', skip=1,
                     date_format='%m/%d/%Y %I:%M:%S.%f %p'):
    """
        Function for loading labview data from '.csv' file
    """
    # datestr2num = def date2num(x):datetime.strptime(x, date_format)
    def datestr2num(date, fmt):
        fmt_num = datetime.strptime(date, fmt)
        return fmt_num

    data_inp = np.loadtxt(fname, delimiter=delimit, skiprows=skip,
                          usecols=(1, 2, 3, 4, 5))
    time_str = np.loadtxt(fname, dtype=str, delimiter=delimit, skiprows=skip,
                          usecols=[0])
    time_inp = np.zeros((len(data_inp[:, 0]), 2)).astype(datetime)
    start = datestr2num(time_str[0], date_format)
    for i in xrange(time_inp.shape[0]):
        time_inp[i, 0] = datestr2num(time_str[i], date_format)
        delta = time_inp[i, 0] - start
        time_inp[i, 1] = delta.seconds
    data = np.concatenate((time_inp, data_inp), axis=1)
    return data


def plot_mmdat(data, splice=0, idx_min=0, idx_max=0,
               time_idx=1, pdiff_idx=2, p_idx=3, T_conf_idx=4,
               T_us_idx=5, T_ds_idx=6):
    """
    Function for plotting data obtained via LabView from the Minimeter
         in UNM's geotech lab

    Input
    data (required): data from minimeter DAQ for plotting
    splice (opt): determines of the whole or just a splice of data will
                  be plotted.
                  splice = 0: all data will be plotted
                  splice = 1: data defined by idx_min and idx_max will be
                  plotted
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

    FIG_ALL, AX1 = plt.subplots(2, sharex=True, figsize=(12, 8))
    AX1[0].plot(time, pdiff_psi, "o-")
    AX1[1].plot(time, T_conf, "o-")
    AX1[1].plot(time, T_us, "o-")
    AX1[1].plot(time, T_ds, "o-")

    lbl_temp = np.array(['Confing Gas', 'Upstream Gas',
                         'Downstream Gas']).astype(str)
    AX1[1].legend(lbl_temp, frameon=1, framealpha=1, loc=0)

    AX1[1].set_xlabel('Index', fontsize=12)
    AX1[0].set_ylabel('Upstream Pressure (psig)', fontsize=12)
    AX1[1].set_ylabel('Temperature (C)', fontsize=12)
    AX1[0].grid(True, which='major')
    AX1[1].grid(True, which='major')
    plt.show()


def visc_n2(T, P):
    """
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    mu = PropsSI('viscosity', 'T', T, 'P', P, 'Nitrogen')  # Pa-s
    return mu


def z_n2(T, P):
    """
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    # Density of Air at standard atmosphere in kg/m^3
    z = PropsSI('Z', 'T', T, 'P', P, 'Nitrogen')
    return z


def rho_n2(T, P):
    """
    returns density of nitrogen (kg/m3)
    T = nitrogen temperature in K
    P = absolute pressure in Pa
    """
    M = 28.01348  # molecular weight of nitrogen (kg/kg-mole)
    R = 8314.0  # gas const (m3-Pa)/(K-kmol)
    z = z_n2(T, P)
    rho = M*P/(z*R*T)  # kg/m3
#     PropsSI('D', 'T', 298.15, 'P', 101325, 'Nitrogen')
    return rho


def runavg(input_vect, window_size):
    """
        Calculates the running mean of a vector

        input:
        input_vect = the input vector
        window_size = number of values to be included in calculate the average

        output:
        out = vector of average values
            - dimension of output vector will be shorter by 'window_size'
            - 'window_size'/2 values will be trimmed from start and end
                of input_vector
            - out dimenions: out = input_vect[1+trim:-trim],
                where trim = window_size/2
    """

    window = np.ones(int(window_size)) / float(window_size)
    avg = np.convolve(input_vect, window, 'same')
    trim = int(window_size/2)
    out = avg[trim:-trim]
    return out
