# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:50:45 2020

@author: smith
"""

#%% load packages
import numpy as np
import pandas as pd
import os
from scipy import integrate

#  get directory of this script file
here = os.path.dirname(os.path.abspath(__file__))
CXRO_CrossSection_dir = os.path.join(here, 'CrossSections')
#%% class definition
class GasCalculations:
    def __init__(self, d=200e-6, j=1, gas='Ar'):
        """
        

        Parameters
        ----------
        d : float, optional
            diameter of gas nozzle in meters. The default is 200e-6.
        j : int, optional
            j=1 for 3D, j=2 for 2D. The default is 1 (3D).
        gas : string, optional
            Ar, Ne, He, N2, none, CO2. The default is 'Ar'.

        Returns
        -------
        None.

        """
        # save input to self
        self.d =  float(d)
        self.gas = gas
        self.j = int(j)
        
        # hard-coded dimensions for LPC
        self.L = 5e-3  # capillary length [m]
        self.R = (0.5*203)*1e-6  # capillary radius [m]

        # hard-coded dimensions for HPC
        self.r1_out = 0.5 * 3.175e-3  # outer radius of high pressure pipe [m]
        self.r2_out = 0.5 * 19.05e-3  # outer radius of outer shroud [m]
        self.d1 = 100e-6  # diameter of laser drilled hole on high pressure pipe [m]
        self.d2 = 600e-6  # diameter of machined hole on outer shroud [m]

        # load gas parameters from tables
        self.load_gas_params()


    def load_gas_params(self):
        """
        Load gas parameters from Scoles tables.

        Saves C1, C2, C3, C4, A, B, phi, x0, xmin to self.

        """
        # load input from self
        j = self.j # j=1 for 3D, j=2 for 2D
        d = self.d  # nozzle diameter [m]
        gas = self.gas

        # gas parameters from Scoles, Table 2.2
        if gas in ['Ar', 'Ne', 'He']:  # monatomic gas
            gamma = 5/3
        elif gas == 'N2':  # diatomic
            gamma = 7/5
        elif gas == 'none':  # triatomic linear
            gamma = 9/7
        elif gas == 'CO2':  # triatomic bent
            gamma = 4/3
        else:
            print('error! unrecognized gas type')
        
        if gamma == 5/3 and j == 1:
            C1 = 3.232
            C2 = -0.7563
            C3 = 0.3937
            C4 = -0.0729
            A = 3.337
            B = -1.541
            phi = 1.365  # scoles table 2.1
            x0 = 0.075*d  # scoles table 2.1
            xmin = 2.5*d  # scoles table 2.1
        elif gamma == 7/5 and j == 1:
            C1 = 3.606
            C2 = -1.742
            C3 = 0.9226
            C4 = -0.2069
            A = 3.190
            B = -1.610
            phi = 1.662  # scoles table 2.1
            x0 = 0.40*d  # scoles table 2.1
            xmin = 6*d  # scoles table 2.1
        elif gamma == 9/7 and j == 1:
            C1 = 3.971
            C2 = -2.327
            C3 = 1.326
            C4 = -0.311
            A = 3.609
            B = -1.950
            phi = 1.888  # scoles table 2.1
            x0 = 0.85*d  # scoles table 2.1
            xmin = 4*d  # scoles table 2.1
        else:
            print('error: unrecognized gamma and j!')
            
        # save parameters to self
        self.gamma = gamma
        self.j = j
        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.C4 = C4
        self.A = A
        self.B = B
        self.phi = phi
        self.x0 = x0
        self.xmin = xmin
        return 


    def M(self, x, d=-1):
        """
        Calculate the Mach number from Scoles Table 2.2.

        Parameters
        ----------
        x : float
            on-axis position [m].
        d : float
            nozzle diameter [m], optional. set to -1 to use self.d value.

        Returns
        -------
        float
            M, mach number.

        """
        # load parameters from self
        if d == -1:
            d = self.d
        A = self.A
        B = self.B
        gamma = self.gamma
        C1 = self.C1
        C2 = self.C2
        C3 = self.C3
        C4 = self.C4
        j = self.j

        # piecewise definition of M
        def M1(x, d):  # for 0 < x < d only
            fact1 = 1.0 + A*(x/d)**2 + B*(x/d)**3
            fact2 = 1 - np.heaviside((x/d)-1, 0.5) * np.heaviside(x/d, 0.5)
            ans = fact1 * fact2
            return ans

        def M2(x, d):  # for x > 0.5*d
            fact1 = (x/d)**((gamma-1)/j)
            fact2 = C1 + C2/(x/d) + C3/(x/d)**2 + C4/(x/d)**3
            #  fact3 = np.heaviside((x/d)-0.5, 0.5)
            ans = fact1 * fact2  # * fact3
            return ans

        # calculate M
        if x < d:
            return M1(x, d)
        else:
            return M2(x, d)
        
    def rho(self, M):
        """
        Calculate the normalized centerline density as a function of Mach 
        number.
    
        Uses Scoles eqn. 2.6.
    
        Parameters
        ----------
        M : float
            Mach number.
    
        Returns
        -------
        rho/rho0 : float
            normalized density (rho0 = backing density).
    
        """
        gamma = self.gamma
        return (1 + 0.5 * M**2 * (gamma-1))**(-1 / (gamma-1))
        

    def pressure(self, M):
        """
        Calculate the normalized centerline pressure as a function of Mach 
        number.
    
        Uses Scoles eqn. 2.5.
    
        Parameters
        ----------
        M : float
            Mach number.
    
        Returns
        -------
        P/P0 : float
            normalized pressure (P0 = backing pressure).
    
        """
        gamma = self.gamma
        return (1.0 + 0.5 * M**2 * (gamma-1))**(-1.0*gamma / (gamma-1))
            
    def calc_Labs(self, gas='He', ener=100, p_Torr=100, Lmed=2.032e-3,
                  rho=2.5049e+25, Verbose=False):
        """
        Calculate the absorption length [m].
    
        Parameters
        ----------
        gas : string, optional
            name of gas. loads from CXRO files.
        ener : float, optional
            photon energy [eV].
        p_Torr : float, optional
            pressure in Torr.
        Lmed : float, optional
            length of medium [m]. The default is 2.032e-3.
        rho : float, optional
            number density [number/m^3]. The default is 2.5049+25.
        Verbose: boolean, optional
            Print the results to screen?
    
        Returns
        -------
        Labs : float
            absorption length [m].
    
        """
        if gas == 'He':
            data = self.load_CXRO_ASF('he.nff.txt', P_Torr=p_Torr, L_nm=Lmed)
        elif gas == 'Ar':
            data = self.load_CXRO_ASF('ar.nff.txt', P_Torr=p_Torr, L_nm=Lmed)
        else:
            print('error! unrecognized gas!')
        # sigma is cross section @ ener [nm^-2]
        sigma = data.mu_a.iloc[np.argmin(np.abs(data.E.values-ener))]
        Labs = 1 / (1e-18*sigma*rho)  # absorption length [m]
        if Verbose:
            print(f'{gas} @ {ener} eV, P0={p_Torr} Torr, rho={rho:1.3e} m^-3:')
            print(f'   Labs = {1e3*Labs:1.3f} mm')
            print(f'   Lmed = {1e3*Lmed:1.3f} mm')
            print(f'   Lmed/Labs = {Lmed/Labs:1.3f}')
        return Labs

    def Poiseuille_Torr_wrapper(self, p1_Torr):
        """
        Wrapper function for Poiseuille_both().
    
        Parameters
        ----------
        p1_Torr : float
            LPC backing pressure [Torr].
        d : float, optional
            LPC laser hole diameter [m]. The default is 200e-6.
        gas : string, optional
            gas type. The default is 'Ar'.
    
        Returns
        -------
        int_Torr : float
            interaction pressure [Torr].
        cham_Torr : float
            chamber pressure [Torr].
        """
        p1_Pa = self.Torr_to_Pa(p1_Torr)  # convert Torr to Pa
        int_Pa, cham_Pa = self.Poiseuille(p1_Pa)
        int_Torr = self.Pa_to_Torr(int_Pa)  # convert to Torr
        cham_Torr = self.Pa_to_Torr(cham_Pa)
        return int_Torr, cham_Torr

    def Poiseuille(self, p1):
        """
        Calculate interaction & chamber pressures for the LPC.
    
        Parameters
        ----------
        p1 : float
            backing pressure of LPC [Pa].
        d : float
            diameter of LPC's laser holes [m].
        gas : string, optional
            DESCRIPTION. The default is 'Ar'.
    
        Returns
        -------
        p2 : float
            interaction pressure of LPC [Pa].
        p3 : float
            chamber pressure [Pa].
    
        """
        # load parameters from self
        d = self.d  # diameter of laser holes [m]
        gas = self.gas
        L = self.L  # capillary length [m]
        R = self.R  # capillary radius [m]
        
        if gas == 'He':
            c = 450  # Scoles table 2.5 [m/s]
            mu = 1.96e-5  # dynamic viscosity [m/s] at 20C, 1atm
            St = 1.2  # pumping speed [m^3/s]
        elif gas == 'Ar':
            c = 140  # Scoles table 2.5 [m/s]
            mu = 2.23e-5  # dynamic visosity [m/s] at 20C, 1atm
            St = 1.0  # pumping speed [m^3/s]
        elif gas == 'N2':
            c = 160  # Scoles table 2.5 [m/s]
            mu = 1.76e-5  # dynamic visosity [m/s] at 20C, 1atm
            St = 1.1  # pumping speed [m^3/s]
        else:
            print('error! unrecognized gas!')

        # p2 = interaction pressure
        fact1 = p1**2 * np.pi**2 * R**8
        fact2 = 256 * c**2 * d**4 * L**2 * mu**2
        num = -16 * c * d**2 * L * mu + np.sqrt(fact1 + fact2)
        denom = np.pi * R**4
        p2 = num / denom
    
        # p3 = chamber pressure
        fact1 = 2 * c * d**2 / St
        fact2 = -16 * c * d**2 * L * mu / (np.pi * R**4)
        fact3 = p1**2
        fact4 = 256 * c**2 * d**4 * L**2 * mu**2 / (np.pi**2 * R**8)
        p3 = fact1 * (fact2 + np.sqrt(fact3 + fact4))
        return p2, p3

    def Torr_to_Pa(self, P_Torr):
        """
        Convert from Torr to Pascal.
    
        Parameters
        ----------
        P_Torr : float
            Pressure in Torr.
    
        Returns
        -------
        float
            Pressure in Pascal.
        """
        return P_Torr*133.32236842105
    
    def Pa_to_Torr(self, P_Pa):
        """
        Convert from Pascal to Torr.
    
        Parameters
        ----------
        P_Pa : float
            Pressure in Pascal.
    
        Returns
        -------
        float
            Pressure in Torr.
        """
        return P_Pa/133.32236842105
    
    def find_nearest(self, array, value):
        '''find nearest index and value of array'''
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx, array[idx]

    def xM(self, d, P0, Pb):
        """
        Calculate Mach Disk Location.
    
        Parameters
        ----------
        d : float
            aperture diameter.
        P0 : float
            backing pressure.
        Pb : float
            chamber pressure.
    
        Returns
        -------
        float
            Mach Disk Location. [same units as d]
    
        """
        return 0.67 * d * np.sqrt(P0/Pb)

    def HPC_onaxis_press(self, x, PH, PM, PL):
        """
        Calculate the on-axis gas pressure of the HPC.
    
        Parameters
        ----------
        x : float
            On-axis distance from the center of the HPC. x=0 corresponds to the
            center of the gas pipe.
        PH : float
            pressure inside the high pressure region [Torr].
        PM : float
            pressure inside the shroud [Torr].
        PL : float
            chamber pressure [Torr].
    
        Returns
        -------
        ans : float
            pressure at x [Torr].
        """
        # load HPC dimensions from self
        r1_out = self.r1_out  # outer radius of high pressure pipe [m]
        r2_out = self.r2_out  # outer radius of outer shroud [m]
        d1 = self.d1  # diameter of laser drilled hole on high pressure pipe [m]
        d2 = self.d2  # diameter of machined hole on outer shroud [m]
        
        # calculate piecewise pressure
        if np.abs(x) <= r1_out:
            # inside the inner pipe
            ans = PH
        elif (np.abs(x) > r1_out) and (np.abs(x) <= r2_out):
            # between the inner pipe & the outer shroud
            MD = self.xM(d=d1, P0=PH, Pb=PM)  # Mach disk location [m]
            if np.abs(x) < r1_out + MD:
                # inside the Mach disk
                # print('inside the Mach disk')
                ans = PH * self.pressure(self.M(np.abs(x)-r1_out, d1))
            else:
                # outside the Mach disk
                ans = PM
        else:
            # outside the outer shroud
            MD = self.xM(d=d2, P0=PM, Pb=PL)  # Mach disk location [m]
            if np.abs(x) < r2_out + MD:
                # inside the Mach disk
                ans = PM * self.pressure(self.M(np.abs(x)-r2_out, d2))
            else:
                # outside the Mach disk
                ans = PL
        return ans
    
    def HPC_onaxis_rho(self, x, PH, PM, PL):
        """
        Calculate the on-axis gas density of the HPC.
    
        Parameters
        ----------
        x : float
            On-axis distance from the center of the HPC. x=0 corresponds to the
            center of the gas pipe.
        PH : float
            pressure inside the high pressure region [Torr].
        PM : float
            pressure inside the shroud [Torr].
        PL : float
            chamber pressure [Torr].
    
        Returns
        -------
        ans : float
            pressure at x [Torr].
        """
        # load HPC dimensions from self
        r1_out = self.r1_out  # outer radius of high pressure pipe [m]
        r2_out = self.r2_out  # outer radius of outer shroud [m]
        d1 = self.d1  # diameter of laser drilled hole on high pressure pipe [m]
        d2 = self.d2  # diameter of machined hole on outer shroud [m]
        
        P0 = 760  # Torr, standard pressure
        rho0 = 2.446e+25  # [number/m3] at 760 Torr
        kBT0 = P0/rho0  # unit: Torr / m3
        # rhoint = PH / kBT0  # unit = 1/m3
        rhoH = PH / kBT0  # number density from ideal gas law in PH [1/m3]
        rhoM = PM / kBT0  # number density from ideal gas law in PM [1/m3]
        rhoL = PL / kBT0 # number density from ideal gas law in PL [1/m3]
        
        # calculate piecewise density
        if np.abs(x) <= r1_out:
            # inside the inner pipe
            ans = rhoH
        elif (np.abs(x) > r1_out) and (np.abs(x) <= r2_out):
            # between the inner pipe & the outer shroud
            MD = self.xM(d=d1, P0=PH, Pb=PM)  # Mach disk location [m]
            if np.abs(x) < r1_out + MD:
                # inside the Mach disk
                # print('inside the Mach disk')
                ans = rhoH * self.rho(self.M(np.abs(x)-r1_out, d1))
            else:
                # outside the Mach disk
                ans = rhoM
        else:
            # outside the outer shroud
            MD = self.xM(d=d2, P0=PM, Pb=PL)  # Mach disk location [m]
            if np.abs(x) < r2_out + MD:
                # inside the Mach disk
                ans = rhoM * self.rho(self.M(np.abs(x)-r2_out, d2))
            else:
                # outside the Mach disk
                ans = rhoL
        return ans
    
    def calc_XUV_abs_PM(self, PH=760):
        
        # define HPC parameters here
        gas = self.gas
        r1_out = self.r1_out  # outer radius of high pressure pipe [m]
        r2_out = self.r2_out  # outer radius of outer shroud [m]
        # d1 = self.d1  # diameter of laser drilled hole on high pressure pipe [m]
        # d2 = self.d2  # diameter of machined hole on outer shroud [m]
    
        # use fitting results to determine PM from PH
        if gas == 'Ar':
            PM_coef = 3.74029938e-04
            # offset = 4.70723320e+00
            PM_offset = 0 
            PL_coef = 3.34e-7
        elif gas == 'He':
            PM_coef = 9.21046250e-04
            # offset = 4.52541328e+00
            PM_offset = 0
            PL_coef = 3.33e-7
        else:
            print('error! unrecognized gas!')
            PM_coef = 1
            PM_offset = 0
            PL_coef = 1
        PM = PM_coef * PH + PM_offset
        PL = PL_coef * PH
    
        # integrate rho*dx over plume
        rho_dx, _ = integrate.quad(
            self.HPC_onaxis_rho, args=(PH, PM, PL), a=r1_out, b=r2_out
            )  # m^-2
        rho_dx = 1e-18*rho_dx  # convert to nm^-2
        
        # calculate transmission from CXRO data
        starting_path = os.getcwd()
        os.chdir(CXRO_CrossSection_dir)
        hc = 1239.84193  # eV*nm
        re = 2.81794032e-6  # classical electron radius, nm
        fname = gas.lower() + '.nff.txt'
        os.chdir(CXRO_CrossSection_dir)
        df = pd.read_csv(fname, delimiter='\t', skiprows=1, usecols=[0, 1, 2],
                         names=['E', 'f1', 'f2'],
                         dtype={'E': np.float64, 'f1': np.float64, 'f2': np.float64}
                         )
        os.chdir(starting_path)
        df['WL'] = hc / df['E']  # wavelength
        df['mu_a'] = 2 * re * df['WL'] * df['f2']  # photoatomic cross section
        df['T'] = np.exp(-1.0*rho_dx*df['mu_a'])  # transmission
        return PM, df

    
    def HPC_PH_PM_Labs_xM(self, PH, energy=30):
        """
        function to calculate P_M from a given P_H using the fitting
        coefficients. then, calculate L_abs, P_M transmission and mach disk 
        locations.
    
        Parameters
        ----------
        PH : float
            backing / interaction pressure of HPC [Torr].
        gas : string, optional
            Name of gas[Ar, He]. The default is 'Ar'.
        energy : float, optional
            photon energy for L_abs [eV]. The default is 30.
    
        Returns
        -------
        None.
    
        """
        # enter HPC dimensions below
        gas = self.gas
        r1_out = self.r1_out  # outer radius of high pressure pipe [m]
        r2_out = self.r2_out  # outer radius of outer shroud [m]
        # d1 = self.d1  # diameter of laser drilled hole on high pressure pipe [m]
        # d2 = self.d2  # diameter of machined hole on outer shroud [m]
    
        # use fitting results to determine PM from PH
        if gas == 'Ar':
            coef = 3.74029938e-04
            # offset = 4.70723320e+00
            offset = 0
        elif gas == 'He':
            coef = 9.21046250e-04
            # offset = 4.52541328e+00
            offset = 0
        else:
            print('error! unrecognized gas!')
            coef = 1
            offset = 0
        PM = coef * PH + offset
    
        # calculate L_abs for this PH
    
        # calculate the absorption lengths [unit = mm] at PH density
        P0 = 760  # Torr
        rho0 = 2.446e+25  # [number/m3] at 760 Torr
        kBT0 = P0/rho0  # unit: Torr / m3
        rhoint = PH / kBT0  # unit = 1/m3
        starting_path = os.getcwd()
        os.chdir(CXRO_CrossSection_dir)
        Labs_PH = 1e3*self.calc_Labs(gas=gas, ener=energy, rho=rhoint)  # unit = mm
        os.chdir(starting_path)
    
        # calculate the density-length product in PM region
        rho_dx, _ = integrate.quad(
            self.HPC_onaxis_rho, args=(PH, PM, 3e-4), a=r1_out, b=r2_out
            )  # m^-2
        rho_dx = 1e-18*rho_dx  # convert to nm^-2
    
        # calculate transmission of PM
        hc = 1239.84193  # eV*nm
        re = 2.81794032e-6  # classical electron radius, nm
        fname = gas.lower() + '.nff.txt'
        os.chdir(CXRO_CrossSection_dir)
        df = pd.read_csv(fname, delimiter='\t', skiprows=1, usecols=[0, 1, 2],
                         names=['E', 'f1', 'f2'],
                         dtype={'E': np.float64, 'f1': np.float64, 'f2': np.float64}
                         )
        os.chdir(starting_path)
        df['WL'] = hc / df['E']  # wavelength
        df['mu_a'] = 2 * re * df['WL'] * df['f2']  # photoatomic cross section
        df['T'] = np.exp(-1.0*rho_dx*df['mu_a'])  # transmission

        # approximate the transmission T(E) by averaging adjacent points
        T_PM = df.iloc[(df['E']-energy).abs().argsort()[:2]]['T'].mean()
        return Labs_PH, PM, rho_dx, T_PM

    def load_CXRO_ASF(self, fname, P_Torr, L_nm):
        '''load atomic scattering file from CXRO
        columns:
            E: energy [eV]
            f1, f2: atomic scattering factors from f = f1 + i*f2
            WL: wavelength, from energy [nm]
            mu_a: photoatomic cross section [nm^2]
        '''    
        re = 2.81794032e-6  # classical electron radius [nm]
        hc = 1239.84193  # [eV*nm]
        old_dir = os.getcwd()
        os.chdir(CXRO_CrossSection_dir)
        df = pd.read_csv(fname, delimiter='\t', skiprows=1, usecols=[0,1,2], names=['E', 'f1', 'f2'], dtype={'E':np.float64, 'f1':np.float64, 'f2':np.float64})
        os.chdir(old_dir)
        df['WL'] = hc / df['E']  # wavelength [nm]
        df['mu_a'] = 2 * re * df['WL'] * df['f2']  # photoabsorption cross section
        df['N'] = (P_Torr/760) * 0.0269  # atomic number density [atoms/nm^3]
        df['n'] = 1 - (df['N']*re*df['WL']**2*(df['f1']+1j*df['f2']))/(2*np.pi)  # refractive index
        df['T'] = np.exp( - df['N']*df['mu_a']*L_nm )  # transmission
        df.name = fname.split('.')[0].capitalize()  # name the dataframe after the element
        return df
        