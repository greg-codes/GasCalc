# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 13:42:57 2020

@author: smith
"""

#%% load things
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# use fancy LaTeX plotting styles
plt.style.use(['science-fixed', 'high-vis'])
plt.rcParams['figure.dpi'] = 240  # fix high-dpi display scaling issues
plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')

#  get directory of this script file
here = os.path.dirname(os.path.abspath(__file__))
CXRO_CrossSection_dir = os.path.join(here, 'CrossSections')

#  add CrossSections directory to PATH
sys.path.insert(0, os.path.normpath(CXRO_CrossSection_dir))

# load my modules
import GasCalculations as GC

# save figures to disk?
save_fig = False
diss_dir = r'C:\dissertation\figures'


#%% Figure 3.2: free expansion jet plots: M, rho, P vs x

# initialize GC class
test = GC.GasCalculations(d=200e-6, j=1, gas='Ar')


myP0 = 63  # backing pressure [Torr]
myPb = 1e-3  # chamber pressure [Torr]
xplot = np.linspace(0, 5e-3, num=1000)  # x-points for plotting
Mplot = np.vectorize(test.M)(xplot)
rhoplot = test.rho(Mplot)
pplot = test.pressure(Mplot)

dtext = f'd={1e6*test.d:3.0f}'+r' $\mu$m'
gtext = r'$\gamma$=' + f'{test.gamma:1.2f}'
P0text = r'$P_0$ = ' + f'{myP0:3.0f} Torr'
Pbtext = r'$P_b$ = ' + f'{myPb:3.0e} Torr'

fig, ax = plt.subplots()
ax.semilogy(1e3*xplot, Mplot, label='M')
ax.semilogy(1e3*xplot, rhoplot, label=r'$\rho/\rho_0$')
ax.semilogy(1e3*xplot, pplot, label=r'$P/P_0$')
ax.legend(frameon=True, loc=(0.65, 0.5))
ax.set_xlabel('off-axis distance, x [mm]')
ax.set_xlim([1e3*np.min(xplot), 1e3*np.max(xplot)])
ax.set_title('Supersonic Expansion')
ax.text(1, 1.5e-2, dtext+'\n'+gtext+'\n'+P0text+'\n'+Pbtext)
if save_fig:
    plt.savefig('M_rho_P_vs_x.pdf', dpi=300)

#%% Figure 3.3: free expansion jet plot: M, rho, P (for fixed x) vs pressure

# initialize GC class
test = GC.GasCalculations(d=200e-6, j=1, gas='Ar')

P0plot = np.linspace(1, 900, num=1000)  # backing pressure [Torr]

fig, ax = plt.subplots(1, 2, figsize=(2*3.5, 2.625), sharey=True)

# left plot: interaction pressure vs backing pressure at different x/d values
for myx in [100e-6, 150e-6, 200e-6, 300e-6]:  # on-axis distance [m]
    myM = test.M(myx)
    myP = test.rho(myM)
    myrho = test.pressure(myM)
    ax[0].plot(P0plot, P0plot*myP, label=rf'x={myx*1e6:3.0f} $\mu$m')
ax[0].axvline(x=250, ls='dashed', c='k')  # 5mTorr limit for He
ax[0].text(x=150, y=150, s='He')
ax[0].axvline(x=804, ls='dashed', c='k')  # 5 mTorr limit for Ar
ax[0].text(x=725, y=375, s='Ar')
ax[0].legend(frameon=True, framealpha=1.0)
ax[0].set_xlabel(r'Backing Pressure, $P_0$ [Torr]')
ax[0].set_ylabel(r'Interaction Pressure, $P_{int}$ [Torr]')
ax[0].set_xlim(0, P0plot.max())
ax[0].set_ylim(0)
ax[0].grid(axis='both', which='major')

# right plot: interaction pressure vs L_abs for He, Ar at 30 eV, 100 eV
Pint = np.linspace(1, P0plot.max(), num=1000)  # range of interaction pressures [Torr]
P0 = 760  # Torr
rho0 = 2.5049e+25  # [number/m3] at 760 Torr
# calculate corresponding range of interaction densities
rhoint = rho0 * (Pint/P0)**(1/(test.gamma))  # number density [number/m3] at Pint

# calculate the absorption lengths for He, Ar at 30, 100 eV at these densities
os.chdir(CXRO_CrossSection_dir)
LabsHe30 = 1e3*np.vectorize(test.calc_Labs)('He', ener=30, rho=rhoint)
LabsHe100 = 1e3*np.vectorize(test.calc_Labs)('He', ener=100, rho=rhoint)
LabsAr30 = 1e3*np.vectorize(test.calc_Labs)('Ar', ener=30, rho=rhoint)
LabsAr100 = 1e3*np.vectorize(test.calc_Labs)('Ar', ener=100, rho=rhoint)
os.chdir(here)
ax[1].semilogx(LabsHe30, Pint, label='He (30 eV)')
ax[1].semilogx(LabsHe100, Pint, label='He (100 eV)')
ax[1].semilogx(LabsAr30, Pint, label='Ar (30 eV)')
ax[1].semilogx(LabsAr100, Pint, label='Ar (100 eV)')
ax[1].axvline(x=1e3*test.d/3, ls='dashed', c='k', label=r'$L_{med}/3$')  # maximum absorption length for phase matching, d/3
ax[1].grid(axis='x', which='both')
ax[1].grid(axis='y', which='major')
ax[1].set_xlabel(r'Absorption Length, $L_{abs}$ [mm]')
ax[1].legend(frameon=True)

fig.suptitle(rf'Free Expansion Nozzle: On-Axis Pressure ($\gamma$ = {test.gamma:0.2f}, d = {test.d*1e6:3.0f} $\mu$m)')
plt.subplots_adjust(wspace=0.0)

if save_fig:
    plt.savefig(os.path.join(diss_dir, 'on-axis-pressure.pdf'), dpi=300)

#%% Figure 3.6: LPC on-axis plot
myP1 = 400  # backing pressure [Torr]
myP2 = 95  # interaction pressure [Torr]
myP3 = 4.35e-3  # chamber pressure [Torr]
LPCd = 400e-6  # laser hole diameter [m]
W = 2.032/1000  # full thickness of LPC interaction region [m]

test = GC.GasCalculations(d=LPCd, j=1, gas='Ar')

xplot = np.linspace(0, 2e-3, num=1000)
Mplot = np.vectorize(test.M)(xplot)
rhoplot = test.rho(Mplot)
pplot = test.pressure(Mplot)

dtext = f'd = {1e6*test.d:3.0f}'+r' $\mu$m'
gtext = f'{test.gas} (' + r'$\gamma$=' + f'{test.gamma:1.2f}' + ')'
P1text = r'$P_1$ = ' + f'{myP1:3.0f} Torr'
P2text = r'$P_2$ = ' + f'{myP2:3.0f} Torr'
P3text = r'$P_3$ = ' + f'{myP3:3.0e} Torr'

fig, ax = plt.subplots()
ax.semilogy(1e3*(xplot+W/2), rhoplot, label=r'$\rho/\rho_2$')
ax.hlines(1, xmin=-1e3*W/2, xmax=1e3*W/2, color='blue')
ax.semilogy(1e3*(-xplot-W/2), rhoplot, color='blue', ls='-')

ax.semilogy(1e3*(xplot+W/2), pplot, label=r'$P/P_2$')
ax.hlines(1, xmin=-1e3*W/2, xmax=1e3*W/2, color='red', ls='--')
ax.semilogy(1e3*(-xplot-W/2), pplot, color='red', ls='--')
ax.legend(frameon=True)
ax.set_xlabel('on-axis distance, x [mm]')
ax.set_xlim([-1e3*np.max(xplot+W/2), 1e3*np.max(xplot+W/2)])
ax.set_title('LPC: On-Axis Parameters')
ax.text(-1, 5e-3, dtext+'\n'+gtext+'\n'+P1text+'\n'+P2text+'\n'+P3text)
if save_fig:
    plt.savefig('LPC_on_axis.pdf', dpi=300)

#%% Figure 3.7: LPC pressures vs hole diameter, phase matching

#
# look at the effect of different laser holes
#

# calculate interaction and chamber pressures
p_Torr = np.linspace(1, 1000, num=1000)  # backing pressure [Torr]
fig, ax = plt.subplots(1, 2, figsize=(2*3.5, 2.625), sharex=True)

for myd in [200e-6, 300e-6, 400e-6, 500e-6]:
    test = GC.GasCalculations(d=myd, j=1, gas='Ar')
    int_Torr, cham_Torr = np.vectorize(test.Poiseuille_Torr_wrapper)(p_Torr)
    
    # find backing pressure where chamber pressure = 5 mTorr
    max_ind, _ = test.find_nearest(cham_Torr, 5e-3)
    ax[0].plot(p_Torr, 1e3*cham_Torr, label=fr'd={1e6*test.d:3.0f} $\mu$m')
    ax[1].plot(p_Torr, int_Torr, label=fr'd={1e6*test.d:3.0f} $\mu$m')
    ax[1].scatter(p_Torr[max_ind], int_Torr[max_ind],)

# plot it
ax[0].axhline(5, ls='dashed', c='k')
ax[0].set_xlabel(r'Backing Pressure, $P_1$ [Torr]')
ax[1].set_xlabel(r'Backing Pressure, $P_1$ [Torr]')
ax[0].set_ylabel(r'Chamber Pressure, $P_3$ [mTorr]')
ax[0].legend(frameon=True)
ax[1].set_ylabel(r'Interaction Pressure, $P_2$ [Torr]')
ax[1].legend(frameon=True)
ax[0].set_xlim(0, p_Torr.max())
ax[0].set_xlim(0, p_Torr.max())
ax[0].set_ylim(0)
ax[1].set_ylim(0, 1000)
ax[0].grid('on')
ax[1].grid('on')
fig.suptitle(f'LPC: Pressures vs. Laser Hole Diameter ({test.gas} gas)')
plt.subplots_adjust(wspace=0.25)
if save_fig:
    plt.savefig(os.path.join(diss_dir, 'LPC_pressure_vs_diameter.pdf'), dpi=300)

#%% Figure 3.8: LPC phase matching

myP1 = 400  # backing pressure [Torr]
myP2 = 95  # interaction pressure [Torr]
myP3 = 4.35e-3  # chamber pressure [Torr]
LPCd = 400e-6  # laser hole diameter [m]
W = 2.032/1000  # full thickness of LPC interaction region [m]

fig, ax = plt.subplots(1, 2, figsize=(2*3.5, 2.625), sharex='col', sharey='row')

# bottom-left: interaction pressure vs backing pressure
p_Torr = np.linspace(1, 1000, num=1000)  # backing pressure [Torr]

# 400 micron case
myd = 400e-6  # LPC laser hole diameter [m]
test_He = GC.GasCalculations(d=myd, j=1, gas='He')
He_int_Torr, He_cham_Torr = np.vectorize(test_He.Poiseuille_Torr_wrapper)(p_Torr)
He_max_ind, _ = test_He.find_nearest(He_cham_Torr, 5e-3)
test_Ar = GC.GasCalculations(d=myd, j=1, gas='Ar')
Ar_int_Torr, Ar_cham_Torr = np.vectorize(test_Ar.Poiseuille_Torr_wrapper)(p_Torr)
Ar_max_ind, _ = test_Ar.find_nearest(Ar_cham_Torr, 5e-3)

ax[0].plot(p_Torr, He_int_Torr, label='He', ls='solid', c='b')
ax[0].scatter(p_Torr[He_max_ind], He_int_Torr[He_max_ind], c='b')
ax[0].plot(p_Torr, Ar_int_Torr, label='Ar', ls='solid', c='r')
ax[0].scatter(p_Torr[Ar_max_ind], Ar_int_Torr[Ar_max_ind], c='r')

# 200 micron case
myd = 200e-6  # LPC laser hole diameter [m]
test_He = GC.GasCalculations(d=myd, j=1, gas='He')
He_int_Torr, He_cham_Torr = np.vectorize(test_He.Poiseuille_Torr_wrapper)(p_Torr)
He_max_ind, _ = test_He.find_nearest(He_cham_Torr, 5e-3)
test_Ar = GC.GasCalculations(d=myd, j=1, gas='Ar')
Ar_int_Torr, Ar_cham_Torr = np.vectorize(test_Ar.Poiseuille_Torr_wrapper)(p_Torr)
Ar_max_ind, _ = test_Ar.find_nearest(Ar_cham_Torr, 5e-3)

ax[0].plot(p_Torr, He_int_Torr, ls='solid', c='b', alpha=0.3)
ax[0].scatter(p_Torr[He_max_ind], He_int_Torr[He_max_ind], c='b', alpha=0.3)
ax[0].plot(p_Torr, Ar_int_Torr, ls='solid', c='r', alpha=0.3)
ax[0].scatter(p_Torr[Ar_max_ind], Ar_int_Torr[Ar_max_ind], c='r', alpha=0.3)

ax[0].set_xlabel(r'Backing Pressure, $P_1$ [Torr]')
ax[0].set_ylabel(r'Interaction Pressure, $P_2$ [Torr]')
ax[0].grid()
ax[0].legend(frameon=True)
ax[0].set_xlim(0, p_Torr.max())


# bottom-right plot: interaction pressure vs L_abs for He, Ar at 30 eV, 100 eV
P0 = 760  # Torr
rho0 = 2.446e+25  # [number/m3] at 760 Torr
kBT0 = P0/rho0 # unit: Torr / m3
rhoint = p_Torr / kBT0
W = 2.032/1000  # full thickness of LPC interaction region [m]

# calculate the absorption lengths for He, Ar at 30, 100 eV at these densities
os.chdir(CXRO_CrossSection_dir)
LabsHe30 = 1e3*np.vectorize(test_He.calc_Labs)('He', ener=30, rho=rhoint)
LabsHe100 = 1e3*np.vectorize(test_He.calc_Labs)('He', ener=100, rho=rhoint)
LabsAr30 = 1e3*np.vectorize(test_Ar.calc_Labs)('Ar', ener=30, rho=rhoint)
LabsAr100 = 1e3*np.vectorize(test_Ar.calc_Labs)('Ar', ener=100, rho=rhoint)
os.chdir(here)

ax[1].semilogx(LabsHe30, p_Torr, label='He (30 eV)', c='b', ls='solid')
ax[1].semilogx(LabsHe100, p_Torr, label='He (100 eV)', c='b', ls='dotted')
ax[1].semilogx(LabsAr30, p_Torr, label='Ar (30 eV)', c='r', ls='solid')
ax[1].semilogx(LabsAr100, p_Torr, label='Ar (100 eV)', c='r', ls='dotted')
ax[1].axvline(x=1e3*W/3, ls='dashed', c='k', label=r'$L_{med}/3$')  # maximum absorption length for phase matching, d/3
ax[1].set_ylim(0, 1000)
ax[1].set_xlim(1e-2, 1e3)
ax[1].grid(axis='x', which='both')
ax[1].grid(axis='y', which='major')
ax[1].set_xlabel(r'Absorption Length, $L_{abs}$ [mm]')
ax[1].legend(frameon=True)
fig.suptitle(fr'Low Pressure Cell (W={1e3*W:1.3f} mm)')
plt.subplots_adjust(hspace=0.10, wspace=0.15)
if save_fig:
    plt.savefig(os.path.join(diss_dir, 'LPC_IntPress_AbsLen.pdf'), dpi=300)

#%% Figure 3.17: HPC: on-axis parameters

myP1 = 760  # backing pressure [Torr]
myP2 = 9.21e-4 * myP1  # 5  # interaction pressure [Torr]
myP3 = 3.34e-7 * myP1  # 3e-4  # chamber pressure [Torr]
LPCd = 400e-6  # laser hole diameter [m]
W = 0.6415e-2  # full thickness of HPC interaction region [m]
PH_t = r'$P_H$'
PM_t = r'$P_M$'
PL_t = r'$P_L$'

test = GC.GasCalculations(d=myd, j=1, gas='Ar')

xplot = np.linspace(-15e-3, 15e-3, num=50000)  # meters
mypress = np.vectorize(test.HPC_onaxis_press)(xplot, myP1, myP2, myP3)
myrho = np.vectorize(test.HPC_onaxis_rho)(xplot, myP1, myP2, myP3)

# plot both pressure and density
fig, ax = plt.subplots()
ax.semilogy(1e3*xplot, mypress, c='b')
ax.set_ylabel('Pressure [Torr]', c='b')
ax.tick_params(axis='y', labelcolor='b')

ax2 = ax.twinx()
ax2.semilogy(1e3*xplot, myrho, c='r')
ax2.set_ylabel(r'Density [m$^{-3}$]', c='r')
ax2.tick_params(axis='y', labelcolor='r')
ax.set_xlim(1e3*np.min(xplot), 1e3*np.max(xplot))
#ax.grid()
ax.text(x=3.5, y=150, s=f'{PH_t} = {myP1} Torr')
ax.text(x=3.5, y=5, s=f'{PM_t} = {myP2:1.1f} Torr')
ax.text(x=0, y=5e-5, s=f'{PL_t} = {myP3:1.0e} Torr')
ax.set_title('HPC: On-Axis Parameters')
ax.set_xlabel('On-Axis position [mm]')
ax.set_ylabel('Pressure [Torr]')
if save_fig:
    plt.savefig(os.path.join(diss_dir, 'HPC_on-axis-pressure.pdf'), dpi=300)

#%% Figure 3.18: HPC: XUV reabsorption in PM region

myPH = 760  # Torr
test_Ar = GC.GasCalculations(d=myd, j=1, gas='Ar')
test_He = GC.GasCalculations(d=myd, j=1, gas='He')

PM_Ar, TM_Ar = test_Ar.calc_XUV_abs_PM(PH=myPH)
PM_He, TM_He = test_He.calc_XUV_abs_PM(PH=myPH)

fig, ax = plt.subplots()
ax.plot(TM_Ar['E'], TM_Ar['T'], label='Ar')
ax.plot(TM_He['E'], TM_He['T'], label='He')
ax.set_xlim([0, 300])
ax.set_xlabel('Energy [eV]')
ax.set_ylabel('Transmission')
ax.set_title(r'HPC: XUV reabsorption in $M$ region')
ax.text(0.3, 0.60, fr'$P_H$ = {myPH} Torr', transform=ax.transAxes)
ax.text(0.3, 0.50, fr'$P_M$ (Ar) = {PM_Ar:1.1f} Torr', transform=ax.transAxes)
ax.text(0.3, 0.40, fr'$P_M$ (He) = {PM_He:1.1f} Torr', transform=ax.transAxes)
ax.legend(frameon=True)
if save_fig:
    plt.savefig(os.path.join(diss_dir, 'HPC_absorption.pdf'), dpi=400)

#%% Figure 3.19: HPC: mega-plot with P_H, P_M, L_abs, and P_M transmission

test_Ar = GC.GasCalculations(d=myd, j=1, gas='Ar')
test_He = GC.GasCalculations(d=myd, j=1, gas='He')



# t_Labs_PH, t_PM, t_xM_PH, t_L_PM, t_T_PM = HPC_PH_PM_Labs_xM(760, gas='Ar', energy=30)

# generate vectors for each of the following:
# PH: interaction pressures [Torr]
# L_abs_PH: absorption length for PH in <gas>
# xM_PH: mach disk location due to PH plume
# PM: absorption region pressure [Torr]
# L_PM: effective length of PM region [meter]
# T_PM: XUV transmission of <energy> in PM region

PH = np.linspace(1, 15000, num=1000)
Labs_PH_Ar_30, PM_Ar, rho_dx_Ar, T_PM_Ar_30 = np.vectorize(test_Ar.HPC_PH_PM_Labs_xM)(PH, energy=30)
Labs_PH_Ar_100, _, _, T_PM_Ar_100 = np.vectorize(test_Ar.HPC_PH_PM_Labs_xM)(PH, energy=100)
Labs_PH_He_30, PM_He, rho_dx_He, T_PM_He_30 = np.vectorize(test_He.HPC_PH_PM_Labs_xM)(PH, energy=30)
Labs_PH_He_100, _, _, T_PM_He_100 = np.vectorize(test_He.HPC_PH_PM_Labs_xM)(PH, energy=100)

Ar_max_ind, _ = test_Ar.find_nearest(PH, 14200)
He_max_ind, _ = test_He.find_nearest(PH, 5400)

# plot it
r1_out = test_He.r1_out  # outer radius of high pressure pipe [m]
fig, ax = plt.subplots(2, 1, figsize=(1.2*3.5, 2*2.625), sharex=True)
# color code: Argon = blue, Helium = red, 30 eV = dashed, 100 eV = dotted

# transmission of PM region
ax[0].semilogy(PH, T_PM_He_100, c='r', ls='dotted')
ax[0].semilogy(PH, T_PM_Ar_100, c='b', ls='dotted')
ax[0].semilogy(PH, T_PM_He_30, label='He', c='r', ls='dashed')
ax[0].semilogy(PH, T_PM_Ar_30, label='Ar', c='b', ls='dashed')
ax[0].axvline(PH[Ar_max_ind], ls='dashdot', c='b')
ax[0].axvline(PH[He_max_ind], ls='dashdot', c='r')
ax[0].set_ylabel(r'Transmission in ${M}$ region')
ax[0].grid(which='both', axis='y')
ax[0].set_ylim(1e-4, 2e0)
ax[0].legend(frameon=True)

# XUV absorption length in PH region
ax[1].semilogy(PH, Labs_PH_He_100, c='r', ls='dotted')
ax[1].semilogy(PH, Labs_PH_Ar_100, c='b', ls='dotted', label='100 eV')
ax[1].semilogy(PH, Labs_PH_He_30, c='r', ls='dashed')
ax[1].semilogy(PH, Labs_PH_Ar_30, c='b', ls='dashed', label='30 eV')
ax[1].axhline(2*r1_out*1e3 / 3, label=r'$L_{med}$/3', ls='dashed', c='k')
ax[1].axvline(PH[Ar_max_ind], ls='dashdot', c='b')
ax[1].axvline(PH[He_max_ind], ls='dashdot', c='r')
ax[1].legend(frameon=True, framealpha=1, ncol=3)
ax[1].grid(which='both', axis='y')
ax[1].set_ylabel(r'$L_{\textrm{abs, H}}$ [mm]')
ax[1].set_xlabel(r'Interaction Pressure, $P_H$ [Torr]')
ax[1].set_ylim(1e-2, 1e1)
ax[1].set_xlim(0, PH.max())
plt.subplots_adjust(hspace=0.1, top=0.95)
ax[0].set_title('HPC: Absorption in $H$ and $M$ regions')

if save_fig:
    plt.savefig(os.path.join(diss_dir, 'HPC-gas-flow-int-length.pdf'), dpi=300)