import numpy as np
import math
import os
import sys
import datetime
import pandas as pd
import numpy.ma as ma
from astropy.table import Table
import matplotlib as mpl
import matplotlib.pyplot as plt
import marvin
from marvin.tools.cube import Cube
import marvin
from multiprocessing.pool import Pool 
from functools import partial
from astropy import wcs
from marvin import config
import pickle 
from provabgs import infer as Infer
from provabgs import models as Models
from provabgs import flux_calib as FluxCalib
fsps_emulator = Models.NMF(burst=False, emulator=True)
from astropy.table import Table
import corner as DFM
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import scipy.optimize as op


def create_mask(lam, z, verbose=False, gal=True, sky=True):
    if (not gal) and (not sky):
        return None
    mask = [True for l in lam]   
    if gal:
        # Emission lines from SDSS table
        # http://classic.sdss.org/dr6/algorithms/linestable.html
        lines = {'ne_vi':[3426.85], 'o_ii':[3727.092, 3729.875], 'he_i':[3889.0], 's_ii':[4072.3, 6718.29, 6732.67],
                 'h_delta':[4102.89], 'h_gamma':[4341.68], 'o_iii':[4364.436, 4932.603, 4960.295, 5008.240],
                 'h_beta':[4862.68], 'o_i':[6302.046, 6365.536], 'n_i':[6529.03], 'n_ii':[6549.86, 6585.27],
                 'h_alpha':[6564.61], 'emi_l':{6410,9820.057746511147,9872, 8951, 9072, 9559, 9532, 3870, 8900, 9126, 9135, 9425, 9465, 9494, 9507, 9711, 9623}}   
        for key in lines.keys():
            for line in lines[key]:
                mask = np.logical_and(mask, np.logical_or(lam < line*(1-2e-3), lam > line*(1+2e-3)))
                if verbose:
                    print('Line: %s\tWavelength: %g\tWidth: %g' %(key, line, line*2e-3))
    if sky:
        # Masked regions have consistently large fit residuals
        # averaged over all runs
        bad = np.array([[5557,5597], [5660,5670]])/(1+z)
        for band in bad:
            mask = np.logical_and(mask, np.logical_or(lam < band[0], lam > band[1]))
            if verbose:
                print('Band masked: %d-%d' %(band[0], band[1]))
    return mask

#create a function that outputs the masked wave, flux and ivar
def masked_w_f_i (wv,fx,iv,z):
    wave=np.asarray(wv)
    rest_wave=wave/(1+z)
    masking=create_mask(rest_wave, z, verbose=False, gal=True, sky=True)
    masking_array=[]
    for i in masking:
        if i==False:
            masking_array.append(1)
        else: 
            masking_array.append(0)
    Wavelength11=np.asarray(wv)   
    flux_=np.asarray(fx)  
    ivar_=np.asarray(iv)
    masked_wave1= ma.masked_array(Wavelength11, mask=masking_array)
    masked_flux=ma.masked_array(flux_, mask=masking_array)
    masked_ivar=ma.masked_array(ivar_, mask=masking_array)    
    return masked_wave1, masked_flux, masked_ivar 

def check_empty (arr):
    non_empty=[]
    for jy in arr:
        if len(jy)!=0:
            non_empty.append(jy)
    return (non_empty)

def checkpoint(xx, yy, a, b):
    p = ((math.pow((xx), 2) / math.pow(a, 2)) + (math.pow((yy), 2) / math.pow(b, 2)))
    return p
def rotates(origin, point, angle):
    ox, oy = origin
    px, py = point
    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy
def rectangles(radius,minor, angle):
    lenss=np.arange(-radius-0.25,radius+0.25,0.5)
    wids=np.arange(-minor,minor+0.25,0.5)
    x=[];y=[];xte=[]; yte=[]; ct=0; outside=[]
    for w in wids:
        for k in lenss:
            x.append(k+0.25)
            y.append(w)
    for vas in x:
        vas=vas+0.25
        yvs=y[ct]+0.25
        if vas>-radius and vas<radius:
            if (checkpoint(vas, yvs, radius, minor) > 1 ):
                outside.append(vas)
            else:
                xte.append(vas)
                yte.append(yvs)
        ct+=1
    truex=[];truey=[];count=0
    for j in range(len(xte)):
        orr=(0,0)
        poin=(xte[count], yte[count])
        truex.append(rotates(orr, poin, math.radians(90+angle))[0])
        truey.append(rotates(orr, poin, math.radians(90+angle))[1])
        count+=1
        
    return truex,truey
####################################################################################################################
####################################################################################################################

os.environ['SAS_BASE_DIR'] = '/global/cfs/cdirs/sdss/data/sdss'
MY_cube=Cube(plateifu='10001-12701')
MaNGa_wave=[]
spaxel = MY_cube.getSpaxel(0.05,0.05)
ccf=spaxel.flux.wavelength
from astropy import units as u
for i in ccf:
    a111=i/ u.AA
    MaNGa_wave.append(float(a111))

#/global/cscratch1/sd/marta_r/2021-10-10/_Multiprocessing/September/All_fits_nersc.csv
MaNGa_wave=MaNGa_wave[: len(MaNGa_wave) - 229]

import random
properties=[]
for j in range(30):
    df = pd.read_csv("/global/cscratch1/sd/marta_r/2021-10-10/_Multiprocessing/September/All_fits_nersc.csv")
    cd=random.randint(0,df[df.columns[0]].count())
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        df.to_csv("/global/cscratch1/sd/marta_r/2021-10-10/_Multiprocessing/September/All_fits_nersc.csv")
    for p in range (df[df.columns[0]].count()):
        #df = pd.read_csv("/global/homes/m/marta_r/Fitting_files/test2.csv")
        if cd <= (df[df.columns[0]].count())-1 and df.loc[cd, 'Validation'] == 0:
            df.loc[cd,'Validation'] ='1'
            df.to_csv("/global/cscratch1/sd/marta_r/2021-10-10/_Multiprocessing/September/All_fits_nersc.csv", index=True)
            z,ifu,ratio=df.loc[cd, 'redhsift'], df.loc[cd, 'Spectra'],df.loc[cd, 'ratio']
            rad,ba,phi=df.loc[cd,'radius'], df.loc[cd,'nsa_elpetro_ba'], df.loc[cd,'nsa_elpetro_phi']
            inner=[z,ifu,ratio,rad, ba, phi]
            break
        elif cd > (df[df.columns[0]].count()-1):
            cd-=1
        else: 
            cd+=1
    properties.append(inner)

os.environ['SAS_BASE_DIR'] = '/global/cfs/cdirs/sdss/data/sdss'
radius=properties[0][3]
ifu_pl=properties[0][1]
red=properties[0][0]
ratios=properties[0][4]; angles=properties[0][5]
err=[]
cube = Cube(plateifu=ifu_pl)
redsh=properties[0][0]
rads=radius; angle=angles; b=ratios*radius
rx=rectangles(rads,b,angle)[0]
ry=rectangles(rads,b,angle)[1]
count=0
flux_val=[]
ivar_val=[]
true_flux=[]
total_list=[] 
total_iv_list=[]
pcount=0
pcountiv=0
for j in range(len(rx)):
    spaxel = cube.getSpaxel(rx[count],ry[count])
    a=spaxel.flux.value
    a=a[: len(a) - 229]
    flux_val.append(a)
    b=spaxel.flux.ivar 
    b=b[: len(b) - 229] 
    ivar_val.append(b)
    count=count+1
for p in range(len(flux_val[0])):
    sar=[]
    for pp in flux_val:
        store=pp[pcount]
        sar.append(store)
    total_flux=sum(sar)
    total_list.append(total_flux)
    pcount=pcount+1
for pi in range(len(ivar_val[0])):
    sar_ivar=[]
    for ppi in ivar_val:
        store_iv=ppi[pcountiv]
        sar_ivar.append(store_iv)
    total_ivar=sum(sar_ivar)
    total_iv_list.append(total_ivar)
    pcountiv=pcountiv+1
flux=np.asarray(total_list)
ivar=np.asarray(total_iv_list)
Wavelength=np.asarray(MaNGa_wave)
zred=redsh
        #use_ivar=masked_w_f_i(Wavelength,flux,ivar,zred)[2]
use_flux=masked_w_f_i(Wavelength,flux,ivar,zred)[1]
use_wave=masked_w_f_i(Wavelength,flux,ivar,zred)[0]
use_ivar=masked_w_f_i(Wavelength,flux,ivar,zred)[2]
priors = Infer.load_priors([
    Infer.UniformPrior(8, 12.5, label='sed'),
    Infer.FlatDirichletPrior(4, label='sed'), 
    Infer.UniformPrior(np.array([6.9e-5, 6.9e-5, 0., 0., -2.2]), np.array([7.3e-3, 7.3e-3, 3.,                               4.,0.4]),label='sed'), 
Infer.UniformPrior(np.array([0.9, 0.9, 0.9]), np.array([1.1, 1.1, 1.1]), label='flux_calib')])
m_nmf = Models.NMF(burst=False, emulator=True)
fluxcalib = FluxCalib.no_flux_factor
desi_mcmc = Infer.desiMCMC(
    model=m_nmf, 
    flux_calib=fluxcalib, 
    prior=priors
)
mcmc = desi_mcmc.run(
        wave_obs=use_wave,
        flux_obs=use_flux,
        flux_ivar_obs=use_ivar,
        zred=zred, 
        sampler='zeus',
        nwalkers=100,  
        burnin=100,
        opt_maxiter=10000,
        niter=1000)
flat_chain = desi_mcmc._flatten_chain(mcmc['mcmc_chain'])       
flat_chain_use=[]
count=0

for i in flat_chain:
    remove_3=flat_chain[count][:len(flat_chain[count])-3]
    flat_chain_use.append(remove_3)
    count=count+1

flat_chain_use=np.asarray(flat_chain_use)

logMstar_true=(mcmc['theta_bestfit'][0])
logMstar_inf=(flat_chain[:,0])


logSSFR_true=(np.log10(m_nmf.avgSFR(mcmc['theta_bestfit'][:len(mcmc['theta_bestfit'])-3], zred,                           dt=0.2)) - mcmc['theta_bestfit'][0])
logSSFR_inf=(np.log10(m_nmf.avgSFR(flat_chain_use, zred, dt=0.2)) - flat_chain_use[:,0])

logZ_MW_true=(np.log10(m_nmf.Z_MW(mcmc['theta_bestfit'][:len(mcmc['theta_bestfit'])-3],                                   m_nmf.cosmo.age(zred).value)))
logZ_MW_inf=(np.log10(m_nmf.Z_MW(flat_chain_use, m_nmf.cosmo.age(zred).value)))
log_tage_true=(np.log10(m_nmf.tage_MW(mcmc['theta_bestfit'][:len(mcmc['theta_bestfit'])-3],                               m_nmf.cosmo.age(zred).value)))
logtage_MW_inf=(np.log10(m_nmf.tage_MW(flat_chain_use, m_nmf.cosmo.age(zred).value)))


flat_chains = np.array(flat_chain_use)

logMstar_true = np.array(logMstar_true)
logMstar_inf = np.array(logMstar_inf)

logSSFR_true = np.array(logSSFR_true).flatten()
logSSFR_inf = np.array(logSSFR_inf)
logZ_MW_true = np.array(logZ_MW_true).flatten()
logZ_MW_inf = np.array(logZ_MW_inf)   
log_tage_true = np.array(log_tage_true).flatten()
logtage_MW_inf = np.array(logtage_MW_inf)

logm_q = np.array([DFM.quantile(logMstar_inf, [0.16, 0.5, 0.84]) for i in                                                 range(flat_chains.shape[0])])
logssfr_q = np.array([DFM.quantile(logSSFR_inf, [0.16, 0.5, 0.84]) for i in                                               range(flat_chains.shape[0])])
logzmw_q = np.array([DFM.quantile(logZ_MW_inf, [0.16, 0.5, 0.84]) for i in                                               range(flat_chains.shape[0])])  
logtage_q = np.array([DFM.quantile(logtage_MW_inf, [0.16, 0.5, 0.84]) for i in                                               range(flat_chains.shape[0])])

mass_yerr=[logm_q[:,1]-logm_q[:,0], logm_q[:,2]-logm_q[:,1]]
sfr_yerr=[logssfr_q[:,1]-logssfr_q[:,0], logssfr_q[:,2]-logssfr_q[:,1]]
zm_yerr=[logzmw_q[:,1] - logzmw_q[:,0], logzmw_q[:,2] - logzmw_q[:,1]]
tage_yerr=[logtage_q[:,1] - logtage_q[:,0], logtage_q[:,2] - logtage_q[:,1]]
sfhs, zhs = [], [] 
for tt in flat_chain_use[-10000:]:
    _, _sfh = fsps_emulator.SFH(tt, zred)
    _, _zh = fsps_emulator.ZH(tt, zred)
    sfhs.append(_sfh)
    zhs.append(_zh)
sfh_q = np.quantile(sfhs, [0.16, 0.50, 0.84], axis=0)
zh_q = np.quantile(zhs, [0.16, 0.5, 0.84], axis=0)

t, sfh = m_nmf.SFH(mcmc['theta_bestfit'][:len(mcmc['theta_bestfit'])-3], zred=zred)
tt, zwh = m_nmf.ZH(mcmc['theta_bestfit'][:len(mcmc['theta_bestfit'])-3], zred=zred)
tlookback = t[-1] - t
tlookback = np.delete(tlookback, 0)
import pandas as pd
import numpy as np
PlateIFU=[ifu_pl]
dataframe=pd.DataFrame(PlateIFU, columns=['Spectra'])
dataframe['ratio']=properties[0][2]
dataframe['Mass']=[logMstar_true]
dataframe['Mass_err']=[[mass_yerr[0][0]]+[mass_yerr[1][0]]]
dataframe['SFR']=[logSSFR_true]
dataframe['SFR_err']=[[sfr_yerr[0][0]]+[sfr_yerr[1][0]]]
dataframe['ZW']=[logZ_MW_true]
dataframe['ZW_err']=[[zm_yerr[0][0]]+[zm_yerr[1][0]]]
dataframe['Age']=[log_tage_true]
dataframe['Age_err']=[[tage_yerr[0][0]]+[tage_yerr[1][0]]]
dataframe.to_csv('Test_fit.csv', mode='a', header=False)



    
#f = open('/global/homes/m/marta_r/MaNGa_files/Internship_SFH.txt', 'a')
#f.write('Plate_ifu '+ str(PlateIFU)+'ratio '+ str(properties[i_obs][2])+' T_lookback '+ str(tlookback)+ ' SFH '+ str(sfh)+' ZWH '+ str(zwh)+' Sfh_q '+ str(sfh_q)+' Zh_q '+ str(zh_q)+str('\n'))
#f.write('Next_galaxy'+str(i_obs)+str('\n'))
#f.close()
