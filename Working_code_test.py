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

i0 = int(sys.argv[1]) 
i1 = int(sys.argv[2])
niter = int(sys.argv[3])
n_cpu = int(sys.argv[4])

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
    df = pd.read_csv("/global/homes/m/marta_r/MaNGa_files/All_fits_nersc.csv")
    cd=random.randint(0,df[df.columns[0]].count())
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
        df.to_csv("/global/homes/m/marta_r/MaNGa_files/All_fits_nersc.csv")
    for p in range (df[df.columns[0]].count()):
        #df = pd.read_csv("/global/homes/m/marta_r/Fitting_files/test2.csv")
        if cd <= (df[df.columns[0]].count())-1 and df.loc[cd, 'Validation'] == 0:
            df.loc[cd,'Validation'] ='1'
            df.to_csv("/global/homes/m/marta_r/MaNGa_files/All_fits_nersc.csv", index=True)
            z,ifu,ratio=df.loc[cd, 'redhsift'], df.loc[cd, 'Spectra'],df.loc[cd, 'ratio']
            rad,ba,phi=df.loc[cd,'radius'], df.loc[cd,'nsa_elpetro_ba'], df.loc[cd,'nsa_elpetro_phi']
            inner=[z,ifu,ratio,rad, ba, phi]
            break
        elif cd > (df[df.columns[0]].count()-1):
            cd-=1
        else: 
            cd+=1
    properties.append(inner)
def getting_data(i_obs):
    radius=properties[i_obs][3]
    ifu_pl=properties[i_obs][1]
    red=properties[i_obs][0]
    ratios=properties[i_obs][4]; angles=properties[i_obs][5]
    print('The radius is: ', radius)
pool = Pool(processes=n_cpu) 
import numpy as np
pool.map(partial(getting_data), np.arange(i0, i1+1))
pool.close()
pool.terminate()
pool.join()
