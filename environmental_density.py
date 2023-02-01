# Sean Dougherty
# 1/31/2023
# convinient separate script to calculate environmental density

### import libraries ###
import pandas as pd
pd.options.mode.chained_assignment = None  # has to do with setting values on a copy of a df slice, not an issue in execution
import numpy as np
from numpy import random
np.seterr(all="ignore") # np.log10(0 or -) occurs in data, but not an issue in execution
import time
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import math as m
import multiprocessing
from multiprocessing import Pool, freeze_support, RLock, RawArray, Manager
from functools import partial
from astropy.cosmology import FlatLambdaCDM
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Table
from scipy import signal
from scipy.interpolate import interp1d
import collections
import sys
import os, psutil

### define universal variables ###
CATALOG_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/CANDELS_COSMOS_CATS/'
CANDELS_PDF_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/Pair Project - Updated Data/'
COSMOS_PDF_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/COSMOS_data/'

# define cosmology
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
z_type='p' # ['p', 'ps', 's'] <-- only use zphot PDFs, zphot PDFs + zspecs, or only zspecs
max_dv = 1000 # max relative line-of-sight velocity for galaxy pairs
num_procs=10 # number of processors you want to use in the environmental density calculations
# initialize global dictionaries to aid in multiprocessing
PDF_dict = {} # for each field, store the array of all zphot PDFs in here for all pooled processors to utilize simultaneously
df_dict = {} # for each field, store the paret sample df in here for pooled processors to utilize simultaneously

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def main():
    print('Beginning main()...')
    start = time.perf_counter()
    # going to do one field at a time
    all_fields = ['GDS','EGS','COS','GDN','UDS','COSMOS'] # COS is for CANDELS COSMOS
    for field in all_fields:
        # calculate the environmetal density
        hm_df = calc_envd(field)
        hm_df['field'] = [field]*len(hm_df)
        # save and combine to field catalogs in post or load them in directly into conv_agn_merger
        hm_df.to_parquet(CATALOG_PATH+field+'_environmental_density.parquet', index=False)
        
    print('Done!')
    print('Final time:', time.perf_counter() - start)

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def calc_envd(field):
    """
    Loads data and PDFs, then calls multiprocessing to calculate environmental densities in parallel
    INPUTS
    : field - string - current field being studied
    RETURNS
    : hm_df - pd.Dataframe - columns include the galaxy's catalog ID and its environmental density
    """
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++ ...Calculating environmental density in {0} in {1}-mode... +++'.format(field, z_type))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Beginning conv_envd()...')
    # load in data
    if field == 'COSMOS':
        df = pd.read_csv(CATALOG_PATH+field+'_data2.csv', dtype={'ZSPEC_R':object})
        df = df.loc[ (df['LP_TYPE'] != 1) & (df['LP_TYPE'] != -99) & (df['MASS'] > 8) & # low cutoff for all gals we will consider
            (df['FLAG_COMBINED'] == 0) & (df['SIG_DIFF'] > 0) & (df['ZPHOT_PEAK'] > 0) & (df['CANDELS_FLAG'] == False) ]
    else:
        df = pd.read_csv(CATALOG_PATH+field+'_data2.csv', dtype={'ZSPEC_R':object})
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > 8) & (df['MASS'] < 15) &
            (df['SIG_DIFF'] > 0) & (df['ZPHOT_PEAK'] > 0) ] 
    df = df.reset_index(drop=True)
    
    # first thing is change df based on z_type
    df['z'] = df['ZPHOT_PEAK']
    if z_type != 'p':
        df.loc[ df['ZBEST_TYPE'] == 's', 'z' ] = df['ZSPEC']
        df.loc[ df['ZBEST_TYPE'] == 's', 'SIG_DIFF' ] = 0.01
    
    # load pdfs into global dictionary so all processors have access
    print('loading PDFs')
    load_pdfs(field)
    # throw the parent sample df onto the universal dictionary for use while multiprocessing:
    df_dict[field] = df
            
    # limit df to the relevant parent sample -> no need to calculate environmental density for all the sources
    hm_df = df.loc[ (df['MASS'] > 9.4) & (df['z'] > 0.5) & (df['z'] < 3.0) ].reset_index(drop=True)
    
    # make field a global variable so all the pool workers get it
    global gfield
    gfield = field
    
    # begin multiprocessing
    print('Filling apple pool in {}'.format(field)) 
    with Pool(processes=num_procs) as pool:
        # return the environmental density array
            result = pool.map(pll_od, np.array(hm_df['ID']))
            
    # close pool and clear dictionaries
    pool.close()
    pool.join()
    PDF_dict.clear()
    df_dict.clear()
    # var_dict.clear()
    
    # add environmental density measurements onto the parent sample df
    hm_df[z_type+'_env'] = np.array(result)
    
    # really all we need are ID and the environmental density
    return hm_df[['ID', z_type+'_env']]

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def pll_od(gal):
    """
    Spawned workers from multiprocessing Pool use this function to individually calculate environmental densities for galaxies
    INPUTS
    : gal - int - ID of galaxy for which the worker is to calculate environmental density
    RETURNS
    : n/np.pi - float - environmental density in units of Mpc^-2
    """
    # supply the pool worker with initial data
    field = gfield
    PDF_array = PDF_dict[field]
    z_01 = PDF_array[0,1:]
    df = df_dict[field]
    
    # what is the maximum arcsecond separation for 1 Mpc? ===> max_R_kpc
    # create SkyCoord objects and find all projected pairs to gal within 1 Mpc
    gal_pos = SkyCoord(df.loc[df['ID'] == gal, 'RA'] ,df.loc[df['ID'] == gal, 'DEC'], unit='deg')
    df_pos = SkyCoord(df['RA'], df['DEC'], unit='deg')
    R_kpc = cosmo.arcsec_per_kpc_proper(df.loc[df['ID'] == gal, 'z']) # arcsec/kpc at z=0.5
    max_R_kpc = 1000*u.kpc * R_kpc
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(gal_pos, max_R_kpc[0])
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    # neatly assemble these pairs into a df
    match_df = pd.DataFrame(matches)
    match_df = match_df.loc[match_df['arc_sep'] != 0].reset_index(drop=True)
    match_df['ID1'] = [gal]*len(match_df)
    match_df['ID2'] = np.array(df.loc[ match_df['partner_index'], 'ID'])

    # calculate all pair probabilities of all projected pairs within 1 Mpc
    match_df['Cp'] = Convdif(z_01, PDF_array[np.array(match_df['ID1']),1:], PDF_array[np.array(match_df['ID2']),1:], dv_lim=max_dv)
    
    # n is the sum of all pair probabilities (Cp) of all projected companions within 1 Mpc
    n = np.sum(match_df['Cp'])
    # print('{0} overdensity = {1}'.format(gal, n / np.pi)) # in Mpc^-2
    
    # return n / Mpc^2 ( / 1 Mpc^2 not printed below)
    return n / np.pi
    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def load_pdfs(field):
    """
    Load in the redshift PDFs for the parent sample and store them globally for use in pll_od()
    INPUTS
    : field - string - current field being studied
    """
    
    dA = np.linspace(0, 200, num=2001)
    
    # load in the PDFs:
    if field == 'COSMOS':
        with fits.open(COSMOS_PDF_PATH+'COSMOS2020_R1/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits') as data:
            COSMOS_PZ_arr = np.array(data[0].data)
        COSMOS_PZ_arrf = COSMOS_PZ_arr.byteswap().newbyteorder()
        COSMOS_PZ = pd.DataFrame(COSMOS_PZ_arrf)
        z_01 = COSMOS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(COSMOS_PZ) # becomes an array in the column case
    else:
        with fits.open(CANDELS_PDF_PATH+'CANDELS_PDFs/'+field+'_mFDa4.fits') as data:
            CANDELS_PZ_arr = np.array(data[0].data)
        CANDELS_PZ_arrf = CANDELS_PZ_arr.byteswap().newbyteorder()
        CANDELS_PZ = pd.DataFrame(CANDELS_PZ_arrf)
        z_01 = CANDELS_PZ.loc[0,1:].to_numpy()
        PDF_array = np.array(CANDELS_PZ)
        
    # zspecs are sensitive beyond 0.01 z, so increase P(z) grid size to 0.001 when including zspecs
    if z_type != 'p':
        # Interpolate to a finer grid:
        z_fine = np.linspace(0,10,10001).round(3)
        # initialize array to put all PDFs in
        PDF_array_ps = np.zeros((len(PDF_array),len(z_fine)+1))
        # add the fine redshift as the first row:
        PDF_array_ps[0,1:] = z_fine
        # add the IDs on the left hand side below:
        PDF_array_ps[:,0] = PDF_array[:,0]
        # fill the phot-zs first: need the IDs of zbest_type = 'p'
        fintp1 = interp1d(z_01, PDF_array[np.array(all_df.loc[ all_df['ZBEST_TYPE'] == 'p', 'ID' ]),1:], kind='linear')
        PDF_array_ps[ np.array(all_df.loc[ all_df['ZBEST_TYPE'] == 'p', 'ID' ]), 1: ] = fintp1(z_fine)
        # now for spec-zs
        # find where in the PDF_array_ps the z-value is our spec-z value and fill - round to 0.001 to match grid sensitivity
        spec_IDs = np.array(all_df.loc[ all_df['ZBEST_TYPE'] == 's', 'ID' ])
        spec_zs = np.array(all_df.loc[ all_df['ZBEST_TYPE'] == 's', 'z' ]).round(3)
        # sort the zspecs to the correct ID and z in PDF_array
        y = spec_zs
        x = PDF_array_ps[0,:]
        xsorted = np.argsort(x)
        ypos = np.searchsorted(x[xsorted], y)
        indices = xsorted[ypos]
        # set the P(z) = 1 where z=zspec when we include zspecs
        PDF_array_ps[spec_IDs, indices ] = 1
        # now normalize
        PDF_array_ps[spec_IDs,1:] = ( PDF_array_ps[spec_IDs,1:] / 
                                     np.array([np.trapz(PDF_array_ps[spec_IDs,1:], x=z_fine)]*PDF_array_ps[:,1:].shape[1]).T )
        # redefine to new 0.001 sensitivity
        z_01 = z_fine
        PDF_array = PDF_array_ps
        
    # add PDF array to the PDF_dict for use later:
    PDF_dict[field] = PDF_array
    
    return

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def dzdv(v):
    """
    Differentiates redshift with respect to line-of-sight velocity (i.e., radial velocity)
    INPUTS
    : v - float - line-of-sight velocity
    RETURNS
    : dzdv - float - derivative of z w.r.t. vel evaluated at v
    """
    c = 2.998e5 # km/s
    dzdv = (1/c) * ((1- (v/c))**(-1.5)) * ((1+ (v/c))**(-0.5))
    return dzdv

def radvel(z):
    """
    Calculates line-of-sight (i.e., radial) velocity from a redshift
    INPUTS
    : z - float - redshift
    RETURNS
    : v - float - line-of-sight velocity
    """
    c = 2.998e5 # km/s
    v = c * ( ((z+1)**2 - 1) / ((z+1)**2 + 1) )
    return v

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def Convdif(z_all, Pz1, Pz2, dv_lim=1000):
    """
    Calculates relative line-of-sight pair probabilities via convolutions of redshift PDFs (photometric or spectroscopic).
    Can input Pz1 and Pz2 as 2D np.arrays and the calculations will be make column-wise.
    INPUTS
    : z_all  - 1D np.array of floats - Redshift grid (z=0-10) with associates probabilities (e.g., Pz1 or Pz2). 
                                       Sensitivity = 0.01 for p-mode and 0.001 otherwise
    : Pz1    - 1D np.array of floats - the redshift PDF (P(z)) of galaxy 1 in a projected pair. 
                                       1D or 2D depends on whether this function is performed as a column-wise operation
    : Pz2    - 1D np.array of floats - the redshift PDF (P(z)) of galaxy 2 in a projected pair.
    : dv_lim - int - relative line-of-sight velocity considered to define a true galaxy pair
    RETURNS
    : prob - float - relative line-of-sight probabilities
    """
    # perform a change of variables into velocity space
    v_all = radvel(z_all)
    Pv1 = Pz1 * dzdv(v_all)
    Pv2 = Pz2 * dzdv(v_all)
    
    # interpolate the velocities to get evenly spaced points, z->v is not a linear map
    v_new = np.linspace(0,radvel(10),num=10000)
    fintp1 = interp1d(v_all, Pv1, kind='linear')
    fintp2 = interp1d(v_all, Pv2, kind='linear')
    
    # extend the inteprolated array into negative velocities to prepare for P(relative line-of-sight vel)
    all_v_neg = -1*v_new[::-1]
    all_ve = np.concatenate((all_v_neg[:-1], v_new))
    
    # convolve with the symmetrical interpolation values
    # the conditions below enure the code runs for single pairs as well as a column-wise operation based on input
    if len(fintp2(v_new).shape) == 1:
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[::-1], mode='full')
    else:
        v_conv = signal.fftconvolve(fintp2(v_new), fintp1(v_new)[:,::-1], mode='full', axes=1)
    
    # clip out negative values <-- very very small negative vakues can appear as a feature of the convolution around 0 probaiblities
    v_conv = np.clip(v_conv, 0, None)
    # normalize distribition
    try:
        v_conv = v_conv / np.trapz(v_conv, x=all_ve)
    except:
        v_conv = v_conv / np.array([np.trapz(v_conv, x=all_ve)]*v_conv.shape[1]).T

    # integrate velocity convolution to find probability rel l.o.s. vel within dv_lim (i.e., pair threshold)
    rnge = tuple(np.where( (all_ve > -dv_lim) & (all_ve < dv_lim)))
    try:
        prob = np.trapz(v_conv[rnge], x=all_ve[rnge])
    except:
        prob = np.trapz(v_conv[:,rnge], x=all_ve[rnge])
    
    return prob

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
if __name__ == '__main__':
    main()