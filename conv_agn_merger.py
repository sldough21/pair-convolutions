# Sean Dougherty
# 1/24/2023
# main script and functions for calculating convolutional pair probabilities and pair-finding algorithm

### import libraries ###
import pandas as pd
pd.options.mode.chained_assignment = None  # has to do with setting values on a copy of a df slice, not an issue with execution
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
SAVE_PATH = '/nobackup/c1029594/CANDELS_AGN_merger_data/agn_merger_output/conv_prob/'

# define cosmology
cosmo = FlatLambdaCDM(H0=70 * u.km / u.s / u.Mpc, Tcmb0=2.725 * u.K, Om0=0.3) # 0.7 for omega
# determine conservative ang separation corresponding to 100 kpc at z = 0.5
R_kpc = cosmo.arcsec_per_kpc_proper(0.5) # arcsec/kpc at z=0.5
max_R_kpc = 100*u.kpc * R_kpc # in arcseconds

mass_lo = 9.4 # parent sample lower mass threshold
gamma = 1.4 # for X-ray luminosity k-correction calculation
max_dv = 1000 # max relative line-of-sight velocity for galaxy pairs
select_controls = True # set to False to run the code without selecting control galaxies
save = True # whether to save the outputs
t_run = False # True to do a test run of the first 500 or so galaxies in a field catalog
z_type = 'ps' # ['p', 'ps', 's'] <-- only use zphot PDFs, zphot PDFs + zspecs, or only zspecs
date = '1.28' # add the date of your run for saving convention
num_procs = 15 # number of processors you want to use in the control galaxy selection
min_pp = 0.1 # minimum pair probability threshold <-- pairs with prob > 0.1 excluded from isolated galaxy pool

# initialize global dictionaries for iso_pools and PDF_array to help with multiprocesing
iso_pool_dict = {} # for each field, store the isolated pool in here for pooled processors to utilize simultaneously
PDF_dict = {} # for each field, store the array of all zphot PDFs in here for all pooled processors to utilize simultaneously
base_dz=0.05 # initial increments for expanding control galaxy search, redshift
base_dM=0.05 # log mass
base_dE=2 # environmental density
base_dS=0.05 # log PDF upper 68 - lower 68
N_controls=3 # number of control pairs to select for each true pair
sig_det = 5 # IRAC channel detection signifigance threshold
var_dict = {} # initialize dict for parallelized workers:

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def main():
    print('Beginning main()')
    # going to do one field at a time
    all_fields = ['GDS','EGS','COS','GDN','UDS','COSMOS'] # COS is for CANDELS COSMOS
    # all_fields = ['EGS'] # if you want to do just one field
    
    for field in all_fields:
        # load in data and calculate some initial values, return parent sample df
        all_df = process_samples(field)
        
        # identify galaxy pairs and calculate probabilities, returns true pair df, proj sep array, and all pair prob array
        gtrue_pairs, pair_Prp_gt0, all_Pp = determine_pairs(all_df, field)
        # add all pair probability array (i.e., probability each parent sample galaxy has any true pair within 100 kpc) to all_df
        all_df['all_pp'] = all_Pp
        
        # control galaxy selection
        if select_controls == True:
            # append broad position information for the pair (i.e., only for the prime) to the main pair df
            # this is used in the control galaxy selection to split COSMOS into quadrants for searching ease
            gtrue_pairs['prime_RA'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'RA'])
            gtrue_pairs['prime_DEC'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'DEC'])
            # control galaxy selection function
            control_df = conv_apples(gtrue_pairs, all_df, field)
            
        # finally, save
        if save == True:
            # save true pair dataframe as a parquet file
            gtrue_pairs.to_parquet(SAVE_PATH+'conv_output/PAIRS_Pp-'+str(min_pp)+'_M-'+str(mass_lo)+'_ztype-'+z_type+'_'+field+'_'+date+'.parquet', index=False)
            print('True pair dataframe saved')
            # save Prp array as a fits file
            hdu_rp = fits.PrimaryHDU(pair_Prp_gt0)
            hdul_rp = fits.HDUList([hdu_rp])
            hdul_rp.writeto(SAVE_PATH+'Prp_output/Prp_Pp-'+str(min_pp)+'_M-'+str(mass_lo)+'_ztype-'+z_type+'_'+field+'_'+date+'.fits', overwrite=True)
            print('P(rp) array saved')
            if select_controls == True:
                # save the control dataframe as well
                control_df.to_parquet(SAVE_PATH+'control_output/APPLES_Pp-'+str(min_pp)+'_M-'+str(mass_lo)+'_N-'+str(N_controls)+'_ztype-'+z_type+'_'+field+'_'+date+'.parquet', index=False)
                print('Control pair dataframe saved')
            
    print('Finished!')
    
    return
    
    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def process_samples(field):
    """
    Creates the parent sample dataframe, all_df. See readme file for which data is needed to perform this analysis.
    INPUTS
    : field  - string - Field currently being studied
    RETURNS
    : all_df - pd.DataFrame - Parent sample df
    """
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('+++ ...Selecting pairs in {0} in {1}-mode... +++'.format(field, z_type))
    print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
    print('Beginning process_samples()...')
    # load in catalogs: <== specify column dtypes
    if field == 'COSMOS':
        df = pd.read_csv(CATALOG_PATH+field+'_data4.csv', dtype={'ZSPEC_R':object})
        df = df.loc[ (df['LP_TYPE'] != 1) & (df['LP_TYPE'] != -99) & (df['MASS'] > mass_lo) &
            (df['FLAG_COMBINED'] == 0) & (df['SIG_DIFF'] > 0) & (df['ZPHOT_PEAK'] > 0) & (df['CANDELS_FLAG'] == False) ]
        df = df.rename(columns={'SPLASH_CH3_FLUX':'IRAC_CH3_FLUX', 'SPLASH_CH3_FLUXERR':'IRAC_CH3_FLUXERR',
                                'SPLASH_CH4_FLUX':'IRAC_CH4_FLUX', 'SPLASH_CH4_FLUXERR':'IRAC_CH4_FLUXERR'})
    else:
        df = pd.read_csv(CATALOG_PATH+field+'_data4.csv', dtype={'ZSPEC_R':object})
        df = df[ (df['CLASS_STAR'] < 0.9) & (df['PHOTFLAG'] == 0) & (df['MASS'] > mass_lo) &
            (df['SIG_DIFF'] > 0) & (df['ZPHOT_PEAK'] > 0) ] 
    df = df.reset_index(drop=True)
    ### ~~~ TEST RUN ~~~ ###
    if t_run == True:
        df = df.iloc[:500] 
        
    # calculate AB mags in IRAC channels:
    df['IRAC_CH1_ABMAG'] = F2m(df['IRAC_CH1_FLUX'], 1)
    df['IRAC_CH2_ABMAG'] = F2m(df['IRAC_CH2_FLUX'], 2)
    df['IRAC_CH3_ABMAG'] = F2m(df['IRAC_CH3_FLUX'], 3)
    df['IRAC_CH4_ABMAG'] = F2m(df['IRAC_CH4_FLUX'], 4)       
        
    # first thing is change df based on z_type
    df['z'] = df['ZPHOT_PEAK']
    if z_type != 'p':
        df.loc[ df['ZBEST_TYPE'] == 's', 'z' ] = df['ZSPEC']
        df.loc[ df['ZBEST_TYPE'] == 's', 'SIG_DIFF' ] = 0.01
        
    # make definite redshift cut and limit to just zspecs if desired
    if z_type == 's':
        all_df = df.loc[ (df['z'] > 0.5) & (df['z'] < 3.0) & (df['ZBEST_TYPE'] == 's') ]
    else:
        all_df = df.loc[ (df['z'] > 0.5) & (df['z'] < 3.0) ]
    print('Size of parent sample is {} galaxies'.format(len(all_df)))
    all_df = all_df.reset_index(drop=True)
    
    # calculate LX
    all_df['LX'] = ( all_df['FX'] * 4 * np.pi * ((cosmo.luminosity_distance(all_df['z']).to(u.cm))**2).value * 
                                                                ((1+all_df['z'])**(gamma-2)) )
    # Flag IR AGN based on Donley12
    all_df['IR_AGN_DON'] = [0]*len(all_df)
    all_df.loc[ (np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']) >= 0.08) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) >= 0.15) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) >= (1.21*np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']))-0.27) &
               (np.log10(all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH2_FLUX']) <= (1.21*np.log10(all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH1_FLUX']))+0.27) &
               (all_df['IRAC_CH2_FLUX'] > all_df['IRAC_CH1_FLUX']) &
               (all_df['IRAC_CH3_FLUX'] > all_df['IRAC_CH2_FLUX']) &
               (all_df['IRAC_CH4_FLUX'] > all_df['IRAC_CH3_FLUX']), 'IR_AGN_DON'] = 1
    
    # set the ones with incomplete data back to 0: POTENTIALLY UNECESSARY NOW (BELOW)
    all_df.loc[ (all_df['IRAC_CH1_FLUX'] <= 0) | (all_df['IRAC_CH2_FLUX'] <= 0) |
               (all_df['IRAC_CH3_FLUX'] <= 0) | (all_df['IRAC_CH4_FLUX'] <= 0), 'IR_AGN_DON' ] = 0
    all_df.loc[ (all_df['IRAC_CH1_FLUX']/all_df['IRAC_CH1_FLUXERR'] < sig_det) | (all_df['IRAC_CH2_FLUX']/all_df['IRAC_CH2_FLUXERR'] < sig_det) |
               (all_df['IRAC_CH3_FLUX']/all_df['IRAC_CH3_FLUXERR'] < sig_det) | (all_df['IRAC_CH4_FLUX']/all_df['IRAC_CH4_FLUXERR'] < sig_det),
              'IR_AGN_DON' ] = 0
        
    # determine projected pairs and their true pair probabilities (below)
    return all_df

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def determine_pairs(all_df, field):
    """
    Identifies projected pairs and calculates pair probabilities.
    An important step of this function is calling pair_pdfs(), which loads and save all galaxy PDFs into one global array 
        (in PDF_dict), which is used later in selecting controls
    INPUTS
    : all_df - pd.DataFrame - parent sample dataframe
    : field  - string - the field currently being studied
    RETURNS
    : gtrue_pairs  - pd.DataFrame - True pair df; gives IDs, relative physical properties, and AGN flags
    : pair_Prp_gt0 - 2D np.array of floats - True pair projected separation probability distributions. 
                                             Each row gives the IDs of the pair followed by P(rp) from 0-200 kpc (steps of 0.1 kpc)
    : all_Pp       - 1D np.array of floats - Stores probabilities parent sample galaxise are in any pair. 
                                             Added to all_df in main function
    """
    print('Beginning determine_pairs()...')
    # match catalogs:
    df_pos = SkyCoord(all_df['RA'],all_df['DEC'],unit='deg')
    idxc, idxcatalog, d2d, d3d = df_pos.search_around_sky(df_pos, max_R_kpc)
    # place galaxy pairs into a df and get rid of duplicate pairs:
    matches = {'prime_index':idxc, 'partner_index':idxcatalog, 'arc_sep': d2d.arcsecond}
    match_df = pd.DataFrame(matches)
    # all galaxies will match to themselves, i.e. with 0 separation, so get rid of these
    pair_df = match_df[ (match_df['arc_sep'] != 0.00) ]
    # calculate mass ratio
    pair_df['mass_ratio'] = (np.array(all_df.loc[pair_df['prime_index'], 'MASS']) - 
                             np.array(all_df.loc[pair_df['partner_index'],'MASS']) )
    
    # get rid of duplicates in pair sample
    pair_df = pair_df[ (pair_df['mass_ratio'] >= 0) ]
    sorted_idx_df = pd.DataFrame(np.sort((pair_df.loc[:,['prime_index','partner_index']]).values, axis=1), 
                                    columns=(pair_df.loc[:,['prime_index','partner_index']]).columns).drop_duplicates()
    pair_df = pair_df.reset_index(drop=True)
    pair_df = pair_df.iloc[sorted_idx_df.index]
    # we only want pairs where the mass ratio is within 10
    pair_df = pair_df[ (pair_df['mass_ratio'] <= 1) ] # log masses
    # calculate projected separation at z
    pair_df['kpc_sep'] = (pair_df['arc_sep']) / (cosmo.arcsec_per_kpc_proper(all_df.loc[pair_df['prime_index'], 'z']).value)
    true_pairs = pair_df
        
    # put the output dataframe together    
    gtrue_pairs = true_pairs.reset_index(drop=True)    
    gtrue_pairs['prime_ID'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ID' ]) # ID as appears in catalog
    gtrue_pairs['partner_ID'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ID' ])
    gtrue_pairs['prime_z'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'z' ])      
    gtrue_pairs['prime_zt'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'ZBEST_TYPE' ]) # i.e., p or s (zphot or zspec)
    gtrue_pairs['partner_z'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'z' ])
    gtrue_pairs['partner_zt'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'ZBEST_TYPE' ])
    gtrue_pairs['prime_M'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'MASS' ])
    gtrue_pairs['partner_M'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'MASS' ])
    # gtrue_pairs['prime_SFR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'LOGSFR_MED' ]) 
    # gtrue_pairs['partner_SFR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'LOGSFR_MED' ])
    gtrue_pairs['prime_LX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'LX' ])
    gtrue_pairs['partner_LX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'LX' ])
    gtrue_pairs['prime_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'SIG_DIFF']) # P(z) width
    gtrue_pairs['partner_PDFsig'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'SIG_DIFF'])
    # # IR data
    # gtrue_pairs['prime_CH1_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_FLUX'])
    # gtrue_pairs['prime_CH2_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_FLUX'])
    # gtrue_pairs['prime_CH3_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_FLUX'])
    # gtrue_pairs['prime_CH4_FLUX'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_FLUX'])
    # gtrue_pairs['partner_CH1_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_FLUX'])
    # gtrue_pairs['partner_CH2_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_FLUX'])
    # gtrue_pairs['partner_CH3_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_FLUX'])
    # gtrue_pairs['partner_CH4_FLUX'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_FLUX'])
    # gtrue_pairs['prime_CH1_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_FLUXERR'])
    # gtrue_pairs['prime_CH2_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_FLUXERR'])
    # gtrue_pairs['prime_CH3_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_FLUXERR'])
    # gtrue_pairs['prime_CH4_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_FLUXERR'])
    # gtrue_pairs['partner_CH1_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_FLUXERR'])
    # gtrue_pairs['partner_CH2_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_FLUXERR'])
    # gtrue_pairs['partner_CH3_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_FLUXERR'])
    # gtrue_pairs['partner_CH4_FLUXERR'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_FLUXERR'])
    # gtrue_pairs['prime_CH1_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH1_ABMAG'])
    # gtrue_pairs['prime_CH2_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH2_ABMAG'])
    # gtrue_pairs['prime_CH3_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH3_ABMAG'])
    # gtrue_pairs['prime_CH4_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['prime_index'], 'IRAC_CH4_ABMAG'])
    # gtrue_pairs['partner_CH1_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH1_ABMAG'])
    # gtrue_pairs['partner_CH2_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH2_ABMAG'])
    # gtrue_pairs['partner_CH3_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH3_ABMAG'])
    # gtrue_pairs['partner_CH4_ABMAG'] = np.array(all_df.loc[ gtrue_pairs['partner_index'], 'IRAC_CH4_ABMAG'])
    gtrue_pairs['prime_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['prime_index'], 'IR_AGN_DON']) # IR AGN flag
    gtrue_pairs['partner_IR_AGN_DON'] = np.array(all_df.loc[gtrue_pairs['partner_index'], 'IR_AGN_DON'])
    
    gtrue_pairs['prime_env'] = np.array(all_df.loc[gtrue_pairs['prime_index'], z_type+'_env']) # evironmental density
    gtrue_pairs['partner_env'] = np.array(all_df.loc[gtrue_pairs['partner_index'], z_type+'_env'])
    gtrue_pairs['field'] = [field] * len(gtrue_pairs)
    
    # major and minor complete samples require the more massive galaxy (prime by my convention) to be > 10 log(solar masses)
    gtrue_pairs = gtrue_pairs.loc[ gtrue_pairs['prime_M'] > 10 ].reset_index(drop=True)

    # get true pair probabilities and the projected separation PDFs (P(r_p) in paper)
    pair_probs, pair_Prp = pair_pdfs(gtrue_pairs, all_df)
    gtrue_pairs['pair_prob'] = pair_probs
    # get rid of 0 probability pairs, NOTE that many have Pp ~= 1e-18, so feel free to change the first 0 to ~= 0
    gtrue_pairs = gtrue_pairs.loc[ gtrue_pairs['pair_prob'] > 0 ]
        
    # keep consistent indexing in the Prp array
    pair_Prp_gt0 = pair_Prp[gtrue_pairs.index]
    gtrue_pairs = gtrue_pairs.reset_index(drop=True)
    
    # determine the probability a galaxy is in any pair (1-P_iso in paper)
    all_Pp = [0]*len(all_df)
    for i, ID in enumerate(all_df['ID']):
        # get parts of true_pairs where ID is prime ID
        gal_match_probs = gtrue_pairs.loc[ (gtrue_pairs['prime_ID'] == ID) | (gtrue_pairs['partner_ID'] == ID), 'pair_prob' ]
        if len(gal_match_probs) == 0:
            all_Pp[i] = 0
        elif len(gal_match_probs) != 0: 
            all_Pp[i] = 1 - np.prod(1-gal_match_probs)
            
    # save the true pairs dataframe for post-processing
    print('There are {} true pairs'.format(len(gtrue_pairs)))    
    
    return gtrue_pairs, pair_Prp_gt0, all_Pp
    
    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def pair_pdfs(pair_df, all_df):
    """
    Loads the PDFs of the parent sample into a global variable (PDF_dict).
    Determines pair probabilities and projected separation probability distributions (Prp).
    INPUTS
    : all_df - pd.DataFrame - parent sample dataframe
    : field  - string - the field currently being studied
    RETURNS
    : Cv_prob  - 1D np.array of floats - array holding the relative line-of-sight pair probabilities for each row of gtrue_pairs, 
                                         added as column in determine_pairs
    : Prp_2sav - 2D np.array of floats - True pair projected separation probability distributions. Each row gives the IDs of the pair
                                         followed by P(rp) from 0 to 200 kpc (in steps of 0.1 kpc)
    """
    field = pair_df['field'].unique()[0]
    print('Beginning pair_pdfs()...')
    
    # I am going to give a probability every 0.1 kpc
    rp = np.linspace(0, 200, num=2001)
    # define array sizes to save distributions
    Prp_2sav = np.zeros((len(pair_df), len(rp)+2)) # so I can add the IDs and field as a check
    
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
    
    # get convolutional pair probabilities
    Cv_prob = Convdif(z_01, PDF_array[pair_df['prime_ID'],1:], PDF_array[pair_df['partner_ID'],1:], dv_lim=max_dv)
    # get proj sep PDF (P(r_p) in paper) and combined z PDF (P(z1=z2) in paper)
    Prp, comb_PDF = Prp_prob(PDF_array[pair_df['prime_ID'],1:], PDF_array[pair_df['partner_ID'],1:], 
                             np.array(pair_df['prime_zt']), np.array(pair_df['partner_zt']), 
                             np.array(pair_df['arc_sep']), z_01, rp)
    # fill the Prp_2sav array
    Prp_2sav[:,0] = np.array(pair_df['prime_ID'])
    Prp_2sav[:,1] = np.array(pair_df['partner_ID'])
    Prp_2sav[:,2:] = Prp    
    
    return Cv_prob, Prp_2sav

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
def Prp_prob(Pz1, Pz2, zt1, zt2, theta, z, rp):
    """
    Create projected separation probability distributions (P(rp)) via a change of variables.
    Can input Pz1 and Pz2 as 2D np.arrays (and zt1, zt2, and theta as 1D arrays of the same length) 
    and the calculations will be make column-wise.
    INPUTS
    : Pz1   - 1D np.array or floats - the redshift PDF (P(z)) of galaxy 1 in a projected pair. 
                                      1D or 2D depends on whether this function is performed as a column-wise operation
    : Pz2   - 1D np.array or floats - the redshift PDF (P(z)) of galaxy 2 in a projected pair.
    : zt1   - string - redshift type (photometric, 'p', or spectroscopic, 's')
    : zt2   - string - redshift type (photometric, 'p', or spectroscopic, 's')
    : theta - float  - angular separation in arcseconds
    : z     - 1D np.array of floats - Redshift grid (z=0-10) with associates probabilities (e.g., Pz1 or Pz2). 
                                      Sensitivity = 0.01 for p-mode and 0.001 otherwise
    : rp    - 1D np.array - projected separation grid (i.e., 0-200 kpc in steps of 0.1 kpc)
    RETURNS
    : Prp      - 1D np.array of floats - projected separation probabilitu distributions interpolated to separations rp
    : comb_PDF - 1D np.array of floats - combined redshift probability distribution function (i.e., P(z1=z2))
    """
    # combined P(z)'s
    comb_PDF = Pz1*Pz2
    # get a middle_z for all rows:
    mid_z_arr = np.vstack( (z[np.argmax(Pz1, axis=1)], z[np.argmax(Pz2, axis=1)]) ).T
    middle_z = np.mean( mid_z_arr, axis=1 )
    
    # if both galaxies have a zspec, default their P(r_p) = 1 based on their mean redshift
    # if zspecs are close enough, then the convolution probability will give 1, and thi middle r_p approximation will be appropriate
    # if zspecs are not close, conv prob = 0, so artificially popl\ulating the middle proj sep value won't matter
    if z_type != 'p':
        s2_idx = np.where((zt1=='s') & (zt2=='s'))[0]
        y = middle_z[s2_idx]
        x = z
        xsorted = np.argsort(x)
        ypos = np.searchsorted(x[xsorted], y)
        indices = xsorted[ypos]
        comb_PDF[ s2_idx, indices] = 1
        
    # normalize
    comb_PDF = (comb_PDF / np.array([np.trapz(comb_PDF, x=z)]*len(z)).T)
    comb_PDF = np.nan_to_num(comb_PDF) # if they don't ever overlap we get badness, which this fixes
    
    # split rp into 0-1.61 (1) and 1.62-10 (2), as this is about where f(z) = r_p turns over
    # i.e., different redshifts can produce the same r_p
    rp1 = ang_diam_dist( z[np.where(z <= 1.61)], theta )
    rp2 = ang_diam_dist( z[np.where(z > 1.61)], theta )
    z1 = z[np.where(z <= 1.61)]
    z2 = z[np.where(z > 1.61)]
    comb_Pz1 = comb_PDF[:,np.where(z <= 1.61)[0]]
    comb_Pz2 = comb_PDF[:,np.where(z > 1.61)[0]] 

    # change of variables from P(z1=z2) -> P(r_p)
    Prp11 = comb_Pz1 * np.abs(dzdrp(rp1, z1, theta))
    Prp11 = np.nan_to_num(Prp11) # fill the nan casued by division ny zero
    Prp12 = comb_Pz2 * np.abs(dzdrp(rp2, z2, theta))
    
    # concatenate 1 and 2 just to find the max
    rp_comb = np.concatenate((rp1, rp2), axis=1)
    Prp_comb = np.concatenate((Prp11, Prp12), axis=1)
    max_rp = rp_comb[np.arange(len(rp_comb)),np.argmax(Prp_comb, axis=1)]
    rp_new = rp
    
    # a for loop is needed to interpolate and place the Prps onto the grid defined above - very short for loop
    print('Running interpolation loop in Prp_prob(), please wait...')
    intr_Prp1 = np.zeros((len(rp1),len(rp_new)))
    intr_Prp2 = np.zeros((len(rp2),len(rp_new)))
    for i in tqdm(range(0,len(rp1)), miniters=100):
        fintp1 = interp1d(rp1[i,:], Prp11[i,:], kind='linear', bounds_error=False, fill_value=0)
        fintp2 = interp1d(rp2[i,:], Prp12[i,:], kind='linear', bounds_error=False, fill_value=0)
        intr_Prp1[i,:] = fintp1(rp_new)
        intr_Prp2[i,:] = fintp2(rp_new)

    # after gridding (via interpolation), add values that fill the same grid (recall rp1 and rp2)
    Prp = intr_Prp1+intr_Prp2
    
    # find value in rp_new that is closest to max_rp in every row:
    rp_new2 = np.array([rp_new]*len(max_rp))
    max_rp2 = np.array([max_rp]*len(rp_new)).T
    # if there is a zspec involved, the interpolation may be off (previously one r_p had P=1; delta function essentially)
    # so find and correct those peaks within the gridded P(r_p) sensitivity
    if z_type != 'p':
        s3_idx = np.where((zt1=='s') | (zt2=='s'))[0]
        max_rp_idxs = np.argmin(np.abs(rp_new2[s3_idx,:]-max_rp2[s3_idx,:]), axis=1)
        Prp[s3_idx, max_rp_idxs] = np.max(Prp_comb[s3_idx,:], axis=1)
        
    # normalize
    Prp = Prp / np.array([np.trapz(Prp, x=rp_new)]*Prp.shape[1]).T
    
    return np.nan_to_num(Prp), comb_PDF

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def F2m(F, ch):
    """
    Calculates AB magnitude from IRAC channel fluxes
    INPUTS
    : F  - float - IRAC flux in uJy
    : ch - int - flux channel (1, 2, 3, or 4)
    RETURNS
    : - float - AB magnitude
    """
    if ch == 1:
        F0 = 280.9
        K = 2.788
    elif ch == 2:
        F0 = 179.7
        K = 3.255
    elif ch == 3:
        F0 = 115
        K = 3.743
    elif ch == 4:
        F0 = 64.9
        K = 4.372
    return 2.5*np.log10(F0/(F*1e-6)) + K

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def ang_diam_dist(z, theta, H0=70, Om_m=0.3, Om_rel=0, Om_lam=0.7, Om_0=1):
    """
    Calculate the angular diameter distance and the projected separation (rp) based on a lambdaCDM cosmology
    INPUTS
    : z      - float - redshift
    : theta  - float - angular separation in arcseconds
    : H0     - float - Hubble constant 
    : Om_m   - float - omega matter
    : Om_rel - float - omega relativistic particles
    : Om_lam - float - omega dark energy
    : Om_0   - float - total density parameter
    RETURNS
    : - float - projeced separation in kpc
    """
    c = 2.998e5 # km/s
    # again, the try-except conditions ensure this works for one pair at a time and as part column-wise operations
    # create redshift grid needed for the change of variables equation below
    try:
        zs = np.linspace(0,z,10, endpoint=True, axis=1)
    except:
        zs = np.linspace(0,z,10, endpoint=True)
    rp = ( c / (H0*(1+z)) ) * np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs )
    # rp is projected separation in Mpc, return it in kpc (need to convert theta from arcseconds to radians)
    try:
        return rp * theta * 1000 / ((180/np.pi)*3600)
    except:
        exp_rp = np.array([rp]*len(theta))
        exp_theta = np.array([theta]*exp_rp.shape[1]).T
        return exp_rp * exp_theta * 1000 / ((180/np.pi)*3600)

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def dzdrp(rp, z, theta, H0=70, Om_m=0.3, Om_rel=0, Om_lam=0.7, Om_0=1):
    """
    Differentiates redshift with respect to projected separation
    INPUTS
    : rp     - float - projected separation in kpc
    : z      - float - redshift
    : theta  - float - angular separation in arcseconds
    : H0     - float - Hubble constant 
    : Om_m   - float - omega matter
    : Om_rel - float - omega relativistic particles
    : Om_lam - float - omega dark energy
    : Om_0   - float - total density parameter
    RETURNS
    : dzdv - float - derivative of z w.r.t. proj sep evaluated at specific projected separations
    """
    c = 2.998e5 # km/s
    # again, the try-except conditions ensure this works for one pair at a time and as part column-wise operations
    # create redshift grid needed to compute the differentiation below
    try:
        zs = np.linspace(0,z,10, endpoint=True, axis=1) # numerically integrate
    except:
        zs = np.linspace(0,z,10, endpoint=True)
    dzdrp = - (c * np.trapz( ( 1 / np.sqrt( Om_m*(1+zs)**3 + Om_rel*(1+zs)**4 + Om_lam + (1-Om_0)*(1+zs)**2 ) ), x=zs )) / (H0*rp**2)
    # return differentiation in kpc units
    try:
        return dzdrp * theta * 1000 / ((180/np.pi)*3600)
    except:
        exp_theta = np.array([theta]*dzdrp.shape[1]).T
        return dzdrp * exp_theta * 1000 / ((180/np.pi)*3600)

    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def conv_apples(pair_df, iso_pool_df, field):
    """
    Prepare for control selection (i.e., apple bobbing) do be done using multiprocessing
    INPUTS
    : pair_df     - pd.DataFrame - true pair df; gives IDs, relative physical properties, and AGN flags
    : iso_pool_df - pd.DataFrame - parent sample df, becomes pool of isolated galaxies following all_pp cut below
    : field       - string - field currently being studied
    RETURNS
    : control_df - pd.DataFrame - control pair df; gives relavent properties (as well as corresponding true pair IDs and probs)
    """
    print('Beginning conv_apples()...')
    # select the isolated pool as all galaxies with probability of being in any true pair < 10% 
    iso_pool_df = iso_pool_df.loc[ iso_pool_df['all_pp'] <= min_pp ].reset_index(drop=True)
        
    # split COSMOS into quadrants to speed up the convolution calculations when searching for controls:
    iso_pool_df['Quadrant'] = [0]*len(iso_pool_df)
    pair_df['Quadrant'] = [0]*len(pair_df)
    # if a pair or isolated galaxy fall into a certain coordinate gride space, assign them that quadrant number
    if field == 'COSMOS':
        percs = np.linspace(0,100,4) # will split COSMOS up into 9 pieces
        RA_perc = np.percentile(iso_pool_df['RA'], percs)
        DEC_perc = np.percentile(iso_pool_df['DEC'], percs)
        for i in range(len(RA_perc)-1):
            for j in range(len(DEC_perc)-1):
                iso_pool_df.loc[ (iso_pool_df['RA'] >= RA_perc[i]) & (iso_pool_df['RA'] <= RA_perc[i+1]) & 
                       (iso_pool_df['DEC'] >= DEC_perc[j]) & (iso_pool_df['DEC'] <= DEC_perc[j+1]), 'Quadrant'] =  int(str(i)+str(j))
                pair_df.loc[ (pair_df['prime_RA'] >= RA_perc[i]) & (pair_df['prime_RA'] <= RA_perc[i+1]) & 
                       (pair_df['prime_DEC'] >= DEC_perc[j]) & (pair_df['prime_DEC'] <= DEC_perc[j+1]), 'Quadrant'] =  int(str(i)+str(j))
        
    # throw the iso_pool df onto the universal dictionary for use while multiprocessing:
    iso_pool_dict[field] = iso_pool_df
    
    # begin the pooling
    print('Filling apple pool...') 
    start = time.perf_counter()
    pair_df = pair_df.drop(columns={'prime_zt','partner_zt','field'}) # RawArray does not work with strings
    
    # set up initial arguments for all the pool workers
    P_shape = np.array(pair_df).shape
    P_data = np.array(pair_df)
    P_cols = np.array(pair_df.columns)
    P_field = field
    P = RawArray('d', P_shape[0] * P_shape[1]) # 'd' is for double
    # wrap P as an numpy array so we can easily manipulates its data
    P_np = np.frombuffer(P).reshape(P_shape)
    # Copy data to the shared array.
    np.copyto(P_np, P_data)
    
    # begin selecting control galaxies using pool workers on the bobbing_pll function below
    with Pool(processes=num_procs, initializer=init_worker, initargs=(P, P_cols, P_shape, P_field)) as pool:
        result = pool.map(bobbing_pll, range(P_shape[0]))
            
    # join pool and clear all universal dictionaries used per field:
    pool.close()
    pool.join()
    iso_pool_dict.clear()
    PDF_dict.clear()
    var_dict.clear()
    print('Pool drained')
    
    # send the list of dfs to byhand function to combine using system storage
    control_df = byhand(result)
    # add field and unique pair_str in post
    control_df['field'] = [field]*len(control_df)
    control_df['pair_ID'] = control_df['field']+'_'+(control_df['ID1'].astype(int)).astype(str)+'+'+(control_df['ID2'].astype(int)).astype(str)
    
    print('Control selection complete... final time:', time.perf_counter() - start)
        
     ### BTW if you're wondering where the 'apple' nomenclature came from... 
    # pairs --sounds like--> pears --what to call control pears--> apples!
    # how are apples selected? bobbing for apples, of course! ###
        
    return control_df
    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def init_worker(P, P_cols, P_shape, P_field):
    """
    Multiprocessing pool worker initialization function, equips all workers with the inputs:
    INPUTS
    : P       - 1D np.array of FLOATS - the true pair dataframe compressed into a wrapped np.array, 
                                        much easier for pool workers to recieve
    : P_cols  - np.array of strings - the columns of the true pair df 
    : P_shape - np.array().shape() - original shape of the true pair df
    : P_field - string - field being studied
    """
    # by spawning all the pool workers in here, they begin their work in bobbing_ppl with this dictionary in memory
    var_dict['field'] = P_field
    var_dict['P'] = P
    var_dict['Pshape'] = P_shape
    var_dict['Pcols'] = P_cols
    
### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def byhand(dfs):
    """
    Use file storage to combine the tens of thousands of control datafames that can be output from bobbing_pll
    INPUTS
    : dfs - array of pd.DataFrames - must all have save columns and only contain one datatype
    RETURNS
    : df_all - pd.DataFrame - a singular combined df
    """
    mtot=0    # save this in storage
    with open(SAVE_PATH+'storage/df_all.bin','wb') as f:
        for df in dfs:
            m,n =df.shape
            mtot += m
            f.write(df.values.tobytes())
            typ=df.values.dtype                
    #delete dfs
    with open(SAVE_PATH+'storage/df_all.bin','rb') as f:
        buffer=f.read()
        data=np.frombuffer(buffer,dtype=typ).reshape(mtot,n)
        df_all=pd.DataFrame(data=data,columns=dfs[0].columns) 
    os.remove(SAVE_PATH+'storage/df_all.bin')
    return df_all

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def PP_diff_func(x):
    """
    1/x function created using scipy optimize to appropriately contrain differences in true and control pair probabilities.
    Matching function as described in paper.
    INPUTS
    : x - float - a true pair probability
    RETURNS
    : - float - the maximum logarithmic distance between the control and true pair prob (i.e., log(Pp-Cp) < the return)
    """
    # these constants were determined using curve_fit from scipy.optimize on a 1/x function
    A = 5.24002529e-05
    C = 1.54419221e-04
    return C*(1/(x+A)) + 0.05

### --------------------------------------------------------- ###
### --------------------------------------------------------- ###
def bobbing_pll(i):    
    """
    Workers spawned by multiprocessing pool individually use this function to select control galaxies
    INPUTS:
    : i - int - the index corresponding to a true pair in the true pair df (pair_arr below)
    RETURNS:
    : control_df - pd.DataFrame - holds three unique control galaxy pairs in each row
    """
    # recover some variables:
    field = var_dict['field']
    iso_pool_df = iso_pool_dict[field]
    PDF_array = PDF_dict[field]
    z_01 = PDF_array[0,1:]    
    # can recover pair_df data from var_dict:
    pair_arr = np.frombuffer(var_dict['P']).reshape(var_dict['Pshape'])
    ID1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_ID')][0][0]
    z1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_z')][0][0]
    M1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_M')][0][0]
    SIG1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_PDFsig')][0][0]
    E1 = pair_arr[i, np.where(var_dict['Pcols'] == 'prime_env')][0][0]
    ID2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_ID')][0][0]
    z2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_z')][0][0]
    M2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_M')][0][0]
    SIG2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_PDFsig')][0][0]
    E2 = pair_arr[i, np.where(var_dict['Pcols'] == 'partner_env')][0][0]
    Pp = pair_arr[i, np.where(var_dict['Pcols'] == 'pair_prob')][0][0]
    Qd = pair_arr[i, np.where(var_dict['Pcols'] == 'Quadrant')][0][0]
    all_Qds = np.unique(pair_arr[:, np.where(var_dict['Pcols'] == 'Quadrant')][0])
    
    # initialize some variables
    got = False
    report = False
    dz = base_dz
    dM = base_dM
    dE = base_dE
    dS = base_dS
    tried_arr = []
    tries = 0
    
    # begin search - see control selection section in paper for more detail
    while got == False:
        # keep track of how many search attempts are made
        tries+=1
        # look for galaxies in the isolated pool that are within z, mass, environment, and P(z) width thresholds for gal 1 and 2
        iso1 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z1) < dz) & (np.abs(iso_pool_df['MASS']-M1) < dM) &
                              (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd) &
                               (np.abs(iso_pool_df[z_type+'_env']-E1) < dE) & 
                               (np.abs(np.log10(iso_pool_df['SIG_DIFF']) - np.log10(SIG1)) < dS), 
                              ['ID','RA','DEC','MASS','z', 'SIG_DIFF','LX','IR_AGN_DON', z_type+'_env'] ] 
        iso1 = iso1.rename(columns={'RA':'RA1', 'DEC':'DEC1', 'z':'z1', 'SIG_DIFF':'SIG1', 'MASS':'MASS1',
                                    'LX':'LX1', 'IR_AGN_DON':'IR_AGN_DON1', z_type+'_env':'ENV1'})
        iso2 = iso_pool_df.loc[ (np.abs(iso_pool_df['z']-z2) < dz) & (np.abs(iso_pool_df['MASS']-M2) < dM) &
                              (iso_pool_df['ID'] != ID1) & (iso_pool_df['ID'] != ID2) & (iso_pool_df['Quadrant'] == Qd) & 
                               (np.abs(iso_pool_df[z_type+'_env']-E2) < dE) & 
                               (np.abs(np.log10(iso_pool_df['SIG_DIFF']) - np.log10(SIG2)) < dS),
                              ['ID','RA','DEC','MASS','z', 'SIG_DIFF','LX','IR_AGN_DON', z_type+'_env'] ]
        iso2 = iso2.rename(columns={'RA':'RA2', 'DEC':'DEC2', 'z':'z2', 'SIG_DIFF':'SIG2', 'MASS':'MASS2',
                                    'LX':'LX2', 'IR_AGN_DON':'IR_AGN_DON2', z_type+'_env':'ENV2'})

        # cross-merge the iso1 and iso2 dfs to get all possible candidate control pair combinations
        apple_df = iso1.merge(iso2, how='cross').rename(columns={'ID_x':'ID1','ID_y':'ID2'})

        # create unique pair strings:
        apple_df['pair_str'] = (apple_df['ID1'].astype(int)).astype(str)+'+'+(apple_df['ID2'].astype(int)).astype(str)
        # add a reuse flag to keep considering pairs that meed the matching function (below) in the next iteration
        # this makes it so we don't need to recalculate conv_prob for galaxies in a parameter space we've already looked and
        #      determined are not good matches in pair probability
        apple_df['reuse_flag'] = [0]*len(apple_df)
        # if we've already determined candidate control pairs are not good matches, discard from the df
        apple_df = apple_df.loc[ (apple_df['pair_str'].isin(tried_arr) == False) ].reset_index(drop=True)
        # initial search safeguard --> if there are no candidates at this step, expand search and loop again
        if len(apple_df) == 0:
            dz = dz + 0.03
            dM = dM + 0.03
            dE = dE + 1
            dS = dS + 0.05
            continue

        # find the arcsecond separation of each candidate pair
        xCOR = SkyCoord(apple_df['RA1'], apple_df['DEC1'], unit='deg')
        yCOR = SkyCoord(apple_df['RA2'], apple_df['DEC2'], unit='deg')
        apple_df['arc_sep'] = xCOR.separation(yCOR).arcsecond

        # if there are more than 50000 candidate pairs, then chunk up the dataframe and calculate convolutional probabilities
        # conv probs are optimally calculated column-wise, but when the column lengths are too long it may cause a memory error
        # this step ensures speed where possible and eases the memory burden of this approach
        merge_length = len(apple_df)
        if merge_length < 50000:
            # calculate the convolutional probability (Cp)
            apple_df['Cp'] = Convdif(z_01, PDF_array[apple_df['ID1'],1:], PDF_array[apple_df['ID2'],1:], dv_lim=max_dv)
        else:
            # loop through and calculate Cp in chunks
            print('Chunking up Cp calculation for pair {0}-{1} into {2} chunks'.format( ID1, ID2, (merge_length//10000)+1 ))
            # initialize an empty array to put Cps into
            Cp_chunks = np.zeros( merge_length )
            N_chunks = merge_length // 10000
            for j in range(N_chunks+1):
                # calculate Cp for this chunk
                Cp_ch = Convdif(z_01, PDF_array[apple_df.loc[ j*10000:(j+1)*10000 , 'ID1'],1:], 
                                PDF_array[apple_df.loc[ j*10000:(j+1)*10000, 'ID2'],1:], dv_lim=max_dv)
                Cp_chunks[j*10000:((j+1)*10000)+1] = Cp_ch[:,0] #shape = [10001,1]
            # now add the chunked Cp values into the df
            apple_df['Cp'] = Cp_chunks
        # if they have 0 probability, throw out
        apple_df = apple_df.loc[ apple_df['Cp'] > 0 ]
        # see what candidates satisfy the matching function and store them in a new df
        apple_df2 = apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < PP_diff_func(Pp)) &
                                                 (apple_df['arc_sep'] > max_R_kpc) & 
                                                 (apple_df['ID1'] != apple_df['ID2']) ].reset_index(drop=True)
        # flag the candidate pairs that satisfy the matching function above, these will be considered in further iterations if 
        #     less than N_controls are in apple_df2
        apple_df.loc[ (np.abs(np.log10(apple_df['Cp']) - np.log10(Pp)) < PP_diff_func(Pp)) & 
                                                 (apple_df['arc_sep'] > max_R_kpc) &
                                                 (apple_df['ID1'] != apple_df['ID2']), 'reuse_flag' ] = 1

        # add pair information:
        apple_df2['P_ID1'] = np.array([ID1]*len(apple_df2), dtype=int)
        apple_df2['P_ID2'] = np.array([ID2]*len(apple_df2), dtype=int)
        apple_df2['Pp'] = np.array([Pp]*len(apple_df2), dtype=float)
        # sort based on the difference function (see paper)
        apple_df2['dif'] = (10*(np.log10(apple_df2['Cp']) - np.log10(Pp))**2 + 
                                (apple_df2['z1'] - z1)**2 + (apple_df2['MASS1'] - M1)**2 + 0.1*(apple_df2['ENV1'] - E1)**2 +
                            (np.log10(apple_df2['SIG1']) - np.log10(SIG1))**2 +
                            (apple_df2['z2'] - z2)**2 + (apple_df2['MASS2'] - M2)**2 + 0.1*(apple_df2['ENV2'] - E2)**2 +
                            (np.log10(apple_df2['SIG2']) - np.log10(SIG2))**2)
        apple_df2.sort_values(by=['dif'], inplace=True, ascending=True, ignore_index=True) # this resets the index

        # we require 6 unique galaxies in 3 galaxy pairs, so drop the repeats (based on higher difference function values)
        apple_df2 = apple_df2.drop_duplicates(['ID1'])
        apple_df2 = apple_df2.drop_duplicates(['ID2'])
        apple_df2 = apple_df2.reset_index(drop=True)

        # finally, if we have inverse pairs (ID1 in pair1 = ID2 in pair 2, etc.) we need to get rid of these as well        
        sort_idx = np.argsort((apple_df2.loc[:,['ID1','ID2']]).values, axis=1)
        arr = np.array(apple_df2.loc[:, ['ID1','ID2']].values)
        arr_ysort = np.take_along_axis(arr, sort_idx, axis=1)
        # unique at axis 0 to get rid of duplicates
        uniq_arr, uniq_idx = np.unique(arr_ysort, return_index=True, return_inverse=False, axis=0)
        # return the unique pairing
        uniq_p = arr[np.sort(uniq_idx)]
        apple_df3 = pd.DataFrame( uniq_p, columns=(apple_df2.loc[:,['ID1','ID2']]).columns )
        apple_df3['pair_str'] = (apple_df3['ID1'].astype(int)).astype(str)+'+'+(apple_df3['ID2'].astype(int)).astype(str)
        # now weed out these duplicate pairs in apple_df2 (i.e., the ones not in apple_df3)
        apple_df2 = apple_df2.loc[ apple_df2['pair_str'].isin(apple_df3['pair_str']) == True ].reset_index(drop=True)
                
        # if N_controls (3 in my case) or more pairs remain, set got equal to true and exit the while loop
        if len(apple_df2) >= N_controls:
            # take the top N_controls and define them as the control_df for this true pair
            control_df = apple_df2.iloc[:N_controls]
            got = True
        else:
            # get an array of str IDs that have been tried already and don't satisfy the matching function
            tried_arr = np.concatenate( (tried_arr, apple_df.loc[ apple_df['reuse_flag'] == 0, 'pair_str' ]) ) 
            # expand search parameter thresholds
            dz = dz + 0.03
            dM = dM + 0.03 
            dE = dE + 1
            dS = dS + 0.05
            # if we are searching at a large mass difference, note to report whether it got matched and to what <-- helpful for tuning
            if dM > 0.6:
                print('dM exceeded 0.6 for pair {0}-{1}'.format(ID1, ID2))
                report = True
                
    # remove pair string to add back in post (can't have mixed data types later when we combine all control df)
    control_df = control_df.drop(columns={'pair_str'})

    # print out the matches for the true pairs that the algorithm had to search beyond 0.6 log(M_solar) for controls
    if report == True:
        print('Poor match to true pair {0}-{1} with Pp={2} found:'.format( ID1, ID2, Pp))
        print(control_df)

    return control_df


### --------------------- ###
### calling main function ###
### --------------------- ###
if __name__ == '__main__':
    main()