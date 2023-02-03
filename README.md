# pair-convolutions

Code repository for the paper “Obscured AGN enhancement in galaxy pairs at cosmic noon: evidence from a probabilistic treatment of photometric redshifts”, S. L. Dougherty, C. M. Harrison, D. D. Kocevski, D. J. Rosario, Monthly Notices of the Royal Astronomical Society, submitted.

## Abstract
Observations of the nearby universe reveal an increasing fraction of active galactic nuclei (AGN) with decreasing projected separation for close galaxy pairs, relative to isolated control galaxies. This implies galaxy interactions play a role in enhancing AGN activity. However, the picture at higher redshift is less established, in part due to limited spectroscopic redshift coverage. Here we combine spectroscopic surveys with photometric redshift probability distribution functions for galaxies in the CANDELS and COSMOS surveys, to produce the largest ever sample of galaxy pairs used in an AGN fraction calculation for cosmic noon (0.5 < $z$ < 3). We present a new technique for assessing galaxy pair probability (based on line-of-sight velocities within ±1000 km s−1) from photometric redshift posterior convolutions and use these to produce weighted AGN fractions. Over projected separations of 5–100 kpc we find no evidence for an enhancement, relative to isolated control galaxies, of X-ray AGN ( $L_X>10^{42}$ erg s $^{−1}$ ) or infrared-selected AGN in major (mass ratios up to 4:1) or minor (mass ratios of 4:1 to 10:1) galaxy pairs. However, defining the most obscured AGN as those detected in the infrared but not in X-rays, we observe a trend of increasing obscured AGN enhancement at decreasing projected separation. The peak enhancement, relative to the isolated controls, is a factor of 2.08 ± 0.61 for separations <25 kpc. If confirmed with improved infrared photometric data (e.g., with _JWST_) and redshifts (e.g., with forthcoming infrared multi-object spectrograph surveys), this would suggest that galaxy interactions play a role in enhancing the most obscured black hole growth at cosmic noon. 

## Architecture
The ultimate goal of this work is to determine whether galaxies with close projected companions are more likely to host AGN than isolated galaxies at high redshift. Due high spectroscopic incompleteness at high redshift, we have devised an algorithm to calculate galaxy pair probabilities in large fields from photometric redshift probability distribution functions on two metrics: (1) how likely they are to have relative line-of-sight velocities within $\pm 1000$ km s $^{-1}$ and (2) how likely galaxies are to be within 100 kpc. These probabilities are then combined into the true pair probability.

The derivation of the relative line-of-sight velocity probability ( $\mathcal{P}_{\Delta V}$ ) is illustrated for three galaxy pairs below (see paper for more details). Briefly, galaxy redshift probability distribution functions ( $P(z)$ ) go through a change of variables to line-of-sight velocity ( $P(V)$ ), which are then convolved to get a probability distribution function of the relative line-of-sight probability ( $P(\Delta V)$ ). The integral of $P(\Delta V)$ within $\pm 1000$ km s $^{-1}$ is taken as the relative line-of-sight probability.

![relative line-of-sight velocity probability derivation](https://github.com/sldough21/pair-convolutions/blob/master/images/git_conv_method.png)

The derivation of the projected separation probability ( $\mathcal{P}_{r}$ ) is illustrated below. In short, we compute the combined redshift probability distribution function by multiplying and normalizing the two $P(z)$ ’s. Then, with a measured angular separation, we perform a change of variables to obtain a projected separation probability distribution function ( $P(r_p)$ ). The integral of this function within certain bin ranges defines the projected separation probability.

![projected separation probability derivation](https://github.com/sldough21/pair-convolutions/blob/master/images/git_Prp_method.png)

### conv_agn_merger.py
This analysis is carried out in **conv_agn_merger.py**. This code identifies projected companions, calculates their true pair probabilities, then identifies control galaxy pairs (in parallel; again see paper for details). In order to run this code, data must be loaded into a pandas DataFrame. This data is loaded in the process_samples() function with the following columns (in no particular order):

- ‘ID’ - photometric catalog ID
- ‘RA’ - right ascension (in degrees)
- ‘DEC’ - declination (in degrees)
- ‘MASS’ - log(stellar mass)
- **‘ZPHOT_PEAK’ - peak photometric redshift $P(z)$ estimate
- **‘SIG_DIFF’ - difference between upper and lower $P(z)$ 68% confidence intervals
- **‘ZSPEC’ - secure spectroscopic redshift.
- **‘ZBEST_TYPE’ - if a galaxy has a secure zspec, then this is ‘s’, otherwise ‘p’
- ‘IRAC_CH1_FLUX’ - IRAC 3.4 um flux (uJy)
- ‘IRAC_CH1_FLUXERR’ - 1sigma channel 1 flux error (uJy)
- ‘IRAC_CH2_FLUX’ - IRAC 4.5 um flux (uJy)
- ‘IRAC_CH2_FLUXERR’ - 1sigma channel 2 flux error (uJy)
- ‘IRAC_CH3_FLUX’ - IRAC 5.8 um flux (uJy)
- ‘IRAC_CH3_FLUXERR’ - 1sigma channel 3 flux error (uJy)
- ‘IRAC_CH4_FLUX’ - IRAC 8.0 um flux (uJy)
- ‘IRAC_CH4_FLUXERR’ - 1sigma channel 4 flux error (uJy)
- **‘FX’ - extrapolated full band (0.5-10 keV) X-ray flux (W/m2)
- **z_type+’_env’ - the environmental density calculated from the similar $P(z)$ convolution probability for all pairs within 1 Mpc. 
  - These can be derived from **environmental_density.py** for a run of just zphot (z_type=’p’) or including zspec (z_type=’ps’). The logic of this code is similar to conv_agn_merger.py, and the exact same DataFrame is required as input.
  - Outputs field+'_environmental_density.parquet', which includes the catalog IDs and the corresponding environmental density of the parent sample.
  - These are heavy calculations, so expect it to take a few days for large fields such as COSMOS.
- Photometric quality flags - the CANDELS and COSMOS photometry have different flags available, see photometric papers for more information. The flags I used are:
  - CANDELS
    - ‘CLASS_STAR’ - SExtractor stellarity parameter (1=star). I choose <0.9 as a conservative cut
    - ‘PHOTFLAG’ - Photometry flag (0 = good photometry, 1 = bright stars and spikes associated with those stars; photometry for objects contaminated by this would be unreliable, 2 = edges of the image as measured from the F160W rms maps)
  - COSMOS
    - ‘LP_TYPE’ - LePhare type (0: galaxy, 1: star, 2: Xray source, -: failure in fit)
    - ‘FLAG_COMBINED’ - 0: clean and inside UVISTA (see COSMOS2020 catalog)
    - ‘CANDELS_FLAG’ - (made myself) True for sources that occupy the CANDELS portion of the COSMOS field

Photometric redshift probability distribution functions must also be input as a 2D numpy array. The first row (beginning on the second column) holds the redshift grid (0-10 in steps of 0.01) and the following rows hold the corresponding probabilities for each galaxy, whose ID is given in the first column. This array is loaded in the pair_pdfs() function.

**These values are being prepared to be released as supplementary material for convenience (including referenced spectroscopic redshifts) and should be released soon. All other input data can be found in the CANDELS and COSMOS photometric catalogs:
- GOODS-S - Guo+13, ApJS, 207, 24
- GOODS-N - Barro+19, ApJS, 243, 22
- EGS - Stefanon+17, ApJS, 229, 32
- UDS - Galametz+13, ApJS, 206, 10
- COSMOS (CANDELS) - Nayyeri+17, ApJS, 228, 7
- COSMOS - Weaver+22, ApJS, 258, 11

And full photometric redshift probability distribution function are available in:
- CANDELS - Kodra+23, ApJ, 942, 36
- COSMOS - Weaver+23, ApJS, 258, 11

When run, the code returns three files for each field:
1. PAIRS_…parquet - the true pair DataFrame
2. APPLES_…parquet - the control pair DataFrame
3. Prp_….fits - projected separation probability distributions for all true pairs

### pam_analysis.ipynb
These files are combined and analyzed in **pam_analysis.ipynb**, a jupyter notebook used for post-processing. We interpret the pair probabilities as weights in calculating a weighted AGN fraction in true galaxy pairs, which we compare to that of the control galaxy pairs to find AGN enhancement in bins of projected separation. This notebook holds many flexible tools and diagnostic outputs for evaluating the output of conv_agn_merger.py, such as measuring the effectiveness of control galaxy selection with a Q-Q plot and calculating AGN enhancements for specific subsample of galaxies (e.g., high redshift, high mass, X-ray luminous AGN; all of this can be done in post). Final results are tabulated into DataFrames and displayed in plots (as shown in paper).

# Requirements
The code was developed using Python 3.8.0, and versions of all relevant python libraries can be found in requirements.txt.

# References
If you use this code, please cite our paper: “Obscured AGN enhancement in galaxy pairs at cosmic noon: evidence from a probabilistic treatment of photometric redshifts”, S. L. Dougherty, C. M. Harrison, D. D. Kocevski, D. J. Rosario, Monthly Notices of the Royal Astronomical Society, submitted.



