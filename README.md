> # Total Blind Search for exoplanet characterization #
>
> Total Blind Search is a package written in Python 3.7 to perform an automatic survey for atomic species on any qualified transition
in exoplanet spectral time series, using the transit spectroscopy technique. It has multiproccesing implemented, and it had been tested 
using data from UVES@VLT, HARPS@LaSilla, HDS@Subaru and HiRes@Keck. It can also be used to search directly for a particular atomic species.
>
> The main class of the code is TBS.py, in which the steps of the transit spectroscopy algorithm are implemented. MultiTBS.py runs TBS using multiprocesing, 
and UpdtReport.py performs an automatic PDF report to summarize the atomic species analyzed. On the other hand, functions_database.py is a bank of useful functions 
ocuppied in the main class, and Manage_plots.py is useful to managing the graphics in the python multiprocessing mode, since it seems that matplotplib does not get 
along well with the latter method. **That is why this code does not work for matplotlib >3. Version 2.2.4 is strongly recommended.**
>
> ## Methods ##
> This algorithm is designed to run in a specific order of the methods. The callable main steps of TBS are the following, in the correct order:
> * check_epoch: It reads the first epoch of the spectral data files, and extracts the local flux near to each transition of the spectral line list. It also qualifies transitions as analyzable or not, comparing them to a gaussian fit of the feature.
> * extract_lines: It extracts all the epochs of the qualified transitions. It performs a Doppler correction using the radial velocity provided as a parameter, and it also performs a misalignment correction using the first epoch feature as a reference.
> * telluric_corrections: It makes a naïve telluric correction, fitting the airmass.
> * snellen: It calculates the relative absorption along time, using the band provided as a parameter. It is based on [Snellen et al, 2008](https://ui.adsabs.harvard.edu/abs/2008A%26A...487..357S/abstract).
> * redfield: It makes a bootstrap analysis to calculate the confidence of the absorption. It is based on [Redfield et al, 2008](https://ui.adsabs.harvard.edu/abs/2008ApJ...673L..87R/abstract).
> * summary_plots: It performs a summary graphic of all the processes of the code.
> 
> ### Definition of detection (in detected_alert method) ###
> > Our definition of detection ensures that a relative absorption will have 3 sigmas of confidence, and that the center of the absorption will not be inside of one of the noise histograms. Then a transition is defined as detected if:
> >  - <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{In-Out}&space;>&space;3" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{In-Out}&space;>&space;3" title="\sigma_{In-Out} > 3" /></a>
> > - <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{rel}&space;>&space;1" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{rel}&space;>&space;1" title="\sigma_{rel} > 1" /></a>
> <br> Where <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{In-Out}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{In-Out}" title="\sigma_{In-Out}" /></a> is the distance between the null hypothesis and the center of the In-Out histogram, in times of the width of this histogram and <a href="https://www.codecogs.com/eqnedit.php?latex=\sigma_{rel}" target="_blank"><img src="https://latex.codecogs.com/gif.latex?\sigma_{rel}" title="\sigma_{rel}" /></a> is the distance between the null hypothesis and the center of the In-Out histogram, but normalized for the maximum width between the In-In and Out-Out histograms.

> 
> ## Initialization ##
> The algorithm has many customizable parameters, but the more important and neccesary to change for each planet are the following:
> * orders: The number of orders of the échelle spectra data.
> * indexFWHM: The typical number of pixels lenght of the FWHM of each feature.
> * error_gain: The gain of the instrument, used to calculate Poisson statistics. It is not necessary when the data includes the error of the flux, which is the case of UVES, for instance.
> * dtransit: The duration of the transits, in days.
> * dlambda_analyze: The length of the bands, in angstrom, to perform the analysis along time [(Snellen et al, 2008)](https://ui.adsabs.harvard.edu/abs/2008A%26A...487..357S/abstract), and to measure the confidence of the possible detection [(Redfield et al, 2008)](https://ui.adsabs.harvard.edu/abs/2008ApJ...673L..87R/abstract).
> * t_transit: The mid transit time in JD.
> * period: The period of the planet, in days.
> * do_1stdopp & radial_velocity: True if a first correction is necessary, using the radial_velocity to calculate a z.
> * dlt_epochs_again: An array used to delete the desired epochs. [0,1], for instance, will delete the first and second epochs.
> * tell_only_out: This boolean parameter is used to decide if the telluric corrections are only performed using the out-transit epochs.
> * merge_epoch_as: One of the ways of the code to make a time binning, combining different epochs. This one is defined by the user, using an array of strings. For example, ['0:2'] will produce only one combined epoch of the first to second epochs.
> * n_epoch_merge & smart_merge: the first defines how many epochs to merge automatically, and the second is whether to use a function that combines them dropping the epochs that lie between the in-transit and out-transit time, until the total of epochs is divisible by n_epoch_merge. This way of time binning is recommended when the exposure time is uniform.
> * norm_width: One of the key parameters of the code. This number times the FWHM will be the kernel of the median filter used to find the continuum for the normalization. Empirically, we found that in HDS the optimized value was 5, **but it should be scaled to other instruments.**
> * norm_high: As the previous parameter, this one is fundamental in the normalization, as it defines how many Poisson statistic sigmas will be defined above and below the median filter to include pixels to find the continuum. 3 is recommended.
> * wave_binning: The number of pixels to bin along wavelength. 0 and 1 will make no binning.
> * test_percent & test_boundary: This parameter is used to qualify transition for the posterior steps of analysis. 0.7 means that the 70% of pixels that are used to find and fit the continuum should be closer than test_boundary from the median of the latter pixels.
> * telescope: the instrument of the observations. This parameter is used by the code to know how to extract the data.
> 
> ## How to run ##
> The way to run TBS without multiprocessing is detailed in tutorial.ipynb

