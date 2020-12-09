# -*- coding: latin-1 -*-
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
from os import listdir
from astropy.io import fits
#from astropy import units as u
#from specutils.io import read_fits
from pyspeckit import Spectrum
from scipy.ndimage.filters import median_filter as mf
from scipy.optimize import leastsq
from scipy.optimize import curve_fit
from scipy.interpolate import InterpolatedUnivariateSpline as uSpline
from matplotlib import gridspec
import uncertainties.unumpy as unumpy
import pandas as pd
from Manage_plots import *
import warnings
warnings.filterwarnings('ignore')



"""
Functions for reading
"""
def no_points(string):
    aux_str = string.split('.')
    new_str = aux_str[0]
    for j in range(1, len(new_str)):
        new_str = new_str + '_point_' + aux_str[j]

def list_archives(strAdress, format = ".fits", key = ''):
    database = []
    for archive in listdir(strAdress):
        if format in archive and key in archive:
            aux = strAdress+'/'+archive
            database.append(aux)
    return database


def extract_from_header(files, str_header = 'MJD', sort=True):
    """
    it sorts the database by time
    """
    array = []
    for file in files:
        header = fits.getheader(file)
        array.append(float(header[str_header]))
    array = np.array(array)
    files = np.array(files)
    if not sort:
        return files, array
    else:
        ind_sort = np.argsort(array)
        return files[ind_sort], array[ind_sort]



def get_lines(lines_adress, min_depth = -1, max_depth = -1, low_limit = 5000, up_limit = 8000, min_loggf = -100,
              max_loggf = 100, delete_near = False, sort_db = True, near_tolerance = 0.2,
              elem_key = '', elem_delete =''):
    db = list_archives(lines_adress, '.dat')
    """
    cols = ['Nr', 'Elem', 'Ion', 'Lam0_air', 'log(gf)', 'ElowEv', 'Rad',
                'Stark', 'Waals', 'Lande', 'Line_depth(%)', 'Weq(mA)',	'Refs.']
    """
    filter = {}
    elems = []
    lines = []
    loggfs = []
    depths = []
    previous_line = 0
    for i in reversed(range(len(db))):
        df = np.genfromtxt(db[i], skip_header=2, usecols = (1,2,3,4,10), dtype=("|S2", int, float, float, float))
        for j in range(len(df)):
            elem, ion, line, loggf, depth = df[j]
            if np.any(sp.isnan(line)):
                print ()
                print (db[i])
                print (previous_line)
            previous_line = line
            #print (elem.decode('UTF-8'), ion, type(elem.decode('UTF-8')), type(ion))
            filter[line] = [elem.decode('UTF-8')+str(ion), loggf, depth]

    for line in list(filter.keys()):
        lines.append(line)
        aux = filter[line]
        elems.append(aux[0])
        loggfs.append(aux[1])
        depths.append(aux[2])

    lines = np.array(lines)
    elems = np.array(elems)
    loggfs = np.array(loggfs)
    depths = np.array(depths)


    if min_depth == -1:
        mask1 = np.array([True for depth in depths])
    else:
        mask1 = np.array([depth > min_depth for depth in depths])
    if max_depth == -1:
        mask2 = np.array([True for depth in depths])
    else:
        mask2 = np.array([depth < max_depth for depth in depths])

    if low_limit == -1 and up_limit == -1:
        mask_lines = np.ones((len(depths)), dtype=bool)
    else:
        mask_lines = np.array([low_limit <= line and line <= up_limit for line in lines])

    if elem_key != '':
        mask_elem = np.array([elem_key in elem for elem in elems])
    else:
        mask_elem = np.ones((len(elems)), dtype=bool)

    if elem_delete != '':
        mask_elem_del = np.array([not elem_delete in elem for elem in elems])
        mask_elem = mask_elem_del*mask_elem

    mask = mask1 * mask2 * mask_lines * mask_elem

    mask_loggf = np.array([min_loggf <= loggf and loggf <= max_loggf for loggf in loggfs])
    mask = mask * mask_loggf

    if delete_near:
        mask3 = [True]

        aux_mask = [abs(lines[j-1] - lines[j]) < near_tolerance for j in range(1, len(lines))]
        mask3.extend(aux_mask)
        mask3 = np.array(mask3)
        mask = mask * mask3

    lines = lines[mask]
    elems = elems[mask]
    loggfs = loggfs[mask]
    depths = depths[mask]
    if sort_db:
        sorted_inds = np.argsort(lines)
        lines = lines[sorted_inds]
        elems = elems[sorted_inds]
        loggfs = loggfs[sorted_inds]
        depths = depths[sorted_inds]

    return lines, elems, loggfs, depths


def get_linedb(str_adress, elem_key= '', elem_delete = '', depth_lims = [], loggf_lims = [], sort_db = True,
              wv_lims = [], mode = 'spectroweb', key = '', _format = ''):
    if mode == 'spectroweb':
        if _format == '':
            _format = '.dat'
        archives = list_archives(str_adress, _format , key = key)
        linedb = []
        str_elems = []
        loggfs = []
        for archive in archives:
            usecols =['Nr', 'Elem', 'Ion', 'Lam0_air', 'log(gf)', 'ElowEv', 'Rad', 'Stark', 'Waals', 'Lande', 'Line_depth(%)', 'Weq(mA)', 'Refs.']
            df = pd.read_table(archive, header =1, sep = '\s+', index_col = 0, usecols=usecols)
            if len(depth_lims) > 0:
                depths = df['Line_depth(%)'].values
                mask = np.array([depth_lims[0] <= aux and aux <=depth_lims[1] for aux in depths])
                df = df[mask]
            if len(loggf_lims) > 0:
                loggfs_ = df['log(gf)'].values
                mask = np.array([loggf_lims[0] <= aux and aux <=loggf_lims[1] for aux in loggfs_])
                df = df[mask]
            if len(wv_lims) > 0:
                wvs = df['Lam0_air'].values
                mask = np.array([wv_lims[0] <= aux and aux <=wv_lims[1] for aux in wvs])
                df = df[mask]
            if elem_delete != '' or elem_key != '':
                elems = df['Elem'].values
                mask = np.array([not elem_delete in aux and elem_key in aux for aux in elems])
                df = df[mask]
            if len(df)>0:
                linedb.extend(df['Lam0_air'].values)
                loggfs.extend(df['log(gf)'].values)
                elems = df['Elem'].values
                ions = df['Ion'].values
                str_elems.extend([str(elems[i])+str(ions[i]) for i in range(len(elems))])
        linedb = np.array(linedb)
        str_elems = np.array(str_elems)
        loggfs = np.array(loggfs)
        if sort_db:
            sorted_inds = np.argsort(linedb)
            linedb = linedb[sorted_inds]
            str_elems = str_elems[sorted_inds]
            loggfs = loggfs[sorted_inds]
        return linedb, str_elems, loggfs
    elif mode == 'vald':
        if _format == '':
            _format = '.txt'
        archives = list_archives(str_adress, _format, key=key)
        linedb = []
        str_elems = []
        loggfs = []
        for archive in archives:
            df = pd.read_csv(archive, sep=",", header = 1)
            df.columns = ['Elm-Ion', 'wv_air', ' excit', 'log_gf', 'rad', 'stark', 'waals', 'lande_factor', 'References']
            if len(loggf_lims) > 0:
                df = df[df.log_gf>=loggf_lims[0]]
                df = df[df.log_gf<=loggf_lims[1]]
            if len(wv_lims)>0:
                df = df[df.wv_air >= wv_lims[0]]
                df = df[df.wv_air <= wv_lims[1]]
            if elem_delete != '' or elem_key != '':
                elems = df['Elm-Ion'].values
                mask = np.array([not elem_delete in aux and elem_key in aux for aux in elems])
                df = df[mask]
            if len(df)>0:
                linedb.extend(df['wv_air'].values)
                loggfs.extend(df['log_gf'].values)
                str_elems.extend([aux.split(' ')[0] + str(aux.split(' ')[1]) for aux in df['Elm-Ion'].values])
        if len(str_elems) > 0:
            str_elems = np.array([aux[1:-1] for aux in str_elems])
        return linedb, str_elems, loggfs

def get_all_linedb(folders = ['spectroweb_lines', 'vald_lines'], modes = ['spectroweb', 'vald'],
                  wv_lims = [], elem_key = '', elem_delete = '', loggf_lims = [], formats = ['.dat', '.txt'],
                  folder_key = ''):
    linedb = []
    str_elems = []
    loggfs = []
    for mode, folder, _format in zip(modes, folders, formats):
        lines, elems, loggf = get_linedb(folder, _format = _format, mode = mode, wv_lims = wv_lims, elem_key = elem_key,
                                         elem_delete = elem_delete, loggf_lims = loggf_lims, key = folder_key)
        linedb.extend(lines)
        str_elems.extend(elems)
        loggfs.extend(loggf)
    return linedb, str_elems, loggfs


def openFile(strFile, order, doPlot = False, orders_together = False, using_keck = False,
             using_uves = False):
    error = []
    if not using_keck:
        if using_uves:
            hdul = fits.open(strFile)
            #header = fits.getheader(strFile)
            wave = np.array(hdul[1].data[0]['WAVE'])
            flux = np.array(hdul[1].data[0]['FLUX'])
            error = np.array(hdul[1].data[0]['ERR'])
        else:
            specFile = Spectrum(str(strFile))
            if not orders_together:
                wave = np.array(specFile[order].xarr)
                flux = np.array(specFile[order].data)
            else:
                wave = np.array(specFile.xarr)
                flux = np.array(specFile.data)

    else:
        hdul = fits.open(strFile)
        header = fits.getheader(strFile)
        if order < header['NAXIS2']:
            if order + 1 > 9:
                aux_str = str(order+1)
            else:
                aux_str = '0'+str(order+1)
            to_header = ['WV_0_'+aux_str, 'WV_4_'+aux_str]
            coefs0 = header[to_header[0]].split(' ')
            coefs4 = header[to_header[1]].split(' ')
            coefs0.extend(coefs4)
            coefs = []
            for str_coef in coefs0:
                if str_coef != '' and str_coef != ' ' and str_coef != '  ':
                    coefs.append(float(str_coef))

            flux = hdul[0].data[order]
            pixels = range(len(flux))
            wave = []
            for pix in pixels:
                aux_pix = 0
                for j in range(len(coefs)):
                    aux_pix = aux_pix + coefs[j]*pix**j
                wave.append(aux_pix)
        else:
            flux = []
            wave = []
            print ('This order was empty', order)

    if doPlot:
        plt.plot(wave, flux)
        plt.show()
    return wave, flux, error


def extract_array(wave, flux, lineWV, N, gain, binning = 0, error = []):
    """
    this function extracts a subarray of a larger array
    """
    l = len(wave)
    if binning != 0:
        N = binning*N

    index = np.argmin(abs(wave-lineWV))
    indexLow = int(index - N/2)
    indexHigh = int(index + N/2)


    if indexHigh >= l:
        dif = indexHigh - (l - 1)
        indexLow = indexLow - dif
        indexHigh = l - 1

    elif (indexLow < 0):
        dif = abs(indexLow)
        indexHigh = indexHigh + dif
        indexLow = 0

    new_wave = wave[indexLow : indexHigh]
    new_flux = flux[indexLow : indexHigh]
    if len(error) == 0:
        error = np.array([np.sqrt(aux / gain) for aux in new_flux])
    else:
        error = error[indexLow:indexHigh]

    if binning != 0:
        uflux = unumpy.uarray(new_flux, error)
        aux_wave = []
        aux_flux = []
        for i in range(0, len(new_wave), binning):
            aux_wave.append(np.mean(new_wave[i:i+binning]))
            aux_flux.append(np.mean(uflux[i:i+binning]))
        new_wave = np.array(aux_wave)
        new_flux = np.array([aux.n for aux in aux_flux])
        error = np.array([aux.s for aux in aux_flux])

    ind_line = np.argmin(np.abs(new_wave-lineWV))

    return new_wave, new_flux, error, ind_line


def extract_array_with_mask(array, ind_center, N):
    """
    this function extracts a subarray of a larger array an returns a mask
    """
    l = len(array)

    indexLow = int(ind_center - N/2)
    indexHigh = int(ind_center + N/2)
    center_array = array[ind_center]

    if indexHigh >= l:
        dif = indexHigh - (l - 1)
        indexLow = indexLow - dif
        indexHigh = l - 1

    elif(indexLow < 0):
        dif = abs(indexLow)
        indexHigh = indexHigh + dif
        indexLow = 0

    mask = np.array(range(indexLow, indexHigh))
    new_array = array[indexLow:indexHigh]
    new_ind_center = np.argmin(np.abs(new_array-center_array))

    return mask, new_ind_center

"""
Functions for normalization
"""



def normalization_filter(y, errP, indexFWHM, min_pixs = 4, high = 3, width = 1.25):
    """
    this function takes an array and its error to make a median filter
    and discard all data up and down the limits made of the median filter
    +- high*errP
    """

    y0 = np.copy(y)
    half = mf(y, size = int(width*indexFWHM))
    vertical = high*errP
    higher = half + vertical
    lower = half - vertical
    filter_mask = np.array([(lower[i] < y0[i]) and (y0[i] < higher[i]) for i in range(len(y))])

    if min_pixs > 1:
        new_mask = np.copy(filter_mask)
        for h in range(2, min_pixs):
            aux_mask = []
            for i in range(len(filter_mask)):
                if i < min_pixs - 1:
                    aux_mask.append(False)
                else:
                    aux = new_mask[i - h + 1]
                    aux_mask.append(aux)
            aux_mask = np.array(aux_mask)
            new_mask = aux_mask*new_mask
        better_mask = new_mask
    else:
        better_mask = filter_mask

    result = y0[better_mask]

    return result, higher, half, lower, better_mask

def f_model_pol3(params, x):
	"""
	This is an order 3 polynomial
	"""
	c0, c1, c2, c3 = params
	return c0 + c1*x + c2*x**2 + c3*x**3

def argument_chi2_pol3(params, xarray, yarray):
	return yarray - f_model_pol3(params, xarray)


def normalize(x, y, y_error, indexFWHM, min_pixs, high, width, polyorder=3):
    y0 = np.copy(y)
    errP0 = np.copy(y_error)
    yfiltered, higher, half, lower, filter_mask = normalization_filter(y, y_error, indexFWHM,
                                                                        min_pixs = min_pixs, high = high, width = width)
    """
    m, n = guess_lineal_params(x, y)
    resultPol = leastsq(argument_chi2_pol3, [sp.median(y), m, 0,0] , args = (x[filter_mask], yfiltered))
    polynomial = f_model_pol3(resultPol[0], x)
    """

    result = np.array([])
    norm_error = np.array([])
    polynomial = []
    if np.all(filter_mask == False):
        message = 'Empty after median filter'
    else:
        message = 'ok'
        resultPol = np.polyfit(x[filter_mask], yfiltered, polyorder)
        polynomial = np.poly1d(resultPol)(x)
        for i in range(len(y0)):
            aux = y0[i]*(polynomial[i])**(-1)
            auxError = errP0[i]*(polynomial[i])**(-1)
            result = np.append(result, aux)
            norm_error = np.append(norm_error, np.abs(auxError))

    return result, norm_error, polynomial, higher, half, lower, filter_mask, message


"""
Functions for Doppler corrections
"""
def f_model_lineal(params, x):
    m, n = params
    return m*x + n


def argument_chi2_lineal(params, xarray, yarray):
    return yarray - f_model_lineal(params, xarray)


def fit_pol1(x, y, p0 = [1, 0], do_plt = False, figname = 'fit.pdf'):
    resultLineal = leastsq(argument_chi2_lineal, p0, args = (x, y))
    params = resultLineal[0]
    y_fit = f_model_lineal(resultLineal[0], x)
    return params, y_fit


def guess_lineal_params(x, y):
    l = len(y)
    indMax = np.argmax(y)
    m = (y[l-1]-y[0])/(x[l-1]-x[0])
    n = y[indMax]
    return m, n


def doppler_fit(waveLab, waveObs, do_plt = True):
    """
    this function takes an array of real wavelengths and another of
    observed wavelengths to return the z of doppler equation
    """

    params = np.polyfit(waveObs, waveLab, 1)
    fit = params[0]*waveLab + params[1]


    return params[0] - 1, to_plot


def shift_array(x, y, xRef, dx, alty=None):
    spline = uSpline(x, y, k = 1)
    shiftedy = spline(xRef + dx)
    if alty is not None:
        spline = uSpline(x, alty, k = 1)
        return shiftedy, spline(xRef + dx)
    return shiftedy


def first_doppler_corr(wave_obs, flux_obs, zdopp, interpolate=False):

    if zdopp==0:
        return wave_obs, flux_obs

    else:
        if interpolate:
            corrected_wave = wave_obs / (zdopp + 1)
            dwave = wave_obs - corrected_wave
            real_flux = shift_array(wave_obs, flux_obs, corrected_wave, dwave)
        else:
            corrected_wave = wave_obs / (zdopp + 1)
            real_flux = flux_obs
        return corrected_wave, real_flux


def normalized_dot(array1, array2):
    """
    this function calculates the dot product of two arrays and normalizes
    it by the product of their norms
    """
    norm1 = np.linalg.norm(array1)
    norm2 = np.linalg.norm(array2)
    Norm = norm1*norm2
    dotProduct = np.dot(array1, array2)
    return  dotProduct / Norm


def second_doppler_corr(wave_ref, real_wave_ref, flux_ref, wave_obs, flux_obs, line, order, inds_plt, wave_order, flux_order,
                        ind_line, j, deltaWmax=10, Nshifts=20, doplt = False, str_adress = 'default', out_folder = 'images/',
                        alty=None):
    '''
    this function takes a flux array of reference and adjust another one
    to make a doppler correction
    '''
    shifts = np.linspace(-deltaWmax, deltaWmax, Nshifts)
    dot_product = [normalized_dot(flux_ref, shift_array(wave_obs, flux_obs, wave_ref, shft)) for shft in shifts]
    max_prod = np.max(dot_product)
    best_shift = shifts[np.argmax(dot_product)]
    if alty is not None:
        real_flux, shifted_alty = shift_array(wave_obs, flux_obs, real_wave_ref, best_shift, alty=alty)
    else:
        real_flux = shift_array(wave_obs, flux_obs, real_wave_ref, best_shift)
        shifted_alty = None
    plt.close('all')
    fig = plt.figure()
    aux_name = str(int(line)) + '_ind_' + str(j) + '_ord_' + str(order) + '_' + str(inds_plt[1])
    file_name = out_folder + str_adress + '/extract_lines/' + aux_name + '.pdf'
    if doplt:
        N_order = 300
        hor_line = np.ones(len(wave_ref))
        bg_color = 'floralwhite'
        gs = gridspec.GridSpec(1,2, width_ratios=[1, 1.5], wspace=0.3)
        gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[1], hspace = 0.01)
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs[0], hspace = 0.2)
        ax_shift = fig.add_subplot(gs1[0], facecolor=bg_color)
        ax_shift.plot(wave_ref, flux_ref, color = 'darkorange', linewidth = 1.5, alpha = 0.6)
        ax_shift.plot(real_wave_ref, flux_obs, color = 'royalblue', linewidth = 1.2)
        ax_shift.plot(real_wave_ref, real_flux, color = 'orangered', linewidth = 0.6)
        ax_shift.axvline(x=real_wave_ref[ind_line], color='deepskyblue', linestyle='--', linewidth=0.7)
        ax_shift.plot(wave_ref, hor_line, '--', color = 'violet', linewidth = 0.7)

        ax_ref = fig.add_subplot(gs1[1], facecolor = bg_color, sharex = ax_shift)
        ax_ref.plot(wave_ref, flux_ref, color = 'darkorange')
        ax_ref.axvline(x=real_wave_ref[ind_line], color = 'deepskyblue', linestyle='--', linewidth=0.7)
        ax_ref.plot(wave_ref, hor_line, '--', color='violet', linewidth = 0.7)
        ax_ref.set_xlabel('Reference Wavelength')
        ax_ref.set_ylabel('Reference Flux')

        ax_factor = fig.add_subplot(gs2[1], facecolor=bg_color)
        ax_factor.plot(shifts, dot_product, color = 'navy', linewidth = 0.7)
        ax_factor.set_xlabel(r'shifts [$\AA$]')
        ax_factor.text(0.45, 0.1, str(best_shift), verticalalignment='center', horizontalalignment='left',
                      transform=ax_factor.transAxes, style='italic',
                      color='darkred', fontsize=6, bbox={'facecolor': 'navajowhite', 'boxstyle': 'round',
                                                         'alpha': 0.5, 'pad': 1, 'edgecolor': 'black'})

        ax_order = fig.add_subplot(gs2[0], facecolor = bg_color)
        l = len(wave_ref)
        dw = wave_ref[1]-wave_ref[0]
        begin = wave_ref[0] - N_order*dw
        end = wave_ref[l-1] + N_order*dw
        mask_order = np.array([begin<= wlth and wlth <= end for wlth in wave_order])
        ax_order.plot(wave_order[mask_order], flux_order[mask_order], color = 'navy', linewidth = 0.5)
        ax_order.axvspan(wave_ref[0], wave_ref[l-1], alpha = 0.5, color = 'coral')

        plt.setp(ax_order.get_yticklabels(), visible = False)
        plt.setp(ax_shift.get_xticklabels(), visible = False)
        plt.suptitle(r'Extract line for ' + str(line))
    fig_name = file_name

    return real_flux, shifted_alty, fig, fig_name, max_prod


"""
Functions for validity tests for lines
"""
def f_model_gauss_and_constant(params, x):
    A, mu, sigma, c = params
    argument = ((x-mu)**2)/(2*sigma**2)
    gauss = A*np.exp(-argument)
    return -gauss + c


def argument_chi2_gauss_and_constant(params, xarray, yarray):
    return yarray - f_model_gauss_and_constant(params, xarray)


def f_model_gauss_and_line(params, x):
    A, mu, sigma, m, n= params
    argument = ((x-mu)**2)/(2*sigma**2)
    gauss = A*np.exp(-argument)
    return -gauss + m*x + n


def argument_chi2_gauss_and_line(params, xarray, yarray):
    return yarray - f_model_gauss_and_line(params, xarray)


def tests_filter(x, y, indLine, near, times_near = 2):
    """
    it makes a median filter near the indLine
    """
    near = int(near)
    y0 = np.copy(y)
    l = len(y)
    i = 0
    result = mf(y, size = int(times_near*near))
    while (i < near):
        if ((indLine + i) >= l) or (indLine - i) < 0:
            break
        result[indLine - i] = y0[indLine - i]
        result[indLine + i] = y0[indLine + i]
        i = i + 1
    return result


def guess_sigma(x, y, A, fraction = 0.1, max_c = 100):
    """
    it guess a sigma for a gaussian fit
    """
    indChange = np.array([])
    c = 0
    while (len(indChange) == 0):
        dmin = fraction*A
        for i in range(len(y)-1):
            dy=(y[i+1]-y[i])
            if (dy > dmin):
                indChange=np.append(indChange, i)
        fraction = 0.5*fraction
        c = c + 1
        if c > max_c:
            break
    if c > max_c:
        sigma = 0
    else:
        l=len(indChange)
        i1 = int(indChange[0])
        i2 = int(indChange[l-1])
        sigma = x[i2]-x[i1]
    return sigma

def guess_gaussian_params(x, y):
    indmin = np.argmin(y)
    indmax = np.argmax(y)
    medianY = sp.median(y)
    minY = y[indmin]
    A = medianY - minY
    mu = x[indmin]
    sigma = guess_sigma(x, y, A)
    C = medianY
    return A, mu, sigma, C


def gaussian_and_constant_fit(x, y, ind_line, near, times_near = 2):
    newY = tests_filter(x, y, ind_line, near, times_near= times_near)
    p0 = guess_gaussian_params(x, newY)
    resultGauss = leastsq(argument_chi2_gauss_and_constant, p0, args = (x, newY))
    return resultGauss[0], f_model_gauss_and_constant(resultGauss[0], x), newY


def gaussian_and_lineal_fit(x, y, ind_line, near):
    newY = tests_filter(x, y, ind_line, near)
    A, mu, sigma, C = guess_gaussian_params(x, newY)
    p0 = A, mu, sigma, 0, C
    resultGauss = leastsq(argument_chi2_gauss_and_line, p0, args = (x, newY))
    return resultGauss[0], f_model_gauss_and_line(resultGauss[0], x), newY


def amplitude_test(y, A, ind_min, dA= 0.06, info = False):
    minY = y[ind_min]
    #medianY = sp.median(y)
    amplitudeY = abs(1 - minY)
    if (info):
        print ("Real amplitude, amplitude of fit, delta A:")
        print (amplitudeY, A, dA)
        print()
    emp_amp = amplitudeY - A
    return abs(emp_amp) < dA, emp_amp


def center_test(x, mu, ind_min, dMu = 0.06, info = False):
    muX = x[ind_min]
    if info:
        print ("Real mu, mu of fit, delta Mu:")
        print (muX, mu, dMu)
        print ()
    emp_mu = abs(muX-mu)
    return emp_mu < dMu, emp_mu


def residuals_test(residuals, percent = 0.7, boundary = 0.009, info = False):
    l = len(residuals)
    k = (np.abs(residuals)<boundary).sum()
    #print (k, l, k/l)
    if info:
        print ("Error percent with respect to fit:")
        print (str(100*k/l), '%')
        print ()
    #if k/l > (1 - percent):
    return k/l >= percent, k/l


"""
Telluric corrections
"""

def label_transit(tarray, dtransit = 0.1, period = 3.5, t_transit = 2500000,
                  max_t_len = 1):
    n = 0
    t0 = tarray[0]
    tf = tarray[len(tarray)-1]
    t_transit = t_transit - 2400000.5
    t_center = t_transit

    if t0 > t_transit:
        while (t0 > t_center):
            n = n + 1
            t_center = t_center + period
    else:
        while (t_center > tf):
            n = n + 1
            t_center = t_center - period

    indexLow = (tarray < t_center - 0.5*dtransit).sum()
    index_peri = (tarray < t_center).sum()
    indexHigh = (tarray < t_center + 0.5*dtransit).sum()

    is_out = np.array([], dtype = bool)
    for i in range(len(tarray)):
        if (indexLow < i < indexHigh):
            is_out = np.append(is_out, False)
        else:
            is_out = np.append(is_out, True)

    if max_t_len != 0:
        len_mask = np.array([t_center - max_t_len/2 <= taux and taux <= t_center + max_t_len/2
                    for taux in tarray])
        tarray = np.array(tarray)
        tarray = tarray[len_mask]
        is_out = is_out[len_mask]
    else:
        len_mask = np.ones((len(tarray)), dtype=bool)

    t_start = t_center - dtransit/2

    return tarray, is_out, index_peri, len_mask, t_start

def f_model_exp(x, k, A):
    return A*np.exp(k*x)


def argument_chi2_exp(params, xarray, yarray):
    return yarray - f_model_exp(params, xarray)


def fit_exp(x, y, p0 = [1, 0]):
    result_exp = leastsq(argument_chi2_exp, p0, args = (x, y))
    return result_exp[0], f_model_exp(result_exp[0], x)



def correct_spectra(s_i, s_ref, arrayM, flux):
    newFlux = np.array([])
    for i in range(len(flux)):
        toDivide = np.exp(arrayM[i]*(s_i - s_ref))
        auxFlux = flux[i]/toDivide
        newFlux = np.append(newFlux, auxFlux)
    return newFlux


"""
Functions for analize lines
"""

""""
def select_zone(wave, flux, beginWV, endWV):
    begin = (wave <= beginWV).sum()
    end = (wave >= endWV).sum()
    end = len(wave) - end
    return wave[begin:end:1], flux[begin:end:1]
"""

def select_zone(wave, flux, beginWV, endWV):
    mask = np.array([beginWV <= wlth and wlth < endWV for wlth in wave])
    return wave[mask], flux[mask]

def calculate_signal_snellen(wave, flux, error, lambda0, dlambda, mask, tests= True,
                             alpha_mid=2./3., alpha_side=1./2.):
    flux_left, flux_mid, flux_right, tests, message = calculate_bands(wave, flux, lambda0, dlambda, mask, mean = False,
                                                    tests=tests, alpha_mid=alpha_mid, alpha_side=alpha_side)
    error_left, error_mid, error_right, _tests, _message = calculate_bands(wave, error, lambda0, dlambda, mask, mean = False,
                                                                           tests = False)
    fleft = unumpy.uarray(flux_left, error_left)
    fmid = unumpy.uarray(flux_mid, error_mid)
    fright = unumpy.uarray(flux_right, error_right)
    ftransit = 2*np.mean(fmid)/(np.mean(fright) + np.mean(fleft))
    return ftransit.n, ftransit.s, tests, message
    

def calculate_bands(wave, flux, lambda0, dlambda, mask, mean = True, tests = False, alpha_mid=2./3.,
                    alpha_side = 1./2., do_print = False):
    low_left = lambda0 - 1.5*dlambda
    high_left = lambda0 - 0.5*dlambda
    low_mid = high_left
    high_mid = lambda0 + 0.5*dlambda
    low_right = high_mid
    high_right = lambda0 + 1.5*dlambda
    N = np.sum([low_left <= wv and wv < high_left for wv in wave])
    wave = wave[mask]
    flux = flux[mask]
    if do_print:
        print ()
        print (wave, flux)
    wave_left, flux_left = select_zone(wave, flux, low_left, high_left)
    wave_mid, flux_mid = select_zone(wave, flux, low_mid, high_mid)
    wave_right, flux_right = select_zone(wave, flux, low_right, high_right)
    if do_print:
        print (flux_mid, flux_left, flux_right)
        print (wave_mid, wave_left, wave_right)

    test = True
    message = 'All ok'
    if tests:
        n_left = len(flux_left)
        n_mid = len(flux_mid)
        n_right = len(flux_right)

        if n_mid < alpha_mid*N:
            test = False
            message = 'Too few N_mid'
        elif n_left < alpha_side*N or n_right < alpha_side*N:
            test = False
            message = 'Too few N_side'

    if mean:
        f_left = np.mean(flux_left)
        f_mid = np.mean(flux_mid)
        f_right = np.mean(flux_right)
    else:
        f_left = flux_left
        f_mid = flux_mid
        f_right = flux_right

    return f_left, f_mid, f_right, test, message


def average_outin(flux_array, is_out):
    flux_out = flux_array[is_out]
    is_in = np.array([not b for b in is_out])
    flux_in = flux_array[is_in]
    average_in = np.mean(flux_in)
    average_out = np.mean(flux_out)
    return average_out, average_in


def average_flux(integratedFlux, isOut):
    l = len(isOut)
    average_out, average_in = average_outin(integratedFlux, isOut)
    average_array = np.array([])
    for i in range(l):
        if isOut[i]:
            aux = average_out
        else:
            aux = average_in
        average_array = np.append(average_array, aux)

    return average_array, average_out, average_in


def change_is_out(len_isout, n_true, high_number = 10000):
    if n_true < 1:
        n_true = 1

    enter_again = True
    while enter_again:
        random_ints = np.random.randint(0, high = high_number, size = len_isout)
        #random_ints0 = np.copy(random_ints)
        sorted_ints = np.sort(random_ints)
        critic_rnd = sorted_ints[n_true]
        aux_array = critic_rnd == random_ints

        if aux_array.sum() == 2:
            random_ints [aux_array][0] +=1
            enter_again = False
        elif aux_array.sum() > 2:
            enter_again = True
            print ('Compra kino')
        elif np.all(random_ints>critic_rnd):
            #print (random_ints)
            enter_again = True
        else:
            enter_again = False

    new_isout = np.array([(rnd_int < critic_rnd) for rnd_int in random_ints])
    if np.all(new_isout==False):
        print (random_ints)
        raise ValueError('Something went wrong: Your bool array was all False.')
    return new_isout

from itertools import combinations
def create_all_isout(len_isout, n_true, max_len = -1, critic_len = 10**7):
    result = []
    if sp.special.comb(len_isout, n_true) < critic_len:
        falses = np.zeros((len_isout), dtype=bool)
        combs = np.array(list(combinations(range(len_isout), n_true)))
        if max_len > 0 and len(combs) >= max_len:
            combs = np.random.permutation(combs)[:max_len]
        for comb in combs:
            aux = np.copy(falses)
            aux[comb] = True
            result.append(aux)
            del aux
        del combs
    else:
        #print ('Force no repeat warning.')
        c = 0
        for i in range(max_len):
            repeat = True
            c -=1
            while repeat:
                new_is_out = change_is_out(len_isout, n_true, high_number=critic_len)
                repeat = new_is_out.tolist() in result
                c += 1
            result.append(new_is_out.tolist())
        #print ('Repetitions:', c)
    return np.array(result)



def mask_redfield(is_out, is_in, mask_remove):
    indexes = np.array(range(len(is_out)))
    ind_in = indexes[is_in]
    ind_out = indexes[is_out]
    remove_in = ind_in[mask_remove]
    mask = np.array([(k in ind_out) or (not k in remove_in) for k in indexes])
    new_is_out = np.delete(is_out, remove_in)
    return mask, new_is_out



def get_histogram_width(data, percent_start = 15.87, percent_end = 84.13):

    start = np.percentile(data, percent_start)
    end = np.percentile(data, percent_end)

    sigma = abs(end - start)/2
    center = (end + start)/2



    return sigma, center


def get_hist_center(centerIO, sigmaIO, center2, sigma2):
    big_length = centerIO - center2
    return big_length/abs(sigmaIO)



#end of functions_database

if __name__ == "__main__":
    low_limit = 3000
    up_limit = 10000
    linedb, str_elems, loggfs = get_all_linedb(folders = ['spectroweb_lines', 'vald_lines'],
                                           wv_lims= [low_limit, up_limit], loggf_lims=[-10, 100],
                                           elem_key= '', elem_delete='')
