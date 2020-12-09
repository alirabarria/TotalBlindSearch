# -*- coding: latin-1 -*-
import numpy as np
import scipy as sp
from scipy.optimize import curve_fit
import matplotlib
import sys
import os
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import gridspec
import pandas as pd
import uncertainties.unumpy as unumpy
from time import gmtime, strftime
from tqdm import tqdm
from Manage_plots import *
from functions_database import *




class TBS:

    def __init__(self, str_adress, linedb, str_elem, log_gf, doppler_db = [], use_bar = True,
                 save_figs = False,
                 orders=22, N_pixs=0, indexFWHM=15, dopplerFWHM = 15, N_FWHM = 30,
                 doppler_pixs = 500, cut_order = 100, t_transit = 55159.33754294,
                 dtransit = 0.1, period = 3.5,
                 norm_high=3.5, norm_width=5, norm_min_pixs = 1, near_indline = 0,
                 test_pixs_near = 15, test_var_amplitude = 0.06,
                 test_var_mu = 0.06, test_percent = 0.7, test_boundary = 0.01,
                 test_times_near = 10,
                 error_gain = 1.36, N_shifts = 2500, delta_max_shift = 1,
                 dlambda_analyze = 0.5, redfield_bins=50, dlt_epochs = 0, max_figs = 1000,
                 dpi_figs = 250, output_folder = 'newimages/', create_folders = False, eq_space_tolerance = 0.001,
                 near_tolerance = 0.001, n_binning = 3, telescope = 'subaru', do_1stdopp = True,
                 wave_binning = 0, tell_limit = 0, alpha_mid = 3./4., alpha_side = 1./2.,
                 tell_only_out = False, interpolate_ref = True, radial_velocity = 0,
                 max_t_len = 1, dlt_epochs_again = [], data_folder = 'exoplanet_data/',
                 get_z_correct = False, doppler_epochs = 0, doppler_times_near = 1,
                 max_dopp_residual = 0.2, n_epoch_merge = 0, merge_in = 'telluric',
                 redf_loops = 3000, omit_hists = False, normalize_hists = False, ignore_tests=False,
                 renorm_scale=1,
                 times_indline=1, merge_epoch_as=[], smart_merge = True,
                 time_average_corr=True, norm_polyorder=3, image_format='.png',
                 plot_time_binning = 0):

        """
        This class make and analysis to seek elements in transits.

        strAdress is a string of the adress of the transit data
        linedb is an array of lines to seek

        :param orders: number of orders in the Echelle spectre
        :param pixs: number of pixs to both sides of the lines
        :param indexFWHM: len of pixs to do the median filter for normalization
        :param norm_high: after the median filter, we take the data between this number times
        the Poisson error in the pix
        :param norm_width: we take the median filter with a len of norm_width times the FWHM
        :param test_pixs_near: for doing the tests to the lines, we take a median filter and rescue
        the data at the center of the line, taking this param to left and right
        :param test_var_amplitude: the maximum difference in amplitude test to discard the line
        :param test_var_mu: the maximum difference in the center of the line to discard the line
        :param test_percent: the minimum percent of pixs that are correct to accept the line
        :param test_boundary: the maximum difference to take the pix like a good one
        :param error_gain: the gain to calculate de Poisson error of a pix
        :param N_shifts: the len of the array of shifts that we test to find the best shift to
        the second Doppler correction
        :param delta_max_shift: the maximum difference (to left and right) of wavelength
        we take to find the best shift in flux to the second Doopler correction
        :param dlamda_analyze: is used to calculate the width of the bands we make for analyze the lines
        """
        #Read files names and set lines arrays
        array_files = np.array(list_archives(data_folder+str_adress))
        #print ('Initial number of files:', len(array_files))
        if telescope == 'harps' or telescope == 'uves':
            str_mjd = 'MJD-OBS'
        else:
            str_mjd = 'MJD'
        self.data_folder = data_folder
        self._files, self._times = extract_from_header(array_files, str_header= str_mjd)
        self.linedb = np.array(linedb)
        self.doppler_db = np.array(doppler_db)
        self.str_elem = str_elem
        self.log_gf = log_gf
        self.use_bar = use_bar

        self.outout_ok = True
        self.inin_ok = True
        self.isout_ok = True

        #init params
        self._params = {"orders": orders, "N_pixs": N_pixs, "N_FWHM": N_FWHM, "indexFWHM": indexFWHM,
                        "dopplerFWHM": dopplerFWHM, "doppler_pixs": doppler_pixs,
                        "t_transit": t_transit, "dtransit": dtransit,
                        "period": period, "cut_order": cut_order,
                        "norm_high": norm_high, "norm_width": norm_width,
                        "norm_min_pixs": norm_min_pixs, "near_indline": near_indline,
                         "test_pixs_near": test_pixs_near, "test_var_amplitude": test_var_amplitude,
                        "test_var_mu": test_var_mu, "test_percent": test_percent,
                        "test_boundary": test_boundary, "error_gain": error_gain,
                        "N_shifts": N_shifts, "delta_max_shift": delta_max_shift,
                        "dlambda_analyze": dlambda_analyze, "redfield_bins": redfield_bins,
                        "dlt_epochs": dlt_epochs, "max_figs": max_figs, "dpi_figs" : dpi_figs,
                        "eq_space_tolerance": eq_space_tolerance, "near_tolerance": near_tolerance,
                        "n_binning": n_binning, "wave_binning": wave_binning,
                        "tell_limit": tell_limit, "alpha_mid": alpha_mid, "alpha_side": alpha_side,
                        "tell_only_out": tell_only_out, "interpolate_ref": interpolate_ref,
                        "radial_velocity": radial_velocity, "max_t_len": max_t_len, "do_1stdopp": do_1stdopp,
                        "test_times_near":test_times_near, "telescope": telescope,
                        "max_dopp_residual":max_dopp_residual, "n_epoch_merge": n_epoch_merge,
                        "merge_in": merge_in, "redf_loops": redf_loops, "omit_hists": omit_hists,
                        "normalize_hists": normalize_hists,
                        "ignore_tests": ignore_tests, "renorm_scale": renorm_scale, "times_indline": times_indline,
                        "merge_epoch_as": merge_epoch_as, "smart_merge": smart_merge,
                        "time_average_corr": time_average_corr, "norm_polyorder": norm_polyorder,
                        "plot_time_binning": plot_time_binning
                        }


        if orders == 0:
            self.all_together = True
        else:
            self.all_together = False

        self.save_figs = save_figs
        self.output_folder = output_folder
        self.general_warnings = ''
        #init some auxiliar arrays
        self._str_adress = str_adress
        self._ind_lines = []
        self.line_centers = []
        self.wave_matrix = []
        self._line_orders = []
        self._useful_data = []
        self._dpline_orders = []
        self._ranges = []
        self.line_valid = np.ones(len(linedb), dtype = bool)
        self._index_fwhm = []
        self.img_format = image_format

        if N_pixs == 0:
            self._params['N_pixs'] = int(self._params['N_FWHM']*self._params['indexFWHM']+0.5)
            N_pixs = self._params['N_pixs']

        if wave_binning > 1:
            self._params['indexFWHM'] = int(self._params['indexFWHM']/wave_binning + 0.5)
            self._params['N_pixs'] = int(self._params['N_pixs']/wave_binning + 0.5)
            N_pixs = self._params['N_pixs']
            indexFWHM = self._params['indexFWHM']
            self.general_warnings += 'indexFWHM and N_pixs changed due to wavelength binning'
            print ('Params corrected: ', self._params['indexFWHM'], self._params['N_pixs'])

        if near_indline == 0:
            self._params['near_indline'] = int(times_indline*self._params['indexFWHM'] + 0.5)


        #init superline and another important arrays

        if ignore_tests:
            print ('WARNING: You will ignore noise tests')


        self._times, self.is_out, self._ind_peri, tlen_mask, t_start = label_transit(self._times, period=self._params['period'],
                        dtransit=self._params['dtransit'], t_transit = self._params['t_transit'],
                        max_t_len = self._params['max_t_len'])
        self._params['t_start'] = t_start
        self._files = np.array(self._files)
        self._files = self._files[tlen_mask]
        if len(dlt_epochs_again) != 0:
            mask_dlt = np.ones((len(self._files)), dtype=bool)
            for ind in dlt_epochs_again:
                mask_dlt[ind] = False
            self._files = self._files[mask_dlt]
            self.is_out = self.is_out[mask_dlt]
            self._times = self._times[mask_dlt]

        print('Final number of files:', len(self._files))
        self._params['n_epochs'] = len(self._files)

        if dlt_epochs != 0:
            self._delete_epochs()
        if do_1stdopp:
            self._z_correct =  np.zeros((self._params['orders'], len(self._files)))
        self.orders_length = []


        self._get_line_orders()
        self.superline = np.zeros((len(self.linedb), len(self._files), N_pixs))
        self.ref_flux = np.zeros((len(self.linedb), N_pixs))
        self.error_matrix = np.zeros((len(self.linedb), len(self._files), N_pixs))
        self._raw_flux = np.zeros((len(self.linedb), len(self._files), N_pixs))
        if self._params['telescope'] == 'uves':
            self._raw_error = np.zeros((len(self.linedb), len(self._files), N_pixs))
        self.tell_corrections = np.zeros((len(self.linedb), N_pixs))
        self.tell_masks = []
        self.snellen_data = []
        self.mean_snellen = []
        self.redfield_data = []
        #self.line_info = []
        #self._get_dplines(lines_adress)

        #self._sigma_data = pd.DataFrame()
        #self._all_sigma_data = pd.DataFrame()


        #arrays_of_plots
        dpi = self._params['dpi_figs']
        self.checking_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)
        self.extraction_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)
        self.telluric_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)
        self.renorm_figs = Manage_plots(max=self._params['max_figs'], dpi=dpi)
        self.summary_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)
        self.sigmas_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)
        self.lcheck_figs = Manage_plots(max = self._params['max_figs'], dpi = dpi)

        #captains_log
        self.log_index = []


        for ind in range(len(self.linedb)):
            aux = str(round(self.linedb[ind], 3)) +'('+str(self._line_orders[ind])+')'
            while aux in self.log_index:
                aux+='b'
            self.log_index.append(aux)

        self.log_index = np.array(self.log_index)
        log_cols = ['elem', 'obs_wv', 'fwhm', 'fwhm_pix', 'status', 'detection', 'warning', 'all_ok', 'log_gf', 'band',
            'norm_tests', 'norm_residuals', 'abs_residuals', 'norm_depth', 'abs_depth',
            'snellen_in', 'snellen_out', 'edge_distance', 'std_residuals', 'snr', 'n_pix_fit', 'mse', 'centered_mse',
            'amplitude_mark', 'center_mark', 'residuals_mark', 'blended_mark', 'blended_valid', 'cont_test_mean',
            'cont_test_std', 'shift_test_mean', 'shift_test_std',
            'centerIO', 'sigmaIO', 'centerOO', 'sigmaOO', 'centerII', 'sigmaII', 'image', 'epoch_image']
        self.captains_log = pd.DataFrame(index=self.log_index,
                                         columns=log_cols)
        self.captains_log['elem'] = self.str_elem
        self.captains_log['band'] = dlambda_analyze
        self.captains_log['log_gf'] = self.log_gf
        self.captains_log['detection'] = False
        self.captains_log['norm_tests'] = False
        self.error_log = pd.DataFrame(index=self.log_index, columns=['message'])
        self.error_log['message'] = 'All ok'

        str_init = 'Not done'
        self.captains_log['status'] = str_init
        self.captains_log['warning'] = 'All ok'
        self.captains_log['all_ok'] = True

        secz = []
        exptime = []
        for epoch in range(len(self._files)):
            if self._params['telescope'] == 'harps' or self._params['telescope'] =='uves':
                aux = np.mean([fits.getheader(self._files[epoch])['HIERARCH ESO TEL AIRM END'],
                               fits.getheader(self._files[epoch])['HIERARCH ESO TEL AIRM START']])
                secz.append(aux)
                exptime.append(float(fits.getheader(self._files[epoch])['EXPTIME']))
            elif self._params['telescope'] == 'keck':
                secz.append(float(fits.getheader(self._files[epoch])['AIRMASS']))
                exptime.append(float(fits.getheader(self._files[epoch])['EXPTIME']))
            else:
                secz.append(float(fits.getheader(self._files[epoch])['SECZ']))
                exptime.append(float(fits.getheader(self._files[epoch])['EXPTIME']))
        self.secz = secz
        self.exptime = np.array(exptime)



        #creating folders
        needed_folders = [
            r''+output_folder+str_adress+'/first_epoch',
            r''+output_folder+str_adress+'/atom_data',
            r''+output_folder + str_adress + '/captains_log',
            r''+output_folder + str_adress + '/extract_lines',
            r''+output_folder + str_adress + '/other_epoch',
            r''+output_folder + str_adress + '/redfield',
            r''+output_folder + str_adress + '/tspectra',
            r''+output_folder + str_adress + '/snellen',
            r''+output_folder + str_adress + '/summary_plots',
            r''+output_folder + str_adress + '/telluric',
            r''+output_folder + str_adress + '/error_log',
            r''+output_folder+str_adress+'/objects',
            r'' + output_folder + str_adress + '/params_log'
        ]
        for folder in needed_folders:
            if not os.path.exists(folder) and create_folders:
                os.makedirs(folder)
                print ('Creating needed folder: '+folder)

        if doppler_epochs > 0:
            ind_dopp = np.argsort(self.doppler_db)
            self.doppler_db = self.doppler_db[ind_dopp]
            self._dpline_orders = self._dpline_orders[ind_dopp]
            self.get_doppler_z(get_z_correct, doppler_epochs, times_near= doppler_times_near)


    def _delete_epochs(self):
        to_dlt = self._params['dlt_epochs']
        is_out = self.is_out
        is_in = np.array([not b for b in is_out])
        for i in range(to_dlt):
            l = len(is_out)
            indexes = np.array(range(l))
            mask = np.ones(l, dtype = bool)
            idx_in = indexes[is_in]
            first = idx_in[0]
            last = idx_in[len(idx_in)-1]
            mask[first] = False
            mask[last] = False
            self.is_out = is_out[mask]
            self._files = self._files[mask]
            self._times = self._times[mask]

    def _check_times(self, figsize = (7,9), figname ='test_times'):
        figname += self.img_format
        archives = list_archives(self.output_folder + self._str_adress + '/params_log/', format=self.img_format, key=figname)
        if len(archives) == 0:
            plt.close('all')
            is_in = np.array([not aux for aux in self.is_out])
            fig = plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(2,1)
            in_times = self._times[is_in]
            out_times = self._times[self.is_out]
            ax = fig.add_subplot(grid[0], facecolor = 'floralwhite')
            x = np.array(range(self._params['n_epochs']))
            ax.plot(x[is_in], in_times, 's', color = 'darkgreen')
            ax.plot(x[self.is_out], out_times, '*', color = 'darkblue')
            ax.set_ylabel(r'Time [MJD]')

            ax_exp = fig.add_subplot(grid[1], facecolor = 'floralwhite')
            ax_exp.plot(in_times, self.exptime[is_in], 's',color = 'darkgreen')
            ax_exp.plot(out_times, self.exptime[self.is_out],'*', color = 'darkblue')
            ax_exp.axhline(y = np.mean(self.exptime[is_in]), color = 'darkgreen')
            ax_exp.axhline(y=np.mean(self.exptime[self.is_out]), color='darkblue')
            ax_exp.set_ylabel(r'Exp time')
            for t, ind in zip(self._times, range(len(self.is_out))):
                ax_exp.text(t, self.exptime[ind], str(ind), color="black", fontsize=12)
            if self.save_figs:
                fig.savefig(self.output_folder+self._str_adress+'/params_log/'+figname)
            else:
                file_name = self.output_folder + self._str_adress + '/params_log/' + figname
                self.checking_figs.add_plot(fig, file_name)

    def _check_flux(self, figsize = (7,7), figname ='test_flux', ords = [0]):
        from mpl_toolkits.mplot3d import Axes3D
        for ord in ords:
            plt.close('all')
            is_in = np.array([not aux for aux in self.is_out])
            fig = plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(3, 1, hspace=0.5)
            ax = fig.add_subplot(grid[0], facecolor='floralwhite')
            flux_b = []
            flux_c = []
            flux_r = []
            tot_flux = []
            for file in self._files:
                wave, flux, error = openFile(file, ord, orders_together=self.all_together,
                                      using_keck=self._params['telescope'] == 'keck',
                                        using_uves=self._params['telescope']=='uves')
                l = len(flux)
                fb = flux[0:int(l / 3)]
                fc = flux[int(l / 3):int(2 * l / 3)]
                fr = flux[int(2 * l / 3):l]
                flux_b.append(np.sum(fb))
                flux_c.append(np.sum(fc))
                flux_r.append(np.sum(fr))
                tot_flux.append(np.sum(flux))
            flux_b = np.array(flux_b) / np.mean(flux_b)
            flux_c = np.array(flux_c) / np.mean(flux_c)
            flux_r = np.array(flux_r) / np.mean(flux_r)
            tot_flux = np.array(tot_flux) / np.mean(tot_flux)

            ax.plot(self._times, flux_b, '--', color='darkblue')

            ax.plot(self._times, flux_c, '--', color='darkgreen')

            ax.plot(self._times, flux_r, '--', color='darkred')

            ax.plot(self._times, tot_flux, '--', color='black')
            ax.axvspan(self._times[is_in][0], self._times[is_in][-1], color='coral',
                       alpha=0.7)
            for t, ind in zip(self._times, range(len(flux_b))):
                ax.text(t, flux_b[ind], str(ind), color="darkblue", fontsize=12)
                ax.text(t, tot_flux[ind], str(ind), color="black", fontsize=12)

            ax2 = fig.add_subplot(grid[1], facecolor='floralwhite')
            secz_out = self.secz[self.is_out]
            secz_in = self.secz[is_in]

            ax2.plot(secz_in, np.log(flux_b[is_in]), '+', color='darkblue')
            ax2.plot(secz_out, np.log(flux_b[self.is_out]), '*', color='darkblue')
            ax2.plot(self.secz, np.log(flux_b), '--', color='darkblue')

            ax2.plot(secz_in, np.log(flux_c[is_in]), '+', color='darkgreen')
            ax2.plot(secz_out, np.log(flux_c[self.is_out]), '*', color='darkgreen')
            ax2.plot(self.secz, np.log(flux_c), '--', color='darkgreen')

            ax2.plot(secz_in, np.log(flux_r[is_in]), '+', color='darkred')
            ax2.plot(secz_out, np.log(flux_r[self.is_out]), '*', color='darkred')
            ax2.plot(self.secz, np.log(flux_r), '--', color='darkred')

            ax2.plot(secz_in, np.log(tot_flux[is_in]), '+', color='black')
            ax2.plot(secz_out, np.log(tot_flux[self.is_out]), '*', color='black')
            ax2.plot(self.secz, np.log(tot_flux), '--', color='black')

            ax.set_ylabel(r'Normalized integrated flux')
            ax2.set_ylabel(r'Normalized integrated flux')
            ax.set_xlabel(r'Times[MJD]')
            ax2.set_xlabel(r'Airmass')


            if self.save_figs:
                fig.savefig(self.output_folder + self._str_adress + '/params_log/' + figname+ '_ord_'+str(ord
                            ) + self.img_format)

            else:
                file_name = self.output_folder + self._str_adress + '/params_log/' + figname + '_ord_'+str(ord
                            )+self.img_format
                self.checking_figs.add_plot(fig, file_name)




    def _set_warning(self, string, pos):

        str_init = self.captains_log.at[pos, 'warning']
        if 'All ok' in str_init:
            self.captains_log.at[pos, 'warning'] = string

        elif 'Error found' not in self.captains_log.at[pos, 'warning']:
            if string != 'Line incorrect by tests':
                print ('Error found!', string)
            sys.stdout.flush()
            aux = self.captains_log.at[pos, 'warning']
            self.captains_log.at[pos, 'warning'] = aux +','+ string

        if not  string == 'Line incorrect by tests' and not string == 'Possible detection':
            self.captains_log.at[pos, 'all_ok'] = False


    def _set_status(self, string, pos):
        str_init = self.captains_log.at[pos, 'status']
        if not isinstance(str_init, (list, tuple, np.ndarray)):
            if str_init == 'Not done' or str_init == 'Success':
                self.captains_log.at[pos, 'status'] = string
            elif str_init == 'Success' and string == 'Success':
                pass
            else:
                aux = self.captains_log.at[pos, 'status']
                self.captains_log.at[pos, 'status'] = aux +',' + string
        else:
            if string == 'Success':
                self.captains_log.at[pos, 'status'] = string
            else:
                aux = self.captains_log.at[pos, 'status']
                self.captains_log.at[pos, 'status'] = aux + ',' + string



    def _update_data(self, update_superline= True, unduplicate = False):
        if unduplicate and len(self.line_centers) != 0:
            duplicates = 0
            tol = self._params['near_tolerance']
            for j in range(1, len(self.linedb)):
                aux_line = self.line_centers[j]
                previous_line = self.line_centers[j-1]
                if not self.all_together:
                    aux_order = self._line_orders[j]
                    previous_order = self._line_orders[j-1]
                else:
                    aux_order = 0
                    previous_order = 0
                if abs(aux_line - previous_line) < tol  and aux_order == previous_order:
                    self.line_valid[j] = False
                    duplicates = duplicates + 1
                elif j != len(self.linedb) - 1:
                    aux_line = self.line_centers[j + 1]
                    previous_line = self.line_centers[j - 1]
                    if not self.all_together:
                        aux_order = self._line_orders[j + 1]
                        previous_order = self._line_orders[j - 1]
                    if abs(aux_line - previous_line) < tol and aux_order == previous_order:
                        self.line_valid[j+1] = False
                        duplicates = duplicates + 1

            print (str(duplicates) + ' duplicates lines were deleted.')
            self._params['duplicates_deleted'] = duplicates


        self.linedb = self.linedb[self.line_valid]
        self._line_orders = self._line_orders[self.line_valid]
        self.line_centers = self.line_centers[self.line_valid]
        self.str_elem = self.str_elem[self.line_valid]
        self.log_gf = self.log_gf[self.line_valid]
        self.log_index = self.log_index[self.line_valid]
        if update_superline:
            self.superline = self.superline[self.line_valid, :, :]
            self.error_matrix = self.error_matrix[self.line_valid, :, :]
            self._raw_flux = self._raw_flux[self.line_valid, :, :]
            if self._params['telescope'] == 'uves':
                self._raw_error = self._raw_error[self.line_valid, :, :]
            self.tell_corrections = self.tell_corrections[self.line_valid, :]
            if len(self.wave_matrix) != 0:
                self.wave_matrix = self.wave_matrix[self.line_valid]
            if len(self.ref_flux) != 0:
                self.ref_flux = self.ref_flux[self.line_valid]
            if len(self.tell_masks) == len(self.line_valid) and len(self.line_valid) > 0:
                self.tell_masks = self.tell_masks[self.line_valid]
        self.line_valid = self.line_valid[self.line_valid]



    def _read_data(self, epoch, do_plt=False):
        """
        This function takes an epoch and it reads the data of all of its
        orders. If you want to plot the order h, so put indPlot = h
        """
        wave_matrix = []
        flux_matrix = []

        if self.all_together:
            wave_matrix, flux_matrix, error = openFile(self._files[epoch], 0, doPlot=do_plt,
                                                orders_together=self.all_together,
                                                using_keck = self._params['telescope']=='keck',
                                                using_uves=self._params['telescope'] == 'uves')
        else:
            for ord in range(self._params['orders']):
                wave, flux, error = openFile(self._files[epoch], ord, doPlot=do_plt,
                                      orders_together=self.all_together,
                                      using_keck=self._params['telescope'] == 'keck',
                                    using_uves=self._params['telescope'] == 'uves')
                wave_matrix.append(wave)
                flux_matrix.append(flux)

        return np.array(wave_matrix), np.array(flux_matrix), np.array(error)




    def _slice_and_normalize(self, wave_matrix, flux_matrix, line_index, epoch,
                             getFWHM = False, print_info= False, error = []):
        """
        this function takes arrays of arrays of wavelength and flux, and normalizes the
        flux of line_index, epoch, and makes the first Doppler correction to them.
        """
        N_pixs = self._params['N_pixs']
        near_indline = self._params['near_indline']
        indexFWHM = self._params['indexFWHM']
        high = self._params['norm_high']
        width = self._params['norm_width']
        min_pixs = self._params['norm_min_pixs']
        near = self._params['test_pixs_near']
        line = self.linedb[line_index]
        line_ord = self._line_orders[line_index]


        if self._params['do_1stdopp']:
            if self._params['radial_velocity'] == 0:
                zdopp = self._z_correct[line_ord][epoch]
            else:
                zdopp = self._params['radial_velocity']/(299792)
        else:
            zdopp = 0

        if not self.all_together:
            wave = wave_matrix[line_ord]
            flux = flux_matrix[line_ord]
        else:
            wave = wave_matrix
            flux = flux_matrix

        ext_wave, ext_flux, error_y, ind_line = extract_array(wave, flux, line, N_pixs,
                                                self._params['error_gain'], error = error, binning=self._params['wave_binning'])


        wave0 = np.copy(ext_wave)
        dw = abs(wave[ind_line + 1] - wave[ind_line])

        if getFWHM:
            if print_info:
                print ('The nearest wavelength to the line is ' + str(ext_wave[ind_line]))
                print ()
            linspace = np.linspace(0, len(ext_wave), len(ext_wave))
            params, fit, new_data = gaussian_and_constant_fit(linspace, ext_flux, ind_line, near)
            A, mu, sigma, const = params
            little_mask_ = np.array(range(int(ind_line - near_indline), int(ind_line + near_indline)))
            self.captains_log.at[self.log_index[line_index], 'abs_depth'] = np.max(ext_flux[little_mask_]) - np.min(ext_flux[little_mask_])
            index_fwhm = np.abs(2*np.sqrt(2*np.log(2)) * sigma)
            self.captains_log.at[self.log_index[line_index], 'fwhm_pix'] = round(index_fwhm, 2)
            self.captains_log.at[self.log_index[line_index], 'fwhm'] = round(index_fwhm*dw, 3)
            """
            plt.plot(ext_wave, ext_flux)
            plt.plot(ext_wave, fit, 'r')
            plt.show()
            """
            if print_info:
                print ("And its optimal FWHM is " + str(index_fwhm))
                print ()

        y0 = np.copy(ext_flux)
        error0 = np.copy(error_y)
        norm_y, norm_error, polynomial, uplimit, mfilter, lowlimit, filter_mask, message = normalize(ext_wave,
                                        ext_flux, error_y, indexFWHM, min_pixs, high, width,
                                        polyorder=self._params['norm_polyorder'])
        if getFWHM and message == 'ok':
            residuals_, _ = get_histogram_width(ext_flux[filter_mask] - polynomial[filter_mask], percent_start= 25.,
                                            percent_end=75.)
            self.captains_log.at[self.log_index[line_index], 'abs_residuals'] = residuals_
        result_wave = []
        result_flux = []
        result_error = []
        ind_min = 0
        arrays_to_plot = y0, error0, [], uplimit, mfilter, lowlimit, wave0, np.ones((len(y0)), dtype=bool)
        if message == 'ok':
            ind_line0 = ind_line
            if self._params['do_1stdopp']:
                corr_wave, corr_flux = first_doppler_corr(ext_wave, norm_y, zdopp)
                ind_line = np.argmin(np.abs(corr_wave - line))
            else:
                corr_wave = ext_wave
                corr_flux = norm_y
            #print (ind_line, line)
            high_index = ind_line + near_indline
            low_index = ind_line - near_indline
            if low_index < 0:
                low_index = 0
            elif high_index > self._params['N_pixs']:
                high_index = self._params['N_pixs']
            elif low_index == high_index:
                message = 'problem'
            little_mask = np.array(range(int(low_index), int(high_index)))
            ind_min = little_mask[np.argmin(corr_flux[little_mask])]

            wave0 = wave0[int(ind_line0 - near_indline): int(ind_line0 + near_indline)]
            arrays_to_plot = y0, error0, polynomial, uplimit, mfilter, lowlimit, wave0
        else:
            corr_wave = ext_wave
            corr_flux = []
            norm_error = []
            ind_min = 0
            arrays_to_plot = y0, error0, [], uplimit, mfilter, lowlimit, wave0

        return corr_wave, corr_flux, norm_error, ind_min, filter_mask, arrays_to_plot, message



    def _get_line_orders(self):
        orders = np.array(range(self._params['orders']))
        cut_order = self._params['cut_order']
        tol_dw = self._params['eq_space_tolerance']
        if not self.all_together:
            for order in orders:
                if self._params['telescope'] == 'keck':
                    wave, data, error = openFile(self._files[0], order, orders_together=False,
                                                 using_keck=True)
                elif self._params['telescope'] == 'uves':
                    from astropy.io import fits
                    data = fits.open(self._files[0])
                    wave = np.array(data[1].data[0])[0]

                else:
                    data = Spectrum(str(self._files[0]))
                    wave = np.array(data[order].xarr)


                self._ranges.append([wave[cut_order], wave[len(wave)-cut_order - 1] ])
                self.orders_length.append([abs(wave[cut_order] - wave[cut_order + 1]),
                                           abs(wave[len(wave) - cut_order-2] - wave[len(wave) -cut_order -1])])
            self._ranges = np.array(self._ranges)

            k = 0
            while (k < len(self.linedb)):

                line = self.linedb[k]
                elem = self.str_elem[k]
                logf = self.log_gf[k]
                bool_array = [(exts[0]<line)*(line<exts[1]) for exts in self._ranges]
                bool_array = np.array(bool_array)
                aux_order = orders[bool_array]
                if len(aux_order) == 0:
                    self.linedb = np.delete(self.linedb, k)
                    self.str_elem = np.delete(self.str_elem, k)
                    self.log_gf = np.delete(self.log_gf, k)
                    self.line_valid = np.delete(self.line_valid, k)

                else:
                    l = len(aux_order)
                    for order in aux_order:
                        self._line_orders.append(order)

                    if l>1:
                        for h in range(l - 1):
                            self.linedb = np.insert(self.linedb, k + h + 1, line)
                            self.str_elem = np.insert(self.str_elem, k+h+1, elem)
                            self.log_gf = np.insert(self.log_gf, k+h+1, logf)
                            self.line_valid = np.insert(self.line_valid, k + h + 1, True)
                        k = k + l
                    else:
                        k = k + 1

            #after while
            ord_len = self.orders_length
            warning = False
            for dw in ord_len:
                if abs(dw[0] - dw[1]) > tol_dw:
                    warning = True
            if warning:
                string = "WARNING: Your data is not equally spaced. "
                #print (string)
                self.general_warnings += string
            k = 0
            while (k < len(self.doppler_db)):
                line = self.doppler_db[k]
                bool_array = [(exts[0]<line)*(line<exts[1]) for exts in self._ranges]
                bool_array = np.array(bool_array)
                if np.all(bool_array == False):
                    print ('The line '+ str(line)+' is at the extreme of the order.')
                    self.doppler_db = np.delete(self.doppler_db, k)
                else:
                    k = k + 1
                    if len(orders[bool_array]) > 1:
                        aux_order = orders[bool_array][1]
                    else:
                        aux_order = orders[bool_array][0]
                    self._dpline_orders.append(aux_order)
        else:
            if self._params['telescope'] == 'uves':
                from astropy.io import fits
                data = fits.open(self._files[0])
                wave = np.array(data[1].data[0])[0]
            else:
                data = Spectrum(str(self._files[0]))
                wave = np.array(data.xarr)
            self._ranges.append([wave[cut_order], wave[len(wave) - cut_order - 1]])
            self.orders_length.append([abs(wave[cut_order] - wave[cut_order + 1]),
                                       abs(wave[len(wave) - cut_order - 2] - wave[len(wave) - cut_order - 1])])
            self._ranges = np.array(self._ranges)

            k = 0
            while k < len(self.linedb):
                line = self.linedb[k]
                if not (self._ranges[0][0] < line and line  < self._ranges[0][1]):
                    self.linedb = np.delete(self.linedb, k)
                    self.str_elem = np.delete(self.str_elem, k)
                    self.log_gf = np.delete(self.log_gf, k)
                    self.line_valid = np.delete(self.line_valid, k)
                else:
                    k = k + 1
            self._line_orders = np.zeros((len(self.linedb)), dtype=int)



        self._line_orders = np.array(self._line_orders)
        self._dpline_orders = np.array(self._dpline_orders)
        #print (self._dpline_orders)

    def get_doppler_z(self, get_z_correct, epochs, times_near = 3, figsize = (10,20), all_plt = False,
                      do_tests = False):
        """
        this funcion returns the z = v/c of all epoch.
        """
        N = self._params['doppler_pixs']
        indexFWHM = self._params['dopplerFWHM']
        high = self._params['norm_high']
        width = self._params['norm_width']
        orders = np.array(range(self._params['orders']))
        min_pixs = self._params['norm_min_pixs']
        near_indline = int(times_near*self._params['near_indline'])
        wave_obs = []

        if all_plt:
            plt.close('all')
            fig_dopp = plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(len(self.doppler_db), 1)
        c = 0
        for epoch in range(epochs):
            wave_matrix, flux_matrix, err_matrix = self._read_data(epoch)
            line_vldt_all = []
            for order in orders:
                """
                if not np.any(order==self._line_orders):
                    continue
                """
                mask_order = np.array([aux == order for aux in self._dpline_orders])
                dplines = np.sort(np.array(self.doppler_db)[mask_order])

                line_vldt_per_order = []
                for dpline in dplines:
                    wave = wave_matrix[order]
                    flux = flux_matrix[order]
                    if len(err_matrix) != 0:
                        error = err_matrix[order]
                    else:
                        error = []
                    ext_wave, ext_flux, error_y, ind_line = extract_array(wave, flux, dpline, N,
                                                            self._params['error_gain'],
                                                            error = error, binning=self._params['wave_binning'])
                    #y0 = np.copy(ext_flux)

                    norm_y, norm_error, polynomial, uplimit, mfilter, lowlimit, mask, message = normalize(ext_wave,
                                                                                       ext_flux, error_y, indexFWHM,
                                                                                      min_pixs, high, width,
                                                                                    polyorder=self._params['norm_polyorder'])

                    params, fit, newData = gaussian_and_constant_fit(ext_wave, norm_y, ind_line, self._params["indexFWHM"],
                                                                     times_near=self._params['norm_width'])
                    A, mu, sigma, C = params
                    ind_low = ind_line-near_indline
                    ind_high = ind_line + near_indline
                    if ind_low < 0:
                        ind_low = 0
                    if ind_high > N:
                        ind_high = N - 1

                    little_mask = np.array(range(ind_low, ind_high))
                    ind_min = little_mask[np.argmin(norm_y[little_mask])]
                    if all_plt:
                        ax = fig_dopp.add_subplot(grid[c], facecolor='floralwhite')
                        c = c + 1
                        dind = 50
                        ax.plot(ext_wave[ind_low - dind:ind_high + dind], norm_y[ind_low - dind:ind_high + dind], color = 'royalblue', linewidth = 0.6)
                        ax.axvspan(ext_wave[ind_low], ext_wave[ind_high], color = 'coral', alpha = 0.9)
                        ax.axvline(x = ext_wave[ind_line], linestyle ='--', color = 'magenta')
                        ax.plot(ext_wave[ind_min], norm_y[ind_min], '*', color = 'magenta')

                    test_info = False
                    if do_tests:
                        amplitude_validity, emp_amp= amplitude_test(norm_y, A, ind_min,
                                                        dA = self._params["test_var_amplitude"],
                                                        info = test_info)
                        center_validity, emp_cdist = center_test(ext_wave, mu, ind_min, dMu = self._params["test_var_mu"],
                                                  info = test_info)
                        residuals_validity, emp_res = residuals_test(norm_y[mask] - 1, percent = self._params["test_percent"],
                                                        boundary = self._params["test_boundary"], info = test_info)
                        line_validity = amplitude_validity and center_validity and residuals_validity
                    else:
                        line_validity = True
                    line_vldt_per_order.append(line_validity)
                    line_vldt_all.extend([line_validity])


                    if line_validity:
                        wave_obs.append(ext_wave[ind_min])
            if all_plt:
                fname = self.output_folder+self._str_adress+'/params_log/check_dopp'+self.img_format
                self.checking_figs.add_plot(fig_dopp, fname)


            if len(wave_obs) > 1:
                wave_obs = np.sort(wave_obs)
                # line_vldt_per_order = np.array(line_vldt_per_order)
                #m, n = guess_lineal_params(dplines, wave_obs)
                doppdb = np.array(self.doppler_db)[line_vldt_all]

                params, fit = fit_pol1(doppdb, wave_obs, p0 = [1, 0])
                while (np.any(wave_obs - fit > self._params['max_dopp_residual'])):
                    mask_fit = wave_obs - fit > self._params['max_dopp_residual']
                    doppdb = doppdb[mask_fit]
                    wave_obs = wave_obs[mask_fit]
                    params, fit = fit_pol1(doppdb, wave_obs, p0=params)
                z = params[0] - 1
                if get_z_correct:
                    self._z_correct[order][epoch] = z
                else:
                    radial_velocity = z*299792

                    print ('z =', str(z), 'so the optimal radial velocity will be', radial_velocity, 'km/s')
                    plt.close('all')
                    fig = plt.figure()
                    ax = fig.add_subplot(1, 2, 1,  facecolor='floralwhite')
                    ax.plot(doppdb, fit, color='navy', linewidth = 0.7)
                    ax.plot(doppdb, wave_obs, '.', color='darkgreen')
                    text = str(radial_velocity)+'\n'+str(z)
                    ax.text(0.5, 0.1, text, verticalalignment='center', horizontalalignment='left',
                                 transform=ax.transAxes, style='italic',
                                 color='darkred', fontsize=6, bbox={'facecolor': 'navajowhite', 'boxstyle': 'round',
                                                                    'alpha': 0.5, 'pad': 1, 'edgecolor': 'black'})
                    ax2 = fig.add_subplot(1,2,2, facecolor = 'floralwhite')
                    ax2.axhline(y=0, color='navy')
                    ax2.plot(wave_obs - fit , '.', color = 'darkgreen')


                    file_name = self.output_folder+self._str_adress+'/params_log/check_RV'+self.img_format
                    self.checking_figs.add_plot(fig, file_name)


    def check_epoch(self, ind_lineplt = -1, plot_all = False, show_plt = False, epoch = 0, print_info = False,
                    checking=True, pbar='None', figsize=(9, 7), fmt='-', fname_verbose=True):
        """
        it makes the validity tests for the first epoch and discards
        the lines of linedb that are wrong by some parameters. If you
        want to plot the line with index i, write ind_lineplt = i, if
        you want to plot them all, put plot_all = True
        """

        near_indline = self._params['near_indline']
        wave_matrix, flux_matrix, err_matrix = self._read_data(epoch)

        print ()
        print ("Doing validity tests (1/8) to " +str(len(self.linedb))+ " lines.")
        print ()

        sys.stdout.flush()
        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1
        for j in range(len(self.linedb)):
            # aux_line_info = []
            try:
                corr_wave, corr_flux, norm_error, ind_line, mask, arrays_to_plot, message = self._slice_and_normalize(wave_matrix,
                                                                                                             flux_matrix, j,
                                                                                                             epoch, error = err_matrix,
                                                                                                             print_info=print_info,
                                                                                                             getFWHM=True)
                ext_flux, ext_error, polynomial, uplimit, mfilter, lowlimit, wave0 = arrays_to_plot
                y0 = ext_flux
                test_info = False
                if ind_lineplt == j or plot_all:
                    test_info = True

                if message != 'ok':
                    if checking:
                        self.line_valid[j] = False
                        self.line_centers.append(self.linedb[j])

                    aux_line = self.log_index[j]
                    self._set_status('Emptiness error', aux_line)
                    self._set_warning(message, aux_line)
                    self.error_log.at[aux_line, 'message'] = message
                    if test_info or plot_all:
                        plt.close('all')
                        fig = plt.figure(figsize=figsize)
                        bg_color = 'floralwhite'
                        ax1 = fig.add_subplot(111, facecolor=bg_color)
                        ax1.plot(corr_wave, y0, fmt, color='royalblue', linewidth=0.8)
                        ax1.plot(corr_wave, uplimit, 'coral', linewidth=0.5)
                        ax1.plot(corr_wave, mfilter, '-', color = 'orangered', linewidth = 0.6)
                        ax1.plot(corr_wave, lowlimit, 'coral', linewidth=0.5)
                        ax1.fill_between(corr_wave, lowlimit, uplimit, facecolor='lightcoral', alpha=0.5)
                        ax1.set_xlabel(r'Wavelength $[\lambda]$')
                        ax1.set_ylabel(r'Flux')
                        self.checking_figs.add_plot(fig, file_name)
                    continue

                little_mask = np.array(range(ind_line - near_indline, ind_line + near_indline))
                if little_mask[-1] >= self._params['N_pixs']:
                    little_mask = little_mask [little_mask<self._params['N_pixs']]
                ind_min = little_mask[np.argmin(corr_flux[little_mask])]
                corr_wave_fit = np.copy(corr_wave)
                corr_flux_fit = np.copy(corr_flux)
                params, gaussian_fit, newData = gaussian_and_constant_fit(corr_wave_fit, corr_flux_fit, ind_min,
                                                                 self._params["indexFWHM"],
                                                                 times_near=self._params['norm_width'])
                A, mu, sigma, C = params

                if ind_min >= self._params['N_pixs']:
                    ind_min = int(self._params['N_pixs'] - 1)
                if j > 0 and corr_wave[ind_min] == self.line_centers[j - 1] and checking:
                    self.captains_log.at[self.log_index[j], 'warning'] = 'Lines too near'
                    self.captains_log.at[self.log_index[j - 1], 'warning'] = 'Lines too near'
                    new_near_indline = int(near_indline / 2)
                    little_mask = np.array(range(ind_line - new_near_indline, ind_line + new_near_indline))
                    ind_min = little_mask[np.argmin(corr_flux[little_mask])]

                if checking:
                    self.line_centers.append(corr_wave[ind_min])
                    residuals_to_log, _ = get_histogram_width(corr_flux[mask] - np.median(corr_flux[mask]), percent_start=25.,
                                                              percent_end=75.)
                    std_residuals, _ = get_histogram_width(corr_flux[mask] - np.median(corr_flux[mask]))
                    self.captains_log.at[self.log_index[j], 'norm_depth'] = abs(1 - corr_flux[ind_min])
                    self.captains_log.at[self.log_index[j], 'norm_residuals'] = residuals_to_log
                    self.captains_log.at[self.log_index[j], 'std_residuals'] = std_residuals
                    in_epochs = np.sum([not aux for aux in self.is_out])
                    #fwhm_pix = self.captains_log.at[self.log_index[j], 'fwhm_pix']
                    fwhm_pix = self._params['indexFWHM']
                    self.captains_log.at[self.log_index[j], 'snr'] = 1/std_residuals*np.sqrt(in_epochs*fwhm_pix)
                    # sigma per pixel * epochs * pixels 
                    self.captains_log.at[self.log_index[j], 'n_pix_fit'] = mask.sum()
                    self.captains_log.at[self.log_index[j], 'mse'] = np.mean(np.square(norm_error))
                    self.captains_log.at[self.log_index[j], 'centered_mse'] = np.mean(np.square(norm_error[little_mask]))

                band_mask = np.ones((len(corr_wave)), dtype=bool)
                if not self._params['ignore_tests']:
                    amplitude_validity, emp_amp = amplitude_test(corr_flux, A, ind_min,
                                                    dA=self._params["test_var_amplitude"],
                                                    info=print_info)
                    center_validity, emp_cdist = center_test(corr_wave, mu, ind_min, dMu=self._params["test_var_mu"],
                                              info=print_info)
                    not_line_mask = np.ones((len(corr_flux)), dtype=bool)
                    not_line_mask[little_mask] = False
                    residuals_validity, emp_res = residuals_test(corr_flux[mask] - sp.median(corr_flux[mask]),
                                                    percent=self._params["test_percent"],
                                                    boundary=self._params["test_boundary"], info=print_info)

                    band_mask[corr_wave < corr_wave[ind_min] - self._params['dlambda_analyze']/2] = False
                    band_mask[corr_wave > corr_wave[ind_min] + self._params['dlambda_analyze']/2] = False
                    blended_valid, emp_blended = residuals_test(corr_flux[band_mask] - gaussian_fit[band_mask],
                                                                percent=self._params['test_percent'],
                                                                boundary=self._params['test_boundary'],
                                                                info=print_info)

                    self.captains_log.at[self.log_index[j], 'amplitude_mark'] = emp_amp
                    self.captains_log.at[self.log_index[j], 'center_mark'] = emp_cdist
                    self.captains_log.at[self.log_index[j], 'residuals_mark'] = emp_res
                    self.captains_log.at[self.log_index[j], 'blended_mark'] = emp_blended
                    self.captains_log.at[self.log_index[j], 'blended_valid'] = blended_valid
                    #print (emp_res)

                    line_validity = amplitude_validity and center_validity and residuals_validity
                    self.captains_log.at[self.log_index[j], 'norm_tests'] = line_validity
                else:
                    line_validity = True
                    self.captains_log.at[self.log_index[j], 'norm_tests'] = True

                if line_validity == False and checking:
                    self.line_valid[j] = False

                if test_info or plot_all:
                    if line_validity:
                        if print_info:
                            print ("The line " + str(self.log_index[j]) + " is correct")
                        text_plot = 'Correct'
                        self._set_status('Success', self.log_index[j])
                    else:
                        if print_info:
                            print ("The line " + str(self.log_index[j]) + " is discarded because of:")
                        tests = np.array([amplitude_validity, center_validity, residuals_validity])
                        text_plot = 'Discarded by:'
                        to_error = ''
                        for i in range(3):
                            if tests[i] == False:
                                if i == 0:
                                    if print_info:
                                        print ("amplitude")
                                    to_error = to_error + 'amplitude, '
                                    text_plot = text_plot + '\n' + 'amplitude'
                                elif (i == 1):
                                    if print_info:
                                        print ("center of gaussean")
                                    to_error = to_error + 'center, '
                                    text_plot = text_plot + '\n' + 'center'
                                else:
                                    if print_info:
                                        print ("error percent")
                                    to_error = to_error + 'error %'
                                    text_plot = text_plot + '\n' + 'error %'

                        aux_line = self.log_index[j]
                        self._set_status('Discarded by ' + to_error, aux_line)
                        self._set_warning('Line incorrect by tests', aux_line)
                        self.error_log.at[aux_line, 'message'] = 'Discarded by ' + to_error
                    if print_info:
                        print ()
                        print ('.......................................')
                        print ()

                    str_adress = self._str_adress
                    if fname_verbose:
                        aux_name = str(int(self.linedb[j])) + '_ind_' + str(j) + '_ord_' + str(self._line_orders[j])
                    else:
                        aux_name = str(int(self.linedb[j]))
                    file_name = self.output_folder + str_adress + '/first_epoch/' + aux_name + self.img_format
                    if epoch != 0:
                        file_name = self.output_folder + str_adress + '/other_epoch/' + aux_name + '_ep' + str(
                            epoch) + +self.img_format

                    x = corr_wave
                    plt.close('all')
                    fig = plt.figure(figsize=figsize)
                    bg_color = 'floralwhite'

                    ax1 = fig.add_subplot(221, facecolor=bg_color)
                    ax1.plot(x, y0, fmt, color = 'royalblue', linewidth=0.8)
                    ax1.plot(x, uplimit, 'coral', linewidth=0.5)
                    # ax1.plot(x, mfilter, '-', color = 'orangered', linewidth = 0.6)
                    ax1.plot(x, lowlimit, 'coral', linewidth=0.5)
                    ax1.fill_between(x, lowlimit, uplimit, facecolor='lightcoral', alpha=0.5)
                    ax1.set_xlabel(r'Wavelength $[\AA]$')
                    ax1.set_ylabel(r'Flux')


                    ax2 = fig.add_subplot(222, sharex=ax1, sharey=ax1, facecolor=bg_color)
                    #ax2 = fig.add_subplot(111, facecolor = bg_color)
                    ax2.plot(x, y0, fmt, color = 'royalblue', linewidth=0.8)
                    ax2.plot(x[mask], y0[mask], '+', color='darkviolet', markersize=1)
                    ax2.plot(x, polynomial, '--', color='darkgreen', linewidth=1, alpha=1)
                    #ax2.plot(x[little_mask], y0[little_mask], '.', color='darkviolet', markersize = 0.8)
                    ax2.axvspan(x[little_mask][0], x[little_mask][-1], alpha=0.5, color='violet')

                    ax3 = fig.add_subplot(223, facecolor=bg_color)
                    ax3.plot(x, corr_flux, 'royalblue', linewidth=0.8)
                    ax3.plot(x, gaussian_fit, 'maroon', linewidth=0.7)
                    ax3.set_ylabel(r'Normalized flux')
                    ax3.set_xlabel(r'Wavelength $[\AA]$')

                    ax4 = fig.add_subplot(224, sharex=ax3, sharey=ax3, facecolor=bg_color)
                    ax4.plot(x, corr_flux, 'royalblue', linewidth=0.8)
                    #ax4.plot(x, gaussian_fit, '--', color='darkviolet', linewidth=0.8)
                    ax4.plot(x[band_mask], corr_flux[band_mask]-gaussian_fit[band_mask] + 1.1, '--', color = 'coral', linewidth=0.8)
                    half_band = self._params['dlambda_analyze']/2
                    ax4.axvspan(x[ind_min] - half_band, x[ind_min] + half_band, color='lightcoral', alpha=0.4)
                    ax4.plot(x[ind_min], corr_flux[ind_min], '*m', alpha=0.5)
                    #ax4.plot(x[mask], corr_flux[mask], '+', color='darkviolet', markersize=0.9)
                    ax4.axvline(x = x[band_mask][0], color = 'coral', linewidth = 0.8)
                    ax4.axvline(x=x[band_mask][-1], color='coral', linewidth = 0.8)
                    ax4.set_xlabel(r'Wavelength $[\AA]$')

                    plt.subplots_adjust(hspace=0.02, wspace=0.02)
                    plt.setp(ax1.get_xticklabels(), visible=False)
                    plt.setp(ax2.get_xticklabels(), visible=False)
                    plt.setp(ax2.get_yticklabels(), visible=False)
                    plt.setp(ax4.get_yticklabels(), visible=False)
                    plt.suptitle(r'Check epock for ' + str(self.linedb[j]))
                    #plt.tight_layout()
                    ax4.text(0.71, 0.2, text_plot,
                             verticalalignment='center', horizontalalignment='left',
                             transform=ax4.transAxes, style='italic',
                             color='darkred', fontsize=6, bbox={'facecolor': 'navajowhite', 'boxstyle': 'round',
                                                                'alpha': 0.5, 'pad': 1, 'edgecolor': 'black'})
                    if self._params['do_1stdopp']:
                        ax2.axvspan(wave0[0], wave0[-1], color = 'plum', alpha = 0.4)
                    self.checking_figs.add_plot(fig, file_name)

                    if show_plt:
                        plt.show()
            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Check epoch: ' + message
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Extraction: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                if len(self.line_centers) != j + 1:
                    self.line_centers.append(self.linedb[j])
                    self._set_warning('Center len problem', self.log_index[j])
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False

            if self.use_bar:
                pbar.update(dbar)
            sys.stdout.flush()
        if self.use_bar:
            pbar.close()
        if self.save_figs:
            self.checking_figs.save_plots()

        self.line_centers = np.array(self.line_centers)
        self.captains_log["obs_wv"] = self.line_centers
        self.line_valid = np.array(self.line_valid)
        self.str_elem = np.array(self.str_elem)
        self.log_gf = np.array(self.log_gf)
        self._update_data(update_superline= True, unduplicate=False)
        self._check_times()
        """
        if test_info:
            print self.captains_log
        """




    def extract_lines(self, inds_plt = [-1, -1], doall_plt = False, all_line_plt = -1, dpi = 400,
                      pbar = 'None', do_normalize=False):
        """
        After checking the first epoch, this function extracts all the epochs of
        the line in linedb that were not discarded. If you want to plot the line j
        in the epoch i, put inds_plt = [j, i]. If you want to plot them all, put
        doall_plt = True.
        """
        print ()
        print ("Doing extraction (2/8) to " + str(len(self.linedb)) + " lines.")
        print ()

        sys.stdout.flush()

        wave_matrix, flux_matrix, err_matrix = self._read_data(0)
        flux_ref_matrix = []
        for j in range(len(self.linedb)):
            wave, flux, error, ind_line, mask, arrays_to_plot, message = self._slice_and_normalize(wave_matrix, flux_matrix, j, 0,
                                                                                                   error = err_matrix)
            self.wave_matrix.append(wave)
            if do_normalize:
                flux_ref_matrix.append(flux)
            else:
                flux_ref_matrix.append(arrays_to_plot[0])
            self._ind_lines.append(ind_line)
        self._ind_lines = np.array(self._ind_lines)
        #self._ind_lines = self._ind_lines[self.line_valid]

        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb)*self._params['n_epochs'])
        dbar = 1
        Nshifts = self._params["N_shifts"]
        shift_test = {}
        for epoch in range(self._params['n_epochs']):
            wave_matrix, flux_matrix, err_matrix = self._read_data(epoch)
            for j in range(len(self.linedb)):
                if epoch == 0:
                    shift_test[self.log_index[j]] = []
                if not self.line_valid[j]:
                    continue

                if self.all_together:
                    wave_order = wave_matrix
                    flux_order = flux_matrix
                    error_order = err_matrix
                else:
                    order = self._line_orders[j]
                    wave_order = wave_matrix[order]
                    flux_order = flux_matrix[order]
                    if len(err_matrix) != 0:
                        error_order = err_matrix[order]
                    else:
                        error_order = err_matrix
                line_info = False
                if np.all([j, epoch] == inds_plt) or epoch == all_line_plt or doall_plt:
                    line_info = True

                try:
                    inds_plt = [j, epoch]
                    if not do_normalize:
                        ext_wave, flux_obs, error_y, ind_line = extract_array(wave_order, flux_order, self.linedb[j],
                                                                              self._params['N_pixs'],
                                                                              self._params['error_gain'], error=error_order,
                                                                              binning=self._params['wave_binning'])
                        alt_flux = None
                        if self._params['do_1stdopp']:
                            if self._params['radial_velocity'] == 0:
                                zdopp = self._z_correct[self._line_orders[j]][epoch]
                            else:
                                zdopp = self._params['radial_velocity'] / (299792)
                            wave_obs, _ = first_doppler_corr(ext_wave, flux_obs, zdopp)
                            ind_line = np.argmin(np.abs(wave_obs - self.linedb[j]))
                        else:
                            wave_obs = ext_wave
                    else:
                        wave_obs, flux_obs, norm_error, ind_line, mask, arrays_to_plot, message = self._slice_and_normalize(
                                                                                            wave_matrix, flux_matrix, j,
                                                                                            epoch, error=err_matrix,
                                                                                            print_info=False,
                                                                                            getFWHM=False)
                        alt_flux = arrays_to_plot[0]
                        error_y = arrays_to_plot[1]

                        if message != 'ok':
                            self.line_valid[j] = False
                            self.error_log.at[self.log_index[j], 'message'] = 'Extraction: ' + message
                            print (message)
                            sys.stdout.flush()
                            self._set_status('Emptiness error', self.log_index[j])
                            self._set_warning('Error found', self.log_index[j])
                            continue

                    if self._params['interpolate_ref']:
                        previous_wvref = self.wave_matrix[j]
                        previous_fluxref = flux_ref_matrix[j]
                        wave_ref = np.linspace(previous_wvref[0], previous_wvref[-1], Nshifts)
                        flux_ref = shift_array(previous_wvref, previous_fluxref, wave_ref, 0)
                        real_wave_ref = previous_wvref

                    else:
                        wave_ref = self.wave_matrix[j]
                        flux_ref = flux_ref_matrix[j]
                        real_wave_ref = wave_ref
                    if epoch != 0:
                        corr_flux, corr_alt_flux, fig, fig_name, max_prod = second_doppler_corr(wave_ref, real_wave_ref,
                                                                                flux_ref, wave_obs, flux_obs,
                                                                    self.linedb[j], self._line_orders[j], inds_plt,
                                                                   wave_order, flux_order, ind_line, j,
                                                                   deltaWmax=self._params["delta_max_shift"],
                                                                   Nshifts=Nshifts, doplt=line_info,
                                                                   str_adress=self._str_adress,
                                                                   out_folder=self.output_folder,
                                                                   alty=alt_flux)
                        shift_test[self.log_index[j]].append(max_prod)
                        if do_normalize:
                            flux_to_save = corr_alt_flux
                        else:
                            flux_to_save = corr_flux

                    else:
                        if do_normalize:
                            flux_to_save = alt_flux
                        else:
                            flux_to_save = flux_obs


                    if line_info and all_line_plt > 0 and epoch != 0:
                        self.extraction_figs.add_plot(fig, fig_name)

                    if self._params['telescope'] != 'uves':
                        self._raw_flux[j, epoch, :] = flux_to_save / self.exptime[epoch]
                    else:
                        self._raw_flux[j, epoch, :] = flux_to_save
                        self._raw_error[j, epoch, :] = error_y
                    self._set_status('Success', self.log_index[j])


                except Exception as ex:
                    template = "{0}. Arguments: {1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    self.error_log.at[self.log_index[j], 'message'] = 'Extraction: ' + message
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                    self._set_status('Extraction: ' + to_log, self.log_index[j])
                    print (to_log, message)
                    sys.stdout.flush()
                    self._set_warning('Error found', self.log_index[j])
                    self.line_valid[j] = False
                if self.use_bar:
                    pbar.update(dbar)
                    sys.stdout.flush()
                if epoch == self._params['n_epochs'] - 1:
                    self.captains_log.at[self.log_index[j], 'shift_test_mean'] = np.mean(shift_test[self.log_index[j]])
                    self.captains_log.at[self.log_index[j],'shift_test_std'] = np.std(shift_test[self.log_index[j]])
        if self.use_bar:
            pbar.close()
        if self.save_figs:
            self.extraction_figs.save_plots()

        self.wave_matrix = np.array(self.wave_matrix)
        self._update_data()

    def _get_secz(self):
        secz = []
        for epoch in range(len(self._files)):
            if self._params['telescope'] == 'harps':
                aux = np.mean([fits.getheader(self._files[epoch])['HIERARCH ESO TEL AIRM END'],
                               fits.getheader(self._files[epoch])['HIERARCH ESO TEL AIRM START']])
                secz.append(aux)
            elif self._params['telescope'] == 'keck':
                secz.append(float(fits.getheader(self._files[epoch])['AIRMASS']))
            else:
                secz.append(float(fits.getheader(self._files[epoch])['SECZ']))
        return secz


    def _merge_epochs(self, n_epoch = 2, N_pixs=-1):
        if N_pixs == -1:
            N_pixs = self._params['N_pixs']
        epoch_len = len(self._files)/n_epoch
        c = 0
        while (epoch_len % n_epoch != 0):
            inds = np.array(range(len(self.is_out)))
            is_in = np.array([not aux for aux in self.is_out])
            inds_in = inds[is_in]
            if c % 2 == 0:
                self._times = np.delete(self._times, inds_in[0])
                self._files = np.delete(self._files, inds_in[0])
                self.exptime = np.delete(self.exptime, inds_in[0])
                self._raw_flux = np.delete(self._raw_flux, inds_in[0], axis=1)
                c = c + 1
            else:
                self._times = np.delete(self._times, inds_in[-1])
                self._files = np.delete(self._files, inds_in[-1])
                self.exptime = np.delete(self.exptime, inds_in[-1])
                self._raw_flux = np.delete(self._raw_flux, inds_in[0], axis=1)
                c = c + 1
            epoch_len = len(self._files)/n_epoch
        print ('You had to delete', str(c), 'epochs.')

        self.general_warnings += 'Deleted epochs: ' + str(c) + '. '
        epoch_len = int(epoch_len)
        self._params['n_epochs'] = epoch_len
        self._times = np.array([np.mean(self._times[i:i+n_epoch]) for i in range(0, len(self._files), n_epoch)])
        new_exptime = np.array([np.sum(self.exptime[i:i + n_epoch] for i in range(0, len(self._files), n_epoch))])
        self.secz = np.array([np.mean(self.secz[i:i + n_epoch]) for i in range(0, len(self._files), n_epoch)])
        self._times, self.is_out, self._ind_peri, tlen_mask, t_start = label_transit(self._times, period=self._params['period'],
                        dtransit=self._params['dtransit'], t_transit = self._params['t_transit'],
                        max_t_len = self._params['max_t_len'])
        self._params['t_start'] = t_start


        new_raw_flux = np.zeros((len(self.linedb), epoch_len, N_pixs))
        for j in range(len(self.linedb)):
            try:
                for pix in range(N_pixs):
                    epoch = 0
                    for i in range(0, len(self._files), n_epoch):
                        exptimes = self.exptime[i:i+n_epoch]
                        aux_flux = np.sum(self._raw_flux[j, i:i+n_epoch, pix]*exptimes)
                        new_raw_flux[j, epoch, pix] = aux_flux/np.sum(exptimes)
                        epoch +=1

            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Epoch merge: ' + message

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ',' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Epoch merge: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
        self._raw_flux = new_raw_flux
        self.exptime = new_exptime
        self._update_data()

    def _smart_merge_epochs(self, merge_as=[], n_epoch_merge=0):
        if len(merge_as) == 0 and n_epoch_merge > 0:
            n_epochs = self._params['n_epochs']
            is_in = np.array([not b for b in self.is_out])
            inds = np.array(range(n_epochs))
            inds_in = inds[is_in]
            c = 0
            while n_epochs % n_epoch_merge != 0:
                if c % 2 == 0:
                    dlt = 0
                else:
                    dlt = -1

                self._raw_flux = np.delete(self._raw_flux, inds_in[dlt], axis=1)
                self.exptime = np.delete(self.exptime, inds_in[dlt])
                self._files = np.delete(self._files, inds_in[dlt])
                self._times = np.delete(self._times, inds_in[dlt])
                self.secz = np.delete(self.secz, inds_in[dlt])
                self.is_out = np.delete(self.is_out, inds_in[dlt])
                if self._params['telescope'] == 'uves':
                    self._raw_error = np.delete(self._raw_error, inds_in[dlt], axis=1)

                n_epochs -= 1
                is_in = np.array([not b for b in self.is_out])
                inds = np.array(range(n_epochs))
                inds_in = inds[is_in]
                c += 1
            if c > 0:
                print ('You had to delete', str(c), 'epochs.')
                self.general_warnings += 'Deleted epochs: ' + str(c) + '. '
            merge_as = [str(i)+':'+str(i+n_epoch_merge+1) for i in range(0, len(inds), n_epoch_merge)]

        new_raw_flux = np.zeros((len(self.linedb), len(merge_as), self._params['N_pixs']))
        if self._params['telescope'] == 'uves':

            new_raw_error = np.zeros((len(self.linedb), len(merge_as), self._params['N_pixs']))
        #new_ematrix = np.zeros((len(self.linedb), len(merge_as), self._params['N_pixs']))
        new_times = []
        new_exptime = []
        new_secz = []
        self._params['n_epochs'] = len(merge_as)
        for inds, new_epoch in zip(merge_as, range(len(merge_as))):
            if ':' in inds:
                inds_ = inds.split(':')
                inds_ = [int(aux) for aux in inds_]
                new_times.append(np.mean(self._times[inds_[0]:inds_[1]]))
                new_exptime.append(np.sum(self.exptime[inds_[0]:inds_[1]]))
                new_secz.append(np.mean(self.secz[inds_[0]:inds_[1]]))
                for j in range(len(self.linedb)):
                    try:
                        for pix in range(self._params['N_pixs']):
                            aux_raw_flux = self._raw_flux[j, inds_[0]:inds_[1], pix]
                            exp_times = self.exptime[inds_[0]:inds_[1]]
                            if self._params['telescope'] != 'uves':
                                new_raw_flux[j, new_epoch, pix] = np.sum(aux_raw_flux*exp_times)/new_exptime[-1]
                            else:
                                aux_raw_error = self._raw_error[j, inds_[0]:inds_[1], pix]
                                uflux = unumpy.uarray(aux_raw_flux, aux_raw_error)
                                aux_uflux = np.sum(uflux*exp_times)/new_exptime[-1]
                                new_raw_flux[j, new_epoch, pix] = aux_uflux.n
                                new_raw_error[j, new_epoch, pix] = aux_uflux.s

                    except Exception as ex:
                        template = "{0}. Arguments: {1!r}"
                        message = template.format(type(ex).__name__, ex.args)
                        self.error_log.at[self.log_index[j], 'message'] = 'Epoch merge: ' + message

                        exc_type, exc_obj, exc_tb = sys.exc_info()
                        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                        to_log = str(type(ex).__name__) + ',' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                        self._set_status('Epoch merge: ' + to_log, self.log_index[j])
                        print (to_log, message)
                        sys.stdout.flush()
                        self._set_warning('Error found', self.log_index[j])
                        self.line_valid[j] = False
            else:
                new_times.append(self._times[int(inds)])
                new_exptime.append(self.exptime[int(inds)])
                new_secz.append(self.secz[int(inds)])
                for j in range(len(self.linedb)):
                    new_raw_flux[j, new_epoch, :] = self._raw_flux[j, int(inds), :]
        self._raw_flux = np.array(new_raw_flux)
        if self._params['telescope'] == 'uves':
            self._raw_error = new_raw_error
        self._times = np.array(new_times)
        self.exptime = np.array(new_exptime)
        self.secz = np.array(new_secz)
        self._times, self.is_out, self._ind_peri, tlen_mask, t_start = label_transit(self._times,
                                                                                     period=self._params['period'],
                                                                                     dtransit=self._params['dtransit'],
                                                                                     t_transit=self._params[
                                                                                         't_transit'],
                                                                                     max_t_len=self._params[
                                                                                         'max_t_len'])
        self._params['t_start'] = t_start
        self._update_data()
        self._check_times(figname='times_after_merge')






    def telluric_corrections(self, ind_plt = -1, all_plt = False, pbar = 'None', pix_plt = [50, 100, 150, 300, 350],
                             use_exp_model=True, debugging=False, epoch_plt=5, figsize=(15,7)):
        """
        After extracting the lines, this function makes the correction to reduce
        the error because of the air
        """

        print ()
        print ("Doing telluric corrections (3/8) to " + str(len(self.linedb)) + " lines.")
        print ()

        sys.stdout.flush()
        self.tell_masks = []
        self.secz = np.array(self.secz)

        if self._params['merge_in'] == 'telluric' and (self._params['n_epoch_merge'] > 1 or len(self._params['merge_epoch_as']) > 0):
            if not self._params['smart_merge']:
                self._merge_epochs(n_epoch=self._params['n_epoch_merge'])
            else:
                self._smart_merge_epochs(merge_as=self._params['merge_epoch_as'],
                                         n_epoch_merge=self._params['n_epoch_merge'])

        if self._params['n_epoch_merge'] == 0 and len(self._params['merge_epoch_as']) == 0:
            self._check_flux()
        is_out = self.is_out
        if self._params['tell_only_out']:
            secz = self.secz[is_out]
        else:
            secz = np.copy(self.secz)
        #tell_figs = Manage_plots(max = self._params['max_figs'])
        #programar mejor el plot
        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1
        use_exp_model0 = use_exp_model
        for j in range(len(self.linedb)):
            tell_info = False
            if ind_plt == j or all_plt:
                tell_info = True
            try:
                for pix in range(self._params['N_pixs']):
                    if self._params['tell_only_out']:
                        flux_epochs = self._raw_flux[j, is_out, pix]
                    else:
                        flux_epochs = self._raw_flux[j, :, pix]

                    if np.any(flux_epochs < 0):
                        warning = 'Negative flux values'
                        use_exp_model = True
                        self._set_warning(warning, self.log_index[j])
                        if debugging:
                            plt.close('all')
                            plt.plot(secz, flux_epochs)
                            plt.savefig('neg' + str(self.log_index[j]) + str(pix) + self.img_format, dpi=250)
                            plt.close('all')
                            plt.plot(self.wave_matrix[j], self.superline[j, epoch_plt, :])
                            plt.savefig('neg_wv' + str(self.log_index[j]) + self.img_format, dpi=250)
                            plt.close('all')

                    if use_exp_model:
                        m, n = guess_lineal_params(secz, np.log(np.abs(flux_epochs)))
                        p0 = [m, np.exp(n)]
                        params, pcov = curve_fit(f_model_exp, secz, flux_epochs, p0)
                        k, A = params
                        if not use_exp_model0:
                            use_exp_model = False
                    else:

                        params = np.polyfit(secz, np.log(np.abs(flux_epochs)), 1, w=np.sqrt(np.abs(flux_epochs)))
                        k = params[0]

                    if pix in pix_plt and self.save_figs:
                        plt.close('all')
                        sorted_secz = np.sort(secz)
                        x = np.linspace(sorted_secz[0], sorted_secz[-1], 50)
                        if use_exp_model:
                            fit = f_model_exp(x, k, A)
                            plt.plot(secz, flux_epochs, '*')
                        else:
                            fit = k*x + params[1]
                            plt.plot(secz, np.log(flux_epochs), '*')
                        plt.plot(x, fit)
                        plt.title('k: '+str(k))
                        plt.savefig(self.output_folder+self._str_adress+'/telluric/pix'+str(pix)+'_line_'+str(self.linedb[j])+ self.img_format)
                    self.tell_corrections[j, pix] = k

                tell_limit = self._params['tell_limit']
                if tell_limit != 0:
                    self.tell_masks.append(
                            np.array([-tell_limit <= aux and aux <= tell_limit for aux in self.tell_corrections[j]]))
                else:
                    self.tell_masks.append(np.ones((len(self.tell_corrections[j])), dtype=bool))




                """
                for epoch in range(len(self._times)):
                    corr_flux = correct_spectra(secz0[epoch], np.mean(self.secz[is_in]), corrections,
                                                self.superline[j, epoch, :])
                    self.superline[j, epoch, :] = corr_flux
                """

                if tell_info:
                    str_adress = self._str_adress
                    # print 'doing telluric corrections'
                    aux_name = str(int(self.linedb[j])) + '_ind_' + str(j) + '_ord_' + str(self._line_orders[j])
                    file_name = self.output_folder + str_adress + '/telluric/' + aux_name + '.png'
                    plt.close('all')
                    fig = plt.figure(figsize=figsize)
                    ax1 = fig.add_subplot(211, facecolor = 'floralwhite')
                    ax1.plot(self.wave_matrix[j], self.superline[j, 0, :], color = 'darkred')
                    ax1.axhline(y = 1.0, color = 'coral')
                    ax1.set_ylabel(r'Normalized flux')
                    ax2 = fig.add_subplot(212, facecolor = 'floralwhite')
                    line_center = self.line_centers[j]
                    dlambda = self._params['dlambda_analyze']
                    start1 = line_center - 1.5 * dlambda
                    end1 = line_center - 0.5 * dlambda
                    start2 = end1
                    end2 = line_center + 0.5 * dlambda
                    start3 = end2
                    end3 = line_center + 1.5 * dlambda
                    ax2.axvspan(start1, end1, alpha=0.3, color='lightcoral')
                    ax2.axvspan(start2, end2, alpha=0.7, color='coral')
                    ax2.axvspan(start3, end3, alpha=0.3, color='lightcoral')

                    ax1.axvspan(start1, end1, alpha=0.3, color='lightcoral')
                    ax1.axvspan(start2, end2, alpha=0.7, color='coral')
                    ax1.axvspan(start3, end3, alpha=0.3, color='lightcoral')

                    ax2.plot(self.wave_matrix[j], self.tell_corrections[j, :], 'royalblue')
                    ax2.set_xlabel(r'Wavelength $[\lambda]$')
                    plt.suptitle(r'Telluric corrections')

                    #plt.savefig(file_name)

                    self.checking_figs.add_plot(fig, file_name)
                    self._set_status('Success', self.log_index[j])

            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Telluric: ' + message

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ',' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Telluric: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
            if self.use_bar:
                pbar.update(dbar)
                sys.stdout.flush()
        if self.use_bar:
            pbar.close()

        if self.save_figs:
            self.telluric_figs.save_plots()

        self.tell_masks = np.array(self.tell_masks)
        self._renormalization(time_average=self._params['time_average_corr'])
        self._update_data()


    def _multiplot(self, j, figsize = (15, 20), fname = 'multiplot_', dpi = 300):
        from mpl_toolkits.mplot3d import Axes3D
        plt.close('all')
        str_adress = self._str_adress
        line = self.linedb[j]
        file_name = self.output_folder+str_adress+'/extract_lines/'+fname+str(line)+'.png'
        fig = plt.figure(figsize = figsize)
        grid = gridspec.GridSpec(2, 1)
        ax_align = fig.add_subplot(grid[0], facecolor = 'floralwhite')
        ax3d = fig.add_subplot(grid[1], projection='3d', facecolor ='floralwhite')
        for epoch in range(len(self._times)):
            ax_align.plot(self.wave_matrix[j], self.superline[j, epoch, :])
            epoch_array = epoch*np.ones((len(self.wave_matrix[j])))
            ax3d.plot(self.wave_matrix[j], epoch_array, zs =self.superline[j, epoch, :])

        fig.savefig(file_name, dpi = dpi)

    def _check_telluric(self, figsize = (15, 20), fname = 'multiplot', dpi = 300, ylims =[-0.08, 0.08],
                        after_telluric=False):
        from mpl_toolkits.mplot3d import Axes3D
        str_adress = self._str_adress
        for j in range(len(self.linedb)):
            plt.close('all')
            line = self.log_index[j]
            file_name = self.output_folder + str_adress + '/extract_lines/' + fname +'_'+ str(line) + '.png'
            fig = plt.figure(figsize=figsize)
            grid = gridspec.GridSpec(2, 1)
            ax_align = fig.add_subplot(grid[0], facecolor='floralwhite')

            if not after_telluric:
                ax3d = fig.add_subplot(grid[1], projection='3d', facecolor='floralwhite')
            else:
                ax_tell = fig.add_subplot(grid[1], facecolor='floralwhite')
            for epoch in range(len(self._times)):
                medians = np.median(self.superline[j, :, :], axis = 0)
                #print (len(medians))
                residuals = self.superline[j, epoch, :] - medians
                ax_align.plot(self.wave_matrix[j], residuals)
                ax_align.set_ylim(ylims)


                if not after_telluric:
                    epoch_array = epoch * np.ones((len(self.wave_matrix[j])))
                    ax3d.plot(self.wave_matrix[j], epoch_array, zs=residuals)
                else:
                    line_center = self.line_centers[j]
                    dlambda = self._params['dlambda_analyze']
                    start1 = line_center - 1.5 * dlambda
                    end1 = line_center - 0.5 * dlambda
                    start2 = end1
                    end2 = line_center + 0.5 * dlambda
                    start3 = end2
                    end3 = line_center + 1.5 * dlambda
                    ax_tell.axvspan(start1, end1, alpha=0.3, color='lightcoral')
                    ax_tell.axvspan(start2, end2, alpha=0.7, color='coral')
                    ax_tell.axvspan(start3, end3, alpha=0.3, color='lightcoral')
                    ax_tell.plot(self.wave_matrix[j], self.tell_corrections[j])
            self.extraction_figs.add_plot(fig, file_name)
            if self.save_figs:
                self.extraction_figs.save_plots()

    def _renormalization(self, epoch_plt=[0, 5, 61, 62, 63, 64], figsize = (10, 7), lw=0.5, N=-1, time_average=True):
        if N == -1:
            N_pixs = self._params['N_pixs']
        else:
            N_pixs = N
        self.superline = np.zeros((len(self.linedb), self._params['n_epochs'], N_pixs))
        self.error_matrix = np.zeros((len(self.linedb), self._params['n_epochs'], N_pixs))
        is_in = np.array([not aux for aux in self.is_out])
        indexFWHM = self._params['indexFWHM']
        min_pixs = self._params['norm_min_pixs']
        high = self._params['norm_high']
        width = self._params['norm_width']
        renorm_scale = self._params['renorm_scale']

        for j in range(len(self.linedb)):
            wave = self.wave_matrix[j]
            ind_min = np.argmin(np.abs(wave - self.line_centers[j]))
            continuum_test = []
            for epoch in range(self._params['n_epochs']):
                corr_flux = correct_spectra(self.secz[epoch], np.mean(self.secz[is_in]), self.tell_corrections[j, :],
                                                self._raw_flux[j, epoch, :])
                if self._params['renorm_scale'] == -1 and epoch == 0:
                    renorm_scale = np.median(self._raw_flux[j, 0, :]) / np.median(corr_flux)
                if self._params['telescope'] != 'uves':
                    new_error = np.array([np.sqrt(aux/(self._params['error_gain']*self.exptime[epoch])) for aux in self._raw_flux[j, epoch, :]])
                else:
                    new_error = self._raw_error[j, epoch, :]
                corr_error = correct_spectra(self.secz[epoch], np.mean(self.secz[is_in]), self.tell_corrections[j, :],
                                                new_error)
                y0 = np.copy(corr_flux)
                norm_flux, norm_error, pol, uplimit, mfilter, lowlimit, filter_mask, message = normalize(wave, corr_flux,
                                                                                        corr_error, indexFWHM, min_pixs,
                                                                                        high/renorm_scale, width,
                                                                                        polyorder=self._params['norm_polyorder'])
                continuum_test.append(filter_mask.sum())
                if len(corr_flux) < N or N == -1:
                    analyze_mask = np.ones(len(corr_flux), dtype=bool)
                else:
                    analyze_mask, ind_min = extract_array_with_mask(wave, ind_min, N)

                if message != 'ok':
                    self.line_valid[j] = False
                    self.error_log.at[self.log_index[j], 'message'] = 'Renorm: ' + message
                    print (message)
                    sys.stdout.flush()
                    self._set_status('Emptiness error', self.log_index[j])
                    self._set_warning('Error found', self.log_index[j])
                    continue
                self.superline[j, epoch, :] = norm_flux[analyze_mask]
                self.error_matrix[j, epoch, :] = norm_error[analyze_mask]

                if epoch in epoch_plt:
                    plt.close('all')
                    fig = plt.figure(figsize=figsize)
                    grid = gridspec.GridSpec(2,3, height_ratios=[0.1, 1])
                    ax_raw = fig.add_subplot(grid[:, 0], facecolor= 'floralwhite')
                    ax_raw.plot(wave, self._raw_flux[j, epoch, :], linewidth=lw)
                    ax_raw.set_title('Raw flux')

                    ax_tell = fig.add_subplot(grid[0,1], facecolor='floralwhite')
                    ax_tell.plot(wave, self.tell_corrections[j, :], linewidth=lw)
                    ax_tell.set_title('Telluric corrections')
                    ax_error = fig.add_subplot(grid[0,2], facecolor='floralwhite')
                    ax_error.plot(wave, corr_error, linewidth=lw)
                    ax_error.set_title('Error')

                    ax_corr = fig.add_subplot(grid[1,1], facecolor='floralwhite', sharex=ax_tell)
                    ax_corr.plot(wave, y0, linewidth=lw)
                    ax_corr.plot(wave[filter_mask], y0[filter_mask], '+', color = 'darkviolet', markersize=1)
                    ax_corr.fill_between(wave, lowlimit, uplimit, facecolor='lightcoral', alpha=0.9)
                    ax_corr.plot(wave, pol, linewidth=lw, color='darkgreen')

                    ax_norm = fig.add_subplot(grid[1,2], facecolor='floralwhite', sharex=ax_error)
                    ax_norm.plot(wave, norm_flux, linewidth=lw)
                    plt.tight_layout()

                    aux_name = str(int(self.linedb[j])) + '_ind_' + str(j) + '_ord_' + str(self._line_orders[j])
                    file_name = self.output_folder + self._str_adress + '/other_epoch/' + aux_name + 'renorm_ep_' + str(
                            epoch) + self.img_format
                    self.renorm_figs.add_plot(fig, file_name)
            self.ref_flux[j, :] = np.copy(self.superline[j, 0, :])
            if time_average:
                for pix in range(N_pixs):
                    median_norm = np.median(self.superline[j, :, pix])
                    self.superline[j, :, pix] /= median_norm
                    self.error_matrix[j, :, pix] /= median_norm
            self.captains_log.at[self.log_index[j], 'cont_test_mean'] = np.mean(continuum_test)
            self.captains_log.at[self.log_index[j], 'cont_test_std'] = np.std(continuum_test)

        if self.save_figs:
            self.renorm_figs.save_plots()



    def snellen(self, do_plt = True, dlambda = -1, pbar = 'None'):
        if dlambda == -1:
            dlambda = self._params['dlambda_analyze']

        """
        this method makes the analysis of Snellen 2008, i.e., it plots the transit near
        the line.
        """
        self.snellen_data = []
        self.mean_snellen = []

        print ()
        print ("Doing Snellen analysis (4/8) to " + str(len(self.linedb)) + " lines.")
        print ()
        sys.stdout.flush()

        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1

        for j in range(len(self.linedb)):
            try:
                wave = self.wave_matrix[j]
                ftransit = np.array([])
                error_transit = np.array([])
                alpha_mid = self._params['alpha_mid']
                alpha_side = self._params['alpha_side']
                first = True
                go_on = True
                mask = self.tell_masks[j]
                for epoch in range(len(self._times)):
                    flux = self.superline[j, epoch, :]
                    error = self.error_matrix[j, epoch, :]
                    mu = self.line_centers[j]
                    signal, err_ftransit, test, message = calculate_signal_snellen(wave, flux, error, mu, dlambda,
                                                        mask, tests = first, alpha_mid=alpha_mid, alpha_side=alpha_side)
                    if first:
                        first = False
                        if not test:
                            self._set_warning(message, self.log_index[j])
                            self.error_log.at[self.log_index[j], 'message'] = 'Discarded: '+message
                            self.line_valid[j] = False
                            go_on = False
                            break

                    if go_on:
                        ftransit = np.append(ftransit, signal)
                        error_transit = np.append(error_transit, err_ftransit)

                if go_on:
                    aux_data = []
                    average_array, av_out, av_in = average_flux(ftransit, self.is_out)
                    ftransit = (ftransit / av_out) - 1
                    average_array = (average_array / av_out) - 1
                    error_transit = error_transit / av_out
                    time_array = np.linspace(0, 1, len(ftransit))

                    aux_data.append(ftransit)
                    aux_data.append(average_array)
                    aux_data.append(error_transit)
                    self.snellen_data.append(aux_data)
                    self.mean_snellen.append([av_in, av_out])
                    if do_plt:
                        str_adress = self._str_adress
                        aux_name = str(int(self.linedb[j])) + '_' + 'ord' + str(self._line_orders[j])
                        file_name = self.output_folder + str_adress + '/snellen/' + aux_name + '.png'
                        title = 'S2008 for line ' + str(self.linedb[j])
                        plt.close('all')
                        plt.errorbar(time_array, ftransit, yerr=error_transit, fmt='o')
                        plt.plot(time_array, average_array)
                        plt.title(title)
                        plt.ylabel(r'$F_{In}/F_{Out}-1$')
                        plt.xlabel('Time')
                        plt.savefig(file_name)
                        plt.clf()
                    self._set_status('Success', self.log_index[j])

            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Snellen: ' + message
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Snellen: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
            if self.use_bar:
                pbar.update(dbar)
                sys.stdout.flush()
        if self.use_bar:
            pbar.close()


        self._update_data()

        self.snellen_data = np.array(self.snellen_data)
        self.mean_snellen = np.array(self.mean_snellen)


    def _get_relative_flux(self, wave, flux_line, bool_array, lambda0, dlambda, mask, tests = False, do_print = False):
        """
        this method takes a matrix of flux to calculate the relative flux of a transit,
        used for the Redfield 2008 analysis.
        """
        fright = np.array([])
        fmid = np.array([])
        fleft = np.array([])


        for i in range(len(bool_array)):
            aux_fleft, aux_fmid, aux_fright, test_, message_ = calculate_bands(wave, flux_line[i], lambda0, dlambda,
                                                                               mask, tests = tests, do_print = do_print)
            fright = np.append(fright, aux_fright)
            fmid = np.append(fmid, aux_fmid)
            fleft = np.append(fleft, aux_fleft)
        avOutLeft, avInLeft = average_outin(fleft, bool_array)
        avOutMid, avInMid = average_outin(fmid, bool_array)
        avOutRight, avInRight = average_outin(fright, bool_array)
        frelLeft = avInLeft / avOutLeft
        frelMid = avInMid / avOutMid
        frelRight = avInRight / avOutRight
        """
        import pdb
        if np.sum(sp.isnan([frelMid, frelLeft, frelRight])) > 0:
            print 'problem!'
            pdb.set_trace()
        """
        return frelMid - (frelLeft + frelRight)/2


    def redfield(self, do_plt = True, dlambda = -1, pbar = 'None', ind_print = -1,
                 max_comb_percent = 0.2, force_no_repeat = True, min_remove = 2):
        """
        this method makes the Redfield 2008 analysis loops times.
        """
        loops = self._params['redf_loops']
        if dlambda == -1:
            dlambda = self._params['dlambda_analyze']

        self.redfield_data = []

        print ()
        print ("Doing Redfield analysis (5/8) to " + str(len(self.linedb)) + " lines.")
        print ()

        sys.stdout.flush()
        is_out = self.is_out
        out_out = is_out[is_out]
        is_in = np.array([not b for b in is_out])
        in_in = is_out[is_in]
        ratio = len(out_out)/len(is_out)
        n_out = int(len(out_out)*ratio + 0.5)
        n_in = int(len(in_in)*ratio + 0.5)
        critic_out = sp.special.comb(len(out_out), n_out)
        critic_in = sp.special.comb(len(in_in), n_in)

        loops_outout = loops
        loops_inin = loops
        critic_ints = range(min_remove, int(len(in_in)/2+0.5) + 1)
        comb_per_int = np.array([sp.special.comb(len(in_in), i) for i in critic_ints])
        critic_inout = comb_per_int.sum()

        if critic_out < loops:
            self.outout_ok = False
            loops_outout = int(critic_out)

        if critic_in < loops:
            self.inin_ok = False
            loops_inin = int(critic_in)

        if not self.outout_ok or force_no_repeat:
            print ('Calculating C(', len(out_out), ',', n_out, ') =', critic_out)
            all_isoutout = np.random.permutation(create_all_isout(len(out_out), n_out, max_len=loops_outout))

        if not self.inin_ok or force_no_repeat:
            print ('Calculating C(', len(in_in), ',', n_in, ') =', critic_in)
            all_isinin = np.random.permutation(create_all_isout(len(in_in), n_in, max_len=loops_inin))

        self._params['isout_ok'] = True
        if critic_inout < loops:
            loops = critic_inout
            self.isout_ok = False
            self._params['isout_ok'] = False


        if (loops > max_comb_percent*critic_inout or force_no_repeat or not self.isout_ok):
            print ('There will not be repetitions.')
            do_changeisout = False
            combs = []
            aux = []
            for step in critic_ints:
                critic_len = int((loops - len(aux))/(critic_ints[-1] + 1 - step) +0.5)
                aux = create_all_isout(len(in_in), step, max_len=critic_len)
                combs.extend(aux)
            self.combs = combs
        else:
            do_changeisout = True
            all_n_remove = []
            critic_len = loops/len(comb_per_int)
            for i in range(len(comb_per_int)):
                curr_int = critic_ints[i]
                if comb_per_int[i] > critic_len:
                    all_n_remove.extend(curr_int*np.ones((int(critic_len + 0.5))))
                else:
                    print (comb_per_int[i])
                    all_n_remove.extend(curr_int*np.ones((int(comb_per_int[i] + 0.5))))
                    critic_len = (loops - len(all_n_remove) + 1)/(len(comb_per_int) - i - 1)
            self.all_n_remove = all_n_remove
            print (len(self.all_n_remove), loops)
            self.comb_per_int = comb_per_int

        self.loops = [loops, loops_inin, loops_outout]
        print (self.loops)


        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = int(len(self.linedb)*(loops + loops_outout + loops_inin)))
        dbar = 1

        print ('Critic comb numbers:', critic_inout, critic_out, critic_in)

        if not self.outout_ok or not self.inin_ok:
            text = 'Not enough data to make out-out or in-in loops'
            print ('Warning: ' + text)
            self.general_warnings += text + '. '

        for j in range(len(self.linedb)):
            try:
                wave = self.wave_matrix[j]
                line_center = self.line_centers[j]
                result_inin = np.array([])
                result_outout = np.array([])
                result_inout = np.array([])
                aux_data = []

                mask_tell = self.tell_masks[j]
                sys.stdout.flush()

                for l in range(int(loops_outout)):
                    # Out-Out
                    flux_out = self.superline[j, is_out, :]
                    if self.outout_ok and not force_no_repeat:
                        new_outout = change_is_out(len(out_out), n_out)
                    else:
                        new_outout = all_isoutout[l]

                    if j == ind_print:
                        do_print = True
                    else:
                        do_print = False
                    frel = self._get_relative_flux(wave, flux_out, new_outout, line_center,
                                                   dlambda, mask_tell, do_print = do_print)
                    result_outout = np.append(result_outout, frel)
                    if self.use_bar:
                        pbar.update(dbar)
                        sys.stdout.flush()

                for l in range(int(loops_inin)):

                    # In-In
                    if self.inin_ok and not force_no_repeat:
                        new_inin = change_is_out(len(in_in), n_in)
                    else:
                        new_inin = all_isinin[l]

                    flux_in = self.superline[j, is_in, :]
                    frel = self._get_relative_flux(wave, flux_in, new_inin, line_center,
                                                   dlambda, mask_tell)
                    result_inin = np.append(result_inin, frel)
                    if self.use_bar:
                        pbar.update(dbar)
                        sys.stdout.flush()
                #print ('loops:', len(all_n_remove))
                for l in range(int(loops)):
                    # In-Out
                    if do_changeisout and self.isout_ok:
                        n_remove = all_n_remove[l]
                        mask_remove = change_is_out(len(in_in), int(n_remove))
                    else:
                        mask_remove = combs[l]

                    mask, new_is_out = mask_redfield(is_out, is_in, mask_remove)
                    flux = self.superline[j, mask, :]
                    frel = self._get_relative_flux(wave, flux, new_is_out, line_center,
                                                   dlambda, mask_tell)
                    result_inout = np.append(result_inout, frel)
                    if self.use_bar:
                        pbar.update(dbar)
                        sys.stdout.flush()

                aux_data.append(result_inout)
                aux_data.append(result_outout)
                aux_data.append(result_inin)
                self.redfield_data.append(aux_data)


            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Redfield: ' + message

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Redfield: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
                if self.use_bar:
                    pbar.update(loops)

        if self.use_bar:
            pbar.close()


        self._update_data()

        self.redfield_data = np.array(self.redfield_data)



    def _band_to_str(self, band):
        str_band = str(band)
        aux_str = str_band.split('.')
        if aux_str[1] != '0':
            return str(aux_str[0])+str(aux_str[1])
        else:
            return str(aux_str[0])


    def summary_plots(self, ind_plt=-1, show_all=False, near_line=20, dlambda=-1, f=np.median,
                      calculate_center = True, pbar = 'None',
                      fontsizes = [27, 24, 25], text_pad = 0.5, do_width_plt = False, binning = True,
                      text_coords = [0.5, 2.15, 0.5, 0.03], lw = 3.5,
                      height_ratios=[1,1,5], width_ratios=[1, 1, 1, 0.1], do_redf_ttest = False,
                      figsize=(13, 7), do_snellen_ttest=False, add_block=False, list_params=False,
                      bnw = False, spectra_offset = 0.8, show_time_average=False,
                      bnw_bgcolor='gainsboro', adjust= [0.15, 0.94, 0.18, 0.98], fname_verbose=True):

        print ()
        print ("Doing Summary plots (6/8) to " + str(len(self.linedb)) + " lines.")
        print ()
        sys.stdout.flush()

        normed = (not self.outout_ok or not self.inin_ok) or self._params['normalize_hists']
        if dlambda == -1:
            dlambda = self._params['dlambda_analyze']

        cols = ['elem', 'band', 'log_gf', 'wave', 'time', 'tspectra', 'err_tspectra',
                'tlight_curve', 'err_tlight_curve']
        self._tspectra = pd.DataFrame(index=self.log_index, columns=cols)
        self._tspectra['band'] = dlambda
        str_adress = self._str_adress
        omit_hists = self._params['omit_hists']

        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1


        for j in range(len(self.linedb)):
            try:
                plt.close('all')
                bg_color = None
                if bnw: bg_color=bnw_bgcolor
                fig = plt.figure(figsize=figsize)
                gs = gridspec.GridSpec(3, 4, height_ratios=height_ratios, width_ratios=width_ratios, hspace=0.02,
                                       wspace=0.02)
                ax_corr = fig.add_subplot(gs[0, 0], facecolor=bg_color)
                ax_norm = fig.add_subplot(gs[1, 0], sharex=ax_corr, facecolor=bg_color)
                ax_wave_transit = fig.add_subplot(gs[2, 0], sharex=ax_norm, facecolor=bg_color)
                ax_snellen = fig.add_subplot(gs[2, 1], sharey=ax_wave_transit, facecolor=bg_color)
                ax_redf = fig.add_subplot(gs[2, 2], sharey=ax_snellen, facecolor=bg_color)
                ax_sigma = fig.add_subplot(gs[2, 3], sharey=ax_redf, facecolor=bg_color)
                if add_block: ax_info = fig.add_subplot(gs[0, 1])

                # General setups
                plt.setp(ax_corr.get_xticklabels(), visible=False)
                plt.setp(ax_norm.get_xticklabels(), visible=False)

                ax_wave_transit.set_ylabel(r'$\mathcal{F}_{in}/\mathcal{F}_{out} - 1$', fontsize=fontsizes[2])
                ax_wave_transit.set_xlabel(r'Wavelength[$\AA$]', fontsize=fontsizes[2])

                plt.setp(ax_snellen.get_yticklabels(), visible=False)
                ax_snellen.set_xlabel(r'Phase', fontsize=fontsizes[2])

                plt.setp(ax_redf.get_yticklabels(), visible=False)
                ax_redf.set_xlabel(r'Number', fontsize=fontsizes[2])

                plt.setp(ax_sigma.get_yticklabels(), visible=False)
                plt.setp(ax_sigma.get_xticklabels(), visible=False)
                ax_sigma.set_xlabel('\n' + r'$\sigma$', fontsize=fontsizes[2])
                ax_snellen.set_title('Light curve', fontsize=fontsizes[2])
                ax_redf.set_title('Bootstrap', fontsize=fontsizes[2])
                #plt.suptitle(r'Summary plots for ' + str(self.linedb[j]) + ' for element ' + str(self.str_elem[j]))

                # some needed parameters
                x = self.wave_matrix[j]
                line_center = self.line_centers[j]
                ind_center = self._ind_lines[j]
                start1 = line_center - 1.5 * dlambda
                end1 = line_center - 0.5 * dlambda
                start2 = end1
                end2 = line_center + 0.5 * dlambda
                start3 = end2
                end3 = line_center + 1.5 * dlambda
                ind_start = np.argmin(np.abs(x - start1)) - near_line
                ind_end = np.argmin(np.abs(x - end3)) + near_line


                # tell corrections
                corr = self.tell_corrections[j]
                mask_near = np.array([ind_start <= i and i <= ind_end for i in range(len(x))])
                mask_tell = self.tell_masks[j][mask_near]
                x_near = x[mask_near]
                corr_near = corr[mask_near]

                if bnw:
                    #             0          1       2         3       4
                    colors = ['lightgray', 'gray', 'grey', 'black', 'gray', 'k', 'dimgray', 'darkgrey']
                else:
                    #             0             1         2         3           4
                    colors = ['lightcoral', 'coral', 'orangered', 'navy', 'forestgreen',
                              'darkgreen', 'darkblue', 'darkred']
                ax_corr.plot(x_near[mask_tell], corr_near[mask_tell], linewidth=lw*0.5, color=colors[3])
                ax_corr.axvspan(start1, end1, alpha=0.3, color=colors[0])
                ax_corr.axvspan(start2, end2, alpha=0.7, color=colors[1])
                ax_corr.axvspan(start3, end3, alpha=0.3, color=colors[0])
                ax_corr.axvline(x=line_center, color=colors[2], linestyle='--', linewidth=lw*0.7)
                tell_limit = self._params['tell_limit']
                #print '1'
                if tell_limit != 0:

                    #ax_corr.plot(x_near[mask_tell], corr_near[mask_tell], ',', color = 'darkred')
                    ax_corr.axhline(y = tell_limit, linewidth = lw*0.4, color = colors[7])
                    ax_corr.axhline(y=-tell_limit, linewidth=lw*0.4, color=colors[7])


                # norm data
                y = self.ref_flux[j, mask_near]
                ax_norm.plot(x_near[mask_tell], y[mask_tell], linewidth=lw*0.7, color=colors[3])
                # ax_norm.plot(x[ind_center], y[ind_center], '*m')
                ax_norm.axvspan(start1, end1, alpha=0.3, color=colors[0])
                ax_norm.axvspan(start2, end2, alpha=0.7, color=colors[1])
                ax_norm.axvspan(start3, end3, alpha=0.3, color=colors[0])
                ax_norm.axvline(x=line_center, color=colors[2], linestyle='--', linewidth=lw*0.7)



                # transit vs wave data
                is_out = self.is_out
                is_in = np.array([not b for b in is_out])
                flux_out = self.superline[j, is_out, :]
                flux_in = self.superline[j, is_in, :]
                err_matrix = np.array(self.error_matrix)
                error_out = err_matrix[j, is_out, :]
                error_in = err_matrix[j, is_in, :]
                l = len(x)
                lo = len(is_out[is_out])
                li = len(is_in[is_in])
                flux_per_lambda_out = np.array([np.mean(flux_out[:, i]) for i in range(l)])
                flux_per_lambda_in = np.array([np.mean(flux_in[:, i]) for i in range(l)])
                err_per_lambda_out = np.array([np.sqrt(np.sum(error_out[:, i] ** 2)) / lo for i in range(l)])
                err_per_lambda_in = np.array([np.sqrt(np.sum(error_in[:, i] ** 2)) / li for i in range(l)])
                u_flux_out = unumpy.uarray(flux_per_lambda_out, err_per_lambda_out)
                u_flux_in = unumpy.uarray(flux_per_lambda_in, err_per_lambda_in)
                flux_to_plot = u_flux_in / u_flux_out

                self._tspectra.at[self.log_index[j], 'elem'] = self.str_elem[j]
                self._tspectra.at[self.log_index[j], 'log_gf'] = self.log_gf[j]
                self._tspectra.at[self.log_index[j], 'wave'] = x_near
                self._tspectra.at[self.log_index[j], 'tspectra'] = np.array([aux.n for aux in flux_to_plot])[mask_near]
                self._tspectra.at[self.log_index[j], 'err_tspectra'] = np.array([aux.s for aux in flux_to_plot])[mask_near]

                elinewidth = lw*0.5
                if self._params['time_average_corr'] and show_time_average:
                    previous_mask_near = np.copy(mask_near)
                    previous_x = np.copy(x)

                if binning:
                    k = 0
                    n_binning = self._params['n_binning'] + 1
                    new_flux_to_plot = []
                    new_x = []
                    while (k < len(flux_to_plot)):
                        new_flux_to_plot.append(np.mean(flux_to_plot[k:k+n_binning]))
                        new_x.append(x[k +int(n_binning/2)])
                        k = k + n_binning
                    flux_to_plot = new_flux_to_plot

                    x = np.array(new_x)
                    ind_start = np.argmin(np.abs(x - start1)) - int(near_line/n_binning)
                    ind_end = np.argmin(np.abs(x - end3)) + int(near_line/n_binning)
                    mask_near = np.array([ind_start <= i and i < ind_end for i in range(len(x))])
                    if self._params['tell_limit'] == 0:
                        mask_tell = np.ones((len(mask_near[mask_near])), dtype=bool)
                    else:
                        mask_tell = np.array([not np.any(mask_tell[i:i+n_binning]==False) for i in range(0, len(mask_tell), n_binning)])
                    elinewidth = lw*0.7


                wt_values = np.array([aux.n for aux in flux_to_plot])[mask_near]
                wt_error = np.array([aux.s for aux in flux_to_plot])[mask_near]
                x_near = x[mask_near]


                ax_wave_transit.axvspan(start1, end1, alpha=0.3, color=colors[0], zorder=0)
                ax_wave_transit.axvspan(start2, end2, alpha=0.7, color=colors[1], zorder=0)
                ax_wave_transit.axvspan(start3, end3, alpha=0.3, color=colors[0], zorder=0)
                ax_wave_transit.axvline(x=line_center, color=colors[2], linestyle='--', linewidth=lw * 0.7, zorder=1)
                if self._params['time_average_corr'] and show_time_average:
                    ax_wave_transit.plot(previous_x[previous_mask_near], self.superline[j, 0, previous_mask_near] - spectra_offset*np.median(self.superline[j, 0, previous_mask_near]),
                                 color=colors[3])
                ax_wave_transit.axhline(y=0.0, linewidth=lw * 0.9, color=colors[4], linestyle='--')
                ax_wave_transit.errorbar(x_near[mask_tell], wt_values[mask_tell] - 1, yerr=wt_error[mask_tell], fmt='.',
                                         color=colors[3], elinewidth=elinewidth, capsize=1.2, zorder=3)




                # snellen data
                snellen_data = self.snellen_data[j]
                dtransit = self._params['dtransit']
                t_center = self._params['t_start'] + dtransit / 2
                time_array = (np.array(self._times) - t_center)/dtransit
                mean_snellen = snellen_data[1]
                self.captains_log.at[self.log_index[j], 'snellen_in'] = self.mean_snellen[j, 0]
                self.captains_log.at[self.log_index[j], 'snellen_out'] = self.mean_snellen[j, 1]

                self._tspectra.at[self.log_index[j], 'time'] = time_array
                self._tspectra.at[self.log_index[j], 'tlight_curve'] = snellen_data[0]
                self._tspectra.at[self.log_index[j], 'err_tlight_curve'] = snellen_data[2]

                ax_snellen.plot(time_array, mean_snellen, color=colors[4])
                if self._params['plot_time_binning'] < 2:
                    ax_snellen.errorbar(time_array, snellen_data[0], yerr=snellen_data[2],
                                        fmt='o', elinewidth=lw*0.7, color=colors[3])
                else:
                    time_binning = self._params['plot_time_binning']
                    binned_time = []
                    binned_flux = []
                    uflux = unumpy.uarray(snellen_data[0], snellen_data[2])
                    k = 0
                    while (k < len(time_array)):
                        binned_flux.append(np.mean(uflux[k:k + time_binning]))
                        binned_time.append(np.mean(time_array[k:k + time_binning]))
                        k += time_binning
                    flux_per_time = np.array([aux.n for aux in binned_flux])
                    error_per_time = np.array([aux.s for aux in binned_flux])
                    ax_snellen.errorbar(binned_time, flux_per_time, yerr=error_per_time,
                                        fmt='o', elinewidth=lw * 0.7, color=colors[3])



                ax_snellen.axhline(y=0.0, linewidth=lw*0.9, color=colors[4], linestyle='--')

                # redfield data
                redf_data = self.redfield_data[j]
                min_data = []
                max_data = []
                for k in range(3):
                    min_data.append(np.min(redf_data[k]))
                    max_data.append(np.max(redf_data[k]))
                mmax = np.max(np.array(max_data))
                mmin = np.min(np.array(min_data))
                n_bins = self._params['redfield_bins']
                num_bins = np.linspace(mmin, mmax, n_bins)

                N_hist = [1, 1, 1]

                (n0, bins0, patches0) = ax_redf.hist(redf_data[0]/N_hist[0], num_bins, edgecolor=colors[5], facecolor='None',
                                                     orientation='horizontal', histtype='step', linewidth=lw*1.2, normed = normed)

                if self.outout_ok or not omit_hists:
                    (n1, bins1, patches1) = ax_redf.hist(redf_data[1]/N_hist[1], num_bins, edgecolor=colors[6], facecolor='None',
                                                     orientation='horizontal', histtype='step', linewidth=lw*0.9, normed = normed)
                if self.inin_ok or not omit_hists:
                    (n2, bins2, patches2) = ax_redf.hist(redf_data[2]/N_hist[2], num_bins, edgecolor=colors[7], facecolor='None',
                                                     orientation='horizontal', histtype='step', linewidth=lw*0.9, normed = normed)
                ax_redf.axhline(y=0.0, linewidth=lw*0.9, color=colors[4], linestyle='--')

                sigma0, center0 = get_histogram_width(redf_data[0])
                sigma1, center1 = get_histogram_width(redf_data[1])
                sigma2, center2= get_histogram_width(redf_data[2])

                s0 = sigma0
                s1 = sigma1
                s2 = sigma2
                if calculate_center:
                    c0 = center0
                    c1 = center1
                    c2 = center2
                else:
                    c0 = f(redf_data[0])
                    c1 = f(redf_data[1])
                    c2 = f(redf_data[2])

                if do_width_plt:
                    ax_width = fig.add_subplot(gs[0:2, 1], facecolor = bg_color)
                    (n0, bins0, patches0) = ax_width.hist(redf_data[0], num_bins, edgecolor='darkgreen', facecolor='None',
                                                      histtype='step', linewidth=lw*1.2)
                    (n1, bins1, patches1) = ax_width.hist(redf_data[1], num_bins, edgecolor='darkblue', facecolor='None',
                                  histtype='step', linewidth=lw*0.9)
                    (n2, bins2, patches2) = ax_width.hist(redf_data[2], num_bins, edgecolor='darkred', facecolor='None',
                                   histtype='step', linewidth=lw*0.9)

                    ax_width.set_xlim()
                    ax_redf.axhline(y = c0, color = 'forestgreen', linestyle = '--', linewidth = lw*1)
                    ax_width.axvline(x = c0, color = 'forestgreen', linestyle = '--', linewidth = lw*1)
                    ax_redf.axhline(y=c0 + s0, color='forestgreen', linestyle = '--', linewidth = lw*0.8)
                    ax_width.axvline(x=c0 + s0, color='forestgreen', linestyle = '--', linewidth=lw*0.8)
                    ax_redf.axhline(y=c0 - s0, color='forestgreen', linestyle='--', linewidth = lw*0.8)
                    ax_width.axvline(x=c0 - s0, color='forestgreen', linestyle = '--', linewidth=lw*0.8)

                    ax_redf.axhline(y=c1, color='navy', linestyle = '--', linewidth = lw*1)
                    ax_width.axvline(x=c1, color='navy', linestyle = '--', linewidth=lw*1)
                    ax_redf.axhline(y=c1 + s1, color='navy', linestyle='--', linewidth = lw*0.8)
                    ax_width.axvline(x=c1 + s1, color='navy', linestyle = '--', linewidth=lw*0.8)
                    ax_redf.axhline(y=c1 - s1, color='navy', linestyle='--', linewidth = lw*0.8)
                    ax_width.axvline(x=c1 - s1, color='navy', linestyle = '--', linewidth=lw*0.8)

                    ax_redf.axhline(y=c2, color='firebrick', linestyle = '--', linewidth = lw*1)
                    ax_width.axvline(x=c2, color='firebrick', linestyle = '--', linewidth=lw*1)
                    ax_redf.axhline(y=c2 + s2, color='firebrick', linestyle='--', linewidth = lw*0.8)
                    ax_width.axvline(x=c2 + s2, color='firebrick', linestyle = '--', linewidth=lw*0.8)
                    ax_redf.axhline(y=c2 - s2, color='firebrick', linestyle='--', linewidth = lw*0.8)
                    ax_width.axvline(x=c2 - s2, color='firebrick', linestyle = '--', linewidth=lw*0.8)

                    fmt = '.'
                    elinewidth = lw*0.9
                    capsize = lw*1.2
                    ypos = np.max(n0)*0.6
                    ax_width.errorbar([center0], ypos, xerr = [sigma0], fmt=fmt, elinewidth=elinewidth,
                                  color='darkgreen', capsize=capsize)
                    ax_width.errorbar([center1], ypos,  xerr=[sigma1], fmt=fmt, elinewidth=elinewidth,
                                      color='darkblue', capsize=capsize)
                    ax_width.errorbar([center2], ypos, xerr=[sigma2], fmt=fmt, elinewidth=elinewidth,
                                      color='darkred', capsize=capsize)
                    aux_lim = np.max([bins1[-1], bins2[-1]])
                    ax_width.set_xlim([aux_lim, bins0[0]])
                    plt.setp(ax_width.get_yticklabels(), visible=False)
                    plt.setp(ax_width.get_xticklabels(), visible=False)

                line = self.log_index[j]
                self.captains_log.at[line, 'centerIO'] = c0
                self.captains_log.at[line, 'centerOO'] = c1
                self.captains_log.at[line, 'centerII'] = c2
                self.captains_log.at[line, 'sigmaIO'] = s0
                self.captains_log.at[line, 'sigmaOO'] = s1
                self.captains_log.at[line, 'sigmaII'] = s2

                n_sigmasOO = np.round(get_hist_center(c0, s0, c1, s1), 3)
                n_sigmasII = np.round(get_hist_center(c0, s0, c2, s2), 3)
                n_sigmas0 = np.round(get_hist_center(c0, s0, 0.0, 1), 2)

                #Writing very important info

                aux_ttext = ''
                if do_redf_ttest:
                    from scipy.stats import norm, kruskal, ks_2samp, ttest_ind
                    #from statsmodels.stats.weightstats import ttest_ind
                    out_out_test = ks_2samp(redf_data[1], redf_data[0])
                    in_in_test = ks_2samp(redf_data[2], redf_data[0])
                    perm_len = np.min([len(redf_data[1]), len(redf_data[2])])
                    all_test = kruskal(np.random.permutation(redf_data[0])[:perm_len], redf_data[1], redf_data[2])
                    aux_ttext += '\n' + r'$\bullet$'+' OO ttest: ' + str(
                        round(out_out_test[0], 5))+ ', '+str(round(norm.ppf(out_out_test[1]),5))
                    aux_ttext += '\n' + r'$\bullet$'+' II ttest: ' + str(
                        round(in_in_test[0], 5))+ ', '+str(round(norm.ppf(in_in_test[1]), 5))
                    aux_ttext += '\n' + r'$\bullet$' + ' All test: ' + str(
                        round(all_test[0], 5)) + ', ' + str(round(norm.ppf(all_test[1]), 5))
                    print (1 - out_out_test[1], norm.ppf(1 - out_out_test[1]))
                    print (1 - in_in_test[1], norm.ppf(1 - in_in_test[1]))
                    print (all_test, norm.ppf(all_test[1]))

                if do_snellen_ttest:
                    from scipy.stats import norm, ttest_ind
                    #from statsmodels.stats.weightstats import ttest_ind
                    delete_mask = np.array([self.is_out[i]==self.is_out[i+1] for i in range(len(self.is_out) - 1)])
                    delete_mask = np.append(delete_mask, True)
                    self.delete_mask = delete_mask
                    aux_isout = self.is_out[delete_mask]
                    aux_isin = np.array([not b for b in aux_isout])
                    aux_snellen = np.array(snellen_data[0])[delete_mask]
                    out_data = aux_snellen[aux_isout]
                    in_data = aux_snellen[aux_isin]
                    snellen_ttest = ttest_ind(out_data, in_data, equal_var=False)


                if add_block:
                    text_plot = r'$\lambda_{center}$: ' + str(round(self.line_centers[j], 3)) + '\n'
                    text_plot += '$C_{In-Out}$: '+str(np.round(c0, 6)) + r'$\pm$' +str(
                                np.round(abs(s0/c0), 3)) + '\n' + '$\sigma_{In-Out}$ : '+str(n_sigmas0)
                    text_plot += aux_ttext
                    ax_info.text(text_coords[0], text_coords[1], text_plot, verticalalignment='center', horizontalalignment='center',
                             transform=ax_info.transAxes, style='italic',
                             color=colors[3], fontsize=fontsizes[1], bbox={'facecolor': 'white', 'boxstyle': 'round',
                                                                 'alpha': 0.5, 'pad': text_pad, 'edgecolor': 'black'})
                    ax_info.axis('off')

                ax_norm.text(text_coords[0], text_coords[1],
                             r'$\lambda_{obs}$: ' + str(round(self.line_centers[j], 3)), fontsize=fontsizes[1],
                             transform=ax_norm.transAxes, horizontalalignment='center')

                ax_redf.text(text_coords[2], text_coords[3], r'$\sigma_{In-Out}$: '+str(-n_sigmas0), fontsize=fontsizes[1],
                             transform=ax_redf.transAxes, horizontalalignment='center')
                ax_sigma.axhline(y=0.0, linewidth=lw*0.9, color=colors[4], linestyle='--')
                fmt = '.'
                elinewidth = lw*0.9
                capsize = lw*1.2
                ax_sigma.errorbar([0], [c0], yerr=[s0], fmt=fmt, elinewidth=elinewidth,
                                  color=colors[5], capsize=capsize)
                if self.outout_ok or not omit_hists:
                    ax_sigma.errorbar([0.1], [c1], yerr=[s1], fmt=fmt, elinewidth=elinewidth,
                                  color=colors[6], capsize=capsize)
                if self.inin_ok or not omit_hists:
                    ax_sigma.errorbar([0.2], [c2], yerr=[s2], fmt=fmt, elinewidth=elinewidth,
                                  color=colors[7], capsize=capsize)
                ax_sigma.set_xlim([-0.1, 0.3])
                ax_snellen.set_xticks([-0.5, 0.5])
                ax_corr.yaxis.set_major_locator(plt.MaxNLocator(2))
                ax_norm.yaxis.set_major_locator(plt.MaxNLocator(2))
                ax_wave_transit.xaxis.set_major_locator(plt.MaxNLocator(2))
                #ax_snellen.xaxis.set_major_locator(plt.MaxNLocator(2))

                axes = [ax_wave_transit, ax_snellen, ax_redf, ax_sigma, ax_norm, ax_corr]

                for axis in axes:
                    axis.xaxis.label.set_size(fontsizes[2])
                    axis.tick_params(labelsize=fontsizes[1])

                lambda_name = str(dlambda)
                if fname_verbose:
                    file_name = str(int(self.linedb[j])) + '_ind_' + str(j) + '_ord_' + str(
                        self._line_orders[j]) + '_band_' + lambda_name
                else:
                    file_name = str(int(self.linedb[j])) + '_' + self._band_to_str(dlambda)
                file_name = self.output_folder + str_adress + '/summary_plots/' + file_name + self.img_format
                plt.subplots_adjust(bottom=adjust[0], top=adjust[1], left=adjust[2], right=adjust[3])
                self.captains_log.at[line, 'image'] = file_name
                self.summary_figs.add_plot(fig, file_name)
                if ind_plt == j or show_all:
                    plt.show()
                plt.close()

            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Summary plots: ' + message

                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Summary plots: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
            if self.use_bar:
                pbar.update(dbar)
                sys.stdout.flush()

        if self.use_bar:
            pbar.close()

        if self.save_figs:
            self.summary_figs.save_plots()


    def _calculate_signal(self, array, is_out, signal_func = np.mean):
        is_out = np.array(is_out)
        is_in = np.array([not b for b in is_out])
        in_flux = array[is_in]
        out_flux = array[is_out]
        return signal_func(in_flux)/signal_func(out_flux) - 1

    def betterfield(self, signal_func = np.mean, bins=50, bloops = 3000, min_remove = 2, lw=1, normed=False):
        #is_in = [not b for b in self.is_out]
        for j in range(len(self.linedb)):
            line_center = self.line_centers[j]
            dlambda = self._params['dlambda_analyze']
            start1 = line_center - 1.5 * dlambda
            end1 = line_center - 0.5 * dlambda
            start2 = end1
            end2 = line_center + 0.5 * dlambda
            end3 = line_center + 1.5 * dlambda
            mask_wv = np.array([start1 < wv and wv < end3 for wv in self.wave_matrix[j]])
            useful_wv = self.wave_matrix[j][mask_wv]
            is_in_wv = np.array([start2 < wv and wv < end2 for wv in useful_wv])
            flux_j = self.superline[j, :, mask_wv]
            master_is_in = np.zeros((self._params['n_epochs'], len(is_in_wv)), dtype=bool)
            master_is_in[:, is_in_wv] = True
            master_is_in[self.is_out, :] = False
            flatten_flux = np.array(flux_j.flatten())
            flatten_is_in = np.array(master_is_in.flatten())
            flatten_is_out = np.array([not b for b in flatten_is_in])

            ratio = flatten_is_out.sum() / len(flatten_is_out)
            n_out = int(flatten_is_out.sum() * ratio + 0.5)
            n_in = int(flatten_is_in.sum() * ratio + 0.5)
            all_isoutout = np.random.permutation(create_all_isout(int(flatten_is_out.sum()), n_out, max_len=bloops))
            all_isinin = np.random.permutation(create_all_isout(int(flatten_is_in.sum()), n_in, max_len=bloops))



            critic_ints = range(min_remove, int(flatten_is_in.sum() / 2 + 0.5) + 1)
            combs = []
            aux = []
            for step in critic_ints:
                critic_len = int((bloops - len(aux))/(critic_ints[-1] + 1 - step) +0.5)
                aux = create_all_isout(len(flatten_is_out), step, max_len=critic_len)
                combs.extend(aux)


            out_out_results = []
            in_in_results = []
            in_out_results = []
            out_flux = flatten_flux[flatten_is_out]
            in_flux = flatten_flux[flatten_is_in]

            for iter in range(bloops):
                out_out_results.append(self._calculate_signal(out_flux, all_isoutout[iter], signal_func=signal_func))
                in_in_results.append(self._calculate_signal(in_flux, all_isinin[iter], signal_func=signal_func))
                in_out_results.append(self._calculate_signal(flatten_flux, combs[iter], signal_func=signal_func))

            plt.close('all')
            plt.hist(in_out_results, bins= bins, color = 'darkgreen',facecolor='None',
                        orientation='horizontal', histtype='step', linewidth=lw*1.2, normed=normed)
            plt.hist(out_out_results, bins= bins, color = 'darkblue', facecolor='None',
                    orientation='horizontal', histtype='step', linewidth=lw*0.9, normed=normed)
            plt.hist(in_in_results, bins= bins, color = 'darkred',
                     facecolor='None',
                     orientation='horizontal', histtype='step', linewidth=lw * 0.9, normed=normed)
            plt.savefig(self.output_folder+self._str_adress+'/redfield/betterfield_'+str(self.linedb[j])+'_'+
                        str(self._params['dlambda_analyze'])+ self.img_format)



    def consolidate_log(self, set_directory = 'captains_log', tspectra = False, error_log = False, use_date = False, use_lines= True, header = True,
                        overwrite = True, params = False):

        if error_log:
            directory = 'error_log'
        elif tspectra:
            directory = 'tspectra'
        elif params:
            directory ='params_log'
        else:
            directory = set_directory

        print ()
        print ('Writing the log in ' + directory)
        print ()
        sys.stdout.flush()

        try:
            directory = self.output_folder+ self._str_adress+'/' + directory+'/'
            dlambda = self._params['dlambda_analyze']

            if overwrite:
                mode = 'w'
            else:
                mode = 'a'

            if use_date:
                date = strftime("%Y-%m-%d_%H:%M:%S", gmtime())
            else:
                date = ''

            lambda_name = 'band_'+str(dlambda)

            if tspectra:
                df = self._tspectra
            elif error_log:
                df = self.error_log
            elif params:
                orders_len = np.mean(self.orders_length)
                self._params['orders_length'] = orders_len
                self._params['in_epochs'] = np.sum([not aux for aux in self.is_out])
                self._params['outout_ok'] = self.outout_ok
                self._params['inin_ok'] = self.inin_ok
                self._params['general_warnings'] = self.general_warnings
                df = pd.DataFrame.from_dict(self._params, orient = 'index')
            else:
                df = self.captains_log

            df_index = df.index.values
            if params:
                df_index = self.captains_log.index.values
            if len(df_index) != 0:
                if use_lines:
                    low_limit = str(df_index[0])
                    up_limit = str(df_index[len(df_index)-1])
                    name = directory+low_limit+'_to_'+up_limit+'_'+date+'_'+lambda_name
                else:
                    name = directory+date+'_'+lambda_name


                df.to_csv(name+'.txt', sep = ' ', header = True, mode = mode)
            else:
                print ('Your log is empty.')

        except Exception as ex:
            template = "{0}. Arguments: {1!r}"
            message = template.format(type(ex).__name__, ex.args)
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
            print ("I couldn't write the log as txt.")
            print (message)
            print(to_log)
            sys.stdout.flush()



    def plot_sigmas(self, fill_file = True, use_lines = False, max_per_fig = 50, dl = 0.04,
                    capsize = 2, fmt = '.', elinewidth = 1.2, figsize=(14,8), overwrite = True,
                    pbar = 'None'):
        print ()
        print ("Doing Sigmas plots (8/8) to " + str(len(self.linedb)) + " lines.")
        print()

        sys.stdout.flush()

        dlambda = self._params['dlambda_analyze']
        str_adress = self._str_adress

        l = len(self.linedb)
        lambda_name = str(dlambda)

        if fill_file and self.save_figs:
            self.consolidate_log(tspectra=True, use_lines = use_lines, overwrite = overwrite)

        h = 0
        k = 0
        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1
        while k < l:
            plt.close('all')
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, facecolor='floralwhite')

            if k + max_per_fig >= l:
                sup = l
            else:
                sup = k + max_per_fig

            h = h + 1
            k0 = k
            for j in range(k, sup):
                k = k + 1
                atom_data = self.captains_log[self.captains_log.norm_tests]
                line = atom_data.index[j]
                c0 = atom_data.at[line, 'centerIO']
                s0 = atom_data.at[line, 'sigmaIO']
                c1 = atom_data.at[line, 'centerOO']
                s1 = atom_data.at[line, 'sigmaOO']
                c2 = atom_data.at[line, 'centerII']
                s2 = atom_data.at[line, 'sigmaII']
                dlambda = atom_data.at[line, 'band']
                try:
                    ax.errorbar([j - dl], [c0], yerr=[s0], fmt=fmt, elinewidth=elinewidth, color='darkgreen',
                                capsize=capsize)
                    ax.errorbar([j], [c1], yerr=[s1], fmt=fmt, elinewidth=elinewidth, color='darkblue', capsize=capsize)
                    ax.errorbar([j + dl], [c2], yerr=[s2], fmt=fmt, elinewidth=elinewidth, color='darkred',
                                capsize=capsize)

                except Exception as ex:
                    template = "{0}. Arguments: {1!r}"
                    message = template.format(type(ex).__name__, ex.args)
                    self.error_log.at[self.log_index[j], 'message'] = 'Sigmas: ' + message
                    exc_type, exc_obj, exc_tb = sys.exc_info()
                    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                    to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                    self._set_status('Sigmas: ' + to_log, self.log_index[j])
                    print (to_log, message)
                    sys.stdout.flush()
                    self._set_warning('Error found', self.log_index[j])
                    self.line_valid[j] = False
                if self.use_bar:
                    pbar.update(dbar)
                    sys.stdout.flush()




            ax.axhline(y=0.0, linewidth=0.9, color='forestgreen', linestyle='--')
            ax.set_xticks(range(k0, sup))
            ax.set_xticklabels(self.linedb[k0:sup], rotation='vertical')
            ax.set_title(r'Widths for band '+ str(dlambda))
            df_index = self.captains_log[self.captains_log.norm_tests].index.values
            directory = self.output_folder + str_adress + '/sigma_data/'
            if len(df_index) != 0:
                if use_lines:

                    low_limit = str(df_index[0])
                    up_limit = str(df_index[len(df_index) - 1])
                    name = directory + low_limit + '_to_' + up_limit + '_' + lambda_name
                else:
                    name = directory + '_' + lambda_name

            else:
                print ('Your log is empty.')
            file_name = name+'_image_'+str(h) + self.img_format
            #plt.savefig(file_name, dpi=dpi)
            self.sigmas_figs.add_plot(fig, file_name)

            plt.close('all')
        if self.use_bar:
            pbar.close()
        if self.save_figs:
            self.sigmas_figs.save_plots()



    def _fill_sigma_data(self, directory = 'sigma_data'):
        str_adress = self.output_folder + self._str_adress + '/' + directory
        database = list_archives(str_adress, format = '.txt')
        self._all_sigma_data = []

        for file in database:
            df = np.genfromtxt(file, delimiter=' ', skip_header= 1, dtype = ("|S3", float, float, float, float, float,
                                                                             float, float, float, float))
            for j in range(len(df)):
                self._all_sigma_data.append(df[j])
        self._all_sigma_data = self._all_sigma_data

    def save_all_plots(self, draw = True, clean_figs = False):
        self.checking_figs.save_plots(draw = draw, clean_figs = clean_figs)
        self.extraction_figs.save_plots(draw = draw, clean_figs = clean_figs)
        self.telluric_figs.save_plots(draw=draw, clean_figs=clean_figs)
        self.renorm_figs.save_plots(draw=draw, clean_figs=clean_figs)
        self.summary_figs.save_plots(draw = draw, clean_figs = clean_figs)
        self.sigmas_figs.save_plots(draw = draw, clean_figs = clean_figs)
        self.lcheck_figs.save_plots(draw = draw, clean_figs = clean_figs)

    def use_previous_work(self, another_TBS):
        if another_TBS._str_adress == self._str_adress:
            mask = another_TBS.line_valid
            self.linedb = np.array(self.linedb)
            if np.all(self.linedb[mask] == another_TBS.line_valid):
                self.superline=another_TBS.superline
                self.captains_log = another_TBS.captains_log

                self.linedb = self.linedb[mask]
                self._line_orders = another_TBS._line_orders[mask]
                self.line_centers = another_TBS.line_centers[mask]
                self.str_elem = another_TBS.str_elem[mask]
                self.log_gf = another_TBS.log_gf[mask]
                self.superline = another_TBS.superline[mask, :, :]
                self.error_matrix = another_TBS.error_matrix[mask, :, :]
                self.line_valid = another_TBS.line_valid[mask]

    def detection_alert(self, string = 'Possible detection', pbar = 'None', lcheck_plot = True,
                        figsize = (7,5), times_near=2, n_sigmas=3, n_sigmas_noise=1):

        print ()
        print ("Doing detection alert (7/8) to " + str(len(self.linedb)) + " lines.")
        print ()

        sys.stdout.flush()
        if self.use_bar and pbar == 'None':
            pbar = tqdm(total = len(self.linedb))
        dbar = 1
        omit_hists = self._params['omit_hists']
        for j in range(len(self.linedb)):
            try:
                line = self.log_index[j]
                log_index = self.log_index[j]
                centerIO = self.captains_log.at[line, 'centerIO']
                centerOO = self.captains_log.at[line, 'centerOO']
                centerII = self.captains_log.at[line, 'centerII']
                sigmaIO = self.captains_log.at[line, 'sigmaIO']
                sigmaOO = self.captains_log.at[line, 'sigmaOO']
                sigmaII = self.captains_log.at[line, 'sigmaII']

                if hasattr(centerIO, '__iter__'):
                    centerIO = centerIO[0]
                    centerOO = centerOO[0]
                    centerII = centerII[0]
                    sigmaIO = sigmaIO[0]
                    sigmaOO = sigmaOO[0]
                    sigmaII = sigmaII[0]
                    print ('Line '+str(line)+ 'is duplicated?')
                    self._set_warning('Duplicated?', line)
                test1 = centerIO +n_sigmas*sigmaIO < 0
                test2 = centerIO < centerII - n_sigmas_noise*sigmaII
                test3 = centerIO < centerOO - n_sigmas_noise*sigmaOO
                total_test = test1 and test2 and test3


                if total_test:
                    self._set_warning(string, line)
                    self.captains_log.at[log_index, 'detection'] = True
                    lcolor = 'darkgreen'
                else:
                    lcolor = 'darkred'
                if lcheck_plot:
                    fig = plt.figure(figsize=figsize)
                    grid = gridspec.GridSpec(2, 2, height_ratios=[2, 5], width_ratios=[1, 0.1], hspace=0.02,
                                       wspace=0.03)
                    ax_corr = fig.add_subplot(grid[0,0], facecolor='floralwhite')
                    ax = fig.add_subplot(grid[1,0], sharex = ax_corr, facecolor='floralwhite')
                    ax_sigma = fig.add_subplot(grid[1,1], facecolor= 'floralwhite')

                    plt.setp(ax_corr.get_xticklabels(), visible=False)
                    plt.setp(ax_sigma.get_xticklabels(), visible=False)

                    wave = self.wave_matrix[j]
                    flux = self.ref_flux[j]
                    line_center = self.line_centers[j]
                    dlambda = self._params['dlambda_analyze']
                    start1 = line_center - 1.5 * dlambda
                    end1 = line_center - 0.5 * dlambda
                    start2 = end1
                    end2 = line_center + 0.5 * dlambda
                    start3 = end2
                    end3 = line_center + 1.5 * dlambda
                    ind_start = np.argmin(np.abs(wave + times_near*dlambda - start1))
                    ind_end = np.argmin(np.abs(wave - times_near*dlambda  - end3))

                    mask_near = np.array([ind_start <= i and i <= ind_end for i in range(len(wave))])


                    ax.axvspan(start1, end1, alpha=0.3, color='lightcoral')
                    ax.axvspan(start2, end2, alpha=0.7, color='coral')
                    ax.axvspan(start3, end3, alpha=0.3, color='lightcoral')
                    ax.axvline(x=line_center, color='orangered', linestyle='--', linewidth=0.7)
                    ax.plot(wave[mask_near], flux[mask_near], color=lcolor)
                    text_plot = round(self.line_centers[j], 3)

                    ax.text(0.8, 0.1, text_plot, verticalalignment='center',
                             horizontalalignment='left',
                             transform=ax.transAxes, style='italic',
                             color='navy', fontsize=10,
                             bbox={'facecolor': 'navajowhite', 'boxstyle': 'round',
                                   'alpha': 0.5, 'pad': 0.5, 'edgecolor': 'black'})
                    ax.set_ylabel(r'Normalized flux')
                    ax.set_xlabel(r'Wavelength $[\AA]$')


                    ax_corr.plot(wave[mask_near], self.tell_corrections[j][mask_near], linewidth=0.5, color='purple')
                    ax_corr.axvspan(start1, end1, alpha=0.3, color='lightcoral')
                    ax_corr.axvspan(start2, end2, alpha=0.7, color='coral')
                    ax_corr.axvspan(start3, end3, alpha=0.3, color='lightcoral')
                    ax_corr.axvline(x=line_center, color='orangered', linestyle='--', linewidth=0.7)

                    ax_sigma.axhline(y=0.0, linewidth=0.9, color='forestgreen', linestyle='--')
                    fmt = '.'
                    elinewidth = 0.9
                    capsize = 1.2
                    ax_sigma.errorbar([0], [centerIO], yerr=[sigmaIO], fmt=fmt, elinewidth=elinewidth,
                                      color='darkgreen', capsize=capsize)
                    if self.outout_ok or not omit_hists:
                        ax_sigma.errorbar([0.1], [centerOO], yerr=[sigmaOO], fmt=fmt, elinewidth=elinewidth,
                                          color='darkblue', capsize=capsize)
                    if self.inin_ok or not omit_hists:
                        ax_sigma.errorbar([0.2], [centerII], yerr=[sigmaII], fmt=fmt, elinewidth=elinewidth,
                                          color='darkred', capsize=capsize)
                    ax_sigma.set_xlim([-0.1, 0.3])

                    lambda_name = str(dlambda)
                    file_name = str(int(self.linedb[j])) + '_ind_' + str(j) + '_ord_' + str(
                        self._line_orders[j]) + '_band_' + lambda_name
                    file_name = self.output_folder + self._str_adress + '/other_epoch/' + file_name + self.img_format
                    self.captains_log.at[line, 'epoch_image'] = file_name
                    self.lcheck_figs.add_plot(fig, file_name)

            except Exception as ex:
                template = "{0}. Arguments: {1!r}"
                message = template.format(type(ex).__name__, ex.args)
                self.error_log.at[self.log_index[j], 'message'] = 'Detection: ' + message
                exc_type, exc_obj, exc_tb = sys.exc_info()
                fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
                to_log = str(type(ex).__name__) + ', ' + str(fname) + ', ' + str(exc_tb.tb_lineno)
                self._set_status('Detection: ' + to_log, self.log_index[j])
                print (to_log, message)
                sys.stdout.flush()
                self._set_warning('Error found', self.log_index[j])
                self.line_valid[j] = False
            if self.use_bar:
                pbar.update(dbar)
        if self.use_bar:
            pbar.close()
        if self.save_figs:
            self.lcheck_figs.save_plots()

    def _get_snr(self, flux, param = 0.02):
        is_in = np.array([not aux for aux in self.is_out])
        in_epochs = is_in.sum()
        return 1/np.std(flux)*np.sqrt(in_epochs*param)







if __name__ == "__main__":

    #Functions tests

    num = 2
    if num == 1:
        str_elems = ['Ca1', 'Sc2', 'Ca1', 'Ca1', 'Ca1', 'Mn1', 'He1', 'Si2', 'Mn1']
        linedb = [5588.75, 5526.8, 6162.17, 6439.5, 6493.8, 6021.8, 6678.2, 6347.1, 6016.7] #indexFWHM = 12, norm_width = 4
        loggfs = range(len(linedb))
    elif num == 2:
        str_elems = ['Ca']
        linedb = [6162.17]
        loggfs = [1]




    main_folder = 'exoplanet_data/'
    str_adress = input('Which folder do you want to analyze?')
    save_figs = True

    # Params of nicola_data

    if str_adress == 'nicola_data':
        ask_more = True
        go_on = True
        TBS_dict = {
            'orders': 22,
            'indexFWHM': 10,
            'error_gain': 1.6,
            'dtransit': 0.127951389,
            't_transit': 2452826.628521,
            'period': 3.52474859,
            'do_1stdopp': True,
            'dlt_epochs_again': [0, 1],
            'tell_only_out': True,
            'norm_width': 5,
            'norm_high': 3,
            'wave_binning': 0,
            'test_percent': 0.7,
            'test_boundary': 0.011,
            'telescope': 'subaru'
        }


    # Params of data
    elif str_adress == 'data':
        orders = 16
        indexFWHM = 11
        error_gain = 1
        dt = 0.1294
        t_transit = 2454884.02817
        period = 21.2163979
        go_on = True
        tell_only_out = False
        # low_limit=6330, up_limit=6799

    elif str_adress == 'wasp-167b':
        orders = 0
        indexFWHM = 50
        error_gain = 1.36
        dt = 0.1135
        t_transit = 2456592.4643
        period = 2.0219596
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'test/'
        ignore_tests = True
        band = 0.75
        radial_velocity = -3.409
        dlt_epochs_again = [17,18]
        save_figs = True
        full_test = True
        ask_more = False
        dlt_epochs = 0

    elif str_adress == 'wasp-117b':
        orders = 0
        indexFWHM = 15
        error_gain = 1.36
        dt = 0.2475
        t_transit = 2456533.82326
        period = 10.02165
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'test/'
        band = 0.5
        dlt_epochs = 0
        radial_velocity = -18.071
        do_1stdopp = True
        ignore_tests = False

        save_figs = True
        full_test = False
        ask_more = False
        only_detected = False

    elif str_adress == 'corot-3b':
        orders = 0
        indexFWHM = 50
        error_gain = 1.36
        dt = 0.1566
        t_transit = 2454283.13388
        period = 4.2567994
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'test/'
        band = 0.75
        dlt_epochs = 0
        near_indline = 15
        test_percent = 0.4
        radial_velocity = -2161.14/1000
        do_1stdopp = True

        save_figs = True
        full_test = True
        ask_more = False
        only_detected = False

    elif str_adress == 'gj-436b':
        orders = 0
        indexFWHM = 50
        error_gain = 1.36
        dt = 0.7608/24
        t_transit = 2454279.436714
        period = 2.643850
        norm_width = 10
        norm_high = 0.5
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'test/'
        band = 0.75
        dlt_epochs = 0
        near_indline = 15
        test_percent = 0.4
        radial_velocity = 9.6
        do_1stdopp = True
        ignore_tests = True

        save_figs = True
        full_test = True
        ask_more = False
        only_detected = False


    elif str_adress == 'hd-3_1':
        go_on = True
        ask_more = True
        TBS_dict = {
            'orders': 0,
            'indexFWHM': 12,
            'error_gain': 1.36,
            'dtransit': 0.07527,
            't_transit': 2454279.436714,
            'period': 2.21857567,
            'do_1stdopp': True,
            'radial_velocity': -2.55,
            'dlt_epochs_again': [],
            'tell_only_out': False,
            'wave_binning': 2,
            'telescope': 'harps',
            'norm_width': 5,
            'test_boundary': 0.013,
            'norm_high': 3

        }



    elif str_adress == 'hd-3_2':
        orders = 0
        indexFWHM = 20
        error_gain = 1.36
        dt = 0.07527
        t_transit = 2454279.436714
        period = 2.21857567
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'test/'
        band = 0.5
        ignore_tests = False
        dlt_epochs = 0
        near_indline = 40
        test_percent = 0.3
        test_pixs_near = 20
        test_times_near = 70
        radial_velocity = -2.2765
        test_boundary = 0.01
        do_1stdopp = True
        tell_only_out = True
        dlt_epochs_again=[0, 1, 2, 35, 36, 45, 46]
        #loops = 200

        save_figs = True
        full_test = False
        ask_more = False
        only_detected = False

    elif str_adress == 'wasp-23b':
        orders = 0
        indexFWHM = 12
        error_gain = 1.36
        dt = 0.09976
        t_transit = 2455320.12363
        period = 2.9444256
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'ca_test/'
        band = 0.5
        dlt_epochs = 0
        near_indline = 15
        ignore_tests = False

        test_pixs_near = 15
        test_times_near = 2
        radial_velocity = 5.69160
        do_1stdopp = True
        tell_only_out = True
        loops = 3000
        max_t_len = 1
        dlt_epochs_again = [0, 27]

        save_figs = True
        full_test = True
        ask_more = False
        only_detected = False

    elif str_adress == 'hd-1490/green' or str_adress == 'hd-1490/blue' \
            or str_adress == 'hd-1490_2/green' or str_adress == 'hd-1490_2/blue':
        orders = 17
        dlt_epochs_again = []
        if 'blue' in str_adress:
            orders = 21
            dlt_epochs_again = [6, 15]
        indexFWHM = 15
        error_gain = 1.36
        dt = 195 / (24 * 60)
        t_transit = 2453527.872
        period = 2.87618
        binning = 1
        telescope = 'keck'
        go_on = True
        out_folder = 'test/'
        band = 0.5
        dlt_epochs = 0
        radial_velocity = 12.3
        do_1stdopp = True
        max_t_len = 1
        doppler_epochs = 0
        n_merge = 1

        save_figs = True
        full_test = True
        ask_more = True
        only_detected = True


    elif str_adress == 'xo-4b':
        orders = 0
        indexFWHM = 15
        error_gain = 1.36
        dt = 0.09976
        t_transit = 2455320.12363
        period = 2.9444256
        norm_width = 10
        norm_high = 0.5
        binning = 2
        use_harps = False
        go_on = True
        out_folder = 'fake/'
        band = 0.5
        dlt_epochs = 0
        near_indline = 15
        test_percent = 0.3

        test_pixs_near = 15
        test_times_near = 10
        radial_velocity = 5.69160
        test_boundary = 0.01
        do_1stdopp = True
        tell_only_out = True
        loops = 1700
        dlt_epochs_again = []

        save_figs = True
        full_test = False
        ask_more = False
        only_detected = False



    elif str_adress == 'xo-3b/green':

        TBS_dict = {
            'orders': 17,
            'indexFWHM': 29,
            'error_gain': 2.09,
            'dtransit': 0.125,
            't_transit': 2454494.549,
            'period': 3.19161,
            'wave_binning': 0,
            'telescope': 'keck',
            'dlt_epochs_again': [],
            'dlt_epochs': 0,
            'test_boundary': 0.026,
            'test_percent': 0.7,
            'use_bar': True,
            'radial_velocity': 58.99
        }
        full_test = False
        out_folder = 'xo_test/'
        ask_more = False
        go_on = True
        band = 0.75


    elif 'hatp-2b/' in str_adress:
        if 'green' in str_adress:
            orders = 17
            dlt_epochs_again = [0, 1, 2, 94, 95, 96]
        elif 'red' in str_adress:
            orders = 10
            dlt_epochs_again = [0, 1, 2,  91, 92, 93, 94, 95, 96]
        elif 'blue' in str_adress:
            orders = 22
            dlt_epochs_again = [0, 1, 2,  91, 92, 93, 94, 95, 96]

        indexFWHM = 50
        error_gain = 2.09
        dt = 0.1787
        t_transit = 2454212.8561
        period = 5.6334729
        norm_width = 10
        norm_high = 0.5
        binning = 2
        telescope = 'keck'
        go_on = True
        out_folder = 'fake/'
        band = 0.75
        dlt_epochs = 0
        near_indline = 25
        test_percent = 0.7

        test_pixs_near = 20
        test_times_near = 10
        radial_velocity = 45
        test_boundary = 0.1
        do_1stdopp = True
        tell_only_out = True
        loops = 3000

        n_merge = 3

        save_figs = True
        full_test = False
        ask_more = False

    elif str_adress=='wasp-28b':
        orders = 0
        indexFWHM = 50
        error_gain = 1.36
        dt = 0.1349
        t_transit = 2455290.40519
        period = 3.4088300
        norm_width = 10
        norm_high = 0.5
        binning = 2
        telescope = 'harps'
        go_on = True
        out_folder = 'na_test/'
        band = 0.75
        dlt_epochs = 0
        near_indline = 25
        test_percent = 0.3
        radial_velocity = 1.2

        ignore_tests = True
        full_test = True
        ask_more = False
        only_detected = False

    elif str_adress == 'wasp-74b':
        ask_more = False
        go_on = True
        band = 0.75
        out_folder='uves_test/'
        full_test = False
        TBS_dict = {
            'orders': 0,
            'indexFWHM': 22,
            'error_gain': 0,
            'dtransit': 0.0955,
            't_transit': 2456506.8918,
            'period': 2.13775,
            'n_epoch_merge': 0,
            'wave_binning': 2,
            'norm_width': 5,
            'norm_high': 3.5,
            'test_boundary': 0.029,
            'telescope': 'uves',
            'dlt_epochs': 0,
            'test_percent': 0.7,
            'radial_velocity': -15.767,
        }


    elif str_adress == 'corot-2b':
        orders = 0
        indexFWHM = 49
        error_gain = 0
        dt = 2.28 * 0.0416667
        t_transit = 2454706.4041
        period = 1.7429964
        norm_width = 10
        norm_high = 3
        binning = 0
        telescope = 'uves'
        go_on = True
        out_folder = 'uves_test/'
        band = 0.75
        dlt_epochs = 0
        near_indline = 25
        test_percent = 0.7
        do_1stdopp = False
        radial_velocity = 23


        ignore_tests = True
        full_test = False
        ask_more = False
        only_detected = False


    elif str_adress == 'tres-2b/green':
        low_limit = 4911
        up_limit = 6422
        TBS_dict = {
            'orders': 17,
            'indexFWHM': 14,
            'error_gain': 2.09,
            'dtransit': 106.68*0.000694444,
            't_transit': 2456230.5980,
            'period': 2.4706132,
            'wave_binning': 0,
            'test_boundary': 0.05,
            'n_epoch_merge': 0,
            'test_percent': 0.7,
            'telescope': 'keck',
            'dlt_epochs_again': [],
            'merge_epoch_as': [],
            'dlt_epochs': 0,
            'radial_velocity': 7.22
        }
        ask_more = True
        go_on = True



    elif 'hatp-1b' in str_adress:
        orders = 17
        indexFWHM = 50
        error_gain = 2.09
        dt = 2.798*0.000694444
        t_transit = 2454363.94656
        period = 4.4652934
        telescope = 'keck'
        go_on = True
        out_folder = 'fake/'
        band = 0.75
        dlt_epochs = 0

        """
        lines_, strs_, loggf_, dep_ = get_lines('spectroweb_lines', min_loggf = -1, max_loggf = 1,
                                                        low_limit=4908, up_limit=6424, min_depth = 20)
        print (len(lines_))
        doppler_db = sorted(lines_)
        doppler_epochs = 1
        """

        radial_velocity = 19.6
        do_1stdopp = True
        tell_only_out = False
        loops = 3000
        test_percent = 0.4
        dlt_epochs_again = []

    elif 'hd-1897_uves' in str_adress:
        TBS_dict = {
            'orders': 0,
            'indexFWHM': 13,
            'error_gain': 1.36,
            'dtransit': 0.07527,
            't_transit': 2454279.436714,
            'period': 2.21857567,
            'do_1stdopp': True,
            'radial_velocity': -2.55,
            'dlt_epochs_again': [],
            'tell_only_out': False,
            'wave_binning': 2,
            'telescope': 'uves',
            'norm_width': 5,
            'test_boundary': 0.013,
            'norm_high': 3,
            'plot_time_binning': 3,
            'n_epoch_merge': 2,
        }
        ask_more = True
        go_on = True

    else:
        print ('Not existing path.')
        go_on = False

    if go_on:
        loops = 3000
        if ask_more:
            #main_folder = input('Main folder?')
            main_folder = 'exoplanet_data/'
            #band = float(input('In what band?'))
            #out_folder = input('Where do you want to save your stuff?')
            band = 0.75
            out_folder = 'tesis_review/'
            save_figs = True
            #full_test = eval(input('Do you want to make a full run?(True/False)'))
            full_test = True
        if full_test:
            linedb, str_elems, loggfs = get_all_linedb(folders = ['spectroweb_lines', 'vald_lines'], elem_key='',
                                                       wv_lims=[], elem_delete='', loggf_lims=[0,100])
            print ('You will analyze ' + str(len(linedb)) + ' lines')


        Totry = TBS(str_adress, linedb, str_elems, loggfs, create_folders=True,  dlambda_analyze = band,
                save_figs=save_figs, output_folder=out_folder, ignore_tests=False, **TBS_dict)

        Totry.check_epoch(plot_all = True, show_plt = False, print_info=False)
        Totry.extract_lines(doall_plt=True, all_line_plt= 4)
        Totry.telluric_corrections(all_plt = True, ind_plt=0)
        Totry.snellen(do_plt = False)
        Totry.redfield(do_plt = False)
        Totry.summary_plots(show_all = False, do_width_plt=False,
                            binning = True, fname_verbose=False)
        Totry.detection_alert()
        if save_figs:
            Totry.consolidate_log()
            Totry.consolidate_log(error_log=True)
            Totry.consolidate_log(params = True)
            Totry.consolidate_log(tspectra=True)

        if save_figs:
            Totry.save_all_plots()
