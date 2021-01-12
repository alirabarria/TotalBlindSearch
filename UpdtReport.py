import pylatex as pl
from pylatex.utils import bold, italic, NoEscape
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import os
from matplotlib import gridspec
from tqdm import tqdm
from functions_database import list_archives, openFile
from astropy.io import fits
import matplotlib.colors as plt_colors
import matplotlib.cm as cmx
import uncertainties.unumpy as unumpy
import sys


class UpdtReport:

    def __init__(self, str_adress, bands, image_format='.png', min_loggf = -100, obs_line_tol = 1, author='A. Lira-Barria',
                 filename='report', fig_width = '17cm', directory = 'atom_data', raw_data_directory='exoplanet_data/',
                 report_geometry={"tmargin": "0.5cm", "lmargin": "2.5cm", "bmargin": "0.5cm"},
                 data_header=['TELESCOP', 'INSTRUME', 'OBJECT', 'DATE'], vmin=1, vmax =3, fig_atom_size = (5,3),
                 using_harps=False, loggf_table_filter = -100, fig_atom_width = '15cm', depth_table_filter=0,
                 number_table_filter = 10, resume_fwhm = True, fig_appendix ='8cm', app_filename = 'appendix',
                 read_fits = True
                 ):
        self.str_adress = str_adress
        self.log_cols = ['elem', 'obs_wv', 'fwhm', 'fwhm_pix', 'status', 'detection', 'warning', 'all_ok', 'log_gf', 'band',
            'norm_tests', 'norm_residuals', 'abs_residuals', 'norm_depth', 'abs_depth',
            'snellen_in', 'snellen_out', 'edge_distance', 'std_residuals', 'snr', 'n_pix_fit', 'mse', 'centered_mse',
            'amplitude_mark', 'center_mark', 'residuals_mark', 'blended_mark', 'blended_valid', 'cont_test_mean',
            'cont_test_std', 'shift_test_mean', 'shift_test_std',
            'centerIO', 'sigmaIO', 'centerOO', 'sigmaOO', 'centerII', 'sigmaII', 'image', 'epoch_image']


        """
        This class makes an automated summary report of your blind search.
        :image_format: to change the format of the plots made by this report. default='.png'
        :min_loggf: in case you want to mask the transitions by loggf
        :obs_line_tol: parameter to associate lab lines with its closest observed line. default= 1, which means 1 FWHM
        """
        self._params = {'image_format': image_format, 'obs_line_tol': obs_line_tol, 'author': author, 'filename':filename,
                        'fig_width': fig_width, 'report_geometry': report_geometry, 'directory': directory,
                        'raw_data_directory': raw_data_directory, 'vmin':vmin, 'vmax':vmax, 'fig_atom_size':fig_atom_size,
                        'loggf_table_filter': loggf_table_filter, 'fig_atom_width': fig_atom_width,
                        'depth_table_filter': depth_table_filter, 'number_table_filter': number_table_filter,
                        'fig_appendix': fig_appendix, 'app_filename': app_filename, 'read_fits': read_fits

        }

        self._add_section = {'object_parameters': True, 'data_info': True, 'object_stretches': True,
                             'data_graphics': True, 'global_elem_analysis': True, 'dist_plots': True,
                             'transitions': True, 'elems_wout_det': True, 'neg_transitions': True,
                             'resume_fwhm': resume_fwhm

        }
        params_logs, _ = self.open_log(directory='params_log', unusual_names=['0'])
        self.params_data = params_logs[0] #reading params log to get important parameters
        self.wv_delta = float(self.params_data.at['orders_length', '0'])
        self.fwhm = float(self.params_data.at['indexFWHM', '0'])
        self.wave_binning = eval(self.params_data.at['wave_binning', '0'])

        atom_data, self.log_file_names = self.open_log()
        self.data_info = self._get_data_info(data_header, using_harps=using_harps)

        if len(atom_data) != 0:
            atom_data = pd.concat(atom_data)
            self._empty = False
            self.atom_data = atom_data #pandas log of all the lines analyzed
            if min_loggf > -100:
                self.atom_data = self.atom_data[self.atom_data['log_gf'] > min_loggf]
            self.str_elem = np.unique(atom_data.elem.values)

            #calculating key values to element analysis
            self.normed_data = self.atom_data[self.atom_data.norm_tests]
            self.normed_data = self.normed_data.dropna(subset=['std_residuals'])
            df = self.normed_data

            self.normed_data['rel_sigmasIO'] = -df['centerIO'] / np.abs(df['sigmaIO'])
            norm_absorption = []

            for index, row in self.normed_data.iterrows():
                max_sigma = np.abs(np.max([np.abs(row['sigmaII']), np.abs(row['sigmaOO'])]))
                if max_sigma == np.abs(row['sigmaII']):
                    center_noise = row['centerII']
                else:
                    center_noise = row['centerOO']
                aux = -(row['centerIO'] - center_noise) / max_sigma
                norm_absorption.append(aux)
            norm_absorption = np.array(norm_absorption)
            self.normed_data['normed_absorption'] = norm_absorption

            #generating atom global analysis and tables per atom
            self.atom_tables = []
            self.elem_counts = []
            eps = self._params['obs_line_tol'] * self.fwhm * self.wv_delta
            for band in bands:
                banded_data = self.atom_data[self.atom_data.band == band] #all lines in band
                normed_data = self.normed_data[self.normed_data.band == band] #only normed lines in band
                normed_elems = np.unique(self.normed_data.elem.values) #all elements with normed lines
                atom_table_dict = {}
                det_counts = pd.DataFrame(index=normed_elems, columns=['sdet', 'snotdet', 'wdet', 'wnotdet']) #summary detections per element
                for elem in normed_elems:
                    sdet=0
                    snotdet=0
                    wdet=0
                    wnotdet=0
                    elem_table = normed_data[normed_data.elem == elem]
                    line_class = []
                    for index, row in elem_table.iterrows():
                        mask_wv = np.array([np.abs(aux - row['obs_wv']) <= eps for aux in banded_data['obs_wv'].values])
                        closest_lines = banded_data[mask_wv]
                        if len(np.unique(closest_lines['elem'].values)) == 1:
                            line_class.append(0) #the transition only can belong to that element
                        elif row['log_gf'] == np.max(closest_lines['log_gf'].values):
                            line_class.append(1) #the strongest element in the log_gf ranking is at that element
                        else:
                            line_class.append(2) #there are stronger elements for that transitions

                        if row['detection'] and line_class[-1] < 2:
                            sdet += 1
                        elif row['detection'] and line_class[-1] == 2:
                            wdet += 1
                        elif not row['detection'] and line_class[-1] < 2:
                            snotdet += 1
                        else:
                            wnotdet += 1
                    line_class = np.array(line_class)
                    elem_table['line_class'] = line_class
                    atom_table_dict[elem] = elem_table
                    det_counts.at[elem, 'sdet'] = sdet
                    det_counts.at[elem, 'wdet'] = wdet
                    det_counts.at[elem, 'snotdet'] = snotdet
                    det_counts.at[elem, 'wnotdet'] = wnotdet
                self.atom_tables.append(atom_table_dict)

                det_counts['considered'] = det_counts['sdet'] + det_counts['snotdet']
                det_counts['det'] = det_counts['sdet'] + det_counts['wdet']
                det_counts['total'] = det_counts['sdet'] + det_counts['snotdet'] + det_counts['wdet'] + det_counts[
                    'wnotdet']
                det_counts['stotal'] = det_counts['sdet'] + det_counts['snotdet']
                det_counts = det_counts[det_counts.total > 0]
                for elem, row in det_counts.iterrows():
                    if row['stotal'] > 0:
                        ratio = row['sdet'] / row['stotal']
                    else:
                        ratio = 0
                    det_counts.at[elem, 'ratio'] = ratio
                det_counts['ratio_dettotal'] = det_counts['det'] / det_counts['total']
                det_counts['ratio_constotal'] = det_counts['considered'] / det_counts['total']
                det_counts['ranking'] = 10 ** 4 * det_counts['ratio']
                dettotal_mask = np.array(det_counts.ratio == 0)
                det_counts['ranking'] += det_counts['ratio_dettotal'] * dettotal_mask
                constotal_mask = np.array(det_counts.ratio_dettotal == 0)
                det_counts['ranking'] -= 100 * det_counts['ratio_constotal'] * constotal_mask

                sorted_counts = det_counts.sort_values(by=['ranking'], ascending=False)
                self.elem_counts.append(sorted_counts)

            #generating the detection images to plot
            self.detection_figs = []
            for band in bands:
                band_det_figs = {}
                banded_data = self.normed_data[self.normed_data.band==band]
                detected_data = banded_data[banded_data.detection]
                if len(detected_data) > 0:
                    detected_data = detected_data.sort_values(by='obs_wv', ascending=True)
                    obs_wv = detected_data['obs_wv'].values
                    all_obs_wv = banded_data['obs_wv'].values
                    k = 0
                    ini_wv = obs_wv[0]
                    while (k < len(obs_wv)):
                        if k+1 == len(obs_wv) or not abs(obs_wv[k+1] - obs_wv[k]) < eps:
                            mask_wv = [ini_wv <= wv and wv <= obs_wv[k] for wv in all_obs_wv]
                            ranged_data = banded_data[mask_wv]
                            wv_key = np.mean(ranged_data['obs_wv'].values)
                            sigmas = ranged_data['rel_sigmasIO'].values
                            if len(sigmas) != 0:
                                max_sigma_ind = np.argmax(sigmas)
                                max_sigma_plot = ranged_data['image'].values[max_sigma_ind]
                                band_det_figs[wv_key] = [max_sigma_plot, ranged_data]
                                if k+1 != len(obs_wv):
                                    ini_wv = obs_wv[k+1] #we change the initial wavelength of the range
                        k += 1
                else:
                    print ('No detections at band', band)
                self.detection_figs.append(band_det_figs)

        else:
            self._empty = True

        self.bands = np.array(bands)*1.0

    def open_log(self, directory='captains_log', format='.txt', key='', unusual_names=[]):
        """
        This function takes a pandas log from the blind search and loads it into the report.
        :param directory: directory where the file is
        :param format: format to search
        :param key: text to search in the file name
        :param unusual_names: in case you don't want to use the default log cols on the __init__
        :return: array with all the pandas logs and the file names
        """
        str_adress = self.str_adress + '/' + directory
        db = list_archives(str_adress, format=format, key=key)
        if len(unusual_names) != 0:
            names = unusual_names
        else:
            names = self.log_cols
        all_logs = []
        for file in db:
            df = pd.read_csv(file, sep=" ", header=0, names=names)
            all_logs.append(df)
        return all_logs, db


    def _fancy_name(self, var):
        if var == 'rel_sigmasIO':
            return 'Relative sigmas In-Out'
        elif var == 'log_gf':
            return 'log(gf)'
        elif var == 'rel_sigmas_noise':
            return 'Relative sigmas out-out & in-in'
        elif var == 'normed_absorption':
            return r'$(C_{Noise} - C_{IO}) / \sigma_{Noise}$'
        elif var == 'abs_residuals':
            return r'IQR(Flux - normalization fit)'
        elif var == 'norm_residuals':
            return r'IQR( 1 - normalized flux )'
        elif var == 'std_residuals':
            return r'$\sigma_c$'
        elif var == 'norm_depth':
            return r'Normed line depth'
        else:
            return var

    def _elem_name(self, str_elem):
        if int(str_elem[-1]) == 1:
            add = ' I'
        elif int(str_elem[-1]) == 2:
            add = ' II'
        else:
            add = str_elem[-1]

        return str_elem[:-1] + add

    def _get_data_folder(self):
        data_folder = self._params['raw_data_directory']
        split_adress = self.str_adress.split('/')
        if len(split_adress) > 2:
            aux_split_adress = split_adress[1] + '/' + split_adress[2]
        else:
            aux_split_adress = split_adress[1]
        if using_harps:
            adress = data_folder + 'log_fits/' + aux_split_adress
        else:
            adress = data_folder + aux_split_adress
        return adress



    def _get_data_info(self, for_header, using_harps=False):
        """
        This function loads the param log to list it on the report
        :param self:
        :param for_header:
        :param using_harps:
        :return:
        """
        adress = self._get_data_folder()
        db = list_archives(adress)
        header = fits.getheader(db[0])
        result = []
        for hdr in for_header:
            val = header[hdr]
            result.append([hdr, val])
        return result

    def _band_to_str(self, band):
        str_band = str(band)
        aux_str = str_band.split('.')
        return str(aux_str[0])+str(aux_str[1])

    def _summary_histogram(self, sorted_counts, colors = ['darkgreen', 'darkred', 'darkblue'], band=-1,  dpi=150, path='',
                           legend=True):

        plt.close('all')
        fig = plt.figure(figsize=(9, 5))
        ax = fig.add_subplot(111)


        #ax_up = ax.twiny()
        #plt.rcParams['axes.axisbelow'] = True
        labels = [r'$d_s/T_s$', r'$(d_s+d_w)/(T_s+T_w)$', r'$T_s/(T_s+T_w)$']

        x = range(len(sorted_counts.index.values))
        #ax_up.plot(x, np.zeros(len(x)), 'None', color='white', zorder=0, linewidth=0)
        ax.plot(x, sorted_counts['ratio'].values, marker='^', linestyle='--', color=colors[0], label=labels[0],
                zorder=3)
        ax.plot(x, sorted_counts['ratio_dettotal'].values, 'v', color=colors[1], label=labels[1],
                zorder=3)
        ax.plot(x, sorted_counts['ratio_constotal'].values, 'H', color=colors[2], label=labels[2],
                zorder=3)

        ax.set_xticks(x)
        #ax_up.set_xticks(x)
        x_ticks = []
        up_ticks = []
        for elem, x_i in zip(sorted_counts.index.values, x):
            x_ticks.append(self._elem_name(elem))
            #up_ticks.append(sorted_counts.at[elem, 'total'])
            ax.text(x_i, 1.15, str(sorted_counts.at[elem, 'total']), horizontalalignment='center', fontsize=19,
                    rotation=90)

        ax.set_xticklabels(x_ticks, rotation='vertical')
        ax.tick_params(axis='x', labelsize=21)
        ax.set_ylim([-0.1, 1.1])

        dif = 0.05
        ax.set_xlim([-dif*len(x), (1+dif)*(len(x)-1)])
        ax.set_ylabel('Ratio', fontsize=21)
        ax.set_xlabel('Elements', fontsize=21)
        if legend:
            fig.legend(bbox_to_anchor=(0.1, 0.89, 0.87, 0.93), loc='lower left', borderaxespad=0., fontsize=21, ncol=3,
                       mode='expand')
        plt.margins(0.3)
        if not legend:
            plt.subplots_adjust(bottom=0.25, top=0.89, left=0.14, right=0.97)
        else:
            plt.subplots_adjust(bottom=0.25, top=0.78, left=0.14, right=0.97)
        short_name = 'summary_atoms' + self._band_to_str(band) + self._params['image_format']
        fig_name = path+ '/' +short_name
        fig.savefig(fig_name, dpi=dpi)
        return short_name

    def _distribution_plot(self, band, var_x, bins=25, dpi=150, path=''):
        plt.close('all')
        fig = plt.figure(figsize=self._params['fig_atom_size'])
        ax = fig.add_subplot(111)
        banded_data = self.normed_data[self.normed_data.band == band]
        detected_data = banded_data[banded_data.detection]
        non_detected_data = banded_data[banded_data.detection == False]

        to_plot = [banded_data[var_x].values, non_detected_data[var_x].values, detected_data[var_x].values]
        colors = ['gray', 'darkred', 'darkgreen']
        facecolors = ['None', 'darkred', 'darkgreen']
        for array, color, fcolor in zip(to_plot, colors, facecolors):
            mask_nan = np.array([not str(aux) == 'nan' for aux in array])
            if np.all(mask_nan):
                filtered_array = array
            else:
                print ('Warning:', var_x, ' contains', color, 'nan')
                filtered_array = array[mask_nan]
            n, bins, patches = ax.hist(filtered_array, bins = bins, edgecolor = color, facecolor = fcolor, alpha = 0.7)
            ax.axvline(x=np.median(filtered_array), linestyle='--', color=color, alpha=0.7)
        ax.set_xlabel(self._fancy_name(var_x))
        ax.set_ylabel('Number')
        short_name = 'dist_plot_'+var_x +'_' + self._band_to_str(band) + self._params['image_format']
        fig_name = path + '/' + short_name
        fig.savefig(fig_name, dpi=dpi)
        return short_name

    def _class_to_latex(self, class_number):
        if class_number == 0:
            return pl.UnsafeCommand('ensuremath', arguments =[r'\bigcirc'])
        elif class_number == 1:
            return pl.UnsafeCommand('ensuremath', arguments =[r'\bigtriangleup'])
        else: #class_number == 2
            return pl.UnsafeCommand('ensuremath', arguments =[r'\bigtriangledown'])

    def summarize_atom(self, str_elem, df, directory = 'atom_data', var_x = 'norm_residuals',
                       var_y = 'abs_depth', var_z = 'rel_sigmasIO', fname_prefix = 'summ_',
                       n_z = 10**3, band_flag = -1, alpha_ini = 0.65, add_lines = False,
                       change_marker=True, fontsizes=[30, 13, 12], block_coords=[0.7, 1.05],
                       lim_scale= [0.1, 0.1], add_block=True, to_title=' normalized absorption',
                       format='.png', smart_text=True, sort_by = 'rel_sigmasIO', sort_ascending=True,
                       fn_varz=[], add_suptitle = False, figsize=None):
        str_adress = self.str_adress
        if len(df) != 0:
            if figsize == None:
                figsize = self._params['fig_atom_size']
            fig = plt.figure(figsize= figsize)
            if sort_by != '':
                df = df.sort_values(by=sort_by, ascending = sort_ascending)
            cm = plt.get_cmap('RdYlGn')
            cNorm = plt_colors.Normalize(vmin=self._params['vmin'], vmax=self._params['vmax'])
            scalar_map = cmx.ScalarMappable(norm=cNorm, cmap=cm)
            ax_circles = fig.add_subplot(111)
            if add_lines:
                texts = []
            for index, row in df.iterrows():
                color = scalar_map.to_rgba(row['rel_sigmasIO'])
                if var_y == 'normed_absorption':
                    ax_circles.axhline(1, linestyle='--', color='gray', alpha=0.7)
                    ax_circles.axhline(-1, linestyle='--', color='gray', alpha=0.7)
                marker = 'o'
                alpha = alpha_ini
                if change_marker and row['line_class'] == 1:
                    marker = '^'
                    alpha = alpha_ini
                elif change_marker and row['line_class'] == 2:
                    marker = 'v'
                    alpha = 0.65*alpha_ini
                if var_z != '' and len(fn_varz)==0:
                    s = row[var_z]
                elif var_z != '':
                    s = row[var_z]
                    for fn in fn_varz:
                        s = fn(s)
                else:
                    s = 10
                ax_circles.scatter(row[var_x], row[var_y], s=n_z*s, c=[color],
                                   alpha=alpha, marker=marker)
                if add_lines:
                    texts.append(plt.text(row[var_x], row[var_y], index, fontsize=fontsizes[1]))

            ax_circles.set_xlabel(self._fancy_name(var_x), fontsize=fontsizes[0])
            ax_circles.set_ylabel(self._fancy_name(var_y), fontsize=fontsizes[0])
            ax_circles.tick_params(axis='both', labelsize=fontsizes[0])

            if add_block:
                box_text = 'Circle radius showing: ' + self._fancy_name(var_z) + '\n'
                box_text += 'Cmap showing: ' + self._fancy_name('rel_sigmasIO') + '\n'
                box_text += 'Detections: ' + str(np.sum(df['detection'])) + '/' + str(len(df))
                ax_circles.text(block_coords[0], block_coords[1], box_text,
                            verticalalignment='center', horizontalalignment='left',
                            transform=ax_circles.transAxes, style='italic',
                            color='navy', fontsize=fontsizes[2],
                            bbox={'facecolor': 'navajowhite', 'boxstyle': 'round',
                            'alpha': 0.5, 'pad': 0.9, 'edgecolor': 'black'})

            if band_flag == -1:
                file_name = 'circles_'+str_elem
            elif band_flag != -1:
                lambda_name = self._band_to_str(band_flag)
                file_name = fname_prefix + str_elem + '_band_' + lambda_name
            short_name = file_name
            minx = np.nanmin(df[var_x].values)
            maxx = np.nanmax(df[var_x].values)
            miny = np.nanmin(df[var_y].values)
            maxy = np.nanmax(df[var_y].values)
            if 'nan' in np.str([minx, maxx, miny, maxy]):
                file_name = 'problem'
            else:
                if add_lines and smart_text and str_elem != 'Fe1':
                    from adjustText import adjust_text
                    adjust_text(texts, only_move={'points': 'y', 'text': 'y'})
                if minx != maxx and miny != maxy:
                    difx = maxx - minx
                    dify = maxy - miny
                    ax_circles.set_xlim([minx - lim_scale[0]*np.abs(difx), maxx + lim_scale[0]*np.abs(difx)])
                    ax_circles.set_ylim([miny - lim_scale[1]*np.abs(dify), maxy + lim_scale[1]*np.abs(dify)])
                if add_suptitle: fig.suptitle(self._elem_name(str_elem)+to_title+' for band '+str(band_flag))
                plt.margins(0.2)
                if var_y == 'normed_absorption':
                    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.16, right = 0.97)
                else:
                    plt.subplots_adjust(bottom=0.2, top=0.95, left=0.2, right=0.97)
                file_name = str_adress + '/' + directory + '/' + file_name + format
                fig.savefig(file_name)
        else:
            file_name = 'problem'
        return short_name

    def create_report(self):
        assert not self._empty
        #Creating the document
        short_path = os.path.abspath('')
        long_path = short_path +'/' + self.str_adress + '/atom_data'
        doc = pl.Document(self.str_adress + '/' + self._params['directory'] + '/' + self._params['filename'],
                          geometry_options=self._params['report_geometry'])
        doc.packages.append(pl.Package('hyperref'))

        doc.preamble.append(pl.Command('title', 'Total blind search automated report (version 2.0)'))
        doc.preamble.append(pl.Command('author', self._params['author']))
        doc.preamble.append(pl.Command('date', NoEscape(r'\today')))
        doc.append(NoEscape(r'\maketitle'))
        doc.append(pl.UnsafeCommand('hypersetup', arguments=[
            'colorlinks=true, citecolor=black, filecolor=black, linkcolor=black, urlcolor=black']))
        doc.append(pl.UnsafeCommand('tableofcontents'))
        doc.append(pl.UnsafeCommand('clearpage'))


        #INTRODUCTION
        with doc.create(pl.Section('Introduction.')):
            first_line = 'Folder ' + self.str_adress
            doc.append(first_line)

            with doc.create(pl.Itemize()) as itemize:
                for band in self.bands:
                    band_fwhm = round(float(band) / (float(self.wv_delta) * float(self.fwhm)), 3)
                    itemize.add_item(str(band) + ' Angstrom (' + str(band_fwhm) + ' times the FWHM).')

            if self._add_section['object_parameters']:
                with doc.create(pl.Subsection('General object parameters:')):
                    #parameters used in this run
                    params_data = self.params_data.sort_index()
                    header_row = ['Parameter name', 'Value']
                    needed_param = ['orders', 'telescope', 'cut_order']
                    with doc.create(pl.LongTabu("X[c] X[c]")) as table:
                        table.add_row(header_row, mapper=[bold])
                        table.add_hline()
                        for param, value in zip(params_data.index.values, params_data.values):
                            if param in needed_param:
                                self._params[param] = value[0]
                            if param != 'dlambda_analyze':
                                table.add_row(param, value[0])
                                table.add_hline()

            if self._add_section['data_info']:
                with doc.create(pl.Subsection('Data information:')):
                    #list transit data information
                    first_row = ['Param name', 'Value']
                    with doc.create(pl.LongTabu("X[c] X[c]")) as table:
                        table.add_row(first_row, mapper=[bold])
                        table.add_hline()
                        for arr in self.data_info:
                            table.add_row(arr)
                        table.add_hline()

            if self._add_section['object_stretches']:
                with doc.create(pl.Subsection('Object stretches:')):
                    first_row = ['ID', 'range']
                    self.log_file_names = np.array(self.log_file_names)
                    for band in self.bands:
                        band_mask = np.array(['band_' + str(band) in filename for filename in self.log_file_names])
                        log_files = self.log_file_names[band_mask]
                        with doc.create(pl.Subsubsection('For band ' + str(band))):
                            with doc.create(pl.LongTabu("X[c] X[c]")) as table:
                                table.add_row(first_row, mapper=[bold])
                                table.add_hline()
                                k_id = 1
                                for log in log_files:
                                    id = str(band) + '_' + str(k_id)
                                    to_add = log.split('__')[0]
                                    to_add = to_add.split('/')[-1]
                                    table.add_row([id, to_add])
                                    k_id = k_id + 1
                                table.add_hline()
                doc.append(pl.UnsafeCommand('clearpage'))

            if self._add_section['data_graphics']:
                with doc.create(pl.Subsection('Data graphics:')):
                    #data graphics generated by the TBS module
                    images = list_archives(self.str_adress + '/params_log', format='.pdf')
                    for image in images:
                        with doc.create(pl.Figure(position='h!')) as data_image:
                            print (image)
                            data_image.add_image(short_path + '/' + image, width=NoEscape(self._params['fig_width']))

                doc.append(pl.UnsafeCommand('clearpage'))

        if self._add_section['global_elem_analysis']:
            with doc.create(pl.Section('Global element analysis')):
                #we list all rates for all elements
                with doc.create(pl.Center()) as summ_center:
                    for ind_band in range(len(self.bands)):

                        summary_histogram = self._summary_histogram(self.elem_counts[ind_band], band=self.bands[ind_band],
                                                                    path=long_path, legend=(ind_band % 2 == 0))
                        print (summary_histogram)
                        with summ_center.create(pl.Figure(position='h!')) as plot:
                            plot.add_image(summary_histogram, width=NoEscape(self._params['fig_width']))
                            plot.add_caption('Summary for elements on band ' + str(self.bands[ind_band]))
                            plt.clf()
            doc.append(pl.UnsafeCommand('clearpage'))

        if self._add_section['dist_plots']:
            with doc.create(pl.Section('Distribution plots')):
                vars = ['log_gf', 'std_residuals', 'norm_depth']
                with doc.create(pl.Center()) as centered:
                    for var_x in vars:
                        with centered.create(pl.Figure(position='h!')) as latex_fig:
                            for band in self.bands:
                                dist_plot = self._distribution_plot(band, var_x, path=long_path)
                                with latex_fig.create(pl.SubFigure(position='b',
                                                                  width=NoEscape(r'0.5\linewidth'))) as plot:
                                    plot.add_image(long_path + '/' + dist_plot, width=NoEscape(r'\linewidth'))
                                    plt.clf()
                doc.append(pl.UnsafeCommand('clearpage'))


        if self._add_section['transitions']:
            with doc.create(pl.Section('Transitions showing absorption')):
                #showing detected transitions
                mod_k = 3
                for ind_band in range(len(self.bands)):
                    banded_dict = self.detection_figs[ind_band]
                    k = 0
                    with doc.create(pl.Subsection('Band ' + str(self.bands[ind_band]))):
                        for obs_wv in np.sort(list(banded_dict.keys())):
                            wv_data = banded_dict[obs_wv]
                            if k % (2*mod_k) == 0 or k == 0:
                                if k != 0:
                                    subsec.append(pl.UnsafeCommand('clearpage'))
                                    doc.append(subsec)
                                subsec = pl.Subsubsection('Starting at ' + str(round(obs_wv, 2)))
                            elif k % 2 == 0 and k != 0:
                                subsec.append(pl.UnsafeCommand('clearpage'))

                            with subsec.create(pl.Center()) as centered_det:
                                with centered_det.create(pl.Figure(position='h!')) as summ_plot:
                                    summ_plot.add_image(short_path+'/'+ wv_data[0], width=NoEscape(self._params['fig_width']))
                                with centered_det.create(pl.Tabu("X[r] X[r] X[r] X[r] X[r] X[r]", to="10cm", pos='h!')) as det_table:
                                    header_row1 = ["Line", "Elem", "log(gf)", "Obs wv", "Sigmas", "Rel Sigmas"]
                                    det_table.add_row(header_row1, mapper=[bold])
                                    det_table.add_hline()
                                    ranged_df = wv_data[1].sort_values(by='log_gf', ascending=False)
                                    for index, row in ranged_df.iterrows():
                                        hyper_elem = pl.UnsafeCommand('hyperref', options='ssubsec:'+row['elem'],
                                                                    arguments=[row['elem']])
                                        det_table.add_row(index, hyper_elem, round(row['log_gf'], 3), round(row['obs_wv'], 3)
                                                          , round(row['rel_sigmasIO'], 2), round(row['normed_absorption'], 2))

                            k += 1
                            if k == len(list(banded_dict.keys())):
                                doc.append(subsec)

            doc.append(pl.UnsafeCommand('clearpage'))
        else:
            print ('You are not showing L plots of detected transitions.')

        with doc.create(pl.Section('Analysis per element')):
            with doc.create(pl.Subsection('With detections')):
                #On the element analysis we include the relative sigmas, the noise plot and the list of lines
                detected_data = self.normed_data[self.normed_data.detection]
                self.detected_elems = np.unique(detected_data.elem.values)
                for elem in np.sort(self.detected_elems):
                    doc.append(pl.UnsafeCommand('clearpage'))
                    with doc.create(pl.Subsubsection(elem)):
                        for ind_band in range(len(self.bands)):
                            elem_table_dict = self.atom_tables[ind_band]
                            elem_table = elem_table_dict[elem]
                            band = self.bands[ind_band] * 1.0

                            figsize = (10, 6)
                            sigmas_plot = self.summarize_atom(elem, elem_table, var_x='log_gf',
                                                              var_y='normed_absorption',
                                                              var_z='', fname_prefix='atom_', n_z=10 ** 2,
                                                              directory=self._params['directory'], band_flag=band,
                                                              add_block=False, add_suptitle=False, figsize=figsize)
                            custom_plot = self.summarize_atom(elem, elem_table, var_x='log_gf', var_y='std_residuals',
                                                              var_z='', fname_prefix='summ_', n_z=10 ** 2,
                                                              add_block=False,
                                                              directory=self._params['directory'], band_flag=band,
                                                              alpha_ini=0.8, add_suptitle=False, figsize=figsize)

                            if not str(sigmas_plot) == 'problem':
                                with doc.create(pl.Center()) as centered:
                                    with centered.create(pl.Figure(position='h!')) as plot:
                                        graphic_name = long_path + '/' + sigmas_plot
                                        plot.add_image(graphic_name, width=NoEscape(self._params['fig_atom_width']))
                                    with centered.create(pl.Figure(position='h!')) as scnd_plot:
                                        graphic_name = long_path + '/' + custom_plot
                                        scnd_plot.add_image(graphic_name, width=NoEscape(self._params['fig_atom_width']))

                                    with centered.create(pl.Tabular("cccccccc")) as table:
                                        header_row1 = ["Band", " ", pl.UnsafeCommand('ensuremath',
                                                        arguments =[r'\lambda_{lab} (\lambda_{obs})']),
                                                       "log(gf)", "normed depth", "sigmas", "rel sigmas",
                                                       "noise"]
                                        table.add_row(header_row1)
                                        if self._params['loggf_table_filter'] > -100:
                                            elem_table = elem_table[
                                                elem_table.log_gf > self._params['loggf_table_filter']]
                                        if self._params['depth_table_filter'] > 0:
                                            elem_table = elem_table[
                                                elem_table.norm_depth > self._params['depth_table_filter']]
                                        elem_table = elem_table.sort_values(by='log_gf', ascending=False)
                                        lrows = len(elem_table)
                                        if self._params['number_table_filter'] != 0:
                                            max_rows = np.min([lrows, self._params['number_table_filter']])
                                        else:
                                            max_rows = lrows
                                        c = 0
                                        for index, row in elem_table.iterrows():
                                            if row['line_class'] != 2:
                                                class_marker = self._class_to_latex(row['line_class'])
                                                if c == 0:
                                                    to_add = [pl.MultiRow(max_rows, data=self.bands[ind_band])]
                                                else:
                                                    to_add = ['']
                                                wobs = str(round(row['obs_wv'], 3)).split('.')
                                                if '(' in index:
                                                    wlab = float(index.split('(')[0])
                                                else:
                                                    wlab = float(index)
                                                one = wobs[0][-1]
                                                decimal = wobs[1]
                                                if one == index.split('.')[0][-1]:
                                                    diff = ('.'+decimal)
                                                else:
                                                    diff = one+'.'+decimal

                                                to_add.extend(
                                                    [class_marker, str(round(wlab, 3)) + '('+diff+')',
                                                     round(row['log_gf'], 2), round(row['norm_depth'], 2),
                                                     round(row['rel_sigmasIO'], 2), round(row['normed_absorption'], 2),
                                                     round(row['std_residuals'], 3)])
                                                table.add_row(to_add)
                                                c += 1
                                            if self._params['number_table_filter'] != 0 and c > max_rows:
                                                break
                                doc.append(pl.UnsafeCommand('clearpage'))

            if self._add_section['elems_wout_det']:
                with doc.create(pl.Subsection('Without detections')):
                    for ind_band in range(len(self.bands)):
                        elem_table_dict = self.atom_tables[ind_band]
                        band = self.bands[ind_band] * 1.0
                        with doc.create(pl.Subsubsection('Band '+str(band))):
                            banded_data = self.normed_data[self.normed_data.band==band]
                            all_elems = np.unique(banded_data['elem'].values)
                            non_detected_mask = np.array([not elem in self.detected_elems for elem in all_elems])
                            non_detected_elems = all_elems[non_detected_mask]
                            n_z = np.median(banded_data['std_residuals'].values)
                            for elem in np.sort(non_detected_elems):
                                elem_table = elem_table_dict[elem]
                                figsize = (10, 6)
                                elem_plot = self.summarize_atom(elem, elem_table, var_x='log_gf',
                                                                  var_y='normed_absorption',
                                                                  var_z='', fname_prefix='atom_', n_z=10 ** 2,
                                                                  directory=self._params['directory'], band_flag=band,
                                                                  add_block=False, add_suptitle=False, figsize=figsize)
                                if str(elem_plot) != 'problem':
                                    with doc.create(pl.Center()) as centered:
                                        with centered.create(pl.Figure(position='h!')) as plot:
                                            graphic_name = long_path + '/' + elem_plot
                                            plot.add_image(graphic_name, width=NoEscape(self._params['fig_atom_width']))
                                else:
                                    doc.append(elem +' presented a code problem unu')

                doc.append(pl.UnsafeCommand('clearpage'))
        if self._add_section['neg_transitions']:
            with doc.create(pl.Section('Negative results for detected elements')):
                for band in self.bands:
                    k = 0
                    banded_data = self.normed_data[self.normed_data.band==band]
                    prev_len = len(banded_data)
                    banded_data = banded_data.dropna(subset=['image'])
                    if prev_len != len(banded_data):
                        print (prev_len - len(banded_data),  'NaN values in neg transitions, at band ', band)

                    with doc.create(pl.Subsection('Band '+str(band))) as subsec:
                        for elem in self.detected_elems:
                            elem_data = banded_data[banded_data.elem==elem]
                            if len(elem_data) != 0:
                                if k != 0:
                                    subsec.append(pl.UnsafeCommand('clearpage'))
                                    k = 0
                                with subsec.create(pl.Subsubsection(elem)) as subsubsec:
                                    with subsubsec.create(pl.Center()) as centered:
                                        non_detected_data = elem_data[elem_data.detection == False]
                                        for index, row in non_detected_data.iterrows():
                                            if row['image'] != 'problem':
                                                if k % 2 == 0 and k != 0:
                                                    centered.append(pl.UnsafeCommand('clearpage'))
                                                with centered.create(pl.Figure(position='h!')) as summ_plot:
                                                    summ_plot.add_image(short_path + '/' + row['image'],
                                                                    width=NoEscape(self._params['fig_width']))
                                                k +=1

        if self._add_section['resume_fwhm']:
            doc.append(pl.UnsafeCommand('clearpage'))
            one_band_data = self.normed_data[self.normed_data.band == self.bands[0]]
            snr_epoch = one_band_data['std_residuals'].values
            one_band_data['snr_epoch'] = 1 / snr_epoch
            snr_epoch = one_band_data['snr_epoch'].values
            if self._params['read_fits'] and not (self._params['telescope'] in ['uves', 'harps']):
                self._ranges = []
                self.orders_length = []
                array_files = np.array(list_archives(self._get_data_folder()))
                for order in range(int(self._params['orders'])):
                    if self._params['telescope'] == 'keck':
                        wave, data, error = openFile(array_files[0], order, orders_together=False,
                                                     using_keck=True)
                    elif self._params['telescope'] == 'uves':
                        from astropy.io import fits
                        data = fits.open(array_files[0])
                        wave = np.array(data[1].data[0])[0]
                    else:
                        from pyspeckit import Spectrum
                        data = Spectrum(str(array_files[0]))
                        wave = np.array(data[order].xarr)
                    cut_order = int(self._params['cut_order'])
                    self._ranges.append([wave[cut_order], wave[len(wave) - cut_order - 1]])
                    self.orders_length.append([abs(wave[cut_order] - wave[cut_order + 1]),
                                               abs(wave[len(wave) - cut_order - 2] - wave[len(wave) - cut_order - 1])])

                orders_length = np.array(self.orders_length)
                for index, row in one_band_data.iterrows():
                    wv = row['obs_wv']
                    mask_wv = np.array([wv_lims[0] < wv and wv < wv_lims[1] for wv_lims in self._ranges])
                    wv_delta = np.mean(orders_length[mask_wv])
                    one_band_data.at[index, 'snr_epoch'] = row['snr_epoch']/np.sqrt(wv_delta)



            with doc.create(pl.Section('Noise analysis')):
                fig = plt.figure(figsize=(10, 7))
                ax = fig.add_subplot(111)
                ax.plot(one_band_data['obs_wv'], one_band_data['snr_epoch'], '.')
                snr_image_name = self.str_adress + '/atom_data/snr_plot.png'
                fig.savefig(snr_image_name, dpi=150)
                with doc.create(pl.Figure(position='h!')) as plot:
                    plot.add_image(short_path + '/' + snr_image_name, width=NoEscape(self._params['fig_width']))


            with doc.create(pl.Section('FWHM vs wavelength for logs')):
                fwhm = one_band_data['fwhm'].values
                fwhm_pix = one_band_data['fwhm_pix'].values
                snr = one_band_data['snr'].values
                snr_epoch = one_band_data['snr_epoch'].values
                scnd_filter = [aux < 100 for aux in fwhm_pix]
                fwhm = fwhm[scnd_filter]
                fwhm_pix = fwhm_pix[scnd_filter]
                snr = snr[scnd_filter]
                snr_epoch = snr_epoch[scnd_filter]
                trd_filter = [aux < 1E308 for aux in snr]
                fwhm = fwhm[trd_filter]
                fwhm_pix = fwhm_pix[trd_filter]
                snr = snr[trd_filter]
                snr_epoch = snr_epoch[trd_filter]
                first_row = ['FWHM (angstrom)', 'FWHM (pix)', 'SNR (transit)', 'SNR (epoch)']
                plt.close('all')
                fig = plt.figure(figsize=(10, 10))
                grid = gridspec.GridSpec(2, 2)
                bins = 10
                ax_fwhm = fig.add_subplot(grid[0, 0], facecolor='floralwhite')
                ax_fwhm_pix = fig.add_subplot(grid[0, 1], facecolor='floralwhite')
                ax_snr = fig.add_subplot(grid[1, 0], facecolor='floralwhite')
                ax_snr_epoch = fig.add_subplot(grid[1, 1], facecolor='floralwhite')
                is_not_nan = np.array([not b for b in sp.isnan(fwhm)])
                ax_fwhm.hist(fwhm[is_not_nan], bins=bins)
                ax_fwhm.set_ylabel('FWHM hist')
                is_not_nan = np.array([not b for b in sp.isnan(fwhm_pix)])
                ax_fwhm_pix.hist(fwhm_pix[is_not_nan], bins=bins)
                ax_fwhm_pix.set_ylabel('FWHM(pix) hist')
                is_not_nan = np.array([not b for b in sp.isnan(snr)])
                ax_snr.hist(snr[is_not_nan], bins=bins)
                ax_snr.set_ylabel('SNR per transit hist')
                is_not_nan = np.array([not b for b in sp.isnan(snr_epoch)])
                ax_snr_epoch.hist(snr_epoch[is_not_nan], bins=bins)
                ax_snr_epoch.set_ylabel('SNR per epoch hist')
                fwhm_image_name = self.str_adress + '/atom_data/summary_log.pdf'
                fig.savefig(fwhm_image_name, dpi=150)

                with doc.create(pl.Figure(position='h!')) as plot:
                    plot.add_image(short_path + '/' + fwhm_image_name, width=NoEscape(self._params['fig_width']))

                with doc.create(pl.LongTabu('X[c] X[c] X[c] X[c]')) as table:
                    table.add_row(first_row, mapper=[bold])
                    table.add_hline()
                    aux_row = [round(np.nanmean(fwhm), 3), round(np.nanmean(fwhm_pix), 2),
                               round(np.nanmean(snr), 2), round(np.nanmean(snr_epoch), 2)]
                    aux_row_median = [round(np.nanmedian(fwhm), 3), round(np.nanmedian(fwhm_pix), 2),
                                      round(np.nanmedian(snr), 2), round(np.nanmedian(snr_epoch), 2)]
                    aux_row_std = [round(np.nanstd(fwhm), 3), round(np.nanstd(fwhm_pix), 2),
                                   round(np.nanstd(snr), 2), round(np.nanstd(snr_epoch), 2)]
                    table.add_row(aux_row, mapper=[bold])
                    table.add_row(aux_row_median, mapper=[bold])
                    table.add_row(aux_row_std, mapper=[bold])

                doc.append('Where the last values were mean, median and std, respectively. ')

                tot_lines_counter = len(self.atom_data[self.atom_data.band == self.bands[0]])
                accepted_lines_counter = len(self.normed_data[self.normed_data.band == self.bands[0]])
                doc.append('\n' + str(tot_lines_counter) + ' lines were tried and ')
                doc.append(str(accepted_lines_counter) + ' were analyzed. (accepted/tried = ')
                doc.append(str(round(accepted_lines_counter / tot_lines_counter, 3)) + ').')
                doc.append(pl.UnsafeCommand('clearpage'))





        doc.generate_tex()
        doc.generate_pdf(clean_tex=False, compiler='pdflatex')

    def create_appendix(self, opt_path=''):
        path = os.path.abspath('')
        path += '/'+self.str_adress+'/atom_data'
        doc = pl.Document(self.str_adress + '/' + self._params['directory'] + '/' + self._params['app_filename'],
                          geometry_options=self._params['report_geometry'])
        doc.packages.append(pl.Package('hyperref'))
        doc.packages.append(pl.Package('graphicx'))
        doc.packages.append(pl.Package('array'))
        doc.append(pl.UnsafeCommand('newcolumntype', options = ['1'],
                         arguments=['X', r'>{\centering\let\newline\\\arraybackslash\hspace{0pt}}m{#1}']))
        detected_data = self.normed_data[self.normed_data.detection]
        self.detected_elems = np.unique(detected_data.elem.values)
        doc.append(pl.UnsafeCommand('graphicspath', arguments=['{' + path + '}']))

        with doc.create(pl.Section('Global element analysis', label=False)):
            # we list all rates for all elements
            with doc.create(pl.Center()) as summ_center:
                for ind_band in range(len(self.bands)):
                    summary_histogram = self._summary_histogram(self.elem_counts[ind_band], band=self.bands[ind_band],
                                                                path=path, legend=(ind_band % 2 == 0))
                    with summ_center.create(pl.Figure(position='!ht')) as plot:
                        plot.add_image(opt_path + summary_histogram, width=NoEscape(self._params['fig_width']))
                        plot.add_caption('Summary for elements on band ' + str(self.bands[ind_band]))
                        plt.clf()
        doc.append(pl.UnsafeCommand('clearpage'))


        skip_params = ['dlambda_analyze', 'alpha_mid', 'alpha_side', 'dopplerFWHM', 'dpi_figs', 'eq_space_tolerance',
                       'ignore_tests', 'inin_ok', 'max_dopp_residual', 'max_figs', 'max_t_len', 'merge_in', 'omit_hists',
                       'tell_limit', 'times_critic_redf', 'isout_ok', 'outout_ok'
                       ]
        with doc.create(pl.Section('Algorithm parameters:', label=False)):
            # parameters used in this run
            params_data = self.params_data.sort_index()
            header_row = ['Parameter name', 'Value']
            with doc.create(pl.LongTabu("X[c] X[c]")) as table:
                table.add_row(header_row, mapper=[bold])
                table.add_hline()
                for param, value in zip(params_data.index.values, params_data.values):
                    if not param in skip_params:
                        table.add_row(param, value[0])
                        table.add_hline()

        with doc.create(pl.Subsection('Object information:', label=False)):
            # list transit data information
            first_row = ['Parameter name', 'Value']
            with doc.create(pl.LongTabu("X[c] X[c]")) as table:
                table.add_row(first_row, mapper=[bold])
                table.add_hline()
                for arr in self.data_info:
                    table.add_row(arr)
                table.add_hline()
        doc.append(pl.UnsafeCommand('clearpage'))


        with doc.create(pl.Section('Elements with detections', label=False)):
            for ind_band in range(len(self.bands)):
                band = self.bands[ind_band] * 1.0
                with doc.create(pl.Subsection(str(band) +' angstrom band', label=False)):
                    with doc.create(pl.LongTabu("m{1cm} m{7cm} m{7cm}")) as master_table:
                        header_row = ['Atom', 'Sigmas plot', 'Noise plot']
                        master_table.add_row(header_row, mapper=[bold])
                        master_table.add_hline()
                        for elem in self.detected_elems:
                            elem_table_dict = self.atom_tables[ind_band]
                            elem_table = elem_table_dict[elem]
                            figsize = (10, 6)
                            sigmas_plot = self.summarize_atom(elem, elem_table, var_x='log_gf', var_y='normed_absorption',
                                                              var_z='', fname_prefix='atom_', n_z=10 ** 2,
                                                              directory=self._params['directory'], band_flag=band,
                                                              add_block=False, add_suptitle=False, figsize=figsize)
                            custom_plot = self.summarize_atom(elem, elem_table, var_x='log_gf', var_y='std_residuals',
                                                              var_z='', fname_prefix='summ_', n_z=10 ** 2, add_block=False,
                                                              directory=self._params['directory'], band_flag=band,
                                                              alpha_ini=0.8, add_suptitle=False, figsize=figsize)

                            fig1 = pl.UnsafeCommand('includegraphics', options='width=' + self._params['fig_appendix'],
                                                    arguments=[opt_path+sigmas_plot])

                            fig2 = pl.UnsafeCommand('includegraphics', options='width=' + self._params['fig_appendix'],
                                                    arguments=[opt_path+custom_plot])
                            master_table.add_row(self._elem_name(elem), fig1, fig2)
                doc.append(pl.UnsafeCommand('clearpage'))

        with doc.create(pl.Section('Detected elements transitions', label=False)):
            for ind_band in range(len(self.bands)):
                band = self.bands[ind_band] * 1.0
                with doc.create(pl.Subsection(str(band) + ' angstrom band', label=False)) as subsec:
                    with subsec.create(pl.LongTabu("C{0.8cm} C{0.2cm} C{2.2cm} C{1cm} C{1.5cm} C{1cm} C{1cm} C{1cm}")) as table:
                        header_row1 = ["Atom", " ", pl.UnsafeCommand('ensuremath',
                                                                     arguments=[r'\lambda_{lab} (\lambda_{obs})']),
                                       "log(gf)", "normed depth", pl.UnsafeCommand('ensuremath',
                                                                     arguments=[r'\sigma_{In-Out}']), pl.UnsafeCommand('ensuremath',
                                                                     arguments=[r'\sigma_{rel}']), pl.UnsafeCommand('ensuremath',
                                                                     arguments=[r'\sigma_c'])]
                        table.add_row(header_row1, strict=False)

                        for elem in self.detected_elems:
                            table.add_hline()
                            elem_table_dict = self.atom_tables[ind_band]
                            elem_table = elem_table_dict[elem]
                            if self._params['loggf_table_filter'] > -100:
                                elem_table = elem_table[
                                    elem_table.log_gf > self._params['loggf_table_filter']]
                            if self._params['depth_table_filter'] > 0:
                                elem_table = elem_table[
                                    elem_table.norm_depth > self._params['depth_table_filter']]
                            elem_table = elem_table[elem_table.line_class != 2]
                            elem_table = elem_table.sort_values(by='log_gf', ascending=False)
                            lrows = len(elem_table)
                            if self._params['number_table_filter'] != 0:
                                max_rows = np.min([lrows, self._params['number_table_filter']])
                            else:
                                max_rows = lrows
                            c = 0
                            for index, row in elem_table.iterrows():
                                class_marker = self._class_to_latex(row['line_class'])
                                if c == 0:
                                    to_add = [pl.MultiRow(max_rows, data=self._elem_name(elem))]
                                else:
                                    to_add = ['']
                                wobs = str(round(row['obs_wv'], 3)).split('.')
                                if '(' in index:
                                    wlab = float(index.split('(')[0])
                                else:
                                    wlab = float(index)
                                one = wobs[0][-1]
                                decimal = wobs[1]
                                if one == index.split('.')[0][-1]:
                                    diff = ('.' + decimal)
                                else:
                                    diff = one + '.' + decimal

                                to_add.extend(
                                    [class_marker, str(round(wlab, 3)) + '(' + diff + ')',
                                     round(row['log_gf'], 2), round(row['norm_depth'], 2),
                                     round(row['rel_sigmasIO'], 2), round(row['normed_absorption'], 2),
                                     round(row['std_residuals'], 3)])
                                table.add_row(to_add, strict=False)
                                c += 1
                                if self._params['number_table_filter'] != 0 and c > max_rows:
                                    break


                doc.append(pl.UnsafeCommand('clearpage'))



        doc.generate_tex()
        doc.generate_pdf(clean_tex=False, compiler='pdflatex')




if __name__ == "__main__":
    #bands = [0.5, 0.75, 1]
    #ylims_atom = [-0.0018, 0.0018]

    import seaborn as sb
    sb.set()
    sb.set_context('poster')

    def_config = eval(input('Do you want to use the configuration default? (True/False)'))
    depth_median = 0
    if def_config:
        str_adress = 'final_test/nicola_data'
        bands = [0.75, 1]
        save_report_obj = True
        app_dir = 'app/hd-2094/'
        print ('Analyzing', str_adress, 'at the bands', bands)
    else:
        str_adress = str(input('What address do you want to summarize?'))
        #save_logs = str(input('Do you want to save the logs as pdf? (True to yes, False to no)'))
        #bands = eval(input('What bands do you want to summarize?(Write an array)'))
        #save_report_obj = eval(input('Do you want to use the report object? (True/False)'))
        bands = [0.75, 1]
        save_report_obj = True
        app_dir = str(input('appendix directory?'))
        if app_dir != '':
            app_dir = 'app/'+ app_dir +'/'


    using_harps = np.any([aux_str in str_adress for aux_str in ['wasp-117b', 'wasp-23b', 'hd_3']])
    report = UpdtReport(str_adress, bands, using_harps=using_harps, filename='updt_report',
                        depth_table_filter=depth_median)
    report.create_report()
    #report.create_appendix(opt_path = app_dir)
    if save_report_obj:
        import pickle
        path = os.path.abspath('')
        fname = path +'/'+ str_adress + '/atom_data/report_obj.pickle'
        filehandler = open(fname, 'wb')
        pickle.dump(report, filehandler)

    print ('done')

