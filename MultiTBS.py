# -*- coding: latin-1 -*-
import multiprocessing as mp
import pickle
import matplotlib
from tqdm import tqdm
import time
matplotlib.use('Agg')
from functions_database import get_linedb
#import seaborn as sb
#sb.set()
from TBS import *


def chunk_it(seq, num):
    avg = len(seq) / float(num)
    out = []
    last = 0.0

    while last < len(seq):
        out.append(seq[int(last):int(last + avg)])
        last += avg

    return out

def save_TBS(obj, directory = 'objects', fname = 'TBS', sp_line = False, use_lines = True):
    band = obj._params["dlambda_analyze"]
    directory = obj.output_folder + directory
    aux_name = str(band)
    if band < 1:
        lambda_name = 'point' + str(aux_name[2:])
    else:
        lambda_name = aux_name
    if use_lines:
        df_index = obj.log_index
        low_limit = str(df_index[0])
        up_limit = str(df_index[len(df_index) - 1])
        name = directory + low_limit + '_to_' + up_limit + '_' + lambda_name
    if sp_line:
        superline = obj.superline
        fname = fname + '_spline'
        filehandler = open(fname, 'wb')
        pickle.dump(superline, filehandler)
    else:
        filehandler = open(fname, 'wb')
        pickle.dump(obj, filehandler)




def worker(str_adress, linedb, str_elems, loggfs, band, queue, consolidate_log = False,
           first = False, plot_sigmas=False, **kwargs):
    print ('start_worker')
    obj = TBS(str_adress, linedb, str_elems, loggfs, dlambda_analyze=band, save_figs=False, **kwargs)
    obj.check_epoch(plot_all=True, show_plt=False)
    if len(obj.linedb) != 0:
        obj.extract_lines(doall_plt=False, all_line_plt=-1)
        obj.telluric_corrections(all_plt=False)
        obj.snellen(do_plt=False)
        obj.redfield(do_plt=False, force_no_repeat=True)
        if len(obj.linedb.tolist()) != 0:
            obj.summary_plots(show_all=False, binning = True, paper = True, lw = 2.5)
            obj.detection_alert()

    else:
        print ("This process didn't reach the end.")

    if consolidate_log:
        obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names)
        obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names, error_log=True)
        obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names, params=True)
        if not plot_sigmas:
            obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names, sigmas=True)

    if len(obj.linedb) != 0 and plot_sigmas:
        obj.plot_sigmas(max_per_fig=50, fill_file=consolidate_log, overwrite=overwrite_logs,
                    use_lines=use_lines_names)
    result = obj
    queue.put(result)
    hour = time.asctime()

    if len(obj.linedb) != 0:
        print ('done at '+str(hour))
        sys.stdout.flush()


def calm_down(linedb, str_elems, log_gf, n_lines):
    index = range(len(linedb))
    masks = np.array(chunk_it(index, n_lines))
    linedb = np.array(linedb)
    str_elems = np.array(str_elems)
    log_gf = np.array(log_gf)
    linedb_array = []
    elems_array = []
    log_gf_array = []
    for mask in masks:
        aux_linedb = linedb[mask]
        aux_elem = str_elems[mask]
        aux_log_gf = log_gf[mask]
        linedb_array.append(aux_linedb.tolist())
        elems_array.append(aux_elem.tolist())
        log_gf_array.append(aux_log_gf.tolist())
    return linedb_array, elems_array, log_gf_array



def run_TBS(str_adress, linedb, str_elems, log_gf, bands, multi = 'bands', n_lines = 10,
            dict = {}):
    m = mp.Manager()
    result_queue = m.Queue()

    jobs = []

    if multi == 'lines':
        band = bands[0]
        use_lines_names = True

        linedb_array, elems_array, log_gf_array = calm_down(linedb, str_elems, log_gf, n_lines)

        for h in range(len(linedb_array)):
            aux_linedb = linedb_array[h]
            aux_elem = elems_array[h]
            aux_log_gf = log_gf_array[h]
            jobs.append(mp.Process(target=worker, args =[str_adress, aux_linedb, aux_elem,
                                                         aux_log_gf, band, result_queue], kwargs = dict))
            jobs[-1].start()

        for p in jobs:
            p.join()

        r = []
        for i in range(result_queue.qsize()):
            r.append(result_queue.get())

        return r


    elif multi == 'bands':
        first = True
        use_lines_names = False
        for band in bands:
            """
            if first:
                plots = [True, True, False, False, False]
                first = False
            else:
                plots = [False, False, False, False, False]
            """
            jobs.append(mp.Process(target=worker, args=[str_adress, linedb, str_elems,
                                                        log_gf, band, result_queue], kwargs=dict))
            jobs[-1].start()

        for p in jobs:
            p.join()


        r = []
        for i in range(result_queue.qsize()):
            r.append(result_queue.get())

        return r

    elif multi == 'both':
        linedb_array, elems_array, log_gf_array = calm_down(linedb, str_elems, log_gf, n_lines)
        use_lines_names = True

        for h in range(len(linedb_array)):
            aux_linedb = linedb_array[h]
            aux_elem = elems_array[h]
            aux_log_gf = log_gf_array[h]
            print ('lengths:', len(aux_linedb), len(aux_elem), len(aux_log_gf))

            for band in bands:
                jobs.append(mp.Process(target=worker, args=[str_adress, aux_linedb, aux_elem,
                                                            aux_log_gf, band, result_queue], kwargs=dict))
                jobs[-1].start()

        for p in jobs:
            p.join()

        r = []
        for i in range(result_queue.qsize()):
            r.append(result_queue.get())

        return r



    else:
        print ('please specify a way to do the multiproccessing')



def consolidate_results(results, str_adress, bands, fname = 'report', obj_name = 'TBS', do_report = False, draw_detections = True,
                        clean_figs = False, consolidate_log = False, save_logs = False, save_obj = False, save_spline = False, out_folder = 'images/'):
    count = 0

    pbar = tqdm(total = len(results))
    for obj in results:
        if save_obj:
            save_TBS(obj, obj_name+str(count))
            print ('Saving object '+ str(count))
        if save_spline:
            save_TBS(obj, obj_name+str(count), sp_line=save_spline)
            print ('Saving superline '+ str(count))

        print ('Saving plots of obj ' + str(count))
        sys.stdout.flush()
        obj.save_all_plots(clean_figs = clean_figs)
        if consolidate_log:
            obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names)
            obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names, error_log=True)
            obj.consolidate_log(overwrite=overwrite_logs, use_lines=use_lines_names, params=True)
        count = count + 1
        pbar.update(1)
    if do_report:
        try:
            report = Report(out_folder+str_adress, bands)
            report.create_report(fname=fname, draw_detections = draw_detections, save_logs = False)
        except Exception as ex:
            template = "{0}. Arguments: {1!r}"
            message = template.format(type(ex).__name__, ex.args)
            print (message)
    pbar.close()


def atomize(linedb, str_elems, log_gf, bands, n_atom, str_adress, clean_figs = True, save_logs=False,
            save_obj = False, save_spline = False, out_folder = 'images/', do_report = False):
    linedb_arrays, elems_arrays, log_gf_arrays = calm_down(linedb, str_elems, log_gf, n_atom)
    print (len(linedb_arrays), len(linedb_arrays[0]))
    for i in range(len(linedb_arrays)):
        print (len(linedb_arrays[i]), len(elems_arrays[i]))
    all_results = []
    for h in range(len(linedb_arrays)):
        aux_linedb= linedb_arrays[h]
        aux_elem = elems_arrays[h]
        aux_log_gf = log_gf_arrays[h]
        results = run_TBS(aux_linedb, aux_elem, aux_log_gf, bands, multi = 'bands')
        all_results.append(results)
        consolidate_results(results, str_adress, bands, fname = 'report'+str(h),
                            clean_figs=clean_figs, consolidate_log=True, save_logs=save_logs, save_obj=save_obj,
                            save_spline = save_spline)

    if do_report:
        report = Report(out_folder+str_adress, bands)
        report.create_report(fname='total_report', draw_detections=True)
    return all_results, linedb_arrays


def start_multiTBS(str_adress, linedb, str_elems, log_gf, bands,  part_before = 2, part_after = 30, clean_figs = True,
                    out_folder = 'images/', multip_way = 'both', save_logs = True, report_name = 'full_report',
                   save_obj = False, save_spline = False, do_report = False, clear_results = True, TBS_dict = {}):
    if part_before == 1:

        results = run_TBS(str_adress, linedb, str_elems, loggfs, bands, multi=multip_way, n_lines=part_after, dict=TBS_dict)

        consolidate_results(results, str_adress, bands, consolidate_log=True, fname=report_name,
                            do_report=do_report, save_logs=save_logs, save_obj=save_obj)

        return results
    elif part_after == 1:
        linedb_arrays, elems_arrays, log_gf_arrays = calm_down(linedb, str_elems, log_gf, part_before)
        print (len(linedb_arrays), len(linedb_arrays[0]))
        for i in range(len(linedb_arrays)):
            print (len(linedb_arrays[i]), len(elems_arrays[i]))
        all_results = []
        for h in range(len(linedb_arrays)):
            aux_linedb = linedb_arrays[h]
            aux_elem = elems_arrays[h]
            aux_log_gf = log_gf_arrays[h]
            results = run_TBS(str_adress, aux_linedb, aux_elem, aux_log_gf, bands, multi=multip_way, dict=TBS_dict)
            all_results.append(results)
            consolidate_results(results, str_adress, bands, fname=report_name + str(h),
                                clean_figs=clean_figs, consolidate_log=True, save_logs=save_logs, save_obj=save_obj,
                                save_spline=save_spline)

        if do_report:
            report = Report(out_folder + str_adress, bands, using_harps=use_harps)
            report.create_report(fname='total_report', draw_detections=True)
        return all_results, linedb_arrays

    else:
        linedb_arrays, elems_arrays, log_gf_arrays = calm_down(linedb, str_elems, log_gf, part_before)
        """
        import pdb
        pdb.set_trace()
        """
        print ('First partition: ')
        print (len(linedb_arrays), len(linedb_arrays[0]))
        for i in range(len(linedb_arrays)):
            print (len(linedb_arrays[i]), len(elems_arrays[i]))
        all_results = []
        for h in range(len(linedb_arrays)):
            aux_linedb = linedb_arrays[h]
            aux_elem = elems_arrays[h]
            aux_log_gf = log_gf_arrays[h]
            results = run_TBS(str_adress, aux_linedb, aux_elem, aux_log_gf, bands, multi=multip_way, n_lines=part_after,
                              dict=TBS_dict)

            if not clear_results:
                all_results.append(results)
            if do_report and h == len(linedb_arrays) - 1:
                do_report = True
            print ('Set', h, 'of', len(linedb_arrays))
            consolidate_results(results, str_adress, bands, fname=report_name + str(h),
                                clean_figs=clean_figs, consolidate_log=True, save_logs=save_logs, save_obj=save_obj,
                                save_spline=save_spline, do_report = do_report, out_folder=out_folder)
            if clear_results:
                del results

        if not clear_results:
            return all_results, linedb_arrays
        else:
            return linedb_arrays



"""
str_elems = ['Sc', 'Ca', 'Ca', 'Ca']
linedb = [5526.8, 6162.2, 6439.5, 6493.8]  # indexFWHM = 12, norm_width = 4
loggfs = [1, 2, 3, 4]
"""





# Params to the object

#test_percent = 0.5
overwrite_logs = True
# plot = [first, extraction, telluric, snellen, redfield]
# plots = [True, False, False, False, False]
use_lines_names = True
out_folder = 'final/'
#####SET YOUR OBJECT HERE#####
str_adress = 'wasp-74b'
##############################
use_bar = False

needed_folders = [
    r'' + out_folder + str_adress + '/first_epoch',
    r'' + out_folder + str_adress + '/atom_data',
    r'' + out_folder + str_adress + '/captains_log',
    r'' + out_folder + str_adress + '/extract_lines',
    r'' + out_folder + str_adress + '/other_epoch',
    r'' + out_folder + str_adress + '/redfield',
    r'' + out_folder + str_adress + '/tspectra',
    r'' + out_folder + str_adress + '/snellen',
    r'' + out_folder + str_adress + '/summary_plots',
    r'' + out_folder + str_adress + '/telluric',
    r'' + out_folder + str_adress + '/error_log',
    r'' + out_folder + str_adress + '/objects',
    r'' + out_folder + str_adress + '/params_log'
]
for folder in needed_folders:
    if not os.path.exists(folder):
        os.makedirs(folder)
        print ('Creating needed folder: ' + folder)

if 'hd-1490' in str_adress:
    orders = 17
    if 'blue' in str_adress:
        orders = 21
    print('hola')
    TBS_dict = {
        'output_folder': out_folder,
        'orders': orders,
        'indexFWHM': 11,
        'error_gain': 2.19,
        'dtransit': 195 / (24 * 60),
        't_transit': 2453527.872,
        'norm_width': 5,
        'norm_high': 3.5,
        'test_boundary': 0.027,
        'period': 2.87618,
        'wave_binning': 0,
        'merge_epoch_as': ['0', '1', '2', '3:5', '5:7', '8', '9', '10', '11:13', '14', '15:17', '17', '18', '19'],
        'telescope': 'keck',
        'radial_velocity': 12.3,
        'do_1stdopp': True,
        'dlt_epochs_again': [],
        'use_bar': use_bar,
        'tell_only_out': False
    }


    if 'green' in str_adress:
        low_limit = 4935
        up_limit = 6421
    elif 'blue' in str_adress:
        low_limit = 3802
        up_limit = 4925

elif 'nicola_data' in str_adress:
    low_limit = 5485
    up_limit = 6891
    TBS_dict = {
        'output_folder': out_folder,
        'orders': 22,
        'indexFWHM': 10,
        'error_gain': 1.6,
        'dtransit': 0.127951389,
        't_transit': 2452826.628521,
        'period': 3.52474859,
        'do_1stdopp': False,
        'dlt_epochs_again': [0, 1],
        'tell_only_out': True,
        'norm_width': 5,
        'norm_high': 3,
        'wave_binning': 0,
        'test_percent': 0.7,
        'test_boundary': 0.011,
        'use_bar': use_bar,
        'telescope': 'subaru'
    }
elif 'data' == str_adress:
    low_limit = 6330
    up_limit = 6799
    TBS_dict = {
        'output_folder': out_folder,
        'orders': 16,
        'indexFWHM': 50,
        'error_gain': 1,
        'dtransit': 0.1294,
        't_transit': 2454884.02817,
        'period': 21.2163979,
        'do_1stdopp': False,
        'dlt_epochs_again': [0, 1],
        'tell_only_out': False,
        'wave_binning': 0,
        'first_epoch_pixs': 700,
        'analyze_pixs':400,
        'norm_width': 1.25,
        'norm_high': 3,
        'test_percent': 0.7,
        'test_boundary': 0.009,
        'use_bar': use_bar,
        'telescope': 'subaru'
    }

elif str_adress == 'hd-3_1':
    low_limit = 3782
    up_limit = 6912
    TBS_dict = {
       'output_folder': out_folder,
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
        'use_bar': use_bar,
        'norm_width':5,
        'test_boundary': 0.013,
        'norm_high': 3
    }

elif str_adress == 'hd-3_2':
    low_limit = 3782
    up_limit = 6912
    TBS_dict = {
       'output_folder': out_folder,
        'orders': 0,
        'indexFWHM': 20,
        'error_gain': 1.36,
        'dtransit': 0.07527,
        't_transit': 2454279.436714,
        'period': 2.21857567,
        'do_1stdopp': False,
        'dlt_epochs_again': [0, 1, 2, 35, 36, 45, 46],
        'tell_only_out': False,
        'wave_binning': 0,
        'first_epoch_pixs': 600,
        'analyze_pixs': 300,
        'telescope': 'harps',
        'norm_high': 1,
        'use_bar': use_bar,
        'redf_loops': 2000
    }


elif str_adress == 'wasp-23b':
    low_limit = 3782
    up_limit = 6912
    TBS_dict = {
       'output_folder': out_folder,
        'orders': 0,
        'indexFWHM': 12,
        'error_gain': 1.36,
        'dtransit': 0.09976,
        't_transit': 2455320.12363,
        'period': 2.9444256,
        'do_1stdopp': True,
        'radial_velocity': 5.69160,
        'dlt_epochs_again': [0, 27],
        'tell_only_out': True,
        'wave_binning': 2,
        'telescope': 'harps',
        'norm_high': 3.5,
        'test_boundary': 0.13,
        'use_bar': use_bar,
        'test_percent': 0.7
    }

elif str_adress == 'wasp-117b':
    low_limit = 3782
    up_limit = 6912
    TBS_dict = {
       'output_folder': out_folder,
        'orders': 0,
        'indexFWHM': 12,
        'error_gain': 1.36,
        'dtransit': 0.2475,
        't_transit': 2456533.82326,
        'period': 10.02165,
        'do_1stdopp': True,
        'radial_velocity': -18.071,
        'dlt_epochs_again': [],
        'tell_only_out': False,
        'wave_binning': 2,
        'telescope': 'harps',
        'norm_high': 3.5,
        'test_boundary': 0.1, #LOOK AT THIS
        'use_bar': use_bar,
        'test_percent': 0.2 #LOOK AT THIS
    }

elif 'hatp-2b' in str_adress:
    if 'blue' in str_adress:
        low_limit = 3761
        up_limit = 4926
        orders = 22
        error_gain = 1.95
    elif 'green' in str_adress:
        low_limit = 4911
        up_limit = 6422
        orders = 17
        error_gain = 2.09
    elif 'red' in str_adress:
        low_limit = 6547
        up_limit = 7990
        orders = 10
        error_gain = 2.09

    TBS_dict = {
        'output_folder': out_folder,
        'orders': orders,
        'indexFWHM': 56,
        'error_gain': error_gain,
        'dtransit': 0.1787,
        't_transit': 2454212.8561,
        'period': 5.6334729,
        'wave_binning': 0,
        'n_epoch_merge': 3,
        'norm_width': 1.25,
        'smart_merge': True,
        'test_percent': 0.7,
        'test_boundary': 0.1,
        'telescope': 'keck',
        'radial_velocity': 45,
        'do_1stdopp': True,
        'dlt_epochs_again': [0, 1, 2, 94, 95, 96],
        'use_bar': use_bar,
        'tell_only_out': False
    }

elif str_adress == 'wasp-74b':
    low_limit = 4726
    up_limit = 6836
    TBS_dict = {
        'output_folder': out_folder,
        'orders': 0,
        'indexFWHM': 22,
        'N_pixs': 400,
        'error_gain': 0,
        'dtransit': 0.0955,
        't_transit': 2456506.8918,
        'period': 2.13775,
        'n_epoch_merge': 0,
        'ignore_tests': False,
        'wave_binning': 2,
        'norm_width': 5,
        'norm_high': 3.5,
        'tell_only_out': True,
        'test_boundary': 0.029,
        'telescope': 'uves',
        'dlt_epochs': 0,
        'test_percent': 0.7,
        'use_bar': use_bar,
        'radial_velocity': -15.767,
        'time_average_corr': True
    }

elif str_adress == 'tres-2b/green':
    low_limit = 4911
    up_limit = 6422
    TBS_dict = {
        'output_folder': out_folder,
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
        'merge_epoch_as': ['0:2', '2', '3:5', '5:7', '7:10', '11:15', '15:19', '19:23', '23:27', '27:31', '31:34',
                '34:37', '37:40', '42:45', '45:48', '48:51', '52:54', '54:56'],
        'dlt_epochs': 0,
        'use_bar': use_bar,
        'radial_velocity': 7.22
    }

#  (11-42)           0     1     2     3       4       5        6        7        8        9       10
#'merge_epoch_as': ['0:2', '2', '3:5', '5:7', '7:10', '11:15', '15:19', '19:23', '23:27', '27:31', '31:34',
#                  '34:37', '37:40', '42:45', '45:48', '48:51', '52:54', '54:56'],
#                          11       12       13       14       15       16       17

elif str_adress == 'xo-3b/green':
    low_limit = 4911
    up_limit = 6422
    TBS_dict = {
        'output_folder': out_folder,
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
        'use_bar': use_bar,
        'radial_velocity': 58.99
    }


elif 'hd-1897_uves' in str_adress:
    if 'red' in str_adress:
        low_limit = 5656
        up_limit = 9462
    elif 'blue' in str_adress:
        low_limit = 3733
        up_limit = 4998
    TBS_dict = {
       'output_folder': out_folder,
        'orders': 0,
        'indexFWHM': 12,
        'error_gain': 1.36,
        'dtransit': 0.07527,
        't_transit': 2454279.436714,
        'period': 2.21857567,
        'do_1stdopp': True,
        'radial_velocity': -2.55,
        'dlt_epochs_again': [],
        'tell_only_out': True,
        'wave_binning': 2,
        'telescope': 'uves',
        'use_bar': use_bar,
        'norm_width': 5,
        'test_boundary': 0.021,
        'norm_high': 3,
        'n_epoch_merge': 4,
    }




linedb, str_elems, loggfs = get_all_linedb(folders = ['spectroweb_lines', 'vald_lines'],
                                           wv_lims= [low_limit, up_limit], loggf_lims=[-5, 100],
                                           elem_key='', elem_delete='')
print ('You will analyze ' +str(len(linedb)) + ' lines')
bands = [0.75, 1]

results = start_multiTBS(str_adress, linedb, str_elems, loggfs, bands, part_before=15, part_after=10, clean_figs=True,
                         out_folder=out_folder, TBS_dict=TBS_dict, multip_way='both')

using_harps = TBS_dict['telescope'] == 'harps'

from UpdtReport import *
import seaborn as sb
sb.set()
sb.set_context('poster')
report = UpdtReport(out_folder+str_adress, bands, using_harps=using_harps, )
fname = 'main_report'
report.create_report()










