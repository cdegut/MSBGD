from networkx import draw
from scipy.__config__ import show
from modules.data_structures import MSData, peak_params
from typing import Tuple
import dearpygui.dearpygui as dpg
from modules.helpers import multi_bi_gaussian
import numpy as np
from modules.dpg_draw import draw_fitted_peaks
from modules.dpg_draw import log
import random

def run_fitting(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    dpg.show_item("Fitting_indicator")
    k = dpg.get_value("fitting_iterations")
    rolling_window_fit(spectrum, k)
    dpg.hide_item("Fitting_indicator")

def initial_peaks_parameters(spectrum:MSData):
    initial_params = []
    working_peak_list = []
    i = 0

    if spectrum.peaks is None:
        log("No peaks are detected. Please run peak detection first")
        return
    
    for peak in spectrum.peaks:
    
        x0_guess = spectrum.peaks[peak].x0_init

        if spectrum.peaks[peak].do_not_fit:
            spectrum.peaks[peak].fitted = False
            continue

        if x0_guess < spectrum.working_data[:,0][0] or x0_guess > spectrum.working_data[:,0][-1]:
            log(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        A_guess = spectrum.peaks[peak].A_init
        sigma_L_guess = sigma_R_guess = spectrum.peaks[peak].width /2
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])       
        working_peak_list.append(peak)
        i += 1
    
    if working_peak_list == []:
        log("No peaks are within the data range. Please adjust the peak detection parameters")
        return

    return initial_params, working_peak_list

def rolling_window_fit(spectrum:MSData, iterations = 1000):
    baseline_window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(baseline_window)
    initial_params, working_peak_list = initial_peaks_parameters(spectrum)
    draw_fitted_peaks(None, None, None, delete=True)
  
    i = 0
    for peak in working_peak_list:
        A_guess, x0_guess, sigma_L_guess, sigma_R_guess =  initial_params[i*4:(i+1)*4]
        log(f"Peak {peak}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}")

        spectrum.peaks[peak].A_refined = A_guess
        spectrum.peaks[peak].x0_refined = x0_guess
        spectrum.peaks[peak].sigma_L = sigma_L_guess
        spectrum.peaks[peak].sigma_R = sigma_R_guess
        spectrum.peaks[peak].fitted = False
        i += 1
 
    fit = refine_peak_parameters(working_peak_list, initial_params, spectrum, iterations)   
    if fit:
        log("Fitting done with no error")
    else:
        log("Error while fitting")
        return
    
    draw_fitted_peaks(None, None, spectrum)


def draw_residual(x_data, residual):
    dpg.show_item("residual")
    dpg.set_value("residual", [x_data, residual])

def refine_peak_parameters(working_peak_list, mbg_params, spectrum:MSData, iterations=1000):
    
    original_peaks: Tuple[int, peak_params] = {peak:spectrum.peaks[peak] for peak in working_peak_list}
    data_x = spectrum.working_data[:,0]
    data_y = spectrum.baseline_corrected[:,1]
    rmse_list = []
    iterations_list = [i for i in range(len(working_peak_list))]
    
    for k in range(iterations):  
        residual = spectrum.baseline_corrected[:,1] - multi_bi_gaussian(spectrum.baseline_corrected[:,0], *mbg_params)
        draw_residual(spectrum.baseline_corrected[:,0].tolist(), residual.tolist())
        
        rmse = np.sqrt(np.mean(residual**2))
        rmse_list.append(rmse)
        if k > 10:
            std = np.std(rmse_list[-10:])
            if std < 0.25:
                dpg.set_value("Fitting_indicator_text","Residual is stable. Done")
                break

            dpg.set_value("Fitting_indicator_text",f"Iteration {k}; RMSE: {rmse:.3f}, Residual std: {std:.3f}")

        random.shuffle(iterations_list) #do not use the same order every iteration

        for i in iterations_list:
            peak = working_peak_list[i]
            original_peak = original_peaks[peak]
            x0_fit = spectrum.peaks[peak].x0_refined
            sigma_L_fit = spectrum.peaks[peak].sigma_L
            sigma_R_fit = spectrum.peaks[peak].sigma_R

            mask = (data_x >= x0_fit - sigma_R_fit/4) & (data_x <= x0_fit + sigma_L_fit/4)
            data_x_peak = data_x[mask]
            data_y_peak = data_y[mask]
            
            if len(data_x_peak) == 0 or len(data_y_peak) == 0:
                log(f"masking error for peak {peak} block 1")              
                continue

            peak_error = np.mean(data_y_peak  - multi_bi_gaussian(data_x_peak, *mbg_params))

            spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (peak_error/10)
            mbg_params[i*4] = spectrum.peaks[peak].A_refined

            # Sharpen the peak
            L_window = original_peak.sigma_L
            R_window = original_peak.sigma_R       
            L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window/4)
            R_mask = (data_x >= x0_fit + R_window/4) & (data_x <= x0_fit + R_window)

            data_x_L = data_x[L_mask]
            data_y_L = data_y[L_mask]
            data_x_R = data_x[R_mask]
            data_y_R = data_y[R_mask]

            if len(data_x_L) == 0 or len(data_x_R) == 0 or len(data_y_L) == 0 or len(data_y_R) == 0:
                log(f"masking error for peak {peak} block 2, iteration {k}")
                log(f"Peak {peak}: x0 = {x0_fit:.3f}, sigma_L = {sigma_L_fit:.3f}, sigma_R = {sigma_R_fit:.3f}")
                log(f"L_mask: {L_mask}, R_mask: {R_mask}")
                log(f"data_x_L: {data_x_L}, data_y_L: {data_y_L}, data_x_R: {data_x_R}, data_y_R: {data_y_R}")
                log(f"data_x_peak: {data_x_peak}, data_y_peak: {data_y_peak}")
                log(f"data_x: {data_x}, data_y: {data_y}")
                log(f"Error while fitting")
                return False
            
            mbg_L =  multi_bi_gaussian(data_x_L , *mbg_params)
            mbg_R = multi_bi_gaussian(data_x_R, *mbg_params)

            error_l = np.mean((data_y_L - mbg_L))
            error_r = np.mean((data_y_R - mbg_R))

            if error_l > 0 and error_r < 0:
                x0_fit = x0_fit - 0.02

            elif error_l <0 and error_r > 0:
                x0_fit = x0_fit + 0.02
            else:
                sigma_L_fit = sigma_L_fit + error_l/1000
                sigma_R_fit = sigma_R_fit + error_r/1000
                      

            if sigma_L_fit > original_peak.width*4:
                sigma_L_fit = sigma_L_fit/2
            if sigma_R_fit > original_peak.width*4:
                sigma_R_fit = sigma_R_fit/2
            if sigma_L_fit < 1:
                sigma_L_fit = original_peak.width * 1.5
            if sigma_R_fit < 1:
                sigma_R_fit = original_peak.width * 1.5
            
            spectrum.peaks[peak].sigma_L = sigma_L_fit
            spectrum.peaks[peak].sigma_R = sigma_R_fit
            spectrum.peaks[peak].x0_refined = x0_fit

            mbg_params[i*4 + 2] = sigma_L_fit
            mbg_params[i*4 + 3] = sigma_R_fit
            mbg_params[i*4 + 1] = x0_fit
    
    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True

    return True

            
def update_peak_params(peak_list, popt, spectrum:MSData):
    i = 0
    for peak in peak_list:   
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]     
        spectrum.peaks[peak].A_refined = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].fitted = True
        i += 1
