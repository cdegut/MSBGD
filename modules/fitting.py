from modules.data_structures import MSData, peak_params
from typing import Tuple
import dearpygui.dearpygui as dpg
from modules.helpers import bi_gaussian
from scipy.integrate import quad
import numpy as np
from modules.intialise import log
import threading
import time
from scipy.signal import savgol_filter

from modules.rendercallback import RenderCallback

def run_fitting(sender = None, app_data = None, user_data:RenderCallback = None):
    render_callback = user_data
    dpg.show_item("Fitting_indicator")
    k = dpg.get_value("fitting_iterations")
    std = dpg.get_value("fitting_std")
    if dpg.get_value("show_residual_checkbox"):
        dpg.show_item("residual")
    use_filtered= dpg.get_value("use_filtered")
    user_data.stop_fitting = False
    rolling_window_fit(render_callback, k, std, use_filtered)
    dpg.hide_item("Fitting_indicator")

def initial_peaks_parameters(spectrum:MSData, asymetry = 1.5):
    initial_params = []
    working_peak_list = []
    i = 0

    if spectrum.peaks is None:
        log("No peaks are detected. Please run peak detection first")
        return None
    
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
        sigma_L_guess =  spectrum.peaks[peak].width /2
        sigma_R_guess = sigma_L_guess * asymetry
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess])       
        working_peak_list.append(peak)
        i += 1
    
    if working_peak_list == []:
        log("No peaks are within the data range. Please adjust the peak detection parameters")
        return None

    return initial_params, working_peak_list

def rolling_window_fit(render_callback, iterations = 1000, std =0.25, use_filtered = True):
    spectrum = render_callback.spectrum
    baseline_window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(baseline_window)
    init = initial_peaks_parameters(spectrum)
    if init is None:
        return
    
    initial_params, working_peak_list = init
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
 
    fit = refine_peak_parameters(working_peak_list, initial_params, render_callback, iterations, std , use_filtered = use_filtered)   
    if fit:
        log("Fitting done with no error")
    else:
        log("Error while fitting")
        return
    
    draw_fitted_peaks(None, None, spectrum)

def draw_residual(x_data, residual):
    dpg.show_item("residual")
    dpg.set_value("residual", [x_data, residual])

def refine_peak_parameters(working_peak_list, mbg_params, render_callback:RenderCallback, iterations=1000, std = 0.25, use_filtered = True):
    spectrum = render_callback.spectrum
    original_peaks: Tuple[int, peak_params] = {peak:spectrum.peaks[peak] for peak in working_peak_list}
    data_x = spectrum.working_data[:,0]
    data_y = spectrum.baseline_corrected[:,1]
    rmse_list = []
    iterations_list = [i for i in range(len(working_peak_list))]
    start = time.time()
    
    if use_filtered:
        window_length = dpg.get_value("smoothing_window")
        data_x = savgol_filter(spectrum.baseline_corrected[:,0], window_length=window_length, polyorder=2)
        data_y = spectrum.baseline_corrected[:,1]
    else:
        data_x = spectrum.baseline_corrected[:,0]
        data_y = spectrum.baseline_corrected[:,1]
    

    for k in range(iterations +1):
        render_callback.execute()
        if render_callback.stop_fitting:
            log("Fitting stopped by user")
            break
        residual = data_y - spectrum.calculate_mbg(data_x,  fitting=True)
        
        if dpg.get_value("show_residual_checkbox"):
            dpg.set_value("residual", [data_x.tolist(), residual.tolist()])
        
        rmse = np.sqrt(np.mean(np.square(residual)))
        rmse_list.append(rmse)

        if k > 10:
            current_std = np.std(rmse_list[-10:])
            if current_std < std:
                time_taken = time.time() - start
                log(f"Residual is stable. Done. Time taken: {time_taken:.2f} seconds")
                dpg.set_value("Fitting_indicator_text",f"Residual is stable. Done after {k} iterations. Time taken: {time_taken:.2f} seconds")
                break

            dpg.set_value("Fitting_indicator_text",f"Iteration {k}; RMS-Residual: {rmse:.3f} Residual std: {current_std:.3f}")

        iterations_list = np.random.permutation(iterations_list) #do not use the same order every iteration
        use_multithreading = dpg.get_value("use_multithreading")
        if use_multithreading:
            thread_list = []
            for i in iterations_list:
                peak = working_peak_list[i]
                original_peak = original_peaks[peak]
                thread = threading.Thread(target=_refine_iteration, args=(peak, data_x, data_y, spectrum, original_peak))
                thread.start()
                thread_list.append(thread)
            
            for thread in thread_list:
                thread.join()
        else:
            for i in iterations_list:
                peak = working_peak_list[i]
                original_peak = original_peaks[peak]
                _refine_iteration(peak, data_x, data_y , spectrum, original_peak)
        
    if k == iterations + 1:
        dpg.set_value("Fitting_indicator_text", f"Fitting did not converge in {k} iterations")
    
    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True
    
    return True

def _refine_iteration(peak:int, data_x, data_y, spectrum:MSData, original_peak):
         
            x0_fit = spectrum.peaks[peak].x0_refined
            sigma_L_fit = spectrum.peaks[peak].sigma_L
            sigma_R_fit = spectrum.peaks[peak].sigma_R

            mask = (data_x >= x0_fit - sigma_R_fit/4) & (data_x <= x0_fit + sigma_L_fit/4)
            data_x_peak = data_x[mask]
            data_y_peak = data_y[mask]
            
            if len(data_x_peak) == 0 or len(data_y_peak) == 0:
                log(f"masking error for peak {peak} block 1")              
                return False

            peak_error = np.mean(data_y_peak  - spectrum.calculate_mbg(data_x_peak, fitting=True))
            spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (peak_error/10)

            # Sharpen the peak
            L_window = original_peak.sigma_L
            R_window = original_peak.sigma_R       
            L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window/2)
            R_mask = (data_x >= x0_fit + R_window/2) & (data_x <= x0_fit + R_window)

            data_x_L = data_x[L_mask]
            data_y_L = data_y[L_mask]
            data_x_R = data_x[R_mask]
            data_y_R = data_y[R_mask]

            if len(data_x_L) == 0 or len(data_x_R) == 0 or len(data_y_L) == 0 or len(data_y_R) == 0:
                log(f"masking error for peak {peak} block 2")
                log(f"Peak {peak}: x0 = {x0_fit:.3f}, sigma_L = {sigma_L_fit:.3f}, sigma_R = {sigma_R_fit:.3f}")
                log(f"L_mask: {L_mask}, R_mask: {R_mask}")
                log(f"data_x_L: {data_x_L}, data_y_L: {data_y_L}, data_x_R: {data_x_R}, data_y_R: {data_y_R}")
                log(f"data_x_peak: {data_x_peak}, data_y_peak: {data_y_peak}")
                log(f"data_x: {data_x}, data_y: {data_y}")
                log(f"Error while fitting")
                return False
            
            mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
            mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

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
            
def update_peak_params(peak_list, popt, spectrum:MSData):
    for peak in peak_list:   
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i*4:(i+1)*4]     
        spectrum.peaks[peak].A_refined = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].fitted = True
        i += 1

def draw_fitted_peaks(sender = None, app_data = None, user_data:MSData = None, delete = False):
    spectrum = user_data
    # Delete previous peaks
    
    for alias in dpg.get_aliases():
        if alias.startswith("fitted_peak_") or alias.startswith("peak_annotation_"):
            dpg.delete_item(alias)
    if delete:
        return    
    # Generate fitted curve    
    peak_list = []
    mbg_param = []

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if x0_fit < spectrum.working_data[:,0][0] or x0_fit > spectrum.working_data[:,0][-1]:
            continue
        if not spectrum.peaks[peak].fitted:
            continue

        A = spectrum.peaks[peak].A_refined
        sigma_L_fit = spectrum.peaks[peak].sigma_L
        sigma_R_fit = spectrum.peaks[peak].sigma_R             
        peak_list.append(peak)

        x_individual_fit = np.linspace(x0_fit - 4*sigma_L_fit, x0_fit + 4*sigma_R_fit, 500)
        y_individual_fit = bi_gaussian(x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit)
        mbg_param.extend([A, x0_fit, sigma_L_fit, sigma_R_fit])

        dpg.add_line_series(x_individual_fit, y_individual_fit, label=f"Peak {peak}", parent="y_axis_plot2", tag = f"fitted_peak_{peak}")
        dpg.bind_item_theme(f"fitted_peak_{peak}", "fitted_peaks_theme")
        dpg.add_plot_annotation(label=f"Peak {peak}", default_value=(x0_fit, A), offset=(-15, -15), color=[255, 255, 0, 255], clamped=False, parent="gaussian_fit_plot", tag=f"peak_annotation_{peak}")
        i+1
    
    x_fit = np.linspace(np.min(spectrum.working_data[:,0]), np.max(spectrum.working_data[:,0]), 500)
    y_fit = spectrum.calculate_mbg(x_fit)
    dpg.show_item("MBG_plot2")
    dpg.set_value("MBG_plot2", [x_fit.tolist(), y_fit.tolist()])
   
    update_peak_table(spectrum)


def update_peak_table(spectrum:MSData):
    for tag in dpg.get_item_children("peak_table")[1]:
        dpg.delete_item(tag)

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        apex = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R
        start = apex - 3 * sigma_L
        end = apex + 3 *  sigma_R
        integral = quad(bi_gaussian, start, end, args=(spectrum.peaks[peak].A_refined, spectrum.peaks[peak].x0_refined, spectrum.
        peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R))[0]
        spectrum.peaks[peak].integral = integral
        
        with dpg.table_row(parent = "peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{start:.2f}")
            dpg.add_text(f"{apex:.2f}")
            dpg.add_text(f"{integral:.2f}")
            dpg.add_text(f"{sigma_L}")
            dpg.add_text(f"{sigma_R}")
