from turtle import width
from einops import reduce
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
import seaborn as sns

def run_fitting(sender = None, app_data = None, user_data:RenderCallback = None):
    render_callback = user_data
    dpg.show_item("Fitting_indicator")
    k = dpg.get_value("fitting_iterations")
    std = dpg.get_value("fitting_std")
    dpg.set_value("stop_fitting_checkbox", False)
    dpg.hide_item("start_fitting_button")
    dpg.show_item("stop_fitting_checkbox")
    if dpg.get_value("show_residual_checkbox"):
        dpg.show_item("residual")
    use_filtered= dpg.get_value("use_filtered")
    user_data.stop_fitting = False
    rolling_window_fit(render_callback, k, std, use_filtered)
    dpg.hide_item("stop_fitting_checkbox")
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")

def initial_peaks_parameters(spectrum:MSData, asymmetry = 1.8):
    initial_params = []
    working_peak_list = []
    i = 0



    reduce_width_var = dpg.get_value("use_reduced")

    if spectrum.peaks is None:
        log("No peaks are detected. Please run peak detection first")
        return None

    width_l = [spectrum.peaks[peak].width for peak in spectrum.peaks]
    std_width = np.std(width_l)
    med_width = np.median(width_l)

    for peak in spectrum.peaks:
    
        x0_guess = spectrum.peaks[peak].x0_init

        if spectrum.peaks[peak].do_not_fit:
            spectrum.peaks[peak].fitted = False
            continue

        if x0_guess < spectrum.working_data[:,0][0] or x0_guess > spectrum.working_data[:,0][-1]:
            log(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        width_init = spectrum.peaks[peak].width
        if reduce_width_var:
            width_init = med_width *0.6 + 0.4*width_init
        

        # Select working data within x0_guess ± width_init
        mask = (spectrum.working_data[:,0] >= x0_guess - width_init*2) & (spectrum.working_data[:,0] <= x0_guess + width_init*2)
        working_data_peak = spectrum.working_data[mask]
        if len(working_data_peak) > 1:
            sampling_rate = np.mean(np.diff(working_data_peak[:,0]))
        else:
            sampling_rate = np.mean(np.diff(spectrum.working_data[:,0]))

        A_guess = spectrum.peaks[peak].A_init
        sigma_L_guess =  width_init / 2
        sigma_R_guess = sigma_L_guess * asymmetry
        initial_params.extend([A_guess, x0_guess, sigma_L_guess, sigma_R_guess, sampling_rate])       
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
        A_guess, x0_guess, sigma_L_guess, sigma_R_guess, sampling_rate =  initial_params[i*5:(i+1)*5]
        print(f"Peak {peak}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}, sampling_rate = {sampling_rate:.3f}")

        spectrum.peaks[peak].A_refined = A_guess
        spectrum.peaks[peak].x0_refined = x0_guess
        spectrum.peaks[peak].sigma_L = sigma_L_guess
        spectrum.peaks[peak].sigma_R = sigma_R_guess
        spectrum.peaks[peak].fitted = False
        spectrum.peaks[peak].sampling_rate = sampling_rate
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
        if dpg.get_value("stop_fitting_checkbox") or render_callback.stop_fitting:
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
            sampling_rate = spectrum.peaks[peak].sampling_rate
            
            R_val = sigma_R_fit/4 if sigma_R_fit > sampling_rate*20 else sampling_rate*5
            L_val = sigma_L_fit/4 if sigma_L_fit > sampling_rate*20 else sampling_rate*5
            mask = (data_x >= x0_fit - R_val) & (data_x <= x0_fit + L_val)
            data_x_peak = data_x[mask]
            data_y_peak = data_y[mask]
            
            # Adjust the amplitude
            if len(data_x_peak) > 0 and len(data_y_peak) > 0:
                peak_error = np.mean(data_y_peak  - spectrum.calculate_mbg(data_x_peak, fitting=True))
                spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (peak_error/50)   

            # Sharpen the peak
            for iteration in range(1, 3):
                L_window = original_peak.sigma_L * iteration
                R_window = original_peak.sigma_R * iteration
                L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window/2)
                R_mask = (data_x >= x0_fit + R_window/2) & (data_x <= x0_fit + R_window)

                data_x_L = data_x[L_mask]
                data_y_L = data_y[L_mask]
                data_x_R = data_x[R_mask]
                data_y_R = data_y[R_mask]
                
                mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
                mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

                error_l = np.mean((data_y_L - mbg_L))
                error_r = np.mean((data_y_R - mbg_R))


                moved = False
                if (iteration ==1):
                    if error_l > 0 and error_r < 0:
                        x0_fit = x0_fit - 0.02
                        moved = True
                    elif error_l <0 and error_r > 0:
                        x0_fit = x0_fit + 0.02
                        moved = True

                if not moved:
                    val = 1000 if iteration ==1 else 2000 
                    sigma_L_fit = sigma_L_fit + error_l/val
                    sigma_R_fit = sigma_R_fit + error_r/val

                # This is supposed to keep the widths in a reasonable range, 
                # but it cause high asymmetry peak to not fit correctly.
                # Removed for now, doesn't seem needed with the two pass approach.
                # if sigma_L_fit > original_peak.width*3:
                #     sigma_L_fit = sigma_L_fit/2
                # if sigma_R_fit > original_peak.width*3:
                #     sigma_R_fit = sigma_R_fit/2
                if sigma_L_fit <  sampling_rate:
                    sigma_L_fit = sampling_rate *4
                if sigma_R_fit <  sampling_rate:
                    sigma_R_fit = sampling_rate *4
                
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
        if alias.startswith("fitted_peak_") or alias.startswith("peak_annotation_") or alias.startswith("fitted_peaks_theme_"):
            dpg.delete_item(alias)
    if delete:
        return    
    # Generate fitted curve    
    peak_list = []
    mbg_param = []
    colors = sns.color_palette("viridis", len(spectrum.peaks))

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if x0_fit < spectrum.working_data[:,0][0] or x0_fit > spectrum.working_data[:,0][-1]:
            continue
        if not spectrum.peaks[peak].fitted:
            continue
       
        color = int(colors[i][0]*255), int(colors[i][1]*255), int(colors[i][2]*255)

        with dpg.theme(tag = f"fitted_peaks_theme_{peak}"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots)

        A = spectrum.peaks[peak].A_refined
        sigma_L_fit = spectrum.peaks[peak].sigma_L
        sigma_R_fit = spectrum.peaks[peak].sigma_R             
        peak_list.append(peak)

        x_individual_fit = np.linspace(x0_fit - 4*sigma_L_fit, x0_fit + 4*sigma_R_fit, 500)
        y_individual_fit = bi_gaussian(x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit)
        mbg_param.extend([A, x0_fit, sigma_L_fit, sigma_R_fit])

        dpg.add_line_series(x_individual_fit, y_individual_fit, label=f"Peak {peak}", parent="y_axis_plot2", tag = f"fitted_peak_{peak}")
        dpg.bind_item_theme(f"fitted_peak_{peak}", f"fitted_peaks_theme_{peak}")
        dpg.add_plot_annotation(label=f"Peak {peak}", default_value=(x0_fit, A), offset=(-15, -15), color=[120,120,120], clamped=False, parent="gaussian_fit_plot", tag=f"peak_annotation_{peak}")
        i+=1

    x_fit = np.linspace(np.min(spectrum.working_data[:,0]), np.max(spectrum.working_data[:,0]), spectrum.working_data.shape[0] // 2)
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
