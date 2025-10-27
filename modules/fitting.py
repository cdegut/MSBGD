from matplotlib import use
from modules.data_structures import MSData, peak_params
from typing import Tuple
import dearpygui.dearpygui as dpg
from modules.helpers import bi_gaussian
from scipy.integrate import quad
import numpy as np
from modules.matching import update_peak_starting_points
from modules.utils import log
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
    use_gaussian = dpg.get_value("use_gaussian")
    if dpg.get_value("show_residual_checkbox"):
        dpg.show_item("residual")
    use_filtered= dpg.get_value("use_filtered")
    user_data.stop_fitting = False
    rolling_window_fit(render_callback, k, std, use_filtered, use_gaussian)
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

def rolling_window_fit(render_callback, iterations = 1000, std =0.25, use_filtered = True, use_gaussian = False):
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
 
    fit = refine_peak_parameters(working_peak_list, initial_params, render_callback, iterations, std , use_filtered = use_filtered, use_gaussian=use_gaussian)   
    if fit:
        log("Fitting done with no error")
    else:
        log("Error while fitting")
        return
    
    draw_fitted_peaks(None, None, spectrum)
    update_peak_starting_points(user_data=render_callback)

def draw_residual(x_data, residual):
    dpg.show_item("residual")
    dpg.set_value("residual", [x_data, residual])


def calculate_fit_quality_metrics(data_x, data_y, spectrum, working_peak_list, rmse_only=False, weights=None):
    """Calculate comprehensive fit quality metrics"""
    residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)
    
    # 1. Weighted RMSE (higher weight for peak regions)
    if weights is None:
        weights = np.ones_like(data_y)
        for peak in working_peak_list:
            if spectrum.peaks[peak].fitted:
                x0 = spectrum.peaks[peak].x0_refined
                sigma_L = spectrum.peaks[peak].sigma_L
                sigma_R = spectrum.peaks[peak].sigma_R
                # Create Gaussian weight centered at peak
                peak_mask = (data_x >= x0 - 3*sigma_L) & (data_x <= x0 + 3*sigma_R)
                weights[peak_mask] *= 5.0  # 5x weight for peak regions
    
    weighted_rmse = np.sqrt(np.average(np.square(residual), weights=weights))
    
    # 2. R-squared (coefficient of determination)
    ss_res = np.sum(np.square(residual))
    ss_tot = np.sum(np.square(data_y - np.mean(data_y)))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if rmse_only:
        return {
            'rmse': np.sqrt(np.mean(np.square(residual))),
            'weighted_rmse': weighted_rmse,
            'r_squared': r_squared,
            'weights': weights
        }
    
    # 3. Chi-squared for noisy data
    degrees_of_freedom = len(data_y) - (len(working_peak_list) * 4)  
    # Method 1: Estimate noise from baseline regions
    noise_mask = np.ones_like(data_y, dtype=bool)
    for peak in working_peak_list:
        if spectrum.peaks[peak].fitted:
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            # Exclude peak regions from noise calculation
            peak_mask = (data_x >= x0 - 4*sigma_L) & (data_x <= x0 + 4*sigma_R)
            noise_mask &= ~peak_mask
    
    # Estimate noise variance from baseline regions
    if np.sum(noise_mask) > 10:  # Need enough baseline points
        baseline_residual = residual[noise_mask]
        noise_variance = np.var(baseline_residual)
    else:
        # Fallback: robust estimate from all residuals using median absolute deviation
        mad = np.median(np.abs(residual - np.median(residual)))
        noise_variance = (1.4826 * mad)**2  # Convert MAD to variance estimate
    
    # Ensure minimum noise level
    min_noise = np.var(data_y) * 0.001  # At least 0.1% of signal variance
    noise_variance = max(noise_variance, min_noise)
    
    if degrees_of_freedom > 0:
        chi_squared_reduced = ss_res / (degrees_of_freedom * noise_variance)
    else:
        chi_squared_reduced = np.inf
    
    # 4. Peak-specific metrics
    peak_quality = {}
    for peak in working_peak_list:
        x0 = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R
        
        # Peak region mask
        peak_mask = (data_x >= x0 - 3*sigma_L) & (data_x <= x0 + 3*sigma_R)
        if np.any(peak_mask):
            peak_residual = residual[peak_mask]
            peak_data = data_y[peak_mask]
            
            # Signal-to-noise ratio at peak
            peak_height = spectrum.peaks[peak].A_refined
            noise_level = np.std(peak_residual)
            snr = peak_height / noise_level if noise_level > 0 else np.inf
            
            # Peak RMSE
            peak_rmse = np.sqrt(np.mean(np.square(peak_residual)))
            
            peak_quality[peak] = {
                'snr': snr,
                'peak_rmse': peak_rmse,
                'relative_error': peak_rmse / peak_height if peak_height > 0 else np.inf
            }
    
    # 5. Akaike Information Criterion (AIC)
    n = len(data_y)
    k = len(working_peak_list) * 4  # number of parameters
    if n > 0 and ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * k
    else:
        aic = np.inf
    
    # 6. Bayesian Information Criterion (BIC)
    if n > 0 and ss_res > 0:
        bic = n * np.log(ss_res / n) + k * np.log(n)
    else:
        bic = np.inf
    
    return {
        'rmse': np.sqrt(np.mean(np.square(residual))),
        'weighted_rmse': weighted_rmse,
        'r_squared': r_squared,
        'chi_squared_reduced': chi_squared_reduced,
        'noise_variance': noise_variance,
        'noise_std': np.sqrt(noise_variance),
        'signal_to_noise': np.mean(np.abs(data_y)) / np.sqrt(noise_variance),
        'aic': aic,
        'bic': bic,
        'peak_quality': peak_quality,
        'residual_autocorr': np.corrcoef(residual[:-1], residual[1:])[0,1] if len(residual) > 1 else 0
    }


def refine_peak_parameters(working_peak_list, mbg_params, render_callback:RenderCallback, iterations=1000, std = 0.25, use_filtered = True, use_gaussian = False):
    spectrum = render_callback.spectrum
    original_peaks: Tuple[int, peak_params] = {peak:spectrum.peaks[peak] for peak in working_peak_list}
    data_x = spectrum.working_data[:,0]
    data_y = spectrum.baseline_corrected[:,1]
    
    # Store quality metrics history
    quality_history = []
    iterations_list = [i for i in range(len(working_peak_list))]
    start = time.time()
    
    if use_filtered:
        window_length = dpg.get_value("smoothing_window")
        data_x = savgol_filter(spectrum.baseline_corrected[:,0], window_length=window_length, polyorder=2)
        data_y = spectrum.baseline_corrected[:,1]
    else:
        data_x = spectrum.baseline_corrected[:,0]
        data_y = spectrum.baseline_corrected[:,1]
    
    ## Iteration loop start here
    ##############################
    for k in range(iterations + 1):
        iteration_start = time.time()
        render_callback.execute()
        if dpg.get_value("stop_fitting_checkbox") or render_callback.stop_fitting:
            log("Fitting stopped by user")
            break
        

        quality_metrics = calculate_fit_quality_metrics(data_x, data_y, spectrum, working_peak_list, rmse_only=True)
        quality_history.append(quality_metrics)
        
        residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)
        
        if dpg.get_value("show_residual_checkbox"):
            dpg.set_value("residual", [data_x.tolist(), residual.tolist()])
        
        # Use weighted RMSE for convergence check
        current_metric = quality_metrics['weighted_rmse']

        if k > 10:
            recent_metrics = [q['weighted_rmse'] for q in quality_history[-10:]]
            current_std = np.std(recent_metrics)
        else:
            current_std = np.inf
        r_squared = quality_metrics['r_squared']

        dpg.set_value("Fitting_indicator_text",f"Iter {k}: wRMSE={current_metric:.4f}, R²={r_squared:.4f}, Change in error={current_std:.4f}, iteration time: {time.time() - iteration_start:.2f}s")
        
        # Check convergence using multiple criteria     
        if k > 10:   
            converged = current_std < std and r_squared > 0.90            
            if converged:
                break

        # Shuffle iterations order for next pass
        iterations_list = np.random.permutation(iterations_list) #do not use the same order every iteration
        for i in iterations_list:
            peak = working_peak_list[i]
            original_peak = original_peaks[peak]
            _refine_iteration(peak, data_x, data_y , spectrum, original_peak, force_gaussian=use_gaussian)
    
    ##############################
    # End of iteration loop
    ##############################   
    quality_metrics = calculate_fit_quality_metrics(data_x, data_y, spectrum, working_peak_list)
    print (quality_metrics)
    chi_squared =  quality_metrics['chi_squared_reduced']

    signal_to_noise = quality_metrics['signal_to_noise']
    peaks_error = [quality_metrics["peak_quality"][peak]['relative_error'] for peak in working_peak_list if peak in quality_metrics["peak_quality"]]
    
    time_taken = time.time() - start
    log(f"Converged: R²={r_squared:.4f}, X²r={chi_squared:.3f}, SNR={signal_to_noise:.1f}, Time: {time_taken:.2f}s")
    dpg.set_value("Fitting_indicator_text",
        f"Converged after {k} iterations. wRMSE={current_metric:.4f}, R²={r_squared:.4f}, "
        f"X²r={chi_squared:.3f}, Median peak error={np.median(peaks_error):.4f}, Time: {time_taken:.2f}s")
    
    print(quality_metrics["peak_quality"])
    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True
        spectrum.peaks[peak].fit_quality = quality_metrics["peak_quality"].get(peak, {})

    
    return True

def _refine_iteration(peak:int, data_x, data_y, spectrum:MSData, original_peak, force_gaussian = False):     
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
                min_window = sampling_rate * 3
                L_window = max(sigma_L_fit * iteration, min_window)
                R_window = max(sigma_R_fit * iteration, min_window)
                L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window/2)
                R_mask = (data_x >= x0_fit + R_window/2) & (data_x <= x0_fit + R_window)

                data_x_L = data_x[L_mask]
                data_y_L = data_y[L_mask]
                data_x_R = data_x[R_mask]
                data_y_R = data_y[R_mask]

                # Check if we have enough data points
                if len(data_x_L) < 3 or len(data_x_R) < 3:
                    continue
                
                mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
                mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

                error_l = np.mean((data_y_L - mbg_L))
                error_r = np.mean((data_y_R - mbg_R))

                if np.isnan(error_l) or np.isnan(error_r) or np.isinf(error_l) or np.isinf(error_r):
                    continue


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

                    max_adjustment = original_peak.width * 0.01  # 1% of original width

                    sigma_L_adjustment = np.clip(error_l/val, -max_adjustment, max_adjustment)
                    sigma_R_adjustment = np.clip(error_r/val, -max_adjustment, max_adjustment)
                    
                    sigma_L_fit = sigma_L_fit + sigma_L_adjustment
                    sigma_R_fit = sigma_R_fit + sigma_R_adjustment

                # This is supposed to keep the widths in a reasonable range, 
                # but it cause high asymmetry peak to not fit correctly.
                # Removed for now, doesn't seem needed with the two pass approach.
                # if sigma_L_fit > original_peak.width*3:
                #     sigma_L_fit = sigma_L_fit/2
                # if sigma_R_fit > original_peak.width*3:
                #     sigma_R_fit = sigma_R_fit/2
                if sigma_L_fit <  sampling_rate:
                    sigma_L_fit = sampling_rate *3
                if sigma_R_fit <  sampling_rate:
                    sigma_R_fit = sampling_rate *3
                
                if np.isnan(sigma_L_fit):
                    sigma_L_fit = sampling_rate *3
                if np.isnan(sigma_R_fit):
                    sigma_R_fit = sampling_rate *3
                
                if force_gaussian:
                    sigma_L_fit = (sigma_L_fit + sigma_R_fit) / 2
                    sigma_R_fit = sigma_L_fit
                
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
    colors = sns.color_palette("plasma", 20)

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if x0_fit < spectrum.working_data[:,0][0] or x0_fit > spectrum.working_data[:,0][-1]:
            continue
        if not spectrum.peaks[peak].fitted:
            continue

        peak_error = spectrum.peaks[peak].fit_quality.get('relative_error', 1.0) * 3
        normalized_error = np.clip(peak_error, 0, 1)
        color_idx = int(normalized_error * (len(colors) - 1))      
        color = [int(c * 255) for c in colors[color_idx]]


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
        rel_error = spectrum.peaks[peak].fit_quality.get('relative_error', np.nan)
        
        with dpg.table_row(parent = "peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{start:.2f}")
            dpg.add_text(f"{apex:.2f}")
            dpg.add_text(f"{integral:.2f}")
            dpg.add_text(f"{sigma_L:.4f}")
            dpg.add_text(f"{sigma_R:.4f}")
            dpg.add_text(f"{rel_error:.4f}")
