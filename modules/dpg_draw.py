import numpy as np
from scipy.integrate import quad
from modules.data_structures import MSData
from modules.helpers import multi_bi_gaussian, bi_gaussian
import dearpygui.dearpygui as dpg

log_string = ""
def log(message:str) -> None:   
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)

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
        dpg.bind_item_theme(f"fitted_peak_{peak}", "fitted_peak_theme")
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
        start = apex - 3 * spectrum.peaks[peak].sigma_L
        end = apex + 3 * spectrum.peaks[peak].sigma_R
        integral = quad(bi_gaussian, start, end, args=(spectrum.peaks[peak].A_refined, spectrum.peaks[peak].x0_refined, spectrum.peaks[peak].sigma_L, spectrum.peaks[peak].sigma_R))[0]
        
        with dpg.table_row(parent = "peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{start:.2f}")
            dpg.add_text(f"{apex:.2f}")
            dpg.add_text(f"{integral:.2f}")


    
            
        
                    
                

        
