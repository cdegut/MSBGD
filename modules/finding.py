import dearpygui.dearpygui as dpg
from modules.data_structures import MSData, peak_params
from typing import Tuple
from scipy.signal import find_peaks
import numpy as np
from modules.dpg_draw import log
from modules.rendercallback import RenderCallback

def add_peak(sender, app_data, user_data:RenderCallback):
    spectrum = user_data.spectrum

    i = 0
    for peak in spectrum.peaks:
        if spectrum.peaks[peak].user_added:
            i += 1
    new_peak_index = 500+i

    mid_spectra = (spectrum.working_data[:,0][0] + spectrum.working_data[:,0][-1]) / 2
    center_slice = spectrum.working_data[:,1][(spectrum.working_data[:,0] > mid_spectra - 0.5) & (spectrum.working_data[:,0] < mid_spectra + 0.5)]
    max_y = max(center_slice) if len(center_slice) > 0 else 0
    width = dpg.get_value("peak_detection_width") 
    new_peak = peak_params(A_init=max_y, x0_init=mid_spectra, width=width, user_added=True)

    spectrum.peaks[new_peak_index] = new_peak

    dpg.add_drag_line(label=f"{new_peak_index}", tag = f"drag_line_{new_peak_index}", parent="data_plot", color=[255, 0, 255, 255], default_value=mid_spectra, callback=drag_peak_callback, user_data=(user_data, new_peak_index))
    update_user_peaks_table(user_data)

def drag_peak_callback(sender, app_data, user_data:RenderCallback):
    spectrum = user_data[0].spectrum
    peak = user_data[1]
    x0_init = dpg.get_value(sender)
    center_slice = spectrum.working_data[:,1][(spectrum.working_data[:,0] > x0_init - 0.5) & (spectrum.working_data[:,0] < x0_init + 0.5)]
    max_y = max(center_slice) if len(center_slice) > 0 else 0
    spectrum.peaks[peak].A_init = max_y
    spectrum.peaks[peak].x0_init = x0_init
    update_user_peaks_table(user_data[0])

def update_user_peaks_table(user_data:RenderCallback):
    spectrum = user_data.spectrum
    dpg.delete_item("user_peak_table", children_only=True)
    
    dpg.add_table_column(label="#", parent="user_peak_table", width=50)
    dpg.add_table_column(label="x0", parent="user_peak_table")
    dpg.add_table_column(label="Width", parent="user_peak_table", width=100)
    dpg.add_table_column(label="Del", parent="user_peak_table")
    
    for peaks in spectrum.peaks:
        if spectrum.peaks[peaks].user_added:
            with dpg.table_row(parent="user_peak_table"):
                dpg.add_text(f"{peaks}")
                dpg.add_text(f"{spectrum.peaks[peaks].x0_init:.1f}")
                #dpg.add_text(f"{spectrum.peaks[peaks].width:.2f}")
                dpg.add_input_int(label="", default_value=int(spectrum.peaks[peaks].width), width=80, tag=f"width_{peaks}", callback=update_peak_width_callback, user_data=(user_data, peaks))
                dpg.add_button(label="Del", callback=delete_user_peak_callback, user_data=(user_data, peaks))

def update_peak_width_callback(sender, app_data, user_data:Tuple[RenderCallback, int]):
    spectrum = user_data[0].spectrum
    peak = user_data[1]
    pass

def delete_user_peak_callback(sender, app_data, user_data:Tuple[RenderCallback, int]):
    spectrum = user_data[0].spectrum
    peak = user_data[1]
    del spectrum.peaks[peak]
    update_user_peaks_table(user_data[0])
    dpg.delete_item(f"drag_line_{peak}")

def peaks_finder_callback(sender, app_data, user_data:RenderCallback):
    spectrum = user_data.spectrum
    threshold = dpg.get_value("peak_detection_threshold")
    width = dpg.get_value("peak_detection_width")
    distance = dpg.get_value("peak_detection_distance")
    filter_window = dpg.get_value("smoothing_window")
    baseline_window = dpg.get_value("baseline_window")
    sampling_rate = np.mean(np.diff(spectrum.working_data[:,0]))
    max_width = 4*width
    width = width / sampling_rate
    distance = distance / sampling_rate
    peaks_finder(spectrum, threshold, width, max_width, distance, filter_window, baseline_window)
    update_found_peaks_table(user_data)

def update_found_peaks_table(user_data:RenderCallback):
    spectrum = user_data.spectrum
    dpg.delete_item("found_peak_table", children_only=True)

    dpg.add_table_column(label="#", parent="found_peak_table")
    dpg.add_table_column(label="x0", parent="found_peak_table")
    dpg.add_table_column(label="Width", parent="found_peak_table")
    dpg.add_table_column(label="Use", parent="found_peak_table")
    
    for peaks in spectrum.peaks:
        with dpg.table_row(parent="found_peak_table"):
            dpg.add_text(f"{peaks}")
            dpg.add_text(f"{spectrum.peaks[peaks].x0_init:.1f}")
            dpg.add_text(f"{spectrum.peaks[peaks].width:.2f}")
            checked = not spectrum.peaks[peaks].do_not_fit
            dpg.add_checkbox(label="", default_value=checked, callback=tick_peak_callback, user_data=(user_data, peaks))

def tick_peak_callback(sender, app_data, user_data:Tuple[RenderCallback, int]):
    spectrum = user_data[0].spectrum
    peak = user_data[1]
    spectrum.peaks[peak].do_not_fit = not spectrum.peaks[peak].do_not_fit
    draw_found_peaks(spectrum)

def peaks_finder(spectrum:MSData, threshold:int, width:int, max_width:int, distance:int, filter_window:int, baseline_window:int):
    filtered = spectrum.get_filterd_data(filter_window)
    spectrum.correct_baseline(baseline_window)
    baseline = spectrum.baseline[:,1] 
    filtered_thresolded = np.where(np.abs(filtered - baseline) <= threshold, 0, (filtered - baseline))
    peaks, peaks_data = find_peaks(filtered_thresolded, width=width, distance=distance)
    
    # Delete previous peaks
    peak_to_delete = []
    for old_peak in spectrum.peaks:
        #Do not delete custom peaks
        if spectrum.peaks[old_peak].user_added:
            continue

        if spectrum.peaks[old_peak].x0_init > spectrum.working_data[:,0][0] and spectrum.peaks[old_peak].x0_init < spectrum.working_data[:,0][-1]:
            peak_to_delete.append(old_peak)
    for peak in peak_to_delete:
        del spectrum.peaks[peak]

    try:
        new_peak_index = max([key for key in spectrum.peaks.keys() if not spectrum.peaks[key].user_added]) + 1
    except ValueError:
        new_peak_index = 0

    # Itterate over the peaks and add them to the dictionary
    i = 0
    for peak in peaks:
        sample = spectrum.working_data[:,0][peak - 250:peak + 250]
        sampling_rate = np.mean(np.diff(sample))
        width = peaks_data["widths"][i] * sampling_rate
        if width > max_width:
            width = max_width

        new_peak = peak_params(A_init=spectrum.working_data[:,1][peak], x0_init=spectrum.working_data[:,0][peak], width=width)
        spectrum.peaks[new_peak_index] = new_peak
        new_peak_index += 1
        i += 1
    
    peaks_centers = spectrum.working_data[:,0][peaks]
    log(f"Detected {len(peaks)} peaks at x = {peaks_centers}")
    draw_found_peaks(spectrum)

def draw_found_peaks(spectrum:MSData):    
    for alias in dpg.get_aliases():
        if alias.startswith(f"found_peak_line") or alias.startswith(f"found_peak_annotation"):
            dpg.delete_item(alias)

    for peak in spectrum.peaks:
        if spectrum.peaks[peak].user_added:
            continue
        x0 = spectrum.peaks[peak].x0_init
        if spectrum.peaks[peak].do_not_fit:
            color = (64,64,90)
        else:
            color = (264,24,24)
        center_slice = spectrum.working_data[:,1][(spectrum.working_data[:,0] > x0 - 0.5) & (spectrum.working_data[:,0] < x0 + 0.5)]
        max_y = max(center_slice) if len(center_slice) > 0 else 0
        dpg.draw_line((x0, 0), (x0, max_y), parent="data_plot", tag=f"found_peak_line_{peak}", color=color, thickness=1.5)
        dpg.add_plot_annotation(label=f"{peak}", default_value=(x0, max_y), parent="data_plot", tag=f"found_peak_annotation_{peak}", color=color)

