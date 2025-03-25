
from json import load
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData
import pandas as pd

def initialise_windows(render_callback):
    spectrum:MSData= render_callback.spectrum
    min_value = min(spectrum.original_data[:,0])
    max_value = max(spectrum.original_data[:,0])
    dpg.configure_item("L_data_clipping", default_value=min_value , min_value=min_value, max_value=max_value)
    dpg.configure_item("R_data_clipping", default_value =max_value, min_value=min_value, max_value=max_value)

    w_x = spectrum.working_data[:,0].tolist()
    dpg.set_value("original_series", [w_x, spectrum.working_data[:,1].tolist()])
    filtered = spectrum.get_filterd_data(50)
    dpg.set_value("filtered_series", [w_x, filtered])
    dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])

    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    dpg.set_value("corrected_series_plot3", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])

def file_dialog(render_callback):   
    with dpg.file_dialog(directory_selector=False, show=False, user_data=render_callback ,callback=open_file_callback, id="file_dialog_id", width=700 ,height=400):
        dpg.add_file_extension(".csv", color=(0, 255, 0, 255), custom_text="[csv]")
        dpg.add_file_extension(".xlsx", color=(0, 255, 0, 255), custom_text="[xlsx]")
        dpg.add_file_extension(".xls", color=(0, 255, 0, 255), custom_text="[xls]")

def open_file_callback(sender, app_data, user_data):
    render_callback = user_data
    spectrum:MSData = user_data.spectrum
    log(f"Path: {app_data['file_path_name']}")
    extension = app_data['file_name'].split('.')[-1]
    
    if extension == 'csv':
        dpg.show_item("file_loading_indicator")
        data = pd.read_csv(app_data['file_path_name'])
        finalise_loading(data, render_callback)

    elif extension == 'xlsx' or extension == 'xls':
        load_excel( app_data['file_path_name'],render_callback)

def load_excel(file_path, render_callback):
    spectrum:MSData = render_callback.spectrum
    xls = pd.ExcelFile(file_path)

    if len(xls.sheet_names) == 1:
        data = pd.read_excel(file_path)
        dpg.show_item("file_loading_indicator")
        finalise_loading(data, render_callback)
    else:
        sheet_names = xls.sheet_names
        show_sheet_selector(file_path, sheet_names, render_callback)

def show_sheet_selector(file_path, sheet_names, render_callback):     
    with dpg.window(label="Excel Sheet Selector", tag="sheet_selector_popup", width=300, height=200):
        dpg.add_text("Select a sheet:")
        dpg.add_combo(sheet_names, tag="sheet_selector")
        dpg.add_button(label="Load Sheet", callback=lambda: load_sheet(file_path, render_callback))

def load_sheet(file_path, render_callback):
    spectrum:MSData = render_callback.spectrum
    selected_sheet = dpg.get_value("sheet_selector")
    dpg.delete_item("sheet_selector_popup")
    dpg.show_item("file_loading_indicator")
    data = pd.read_excel(file_path, sheet_name=selected_sheet)
    finalise_loading(data, render_callback)

log_string = ""
def log(message:str) -> None:   
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)

def finalise_loading(df:pd.DataFrame, render_callback):
    spectrum:MSData = render_callback.spectrum
    spectrum.initialise_dataframe(df)
    initialise_windows(render_callback)
    dpg.hide_item("file_loading_indicator")