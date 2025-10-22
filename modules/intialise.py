
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData
from modules.matching import draw_mz_lines
import pandas as pd

def initialise_windows(render_callback):
    spectrum:MSData= render_callback.spectrum
    min_value = min(spectrum.original_data[:,0])
    max_value = max(spectrum.original_data[:,0])
    default_L = min(spectrum.working_data[:,0])
    default_R = max(spectrum.working_data[:,0])
    dpg.configure_item("L_data_clipping", default_value=default_L, min_value=min_value, max_value=max_value)
    dpg.configure_item("R_data_clipping", default_value=default_R, min_value=min_value, max_value=max_value)

    w_x = spectrum.working_data[:,0].tolist()
    dpg.set_value("original_series", [w_x, spectrum.working_data[:,1].tolist()])
    filtered = spectrum.get_filterd_data(900)
    dpg.set_value("filtered_series", [w_x, filtered])
    dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])

    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    dpg.set_value("corrected_series_plot3", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])

    for i in range(5):
        if spectrum.matching_data[i] != []:          
            dpg.configure_item(f"molecular_weight_{i}", default_value = spectrum.matching_data[i][0])
            dpg.configure_item(f"charges_{i}", default_value = spectrum.matching_data[i][1])
            dpg.configure_item(f"nb_peak_show_{i}", default_value = spectrum.matching_data[i][2])
            draw_mz_lines(sender=None,app_data=None, user_data=(render_callback, i))
            

def file_dialog(render_callback):   
    with dpg.file_dialog(directory_selector=False, show=False, user_data=render_callback ,callback=open_file_callback, id="file_dialog_id", width=700 ,height=400):
        dpg.add_file_extension(".csv", color=(110,79,46, 255), custom_text="[csv]")
        dpg.add_file_extension(".xlsx", color=(16,124,65, 255), custom_text="[xlsx]")
        dpg.add_file_extension(".xls", color=(16,124,65, 255), custom_text="[xls]")

def file_dialog_load_saved_data(render_callback):
    with dpg.file_dialog(directory_selector=False, show=False, user_data=render_callback ,callback=open_pkl_callback, id="file_dialog_id_saved_data", width=700 ,height=400):
        dpg.add_file_extension(".pkl", color=(54,92,45, 255), custom_text="[pkl]")

def file_dialog_save_data(render_callback):
    with dpg.file_dialog(directory_selector=False, show=False, user_data=render_callback ,callback=save_pkl_callback, id="file_dialog_id_save_data", width=700 ,height=400):
        dpg.add_file_extension(".pkl", color=(54,92,45, 255), custom_text="[pkl]")

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

def open_pkl_callback(sender, app_data, user_data):
    render_callback = user_data
    spectrum:MSData = user_data.spectrum
    print(spectrum.original_data)
    log(f"Path: {app_data['file_path_name']}")
    spectrum.load_from_file(app_data['file_path_name'])
    initialise_windows(render_callback)

def save_pkl_callback(sender, app_data, user_data):
    render_callback = user_data
    spectrum:MSData = user_data.spectrum
    log(f"Path: {app_data['file_path_name']}")
    spectrum.save_to_file(app_data['file_path_name'])

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