
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData

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

def open_file_callback(sender, app_data, user_data):
    log(f"Path: {app_data['file_path_name']}")
    spectrum = user_data.spectrum
    spectrum.import_csv(app_data['file_path_name'])
    initialise_windows(user_data)


log_string = ""
def log(message:str) -> None:   
    global log_string
    log_string += message + "\n"
    dpg.set_value("message_box", log_string)
