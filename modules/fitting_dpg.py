import dearpygui.dearpygui as dpg
from modules.fitting import run_fitting
from modules.fitting import draw_fitted_peaks
from modules.rendercallback import RenderCallback

def fitting_window(render_callback):
    spectrum = render_callback.spectrum
    with dpg.child_window(label="Peak fitting", tag="Peak fitting"):
        # Create a plot for the raw data
        with dpg.plot(label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot") as plot2:
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")        
            dpg.add_line_series([], [], label="Corrected Data Series", parent="y_axis_plot2", tag="corrected_series_plot2")
            dpg.add_line_series([], [], label="Residual", parent="y_axis_plot2", tag="residual", show=False)
            dpg.add_line_series([], [], label="MBG", parent="y_axis_plot2", tag="MBG_plot2", show=False)

        with dpg.group(horizontal=True, horizontal_spacing= 50):
            with dpg.group(horizontal=False):     
                dpg.add_button(label="Start Multi Bi Gaussian Deconvolution",callback=run_fitting, user_data=render_callback, tag="start_fitting_button")
                dpg.add_checkbox(label = "<--- Tick to Stop Fitting", user_data=render_callback, tag="stop_fitting_checkbox", show=False)

                with dpg.group(horizontal=False):
                    dpg.add_checkbox(label="Use multithreading", default_value=True, tag="use_multithreading")
                    dpg.add_checkbox(label="Fit on filtered data", default_value=True, tag="use_filtered")
                    dpg.add_checkbox(label="Reduce width variance", default_value=True, tag="use_reduced")
            dpg.add_loading_indicator(style=5,radius=3, show=False, tag="Fitting_indicator")
            dpg.add_text("", tag="Fitting_indicator_text")
        with dpg.group(horizontal=True, horizontal_spacing= 50):
            with dpg.group(horizontal=False):    
                dpg.add_text("Fitting Iterations:")
                dpg.add_input_int(label="", default_value=1000, min_value=50, max_value=10000, width=200, tag="fitting_iterations")
            with dpg.group(horizontal=False):    
                dpg.add_text("OR residual variation bellow:")
                dpg.add_input_float(label="", default_value=0.25, min_value=0.01, max_value=1, width=200, tag="fitting_std")
        dpg.add_checkbox(label="Show residual", default_value=True, tag="show_residual_checkbox", callback=show_residual_callback, user_data=spectrum)
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks, user_data=spectrum)

        with dpg.child_window(height=100, width=900, tag="peak_table_window"):
            with dpg.table(header_row=True, tag="peak_table"):
                dpg.add_table_column(label="Peak Label")
                dpg.add_table_column(label="Peak Start")
                dpg.add_table_column(label="Peak Apex")
                dpg.add_table_column(label="Peak Integral")
                dpg.add_table_column(label="Sigma Left")
                dpg.add_table_column(label="Sigma Right")

    dpg.bind_item_theme("corrected_series_plot2", "data_theme")
    dpg.bind_item_theme("residual", "residual_theme")
    dpg.bind_item_theme("MBG_plot2", "fitting_MBG_theme")

def show_residual_callback(sender, app_data):
    if app_data:
        dpg.show_item("residual")
    else:
        dpg.hide_item("residual")
    return 

def stop_fitting(sender, app_data, user_data:RenderCallback):
    user_data.stop_fitting = True
    print("Fitting stopped")
    print(user_data.stop_fitting)
    return
