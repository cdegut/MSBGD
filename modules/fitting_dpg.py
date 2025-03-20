import dearpygui.dearpygui as dpg
from modules.fitting import run_fitting
from modules.dpg_draw import draw_fitted_peaks

def fitting_window(render_callback):
    spectrum = render_callback.spectrum
    with dpg.child_window(label="Peak fitting", tag="Peak fitting"):
        # Create a plot for the raw data
        with dpg.plot(label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot") as plot2:
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")        
            dpg.add_line_series(spectrum.baseline_corrected[:,0], spectrum.baseline_corrected[:,1], label="Corrected Data Series", parent="y_axis_plot2", tag="corrected_series_plot2")
            dpg.add_line_series([0], [0], label="Residual", parent="y_axis_plot2", tag="residual", show=False)
            dpg.add_line_series([], [], label="MBG", parent="y_axis_plot2", tag="MBG_plot2", show=False)

        with dpg.group(horizontal=True, horizontal_spacing= 50):            
            dpg.add_button(label="Windowed Multi Bi Gaussian Deconvolution",callback=run_fitting, user_data=spectrum)
            dpg.add_loading_indicator(style=5,radius=3, show=False, tag="Fitting_indicator")
            dpg.add_text("", tag="Fitting_indicator_text")
            
        dpg.add_text("Fitting Iterations:")
        dpg.add_input_int(label="", default_value=1000, min_value=50, max_value=10000, tag="fitting_iterations")
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks, user_data=spectrum)

        with dpg.child_window(height=100, width=900, tag="peak_table_window"):
            with dpg.table(header_row=True, tag="peak_table"):
                dpg.add_table_column(label="Peak Label")
                dpg.add_table_column(label="Peak Start")
                dpg.add_table_column(label="Peak Apex")
                dpg.add_table_column(label="Peak Integral")

    dpg.bind_item_theme("corrected_series_plot2", "data_theme")
    dpg.bind_item_theme("residual", "residual_theme")
    dpg.bind_item_theme("MBG_plot2", "fitting_MBG_theme")