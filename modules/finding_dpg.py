from dearpygui import dearpygui as dpg
from modules.data_structures import MSData
from modules.finding import peaks_finder_callback, add_peak


def finding_window(render_callback):
    spectrum:MSData = render_callback.spectrum
    with dpg.child_window(label="Data Filtering and peak finding", tag="Data Filtering"):
        with dpg.group(horizontal=True, horizontal_spacing= 50):
            dpg.add_text("Data Clipping:")
            dpg.add_slider_int(label="Data clipping left", width=400, default_value=0, min_value=0, max_value=12000, tag="L_data_clipping", callback=data_clipper, user_data=spectrum)
            dpg.add_slider_int(label="Data clipping right", width=400, default_value=12000, min_value=0, max_value=12000, tag="R_data_clipping", callback=data_clipper, user_data=spectrum)
    
        # Create a plot for the data
        with dpg.plot(label="Data Filtering", width=1430, height=600, tag="data_plot") as plot1:
            # Add x and y axes
            x_axis = dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag= "x_axis_plot1")
            y_axis = dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag = "y_axis_plot1")
                       
            dpg.add_line_series([], [], label="Original Data Series", parent=y_axis, tag="original_series")
            dpg.add_line_series([], [], label="Filtered Data Series", parent=y_axis, tag="filtered_series")
            dpg.add_line_series([], [], label="Snip Baseline", parent=y_axis, tag="baseline")
            dpg.add_line_series([], [], label="2nd Order Derivative", parent=y_axis, tag="derivative2nd")
            dpg.add_line_series([], [], label="Fitting Residual", parent=y_axis, tag="fitting_residual_plot1", show=False)  
        
        with dpg.group(horizontal=True, horizontal_spacing= 50):
            with dpg.child_window(height=230, width=300):     
                dpg.add_text("Data Filtering:")
                dpg.add_text("Smoothing window:")
                dpg.add_slider_int(label="", default_value=300, min_value=3, max_value=5000, width=250, callback=filter_data, user_data=spectrum, tag="smoothing_window")
                dpg.add_text("")
                dpg.add_text("Baseline estimation:")
                with dpg.group(horizontal=True, horizontal_spacing= 50):
                    dpg.add_checkbox(label="Correct baseline", callback=toggle_baseline, user_data=spectrum, default_value=False, tag="baseline_correction_checkbox")
                dpg.add_text("Baseline window:") 
                dpg.add_slider_int(label="", default_value=1000, min_value=10, max_value=2000, width=250, callback=spectrum.request_baseline_update, user_data=spectrum, tag="baseline_window")
                dpg.add_checkbox(label="Show Fit Residual", default_value=False, tag="fitting_residual_plot1_checkbox", callback=get_residual)
                dpg.add_button(label="Calculate 2nd order derivative", callback=sec_order_derivative, user_data=spectrum)
            with dpg.child_window(height=230, width=300):           
                dpg.add_text("Peak detection:")
                dpg.add_text("Peak detection threshold:")
                with dpg.group(horizontal=True, horizontal_spacing= 50):
                    dpg.add_slider_int(label="", width=200, default_value=100, min_value=1, max_value=1000, tag="peak_detection_threshold", callback=move_threshold_callback)
                    dpg.add_button(label="Show", callback=show_threshold_callback, tag="show_threshold_button")
                dpg.add_text("Peak detection width:")
                dpg.add_slider_int(label="", width=200, default_value=10, min_value=2, max_value=100, tag="peak_detection_width")
                dpg.add_text("Min Peak Distance:")
                dpg.add_slider_int(label="", width=200, default_value=10, min_value=1, max_value=100, tag="peak_detection_distance")
                dpg.add_checkbox(label="Use 2nd order derivative", default_value=True, tag="use_2nd_derivative_checkbox")
                dpg.add_button(label="Find Peaks", callback=peaks_finder_callback, user_data=render_callback)
            
            with dpg.child_window(height=230, width=300):
                dpg.add_text("Peaks:")
                dpg.add_table(header_row=True, tag="found_peak_table", resizable=False, policy=dpg.mvTable_SizingStretchProp)


            with dpg.child_window(height=230, width=300):
                dpg.add_text("User Peaks:")
                dpg.add_button(label="Add Peak", callback=add_peak, user_data=render_callback)
                dpg.add_table(header_row=True, tag="user_peak_table", policy=dpg.mvTable_SizingStretchProp)


    dpg.bind_item_theme("original_series", "data_theme")
    dpg.bind_item_theme("filtered_series", "filtered_data_theme")
    dpg.bind_item_theme("baseline", "baseline_theme")
    dpg.bind_item_theme("fitting_residual_plot1", "residual_theme")
    dpg.bind_item_theme("derivative2nd", "derivative2nd_theme")



def data_clipper(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    L_clip = dpg.get_value("L_data_clipping")
    R_clip = dpg.get_value("R_data_clipping")
    if L_clip >= R_clip:
        return
    if len(spectrum.original_data) <= 2:
        return
    
    spectrum.clip_data(L_clip, R_clip)
    spectrum.request_baseline_update()

    dpg.set_value("original_series", [spectrum.working_data[:,0].tolist(), spectrum.working_data[:,1].tolist()])
    dpg.set_value("baseline", [spectrum.baseline[:,0].tolist(), spectrum.baseline[:,1].tolist()])
    dpg.set_value("corrected_series_plot2", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    dpg.set_value("corrected_series_plot3", [spectrum.baseline_corrected[:,0].tolist(), spectrum.baseline_corrected[:,1].tolist()])
    filter_data(user_data=spectrum)
    dpg.fit_axis_data("y_axis_plot1")
    dpg.fit_axis_data("x_axis_plot1")
    dpg.fit_axis_data("y_axis_plot2")
    dpg.fit_axis_data("x_axis_plot2")
    dpg.fit_axis_data("y_axis_plot3")
    dpg.fit_axis_data("x_axis_plot3")

def filter_data(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    window_length = dpg.get_value("smoothing_window")
    filtered = spectrum.get_filterd_data(window_length)
    dpg.set_value("filtered_series", [spectrum.working_data[:,0].tolist(),  filtered])

def toggle_baseline(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    spectrum.baseline_toggle = not spectrum.baseline_toggle
    spectrum.request_baseline_update()

def get_residual(sender = None, app_data = None):
    if app_data:
        value = dpg.get_value("residual")
        dpg.set_value("fitting_residual_plot1", value)
        dpg.show_item("fitting_residual_plot1")
    else:
        dpg.hide_item("fitting_residual_plot1")

def sec_order_derivative(sender = None, app_data = None, user_data:MSData = None):
    spectrum = user_data
    window_length = dpg.get_value("smoothing_window")
    derivative2nd = spectrum.get_2nd_derivative(window_length)
    dpg.set_value("derivative2nd", [spectrum.working_data[:,0].tolist(),  derivative2nd])

def show_threshold_callback():
    threshold = dpg.get_value("peak_detection_threshold")

    if dpg.does_item_exist("threshold_line_plot1"):
        dpg.delete_item("threshold_line_plot1")
        dpg.set_item_label("show_threshold_button", "Show")
        return
        
    if not dpg.does_item_exist("threshold_line_theme"):
        with dpg.theme(tag="threshold_line_theme"):       
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(dpg.mvPlotStyleVar_LineWeight, 2, category=dpg.mvThemeCat_Plots)
                dpg.add_theme_color(dpg.mvPlotCol_Line, (255,116,108), category=dpg.mvThemeCat_Plots)
       
    dpg.add_line_series([dpg.get_axis_limits("x_axis_plot1")[0], dpg.get_axis_limits("x_axis_plot1")[1]], [threshold, threshold], label="Threshold", parent="y_axis_plot1", tag="threshold_line_plot1")
    dpg.bind_item_theme("threshold_line_plot1", "threshold_line_theme")
    dpg.set_item_label("show_threshold_button", "Hide")

def move_threshold_callback():
    threshold = float(dpg.get_value("peak_detection_threshold"))
    for alias in dpg.get_aliases():
        if alias.startswith("threshold_line_"):
            dpg.set_value("threshold_line_plot1",[[dpg.get_axis_limits("x_axis_plot1")[0], dpg.get_axis_limits("x_axis_plot1")[1]], [threshold, threshold]])
            return
