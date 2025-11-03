import dearpygui.dearpygui as dpg
from modules.fitting import draw_base_projection, draw_fitted_peaks, MBG_fit
from modules.helpers import bi_gaussian
from modules.rendercallback import RenderCallback
import numpy as np
from modules.data_structures import get_global_msdata_ref


def fitting_window(render_callback):
    spectrum = render_callback.spectrum
    with dpg.child_window(label="Peak fitting", tag="Peak fitting"):
        # Create a plot for the raw data
        with dpg.plot(
            label="Gaussian Fit", width=1430, height=600, tag="gaussian_fit_plot"
        ) as plot2:
            dpg.add_plot_axis(dpg.mvXAxis, label="m/z", tag="x_axis_plot2")
            dpg.add_plot_axis(dpg.mvYAxis, label="Y Axis", tag="y_axis_plot2")
            dpg.add_line_series(
                [],
                [],
                label="Corrected Data Series",
                parent="y_axis_plot2",
                tag="corrected_series_plot2",
            )
            dpg.add_line_series(
                [],
                [],
                label="Residual",
                parent="y_axis_plot2",
                tag="residual",
                show=False,
            )
            dpg.add_line_series(
                [], [], label="MBG", parent="y_axis_plot2", tag="MBG_plot2", show=False
            )

        with dpg.group(horizontal=True, horizontal_spacing=50):
            with dpg.group(horizontal=False):
                dpg.add_button(
                    label="Start Multi Bi Gaussian Deconvolution",
                    callback=run_fitting_callback,
                    user_data=render_callback,
                    tag="start_fitting_button",
                )
                dpg.add_checkbox(
                    label="<--- Tick to Stop Fitting",
                    user_data=render_callback,
                    tag="stop_fitting_checkbox",
                    show=False,
                )

                with dpg.group(horizontal=False):
                    dpg.add_checkbox(
                        label="Use Symmetric Gaussian",
                        default_value=False,
                        tag="use_gaussian",
                    )
                    dpg.add_checkbox(
                        label="Fit on filtered data",
                        default_value=False,
                        tag="use_filtered",
                    )
                    dpg.add_checkbox(
                        label="Reduce width variance",
                        default_value=True,
                        tag="use_reduced",
                    )
            dpg.add_loading_indicator(
                style=5, radius=3, show=False, tag="Fitting_indicator"
            )
            dpg.add_text("", tag="Fitting_indicator_text")
        with dpg.group(horizontal=True, horizontal_spacing=50):
            with dpg.group(horizontal=False):
                dpg.add_text("Fitting Iterations:")
                dpg.add_input_int(
                    label="",
                    default_value=1000,
                    min_value=50,
                    max_value=10000,
                    width=200,
                    tag="fitting_iterations",
                )
            with dpg.group(horizontal=False):
                dpg.add_text("OR residual variation bellow:")
                dpg.add_button(
                    label="Draw Initial Peaks",
                    callback=draw_initial_peaks_callback,
                    user_data=spectrum,
                )
                dpg.add_input_float(
                    label="",
                    default_value=0.25,
                    min_value=0.01,
                    max_value=1,
                    width=200,
                    tag="fitting_std",
                )
        dpg.add_checkbox(
            label="Show residual",
            default_value=True,
            tag="show_residual_checkbox",
            callback=show_residual_callback,
            user_data=spectrum,
        )
        dpg.add_checkbox(
            label="Show baseline projections",
            callback=draw_base_projection,
            tag="show_projection_checkbox",
            default_value=False,
        )
        dpg.add_button(label="Redraw Peaks", callback=draw_fitted_peaks_callback)

        with dpg.child_window(height=100, width=1200, tag="peak_table_window"):
            with dpg.table(header_row=True, tag="peak_table"):
                dpg.add_table_column(label="Peak Label")
                dpg.add_table_column(label="Peak Start")
                dpg.add_table_column(label="Peak Apex")
                dpg.add_table_column(label="Peak Integral")
                dpg.add_table_column(label="Sigma Left")
                dpg.add_table_column(label="Sigma Right")
                dpg.add_table_column(label="Relative Error")

    dpg.bind_item_theme("corrected_series_plot2", "data_theme")
    dpg.bind_item_theme("residual", "residual_theme")
    dpg.bind_item_theme("MBG_plot2", "fitting_MBG_theme")


def draw_fitted_peaks_callback():
    draw_fitted_peaks(False)


def show_residual_callback(sender, app_data):
    if app_data:
        dpg.show_item("residual")
    else:
        dpg.hide_item("residual")
    return


def stop_fitting(sender, app_data, user_data: RenderCallback):
    user_data.stop_fitting = True
    print("Fitting stopped")
    print(user_data.stop_fitting)
    return


def draw_initial_peaks_callback(sender, app_data, user_data: RenderCallback):
    spectrum = get_global_msdata_ref()
    for alias in dpg.get_aliases():
        if alias.startswith("initial_peak_"):
            dpg.delete_item(alias)

    i = 0
    for peak in spectrum.peaks:
        x0_init = spectrum.peaks[peak].x0_init
        if (
            x0_init < spectrum.working_data[:, 0][0]
            or x0_init > spectrum.working_data[:, 0][-1]
        ):
            continue

        A = spectrum.peaks[peak].A_init
        sigma_L = spectrum.peaks[peak].width / 2
        sigma_R = spectrum.peaks[peak].width / 2

        x_individual_fit = np.linspace(
            x0_init - 4 * sigma_L, x0_init + 4 * sigma_R, 500
        )
        y_individual_fit = bi_gaussian(x_individual_fit, A, x0_init, sigma_L, sigma_R)

        dpg.add_line_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak}",
            parent="y_axis_plot2",
            tag=f"initial_peak_{peak}",
        )
        dpg.bind_item_theme(f"initial_peak_{peak}", f"initial_peaks_theme_{peak}")
        dpg.add_plot_annotation(
            label=f"{peak}",
            default_value=(x0_init, A),
            offset=(-15, -15),
            color=[120, 120, 120],
            clamped=False,
            parent="gaussian_fit_plot",
            tag=f"initial_peak_annotation_{peak}",
        )
        i += 1


def run_fitting_callback(sender, app_data, user_data: RenderCallback):
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

    if dpg.does_alias_exist("MBG_plot2"):
        dpg.hide_item("MBG_plot2")

    if dpg.does_alias_exist("fitting_residual_plot1"):
        dpg.delete_item("fitting_residual_plot1")
        dpg.set_item_label("fitting_residual_plot1_button", "Show Fitting Residual")

    use_filtered = dpg.get_value("use_filtered")
    user_data.stop_fitting = False
    MBG_fit(render_callback, k, std, use_filtered, use_gaussian)
    dpg.hide_item("stop_fitting_checkbox")
    dpg.hide_item("Fitting_indicator")
    dpg.show_item("start_fitting_button")
