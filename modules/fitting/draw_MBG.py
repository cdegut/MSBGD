import numpy as np
import dearpygui.dearpygui as dpg
from modules.data_structures import MSData


def show_MBG(spectrum: MSData, fitting=False):
    x_fit = np.linspace(
        np.min(spectrum.working_data[:, 0]),
        np.max(spectrum.working_data[:, 0]),
        spectrum.working_data.shape[0] // 2,
    )
    y_fit = spectrum.calculate_mbg(x_fit, fitting=fitting)
    dpg.show_item("MBG_plot2")
    dpg.set_value("MBG_plot2", [x_fit.tolist(), y_fit.tolist()])
