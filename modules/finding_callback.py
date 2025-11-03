import math
import dearpygui.dearpygui as dpg


def get_smoothing_window():
    return 10 ** dpg.get_value("smoothing_window")


def set_smoothing_window(value):
    dpg.set_value("smoothing_window", math.log10(value))
