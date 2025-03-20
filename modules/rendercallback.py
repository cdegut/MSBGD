from dataclasses import dataclass
from typing import Dict
from modules.data_structures import MSData
import time as time
import dearpygui.dearpygui as dpg


@dataclass
class RenderCallback:
    spectrum:MSData
    last_baseline_corrected = time.time()
    baseline_correct_requested = False
    mz_lines = {}

    def execute(self):
        now = time.time()

        if self.spectrum.baseline_need_update == True: 
            if  now - self.last_baseline_corrected > 0.5:
                self.correct_baseline()
    
    def request_baseline_correction(self):
        self.baseline_correct_requested = True
    
    def correct_baseline(self):
        window = dpg.get_value("baseline_window")
        self.spectrum.correct_baseline(window)
        dpg.set_value("baseline", [self.spectrum.baseline[:,0].tolist(), self.spectrum.baseline[:,1].tolist()])
        dpg.set_value("corrected_series_plot2", [self.spectrum.baseline_corrected[:,0].tolist(), self.spectrum.baseline_corrected[:,1].tolist()])
        dpg.set_axis_limits("y_axis_plot2", min(self.spectrum.baseline_corrected[:,1]) - 1, max(self.spectrum.baseline_corrected[:,1]))
