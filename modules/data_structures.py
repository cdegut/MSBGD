from pybaselines import Baseline
from scipy.signal import savgol_filter, medfilt
import pandas as pd
import numpy as np
from dataclasses import dataclass
from modules.helpers import multi_bi_gaussian
from typing import Dict, Tuple
from dataclasses import field
import pickle

# Define a class to store mass spectrometry data
class MSData():
    def __init__(self):
        self.original_data = [[],[]]
        self.working_data = [[],[]]
        self.baseline = [[],[]]
        self.baseline_corrected = [[],[]]
        self.peaks: Dict[int : peak_params] = {}
        self.baseline_toggle = False
        self.baseline_need_update = False
        self.matching_data = [[],[],[],[],[]]
                  
    def import_csv(self, path:str):
        return True
    
    def initialise_dataframe(self, data:np.ndarray):
        data = data.dropna()
        self.original_data = data.to_numpy()
        self.working_data = self.original_data
        self.correct_baseline(0)
    
    def clip_data(self, L_clip:int, R_clip:int):
        self.working_data = self.original_data[(self.original_data[:,0] > L_clip) & (self.original_data[:,0] < R_clip)]

    def get_filterd_data(self, window_length, polyorder=2):
        if window_length % 2 == 0:
            window_length += 1  # Ensure window_length is odd
        filtered =  savgol_filter(self.working_data[:,1], window_length=window_length, polyorder=polyorder)
        # filtered = remove_spikes_median(filtered, 31)
        return filtered.tolist()

    def correct_baseline(self, window):
        if not self.baseline_toggle:
            self.baseline_corrected = self.working_data
            self.baseline = np.column_stack((self.working_data[:,0], [0]*len(self.working_data)))
            self.baseline_need_update = False
            return
          
        baseline_fitter = Baseline(x_data=self.working_data[:,0])
        bkg_4, params_4 = baseline_fitter.snip(self.working_data[:,1], max_half_window=window, decreasing=True, smooth_half_window=3)    
        self.baseline = np.column_stack((self.working_data[:,0], bkg_4 ))
        self.baseline_corrected = np.column_stack((self.working_data[:,0], self.working_data[:,1] - bkg_4))
        self.baseline_need_update = False        
    
    def guess_sampling_rate(self):
        sampling_rate = np.mean(np.diff(self.working_data[:,0]))
        return sampling_rate
    
    def request_baseline_update(self) -> None:
        self.baseline_need_update = True
    
    def calculate_mbg(self, data_x: np.ndarray, fitting = False) -> np.ndarray:
        mbg_params = []

        for peak in self.peaks:
            if self.peaks[peak].do_not_fit:
                continue
            
            if fitting or self.peaks[peak].fitted:
                mbg_params.append(self.peaks[peak].A_refined)
                mbg_params.append(self.peaks[peak].x0_refined)
                mbg_params.append(self.peaks[peak].sigma_L)
                mbg_params.append(self.peaks[peak].sigma_R )
        
        mbg = multi_bi_gaussian(data_x, *mbg_params)
        return mbg
 
    def save_to_file(self, path: str):
        """Save the entire MSData object to a file using pickle."""
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    #@staticmethod
    #def load_from_file(path: str) -> "MSData":
    #    """Load an MSData object from a file using pickle."""
    #    with open(path, 'rb') as f:
    #        data: MSData = pickle.load(f)
    #    return data

    def load_from_file(self, path: str):
        with open(path, 'rb') as f:
            new_data: MSData = pickle.load(f)
        self.original_data = new_data.original_data
        self.working_data = new_data.working_data
        self.baseline = new_data.baseline
        self.baseline_corrected = new_data.baseline_corrected
        self.peaks = new_data.peaks
        self.baseline_toggle = new_data.baseline_toggle
        self.baseline_need_update = new_data.baseline_need_update
        self.matching_data = new_data.matching_data
        

@dataclass
class peak_params:
    A_init: float
    x0_init: float
    width: np.ndarray
    A_refined: float = 0
    x0_refined: float = 0
    sigma_L: float = 0
    sigma_R: float =0
    fitted: bool = False
    integral: float = 0
    start_range: Tuple[int, int] = (0, 0)
    regression_fct : Tuple[float, float] = (1.0, 0.0)
    do_not_fit: bool = False
    user_added: bool = False
    matched_with: list = field(default_factory=lambda: [0,0,0])

if __name__ == "__main__":
    ms = MSData()
    ms.import_csv(rf"D:\MassSpec\Um_2-1_1x.csv")
    #ms.clip_data(10000, 12000)
    ms.baseline_toggle = True
    ms.correct_baseline(40)
    ms.guess_sampling_rate()
    print(ms.working_data[:,0])

def remove_spikes_median(y, kernel_len=11):
    """
    Median filter. kernel_len must be odd.
    - kernel_len ~ a bit larger than spike width in samples.
    """
    if kernel_len % 2 == 0:
        kernel_len += 1
    return medfilt(y, kernel_len)