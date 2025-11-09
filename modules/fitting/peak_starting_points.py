import numpy as np
from sklearn.linear_model import LinearRegression
from modules.data_structures import MSData
from modules.math import bi_gaussian, bi_Lorentzian


def update_peak_starting_points(spectrum: MSData):

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue

        for i in range(2):
            if i == 1:
                k = -1
            else:
                k = 1
            A = spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex
            sigma_L = spectrum.peaks[peak].sigma_L

            while True:
                if spectrum.peak_model == "lorentzian":
                    A_current = bi_Lorentzian(
                        start,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )
                else:
                    A_current = bi_gaussian(
                        start,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )
                start -= 0.02 * k
                if A_current <= 0.95 * A:
                    break

            mz80pcs = start

            while True:
                if spectrum.peak_model == "lorentzian":
                    A_current = bi_Lorentzian(
                        start,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )
                else:
                    A_current = bi_gaussian(
                        start,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )

                start -= 0.02 * k
                if A_current <= 0.30 * A:
                    break

            mz25pcs = start

            sample_points = np.linspace(mz80pcs, mz25pcs, 10)
            mz_samples = []
            A_samples = []

            for sample_mz in sample_points:
                if spectrum.peak_model == "lorentzian":
                    A_sample = bi_Lorentzian(
                        sample_mz,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )
                else:
                    A_sample = bi_gaussian(
                        sample_mz,
                        spectrum.peaks[peak].A_refined,
                        apex,
                        spectrum.peaks[peak].sigma_L,
                        spectrum.peaks[peak].sigma_R,
                    )
                mz_samples.append(sample_mz)
                A_samples.append(A_sample)

            X = np.array(mz_samples).reshape(-1, 1)
            y = np.array(A_samples)
            reg = LinearRegression().fit(X, y)

            a = float(reg.coef_[0])
            b = float(reg.intercept_)

            if i == 0:
                spectrum.peaks[peak].regression_fct = (a, b)
                spectrum.peaks[peak].start_end = (-b / a, 0.0)
            else:
                spectrum.peaks[peak].start_end = (
                    spectrum.peaks[peak].start_end[0],
                    -b / a,
                )
