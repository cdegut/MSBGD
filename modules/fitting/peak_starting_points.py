import numpy as np
from sklearn.linear_model import LinearRegression
from modules.math import bi_gaussian, bi_Lorentzian


def update_peak_starting_points(spectrum):

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue

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
            start -= 0.02
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

            start -= 0.02
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

        spectrum.peaks[peak].regression_fct = (a, b)
