from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from modules.data_structures import (
    FitQualityPeakMetrics,
    MSData,
    peak_params,
    get_global_msdata_ref,
)
from typing import Literal, Tuple, overload
import dearpygui.dearpygui as dpg
from modules.finding_callback import get_smoothing_window
from modules.helpers import bi_gaussian, bi_gaussian_integral
from scipy.integrate import quad
import numpy as np
from modules.matching import redraw_blocks
from modules.utils import log
import time
from modules.rendercallback import RenderCallback
import seaborn as sns
from copy import deepcopy


def initial_peaks_parameters(spectrum: MSData, asymmetry=1.8):
    initial_params = []
    working_peak_list = []
    i = 0

    reduce_width_var = dpg.get_value("use_reduced")

    if spectrum.peaks is None:
        log("No peaks are detected. Please run peak detection first")
        return None

    width_l = [spectrum.peaks[peak].width for peak in spectrum.peaks]
    std_width = np.std(width_l)
    med_width = np.median(width_l)

    for peak in spectrum.peaks:

        x0_guess = spectrum.peaks[peak].x0_init

        if spectrum.peaks[peak].do_not_fit:
            spectrum.peaks[peak].fitted = False
            continue

        if (
            x0_guess < spectrum.working_data[:, 0][0]
            or x0_guess > spectrum.working_data[:, 0][-1]
        ):
            log(f"Peak {i} is out of bounds. Skipping")
            i += 1
            continue

        width_init = spectrum.peaks[peak].width
        if reduce_width_var:
            width_init = med_width * 0.8 + 0.2 * width_init

        # Select working data within x0_guess ± width_init
        mask = (spectrum.working_data[:, 0] >= x0_guess - width_init * 2) & (
            spectrum.working_data[:, 0] <= x0_guess + width_init * 2
        )
        working_data_peak = spectrum.working_data[mask]
        if len(working_data_peak) > 1:
            sampling_rate = np.mean(np.diff(working_data_peak[:, 0]))
        else:
            sampling_rate = np.mean(np.diff(spectrum.working_data[:, 0]))

        A_guess = spectrum.peaks[peak].A_init
        sigma_L_guess = width_init / 2
        sigma_R_guess = sigma_L_guess * asymmetry
        initial_params.extend(
            [A_guess, x0_guess, sigma_L_guess, sigma_R_guess, sampling_rate]
        )
        working_peak_list.append(peak)
        i += 1

    if working_peak_list == []:
        log(
            "No peaks are within the data range. Please adjust the peak detection parameters"
        )
        return None

    return initial_params, working_peak_list


def MBG_fit(
    render_callback, iterations=1000, std=0.25, use_filtered=True, use_gaussian=False
):
    spectrum = render_callback.spectrum
    baseline_window = dpg.get_value("baseline_window")
    spectrum.correct_baseline(baseline_window)
    init = initial_peaks_parameters(spectrum)
    if init is None:
        return

    initial_params, working_peak_list = init

    draw_fitted_peaks(delete=True)

    i = 0
    for peak in working_peak_list:
        A_guess, x0_guess, sigma_L_guess, sigma_R_guess, sampling_rate = initial_params[
            i * 5 : (i + 1) * 5
        ]
        print(
            f"Peak {peak}: A = {A_guess:.3f}, x0 = {x0_guess:.3f}, sigma_L = {sigma_L_guess:.3f}, sigma_R = {sigma_R_guess:.3f}, sampling_rate = {sampling_rate:.3f}"
        )

        spectrum.peaks[peak].A_refined = A_guess
        spectrum.peaks[peak].x0_refined = x0_guess
        spectrum.peaks[peak].sigma_L = sigma_L_guess
        spectrum.peaks[peak].sigma_R = sigma_R_guess
        spectrum.peaks[peak].fitted = False
        spectrum.peaks[peak].sampling_rate = sampling_rate
        i += 1

    fit = refine_peak_parameters(
        working_peak_list,
        initial_params,
        render_callback,
        iterations,
        std,
        use_filtered=use_filtered,
        use_gaussian=use_gaussian,
    )
    if fit:
        log("Fitting done with no error")
    else:
        log("Error while fitting")
        return

    update_peak_starting_points()
    draw_fitted_peaks()
    redraw_blocks()


def draw_base_projection():
    if not dpg.get_value("show_projection_checkbox"):
        for alias in dpg.get_aliases():
            if alias.startswith("fitted_regression_"):
                dpg.delete_item(alias)
        return

    spectrum: MSData = get_global_msdata_ref()

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        spectrum.peaks[peak].matched_with = []  # Reset matched_with for all peaks

        regression_0 = (
            -spectrum.peaks[peak].regression_fct[1]
            / spectrum.peaks[peak].regression_fct[0]
        )
        regression_x = regression_0 + spectrum.peaks[peak].sigma_L * 5
        regression_y = (
            spectrum.peaks[peak].regression_fct[0] * regression_x
            + spectrum.peaks[peak].regression_fct[1]
        )

        dpg.draw_line(
            (regression_0, 0),
            (regression_x, regression_y),
            parent="gaussian_fit_plot",
            color=(237, 43, 43),
            thickness=1,
            tag=f"fitted_regression_{peak}",
        )


def draw_residual(x_data, residual):
    dpg.show_item("residual")
    dpg.set_value("residual", [x_data, residual])


@dataclass
class FitQualityMetrics:
    rmse: float
    weighted_rmse: float
    r_squared: float
    chi_squared_reduced: float
    noise_variance: float
    noise_std: float
    signal_to_noise: float
    aic: float
    bic: float
    peak_quality: dict
    residual_autocorr: float


@dataclass
class FitQualityMetricsReduced:
    rmse: float
    weighted_rmse: float
    r_squared: float
    weights: np.ndarray
    sigma_L_mean: float = 0.0
    sigma_R_mean: float = 0.0
    sigma_L_std: float = 0.0
    sigma_R_std: float = 0.0


@overload
def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only: Literal[True],
) -> FitQualityMetricsReduced: ...


@overload
def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only: Literal[False] = False,
) -> FitQualityMetrics: ...


def calculate_bootstrap_standard_errors_residual(
    spectrum: MSData,
    working_peak_list: list,
    data_x: np.ndarray,
    data_y: np.ndarray,
    render_callback: RenderCallback,
    n_bootstrap=100,
) -> dict:
    """
    Calculate standard errors via residual bootstrap resampling.

    This method:
    1. Computes residuals from the current fit
    2. Resamples residuals with replacement
    3. Adds resampled residuals to the fitted curve
    4. Refits the model to the synthetic data
    5. Repeats and computes standard deviation of parameters

    Args:
        spectrum: MSData object with fitted peaks
        working_peak_list: List of peak indices to analyze
        data_x: X-axis data
        data_y: Y-axis data (observed)
        n_bootstrap: Number of bootstrap samples (50-200 typical)
        render_callback: Optional callback for progress updates

    Returns:
        dict: Standard errors for each peak's parameters
    """
    from copy import deepcopy

    # Store original fitted values
    original_peaks = {
        peak: {
            "A": spectrum.peaks[peak].A_refined,
            "x0": spectrum.peaks[peak].x0_refined,
            "sigma_L": spectrum.peaks[peak].sigma_L,
            "sigma_R": spectrum.peaks[peak].sigma_R,
        }
        for peak in working_peak_list
    }

    # Compute fitted values and residuals
    y_fitted = spectrum.calculate_mbg(data_x, fitting=True)
    residuals = data_y - y_fitted

    # Storage for bootstrap parameter estimates
    bootstrap_params = {
        peak: {
            "A": [],
            "x0": [],
            "sigma_L": [],
            "sigma_R": [],
        }
        for peak in working_peak_list
    }

    log(f"Starting bootstrap with {n_bootstrap} resamples...")
    successful_fits = 0

    for b in range(n_bootstrap):
        if render_callback:
            render_callback.execute()
            dpg.set_value(
                "Fitting_indicator_text",
                f"Bootstrap: {b+1}/{n_bootstrap} ({successful_fits} successful)",
            )

        # Resample residuals with replacement
        bootstrap_indices = np.random.choice(
            len(residuals), size=len(residuals), replace=True
        )
        resampled_residuals = residuals[bootstrap_indices]

        # Create synthetic data: fitted + resampled residuals
        y_bootstrap = y_fitted + resampled_residuals

        # Reset to perturbed parameters before refitting
        for peak in working_peak_list:
            # Add small random perturbation (15% noise)
            noise_scale_A = 0.05
            noise_scale_X0 = 0.05
            noise_scale_w = 0.10
            spectrum.peaks[peak].A_refined = original_peaks[peak]["A"] * (
                1 + np.random.randn() * noise_scale_A
            )
            spectrum.peaks[peak].x0_refined = original_peaks[peak]["x0"] + (
                np.random.randn() * original_peaks[peak]["sigma_L"] * noise_scale_X0
            )
            spectrum.peaks[peak].sigma_L = original_peaks[peak]["sigma_L"] * (
                1 + np.random.randn() * noise_scale_w
            )
            spectrum.peaks[peak].sigma_R = original_peaks[peak]["sigma_R"] * (
                1 + np.random.randn() * noise_scale_w
            )
        # Refit with bootstrap data (simplified - fewer iterations)
        try:
            # Create temporary spectrum copy to avoid modifying the original
            temp_spectrum = deepcopy(spectrum)

            # Run a quick refinement (fewer iterations for speed)
            for iteration in range(10):  # Reduced iterations for speed
                for peak in working_peak_list:
                    original_peak = peak_params(
                        A_init=original_peaks[peak]["A"],
                        x0_init=original_peaks[peak]["x0"],
                        width=original_peaks[peak]["sigma_L"]
                        + original_peaks[peak]["sigma_R"],
                    )
                    _refine_iteration(
                        peak,
                        data_x,
                        y_bootstrap,
                        temp_spectrum,
                        original_peak,
                        force_gaussian=False,
                        widths=(-1, -1, -1, -1),
                    )

            # Store the fitted parameters
            for peak in working_peak_list:
                A = temp_spectrum.peaks[peak].A_refined
                x0 = temp_spectrum.peaks[peak].x0_refined
                sigma_L = temp_spectrum.peaks[peak].sigma_L
                sigma_R = temp_spectrum.peaks[peak].sigma_R

                # Sanity check: reject obviously bad fits
                if (
                    A > 0
                    and sigma_L > 0
                    and sigma_R > 0
                    and not np.isnan(A)
                    and not np.isnan(x0)
                    and not np.isnan(sigma_L)
                    and not np.isnan(sigma_R)
                ):

                    bootstrap_params[peak]["A"].append(A)
                    bootstrap_params[peak]["x0"].append(x0)
                    bootstrap_params[peak]["sigma_L"].append(sigma_L)
                    bootstrap_params[peak]["sigma_R"].append(sigma_R)

            successful_fits += 1

        except Exception as e:
            log(f"Bootstrap iteration {b+1} failed: {e}")
            continue

    # Restore original parameters
    for peak in working_peak_list:
        spectrum.peaks[peak].A_refined = original_peaks[peak]["A"]
        spectrum.peaks[peak].x0_refined = original_peaks[peak]["x0"]
        spectrum.peaks[peak].sigma_L = original_peaks[peak]["sigma_L"]
        spectrum.peaks[peak].sigma_R = original_peaks[peak]["sigma_R"]

    # Compute standard errors from bootstrap distribution
    result = {}
    for peak in working_peak_list:
        if len(bootstrap_params[peak]["A"]) < n_bootstrap * 0.5:
            log(
                f"Warning: Peak {peak} had only {len(bootstrap_params[peak]['A'])} successful fits"
            )
            result[peak] = None
            continue

        result[peak] = {
            "A": float(np.std(bootstrap_params[peak]["A"])),
            "x0": float(np.std(bootstrap_params[peak]["x0"])),
            "sigma_L": float(np.std(bootstrap_params[peak]["sigma_L"])),
            "sigma_R": float(np.std(bootstrap_params[peak]["sigma_R"])),
            # Also store percentile confidence intervals
            "A_ci": (
                float(np.percentile(bootstrap_params[peak]["A"], 2.5)),
                float(np.percentile(bootstrap_params[peak]["A"], 97.5)),
            ),
            "x0_ci": (
                float(np.percentile(bootstrap_params[peak]["x0"], 2.5)),
                float(np.percentile(bootstrap_params[peak]["x0"], 97.5)),
            ),
            "n_samples": len(bootstrap_params[peak]["A"]),
        }

    log(f"Bootstrap complete: {successful_fits}/{n_bootstrap} successful fits")
    return result


def calculate_fit_quality_metrics(
    data_x,
    data_y,
    spectrum,
    working_peak_list,
    weights=None,
    *,
    rmse_only=False,
):  # -> FitQualityMetricsReduced | dict[str, Any]:
    """Calculate comprehensive fit quality metrics"""
    residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)

    sigma_L_list = []
    sigma_R_list = []
    # 1. Weighted RMSE (higher weight for peak regions)
    if weights is None:
        weights = np.ones_like(data_y)
        for peak in working_peak_list:
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            # Create Gaussian weight centered at peak
            peak_mask = (data_x >= x0 - 3 * sigma_L) & (data_x <= x0 + 3 * sigma_R)
            weights[peak_mask] *= 5.0  # 5x weight for peak regions
            sigma_L_list.append(sigma_L)
            sigma_R_list.append(sigma_R)

    weighted_rmse = np.sqrt(np.average(np.square(residual), weights=weights))

    # 2. R-squared (coefficient of determination)
    ss_res = np.sum(np.square(residual))
    ss_tot = np.sum(np.square(data_y - np.mean(data_y)))
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    if rmse_only:
        return FitQualityMetricsReduced(
            rmse=np.sqrt(np.mean(np.square(residual))),
            weighted_rmse=weighted_rmse,
            r_squared=r_squared,
            weights=weights,
            sigma_L_mean=float(np.mean(sigma_L_list) if sigma_L_list else 0.0),
            sigma_R_mean=float(np.mean(sigma_R_list) if sigma_R_list else 0.0),
            sigma_L_std=float(np.std(sigma_L_list) if sigma_L_list else 0.0),
            sigma_R_std=float(np.std(sigma_R_list) if sigma_R_list else 0.0),
        )

    # 3. Chi-squared for noisy data
    degrees_of_freedom = len(data_y) - (len(working_peak_list) * 4)
    # Method 1: Estimate noise from baseline regions
    noise_mask = np.ones_like(data_y, dtype=bool)
    for peak in working_peak_list:
        if spectrum.peaks[peak].fitted:
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            # Exclude peak regions from noise calculation
            peak_mask = (data_x >= x0 - 4 * sigma_L) & (data_x <= x0 + 4 * sigma_R)
            noise_mask &= ~peak_mask

    # Estimate noise variance from baseline regions
    if np.sum(noise_mask) > 10:  # Need enough baseline points
        baseline_residual = residual[noise_mask]
        noise_variance = np.var(baseline_residual)
    else:
        # Fallback: robust estimate from all residuals using median absolute deviation
        mad = np.median(np.abs(residual - np.median(residual)))
        noise_variance = (1.4826 * mad) ** 2  # Convert MAD to variance estimate

    # Ensure minimum noise level
    min_noise = np.var(data_y) * 0.001  # At least 0.1% of signal variance
    noise_variance = max(noise_variance, min_noise)

    if degrees_of_freedom > 0:
        chi_squared_reduced = ss_res / (degrees_of_freedom * noise_variance)
    else:
        chi_squared_reduced = np.inf

    # 4. Peak-specific metrics
    peak_quality = {}
    for peak in working_peak_list:
        x0 = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R

        # Peak region mask
        peak_mask = (data_x >= x0 - 3 * sigma_L) & (data_x <= x0 + 3 * sigma_R)
        if np.any(peak_mask):
            peak_residual = residual[peak_mask]
            peak_data = data_y[peak_mask]

            # Signal-to-noise ratio at peak
            peak_height = spectrum.peaks[peak].A_refined
            noise_level = np.std(peak_residual)
            snr = peak_height / noise_level if noise_level > 0 else np.inf

            # Peak RMSE
            peak_rmse = np.sqrt(np.mean(np.square(peak_residual)))

            peak_quality[peak] = FitQualityPeakMetrics(
                snr=snr,
                peak_rmse=peak_rmse,
                relative_error=(peak_rmse / peak_height if peak_height > 0 else np.inf),
            )

    # 5. Akaike Information Criterion (AIC)
    n = len(data_y)
    k = len(working_peak_list) * 4  # number of parameters
    if n > 0 and ss_res > 0:
        aic = n * np.log(ss_res / n) + 2 * k
    else:
        aic = np.inf

    # 6. Bayesian Information Criterion (BIC)
    if n > 0 and ss_res > 0:
        bic = n * np.log(ss_res / n) + k * np.log(n)
    else:
        bic = np.inf

    return FitQualityMetrics(
        rmse=np.sqrt(np.mean(np.square(residual))),
        weighted_rmse=weighted_rmse,
        r_squared=r_squared,
        chi_squared_reduced=chi_squared_reduced,
        noise_variance=float(noise_variance),
        noise_std=np.sqrt(noise_variance),
        signal_to_noise=np.mean(np.abs(data_y)) / np.sqrt(noise_variance),
        aic=aic,
        bic=bic,
        peak_quality=peak_quality,
        residual_autocorr=(
            np.corrcoef(residual[:-1], residual[1:])[0, 1] if len(residual) > 1 else 0
        ),
    )


def refine_peak_parameters(
    working_peak_list,
    mbg_params,
    render_callback: RenderCallback,
    iterations=1000,
    std=0.25,
    use_filtered=True,
    use_gaussian=False,
):
    spectrum = render_callback.spectrum
    original_peaks: dict[int, peak_params] = {
        peak: spectrum.peaks[peak] for peak in working_peak_list
    }
    data_x = spectrum.working_data[:, 0]
    data_y = spectrum.baseline_corrected[:, 1]

    # Store quality metrics history
    quality_history = []
    iterations_list = [i for i in range(len(working_peak_list))]
    start = time.time()

    if use_filtered:
        window_length = get_smoothing_window()
        data_x = spectrum.baseline_corrected[:, 0]
        data_y = np.array(
            spectrum.get_filtered_data(window_length=window_length, baseline=True)
        )

    else:
        data_x = spectrum.baseline_corrected[:, 0]
        data_y = spectrum.baseline_corrected[:, 1]

    ## Iteration loop start here
    ##############################
    k = 0
    sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std = -1, -1, -1, -1

    current_metric = 0.0
    for k in range(iterations + 1):
        iteration_start = time.time()
        render_callback.execute()
        if dpg.get_value("stop_fitting_checkbox") or render_callback.stop_fitting:
            log("Fitting stopped by user")
            break

        quality_metrics: FitQualityMetricsReduced = calculate_fit_quality_metrics(
            data_x, data_y, spectrum, working_peak_list, rmse_only=True
        )
        if dpg.get_value("use_reduced"):
            sigma_L_mean = quality_metrics.sigma_L_mean
            sigma_R_mean = quality_metrics.sigma_R_mean
            sigma_L_std = quality_metrics.sigma_L_std
            sigma_R_std = quality_metrics.sigma_R_std

        widths = (sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std)
        quality_history.append(quality_metrics)
        residual = data_y - spectrum.calculate_mbg(data_x, fitting=True)
        show_MBG(spectrum, True)

        if dpg.get_value("show_residual_checkbox"):
            dpg.set_value("residual", [data_x.tolist(), residual.tolist()])

        # Use weighted RMSE for convergence check
        current_metric = quality_metrics.weighted_rmse

        if k > 10:
            recent_metrics = [q.weighted_rmse for q in quality_history[-10:]]
            current_std = np.std(recent_metrics)
        else:
            current_std = np.inf
        r_squared = quality_metrics.r_squared

        dpg.set_value(
            "Fitting_indicator_text",
            f"Iter {k}: wRMSE={current_metric:.4f}, R²={r_squared:.4f}, Change in error={current_std:.4f}, iteration time: {time.time() - iteration_start:.2f}s",
        )
        print(widths)

        # Check convergence using multiple criteria
        if k > 10:
            converged = current_std < std and r_squared > 0.90
            if converged:
                break

        # Shuffle iterations order for next pass
        iterations_list = np.random.permutation(
            iterations_list
        )  # do not use the same order every iteration
        for i in iterations_list:
            peak = working_peak_list[i]
            original_peak = original_peaks[peak]
            _refine_iteration(
                peak,
                data_x,
                data_y,
                spectrum,
                original_peak,
                force_gaussian=use_gaussian,
                widths=widths,
            )

    ##############################
    # End of iteration loop
    ##############################
    full_quality_metrics: FitQualityMetrics = calculate_fit_quality_metrics(
        data_x,
        data_y,
        spectrum,
        working_peak_list,
    )

    # Calculate standard errors using Hessian
    standard_errors = calculate_bootstrap_standard_errors_residual(
        spectrum, working_peak_list, data_x, data_y, render_callback, n_bootstrap=100
    )

    # Store standard errors in peak parameters
    for peak in working_peak_list:
        if standard_errors[peak] is not None:
            spectrum.peaks[peak].se_A = standard_errors[peak]["A"]
            spectrum.peaks[peak].se_x0 = standard_errors[peak]["x0"]
            spectrum.peaks[peak].se_sigma_L = standard_errors[peak]["sigma_L"]
            spectrum.peaks[peak].se_sigma_R = standard_errors[peak]["sigma_R"]

            # Propagate uncertainty to integral
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            A_refined = spectrum.peaks[peak].A_refined

            se_A = spectrum.peaks[peak].se_A
            se_sigma_L = spectrum.peaks[peak].se_sigma_L
            se_sigma_R = spectrum.peaks[peak].se_sigma_R
            # Approximate integral SE (simplified)
            se_integral = np.sqrt(2 * np.pi) * np.sqrt(
                ((sigma_L + sigma_R) / 2 * se_A) ** 2
                + (A_refined / 2 * se_sigma_L) ** 2
                + (A_refined / 2 * se_sigma_R) ** 2
            )
            spectrum.peaks[peak].se_integral = se_integral

    chi_squared = full_quality_metrics.chi_squared_reduced

    signal_to_noise = full_quality_metrics.signal_to_noise
    peaks_error = [
        full_quality_metrics.peak_quality[peak].relative_error
        for peak in working_peak_list
        if peak in full_quality_metrics.peak_quality
    ]
    r_squared = full_quality_metrics.r_squared

    time_taken = time.time() - start
    log(
        f"Converged: R²={r_squared:.4f}, X²r={chi_squared:.3f}, SNR={signal_to_noise:.1f}, Time: {time_taken:.2f}s"
    )

    dpg.set_value(
        "Fitting_indicator_text",
        f"Converged after {k} iterations. wRMSE={current_metric:.4f}, R²={r_squared:.4f}, "
        f"X²r={chi_squared:.3f}, Median peak error={np.median(peaks_error):.4f}, Time: {time_taken:.2f}s",
    )

    print(full_quality_metrics.peak_quality)
    for peak in working_peak_list:
        spectrum.peaks[peak].fitted = True
        spectrum.peaks[peak].fit_quality = full_quality_metrics.peak_quality.get(
            peak, {}
        )

    return True


def _refine_iteration(
    peak: int,
    data_x,
    data_y,
    spectrum: MSData,
    original_peak,
    force_gaussian=False,
    widths=(-1, -1, -1, -1),
):
    x0_fit = spectrum.peaks[peak].x0_refined
    sigma_L_fit = spectrum.peaks[peak].sigma_L
    sigma_R_fit = spectrum.peaks[peak].sigma_R
    sampling_rate = spectrum.peaks[peak].sampling_rate

    # Adjust the amplitude
    R_val = sigma_R_fit / 4 if sigma_R_fit > sampling_rate * 20 else sampling_rate * 5
    L_val = sigma_L_fit / 4 if sigma_L_fit > sampling_rate * 20 else sampling_rate * 5
    mask = (data_x >= x0_fit - R_val) & (data_x <= x0_fit + L_val)
    data_x_peak = data_x[mask]
    data_y_peak = data_y[mask]
    if len(data_x_peak) > 0 and len(data_y_peak) > 0:
        peak_error = np.mean(
            data_y_peak - spectrum.calculate_mbg(data_x_peak, fitting=True)
        )
        spectrum.peaks[peak].A_refined = spectrum.peaks[peak].A_refined + (
            peak_error / 10
        )

    # Sharpen the peak
    for iteration in range(1, 4):
        min_window = sampling_rate * 3
        L_window = max(sigma_L_fit * iteration, min_window)
        R_window = max(sigma_R_fit * iteration, min_window)
        L_mask = (data_x >= x0_fit - L_window) & (data_x <= x0_fit - L_window / 2)
        R_mask = (data_x >= x0_fit + R_window / 2) & (data_x <= x0_fit + R_window)

        data_x_L = data_x[L_mask]
        data_y_L = data_y[L_mask]
        data_x_R = data_x[R_mask]
        data_y_R = data_y[R_mask]

        # Check if we have enough data points
        if len(data_x_L) < 3 or len(data_x_R) < 3:
            continue

        mbg_L = spectrum.calculate_mbg(data_x_L, fitting=True)
        mbg_R = spectrum.calculate_mbg(data_x_R, fitting=True)

        error_l = np.mean((data_y_L - mbg_L))
        error_r = np.mean((data_y_R - mbg_R))

        if (
            np.isnan(error_l)
            or np.isnan(error_r)
            or np.isinf(error_l)
            or np.isinf(error_r)
        ):
            continue

        # Old approach
        # moved = False
        if iteration == 1:
            offset = abs((sigma_L_fit + sigma_R_fit) / 2000)
            if error_l > 0 and error_r < 0:
                x0_fit = x0_fit - offset
                moved = True
            elif error_l < 0 and error_r > 0:
                x0_fit = x0_fit + offset
                moved = True
        # # New simplified approach
        # asym_error = error_r - error_l
        # if iteration == 1:
        #     offset = abs((sigma_L_fit + sigma_R_fit) / 2000)
        #     x0_fit = x0_fit + offset * np.sign(asym_error)

        # if not moved:
        val = 1000 * iteration

        max_adjustment = original_peak.width * 0.01  # 1% of original width

        sigma_L_adjustment = np.clip(error_l / val, -max_adjustment, max_adjustment)
        sigma_R_adjustment = np.clip(error_r / val, -max_adjustment, max_adjustment)

        sigma_L_fit = sigma_L_fit + sigma_L_adjustment
        sigma_R_fit = sigma_R_fit + sigma_R_adjustment

        if sigma_L_fit < sampling_rate:
            sigma_L_fit = sampling_rate * 3
        if sigma_R_fit < sampling_rate:
            sigma_R_fit = sampling_rate * 3

        if np.isnan(sigma_L_fit):
            sigma_L_fit = sampling_rate * 3
        if np.isnan(sigma_R_fit):
            sigma_R_fit = sampling_rate * 3

        if force_gaussian:
            sigma_L_fit = (sigma_L_fit + sigma_R_fit) / 2
            sigma_R_fit = sigma_L_fit

        # Apply width regularization if enabled
        if any(w != -1 for w in widths):
            factor = 4
            sigma_L_mean, sigma_R_mean, sigma_L_std, sigma_R_std = widths
            if sigma_L_fit > sigma_L_mean + sigma_L_std * factor:
                sigma_L_fit = sigma_L_fit - sigma_L_std / factor
            if sigma_L_fit < sigma_L_mean - sigma_L_std * factor:
                sigma_L_fit = sigma_L_fit + sigma_L_std / factor
            if sigma_R_fit > sigma_R_mean + sigma_R_std * factor:
                sigma_R_fit = sigma_R_fit - sigma_R_std / factor
            if sigma_R_fit < sigma_R_mean - sigma_R_std * factor:
                sigma_R_fit = sigma_R_fit + sigma_R_std / factor

        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].x0_refined = x0_fit


def update_peak_params(peak_list, popt, spectrum: MSData):
    i = 0
    for peak in peak_list:
        A_fit, x0_fit, sigma_L_fit, sigma_R_fit = popt[i * 4 : (i + 1) * 4]
        spectrum.peaks[peak].A_refined = A_fit
        spectrum.peaks[peak].x0_refined = x0_fit
        spectrum.peaks[peak].sigma_L = sigma_L_fit
        spectrum.peaks[peak].sigma_R = sigma_R_fit
        spectrum.peaks[peak].fitted = True
        i += 1


def draw_fitted_peaks(delete=False):
    spectrum = get_global_msdata_ref()
    # Delete previous peaks
    for alias in dpg.get_aliases():
        if (
            alias.startswith("fitted_peak_")
            or alias.startswith("peak_annotation_")
            or alias.startswith("fitted_peaks_theme_")
        ):
            dpg.delete_item(alias)
    if delete:
        return
    # Generate fitted curve
    peak_list = []
    mbg_param = []
    colors = sns.color_palette("plasma", 20)

    i = 0
    for peak in spectrum.peaks:
        x0_fit = spectrum.peaks[peak].x0_refined
        if (
            x0_fit < spectrum.working_data[:, 0][0]
            or x0_fit > spectrum.working_data[:, 0][-1]
        ):
            continue
        if not spectrum.peaks[peak].fitted:
            continue

        peak_error = spectrum.peaks[peak].fit_quality.relative_error * 3
        normalized_error = np.clip(peak_error, 0, 1)
        color_idx = int(normalized_error * (len(colors) - 1))
        color = [int(c * 255) for c in colors[color_idx]]

        shade_color = color.copy()
        shade_color.append(100)

        with dpg.theme(tag=f"fitted_peaks_theme_{peak}"):
            with dpg.theme_component(dpg.mvAll):
                dpg.add_theme_style(
                    dpg.mvPlotStyleVar_LineWeight, 3, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Line, color, category=dpg.mvThemeCat_Plots
                )
                dpg.add_theme_color(
                    dpg.mvPlotCol_Fill, shade_color, category=dpg.mvThemeCat_Plots
                )

        A = spectrum.peaks[peak].A_refined
        sigma_L_fit = spectrum.peaks[peak].sigma_L
        sigma_R_fit = spectrum.peaks[peak].sigma_R
        peak_list.append(peak)

        x_individual_fit = np.linspace(
            x0_fit - 4 * sigma_L_fit, x0_fit + 4 * sigma_R_fit, 500
        )
        y_individual_fit = bi_gaussian(
            x_individual_fit, A, x0_fit, sigma_L_fit, sigma_R_fit
        )
        mbg_param.extend([A, x0_fit, sigma_L_fit, sigma_R_fit])

        dpg.add_line_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak}",
            parent="y_axis_plot2",
            tag=f"fitted_peak_{peak}",
        )
        dpg.bind_item_theme(f"fitted_peak_{peak}", f"fitted_peaks_theme_{peak}")
        dpg.add_shade_series(
            x_individual_fit.tolist(),
            y_individual_fit.tolist(),
            label=f"Peak {peak} area",
            parent="y_axis_plot2",
            tag=f"fitted_peak_{peak}_area",
            show=True,
        )
        dpg.bind_item_theme(f"fitted_peak_{peak}_area", f"fitted_peaks_theme_{peak}")
        dpg.add_plot_annotation(
            label=f"Peak {peak}",
            default_value=(x0_fit, A),
            offset=(-15, -15),
            color=[120, 120, 120],
            clamped=False,
            parent="gaussian_fit_plot",
            tag=f"peak_annotation_{peak}",
        )
        i += 1

    show_MBG(spectrum)
    update_peak_table(spectrum)


def show_MBG(spectrum: MSData, fitting=False):
    x_fit = np.linspace(
        np.min(spectrum.working_data[:, 0]),
        np.max(spectrum.working_data[:, 0]),
        spectrum.working_data.shape[0] // 2,
    )
    y_fit = spectrum.calculate_mbg(x_fit, fitting=fitting)
    dpg.show_item("MBG_plot2")
    dpg.set_value("MBG_plot2", [x_fit.tolist(), y_fit.tolist()])


def update_peak_table(spectrum: MSData):
    children = dpg.get_item_children("peak_table")
    if children and len(children) > 1 and isinstance(children[1], list):
        for tag in children[1]:
            dpg.delete_item(tag)

    for peak in spectrum.peaks:
        if not spectrum.peaks[peak].fitted:
            continue
        apex = spectrum.peaks[peak].x0_refined
        sigma_L = spectrum.peaks[peak].sigma_L
        sigma_R = spectrum.peaks[peak].sigma_R
        A_refined = spectrum.peaks[peak].A_refined

        integral = bi_gaussian_integral(A_refined, sigma_L, sigma_R)
        spectrum.peaks[peak].integral = integral
        rel_error = spectrum.peaks[peak].fit_quality.relative_error

        # Get standard errors if available
        se_x0 = getattr(spectrum.peaks[peak], "se_x0", None)
        se_integral = spectrum.peaks[peak].se_integral

        if spectrum.peaks[peak].regression_fct[0] == 0:
            regression_0 = 0
        else:
            regression_0 = (
                -spectrum.peaks[peak].regression_fct[1]
                / spectrum.peaks[peak].regression_fct[0]
            )

        with dpg.table_row(parent="peak_table"):
            dpg.add_text(f"Peak {peak}")
            dpg.add_text(f"{regression_0:.2f}")
            apex_text = f"{apex:.2f}" + (f" ± {se_x0:.2f}" if se_x0 else "")
            dpg.add_text(apex_text)
            integral_text = f"{integral:.0f}" + (
                f" ± {(se_integral / integral * 100):.2f}%" if se_integral else ""
            )
            dpg.add_text(integral_text)
            dpg.add_text(f"{sigma_L:.4f}")
            dpg.add_text(f"{sigma_R:.4f}")
            dpg.add_text(f"{rel_error:.4f}")


def update_peak_starting_points():
    spectrum = get_global_msdata_ref()
    regression_projection = True

    if dpg.get_value("show_centers"):
        center_width = dpg.get_value("center_width") / 100
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue
            apex = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R
            spectrum.peaks[peak].start_range = (
                apex - sigma_L * center_width,
                apex + sigma_R * center_width,
            )

    elif regression_projection:
        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            A = spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex
            sigma_L = spectrum.peaks[peak].sigma_L

            while True:
                A_current = bi_gaussian(
                    start,
                    spectrum.peaks[peak].A_refined,
                    apex,
                    spectrum.peaks[peak].sigma_L,
                    spectrum.peaks[peak].sigma_R,
                )
                if A_current <= 0.8 * A:
                    break
                start -= 0.02
            mz80pcs = start

            while True:
                A_current = bi_gaussian(
                    start,
                    spectrum.peaks[peak].A_refined,
                    apex,
                    spectrum.peaks[peak].sigma_L,
                    spectrum.peaks[peak].sigma_R,
                )
                if A_current <= 0.20 * A:
                    break
                start -= 0.02
            mz25pcs = start

            sample_points = np.linspace(mz80pcs, mz25pcs, 10)
            mz_samples = []
            A_samples = []

            for sample_mz in sample_points:
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

    else:

        for peak in spectrum.peaks:
            if not spectrum.peaks[peak].fitted:
                continue

            A = spectrum.peaks[peak].A_refined
            apex = spectrum.peaks[peak].x0_refined
            start = apex - spectrum.peaks[peak].sigma_L

            while (
                bi_gaussian(
                    start,
                    spectrum.peaks[peak].A_refined,
                    apex,
                    spectrum.peaks[peak].sigma_L,
                    spectrum.peaks[peak].sigma_R,
                )
                > 0.1 * A
            ):
                start -= 0.01
            start10pcs = start
            while (
                bi_gaussian(
                    start,
                    spectrum.peaks[peak].A_refined,
                    apex,
                    spectrum.peaks[peak].sigma_L,
                    spectrum.peaks[peak].sigma_R,
                )
                > 5 * A
            ):
                start -= 0.01
            start1pcs = start
            spectrum.peaks[peak].start_range = (start1pcs, start10pcs)
