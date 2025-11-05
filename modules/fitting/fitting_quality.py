from calendar import c
from dataclasses import dataclass
import os
from typing import Dict, Literal, overload
import numpy as np
from modules.data_structures import FitQualityPeakMetrics, MSData, peak_params
from modules.math import bi_Lorentzian_integral, bi_gaussian_integral, standard_error
from modules.rendercallback import RenderCallback
from modules.utils import log
import dearpygui.dearpygui as dpg
from modules.fitting.refiner import refine_iteration
from copy import deepcopy
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock


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

            # Per-peak R-squared
            ss_res_peak = np.sum(np.square(peak_residual))
            ss_tot_peak = np.sum(np.square(peak_data - np.mean(peak_data)))
            peak_r_squared = 1 - (ss_res_peak / ss_tot_peak) if ss_tot_peak > 0 else 0

            peak_quality[peak] = FitQualityPeakMetrics(
                snr=snr,
                peak_rmse=peak_rmse,
                relative_error=(peak_rmse / peak_height if peak_height > 0 else np.inf),
                r_squared=peak_r_squared,
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


@dataclass
class PerturbationFitErrorMetrics:
    A: list[float]
    x0: list[float]
    sigma_L: list[float]
    sigma_R: list[float]
    integral: list[float]


def advanced_statistical_analysis(
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    data_y: np.ndarray,
    render_callback: RenderCallback,
    macro_iteration=100,
    micro_iteration=10,
    method: Literal[
        "bootstrap-parametric", "bootstrap-residual", "initial"
    ] = "bootstrap-parametric",
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False] = False,
    convergence_threshold=1e-4,
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
        macro_iteration: Number of bootstrap samples (50-200 typical)
        render_callback: Optional callback for progress updates

    Returns:
        dict: Standard errors for each peak's parameters
    """

    # Compute fitted values and residuals
    y_fitted = spectrum.calculate_mbg(data_x, fitting=True)
    residuals = data_y - y_fitted
    mad = np.median(np.abs(residuals - np.median(residuals)))
    sigma_hat = 1.4826 * mad

    initial_peaks: dict[int, peak_params] = {}
    for peak in working_peak_list:
        initial_peaks[peak] = peak_params(
            A_refined=spectrum.peaks[peak].A_refined,
            x0_refined=spectrum.peaks[peak].x0_refined,
            sigma_L=spectrum.peaks[peak].sigma_L,
            sigma_R=spectrum.peaks[peak].sigma_R,
        )

    perturbation_results: dict[int, PerturbationFitErrorMetrics] = {
        peak: PerturbationFitErrorMetrics(
            A=[], x0=[], sigma_L=[], sigma_R=[], integral=[]
        )
        for peak in working_peak_list
    }

    log(f"Starting bootstrap with {macro_iteration} resamples...")
    successful_fits = 0

    noise_scale_A = 0
    noise_scale_X0 = 0
    noise_scale_w = 0

    task_pool = []
    # Thread-safe counters
    cpu_count = os.cpu_count() or 1
    completed_lock = Lock()
    completed_tasks = {"count": 0, "successful": 0}

    for b in range(macro_iteration):
        working_peaks: Dict[int, peak_params] = {}

        if method == "bootstrap-residual" or method == "bootstrap-parametric":
            noise_scale_A = 0.0
            noise_scale_X0 = 0.0
            noise_scale_w = 0.0

            if render_callback:
                render_callback.execute()
                dpg.set_value(
                    "Fitting_indicator_text",
                    f"Setting up Bootstrap 0 to {cpu_count}/{macro_iteration}",
                )

            if method == "bootstrap-parametric":
                noise = np.random.normal(0, sigma_hat, size=len(residuals))
                task_data_y = y_fitted + noise

            else:  # residual bootstrap
                bootstrap_indices = np.random.choice(
                    len(residuals), size=len(residuals), replace=True
                )
                resampled_residuals = residuals[bootstrap_indices]
                task_data_y = y_fitted + resampled_residuals

        elif method == "initial":
            noise_scale_A = 0.2
            noise_scale_X0 = 0.05
            noise_scale_w = 0.2

            if render_callback:
                render_callback.execute()
                dpg.set_value(
                    "Fitting_indicator_text",
                    f"Setting up Initial parameter test task 0 to {cpu_count}/{macro_iteration}",
                )
            task_data_y = data_y

        for peak in working_peak_list:
            search_width = spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
            working_peaks[peak] = peak_params(
                A_refined=spectrum.peaks[peak].A_init
                * (1 + np.random.randn() * noise_scale_A),
                x0_refined=spectrum.peaks[peak].x0_init
                + (np.random.randn() * search_width * noise_scale_X0),
                sigma_L=spectrum.peaks[peak].sigma_L_init
                * (1 + np.random.randn() * noise_scale_w),
                sigma_R=spectrum.peaks[peak].sigma_R_init
                * (1 + np.random.randn() * noise_scale_w),
            )

        task_spectrum = deepcopy(spectrum)

        task = QuickFitInterface(
            spectrum=task_spectrum,
            working_peaks=working_peaks,
            data_x=data_x,
            data_y=task_data_y,
            n_iterations=micro_iteration,
            check_convergence=check_convergence,
            theta_threshold=1e-4,
            wRMSE_threshold=convergence_threshold,
            width_regularization=True,
            it_index=b,
            indicator=None,
        )
        task_pool.append(task)

    with ThreadPoolExecutor(max_workers=cpu_count) as executor:
        futures = {executor.submit(execute_quick_fit, task): task for task in task_pool}

        for future in as_completed(futures):
            task = futures[future]
            fitted_peaks, converged = future.result()

            # Update progress (thread-safe)
            with completed_lock:
                completed_tasks["count"] += 1
                if fitted_peaks != {}:
                    completed_tasks["successful"] += 1

                # Update GUI
                if render_callback:
                    render_callback.execute()
                    if method == "initial":
                        dpg.set_value(
                            "Fitting_indicator_text",
                            f"Refit with randomisation: {completed_tasks['count']}/{macro_iteration} "
                            f"({completed_tasks['successful']} successful)",
                        )
                    else:
                        dpg.set_value(
                            "Fitting_indicator_text",
                            f"Bootstrap: {completed_tasks['count']}/{macro_iteration} "
                            f"({completed_tasks['successful']} successful)",
                        )

                if fitted_peaks == {}:
                    continue  # Skip failed fits
                for peak in fitted_peaks:
                    if spectrum.peak_model == "lorentzian":
                        integral = bi_Lorentzian_integral(
                            fitted_peaks[peak].A_refined,
                            fitted_peaks[peak].sigma_L,
                            fitted_peaks[peak].sigma_R,
                        )
                    else:
                        integral = bi_gaussian_integral(
                            fitted_peaks[peak].A_refined,
                            fitted_peaks[peak].sigma_L,
                            fitted_peaks[peak].sigma_R,
                        )
                    perturbation_results[peak].A.append(fitted_peaks[peak].A_refined)
                    perturbation_results[peak].x0.append(fitted_peaks[peak].x0_refined)
                    perturbation_results[peak].sigma_L.append(
                        fitted_peaks[peak].sigma_L
                    )
                    perturbation_results[peak].sigma_R.append(
                        fitted_peaks[peak].sigma_R
                    )
                    perturbation_results[peak].integral.append(integral)

    # Compute standard errors from bootstrap distribution
    final_result = {}
    for peak in working_peak_list:
        if len(perturbation_results[peak].A) < macro_iteration * 0.5:
            log(
                f"Warning: Peak {peak} had only {len(perturbation_results[peak].A)} successful fits"
            )
            final_result[peak] = None
            continue

        final_result[peak] = {
            "A": standard_error((perturbation_results[peak].A)),
            "x0": standard_error((perturbation_results[peak].x0)),
            "sigma_L": standard_error((perturbation_results[peak].sigma_L)),
            "sigma_R": standard_error((perturbation_results[peak].sigma_R)),
            "integral": standard_error((perturbation_results[peak].integral)),
            "n_samples": len(perturbation_results[peak].A),
        }

    log(f"Perturbation complete: {successful_fits}/{macro_iteration} successful fits")
    return final_result


@dataclass
class QuickFitInterface:
    spectrum: MSData
    working_peaks: Dict[int, peak_params]
    data_x: np.ndarray
    data_y: np.ndarray
    n_iterations: int
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False]
    theta_threshold: float
    wRMSE_threshold: float
    width_regularization: bool
    it_index: int
    indicator: Optional[str]


def execute_quick_fit(task: QuickFitInterface):
    return quick_fit_model(
        task.spectrum,
        task.working_peaks,
        task.data_x,
        task.data_y,
        task.n_iterations,
        task.check_convergence,
        task.theta_threshold,
        task.wRMSE_threshold,
        task.width_regularization,
        task.it_index,
        task.indicator,
    )


def quick_fit_model(
    spectrum: MSData,
    working_peaks: Dict[int, peak_params],
    data_x: np.ndarray,
    data_y: np.ndarray,
    n_iterations=10,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False] = False,
    theta_threshold: float = 1e-4,
    wRMSE_threshold: float = 1e-4,
    width_regularization: bool = True,
    it_index: int = 0,
    indicator: Optional[str] = None,
) -> tuple[Dict[int, peak_params], bool]:
    try:
        converged = False
        # Create temporary spectrum copy to avoid modifying the original

        for peak in working_peaks:
            spectrum.peaks[peak].A_refined = working_peaks[peak].A_refined
            spectrum.peaks[peak].x0_refined = working_peaks[peak].x0_refined
            spectrum.peaks[peak].sigma_L = working_peaks[peak].sigma_L
            spectrum.peaks[peak].sigma_R = working_peaks[peak].sigma_R
            spectrum.peaks[peak].width = (
                working_peaks[peak].sigma_L + working_peaks[peak].sigma_R
            )

        widths = (-1, -1, -1, -1)
        quality_metrics: Optional[FitQualityMetricsReduced] = None
        # Run a quick refinement (fewer iterations for speed)
        for iteration in range(n_iterations):
            old_theta = spectrum.get_packed_parameters()
            random_list = np.random.permutation(list(working_peaks.keys()))
            if width_regularization and quality_metrics:
                widths = (
                    quality_metrics.sigma_L_mean,
                    quality_metrics.sigma_R_mean,
                    quality_metrics.sigma_L_std,
                    quality_metrics.sigma_R_std,
                )

            for peak in random_list:
                refine_iteration(
                    peak=peak,
                    data_x=data_x,
                    data_y=data_y,
                    spectrum=spectrum,
                    original_peak_width=spectrum.peaks[peak].width,
                    force_gaussian=False,
                    widths=widths,
                )

            quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                spectrum,
                list(working_peaks.keys()),
                rmse_only=True,
            )
            if indicator:
                dpg.set_value(
                    indicator,
                    f"Quick fit cycle {it_index}, {iteration+1}/{n_iterations}, RMSE: {quality_metrics.weighted_rmse:.4f} target {wRMSE_threshold:.4f}",
                )

            if check_convergence:
                rmse_converged = False
                theta_converged = False
                if check_convergence == "wRMSE" or check_convergence == "both":
                    rmse_converged = quality_metrics.weighted_rmse < wRMSE_threshold

                if check_convergence == "theta-gradient" or check_convergence == "both":
                    new_theta = spectrum.get_packed_parameters()
                    theta_converged, delta_theta = check_local_convergence(
                        old_theta,
                        new_theta,
                        tol_theta=theta_threshold,
                    )

                if check_convergence == "both":
                    converged = bool(rmse_converged and theta_converged)
                elif check_convergence == "wRMSE":
                    converged = rmse_converged
                elif check_convergence == "theta-gradient":
                    converged = bool(theta_converged)
                if converged:
                    break

        if check_convergence and not converged:
            log(f"Quick fit iteration {it_index} did not converge")
            # return False

        # Store the fitted parameters
        for peak in working_peaks:
            A = spectrum.peaks[peak].A_refined
            x0 = spectrum.peaks[peak].x0_refined
            sigma_L = spectrum.peaks[peak].sigma_L
            sigma_R = spectrum.peaks[peak].sigma_R

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
                working_peaks[peak].A_refined = A
                working_peaks[peak].x0_refined = x0
                working_peaks[peak].sigma_L = sigma_L
                working_peaks[peak].sigma_R = sigma_R

        return working_peaks, converged

    except Exception as e:
        print(f"Quick fit iteration {it_index} failed: {e}")
        return {}, False


def pack_parameters(peaks_dict: dict[int, peak_params]) -> np.ndarray:
    """Flatten all peak parameters into a single 1D array."""
    theta = []
    for pid in sorted(peaks_dict.keys()):
        p = peaks_dict[pid]
        theta.extend([p.A_refined, p.x0_refined, p.sigma_L, p.sigma_R])
    return np.array(theta, dtype=float)


def check_local_convergence(theta_old, theta_new, tol_theta=1e-4, eps=1e-12):
    """
    Determine whether local optimization has converged.

    Parameters
    ----------
    theta_old : np.ndarray
        Previous parameter vector (flattened).
    theta_new : np.ndarray
        New parameter vector after an iteration.
    tol_theta : float
        Relative tolerance on parameter change.
    eps : float
        Small constant to avoid division by zero.

    Returns
    -------
    converged : bool
        True if both criteria are satisfied.
    delta_obj : float
        Relative change in objective (for logging/debug).
    delta_theta : float
        Relative change in parameters (for logging/debug).
    """

    # Parameter change (normalized L2 norm)
    # Convert to numpy arrays if needed
    theta_old = np.asarray(theta_old)
    theta_new = np.asarray(theta_new)
    diff = np.linalg.norm(theta_new - theta_old)
    norm = np.linalg.norm(theta_old) + eps
    delta_theta = diff / norm

    converged = delta_theta < tol_theta
    return converged, delta_theta
