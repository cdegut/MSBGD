from calendar import c
from dataclasses import dataclass
import os
from typing import Dict, Literal, overload
import numpy as np
from sklearn import base
from modules.data_structures import FitQualityPeakMetrics, MSData, peak_params
from modules.fitting.peak_starting_points import update_peak_starting_points
from modules.math import (
    bi_Lorentzian_integral,
    bi_gaussian_integral,
    standard_error,
    check_theta_convergence,
)
from modules.rendercallback import RenderCallback
from modules.utils import log
import dearpygui.dearpygui as dpg
from modules.fitting.refiner import refine_iteration
from copy import deepcopy
from typing import Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    base: list[float]


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
    wRMSE_threshold: float = 100.0,
    theta_threshold: float = 1e-4,
) -> dict | Literal[False]:
    """Perform advanced statistical analysis using bootstrap and random start methods."""

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
            A=[], x0=[], sigma_L=[], sigma_R=[], integral=[], base=[]
        )
        for peak in working_peak_list
    }

    log(f"Starting bootstrap with {macro_iteration} resamples...")
    successful_fits = 0

    task_pool = []
    # Thread-safe counters
    cpu_count = os.cpu_count() or 2
    completed_lock = Lock()
    completed_tasks = {"count": 0, "successful": 0, "rejected": 0}
    BATCH_SIZE = min(cpu_count * 2, 50)  # Process 4x CPU cores or max 50 at once
    num_batches = (macro_iteration + BATCH_SIZE - 1) // BATCH_SIZE

    avg_iterations = None
    batch_iterations = []
    rescale = 1.0

    if method == "bootstrap-parametric" or method == "bootstrap-residual":
        it = 0
        for test in range(0, 10):
            dpg.set_value(
                "Fitting_indicator_sub_text",
                (
                    f"Running initial rescale tests {test}/10 "
                    + (f"last it: {it} rescale: {rescale:.2f}" if it else "")
                ),
            )
            if dpg.does_alias_exist("noise"):
                dpg.delete_item("noise")
            dpg.add_line_series(
                x=residuals.tolist(),
                y=(np.random.normal(0, sigma_hat, size=len(residuals)) + 250).tolist(),
                label="MBG",
                parent="y_axis_plot2",
                tag="noise",
                show=False,
            )

            test_task = _make_bootstrap_task(
                spectrum=spectrum,
                working_peak_list=working_peak_list,
                data_x=data_x,
                method=method,
                b=test,
                y_fitted=y_fitted,
                wRMSE_threshold=wRMSE_threshold,
                residuals=residuals,
                sigma_hat=float(sigma_hat),
                rescale=rescale,
                micro_iteration=25,
                check_convergence=check_convergence,
                theta_threshold=theta_threshold,
            )
            _, _, it = execute_quick_fit(test_task)

            if 5 < it > 10:
                break
            if it > 10:
                rescale = rescale * 0.75
            elif it > 24:
                rescale = rescale * 0.5
            elif it < 5:
                rescale = rescale * 1.25
            elif it <= 2:
                rescale = rescale * 2.0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, macro_iteration)

        if batch_iterations != []:
            avg_iterations = int(np.mean(batch_iterations))
            batch_iterations = []

        if avg_iterations is not None:
            if avg_iterations > 15:
                rescale = rescale * 0.75
            elif avg_iterations > 25:
                rescale = rescale * 0.5
            elif avg_iterations < 5:
                rescale = rescale * 1.25
            elif avg_iterations <= 2:
                rescale = rescale * 2.0

        if render_callback:
            dpg.set_value(
                "Fitting_indicator_sub_text",
                (
                    f"Running {start_idx} to {end_idx}"
                    + f" average iter last batch: {avg_iterations} rescale: {rescale:.2f}"
                    if avg_iterations and method != "initial"
                    else ""
                ),
            )

        task_pool = []
        for b in range(start_idx, end_idx):
            if method == "bootstrap-parametric" or method == "bootstrap-residual":
                task = _make_bootstrap_task(
                    spectrum=spectrum,
                    working_peak_list=working_peak_list,
                    data_x=data_x,
                    method=method,
                    b=b,
                    y_fitted=y_fitted,
                    wRMSE_threshold=wRMSE_threshold,
                    residuals=residuals,
                    sigma_hat=float(sigma_hat),
                    rescale=rescale,
                    micro_iteration=micro_iteration,
                    check_convergence=check_convergence,
                    theta_threshold=theta_threshold,
                )

            elif method == "initial":
                task = _make_initial_refit_task(
                    spectrum=spectrum,
                    working_peak_list=working_peak_list,
                    data_x=data_x,
                    data_y=data_y,
                    b=b,
                    micro_iteration=micro_iteration,
                    check_convergence=check_convergence,
                    wRMSE_threshold=wRMSE_threshold,
                    theta_threshold=theta_threshold,
                )

            task_pool.append(task)

        with ProcessPoolExecutor(max_workers=cpu_count) as executor:
            futures = {
                executor.submit(execute_quick_fit, task): task for task in task_pool
            }

            for future in as_completed(futures):
                task = futures[future]
                fitted_peaks, converged, iteration = future.result()

                # Update progress (thread-safe)
                with completed_lock:
                    completed_tasks["count"] += 1
                    if converged:
                        completed_tasks["successful"] += 1
                    batch_iterations.append((iteration))

                    # Update GUI
                    if render_callback:
                        render_callback.execute()
                        if dpg.get_value("stop_fitting_checkbox"):
                            log("Fitting stopped by user.")
                            dpg.set_value("Fitting_indicator_text", "Stopping ...")
                            # Cancel remaining futures
                            for f in futures:
                                f.cancel()
                            executor.shutdown(wait=False, cancel_futures=True)

                            return False

                        if method == "initial":
                            dpg.set_value(
                                "Fitting_indicator_text",
                                f"Refit with randomisation: {completed_tasks['count']}/{macro_iteration} "
                                f"({completed_tasks['successful']} converged, {completed_tasks['rejected']} rejected)",
                            )
                        else:
                            dpg.set_value(
                                "Fitting_indicator_text",
                                f"Bootstrap: {completed_tasks['count']}/{macro_iteration} "
                                f"({completed_tasks['successful']} successful)",
                            )

                    if fitted_peaks == {}:
                        completed_tasks["rejected"] += 1
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
                        perturbation_results[peak].A.append(
                            fitted_peaks[peak].A_refined
                        )
                        perturbation_results[peak].x0.append(
                            fitted_peaks[peak].x0_refined
                        )
                        perturbation_results[peak].sigma_L.append(
                            fitted_peaks[peak].sigma_L
                        )
                        perturbation_results[peak].sigma_R.append(
                            fitted_peaks[peak].sigma_R
                        )
                        perturbation_results[peak].integral.append(integral)
                        perturbation_results[peak].base.append(
                            -fitted_peaks[peak].regression_fct[1]
                            / fitted_peaks[peak].regression_fct[0]
                        )

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
            "base": standard_error((perturbation_results[peak].base)),
        }
    if dpg.does_alias_exist("noise"):
        dpg.delete_item("noise")

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
) -> tuple[Dict[int, peak_params], bool, int]:
    iteration = 0
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

            if check_convergence:
                rmse_converged = False
                theta_converged = False
                if check_convergence == "wRMSE" or check_convergence == "both":
                    rmse_converged = quality_metrics.weighted_rmse < wRMSE_threshold

                if check_convergence == "theta-gradient" or check_convergence == "both":
                    new_theta = spectrum.get_packed_parameters()
                    theta_converged, delta_theta = check_theta_convergence(
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
            full_quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                spectrum,
                list(working_peaks.keys()),
            )

            print(
                f"Quick fit did not converge in {n_iterations} iterations \n",
                "chi squared:",
                full_quality_metrics.chi_squared_reduced,
                "\n",
                "R²:",
                full_quality_metrics.r_squared,
                "\n",
                "wRMSE:",
                full_quality_metrics.weighted_rmse,
                f"/ {wRMSE_threshold} \n",
                "RMSE:",
                full_quality_metrics.rmse,
                "\n",
            )
            if (
                full_quality_metrics.chi_squared_reduced > 1.0
                or full_quality_metrics.r_squared < 0.8
                or full_quality_metrics.weighted_rmse > wRMSE_threshold
            ):
                print("Very poor fit detected, rejecting results.")
                return {}, False, iteration

        # Store the fitted parameters
        update_peak_starting_points(spectrum)
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
                working_peaks[peak].regression_fct = spectrum.peaks[peak].regression_fct

        return working_peaks, converged, iteration

    except Exception as e:
        print(f"Quick fit iteration {it_index} failed: {e}")
        return {}, False, iteration


def pack_parameters(peaks_dict: dict[int, peak_params]) -> np.ndarray:
    """Flatten all peak parameters into a single 1D array."""
    theta = []
    for pid in sorted(peaks_dict.keys()):
        p = peaks_dict[pid]
        theta.extend([p.A_refined, p.x0_refined, p.sigma_L, p.sigma_R])
    return np.array(theta, dtype=float)


def _make_bootstrap_task(
    method: Literal["bootstrap-parametric", "bootstrap-residual"],
    b: int,
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    rescale: float,
    residuals: np.ndarray,
    y_fitted: np.ndarray,
    sigma_hat: float,
    micro_iteration: int,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False],
    theta_threshold: float,
    wRMSE_threshold: float,
):
    working_peaks: Dict[int, peak_params] = {}
    if method == "bootstrap-residual" or method == "bootstrap-parametric":
        noise_scale_A = 0.01 * rescale
        noise_scale_X0 = 0.02 * rescale
        noise_scale_w = 0.02 * rescale

        if method == "bootstrap-parametric":
            noise = np.random.normal(0, sigma_hat, size=len(residuals))
            task_data_y = y_fitted + noise

        else:  # residual bootstrap
            bootstrap_indices = np.random.choice(
                len(residuals), size=len(residuals), replace=True
            )
            resampled_residuals = residuals[bootstrap_indices]
            task_data_y = y_fitted + resampled_residuals

        for peak in working_peak_list:
            search_width = spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
            working_peaks[peak] = peak_params(
                A_refined=spectrum.peaks[peak].A_refined
                * (1 + np.random.randn() * noise_scale_A),
                x0_refined=spectrum.peaks[peak].x0_refined
                + (np.random.randn() * search_width * noise_scale_X0),
                sigma_L=spectrum.peaks[peak].sigma_L
                * (1 + np.random.randn() * noise_scale_w),
                sigma_R=spectrum.peaks[peak].sigma_R
                * (1 + np.random.randn() * noise_scale_w),
            )
        task_spectrum = deepcopy(spectrum)

        task = QuickFitInterface(
            spectrum=task_spectrum,
            working_peaks=working_peaks,
            data_x=data_x.copy(),
            data_y=task_data_y,
            n_iterations=micro_iteration,
            check_convergence=check_convergence,
            theta_threshold=theta_threshold,
            wRMSE_threshold=wRMSE_threshold,
            width_regularization=True,
            it_index=b,
        )

        return task


def _make_initial_refit_task(
    data_y: np.ndarray,
    b: int,
    spectrum: MSData,
    working_peak_list: list[int],
    data_x: np.ndarray,
    micro_iteration: int,
    check_convergence: Literal["wRMSE", "theta-gradient", "both", False],
    theta_threshold: float,
    wRMSE_threshold: float,
):
    working_peaks: Dict[int, peak_params] = {}
    noise_scale_A = 0.2
    noise_scale_X0 = 0.1
    noise_scale_w = 0.2  # this is the critical value

    task_data_y = data_y.copy()
    task_spectrum = deepcopy(spectrum)

    for peak in working_peak_list:
        search_width = spectrum.peaks[peak].sigma_L + spectrum.peaks[peak].sigma_R
        noise_scale_X0_applied = (
            noise_scale_X0 if peak < 500 else noise_scale_X0 * 2.0
        )  # be harder on user added peaks
        working_peaks[peak] = peak_params(
            A_refined=spectrum.peaks[peak].A_init
            * (1 + np.random.randn() * noise_scale_A),
            x0_refined=spectrum.peaks[peak].x0_init
            + (np.random.randn() * search_width * noise_scale_X0_applied),
            sigma_L=spectrum.peaks[peak].sigma_L_init
            * (1 + np.random.randn() * noise_scale_w),
            sigma_R=spectrum.peaks[peak].sigma_R_init
            * (1 + np.random.randn() * noise_scale_w),
        )
    task = QuickFitInterface(
        spectrum=task_spectrum,
        working_peaks=working_peaks,
        data_x=data_x.copy(),
        data_y=task_data_y,
        n_iterations=micro_iteration,
        check_convergence=check_convergence,
        theta_threshold=theta_threshold,
        wRMSE_threshold=wRMSE_threshold,
        width_regularization=True,
        it_index=b,
    )

    return task
