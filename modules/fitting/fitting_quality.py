from dataclasses import dataclass
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
    method: Literal["bootstrap", "initial"] = "bootstrap",
    check_convergence=False,
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

    perturbation_results: dict[int, PerturbationFitErrorMetrics] = {
        peak: PerturbationFitErrorMetrics(
            A=[], x0=[], sigma_L=[], sigma_R=[], integral=[]
        )
        for peak in working_peak_list
    }

    log(f"Starting bootstrap with {macro_iteration} resamples...")
    successful_fits = 0

    for b in range(macro_iteration):
        noise_scale_A = 0.05
        noise_scale_X0 = 0.05
        noise_scale_w = 0.1
        working_peaks: Dict[int, peak_params] = {}
        # Resample residuals with replacement
        if method == "bootstrap":
            if render_callback:
                render_callback.execute()
                dpg.set_value(
                    "Fitting_indicator_text",
                    f"Bootstrap: {b+1}/{macro_iteration} ({successful_fits} successful)",
                )
            bootstrap_indices = np.random.choice(
                len(residuals), size=len(residuals), replace=True
            )
            resampled_residuals = residuals[bootstrap_indices]
            data_y = y_fitted + resampled_residuals

            # Reset to perturbed parameters before refitting

            for peak in working_peak_list:
                working_peaks[peak] = peak_params(
                    A_refined=spectrum.peaks[peak].A_refined,
                    x0_refined=spectrum.peaks[peak].x0_refined,
                    sigma_L=spectrum.peaks[peak].sigma_L,
                    sigma_R=spectrum.peaks[peak].sigma_R,
                )
        elif method == "initial":

            if render_callback:
                render_callback.execute()
                dpg.set_value(
                    "Fitting_indicator_text",
                    f"Initial parameter randomization: {b+1}/{macro_iteration} ({successful_fits} successful)",
                )
            for peak in working_peak_list:
                working_peaks[peak] = peak_params(
                    A_refined=spectrum.peaks[peak].A_init
                    * (1 + np.random.randn() * noise_scale_A),
                    x0_refined=spectrum.peaks[peak].x0_init
                    + (np.random.randn() * spectrum.peaks[peak].width * noise_scale_X0),
                    sigma_L=spectrum.peaks[peak].sigma_L_init
                    * (1 + np.random.randn() * noise_scale_w),
                    sigma_R=spectrum.peaks[peak].sigma_R_init
                    * (1 + np.random.randn() * noise_scale_w),
                )

        # Refit with bootstrap data (simplified - fewer iterations)
        fitted_peaks = quick_fit_model(
            spectrum,
            working_peaks,
            data_x,
            data_y,
            n_iterations=micro_iteration,
            it_index=b,
            check_convergence=check_convergence,
            convergence_threshold=convergence_threshold,
            indicator="Fitting_indicator_sub_text",
        )

        if not fitted_peaks:
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
            perturbation_results[peak].sigma_L.append(fitted_peaks[peak].sigma_L)
            perturbation_results[peak].sigma_R.append(fitted_peaks[peak].sigma_R)
            perturbation_results[peak].integral.append(integral)

        successful_fits += 1

    # Compute standard errors from bootstrap distribution
    result = {}
    for peak in working_peak_list:
        if len(perturbation_results[peak].A) < macro_iteration * 0.5:
            log(
                f"Warning: Peak {peak} had only {len(perturbation_results[peak].A)} successful fits"
            )
            result[peak] = None
            continue

        result[peak] = {
            "A": standard_error((perturbation_results[peak].A)),
            "x0": standard_error((perturbation_results[peak].x0)),
            "sigma_L": standard_error((perturbation_results[peak].sigma_L)),
            "sigma_R": standard_error((perturbation_results[peak].sigma_R)),
            "integral": standard_error((perturbation_results[peak].integral)),
            "n_samples": len(perturbation_results[peak].A),
        }

    log(f"Perturbation complete: {successful_fits}/{macro_iteration} successful fits")
    return result


# def advanced_statistical_analysis(
#     spectrum: MSData,
#     working_peak_list: list[int],
#     data_x: np.ndarray,
#     data_y: np.ndarray,
#     render_callback: RenderCallback,
#     macro_iteration=100,
#     micro_iteration=10,
#     method: Literal["bootstrap", "initial"] = "bootstrap",
#     check_convergence=False,
#     convergence_threshold=1e-4,
#     max_workers: Optional[int] = None,  # None = auto-detect CPU count
# ) -> dict:
#     """
#     Calculate standard errors via residual bootstrap resampling.
#     Now parallelized for faster execution.
#     """
#     y_fitted = spectrum.calculate_mbg(data_x, fitting=True)
#     residuals = data_y - y_fitted

#     perturbation_results: dict[int, PerturbationFitErrorMetrics] = {
#         peak: PerturbationFitErrorMetrics(A=[], x0=[], sigma_L=[], sigma_R=[])
#         for peak in working_peak_list
#     }

#     log(f"Starting bootstrap with {macro_iteration} resamples (multithreaded)...")
#     successful_fits = 0
#     completed_iterations = 0
#     lock = Lock()  # Thread-safe counter updates

#     # Create thread pool
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         # Submit all iterations
#         futures = {
#             executor.submit(
#                 _perturbation_iteration,
#                 b,
#                 method,
#                 spectrum,
#                 working_peak_list,
#                 data_x,
#                 data_y,
#                 y_fitted,
#                 residuals,
#                 micro_iteration,
#                 check_convergence,
#                 convergence_threshold,
#             ): b
#             for b in range(macro_iteration)
#         }

#         # Process completed iterations
#         for future in as_completed(futures):
#             iteration_num, fitted_peaks = future.result()

#             with lock:
#                 completed_iterations += 1

#                 # Update UI
#                 if render_callback:
#                     render_callback.execute()
#                     method_name = (
#                         "Bootstrap"
#                         if method == "bootstrap"
#                         else "Initial parameter randomization"
#                     )
#                     dpg.set_value(
#                         "Fitting_indicator_text",
#                         f"{method_name}: {completed_iterations}/{macro_iteration} ({successful_fits} successful)",
#                     )

#                 # Store results
#                 if fitted_peaks:
#                     for peak in fitted_peaks:
#                         perturbation_results[peak].A.append(
#                             fitted_peaks[peak].A_refined
#                         )
#                         perturbation_results[peak].x0.append(
#                             fitted_peaks[peak].x0_refined
#                         )
#                         perturbation_results[peak].sigma_L.append(
#                             fitted_peaks[peak].sigma_L
#                         )
#                         perturbation_results[peak].sigma_R.append(
#                             fitted_peaks[peak].sigma_R
#                         )
#                     successful_fits += 1

#     # Compute standard errors from bootstrap distribution
#     result = {}
#     for peak in working_peak_list:
#         if len(perturbation_results[peak].A) < macro_iteration * 0.5:
#             log(
#                 f"Warning: Peak {peak} had only {len(perturbation_results[peak].A)} successful fits"
#             )
#             result[peak] = None
#             continue

#         result[peak] = {
#             "A": float(np.std(perturbation_results[peak].A)),
#             "x0": float(np.std(perturbation_results[peak].x0)),
#             "sigma_L": float(np.std(perturbation_results[peak].sigma_L)),
#             "sigma_R": float(np.std(perturbation_results[peak].sigma_R)),
#             "A_ci": (
#                 float(np.percentile(perturbation_results[peak].A, 2.5)),
#                 float(np.percentile(perturbation_results[peak].A, 97.5)),
#             ),
#             "x0_ci": (
#                 float(np.percentile(perturbation_results[peak].x0, 2.5)),
#                 float(np.percentile(perturbation_results[peak].x0, 97.5)),
#             ),
#             "n_samples": len(perturbation_results[peak].A),
#         }

#     log(f"Bootstrap complete: {successful_fits}/{macro_iteration} successful fits")
#     return result


# def _perturbation_iteration(
#     iteration: int,
#     method: Literal["bootstrap", "initial"],
#     spectrum: MSData,
#     working_peak_list: list[int],
#     data_x: np.ndarray,
#     data_y: np.ndarray,
#     y_fitted: np.ndarray,
#     residuals: np.ndarray,
#     micro_iteration: int,
#     check_convergence: bool,
#     convergence_threshold: float,
#     noise_scale_A: float = 0.05,
#     noise_scale_X0: float = 0.05,
#     noise_scale_w: float = 0.1,
# ) -> tuple[int, Optional[Dict[int, peak_params]]]:
#     """Single bootstrap iteration - thread-safe worker function"""
#     try:
#         working_peaks: Dict[int, peak_params] = {}

#         if method == "bootstrap":
#             bootstrap_indices = np.random.choice(
#                 len(residuals), size=len(residuals), replace=True
#             )
#             resampled_residuals = residuals[bootstrap_indices]
#             current_data_y = y_fitted + resampled_residuals

#             for peak in working_peak_list:
#                 working_peaks[peak] = peak_params(
#                     A_refined=spectrum.peaks[peak].A_refined,
#                     # * (1 + np.random.randn() * noise_scale_A),
#                     x0_refined=spectrum.peaks[peak].x0_refined,
#                     # + (np.random.randn() * spectrum.peaks[peak].width * noise_scale_X0),
#                     sigma_L=spectrum.peaks[peak].sigma_L,
#                     # * (1 + np.random.randn() * noise_scale_w),
#                     sigma_R=spectrum.peaks[peak].sigma_R,
#                     # * (1 + np.random.randn() * noise_scale_w),
#                 )
#         else:  # method == "initial"
#             current_data_y = data_y
#             for peak in working_peak_list:
#                 working_peaks[peak] = peak_params(
#                     A_refined=spectrum.peaks[peak].A_init
#                     * (1 + np.random.randn() * noise_scale_A),
#                     x0_refined=spectrum.peaks[peak].x0_init
#                     + (np.random.randn() * spectrum.peaks[peak].width * noise_scale_X0),
#                     sigma_L=spectrum.peaks[peak].sigma_L_init
#                     * (1 + np.random.randn() * noise_scale_w),
#                     sigma_R=spectrum.peaks[peak].sigma_R_init
#                     * (1 + np.random.randn() * noise_scale_w),
#                 )

#         fitted_peaks = quick_fit_model(
#             spectrum,
#             working_peaks,
#             data_x,
#             current_data_y,
#             n_iterations=micro_iteration,
#             it_index=iteration,
#             check_convergence=check_convergence,
#             convergence_threshold=convergence_threshold,
#             indicator=None,  # Disable per-thread UI updates
#         )

#         return iteration, fitted_peaks
#     except Exception as e:
#         log(f"Bootstrap iteration {iteration} failed: {e}")
#         return iteration, None


def quick_fit_model(
    spectrum: MSData,
    working_peaks: Dict[int, peak_params],
    data_x: np.ndarray,
    data_y: np.ndarray,
    n_iterations=10,
    check_convergence=False,
    convergence_threshold: float = 95,
    width_regularization: bool = True,
    it_index: int = 0,
    indicator: Optional[str] = None,
):
    try:
        converged = False
        # Create temporary spectrum copy to avoid modifying the original
        temp_spectrum = deepcopy(spectrum)
        for peak in working_peaks:
            temp_spectrum.peaks[peak].A_refined = working_peaks[peak].A_refined
            temp_spectrum.peaks[peak].x0_refined = working_peaks[peak].x0_refined
            temp_spectrum.peaks[peak].sigma_L = working_peaks[peak].sigma_L
            temp_spectrum.peaks[peak].sigma_R = working_peaks[peak].sigma_R
            temp_spectrum.peaks[peak].width = (
                working_peaks[peak].sigma_L + working_peaks[peak].sigma_R
            )
        widths = (-1, -1, -1, -1)
        quality_metrics: Optional[FitQualityMetricsReduced] = None
        # Run a quick refinement (fewer iterations for speed)
        for iteration in range(n_iterations):
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
                    spectrum=temp_spectrum,
                    original_peak_width=temp_spectrum.peaks[peak].width,
                    force_gaussian=False,
                    widths=widths,
                )

            quality_metrics = calculate_fit_quality_metrics(
                data_x,
                data_y,
                temp_spectrum,
                list(working_peaks.keys()),
                rmse_only=True,
            )
            if indicator:
                dpg.set_value(
                    indicator,
                    f"Quick fit cycle {it_index}, {iteration+1}/{n_iterations}, RMSE: {quality_metrics.weighted_rmse:.4f} target {convergence_threshold:.4f}",
                )

            if check_convergence and (
                quality_metrics.weighted_rmse < convergence_threshold
            ):
                converged = True
                break

        if check_convergence and not converged:
            log(f"Quick fit iteration {it_index} did not converge")
            return False

        # Store the fitted parameters
        for peak in working_peaks:
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
                working_peaks[peak].A_refined = A
                working_peaks[peak].x0_refined = x0
                working_peaks[peak].sigma_L = sigma_L
                working_peaks[peak].sigma_R = sigma_R

        return working_peaks

    except Exception as e:
        log(f"Quick fit iteration {it_index} failed: {e}")
        return False
