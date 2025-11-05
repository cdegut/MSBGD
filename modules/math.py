import numpy as np


# Define multi-peak bi-Gaussian function
def multi_bi_gaussian(x, *params):
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        A, x0, sigma_L, sigma_R = params[i * 4 : (i + 1) * 4]
        y += np.where(
            x < x0,
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_L**2)),
            A * np.exp(-((x - x0) ** 2) / (2 * sigma_R**2)),
        )
    return y


# Define bi-Gaussian function
def bi_gaussian(x, A, x0, sigma_L, sigma_R):
    return np.where(
        x < x0,
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_L**2)),
        A * np.exp(-((x - x0) ** 2) / (2 * sigma_R**2)),
    )


def bi_gaussian_integral(A, sigmaL, sigmaR):
    """
    Compute total integral (area under curve) of an asymmetric bi-Gaussian.

    f(x) = A * exp(-((x - mu)^2)/(2*sigma_L^2)) for x < mu
         = A * exp(-((x - mu)^2)/(2*sigma_R^2)) for x >= mu

    Parameters
    ----------
    A : float
        Amplitude at x = mu
    sigmaL, sigmaR : float
        Left and right standard deviations

    Returns
    -------
    float
        Total integral over all x
    """
    return A * np.sqrt(np.pi / 2) * (sigmaL + sigmaR)


def bi_Lorentzian(x, amplitude, center, width_left, width_right):
    """
    Asymmetric Lorentzian function with different widths on each side

    Parameters:
    x: array of x values
    amplitude: peak height
    center: center position (x0)
    width_left: half-width at half-maximum for x < center (left side)
    width_right: half-width at half-maximum for x >= center (right side)
    """
    y = np.zeros_like(x)

    # # Left side (x < center)
    # mask_left = x < center
    # y[mask_left] = (
    #     amplitude * (width_left**2) / ((x[mask_left] - center) ** 2 + width_left**2)
    # )

    # # Right side (x >= center)
    # mask_right = x >= center
    # y[mask_right] = (
    #     amplitude * (width_right**2) / ((x[mask_right] - center) ** 2 + width_right**2)
    # )

    y += np.where(
        x < center,
        amplitude * (width_left**2) / ((x - center) ** 2 + width_left**2),
        amplitude * (width_right**2) / ((x - center) ** 2 + width_right**2),
    )

    return y


def multi_bi_Lorentzian(x, *params):
    """
    Multi-peak asymmetric Lorentzian function.

    Parameters
    ----------
    x : array-like
        Independent variable
    *params : tuple
        Flattened parameters: (A1, x0_1, width_L1, width_R1, A2, x0_2, width_L2, width_R2, ...)
        Each peak requires 4 parameters: amplitude, center, left width, right width

    Returns
    -------
    array-like
        Sum of all asymmetric Lorentzian peaks
    """
    n_peaks = len(params) // 4
    y = np.zeros_like(x, dtype=float)
    for i in range(n_peaks):
        A, x0, width_L, width_R = params[i * 4 : (i + 1) * 4]
        y += bi_gaussian(x, A, x0, width_L, width_R)
    return y


def bi_Lorentzian_integral(A, width_L, width_R):
    return A * np.pi * (width_L + width_R)


def standard_error(values: list[float]) -> float:
    """
    Calculate the standard error of the mean for a list of values.

    Standard Error (SE) = Standard Deviation (SD) / sqrt(n)

    where n is the number of samples.

    Parameters
    ----------
    values : list of float
        The list of values to calculate the standard error for.

    Returns
    -------
    float
        The standard error of the mean.
    """
    n = len(values)
    if n == 0:
        return float("nan")
    sd = np.std(values, ddof=1)  # Sample standard deviation
    se = sd / np.sqrt(n)
    return se
