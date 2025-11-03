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
