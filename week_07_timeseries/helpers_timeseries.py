"""
Helper functions for Time Series Analysis
"""

import warnings

import numpy as np
import pandas as pd
import seaborn as sns
from dateutil.relativedelta import relativedelta
from statsmodels.tsa.stattools import adfuller, kpss

warnings.simplefilter("ignore")


def qcd_variance(series: pd.Series, window: int = 10) -> float:
    """
    Function that returns the quartile coefficient of dispersion
    of the rolling variance of a series in a given window range.
    """

    # Rolling variance for a given window
    variances = series.rolling(window).var().dropna()

    # First quartile
    quartile_1 = np.percentile(variances, 25, method="midpoint")

    # Third quartile
    quartile_3 = np.percentile(variances, 75, method="midpoint")

    # Quartile coefficient of dispersion
    qcd = (quartile_3 - quartile_1) / (quartile_3 + quartile_1)

    return round(qcd, 6)


def p_values(series: pd.Series) -> tuple[float, float]:
    """
    Returns p-values for ADF and KPSS Tests on a time series.
    """

    # p-value from Augmented Dickey-Fuller (ADF) Test
    p_adf = adfuller(series, autolag="AIC")[1]

    # p-value from Kwiatkowski–Phillips–Schmidt–Shin (KPSS) Test
    p_kpss = kpss(series, regression="c", nlags="auto")[1]

    return round(p_adf, 6), round(p_kpss, 6)


def test_stationarity(series: pd.Series, qcd_window=10) -> None:
    """
    Prints likely conclusions about series stationarity.
    """

    # Test heteroscedasticity with qcd
    qcd = qcd_variance(series, window=qcd_window)

    if qcd >= 0.50:
        print(f"\n Non-stationary: heteroscedastic (qcd = {qcd})\n")

    # Test stationarity
    else:
        p_adf, p_kpss = p_values(series)

        # Print p-values
        print(f"p-values: {p_adf} (ADF), {p_kpss} (KPSS)")
        print(f"Quartile coefficient of dispersion (QCD): {qcd}\n")

        if (p_adf < 0.01) and (p_kpss >= 0.05):
            print("Stationary or seasonal-stationary")

        elif (p_adf >= 0.1) and (p_kpss < 0.05):
            print("Difference-stationary")

        elif (p_adf < 0.1) and (p_kpss < 0.05):
            print("Trend-stationary")

        else:
            print("Non-stationary; no robust conclusions")


def clean_temp(dataframe: pd.DataFrame, series: pd.Series):
    """
    Function to impute missing temperature values.

    Returns a temperature column with missing values imputed.
    Imputation is done with the average of the temperatures on the same
    day over all the reference years.
    """

    # Reference years are all years 5 years before and 5 years after
    reference_years = list(range(-5, 0)) + list(range(1, 6))

    # If missing value occurs
    if series["quality"] == 9:
        # List reference dates
        reference_dates = [
            series["date"] + relativedelta(years=y) for y in reference_years
        ]
        # Calculate mean temperatue over the referenced dates
        temp_value = dataframe[dataframe["date"].isin(reference_dates)][
            "mean_temp"
        ].mean()
        return int(temp_value)

    # Else return unchanged value
    return series["mean_temp"]


def resampled_mean_plot(series: pd.Series, years: int):
    """
    Function to plot resampled data by year for a given period.

    Parameters
    ----------
    series : pd.Series
        Time series to be resampled.
    years : int
        Number of years to resample.
    """

    series_ = series.copy()
    series_.index = pd.to_datetime(series_.index)
    resampled = series_.resample(str(years) + "Y").mean()
    resampled.plot(legend=True)
    sns.despine()


def rolling_mean_plot(series, years):
    """
    Function to plot the rolling average.
    """

    window = int(years * 365.24)
    rolling = series.rolling(window).mean()
    rolling.plot(legend=True)
    sns.despine()


def calc_seas(
    dataframe: pd.DataFrame, col_ts: str = "timestep", frequency: float = 1 / 365.24
) -> pd.DataFrame:
    """
    Function to calculate sin/cos values based on a given frequency.
    """
    df_ = dataframe.copy()

    # Convert to list if is str
    frequencies = [frequency] if isinstance(frequency, str) else frequency

    for i, freq in enumerate(frequencies):
        col_name = f"sin_{i}"
        df_[col_name] = np.sin(2 * np.pi * freq * df_[col_ts])

        col_name = f"cos_{i}"
        df_[col_name] = np.cos(2 * np.pi * freq * df_[col_ts])

    return df_
