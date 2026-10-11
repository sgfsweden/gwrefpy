import numpy as np
import pandas as pd
import pytest

from gwrefpy import Model, Well


@pytest.fixture()
def timeseries() -> pd.Series:
    """Fixture for generating a sinusoidal timeseries for testing."""
    # Generate 334 days of data starting from 2024-10-01 (matching original data length)
    start_date = pd.Timestamp("2024-10-01")
    dates = pd.date_range(start=start_date, periods=334, freq="D")

    # Generate sinusoidal data with some variation (similar to groundwater patterns)
    # Base level around 25, with seasonal variation and some noise
    t = np.arange(334)
    base_level = 25.0
    seasonal_amplitude = 2.0
    seasonal_period = 365.25  # days per year
    trend = 0.0001 * t  # slight upward trend
    noise = 0.05 * np.random.RandomState(42).randn(334)  # reproducible noise

    values = (
        base_level
        + seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
        + trend
        + noise
    )

    return pd.Series(values, index=dates, name="lagga2")


@pytest.fixture()
def strandangers_example() -> tuple[pd.Series, pd.Series]:
    """Test fixture to create Strandangers example."""
    obs = pd.Series(
        index=[
            pd.Timestamp("2023-01-07"),
            pd.Timestamp("2023-02-01"),
            pd.Timestamp("2023-02-25"),
        ],
        data=[11.4, 11.7, 11.8],
        name="obs",
    )
    ref = pd.Series(
        index=[
            pd.Timestamp("2023-01-08"),
            pd.Timestamp("2023-02-03"),
            pd.Timestamp("2023-02-08"),
            pd.Timestamp("2023-02-25"),
            pd.Timestamp("2023-02-28"),
        ],
        data=[8.9, 9.2, 9.3, 9.3, 9.5],
        name="ref",
    )
    return obs, ref


@pytest.fixture()
def strandangers_model(strandangers_example) -> Model:
    """Fixture for creating Strandangers test case."""
    obs, ref = strandangers_example

    obs_well = Well("obs", is_reference=False)
    obs_well.add_timeseries(obs)

    ref_well = Well("ref", is_reference=True)
    ref_well.add_timeseries(ref)

    model = Model(name="Strandangers")
    model.add_well(obs_well)
    model.add_well(ref_well)

    return model


@pytest.fixture()
def small_example_model() -> Model:
    """Fixture for creating a small example model."""

    # Create sample time series data for testing
    dates = pd.date_range(start="2020-01-01", periods=365, freq="D")

    obs_series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 365)) + (np.random.normal(0, 0.1, 365) * 2), index=dates)
    ref_series = pd.Series(np.sin(np.linspace(0, 4 * np.pi, 365)) + np.random.normal(0, 0.1, 365), index=dates)


    # Create a Model object, add wells, and fit the model
    model = Model(name="Small Example")
    ref = Well(name="ref", timeseries=ref_series, is_reference=True)
    obs = Well(name="obs", timeseries=obs_series, is_reference=False)
    model.add_well([ref, obs])

    return model