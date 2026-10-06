# Import the packages
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import src.gwrefpy as gr

# Create sample time series data for testing
rng = np.random.default_rng(42)
days = np.arange(400)
period_days = 100
phase = (days % period_days) / period_days
x = 2 * phase - 1
polynomial_shape = (1 - x**2) * (1 - 5 * x**2 + 0.6 * x + 0.4 * x**3)
noise_scale = 0.05 + 0.35 * np.sin(np.pi * phase)
mean_head = 12.0


def center_period_variation(variation):
    variation = variation.copy()
    for start in range(0, len(days), period_days):
        stop = min(start + period_days, len(days))
        variation[start] = 0
        variation[stop - 1] = 0
        interior = slice(start + 1, stop - 1)
        variation[interior] -= variation[interior].mean()
    return variation


def make_head_data():
    variation = 1.5 * polynomial_shape + rng.normal(0, noise_scale)
    return mean_head + center_period_variation(variation)


head = make_head_data()
head_deviation = head - mean_head
observed_variation = (
    0.6 * head_deviation**2
    + 0.08 * head_deviation**5
    + rng.normal(0, 0.03 + 0.05 * np.sin(np.pi * phase))
)
head2 = mean_head + center_period_variation(observed_variation)
dates = pd.date_range(start="2020-01-01", periods=len(days), freq="D")
timeseries = pd.Series(data=head, index=dates, name="obs")
timeseries2 = pd.Series(data=head2, index=dates, name="obs2")

# Create a Model object, add wells, and fit the model
model = gr.Model(name="Small Example")
ref = gr.Well(name="ref2", timeseries=timeseries, is_reference=True)
obs = gr.Well(name="ref3", timeseries=timeseries2, is_reference=False)
model.add_well([ref, obs])

# make the ref empty
# ref.timeseries = pd.Series([], dtype=float, name="obs")

# model.fit(obs_well=obs, ref_well=ref, method="linearregression", degree=3, offset="0D")
model.fit(
    obs_well=obs,
    ref_well=ref,
    method="npolyfit",
    degree=3,
    offset="0D",
    tmax="2020-02-01",
)
# model.best_fit(obs_well=obs, ref_wells=[ref, ref2, ref3], method="npolyfit", degree=3, offset="0D", skip_raise_error=False, tmax="2020-03-01")

# Plot the results
model.plot_wells()
model.plot_fits(plot_style="fancy", color_style="color", show_initiation_period=True)
model.plot_fitmethod()

plt.show()
