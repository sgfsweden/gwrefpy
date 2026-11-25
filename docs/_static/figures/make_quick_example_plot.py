"""Make quick_example_plot.png, the figure for the quick example in docs/index.rst.

The wells come from the bundled ``small_example.gwref``.

Run from the repository root:

    uv run python docs/_static/figures/make_quick_example_plot.py
"""

from importlib.resources import files
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import gwrefpy as gr  # noqa: E402

model = gr.Model(str(files("gwrefpy.examples") / "small_example.gwref"))
obs, ref = model.wells["12GIPGW"], model.wells["45LOGW"]

fit = model.fit(
    obs_well=obs, ref_well=ref, offset="0D", tmin="2020-01-01", tmax="2020-03-23"
)
ax = fit.plot(outliers=True, calibration_period=True, figsize=(10, 6))

out = Path(__file__).with_name("quick_example_plot.png")
ax.figure.savefig(out, dpi=150, bbox_inches="tight")
print(f"Saved {out}")
