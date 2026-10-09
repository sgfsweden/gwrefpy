import pandas as pd

from gwrefpy import Model, Well


def test_add_well_to_model() -> None:
    model = Model("Test Model")
    well1 = Well("Well 1", is_reference=True)
    well2 = Well("Well 2", is_reference=True)
    model.add_well([well1, well2])

    assert len(model.wells) == 2
    assert model.wells[0].name == "Well 1"
    assert model.wells[1].name == "Well 2"

    try:
        model.add_well("Not a well")  # type: ignore
    except TypeError as e:
        assert str(e) == "Only Well instances can be added to the model."

    try:
        model.add_well(well1)
    except ValueError as e:
        assert str(e) == "Well 'Well 1' is already in the model."


def test_strandangers_model(strandangers_model) -> None:
    assert strandangers_model.name == "Strandangers"
    assert len(strandangers_model.wells) == 2
    obs = strandangers_model.get_wells("obs")
    ref = strandangers_model.get_wells("ref")

    assert isinstance(obs, Well)
    assert isinstance(ref, Well)
    assert obs.name == "obs"
    assert ref.name == "ref"


def test_obs_wells_summary_empty() -> None:
    """Test obs_wells_summary with no observation wells."""
    model = Model("Test Model")
    summary = model.obs_wells_summary()
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty


def test_obs_wells_summary_with_data() -> None:
    """Test obs_wells_summary with observation wells containing data."""
    model = Model("Test Model")

    # Create timeseries data
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    obs_data = pd.Series(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0], index=dates
    )

    # Create observation well with data
    obs_well = Well("OBS001", is_reference=False, timeseries=obs_data)
    obs_well.latitude = 55.123
    obs_well.longitude = 12.456
    obs_well.elevation = 20.5

    model.add_well(obs_well)

    summary = model.obs_wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "OBS001"
    assert summary.iloc[0]["well_type"] == "observation"
    assert summary.iloc[0]["data_points"] == 10
    assert summary.iloc[0]["mean_level"] == 5.5
    assert summary.iloc[0]["latest_value"] == 10.0
    assert summary.iloc[0]["latitude"] == 55.123
    assert summary.iloc[0]["longitude"] == 12.456
    assert summary.iloc[0]["elevation"] == 20.5
    assert pd.isna(summary.iloc[0]["best_fit_ref_well"])
    assert pd.isna(summary.iloc[0]["best_rmse"])


def test_obs_wells_summary_without_timeseries() -> None:
    """Test obs_wells_summary with observation wells without timeseries."""
    model = Model("Test Model")
    obs_well = Well("OBS001", is_reference=False)
    model.add_well(obs_well)

    summary = model.obs_wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "OBS001"
    assert summary.iloc[0]["data_points"] == 0
    assert pd.isna(summary.iloc[0]["start_date"])
    assert pd.isna(summary.iloc[0]["end_date"])
    assert pd.isna(summary.iloc[0]["mean_level"])


def test_ref_wells_summary_empty() -> None:
    """Test ref_wells_summary with no reference wells."""
    model = Model("Test Model")
    summary = model.ref_wells_summary()
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty


def test_ref_wells_summary_with_data() -> None:
    """Test ref_wells_summary with reference wells containing data."""
    model = Model("Test Model")

    # Create timeseries data
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    ref_data = pd.Series(
        [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0], index=dates
    )

    # Create reference well with data
    ref_well = Well("REF001", is_reference=True, timeseries=ref_data)
    ref_well.latitude = 55.234
    ref_well.longitude = 12.567
    ref_well.elevation = 25.0

    model.add_well(ref_well)

    summary = model.ref_wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "REF001"
    assert summary.iloc[0]["well_type"] == "reference"
    assert summary.iloc[0]["data_points"] == 10
    assert summary.iloc[0]["mean_level"] == 19.5
    assert summary.iloc[0]["latest_value"] == 24.0
    assert summary.iloc[0]["latitude"] == 55.234
    assert summary.iloc[0]["longitude"] == 12.567
    assert summary.iloc[0]["elevation"] == 25.0
    assert summary.iloc[0]["num_fits"] == 0
    assert pd.isna(summary.iloc[0]["avg_rmse"])


def test_ref_wells_summary_without_timeseries() -> None:
    """Test ref_wells_summary with reference wells without timeseries."""
    model = Model("Test Model")
    ref_well = Well("REF001", is_reference=True)
    model.add_well(ref_well)

    summary = model.ref_wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "REF001"
    assert summary.iloc[0]["data_points"] == 0
    assert pd.isna(summary.iloc[0]["start_date"])
    assert pd.isna(summary.iloc[0]["end_date"])
    assert pd.isna(summary.iloc[0]["mean_level"])


def test_wells_summary_empty() -> None:
    """Test wells_summary with no wells."""
    model = Model("Test Model")
    summary = model.wells_summary()
    assert isinstance(summary, pd.DataFrame)
    assert summary.empty


def test_wells_summary_only_obs() -> None:
    """Test wells_summary with only observation wells."""
    model = Model("Test Model")

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    obs_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    obs_well = Well("OBS001", is_reference=False, timeseries=obs_data)

    model.add_well(obs_well)

    summary = model.wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "OBS001"
    assert summary.iloc[0]["well_type"] == "observation"


def test_wells_summary_only_ref() -> None:
    """Test wells_summary with only reference wells."""
    model = Model("Test Model")

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    ref_data = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)
    ref_well = Well("REF001", is_reference=True, timeseries=ref_data)

    model.add_well(ref_well)

    summary = model.wells_summary()
    assert len(summary) == 1
    assert summary.iloc[0]["name"] == "REF001"
    assert summary.iloc[0]["well_type"] == "reference"


def test_wells_summary_combined() -> None:
    """Test wells_summary with both observation and reference wells."""
    model = Model("Test Model")

    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    obs_data = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=dates)
    ref_data = pd.Series([10.0, 11.0, 12.0, 13.0, 14.0], index=dates)

    obs_well = Well("OBS001", is_reference=False, timeseries=obs_data)
    ref_well = Well("REF001", is_reference=True, timeseries=ref_data)

    model.add_well([obs_well, ref_well])

    summary = model.wells_summary()
    assert len(summary) == 2

    # Check that both well types are present
    well_types = summary["well_type"].tolist()
    assert "observation" in well_types
    assert "reference" in well_types

    # Check specific wells
    obs_row = summary[summary["name"] == "OBS001"].iloc[0]
    ref_row = summary[summary["name"] == "REF001"].iloc[0]

    assert obs_row["well_type"] == "observation"
    assert obs_row["data_points"] == 5
    assert obs_row["mean_level"] == 3.0

    assert ref_row["well_type"] == "reference"
    assert ref_row["data_points"] == 5
    assert ref_row["mean_level"] == 12.0


def test_fits_to_dataframe_empty() -> None:
    """Test fits.to_dataframe with no fits."""
    model = Model(name="test_model")
    summary = model.fits.to_dataframe()
    assert summary.empty
    assert isinstance(summary, pd.DataFrame)


def test_fits_to_dataframe_with_fits(strandangers_model) -> None:
    """Test fits.to_dataframe with actual fit results using the Strandangers example."""
    # Use the existing fixture that has working data
    [obs, ref] = strandangers_model.wells

    # Perform a fit to generate FitResultData
    strandangers_model.fit(obs, ref, offset="3.5D")

    # Test fits.to_dataframe
    summary = strandangers_model.fits.to_dataframe()
    assert len(summary) == 1

    # Check common columns
    row = summary.iloc[0]
    assert row["ref_well_name"] == "ref"
    assert row["obs_well_name"] == "obs"
    assert row["method"] == "LinRegResult"
    assert "rmse" in row
    assert "n_points" in row
    assert "stderr" in row
    assert "confidence_level" in row
    assert "calibration_start" in row
    assert "calibration_end" in row
    assert "time_offset" in row
    assert "t_a" in row
    assert "pred_const" in row

    # Check LinRegResult-specific columns with prefix
    assert "linreg_slope" in row
    assert "linreg_intercept" in row
    assert "linreg_rvalue" in row
    assert "linreg_pvalue" in row
    assert "linreg_stderr" in row

    # Check specific values from the known fit
    assert row["n_points"] == 3  # Known from test_fitting.py
    assert row["time_offset"] == "3.5D"
    assert row["confidence_level"] == 0.95  # Default confidence level


def test_fits_to_dataframe_multiple_fits(strandangers_model) -> None:
    """Test fits.to_dataframe with multiple fits using the Strandangers example."""
    # Use the existing fixture and add another reference well
    [obs, ref] = strandangers_model.wells

    # Create second reference well based on existing one
    ref2_data = ref.timeseries + 0.5
    ref2 = Well(name="ref2", is_reference=True, timeseries=ref2_data)
    strandangers_model.add_well(ref2)

    # Perform fits with both reference wells
    strandangers_model.fit(obs, ref, offset="3.5D")
    strandangers_model.fit(obs, ref2, offset="3.5D")

    # Test fits.to_dataframe
    summary = strandangers_model.fits.to_dataframe()
    assert len(summary) == 2

    # Check that both fits are represented
    ref_well_names = summary["ref_well_name"].tolist()
    assert "ref" in ref_well_names
    assert "ref2" in ref_well_names

    # All obs_well_name should be the same
    assert all(name == "obs" for name in summary["obs_well_name"])

    # All methods should be LinRegResult for now
    assert all(method == "LinRegResult" for method in summary["method"])

    # Check that both have the expected number of points and offset
    assert all(row == 3 for row in summary["n_points"])  # From fixture data
    assert all(offset == "3.5D" for offset in summary["time_offset"])
