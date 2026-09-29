import pandas as pd

import gwrefpy as gr


def test_datetime_to_float():
    time = pd.Timestamp("2023-08-27 12:00:00", tz="UTC")
    float = gr.datetime_to_float(time)
    check = 1693137600.0  # Correct UTC timestamp for the given datetime
    assert abs(float - check) < 1e-6  # Allow small numerical error


def test_float_to_datetime():
    timestamp = 1693147200.0  # Corresponds to 2023-08-27 12:00:00 UTC
    expected = pd.Timestamp(year=2023, month=8, day=27, hour=14, minute=40, second=0)
    actual = gr.float_to_datetime(timestamp)
    assert actual == expected


def test_save_load_project(tmp_path, strandangers_model):
    model_str = str(strandangers_model)
    strandangers_model.save_project(str(tmp_path / "test_project"))
    # Check that the project was saved correctly by loading it back
    loaded_project = gr.Model(name="temp")
    loaded_project.open_project(str(tmp_path / "test_project"))
    assert loaded_project.name == strandangers_model.name
    assert len(loaded_project.wells) == len(strandangers_model.wells)
    assert str(loaded_project) == model_str
    for well in strandangers_model.wells:
        loaded_well = loaded_project.get_wells(well.name)
        assert well.name == loaded_well.name
        assert well.is_reference == loaded_well.is_reference
        pd.testing.assert_series_equal(well.timeseries, loaded_well.timeseries)


def test_save_load_polynomial_prediction_constants(tmp_path, strandangers_model):
    strandangers_model.fit("obs", "ref", offset="3.5D", method="npolyfit", degree=1)
    expected = strandangers_model.fits[0].pred_const
    ref = strandangers_model.get_wells("ref")
    new_index = ref.timeseries.index.max() + pd.Timedelta(days=30)
    ref.append_timeseries(pd.Series([12.0], index=[new_index], name="ref"))

    project_path = str(tmp_path / "polynomial_project")
    strandangers_model.save_project(project_path)
    loaded_project = gr.Model(name="temp")
    loaded_project.open_project(project_path)

    actual = loaded_project.fits[0].pred_const
    assert isinstance(actual, pd.Series)
    pd.testing.assert_series_equal(actual, expected)

    new_reference = pd.Series([12.0], index=[new_index], name="ref")
    pd.testing.assert_series_equal(
        loaded_project.fits[0].get_prediction_constant(new_reference),
        strandangers_model.fits[0].get_prediction_constant(new_reference),
    )
