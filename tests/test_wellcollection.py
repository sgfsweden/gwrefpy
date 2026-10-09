import pandas as pd
import pytest

from gwrefpy import Model, Well, WellCollection


@pytest.fixture()
def model() -> Model:
    """Model with two observation wells and two reference wells, interleaved."""
    obs = pd.Series(
        [11.4, 11.7, 11.8],
        index=pd.to_datetime(["2023-01-07", "2023-02-01", "2023-02-25"]),
    )
    ref = pd.Series(
        [8.9, 9.2, 9.3, 9.3, 9.5],
        index=pd.to_datetime(
            ["2023-01-08", "2023-02-03", "2023-02-08", "2023-02-25", "2023-02-28"]
        ),
    )
    model = Model(name="wells")
    model.add_well(
        [
            Well("obsA", is_reference=False, timeseries=obs),
            Well("refA", is_reference=True, timeseries=ref),
            Well("obsB", is_reference=False, timeseries=obs + 0.1),
            Well("refB", is_reference=True, timeseries=ref + 0.2),
        ]
    )
    return model


def fit(model: Model, obs: str, ref: str):
    return model.fit(obs, ref, offset="3.5D", report=False)


def names(collection) -> list[str]:
    return [w.name for w in collection]


# --- list-like access ---------------------------------------------------------


def test_wells_is_a_list_like_collection(model) -> None:
    wells = model.wells

    assert isinstance(wells, WellCollection)
    assert len(wells) == 4
    assert bool(wells)
    assert not Model("empty").wells
    assert names(wells) == ["obsA", "refA", "obsB", "refB"]
    assert wells[0].name == "obsA"
    assert wells[-1].name == "refB"
    assert names(wells[1:3]) == ["refA", "obsB"]
    assert isinstance(wells[1:3], WellCollection)


def test_wells_lookup_by_name(model) -> None:
    obs_a = model.wells["obsA"]

    assert isinstance(obs_a, Well)
    assert obs_a.name == "obsA"
    assert "obsA" in model.wells
    assert obs_a in model.wells
    assert "missing" not in model.wells
    assert Well("obsA", is_reference=False) not in model.wells
    with pytest.raises(KeyError):
        model.wells["missing"]


def test_names_lists_the_well_names_in_order(model) -> None:
    assert model.wells.names() == ["obsA", "refA", "obsB", "refB"]


# --- filter -----------------------------------------------------------------


def test_obs_and_ref_split_by_kind(model) -> None:
    assert names(model.wells.obs()) == ["obsA", "obsB"]
    assert names(model.wells.ref()) == ["refA", "refB"]
    assert isinstance(model.wells.obs(), WellCollection)


def test_filter_by_kind_name_and_predicate(model) -> None:
    assert names(model.wells.filter(kind="ref")) == ["refA", "refB"]
    assert names(model.wells.filter(name="obsB")) == ["obsB"]
    assert names(model.wells.filter(name=["refB", "obsA"])) == ["obsA", "refB"]
    assert names(model.wells.filter(lambda w: w.name.endswith("B"))) == [
        "obsB",
        "refB",
    ]


def test_filter_combines_keywords_with_and(model) -> None:
    result = model.wells.filter(lambda w: "A" in w.name, kind="obs", name=["obsA"])

    assert names(result) == ["obsA"]
    assert names(model.wells.filter(kind="obs", name="refA")) == []


def test_filter_unknown_kind_raises(model) -> None:
    with pytest.raises(ValueError):
        model.wells.filter(kind="observation")


def test_filter_without_matches_warns_and_is_empty(model, caplog) -> None:
    result = model.wells.filter(name="missing")

    assert not result
    assert "No wells matched" in caplog.text


def test_filters_can_be_chained(model) -> None:
    assert names(model.wells.obs().filter(name=["obsB", "refB"])) == ["obsB"]


def test_get_returns_the_single_match(model) -> None:
    assert model.wells.get(name="refB") is model.wells["refB"]
    assert model.wells.get(lambda w: w.name == "obsB", kind="obs").name == "obsB"


def test_get_raises_unless_exactly_one_match(model) -> None:
    with pytest.raises(ValueError, match="No well matched"):
        model.wells.get(name="missing")
    with pytest.raises(ValueError, match="2 wells matched"):
        model.wells.get(kind="obs")


# --- remove -----------------------------------------------------------------


def test_remove_by_well_name_or_collection_returns_removed_wells(model) -> None:
    obs_a = model.wells["obsA"]

    removed = model.wells.remove(obs_a)
    assert names(removed) == ["obsA"]
    assert isinstance(removed, WellCollection)

    assert names(model.wells.remove("refB")) == ["refB"]
    assert names(model.wells.remove(model.wells.ref())) == ["refA"]
    assert model.wells.names() == ["obsB"]


def test_removed_well_no_longer_belongs_to_the_model(model) -> None:
    obs_a = model.wells["obsA"]

    model.wells.remove(obs_a)

    assert model not in obs_a.model
    model.add_well(obs_a)
    assert model.wells.names() == ["refA", "obsB", "refB", "obsA"]


def test_remove_also_removes_the_fits_of_the_well(model, caplog) -> None:
    fit(model, "obsA", "refA")
    fit(model, "obsB", "refA")
    fit(model, "obsB", "refB")

    model.wells.remove("refA")

    assert [f.name for f in model.fits] == ["obsB~refB:linearregression"]
    assert "Removed 2 fit(s)" in caplog.text


def test_remove_dry_run_changes_nothing(model, caplog) -> None:
    fit(model, "obsA", "refA")

    removed = model.wells.remove(["obsA", "refB"], dry_run=True)

    assert names(removed) == ["obsA", "refB"]
    assert len(model.wells) == 4
    assert len(model.fits) == 1
    assert model in removed[0].model
    assert "1 fit(s)" in caplog.text


def test_remove_missing_target_raises_and_removes_nothing(model) -> None:
    with pytest.raises(KeyError):
        model.wells.remove(["obsA", "missing"])
    with pytest.raises(KeyError):
        model.wells.remove(Well("refA", is_reference=True))

    assert len(model.wells) == 4


def test_detached_collections_cannot_be_modified(model) -> None:
    with pytest.raises(TypeError):
        model.wells.obs().remove("obsA")
    with pytest.raises(TypeError):
        WellCollection([model.wells["obsA"]]).remove("obsA")

    assert len(model.wells) == 4


def test_model_loads_after_removing_a_fitted_well(model, tmp_path) -> None:
    fit(model, "obsA", "refA")
    fit(model, "obsB", "refB")
    model.wells.remove("refA")
    model.save_project(str(tmp_path / "project"))

    loaded = Model(name=str(tmp_path / "project.gwref"))

    assert loaded.wells.names() == ["obsA", "obsB", "refB"]
    assert [f.name for f in loaded.fits] == ["obsB~refB:linearregression"]


# --- to_dataframe and repr --------------------------------------------------


def test_to_dataframe_has_a_row_per_well_in_collection_order(model) -> None:
    df = model.wells.to_dataframe()

    assert df["name"].tolist() == ["obsA", "refA", "obsB", "refB"]
    assert df["well_type"].tolist() == [
        "observation",
        "reference",
        "observation",
        "reference",
    ]
    row = df.iloc[0]
    assert row["data_points"] == 3
    assert row["start_date"] == pd.Timestamp("2023-01-07")
    assert row["end_date"] == pd.Timestamp("2023-02-25")
    assert row["latest_value"] == 11.8
    assert row["mean_level"] == pytest.approx((11.4 + 11.7 + 11.8) / 3)


def test_to_dataframe_includes_fit_columns(model) -> None:
    good = fit(model, "obsA", "refA")
    other = fit(model, "obsA", "refB")
    best, worse = sorted([good, other], key=lambda f: f.rmse)

    df = model.wells.to_dataframe().set_index("name")

    assert df.loc["obsA", "best_fit_ref_well"] == best.ref_well.name
    assert df.loc["obsA", "best_rmse"] == best.rmse
    assert pd.isna(df.loc["obsB", "best_fit_ref_well"])
    assert df.loc["refA", "num_fits"] == 1
    assert df.loc["refA", "avg_rmse"] == good.rmse
    assert df.loc["refB", "num_fits"] == 1
    assert pd.isna(df.loc["obsA", "num_fits"])


def test_to_dataframe_of_a_filtered_collection(model) -> None:
    fit(model, "obsA", "refA")

    df = model.wells.ref().to_dataframe()

    assert df["name"].tolist() == ["refA", "refB"]
    assert df["num_fits"].tolist() == [1, 0]


def test_to_dataframe_without_a_model_has_empty_fit_columns(model) -> None:
    fit(model, "obsA", "refA")

    df = WellCollection([model.wells["obsA"]]).to_dataframe()

    assert df["name"].tolist() == ["obsA"]
    assert pd.isna(df.loc[0, "best_fit_ref_well"])


def test_to_dataframe_of_empty_collection_is_empty() -> None:
    assert Model("empty").wells.to_dataframe().empty


def test_repr_summarises_the_collection(model) -> None:
    assert repr(model.wells) == "WellCollection(4 wells: 2 obs, 2 ref)"
    assert "obsA" in model.wells._repr_html_()


# --- deprecated model API ---------------------------------------------------


def test_obs_wells_and_ref_wells_are_deprecated_but_return_lists(model) -> None:
    with pytest.deprecated_call():
        obs_wells = model.obs_wells
    with pytest.deprecated_call():
        ref_wells = model.ref_wells

    assert isinstance(obs_wells, list)
    assert names(obs_wells) == ["obsA", "obsB"]
    assert names(ref_wells) == ["refA", "refB"]


def test_well_names_is_deprecated(model) -> None:
    with pytest.deprecated_call():
        assert model.well_names == ["obsA", "refA", "obsB", "refB"]


def test_get_wells_is_deprecated_but_keeps_its_shape(model) -> None:
    with pytest.deprecated_call():
        single = model.get_wells("refA")
    with pytest.deprecated_call():
        several = model.get_wells(["obsB", "refA"])

    assert single is model.wells["refA"]
    assert names(several) == ["obsB", "refA"]
    with pytest.deprecated_call(), pytest.raises(ValueError):
        model.get_wells("missing")


def test_delete_well_is_deprecated_and_removes_fits(model) -> None:
    fit(model, "obsA", "refA")

    with pytest.deprecated_call():
        model.delete_well([model.wells["refA"]])

    assert model.wells.names() == ["obsA", "obsB", "refB"]
    assert not model.fits


@pytest.mark.parametrize(
    ("method", "expected"),
    [
        ("obs_wells_summary", ["obsA", "obsB"]),
        ("ref_wells_summary", ["refA", "refB"]),
        ("wells_summary", ["obsA", "obsB", "refA", "refB"]),
    ],
)
def test_summaries_are_deprecated_but_keep_their_shape(model, method, expected) -> None:
    with pytest.deprecated_call():
        df = getattr(model, method)()

    assert df["name"].tolist() == expected


def test_model_internals_do_not_use_deprecated_api(model, tmp_path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        fit(model, "obsA", "refA")
        model.fit(["obsA", "obsB"], ["refA", "refB"], offset="3.5D", report=False)
        model.best_fit("obsB", offset="3.5D", report=False)
        str(model)
        model.save_project(str(tmp_path / "project"))
        Model(name=str(tmp_path / "project.gwref"))


# --- plotting ---------------------------------------------------------------


def test_plot_wells_accepts_a_collection(model) -> None:
    import matplotlib.pyplot as plt

    fig, ax = model.plot_wells(model.wells.obs())
    _, default_ax = model.plot_wells()

    assert {line.get_label() for line in ax.get_lines()} >= {"obsA", "obsB"}
    assert {line.get_label() for line in default_ax.get_lines()} >= {
        "obsA",
        "refA",
        "obsB",
        "refB",
    }
    plt.close("all")


def test_remove_counts_a_well_listed_twice_once(model) -> None:
    obs_a = model.wells["obsA"]

    removed = model.wells.remove([obs_a, "obsA"])

    assert names(removed) == ["obsA"]


def test_remove_rejects_targets_that_are_not_wells_or_names(model) -> None:
    for bad in (5, None, [5]):
        with pytest.raises(TypeError, match="Well objects or well names"):
            model.wells.remove(bad)


def test_delete_well_keeps_its_old_error_types(model) -> None:
    with pytest.deprecated_call(), pytest.raises(ValueError):
        model.delete_well(Well("stranger", is_reference=False))
    with pytest.deprecated_call(), pytest.raises(TypeError):
        model.delete_well("obsA")
